"""Cross policy training with evaluation approximation and sample size.

All market-based inputs and dependent outputs remain private. The fixed design
compares observed workloads, not guaranteed equal wall-clock budgets.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_value_martingale import planned_inputs
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale, evaluate_value_dual,
)
from qf_research.optimal_stopping.training_evaluation import (
    paired_difference, policy_gain, replicate_summary,
)
from qf_research.optimal_stopping.cost_budget import independent_difference

METHODS = ('coarse', 'fine')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
SOURCES = ('scripts/study_training_evaluation.py', 'scripts/audit_training_evaluation.py',
           'src/qf_research/optimal_stopping/training_evaluation.py',
           'src/qf_research/optimal_stopping/cost_budget.py',
           'tests/test_training_evaluation.py',
           'projects/01_optimal_stopping/reports/training_evaluation.md')


def seed(config, role, case, repetition, index):
    return np.random.SeedSequence(config['root_seed'], spawn_key=(role, case, repetition, index))


def stream_record(s, paths):
    return dict(entropy=s.entropy, spawn_key=list(s.spawn_key), stream_id=paths.stream_id,
                paths=len(paths.states))


def train(base, config, count, s):
    gc.collect()
    rng = np.random.Generator(np.random.PCG64(s))
    start, cpu = time.perf_counter(), time.process_time()
    paths = simulate_paths(base.model, base.option, base.decisions, count, rng)
    split = time.perf_counter()
    policy = fit_policy(base.option, paths, config['degree'], **config['regression'])
    elapsed, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
    return policy, dict(policy=asdict(policy), fingerprint=policy.fingerprint,
        training_paths=count, stream=stream_record(s, paths), elapsed_seconds=elapsed,
        cpu_seconds=cpu_seconds, simulation_seconds=split-start,
        fitting_seconds=elapsed-(split-start))


def measure(policy, config, method, count, s):
    """Fresh complete evaluation of one frozen policy, with no cached setup."""
    gc.collect()
    rng = np.random.Generator(np.random.PCG64(s))
    start, cpu = time.perf_counter(), time.process_time()
    phases = {}
    approximation = build_values(policy.model, policy.option, policy.decisions,
                                 **config['approximations'][method])
    phases['build_seconds'] = time.perf_counter()-start
    mark = time.perf_counter()
    observations = simulate_paths(policy.model, policy.option, observation_rights(approximation), count, rng)
    phases['simulation_seconds'] = time.perf_counter()-mark
    mark = time.perf_counter()
    sample = value_martingale(approximation, observations)
    phases['martingale_seconds'] = time.perf_counter()-mark
    mark = time.perf_counter()
    result = evaluate_value_dual(policy, sample)
    phases['policy_and_estimation_seconds'] = time.perf_counter()-mark
    elapsed, cpu_seconds = time.perf_counter()-start, time.process_time()-cpu
    factor = policy.model.premium_factor
    estimates = {}
    for name in ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale'):
        e = asdict(getattr(result, name))
        for key in ('value', 'standard_error', 'lower', 'upper'):
            e[key] /= factor
        estimates[name] = e
    # Evidence hashes are outside the declared numerical workload.
    path_hash = hashlib.sha256()
    path_hash.update(observations.states.tobytes())
    path_hash.update(observations.detached_payments.tobytes())
    path_hash.update(json.dumps([asdict(d) for d in observations.decisions], sort_keys=True).encode())
    record = dict(training_paths=policy.training_paths, policy_fingerprint=policy.fingerprint,
        method=method, evaluation_paths=count, estimates_quote=estimates,
        stream=stream_record(s, observations), paths_sha256=path_hash.hexdigest(),
        approximation_fingerprint=approximation.fingerprint, elapsed_seconds=elapsed,
        cpu_seconds=cpu_seconds, phases=phases,
        minimum_gap_quote=float(result.gap_payoffs.min()/factor))
    return record, result


def execute_block(task, config, repetition):
    base = load_frozen(task['source_policy'])[0]
    case = task['case']; forbidden = set(task['prior_streams'])
    policies, trained = {}, {}
    order = [(case+repetition+i) % 2 for i in range(2)]
    for index in order:
        count = config['training_paths'][index]
        policies[count], trained[count] = train(base, config, count, seed(config, 1, case, repetition, index))
        if trained[count]['stream']['stream_id'] in forbidden:
            raise ValueError('Training stream reuses earlier research evidence')
        forbidden.add(trained[count]['stream']['stream_id'])
    choices = [(n, method, e) for n in config['training_paths']
               for method in METHODS for e in config['evaluation_paths']]
    offset = (case+repetition) % len(choices)
    ordered = choices[offset:]+choices[:offset]
    rows, results = {}, {}
    for n, method, e in ordered:
        index = config['evaluation_paths'].index(e)
        key = (n, method, e)
        rows[key], results[key] = measure(policies[n], config, method, e,
                                         seed(config, 2, case, repetition, index))
        r = rows[key]
        if r['stream']['stream_id'] in forbidden:
            raise ValueError('Evaluation overlaps training or earlier research evidence')
        r['total_elapsed_seconds'] = trained[n]['elapsed_seconds']+r['elapsed_seconds']
        r['total_cpu_seconds'] = trained[n]['cpu_seconds']+r['cpu_seconds']
    factor = base.model.premium_factor
    low, high = config['training_paths']
    gains, evaluations, consistency, sampling = [], [], [], []
    for e in config['evaluation_paths']:
        related = [rows[n, m, e] for n in config['training_paths'] for m in METHODS]
        if len({r['paths_sha256'] for r in related}) != 1:
            raise ValueError('Paired workloads have different realised paths or calendars')
        for method in METHODS:
            g = policy_gain(results[low, method, e], results[high, method, e], factor)
            gains.append(dict(method=method, evaluation_paths=e, **g))
        for n in config['training_paths']:
            evaluations.append(dict(training_paths=n, evaluation_paths=e,
                **paired_difference(results[n, 'fine', e].gap_payoffs/factor,
                                    results[n, 'coarse', e].gap_payoffs/factor)))
        gain_vectors = {m: (results[high, m, e].adjusted_policy_payoffs-
                           results[low, m, e].adjusted_policy_payoffs)/factor for m in METHODS}
        consistency.append(dict(evaluation_paths=e,
            **paired_difference(gain_vectors['fine'], gain_vectors['coarse'])))
    lo_e, hi_e = config['evaluation_paths']
    if rows[low, 'coarse', lo_e]['stream']['stream_id'] == rows[low, 'coarse', hi_e]['stream']['stream_id']:
        raise ValueError('Evaluation sample sizes require independent streams')
    for n in config['training_paths']:
        for m in METHODS:
            left, right = rows[n, m, lo_e], rows[n, m, hi_e]
            sampling.append(dict(training_paths=n, method=m,
                gap_mean_difference=independent_difference(right['estimates_quote']['gap_bound'],
                                                          left['estimates_quote']['gap_bound']),
                upper_endpoint_difference=right['estimates_quote']['gap_bound']['upper']-
                                          left['estimates_quote']['gap_bound']['upper']))
    return dict(case=case, repetition=repetition, premium_factor=factor,
                training=[trained[n] for n in config['training_paths']],
                workloads=[rows[k] for k in choices], execution_order=[list(k) for k in ordered],
                policy_gains=gains, evaluation_effects=evaluations,
                gain_consistency=consistency, sampling_effects=sampling)


def summarise(blocks, config, cases):
    expected={(c,r) for c in cases for r in range(config['training_repetitions'])}
    if len(blocks) != len(expected) or {(b['case'],b['repetition']) for b in blocks} != expected:
        raise ValueError('Incomplete or duplicated experimental coverage')
    records = []
    for case in cases:
        chosen = sorted((b for b in blocks if b['case']==case), key=lambda b:b['repetition'])
        primary = [next(g['estimate'] for g in b['policy_gains'] if
                        g['method']=='fine' and g['evaluation_paths']==config['evaluation_paths'][-1]) for b in chosen]
        workloads = []
        for n in config['training_paths']:
            for m in METHODS:
                for e in config['evaluation_paths']:
                    rows=[next(w for w in b['workloads'] if (w['training_paths'],w['method'],w['evaluation_paths'])==(n,m,e)) for b in chosen]
                    def extent(values):
                        return dict(minimum=float(min(values)), median=float(np.median(values)), maximum=float(max(values)))
                    workloads.append(dict(training_paths=n,method=m,evaluation_paths=e,
                        total_elapsed_seconds=extent([w['total_elapsed_seconds'] for w in rows]),
                        mean_gap_quote=float(np.mean([w['estimates_quote']['gap_bound']['value'] for w in rows])),
                        mean_upper_limit_quote=float(np.mean([w['estimates_quote']['gap_bound']['upper'] for w in rows])),
                        mean_sampling_radius_quote=float(np.mean([w['estimates_quote']['gap_bound']['upper']-w['estimates_quote']['gap_bound']['value'] for w in rows]))))
        records.append(dict(case=case,primary_training_gain_quote=replicate_summary(primary),
            additional_training_seconds=[b['training'][1]['elapsed_seconds']-b['training'][0]['elapsed_seconds'] for b in chosen],
            workloads=workloads))
    return dict(models=records, blocks=len(blocks), workloads=sum(len(b['workloads']) for b in blocks),
        pointwise_intervals=True, equal_time_design=False, empirical_calibration_accepted=False,
        publication_clearance=False)


def numerical_record(block):
    """Remove observed timings while preserving every numerical outcome."""
    timing={'elapsed_seconds','cpu_seconds','simulation_seconds','fitting_seconds','phases',
            'total_elapsed_seconds','total_cpu_seconds'}
    def strip(x):
        if isinstance(x,dict): return {k:strip(v) for k,v in x.items() if k not in timing}
        if isinstance(x,list): return [strip(v) for v in x]
        return x
    return strip(block)


def validate_config(c):
    if (c['training_paths'] != [5000,20000] or c['evaluation_paths'] != [20000,80000]
            or c['training_repetitions'] != 8 or c['degree'] != 4
            or c['regression'] != dict(minimum_itm_paths=32,condition_limit=1e8)
            or c['approximations'] != dict(coarse=dict(core_steps=400,wing_steps=40,tolerance=.005),
                                           fine=dict(core_steps=800,wing_steps=40,tolerance=.001))
            or c['root_seed'] != 202609262041 or c['confidence'] != .95
            or c['publication_clearance'] or c['empirical_calibration_accepted']):
        raise ValueError('Configuration differs from the declared crossed design')


def register(configuration, protocol, output):
    configuration, protocol, output = map(private,(configuration,protocol,output))
    if output.exists(): raise FileExistsError('Use a fresh registration directory')
    c=read(configuration); validate_config(c)
    prior, inputs, sources=planned_inputs()
    tasks=[]
    for task in prior:
        entries=[p for r in task['prior_rows'] for p in r['policies']]
        first=load_frozen(entries[0])[0]
        if any((p.model,p.option,p.decisions)!=(first.model,first.option,first.decisions)
               for p,_ in map(load_frozen,entries)):
            raise ValueError('Inconsistent retained financial model')
        tasks.append(dict(case=task['case'],source_policy=entries[0],
            prior_streams=sorted({s['stream_id'] for r in task['prior_rows'] for s in r['streams']})))
    for f in SOURCES: sources[f]=sha(ROOT/f)
    for p in (configuration,protocol): inputs[str(p.relative_to(ROOT))]=sha(p)
    verify_mapping(inputs); verify_mapping(sources)
    output.mkdir(parents=True)
    save(output/'configuration.json',c);save(output/'tasks.json',tasks)
    (output/'protocol.md').write_text(protocol.read_text())
    save(output/'manifest.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs,source_sha256=sources,
        artifact_sha256={f:sha(output/f) for f in ('configuration.json','tasks.json','protocol.md')},
        public_registration=False,publication_clearance=False))


def run(registration, output):
    registration,output=private(registration),private(output)
    if output.exists(): raise FileExistsError('Use a fresh output directory')
    manifest=read(registration/'manifest.json')
    for key in ('input_sha256','source_sha256'): verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'],registration)
    if any(os.environ.get(n)!='1' for n in THREADS): raise ValueError('Use the declared single-thread environment')
    c=read(registration/'configuration.json');validate_config(c);tasks=read(registration/'tasks.json')
    output.mkdir(parents=True);(output/'blocks').mkdir()
    started=time.monotonic()
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),
        registration=str(registration.relative_to(ROOT)),registration_sha256=sha(registration/'manifest.json'),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),processor=platform.processor(),
        threads={n:os.environ[n] for n in THREADS},single_worker=True))
    base=load_frozen(tasks[0]['source_policy'])[0]
    warm_policy,training=train(base,c,1000,seed(c,0,tasks[0]['case'],0,0))
    warmup=[]
    for index,method in enumerate(METHODS):
        row,_=measure(warm_policy,c,method,1000,seed(c,0,tasks[0]['case'],1,index))
        warmup.append(dict(method=method,elapsed_seconds=row['elapsed_seconds'],cpu_seconds=row['cpu_seconds']))
    save(output/'warmup.json',dict(training_seconds=training['elapsed_seconds'],evaluations=warmup))
    blocks=[]
    for task in tasks:
        for repetition in range(c['training_repetitions']):
            block=execute_block(task,c,repetition);blocks.append(block)
            save(output/'blocks'/f'case_{task["case"]:03d}_rep_{repetition:03d}.json',block)
            print(f'Completed model {task["case"]}, repetition {repetition+1}/{c["training_repetitions"]}',flush=True)
    for key in ('input_sha256','source_sha256'): verify_mapping(manifest[key])
    if sha(registration/'manifest.json')!=read(output/'launch.json')['registration_sha256']:
        raise ValueError('Registration changed during execution')
    save(output/'summary.json',summarise(blocks,c,[t['case'] for t in tasks]))
    files=['launch.json','warmup.json','summary.json']+[str(p.relative_to(output)) for p in sorted((output/'blocks').glob('*.json'))]
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,registration=str(registration.relative_to(ROOT)),
        registration_sha256=sha(registration/'manifest.json'),input_sha256=manifest['input_sha256'],
        source_sha256=manifest['source_sha256'],artifact_sha256={f:sha(output/f) for f in files},
        publication_clearance=False,empirical_calibration_accepted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('register');r.add_argument('--configuration',type=Path,required=True)
    r.add_argument('--protocol',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--registration',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='register':register(a.configuration,a.protocol,a.output)
    else:run(a.registration,a.output)
