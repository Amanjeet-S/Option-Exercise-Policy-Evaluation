"""Compare frozen-policy loss bounds under pilot-calibrated time budgets."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_value_martingale import planned_inputs
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.funded_dual import evaluate_dual
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale, evaluate_value_dual,
)
from qf_research.optimal_stopping.cost_budget import timing_model, allocate_paths, independent_difference

METHODS = ('baseline', 'coarse', 'fine')
OWN_SOURCES = ('scripts/study_cost_budget.py', 'src/qf_research/optimal_stopping/cost_budget.py')


def measure(task, method, count, seed):
    """One complete in-memory workload, including setup and every frozen policy.

    Loading, garbage collection and output serialisation fall outside timing.
    Simulation, all numerical validation and estimator construction are inside.
    No value approximation or martingale is reused between timed workloads.
    """
    if method not in METHODS:
        raise ValueError('Unknown method')
    entries = [(r['repetition'], p) for r in task['prior_rows'] for p in r['policies']]
    loaded = [load_frozen(p) for _, p in entries]
    first = loaded[0][0]
    if any((p.model, p.option, p.decisions) != (first.model, first.option, first.decisions) for p, _ in loaded):
        raise ValueError('Mixed models in one workload')
    prior = {s['stream_id'] for row in task['prior_rows'] for s in row['streams']}
    rng = np.random.Generator(np.random.PCG64(seed))
    gc.collect()
    start, cpu = time.perf_counter(), time.process_time()
    phases = {}
    if method == 'baseline':
        approximation = None
        rights = first.decisions
    else:
        approximation = build_values(first.model, first.option, first.decisions,
                                    **task['configuration']['approximations'][method])
        rights = observation_rights(approximation)
    phases['build_seconds'] = time.perf_counter()-start
    mark = time.perf_counter()
    observations = simulate_paths(first.model, first.option, rights, count, rng)
    if observations.stream_id in prior:
        raise ValueError('Old stream reused')
    phases['simulation_seconds'] = time.perf_counter()-mark
    mark = time.perf_counter()
    sample = None if approximation is None else value_martingale(approximation, observations)
    phases['martingale_seconds'] = time.perf_counter()-mark
    mark = time.perf_counter()
    results = [evaluate_dual(p, observations, c, hedge='policy_slope') if sample is None
               else evaluate_value_dual(p, sample) for p, c in loaded]
    phases['policy_seconds'] = time.perf_counter()-mark
    elapsed = time.perf_counter()-start
    cpu_seconds = time.process_time()-cpu
    # Inferential summaries do not decide the size or length of this workload.
    policies = []
    for (repetition, _), (policy, _), result in zip(entries, loaded, results):
        record = {}
        for name in ('gap_bound', 'policy_value', 'upper_value', 'stopped_martingale', 'terminal_martingale'):
            estimate = asdict(getattr(result, name))
            for key in ('value', 'standard_error', 'lower', 'upper'):
                estimate[key] /= first.model.premium_factor
            record[name] = estimate
        policies.append(dict(fingerprint=policy.fingerprint, training_repetition=repetition,
            training_paths=policy.training_paths, degree=policy.requested_degree, estimates_quote=record,
            minimum_gap_quote=float(result.gap_payoffs.min()/first.model.premium_factor)))
    return dict(case=task['case'], method=method, paths=count, elapsed_seconds=elapsed,
        cpu_seconds=cpu_seconds, phases=phases,
        stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key), stream_id=observations.stream_id),
        approximation_fingerprint=None if approximation is None else approximation.fingerprint,
        policies=policies)


def allocation_table(pilot, config):
    rows = []
    for case in sorted({r['case'] for r in pilot}):
        for method in METHODS:
            groups = [[r['elapsed_seconds'] for r in pilot if r['case'] == case
                       and r['method'] == method and r['paths'] == n] for n in config['pilot_paths']]
            if any(len(g) != config['pilot_repetitions'] for g in groups):
                raise ValueError('Incomplete timing pilot')
            model = timing_model(config['pilot_paths'], groups)
            for budget in config['target_seconds']:
                allocation = allocate_paths(model, budget, **config['path_limits'])
                rows.append(dict(case=case, method=method, budget_seconds=budget,
                                 timing_model=model, **allocation))
    return rows


def summarise(rows):
    if not rows or len({r['stream']['stream_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or reused evaluation streams')
    keys = [(r['case'], r['budget_seconds'], r['evaluation_repetition'], r['method']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate workload')
    groups = []
    for budget in sorted({r['budget_seconds'] for r in rows}):
        methods = {}
        for method in METHODS:
            selected = [r for r in rows if r['budget_seconds'] == budget and r['method'] == method]
            estimates = [p['estimates_quote']['gap_bound'] for r in selected for p in r['policies']]
            def extent(values):
                return dict(minimum=float(min(values)), median=float(np.median(values)), maximum=float(max(values)))
            methods[method] = dict(workloads=len(selected), policy_evaluations=len(estimates),
                paths=extent([r['paths'] for r in selected]),
                seconds=extent([r['elapsed_seconds'] for r in selected]),
                time_ratio=extent([r['elapsed_seconds']/budget for r in selected]),
                gap_quote=extent([e['value'] for e in estimates]),
                upper_quote=extent([e['upper'] for e in estimates]),
                sampling_radius_quote=extent([e['upper']-e['value'] for e in estimates]),
                overshoots=sum(r['elapsed_seconds'] > budget for r in selected))
        groups.append(dict(budget_seconds=budget, methods=methods))
    lookup = dict(zip(keys, rows))
    comparisons = []
    for case, budget, rep in sorted({k[:3] for k in keys}):
        for right, left in (('coarse', 'baseline'), ('fine', 'baseline'), ('fine', 'coarse')):
            r, l = lookup[case, budget, rep, right], lookup[case, budget, rep, left]
            if [p['fingerprint'] for p in r['policies']] != [p['fingerprint'] for p in l['policies']]:
                raise ValueError('Policy order changed')
            for rp, lp in zip(r['policies'], l['policies']):
                er, el = rp['estimates_quote']['gap_bound'], lp['estimates_quote']['gap_bound']
                comparisons.append(dict(case=case, budget_seconds=budget, evaluation_repetition=rep,
                    fingerprint=rp['fingerprint'], comparison=f'{right}_minus_{left}',
                    gap_difference_quote=independent_difference(er, el),
                    upper_endpoint_difference_quote=er['upper']-el['upper']))
    return dict(groups=groups, comparisons=comparisons, pointwise_intervals=True,
                policies_are_fixed=True, empirical_calibration_accepted=False, publication_clearance=False)


def register(config_path, protocol_path, output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh registration directory')
    cfg = read(private(config_path))
    if (cfg['pilot_paths'] != [8000, 24000] or cfg['pilot_repetitions'] != 3
            or cfg['target_seconds'] != [1., 3.] or cfg['evaluation_repetitions'] != 3
            or cfg['root_seed'] != 202609261337 or cfg['publication_clearance']
            or cfg['empirical_calibration_accepted']
            or cfg['path_limits'] != dict(minimum=2000, maximum=300000, quantum=1000)):
        raise ValueError('Unsupported registered design')
    tasks, inputs, sources = planned_inputs()
    for name in OWN_SOURCES:
        sources[name] = sha(ROOT/name)
    inputs.update({str(private(p).relative_to(ROOT)): sha(private(p)) for p in (config_path, protocol_path)})
    verify_mapping(inputs)
    verify_mapping(sources)
    output.mkdir(parents=True)
    save(output/'configuration.json', cfg)
    save(output/'tasks.json', tasks)
    (output/'protocol.md').write_text(private(protocol_path).read_text())
    save(output/'manifest.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={n: sha(output/n) for n in ('configuration.json', 'tasks.json', 'protocol.md')},
        public_registration=False, publication_clearance=False))


def run(registration, output):
    registration, output = private(registration), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh experiment directory')
    manifest = read(registration/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], registration)
    if any(os.environ.get(n) != '1' for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Set all recorded numerical thread limits to one')
    cfg, tasks = read(registration/'configuration.json'), read(registration/'tasks.json')
    output.mkdir(parents=True)
    launch = dict(started_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(), machine=platform.machine(),
        single_worker=True, thread_environment={n: os.environ[n] for n in
        ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        registration=str(registration.relative_to(ROOT)), registration_sha256=sha(registration/'manifest.json'))
    save(output/'launch.json', launch)
    def execute(task, method, count, key):
        return measure(task, method, count, np.random.SeedSequence(cfg['root_seed'], spawn_key=key))
    warmup = []
    for mi, method in enumerate(METHODS):
        row = execute(tasks[0], method, 1000, (0, tasks[0]['case'], mi))
        warmup.append({k: v for k, v in row.items() if k not in ('policies', 'approximation_fingerprint')})
    save(output/'warmup.json', warmup)
    pilot = []
    for task in tasks:
        for rep in range(cfg['pilot_repetitions']):
            for ni, count in enumerate(cfg['pilot_paths']):
                order = [(rep+ni+j) % len(METHODS) for j in range(len(METHODS))]
                for mi in order:
                    row = execute(task, METHODS[mi], count, (1, task['case'], rep, ni, mi))
                    # Only timings and stream identities enter allocation. No pilot outcomes are retained or inspected.
                    pilot.append({k: v for k, v in row.items() if k not in ('policies', 'approximation_fingerprint')})
        save(output/'pilot.json', pilot)
        print(f'Timing pilot completed for model {task["case"]}', flush=True)
    allocations = allocation_table(pilot, cfg)
    save(output/'allocations.json', allocations)
    save(output/'allocation_lock.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        pilot_sha256=sha(output/'pilot.json'), allocations_sha256=sha(output/'allocations.json'),
        source_sha256=manifest['source_sha256']))
    if any(r['status'] != 'allocated' for r in allocations):
        raise ValueError('Infeasible budget retained; do not change the design within this run')
    indexed = {(r['case'], r['budget_seconds'], r['method']): r for r in allocations}
    rows = []
    for task in tasks:
        for rep in range(cfg['evaluation_repetitions']):
            for bi, budget in enumerate(cfg['target_seconds']):
                order = [(rep+bi+j) % len(METHODS) for j in range(len(METHODS))]
                for mi in order:
                    allocation = indexed[task['case'], budget, METHODS[mi]]
                    row = execute(task, METHODS[mi], allocation['paths'], (2, task['case'], rep, bi, mi))
                    row.update(budget_seconds=budget, evaluation_repetition=rep,
                               predicted_seconds=allocation['predicted_seconds'])
                    rows.append(row)
        save(output/'partial_rows.json', rows)
        print(f'Budget evaluation completed for model {task["case"]}', flush=True)
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    if sha(output/'allocations.json') != read(output/'allocation_lock.json')['allocations_sha256']:
        raise ValueError('Frozen allocation changed')
    save(output/'rows.json', rows)
    save(output/'summary.json', summarise(rows))
    names = ('launch.json', 'warmup.json', 'pilot.json', 'allocations.json', 'allocation_lock.json', 'rows.json', 'summary.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        registration=launch['registration'], registration_sha256=launch['registration_sha256'],
        source_sha256=manifest['source_sha256'], input_sha256=manifest['input_sha256'],
        artifact_sha256={n: sha(output/n) for n in names}, publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('register')
    p.add_argument('--configuration', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('run')
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        register(args.configuration, args.protocol, args.output)
    else:
        run(args.registration, args.output)
