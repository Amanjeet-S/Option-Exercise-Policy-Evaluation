"""Matched retraining against an integrated terminal reward."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from study_terminal_repair import verified_results as prior_results
from study_expiry_extension import scenario_calendars, YEAR
from study_policy_input_sensitivity import restore_scenario
from study_value_martingale import serialise_dual
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import simulate_paths, prefix_paths
from qf_research.optimal_stopping.terminal_training import fit_terminal_policy
from qf_research.optimal_stopping.terminal_repair import evaluate_terminal_repair
from qf_research.optimal_stopping.expiry_extension import restrict_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples

CONFIG=PROJECT/'configs/terminal_training_2026-09-20.json'


def planned_inputs():
    cfg=read(CONFIG); prior=ROOT/cfg['prior_run']; rows,_=prior_results(prior)
    if cfg['extensions_seconds']!=[0,900,5400] or cfg['bit_generator']!='PCG64' or cfg['family_size']!=3:
        raise ValueError('Unsupported matched-training protocol')
    check=read(ROOT/cfg['prior_verification'])
    if check['experiment_manifest_sha256']!=sha(prior/'manifest.json'):
        raise ValueError('Previous experiment changed')
    verify_mapping(check['source_sha256'])
    manifest=read(prior/'manifest.json')
    sources={**manifest['source_sha256'],**check['source_sha256']}; inputs=dict(manifest['input_sha256'])
    inputs.update({str((prior/name).relative_to(ROOT)):digest for name,digest in manifest['artifact_sha256'].items()})
    for p in (CONFIG,prior/'manifest.json',ROOT/cfg['prior_verification']): inputs[str(p.relative_to(ROOT))]=sha(p)
    for name in ('scripts/study_terminal_training.py','src/qf_research/optimal_stopping/terminal_training.py'): sources[name]=sha(ROOT/name)
    tasks=[]
    for old,row in zip(read(prior/'tasks.json'),rows):
        tasks.append({**old,'configuration':cfg,'previous_streams':sorted(set(old['previous_streams'])|{row['stream']['stream_id']})})
    return tasks,inputs,sources


def train_case(task):
    """Reproduce original coefficients before changing any terminal target."""
    model,option,rights=restore_scenario(task['baseline']); records=[]; objects=[]
    for repetition in task['policy_rows']:
        stream=next(s for s in repetition['streams'] if s['role']=='training')
        seed=np.random.SeedSequence(stream['entropy'],spawn_key=tuple(stream['spawn_key']))
        training=simulate_paths(model,option,rights,stream['paths'],np.random.Generator(np.random.PCG64(seed)))
        if training.stream_id!=stream['stream_id']: raise ValueError('Original training stream differs')
        for source in repetition['policies']:
            original=load_frozen(source)[0]
            subset=prefix_paths(training,original.training_paths)
            settings=dict(minimum_itm_paths=original.minimum_itm_paths,condition_limit=original.condition_limit)
            start=time.perf_counter()
            baseline=fit_terminal_policy(option,subset,original.requested_degree,0.,**settings)
            baseline_seconds=time.perf_counter()-start
            if baseline.policy.fingerprint!=original.fingerprint:
                raise ValueError('Original regression coefficients do not reproduce')
            fitted={0:baseline}; retained=[]
            for seconds in task['configuration']['extensions_seconds']:
                start=time.perf_counter()
                if seconds: fitted[seconds]=fit_terminal_policy(option,subset,original.requested_degree,seconds/YEAR,**settings)
                elapsed=time.perf_counter()-start if seconds else baseline_seconds
                trained=fitted[seconds]
                retained.append(dict(extension_seconds=seconds,extension=trained.extension,policy=asdict(trained.policy),
                    fingerprint=trained.fingerprint,policy_fingerprint=trained.policy.fingerprint,training_seconds=elapsed))
            records.append(dict(repetition=repetition['repetition'],training_paths=original.training_paths,degree=original.requested_degree,
                original_fingerprint=original.fingerprint,training_stream=stream,scenarios=retained))
            objects.append((original,fitted))
    return records,objects


def training_record(records):
    return [{**r,'scenarios':[{k:v for k,v in s.items() if k!='training_seconds'} for s in r['scenarios']]} for r in records]


def evaluate(task):
    cfg=task['configuration']; model,option,rights=restore_scenario(task['baseline']); factor=model.premium_factor
    training,policies=train_case(task)  # All policies frozen before fresh evaluation draws.
    calendars=scenario_calendars(model,option,rights,cfg)
    start=time.perf_counter()
    approximations={seconds:build_values(model,contract,calendar,**cfg['value_approximation']) for seconds,contract,calendar in calendars}
    build_seconds=time.perf_counter()-start
    for ref in task['references']:
        if approximations[ref['extension_seconds']].fingerprint!=ref['approximation_fingerprint']:
            raise ValueError('Value approximation changed')
    seed=np.random.SeedSequence(cfg['root_seed'],spawn_key=(task['case'],))
    start=time.perf_counter()
    observations=simulate_paths(model,calendars[-1][1],observation_rights(approximations[5400]),cfg['evaluation_paths'],np.random.Generator(np.random.PCG64(seed)))
    simulation_seconds=time.perf_counter()-start
    if observations.stream_id in task['previous_streams']: raise ValueError('Evaluation stream reused')
    rows=[]
    for (seconds,contract,_),ref in zip(calendars,task['references']):
        a=approximations[seconds]; restricted=restrict_paths(observations,contract,observation_rights(a))
        start=time.perf_counter(); sample=value_martingale(a,restricted); martingale_seconds=time.perf_counter()-start
        start=time.perf_counter(); records=[]
        for entry,(original,fitted) in zip(training,policies):
            old=evaluate_terminal_repair(original,sample,seconds/YEAR)
            new=evaluate_terminal_repair(fitted[seconds].policy,sample,seconds/YEAR) if seconds else old
            if not np.array_equal(old.repaired.upper_payoffs,new.repaired.upper_payoffs):
                raise ValueError('Comparison changed the common upper payoff')
            gain=(old.repaired.gap_payoffs-new.repaired.gap_payoffs)/factor
            duals={name:serialise_dual(result.repaired,factor) for name,result in [('terminal_only',old),('retrained',new)]}
            limits={name:family_upper_limit(result.repaired.gap_bound,cfg['family_size'],cfg['family_error_probability'])/factor
                    for name,result in [('terminal_only',old),('retrained',new)]}
            records.append(dict(repetition=entry['repetition'],training_paths=entry['training_paths'],degree=entry['degree'],
                original_fingerprint=original.fingerprint,retrained_fingerprint=fitted[seconds].fingerprint,
                duals=duals,family_upper_quote=limits,paired_gain=asdict(estimate_samples(gain)),
                changed_stops=int(np.count_nonzero(new.stopping_indices!=old.stopping_indices)),
                moved_earlier=int(np.count_nonzero(new.stopping_indices<old.stopping_indices)),
                moved_later=int(np.count_nonzero(new.stopping_indices>old.stopping_indices)),
                stopping_counts={name:np.bincount(result.stopping_indices,minlength=len(rights)).tolist()
                                 for name,result in [('terminal_only',old),('retrained',new)]}))
        rows.append(dict(**ref,policies=records,martingale_seconds=martingale_seconds,policy_evaluation_seconds=time.perf_counter()-start))
    spread=task['baseline']['fit']['put_ask']-task['baseline']['fit']['put_bid']; families=[]
    for i,entry in enumerate(training):
        maximum={name:max(s['policies'][i]['family_upper_quote'][name] for s in rows) for name in ('terminal_only','retrained')}
        families.append(dict(repetition=entry['repetition'],training_paths=entry['training_paths'],degree=entry['degree'],
            maximum_upper_quote=maximum,below_spread={name:value<spread for name,value in maximum.items()}))
    return dict(case=task['case'],premium_factor=factor,spread=spread,training=training,scenarios=rows,families=families,
        stream=dict(entropy=seed.entropy,spawn_key=list(seed.spawn_key),stream_id=observations.stream_id,paths=cfg['evaluation_paths']),
        build_seconds=build_seconds,simulation_seconds=simulation_seconds,refit=False)


def numerical_record(row):
    return {**{k:v for k,v in row.items() if k not in ('build_seconds','simulation_seconds','training','scenarios')},
        'training':training_record(row['training']),
        'scenarios':[{k:v for k,v in s.items() if k not in ('martingale_seconds','policy_evaluation_seconds')} for s in row['scenarios']]}


def summarise(rows):
    if not rows or len({r['case'] for r in rows})!=len(rows) or len({r['stream']['stream_id'] for r in rows})!=len(rows):
        raise ValueError('Unique cases and streams required')
    scenarios=[]
    for seconds in (0,900,5400):
        group=[(r,next(s for s in r['scenarios'] if s['extension_seconds']==seconds)) for r in rows]; variants=[]
        for n,d in sorted({(p['training_paths'],p['degree']) for _,s in group for p in s['policies']}):
            chosen=[(r,p) for r,s in group for p in s['policies'] if (p['training_paths'],p['degree'])==(n,d)]
            variants.append(dict(training_paths=n,degree=d,policies=len(chosen),mean_gain=float(np.mean([p['paired_gain']['value'] for _,p in chosen])),
                gain_range=[min(p['paired_gain']['value'] for _,p in chosen),max(p['paired_gain']['value'] for _,p in chosen)],
                positive_intervals=sum(p['paired_gain']['lower']>0 for _,p in chosen),negative_intervals=sum(p['paired_gain']['upper']<0 for _,p in chosen),
                mean_gap_quote={name:float(np.mean([p['duals'][name]['gap_bound']['value']/r['premium_factor'] for r,p in chosen])) for name in ('terminal_only','retrained')},
                adjusted_below_spread={name:sum(p['family_upper_quote'][name]<r['spread'] for r,p in chosen) for name in ('terminal_only','retrained')},
                mean_changed_stop_fraction=float(np.mean([p['changed_stops']/r['stream']['paths'] for r,p in chosen]))))
        scenarios.append(dict(extension_seconds=seconds,variants=variants))
    families=[f for r in rows for f in r['families']]
    return dict(models=len(rows),policy_scenarios=sum(len(s['policies']) for r in rows for s in r['scenarios']),
        scenarios=scenarios,families=[dict(training_paths=n,degree=d,policies=sum((f['training_paths'],f['degree'])==(n,d) for f in families),
            below_spread={name:sum(f['below_spread'][name] for f in families if (f['training_paths'],f['degree'])==(n,d)) for name in ('terminal_only','retrained')})
            for n,d in sorted({(f['training_paths'],f['degree']) for f in families})],
        empirical_calibration_accepted=False,publication_clearance=False)


def verified_results(folder):
    folder=private(folder); manifest=read(folder/'manifest.json')
    for key in ('input_sha256','source_sha256'): verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'],folder)
    rows=read(folder/'rows.json'); tasks=read(folder/'tasks.json')
    if [r['case'] for r in rows]!=[t['case'] for t in tasks] or summarise(rows)!=read(folder/'summary.json'):
        raise ValueError('Experiment coverage or summary differs')
    return rows,read(folder/'summary.json')


def run(output,workers):
    output=private(output)
    if output.exists(): raise FileExistsError('Use a fresh private output directory')
    if not 1<=workers<=2: raise ValueError('Use one or two workers')
    tasks,inputs,sources=planned_inputs(); output.mkdir(parents=True)
    save(output/'tasks.json',tasks);save(output/'configuration.json',read(CONFIG))
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,source_sha256=sources))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows);print(f'Completed {len(rows)}/{len(tasks)} matched-training cases',flush=True)
    verify_mapping(inputs);verify_mapping(sources)
    save(output/'rows.json',rows);save(output/'summary.json',summarise(rows))
    names=('tasks.json','configuration.json','launch.json','rows.json','summary.json')
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs,source_sha256=sources,artifact_sha256={name:sha(output/name) for name in names},publication_clearance=False))
    verified_results(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();run(args.output,args.workers)
