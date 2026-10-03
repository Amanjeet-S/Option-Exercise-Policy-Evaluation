"""Register, calibrate and execute a private allocation of policy evaluations."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import time

import numpy as np
import scipy

from audit_training_evaluation import verify as verify_previous, check_pair
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import (
    train_pair, build_approximation, evaluate_pair,
)
from qf_research.optimal_stopping.training_allocation import choose_allocation
from qf_research.optimal_stopping.training_evaluation import replicate_summary
from review_training_inference import delete_one

THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
SOURCES = ('scripts/study_training_allocation.py',
    'scripts/audit_training_allocation.py',
    'scripts/validate_funded_dual.py',
    'src/qf_research/optimal_stopping/allocation_workload.py',
    'src/qf_research/optimal_stopping/training_allocation.py',
    'tests/test_allocation_workload.py', 'tests/test_training_allocation.py',
    'tests/test_training_allocation_study.py', 'tests/test_training_allocation_audit.py',
    'scripts/review_training_inference.py',
    'reports/training_allocation.md')


def seed(cfg, *key):
    return np.random.SeedSequence(cfg['root_seed'], spawn_key=key)


def validate_config(cfg):
    expected = dict(training_paths=[5000,20000], degree=4, minimum_itm_paths=32,
        condition_limit=1e8, approximation=dict(core_steps=800,wing_steps=40,tolerance=.001),
        pilot_repetitions=32, pilot_paths=4096, timing_repetitions=5,
        evaluation_grid=[128,256,512,1024,2048,4096],
        reference_repetitions=64, reference_paths=4096, minimum_repetitions=64,
        maximum_repetitions=1024, root_seed=2026092700644096,
        budget_ratio_limits=[.9,1.1], pilot_cpu_limit=1200.,pilot_elapsed_limit=3600.,
        main_cpu_limit=5400.,main_elapsed_limit=14400.,publication_clearance=False)
    if cfg != expected:
        raise ValueError('Configuration differs from the declared allocation design')


def checked(folder):
    folder=private(folder); m=read(folder/'manifest.json')
    for key in ('input_sha256','source_sha256'):
        verify_mapping(m[key])
    verify_mapping(m['artifact_sha256'],folder)
    return m


def stage_registration(folder):
    m=checked(folder);cfg=read(folder/'configuration.json');validate_config(cfg)
    tasks=read(folder/'tasks.json')
    if len(tasks)!=13 or [t['case'] for t in tasks]!=list(range(13)):
        raise ValueError('The registered model coverage differs')
    return m,cfg,tasks


def register(previous,configuration,protocol,output):
    previous,configuration,protocol,output=map(private,(previous,configuration,protocol,output))
    if output.exists():raise FileExistsError('Use a fresh registration directory')
    cfg=read(configuration);validate_config(cfg)
    blocks,_,tasks,_=verify_previous(previous)
    prior=read(previous/'manifest.json')
    forbidden={s['stream']['stream_id'] for b in blocks for s in b['training']+b['workloads']}
    for task in tasks:
        task['prior_streams']=sorted(set(task['prior_streams'])|forbidden)
    sources=dict(prior['source_sha256'])
    for name in SOURCES:sources[name]=sha(ROOT/name)
    inputs=dict(prior['input_sha256'])
    for p in [previous/'manifest.json',configuration,protocol]:inputs[str(p.relative_to(ROOT))]=sha(p)
    verify_mapping(sources);verify_mapping(inputs)
    output.mkdir(parents=True)
    save(output/'configuration.json',cfg);save(output/'tasks.json',tasks)
    (output/'protocol.md').write_text(protocol.read_text())
    save(output/'manifest.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        stage='pilot_design',input_sha256=inputs,source_sha256=sources,
        artifact_sha256={n:sha(output/n) for n in ['configuration.json','tasks.json','protocol.md']},
        public_registration=False,publication_clearance=False))


def launch(registration,output,stage):
    registration,output=private(registration),private(output)
    if output.exists():raise FileExistsError('Use a fresh execution directory')
    m,cfg,tasks=stage_registration(registration)
    expected='pilot_design' if stage=='pilot' else 'main_allocation'
    if m.get('stage')!=expected:
        raise ValueError('Registration does not authorise this execution stage')
    if stage=='main':
        _,_,_,summaries,seen=verify_pilot(ROOT/m['pilot'])
        registered=read(registration/'allocations.json')
        if registered['models']!=summaries or registered['forbidden_streams']!=sorted(seen):
            raise ValueError('Main allocations differ from verified pilot calculations')
    if any(os.environ.get(k)!='1' for k in THREADS):
        raise ValueError('Set every recorded numerical-library thread environment variable to one')
    output.mkdir(parents=True);(output/'blocks').mkdir()
    record=dict(started_at=datetime.now(timezone.utc).isoformat(),stage=stage,
        registration=str(registration.relative_to(ROOT)),registration_sha256=sha(registration/'manifest.json'),
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),numpy_build=np.show_config(mode='dicts'),
        thread_environment={k:os.environ[k] for k in THREADS},python_workers=1,
        effective_native_thread_count=None)
    save(output/'launch.json',record)
    return m,cfg,tasks,time.process_time(),time.monotonic()


def guard(cfg,stage,cpu,wall):
    if time.process_time()-cpu>cfg[stage+'_cpu_limit'] or time.monotonic()-wall>cfg[stage+'_elapsed_limit']:
        raise RuntimeError('Declared resource limit reached; retained execution is incomplete')


def fit(base,cfg,role,case,rep,arm=0):
    return train_pair(base,tuple(cfg['training_paths']),degree=cfg['degree'],
        minimum_itm_paths=cfg['minimum_itm_paths'],condition_limit=cfg['condition_limit'],
        seeds=(seed(cfg,role,case,arm,rep,0),seed(cfg,role,case,arm,rep,1)))


def check_streams(training,evaluation,seen):
    ids=[s['stream_id'] for s in training['streams']]+[evaluation['stream']['stream_id']]
    if len(ids)!=len(set(ids)) or seen.intersection(ids):
        raise ValueError('Independent stream reused')
    seen.update(ids)


def warmup(base,cfg,stage_id=0):
    a,c=build_approximation(base,**cfg['approximation'])
    policies,t=train_pair(base,(512,1024),degree=cfg['degree'],
        minimum_itm_paths=cfg['minimum_itm_paths'],condition_limit=cfg['condition_limit'],
        seeds=(seed(cfg,0,stage_id,0),seed(cfg,0,stage_id,1)))
    e=evaluate_pair(policies,a,128,seed(cfg,0,stage_id,2))
    return dict(setup=c,training_cpu_seconds=t['cpu_seconds'],evaluation_cpu_seconds=e['cpu_seconds'],
                training_elapsed_seconds=t['elapsed_seconds'],evaluation_elapsed_seconds=e['elapsed_seconds'])


def infer(records,paths):
    estimates=[r['evaluation']['gain']['estimate'] for r in records]
    for e in estimates:
        if e['observations']!=paths:raise ValueError('Evaluation sample count differs')
    result=replicate_summary(estimates)
    result['evaluation_path_variance']=result['mean_conditional_evaluation_variance']*paths
    return result


def pilot_summary(row,cfg):
    expected=list(range(cfg['pilot_repetitions']))
    if [r['repetition'] for r in row['statistical']]!=expected:
        raise ValueError('Pilot repetition coverage differs')
    if len(row['setup'])!=cfg['timing_repetitions']:
        raise ValueError('Setup timing coverage differs')
    if len({r['approximation_fingerprint'] for r in row['setup']})!=1:
        raise ValueError('Repeated construction changed the value approximation')
    fingerprint=row['setup'][0]['approximation_fingerprint']
    for record in row['statistical']:
        training,evaluation=record['training'],record['evaluation']
        if training['training_paths']!=cfg['training_paths'] or evaluation['training_paths']!=cfg['training_paths']:
            raise ValueError('Training path counts differ')
        if training['policy_fingerprints']!=evaluation['policy_fingerprints']:
            raise ValueError('Evaluation does not use the fitted policies')
        if evaluation['approximation_fingerprint']!=fingerprint:
            raise ValueError('Evaluation approximation differs')
    index={(r['repetition'],r['evaluation']['evaluation_paths']):r for r in row['timing']}
    expected={(r,n) for r in range(cfg['timing_repetitions']) for n in cfg['evaluation_grid']}
    if len(index)!=len(row['timing']) or set(index)!=expected:
        raise ValueError('Timing grid coverage differs')
    for (rep,n),record in index.items():
        evaluation=record['evaluation'];training=row['statistical'][rep]['training']
        if (evaluation['training_paths']!=cfg['training_paths'] or
                evaluation['policy_fingerprints']!=training['policy_fingerprints'] or
                evaluation['approximation_fingerprint']!=fingerprint):
            raise ValueError('Timing evaluation does not match its registered pilot pair')
    costs={}
    for n in cfg['evaluation_grid']:
        costs[n]=median(row['statistical'][r]['training']['cpu_seconds']+
                        index[r,n]['evaluation']['cpu_seconds'] for r in range(cfg['timing_repetitions']))
    inference=infer(row['statistical'],cfg['pilot_paths'])
    allocation=choose_allocation(inference['training_variance_moment_estimate'],
        inference['evaluation_path_variance'],median(x['cpu_seconds'] for x in row['setup']),costs,
        reference_repetitions=cfg['reference_repetitions'],reference_paths=cfg['reference_paths'],
        min_repetitions=cfg['minimum_repetitions'],max_repetitions=cfg['maximum_repetitions'])
    return dict(case=row['case'],pilot_inference=inference,allocation=allocation)


def finish(registration,output,m,cpu,wall,extra=None):
    for key in ('input_sha256','source_sha256'):verify_mapping(m[key])
    verify_mapping(m['artifact_sha256'],registration)
    if sha(registration/'manifest.json')!=read(output/'launch.json')['registration_sha256']:
        raise ValueError('Registration changed during execution')
    files=[p for p in output.rglob('*.json') if p.name!='manifest.json']
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        process_cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-wall,
        registration=str(registration.relative_to(ROOT)),registration_sha256=sha(registration/'manifest.json'),
        input_sha256=m['input_sha256'],source_sha256=m['source_sha256'],
        artifact_sha256={str(p.relative_to(output)):sha(p) for p in sorted(files)},
        complete=True,publication_clearance=False,**(extra or {})))


def run_pilot(registration,output):
    registration,output=private(registration),private(output)
    m,cfg,tasks,cpu,wall=launch(registration,output,'pilot')
    (output/'checkpoints').mkdir()
    save(output/'warmup.json',warmup(load_frozen(tasks[0]['source_policy'])[0],cfg,stage_id=0))
    summaries=[];seen={s for t in tasks for s in t['prior_streams']}
    for task in tasks:
        base=load_frozen(task['source_policy'])[0];case=task['case'];setups=[]
        for _ in range(cfg['timing_repetitions']):
            approximation,cost=build_approximation(base,**cfg['approximation']);setups.append(cost)
        rows=[];timing=[];retained=[]
        for rep in range(cfg['pilot_repetitions']):
            guard(cfg,'pilot',cpu,wall)
            policies,training=fit(base,cfg,10,case,rep)
            evaluation=evaluate_pair(policies,approximation,cfg['pilot_paths'],seed(cfg,11,case,rep))
            check_streams(training,evaluation,seen)
            rows.append(dict(repetition=rep,training=training,evaluation=evaluation))
            save(output/'checkpoints'/f'case_{case:03d}_stat_{rep:03d}.json',rows[-1])
            if rep<cfg['timing_repetitions']:retained.append(policies)
        for rep,policies in enumerate(retained):
            grid=cfg['evaluation_grid'];offset=(case+rep)%len(grid)
            for n in grid[offset:]+grid[:offset]:
                guard(cfg,'pilot',cpu,wall)
                e=evaluate_pair(policies,approximation,n,seed(cfg,12,case,rep,grid.index(n)))
                identifier=e['stream']['stream_id']
                if identifier in seen:raise ValueError('Timing evaluation stream reused')
                seen.add(identifier);timing.append(dict(repetition=rep,evaluation=e))
                save(output/'checkpoints'/f'case_{case:03d}_timing_{rep:03d}_paths_{n:05d}.json',timing[-1])
        row=dict(case=case,setup=setups,statistical=rows,timing=timing)
        save(output/'blocks'/f'case_{case:03d}.json',row)
        summaries.append(pilot_summary(row,cfg))
        print(f'Completed pilot model {case}: {len(rows)} independent pairs and {len(timing)} timing measurements',flush=True)
    save(output/'summary.json',dict(models=summaries,statistical_pairs=sum(len(read(p)['statistical']) for p in sorted((output/'blocks').glob('*.json'))),
        publication_clearance=False,main_outcomes_inspected=False))
    guard(cfg,'pilot',cpu,wall)
    finish(registration,output,m,cpu,wall)


def verify_pilot(folder):
    folder=private(folder);m=checked(folder)
    registration=ROOT/m['registration'];_,cfg,tasks=stage_registration(registration)
    if (m.get('complete') is not True or read(folder/'launch.json').get('stage')!='pilot'
            or read(registration/'manifest.json').get('stage')!='pilot_design'):
        raise ValueError('A completed pilot execution is required')
    if sha(registration/'manifest.json')!=m['registration_sha256']:
        raise ValueError('Pilot registration identity differs')
    summaries=[];seen={s for t in tasks for s in t['prior_streams']}
    expected={f'case_{t["case"]:03d}.json' for t in tasks}
    if {p.name for p in (folder/'blocks').glob('*.json')}!=expected:
        raise ValueError('Pilot block-file coverage differs')
    for task in tasks:
        row=read(folder/'blocks'/f'case_{task["case"]:03d}.json')
        if row['case']!=task['case']:raise ValueError('Pilot model identity differs')
        for r in row['statistical']:
            if r!=read(folder/'checkpoints'/f'case_{task["case"]:03d}_stat_{r["repetition"]:03d}.json'):
                raise ValueError('Statistical checkpoint differs')
            check_streams(r['training'],r['evaluation'],seen)
            check_pair(r['evaluation']['gain'],cfg['pilot_paths'])
        for r in row['timing']:
            e=r['evaluation'];check_pair(e['gain'],e['evaluation_paths'])
            if r!=read(folder/'checkpoints'/f'case_{task["case"]:03d}_timing_{r["repetition"]:03d}_paths_{e["evaluation_paths"]:05d}.json'):
                raise ValueError('Timing checkpoint differs')
            if e['stream']['stream_id'] in seen:raise ValueError('Timing stream reused')
            seen.add(e['stream']['stream_id'])
        summaries.append(pilot_summary(row,cfg))
    if summaries!=read(folder/'summary.json')['models']:
        raise ValueError('Pilot summary or allocation differs')
    return m,cfg,tasks,summaries,seen


def allocate(pilot,output):
    pilot,output=private(pilot),private(output)
    if output.exists():raise FileExistsError('Use a fresh main registration directory')
    m,cfg,tasks,summaries,seen=verify_pilot(pilot)
    old=ROOT/m['registration'];design=read(old/'manifest.json')
    output.mkdir(parents=True)
    save(output/'configuration.json',cfg);save(output/'tasks.json',tasks)
    save(output/'allocations.json',dict(models=summaries,forbidden_streams=sorted(seen),
        estimand='Mean quoted policy-value gain of the fixed independently sampled training procedures',
        main_outcomes_inspected=False,publication_clearance=False))
    (output/'protocol.md').write_text((old/'protocol.md').read_text())
    inputs=dict(design['input_sha256']);inputs[str((pilot/'manifest.json').relative_to(ROOT))]=sha(pilot/'manifest.json')
    save(output/'manifest.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),stage='main_allocation',
        pilot=str(pilot.relative_to(ROOT)),input_sha256=inputs,source_sha256=design['source_sha256'],
        artifact_sha256={n:sha(output/n) for n in ['configuration.json','tasks.json','protocol.md','allocations.json']},
        public_registration=False,publication_clearance=False))
    print('Main counts frozen for every retained model before independent main draws.')


def main_summary(blocks,setup,cfg):
    if not blocks:raise ValueError('No completed main repetitions')
    n=blocks[0]['evaluation']['evaluation_paths'];result=infer(blocks,n)
    cpu=setup['cpu_seconds']+sum(r['training']['cpu_seconds']+r['evaluation']['cpu_seconds'] for r in blocks)
    elapsed=setup['elapsed_seconds']+sum(r['training']['elapsed_seconds']+r['evaluation']['elapsed_seconds'] for r in blocks)
    diagnostic=delete_one([dict(repetition=r['repetition'],value=r['evaluation']['gain']['estimate']['value'],
        standard_error=r['evaluation']['gain']['estimate']['standard_error']) for r in blocks])
    return dict(inference=result,standalone_cpu_seconds=cpu,standalone_elapsed_seconds=elapsed,
        zero_gain_sample_pairs=sum(r['evaluation']['nonzero_gain_count']==0 for r in blocks),
        influence=diagnostic)


def run_main(registration,output):
    registration,output=private(registration),private(output)
    m,cfg,tasks,cpu,wall=launch(registration,output,'main')
    if m.get('stage')!='main_allocation':raise ValueError('Main allocation is not registered')
    allocations=read(registration/'allocations.json');bycase={r['case']:r for r in allocations['models']}
    seen=set(allocations['forbidden_streams']);models=[]
    save(output/'warmup.json',warmup(load_frozen(tasks[0]['source_policy'])[0],cfg,stage_id=1))
    for task in tasks:
        case=task['case'];base=load_frozen(task['source_policy'])[0];arms={}
        names=('reference','selected') if case%2==0 else ('selected','reference')
        for arm in names:
            arm_id=('reference','selected').index(arm)
            allocation=bycase[case]['allocation'][arm];R=allocation['repetitions'];N=allocation['evaluation_paths']
            approximation,setup=build_approximation(base,**cfg['approximation']);blocks=[]
            for rep in range(R):
                guard(cfg,'main',cpu,wall)
                policies,training=fit(base,cfg,20,case,rep,arm_id)
                evaluation=evaluate_pair(policies,approximation,N,seed(cfg,21,case,arm_id,rep))
                check_streams(training,evaluation,seen)
                block=dict(case=case,arm=arm,repetition=rep,training=training,evaluation=evaluation)
                save(output/'blocks'/f'case_{case:03d}_{arm}_rep_{rep:04d}.json',block);blocks.append(block)
            row=main_summary(blocks,setup,cfg);budget=bycase[case]['allocation']['budget']['cpu_seconds']
            ratio=row['standalone_cpu_seconds']/budget
            row.update(setup=setup,allocation=allocation,actual_to_budget_ratio=ratio,
                       material_budget_mismatch=not cfg['budget_ratio_limits'][0]<=ratio<=cfg['budget_ratio_limits'][1])
            arms[arm]=row
            print(f'Completed main model {case}, {arm}: {R} pairs, {N} evaluation paths',flush=True)
        a,b=arms['reference'],arms['selected']
        va=a['inference']['estimate']['standard_error']**2;vb=b['inference']['estimate']['standard_error']**2
        comparison=dict(estimated_variance_ratio=vb/va if va>0 else None,
            actual_cpu_ratio=b['standalone_cpu_seconds']/a['standalone_cpu_seconds'],
            estimated_variance_cpu_ratio=vb*b['standalone_cpu_seconds']/(va*a['standalone_cpu_seconds']) if va>0 else None,
            measured_mean_squared_error=False,validated_interval_coverage=False)
        models.append(dict(case=case,arms=arms,comparison=comparison))
        save(output/f'model_{case:03d}.json',models[-1])
    save(output/'summary.json',dict(models=models,publication_clearance=False,independent_market_validation=False))
    guard(cfg,'main',cpu,wall)
    finish(registration,output,m,cpu,wall)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register')
    for key in ('previous','configuration','protocol','output'):p.add_argument('--'+key,type=Path,required=True)
    for name in ('pilot','main'):
        p=sub.add_parser(name);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('allocate');p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify-pilot');p.add_argument('--pilot',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='register':register(args.previous,args.configuration,args.protocol,args.output)
    elif args.command=='pilot':run_pilot(args.registration,args.output)
    elif args.command=='main':run_main(args.registration,args.output)
    elif args.command=='allocate':allocate(args.pilot,args.output)
    else:
        _,cfg,tasks,_,_=verify_pilot(args.pilot)
        print(f'Verified {len(tasks)} models and {cfg["pilot_repetitions"]} statistical repetitions per model.')
