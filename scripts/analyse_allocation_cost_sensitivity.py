"""Bind retained costs and compute retrospective fixed-budget allocation sensitivity."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from qf_research.optimal_stopping.allocation_sensitivity import choose, evaluate, frontier

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/private/allocation_cost_sensitivity/2026-09-27'
TIMING=ROOT/'data/private/cpu_budget_transfer/2026-09-27'
ORIGINAL=ROOT/'data/private/training_allocation/2026-09-27'
GRID=(128,256,4096)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value


def save(path,value):
    path.write_text(json.dumps(encode(value),indent=2,allow_nan=False)+'\n')


def private(path):
    path=path.resolve()
    if not path.is_relative_to(ROOT/'data/private'):
        raise ValueError('Keep this analysis in private storage')
    return path


def verify(mapping,base=ROOT):
    for name,digest in mapping.items():
        path=(base/name).resolve()
        if not path.is_relative_to(base.resolve()) or sha(path)!=digest:
            raise ValueError('Changed or unsafe evidence: '+name)


def manifest(folder,evidence):
    value=read(folder/'manifest.json')
    for group,base in (('input_sha256',ROOT),('source_sha256',ROOT),('artifact_sha256',folder)):
        verify(value[group],base)
        for name,digest in value[group].items():
            relative=str((base/name).relative_to(ROOT))
            if relative in evidence and evidence[relative]!=digest:raise ValueError('Conflicting evidence')
            evidence[relative]=digest
    evidence[str((folder/'manifest.json').relative_to(ROOT))]=sha(folder/'manifest.json')
    return value


def prepare(output):
    output=private(output)
    if output.exists():raise FileExistsError('Use a fresh input directory')
    evidence={}
    completion=TIMING/'local_timing_completion_v1.json'
    previous=read(completion)
    if previous['complete'] is not True:raise ValueError('Complete timing stage required')
    verify(previous['evidence_sha256'])
    evidence.update(previous['evidence_sha256'])
    evidence[str(completion.relative_to(ROOT))]=sha(completion)
    for folder in (ORIGINAL/'main_registration_v1',ORIGINAL/'pilot_v1',TIMING/'local_timing_registration_v1',TIMING/'local_timing_run_v1'):
        m=manifest(folder,evidence)
        if folder.name in ('pilot_v1','local_timing_run_v1') and m['complete'] is not True:
            raise ValueError('Completed evidence required')
    registration=read(ORIGINAL/'main_registration_v1/allocations.json')
    pilot=read(ORIGINAL/'pilot_v1/summary.json')
    if registration['models']!=pilot['models']:raise ValueError('Frozen variance or allocations changed')
    tasks=read(ORIGINAL/'main_registration_v1/tasks.json')
    recent=read(TIMING/'local_timing_registration_v1/tasks.json')
    if [t['source_policy'] for t in tasks]!=[t['source_policy'] for t in recent]:
        raise ValueError('Financial cases differ between studies')
    cfg=read(BASE/'design_v1/configuration.json')
    if cfg['cases']!=list(range(13)) or cfg['evaluation_grid']!=list(GRID) or cfg['minimum_repetitions']!=64 or cfg['maximum_repetitions']!=1024:
        raise ValueError('Analysis grid differs')
    schedule=read(TIMING/'local_timing_registration_v1/schedule.json')
    lookup={(r['case'],r['evaluation_paths']):r['position'] for r in schedule}
    models=[]
    for original in registration['models']:
        case=original['case'];allocation=original['allocation'];stat=original['pilot_inference']
        A,B=Fraction(stat['training_variance_moment_estimate']),Fraction(stat['evaluation_path_variance'])
        scenarios={q:{} for q in ('pilot','validation')};screening=Fraction(0)
        for n in GRID:
            folder=TIMING/'local_timing_run_v1'/f'cell_{lookup[case,n]:02d}'
            for q,count in (('pilot',8),('validation',64)):
                rows=[read(folder/f'{q}_{rep:03d}.json') for rep in range(count)]
                if any((r['case'],r['evaluation_paths'],r['phase'],r['repetition'])!=(case,n,q,rep) for rep,r in enumerate(rows)):
                    raise ValueError('Timing cell coverage differs')
                setup=Fraction(read(folder/f'{q}_setup.json')['cpu_seconds'])
                total=sum((Fraction(r['pair_cpu_seconds']) for r in rows),Fraction(0))
                scenarios[q][n]=dict(setup=setup,pair=total/count)
                if q=='pilot':screening+=setup+total
        models.append(dict(case=case,A=A,B=B,budget=Fraction(allocation['budget']['cpu_seconds']),
            original={name:{k:allocation[name][k] for k in ('evaluation_paths','repetitions')}
                      for name in ('reference','selected')},scenarios=scenarios,screening_charge=screening,
            variance_estimates_resolved=A>0 and B>0))
    if [m['case'] for m in models]!=list(range(13)):raise ValueError('All original cases required')
    for path in (BASE/'design_v1/configuration.json',BASE/'design_v1/protocol.md',BASE/'theory_v1/finite_cost_scenarios.md',
                 Path(__file__).resolve(),ROOT/'src/qf_research/optimal_stopping/allocation_sensitivity.py',
                 ROOT/'tests/test_allocation_sensitivity.py'):
        evidence[str(path.relative_to(ROOT))]=sha(path)
    verify(evidence)
    output.mkdir(parents=True)
    save(output/'inputs.json',dict(schema=1,configuration=cfg,models=models,retrospective=True))
    save(output/'manifest.json',dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_sha256=evidence,artifact_sha256={'inputs.json':sha(output/'inputs.json')},
        new_financial_simulations=0,new_financial_estimates=0))


def load_inputs(folder):
    folder=private(folder);m=read(folder/'manifest.json')
    verify(m['evidence_sha256']);verify(m['artifact_sha256'],folder)
    if m['complete'] is not True:raise ValueError('Complete input record required')
    inputs=read(folder/'inputs.json')
    for row in inputs['models']:
        for field in ('A','B','budget','screening_charge'):row[field]=Fraction(row[field])
        row['scenarios']={q:{int(n):{k:Fraction(v) for k,v in values.items()} for n,values in costs.items()}
                          for q,costs in row['scenarios'].items()}
    return inputs,m


def run(inputs,output):
    inputs,output=private(inputs),private(output)
    if output.exists():raise FileExistsError('Use a fresh analysis directory')
    payload,source=load_inputs(inputs)
    output.mkdir(parents=True)
    results=[]
    try:
        for model in payload['models']:
            case,A,B,budget,scenarios=(model[k] for k in ('case','A','B','budget','scenarios'))
            record=dict(case=case,status='resolved' if model['variance_estimates_resolved'] else 'unresolved_variance',accounting={})
            if not model['variance_estimates_resolved']:
                results.append(record);continue
            for accounting,H in (('marginal',Fraction(0)),('screening_charged',model['screening_charge'])):
                choices,curves={},{}
                for scope in ('pilot','validation','joint'):
                    costs=scenarios if scope=='joint' else {scope:scenarios[scope]}
                    choices[scope]=choose(A,B,budget,costs,H)
                    curves[scope]=frontier(A,B,costs,H)
                    prior=[r for r in curves[scope] if r['required_budget']<=budget]
                    chosen=choices[scope]['selected']
                    if bool(prior)!=(chosen is not None) or (prior and any(prior[-1][k]!=chosen[k] for k in ('evaluation_paths','repetitions','variance'))):
                        raise ValueError('Frontier and direct integer selection differ')
                cross={}
                for name,original in model['original'].items():
                    cross['original_'+name]=evaluate(A,B,original['evaluation_paths'],original['repetitions'],budget,scenarios,H)
                for name,result in choices.items():
                    chosen=result['selected']
                    cross[name+'_choice']=(evaluate(A,B,chosen['evaluation_paths'],chosen['repetitions'],budget,scenarios,H)
                                           if chosen else None)
                joint=choices['joint']['selected']
                matched=next(r for r in choices['joint']['candidates'] if r['evaluation_paths']==4096)
                penalties={q:(joint['variance']/choices[q]['selected']['variance'] if joint and choices[q]['selected'] else None)
                           for q in ('pilot','validation')}
                record['accounting'][accounting]=dict(overhead=H,choices=choices,cross_evaluation=cross,
                    joint_to_single_scenario_variance_ratio=penalties,
                    joint_to_original_reference_variance_ratio=(joint['variance']/cross['original_reference']['variance'] if joint else None),
                    joint_to_budget_matched_reference_variance_ratio=(joint['variance']/matched['variance'] if joint and matched['feasible'] else None))
                save(output/f'case_{case:02d}_{accounting}_frontiers.json',curves)
            # A common acquisition charge translates the full frontier exactly.
            base=read(output/f'case_{case:02d}_marginal_frontiers.json')
            charged=read(output/f'case_{case:02d}_screening_charged_frontiers.json')
            for scope in base:
                if len(base[scope])!=len(charged[scope]):raise ValueError('Frontier translation length differs')
                for x,y in zip(base[scope],charged[scope]):
                    if any(x[k]!=y[k] for k in ('evaluation_paths','repetitions','variance','strict_variance_improvement')) or Fraction(y['required_budget'])-Fraction(x['required_budget'])!=model['screening_charge']:
                        raise ValueError('Common-charge frontier translation failed')
            save(output/f'case_{case:02d}.json',record);results.append(record)
        summary=dict(complete=True,cases=len(results),retrospective=True,new_financial_simulations=0,new_financial_estimates=0,
                     frozen_original_budgets=True,frontier_translation_verified=True,models=results)
        save(output/'summary.json',summary)
        load_inputs(inputs)
        evidence={**source['evidence_sha256'],str((inputs/'manifest.json').relative_to(ROOT)):sha(inputs/'manifest.json'),
                  str((inputs/'inputs.json').relative_to(ROOT)):sha(inputs/'inputs.json')}
        save(output/'manifest.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
            evidence_sha256=evidence,artifact_sha256={p.name:sha(p) for p in sorted(output.glob('*.json'))},
            retrospective=True,new_financial_simulations=0,new_financial_estimates=0,publication_clearance=False))
        print(json.dumps(dict(complete=True,cases=len(results),new_financial_simulations=0)))
    except BaseException as exc:
        save(output/'incomplete.json',dict(complete=False,reason=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    prepare_parser=commands.add_parser('prepare');prepare_parser.add_argument('--output',type=Path,required=True)
    runner=commands.add_parser('run');runner.add_argument('--inputs',type=Path,required=True);runner.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:run(args.inputs,args.output)
