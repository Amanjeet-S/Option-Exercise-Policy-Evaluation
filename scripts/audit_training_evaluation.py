"""Reconstruct the crossed experiment and repeat selected numerical blocks."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from run_conditional_calibration import private, read, save, sha, verify_mapping, ROOT
from study_training_evaluation import execute_block, numerical_record, summarise, METHODS
from qf_research.optimal_stopping.cost_budget import independent_difference
from qf_research.optimal_stopping.simulation import estimate_samples


def close(actual, expected):
    if not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=2e-11):
        raise ValueError('Statistical reconstruction differs')


def check_pair(row, count):
    e=row['estimate']
    if e['observations']!=count:raise ValueError('Paired sample count differs')
    close(e['value'],row['right_mean']-row['left_mean'])
    variance=row['right_variance']+row['left_variance']-2*row['covariance']
    if variance < -1e-9*max(1.,row['right_variance'],row['left_variance']):
        raise ValueError('Invalid covariance reconstruction')
    close(e['standard_error']**2,max(0.,variance)/count)
    radius=float(t.ppf(.975,count-1))*e['standard_error']
    close(e['lower'],e['value']-radius);close(e['upper'],e['value']+radius)


def verify(folder):
    folder=private(folder);manifest=read(folder/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(manifest[k])
    verify_mapping(manifest['artifact_sha256'],folder)
    registration=ROOT/manifest['registration']
    if sha(registration/'manifest.json')!=manifest['registration_sha256']:
        raise ValueError('Registration differs')
    registered=read(registration/'manifest.json')
    verify_mapping(registered['artifact_sha256'],registration)
    cfg=read(registration/'configuration.json');tasks=read(registration/'tasks.json')
    names=sorted(n for n in manifest['artifact_sha256'] if n.startswith('blocks/'))
    if names!=sorted(str(p.relative_to(folder)) for p in (folder/'blocks').glob('*.json')):
        raise ValueError('Unexpected or missing block files')
    blocks=[read(folder/n) for n in names]
    global_training=set();global_evaluation=set()
    pairs=0
    for b in blocks:
        training={r['training_paths']:r for r in b['training']}
        if set(training)!=set(cfg['training_paths']):raise ValueError('Training coverage differs')
        ids=[r['stream']['stream_id'] for r in training.values()]
        if len(set(ids))!=len(ids) or global_training.intersection(ids):raise ValueError('Training stream reused')
        global_training.update(ids)
        ws={(r['training_paths'],r['method'],r['evaluation_paths']):r for r in b['workloads']}
        expected={(n,m,e) for n in cfg['training_paths'] for m in METHODS for e in cfg['evaluation_paths']}
        if len(ws)!=len(b['workloads']) or set(ws)!=expected:raise ValueError('Workload coverage differs')
        for (n,m,e),r in ws.items():
            if r['policy_fingerprint']!=training[n]['fingerprint']:raise ValueError('Policy changed during evaluation')
            if r['minimum_gap_quote']<0:raise ValueError('Negative pathwise gap')
            close(r['total_elapsed_seconds'],r['elapsed_seconds']+training[n]['elapsed_seconds'])
            if r['elapsed_seconds']<=0 or sum(r['phases'].values())>r['elapsed_seconds']+1e-8:
                raise ValueError('Invalid timing accounting')
            estimates=r['estimates_quote']
            if any(x['observations']!=e for x in estimates.values()):raise ValueError('Sample size differs')
            close(estimates['gap_bound']['value'],estimates['upper_value']['value']-estimates['policy_value']['value'])
        for e in cfg['evaluation_paths']:
            selected=[ws[n,m,e] for n in cfg['training_paths'] for m in METHODS]
            streams={r['stream']['stream_id'] for r in selected}
            if len(streams)!=1 or len({r['paths_sha256'] for r in selected})!=1:
                raise ValueError('Pairing evidence differs')
            if streams & global_evaluation:raise ValueError('Evaluation stream reused across independent groups')
            global_evaluation.update(streams)
        for name in ('policy_gains','evaluation_effects','gain_consistency'):
            for r in b[name]:check_pair(r,r['evaluation_paths']);pairs+=1
        for r in b['sampling_effects']:
            lo,hi=cfg['evaluation_paths'];n,m=r['training_paths'],r['method']
            reference=independent_difference(ws[n,m,hi]['estimates_quote']['gap_bound'],ws[n,m,lo]['estimates_quote']['gap_bound'])
            if reference!=r['gap_mean_difference']:raise ValueError('Independent comparison differs')
    if global_training & global_evaluation:raise ValueError('Training/evaluation streams overlap')
    summary=read(folder/'summary.json')
    if summarise(blocks,cfg,[task['case'] for task in tasks])!=summary:raise ValueError('Summary differs')
    # Independently reconstruct the primary repetition inference and variance components.
    for model in summary['models']:
        chosen=[b for b in blocks if b['case']==model['case']]
        es=[next(g['estimate'] for g in b['policy_gains'] if g['method']=='fine' and
                 g['evaluation_paths']==cfg['evaluation_paths'][-1]) for b in chosen]
        means=np.array([e['value'] for e in es]);variance=means.var(ddof=1)
        rec=model['primary_training_gain_quote'];evaluation=np.mean([e['standard_error']**2 for e in es])
        if asdict(estimate_samples(means))!=rec['estimate']:raise ValueError('Repetition inference differs')
        close(rec['training_variance_moment_estimate'],variance-evaluation)
        close(rec['evaluation_variance_of_grand_mean'],evaluation/len(es))
    return blocks,cfg,tasks,dict(blocks=len(blocks),paired_reconstructions=pairs,
        training_streams=len(global_training),independent_evaluation_groups=len(global_evaluation))


def audit(folder,output,rerun=False):
    folder,output=private(folder),private(output)
    if output.exists():raise FileExistsError('Use a fresh audit directory')
    blocks,cfg,tasks,checks=verify(folder)
    output.mkdir(parents=True)
    repeated=[]
    if rerun:
        # Fixed coverage selection, not selection on performance or timing.
        for task in (tasks[0],tasks[-1]):
            for repetition in (0,cfg['training_repetitions']-1):
                row=execute_block(task,cfg,repetition)
                old=next(b for b in blocks if (b['case'],b['repetition'])==(task['case'],repetition))
                if numerical_record(row)!=numerical_record(old):raise ValueError('Numerical rerun differs')
                save(output/f'repeated_case_{task["case"]}_rep_{repetition}.json',row)
                repeated.append([task['case'],repetition])
                print(f'Reproduced model {task["case"]}, repetition {repetition+1}',flush=True)
    save(output/'verification.json',dict(verified_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(folder/'manifest.json'),checks=checks,
        repeated_blocks=repeated,full_numerical_rerun=False,timings_expected_to_match=False,
        source_sha256=read(folder/'manifest.json')['source_sha256'],publication_clearance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rerun',action='store_true')
    a=p.parse_args();audit(a.run,a.output,a.rerun)
