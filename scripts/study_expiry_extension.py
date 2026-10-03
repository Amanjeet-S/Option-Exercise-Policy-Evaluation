"""Compare later expiry decisions with unchanged models and frozen policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import numpy as np

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from study_policy_input_sensitivity import verified_results as prior_results, restore_scenario
from study_value_martingale import serialise_dual
from validate_funded_dual import load_frozen
from qf_research.computational.funded_kernel import NativeKernel, prepare
from qf_research.market_data.calendar_robustness import mesh_changes
from qf_research.optimal_stopping.expiry_extension import (
    price_terminal_extension, extended_calendar, restrict_paths, evaluate_embedded_policy, european_increment,
)
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples

CONFIG=PROJECT/'configs/expiry_extension_2026-09-20.json'
YEAR=365*86400


def planned_inputs():
    cfg=read(CONFIG); prior=ROOT/cfg['prior_run']
    previous,_=prior_results(prior)
    check=read(ROOT/cfg['prior_verification'])
    if check['experiment_manifest_sha256']!=sha(prior/'manifest.json'):
        raise ValueError('Preceding verified experiment changed')
    verify_mapping(check['source_sha256'])
    m=read(prior/'manifest.json'); inputs=dict(m['input_sha256']);sources={**m['source_sha256'],**check['source_sha256']}
    inputs.update({str((prior/k).relative_to(ROOT)):v for k,v in m['artifact_sha256'].items()})
    old_tasks=read(prior/'tasks.json')
    fit_tasks=read(prior.parent/'calibration_v1/tasks.json')
    source=ROOT/cfg['source_snapshot']; sm=read(source/'manifest.json')
    for row in sm['requests']:
        if row['status']!='retained' or sha(source/row['file'])!=row['sha256']:
            raise ValueError('Timing source snapshot incomplete or changed')
        inputs[str((source/row['file']).relative_to(ROOT))]=row['sha256']
    for p in (CONFIG, prior/'manifest.json', ROOT/cfg['prior_verification'], source/'manifest.json'):
        inputs[str(p.relative_to(ROOT))]=sha(p)
    if cfg['extensions_seconds'][0]!=0 or cfg['family_size']!=len(cfg['extensions_seconds']):
        raise ValueError('Invalid expiry scenario family')
    tasks=[]
    for old,row in zip(old_tasks,previous):
        baseline=old['scenarios'][0]
        if baseline['fraction']!=.5 or baseline['funding_path']!='midpoint' or not baseline['baseline_reproduced']:
            raise ValueError('Require unchanged baseline financial construction')
        fit=next(t for t in fit_tasks if t['case']==old['case'] and t['scenario']==baseline['scenario'])['fit_task']
        tasks.append(dict(case=old['case'],baseline=baseline,policy_rows=old['policy_rows'],configuration=cfg,
            library=fit['library'],native_tolerance=fit['native_tolerance'],
            previous_streams=sorted(set(old['previous_streams'])|{row['stream']['stream_id']}),
            repeat_python=old['case'] in (old_tasks[0]['case'],old_tasks[-1]['case'])))
    for name in ('scripts/study_expiry_extension.py','src/qf_research/optimal_stopping/expiry_extension.py'):
        sources[name]=sha(ROOT/name)
    return tasks,inputs,sources


def scenario_calendars(model,option,rights,cfg):
    result=[]; extra=[]
    for seconds in cfg['extensions_seconds']:
        if seconds:
            extra.append(seconds/YEAR)
            contract,calendar=extended_calendar(model,option,rights,tuple(extra))
        else:contract,calendar=option,rights
        result.append((seconds,contract,calendar))
    return result


def evaluate(task):
    cfg=task['configuration']; original=task['baseline']; fitted=original['fit']
    model,option,rights=restore_scenario(original)
    calendars=scenario_calendars(model,option,rights,cfg)
    entries=[(r['repetition'],p) for r in task['policy_rows'] for p in r['policies']]
    policies=[load_frozen(p)[0] for _,p in entries]
    if any((p.model,p.option,p.decisions)!=(model,option,rights) for p in policies):
        raise ValueError('Original financial model or frozen policy changed')
    native=NativeKernel(Path(task['library']))
    threshold=max(model.initial_state,*(option.strike*d.settlement_factor for d in rights))+sum(d.discounted_cash for d in model.dividends)
    prepared={name:prepare(model,option,g['space_steps'],g['time_steps'],g['domain_multiple']*threshold,rights)
              for name,g in cfg['meshes'].items()}
    approximations={seconds:build_values(model,contract,calendar,**cfg['value_approximation']) for seconds,contract,calendar in calendars}
    all_observations=observation_rights(approximations[cfg['extensions_seconds'][-1]])
    seed=np.random.SeedSequence(cfg['root_seed'],spawn_key=(task['case'],))
    draws=simulate_paths(model,calendars[-1][1],all_observations,cfg['evaluation_paths'],np.random.Generator(np.random.PCG64(seed)))
    if draws.stream_id in task['previous_streams']:
        raise ValueError('Previous random-stream role reused')
    baseline_gaps=[];raw_payoffs=[];rows=[]
    old_time=datetime.fromisoformat(original['cutoff'])+timedelta(seconds=option.maturity*YEAR)
    for seconds,contract,calendar in calendars:
        prices={name:price_terminal_extension(p,seconds/YEAR,native).premium_quote for name,p in prepared.items()}
        if seconds==0:
            tolerance=task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(prices['fine'])
            if abs(prices['fine']-fitted['put_prices'][1])>tolerance:
                raise ValueError('Original numerical price does not reproduce')
        repeats={}
        if task['repeat_python']:
            price=price_terminal_extension(prepared['fine'],seconds/YEAR).premium_quote
            tolerance=task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(price)
            if abs(price-prices['fine'])>tolerance:raise ValueError('Python/native terminal replacement differs')
            repeats=dict(python_price=price,native_price=prices['fine'],difference=prices['fine']-price,tolerance=tolerance)
        euro=european_increment(model,option,rights[-1],seconds/YEAR)
        call=european_increment(model,replace(option,kind='call'),rights[-1],seconds/YEAR)
        condition=terminal_call_dominance(model,replace(contract,kind='call'),calendar)
        if not condition.sufficient_condition_satisfied:raise ValueError('Terminal-call reduction unavailable')
        approx=approximations[seconds]
        sub=restrict_paths(draws,contract,observation_rights(approx))
        start=time.perf_counter();sample=value_martingale(approx,sub);shared_seconds=time.perf_counter()-start
        evaluations=[]
        for index,((rep,record),policy) in enumerate(zip(entries,policies)):
            result,realised=evaluate_embedded_policy(policy,sample)
            gap=result.gap_payoffs/model.premium_factor
            if seconds==0:
                baseline_gaps.append(gap.copy());raw_payoffs.append(realised.discounted_payoffs.copy())
            elif not np.array_equal(raw_payoffs[index],realised.discounted_payoffs):
                raise ValueError('Frozen policy payoff changed across expiry scenarios')
            evaluations.append(dict(repetition=rep,training_paths=policy.training_paths,degree=policy.requested_degree,
                fingerprint=record['fingerprint'],result=serialise_dual(result,model.premium_factor),
                family_adjusted_upper_quote=family_upper_limit(result.gap_bound,cfg['family_size'],cfg['family_error_probability'])/model.premium_factor,
                paired_gap_change=asdict(estimate_samples(gap-baseline_gaps[index])),
                stopping_counts=np.bincount(realised.stopping_indices,minlength=len(rights)).tolist()))
        changes=mesh_changes(prices);eta=fitted['numerical_allowance'];tol=fitted['call_tolerance']
        increment={k:v-rows[0]['prices'][k] for k,v in prices.items()} if rows else {k:0. for k in prices}
        screens=dict(space=all(abs(changes[k])<=eta for k in ('space_at_coarse_time','space_at_fine_time')),
            time=all(abs(changes[k])<=eta for k in ('time_at_coarse_space','time_at_fine_space')),
            domain=abs(changes['domain'])<=eta,
            increment_refinement=all(abs(v-increment['fine'])<=eta for v in increment.values()),
            european_increment=all(-2*eta<=v<=euro['increment_quote']+2*eta for v in increment.values()),
            call_reference=(call['expanded'].quadrature_error_estimate+call['expanded'].omitted_tail_bound)/model.premium_factor<=tol)
        call_value=call['expanded'].premium_quote
        call_fits=fitted['call_bid']<=call_value-tol and call_value+tol<=fitted['call_ask']
        put_fits=fitted['put_bid']<=prices['fine']-eta and prices['fine']+eta<=fitted['put_ask']
        rows.append(dict(extension_seconds=seconds,endpoint_utc=(old_time+timedelta(seconds=seconds)).isoformat(),
            decisions=[asdict(d) for d in calendar],maturity=contract.maturity,prices=prices,
            increment_prices=increment,mesh_changes=changes,european={k:asdict(v) if k in ('baseline','expanded') else v for k,v in euro.items()},
            call={k:asdict(v) if k in ('baseline','expanded') else v for k,v in call.items()},
            terminal_call_condition=asdict(condition),python_repeat=repeats,numerical_screens=screens,
            call_quote_with_margin=call_fits,put_quote_with_margin=put_fits,
            conditional_joint_compatibility=bool(call_fits and put_fits and all(screens.values())),
            approximation=asdict(approx),approximation_fingerprint=approx.fingerprint,
            shared_martingale_seconds=shared_seconds,policies=evaluations))
    families=[]
    for i,(rep,record) in enumerate(entries):
        maximum,seconds=max((r['policies'][i]['family_adjusted_upper_quote'],r['extension_seconds']) for r in rows)
        families.append(dict(repetition=rep,training_paths=record['training_paths'],degree=record['degree'],fingerprint=record['fingerprint'],
            maximum_upper_quote=maximum,maximising_extension_seconds=seconds,
            below_spread=maximum < fitted['put_ask']-fitted['put_bid']))
    return dict(case=task['case'],cutoff=original['cutoff'],strike=original['strike'],premium_factor=model.premium_factor,
        spread=fitted['put_ask']-fitted['put_bid'],stream=dict(entropy=seed.entropy,spawn_key=list(seed.spawn_key),stream_id=draws.stream_id,paths=cfg['evaluation_paths']),
        scenarios=rows,families=families,raw_policy_payoffs_identical=True,refit=False)


def numerical_record(row):
    return dict(row,scenarios=[{k:v for k,v in s.items() if k!='shared_martingale_seconds'} for s in row['scenarios']])


def summarise(rows):
    if not rows or len({r['case'] for r in rows})!=len(rows) or len({r['stream']['stream_id'] for r in rows})!=len(rows):
        raise ValueError('Unique model cases and independent stream identities required')
    result=[]
    for seconds in [s['extension_seconds'] for s in rows[0]['scenarios']]:
        group=[(r,next(s for s in r['scenarios'] if s['extension_seconds']==seconds)) for r in rows]
        variants=[]
        for n,d in sorted({(p['training_paths'],p['degree']) for _,s in group for p in s['policies']}):
            selected=[(r,p) for r,s in group for p in s['policies'] if (p['training_paths'],p['degree'])==(n,d)]
            variants.append(dict(training_paths=n,degree=d,policies=len(selected),
                mean_gap_quote=float(np.mean([p['result']['gap_bound']['value']/r['premium_factor'] for r,p in selected])),
                pointwise_below_spread=sum(p['result']['loss_upper_97_5_quote']<r['spread'] for r,p in selected),
                adjusted_below_spread=sum(p['family_adjusted_upper_quote']<r['spread'] for r,p in selected)))
        result.append(dict(extension_seconds=seconds,models=len(group),
            jointly_compatible=sum(s['conditional_joint_compatibility'] for _,s in group),
            numerical_failures=sum(not all(s['numerical_screens'].values()) for _,s in group),
            pde_increment_range=[min(s['increment_prices']['fine'] for _,s in group),max(s['increment_prices']['fine'] for _,s in group)],
            european_increment_bound_range=[min(s['european']['increment_quote'] for _,s in group),max(s['european']['increment_quote'] for _,s in group)],variants=variants))
    families=[f for r in rows for f in r['families']]
    return dict(models=len(rows),policy_evaluations=sum(len(s['policies']) for r in rows for s in r['scenarios']),
        scenarios=result,families=[dict(training_paths=n,degree=d,policies=sum((f['training_paths'],f['degree'])==(n,d) for f in families),
            bounds_below_spread=sum(f['below_spread'] for f in families if (f['training_paths'],f['degree'])==(n,d))) for n,d in sorted({(f['training_paths'],f['degree']) for f in families})],
        refit=False,empirical_calibration_accepted=False,publication_clearance=False)


def verified_results(folder):
    folder=private(folder);m=read(folder/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(m[k])
    verify_mapping(m['artifact_sha256'],folder)
    rows=read(folder/'rows.json');tasks=read(folder/'tasks.json')
    if [r['case'] for r in rows]!=[t['case'] for t in tasks] or summarise(rows)!=read(folder/'summary.json'):
        raise ValueError('Experiment coverage or summary differs')
    return rows,read(folder/'summary.json')


def run(output,workers):
    output=private(output)
    if output.exists():raise FileExistsError('Use a fresh private output directory')
    if not 1<=workers<=2:raise ValueError('Use one or two workers')
    tasks,inputs,sources=planned_inputs();output.mkdir(parents=True)
    save(output/'configuration.json',read(CONFIG));save(output/'tasks.json',tasks)
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,source_sha256=sources,workers=workers))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows)
            print(f'Completed {len(rows)}/{len(tasks)} expiry comparisons',flush=True)
    verify_mapping(inputs);verify_mapping(sources)
    save(output/'rows.json',rows);save(output/'summary.json',summarise(rows))
    names=('configuration.json','tasks.json','launch.json','rows.json','summary.json')
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs,source_sha256=sources,artifact_sha256={n:sha(output/n) for n in names},publication_clearance=False))
    verified_results(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();run(args.output,args.workers)
