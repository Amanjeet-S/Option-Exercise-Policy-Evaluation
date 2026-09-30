"""Freeze constructed joint fits and refine their exercise opportunities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
import time

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from construct_joint_funding_witnesses import verified_results as witness_results
from cover_joint_cash_funding import model_at
from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.exercise_calendar_sensitivity import core_session_schedules
from qf_research.market_data.calendar_robustness import compare_calendars
from qf_research.market_data.settlement_comparison import restore_problem
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_cash import european_reference, finite_difference_price
from qf_research.computational.funded_kernel import NativeKernel, prepare, price_prepared

CONFIG=PROJECT/'configs/fitted_calendar_robustness_2026-09-20.json'


def planned_inputs():
    cfg=read(CONFIG);prior=ROOT/cfg['witness_run'];rows,summary=witness_results(prior)
    check=read(ROOT/cfg['prior_verification'])
    if check['witness_manifest_sha256']!=sha(prior/'manifest.json'):raise ValueError('Verified candidate experiment changed')
    verify_mapping(check['source_sha256'])
    m=read(prior/'manifest.json');inputs=dict(m['input_sha256']);sources=dict(m['source_sha256'])
    inputs.update({str((prior/n).relative_to(ROOT)):v for n,v in m['artifact_sha256'].items()})
    for p in (CONFIG,prior/'manifest.json',ROOT/cfg['prior_verification'],ROOT/cfg['calendar_config']):
        inputs[str(p.relative_to(ROOT))]=sha(p)
    previous=read(prior/'tasks.json')
    if len(previous)!=len(rows):raise ValueError('Candidate task/row coverage differs')
    success={r['candidate_id'] for r in summary['candidates'] if r['all_pairs_pass']}
    tasks=[]
    for old,row in zip(previous,rows):
        if (old['candidate_id'],old['cutoff'],old['strike'])!=(row['candidate_id'],row['cutoff'],row['strike']):
            raise ValueError('Candidate task order differs')
        if row['candidate_id'] not in success:continue
        if row['status']!='conditional_numerical_witness':raise ValueError('Incomplete successful candidate')
        tasks.append(dict(original_task=old,retained=row,configuration=cfg,
                          calendar_config=read(ROOT/cfg['calendar_config']),repeat_python=False))
    if not tasks:raise ValueError('No successful complete candidate to study')
    tasks.sort(key=lambda t:(t['retained']['candidate_id'],t['retained']['cutoff'],t['retained']['strike']))
    tasks[0]['repeat_python']=tasks[-1]['repeat_python']=True
    for p in (Path(__file__).resolve(),ROOT/'src/qf_research/market_data/calendar_robustness.py',
              ROOT/'src/qf_research/market_data/exercise_calendar_sensitivity.py'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    return tasks,inputs,sources


def task_problem(task):
    old,r=task['original_task'],task['retained']
    p=restore_problem(old['variants']['midpoint']['raw_call'])
    model=model_at(old,'midpoint',r['nominal_cash'],r['state_at_cutoff'],r['volatility'])
    calendar=EventCalendar.from_config(task['calendar_config'])
    schedules=core_session_schedules(calendar,datetime.fromisoformat(r['cutoff']),
        date.fromisoformat(task['configuration']['expiry']),
        p.decisions,tuple(task['configuration']['intraday_intervals_minutes']))
    return model,p,schedules


def evaluate(task):
    old,r,cfg=task['original_task'],task['retained'],task['configuration']
    model,p,schedules=task_problem(task);put=replace(p.option,kind='put')
    if list(schedules)!=cfg['calendars']:raise ValueError('Unexpected calendar ordering')
    anchors=tuple(sorted({d.time for rights in schedules.values() for d in rights}))
    condition=terminal_call_dominance(model,p.option,schedules[cfg['calendars'][-1]])
    call_reference=european_reference(model,p.option,p.decisions[-1])
    put_reference=european_reference(model,put,p.decisions[-1])
    if call_reference.premium_quote!=r['call_value']:raise ValueError('Frozen call reference changed')
    threshold=max(model.initial_state,*(put.strike*d.settlement_factor for d in p.decisions))+sum(d.discounted_cash for d in model.dividends)
    native=NativeKernel(Path(old['library']))
    def price(rights,g,extra=anchors):
        prepared=prepare(model,put,g['space_steps'],g['time_steps'],g['domain_multiple']*threshold,rights,propagation_times=extra)
        return price_prepared(prepared,native).premium_quote,prepared.time_intervals
    restored,_=price(p.decisions,cfg['meshes']['fine'],())
    native_tol=old['native_tolerance']['absolute']+old['native_tolerance']['relative']*abs(restored)
    if abs(restored-r['put_prices'][1])>native_tol:raise ValueError('Retained close price does not reproduce')
    prices={};counts={};repeats=[]
    for name,rights in schedules.items():
        prices[name]={};counts[name]={}
        for mesh,g in cfg['meshes'].items():
            prices[name][mesh],counts[name][mesh]=price(rights,g)
        if task['repeat_python'] and name in ('core_close',cfg['calendars'][-1]):
            g=cfg['meshes']['fine']
            value=finite_difference_price(model,put,g['space_steps'],g['time_steps'],g['domain_multiple']*threshold,
                                          rights,propagation_times=anchors).premium_quote
            difference=prices[name]['fine']-value
            tolerance=old['native_tolerance']['absolute']+old['native_tolerance']['relative']*abs(value)
            if abs(difference)>tolerance:raise ValueError('Original Python price differs')
            repeats.append(dict(calendar=name,python_price=value,native_price=prices[name]['fine'],difference=difference,tolerance=tolerance))
    if len({tuple(counts[n].items()) for n in counts})!=1:raise ValueError('Propagation intervals differ across calendars')
    diagnostics=compare_calendars(prices,cfg['calendars'],r['put_prices'][1],r['put_bid'],r['put_ask'],r['numerical_allowance'])
    eta=r['numerical_allowance'];tol=r['call_tolerance']
    oracle_gaps={m:v-put_reference.premium_quote for m,v in prices['terminal_only'].items()}
    screens=dict(space=all(x['space_screen'] for x in diagnostics['calendars']),
        time=all(x['time_screen'] for x in diagnostics['calendars']),domain=all(x['domain_screen'] for x in diagnostics['calendars']),
        terminal_reference=all(abs(v)<=eta for v in oracle_gaps.values()),anchor=diagnostics['anchor_screen'],
        nested_ordering=all(x['ordering_with_allowance'] for x in diagnostics['increments']),
        increment_refinement=all(x['refinement_screen'] for x in diagnostics['increments']),
        call_reference=call_reference.quadrature_error_estimate+call_reference.omitted_tail_bound<=tol,
        put_reference=put_reference.quadrature_error_estimate+put_reference.omitted_tail_bound<=tol)
    call_fits=r['call_bid']<=call_reference.premium_quote-tol and call_reference.premium_quote+tol<=r['call_ask']
    compatibility={x['calendar']:bool(condition.sufficient_condition_satisfied and all(screens.values()) and call_fits and x['quote_with_margin'])
                   for x in diagnostics['calendars']}
    return dict(candidate_id=r['candidate_id'],date=r['date'],cutoff=r['cutoff'],strike=r['strike'],
        nominal_cash=r['nominal_cash'],state=r['state_at_cutoff'],volatility=r['volatility'],
        call_bid=r['call_bid'],call_ask=r['call_ask'],put_bid=r['put_bid'],put_ask=r['put_ask'],
        numerical_allowance=eta,call_tolerance=tol,call_reference=asdict(call_reference),put_reference=asdict(put_reference),
        terminal_call_condition=asdict(condition),prices=prices,decision_counts={k:len(v) for k,v in schedules.items()},
        common_anchor_count=len(anchors),actual_time_intervals=counts,diagnostics=diagnostics,
        restored_close_price=restored,retained_close_price=r['put_prices'][1],terminal_reference_gaps=oracle_gaps,
        python_repeats=repeats,numerical_screens=screens,conditional_joint_compatibility=compatibility,refit=False)


def summarise(rows):
    if not rows or len({(r['candidate_id'],r['cutoff'],r['strike']) for r in rows})!=len(rows):raise ValueError('Require unique candidate observations')
    return dict(observations=len(rows),native_put_valuations=sum(sum(len(v) for v in r['prices'].values())+1 for r in rows),
        python_repeats=sum(len(r['python_repeats']) for r in rows),
        numerical_failures={k:sum(not r['numerical_screens'][k] for r in rows) for k in rows[0]['numerical_screens']},
        compatibility_counts={k:sum(r['conditional_joint_compatibility'][k] for r in rows) for k in rows[0]['conditional_joint_compatibility']},
        session_candidates=[dict(candidate_id=identity,observations=sum(r['candidate_id']==identity for r in rows),
            all_core_calendars_pass=all(all(v for k,v in r['conditional_joint_compatibility'].items() if k!='terminal_only') for r in rows if r['candidate_id']==identity))
            for identity in sorted({r['candidate_id'] for r in rows})],
        refit=False,continuous_american_limit_established=False,empirical_calibration_accepted=False,publication_clearance=False)


def verified_results(folder):
    folder=private(folder);manifest=read(folder/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(manifest[k])
    verify_mapping(manifest['artifact_sha256'],folder)
    rows=read(folder/'rows.json')
    if summarise(rows)!=read(folder/'summary.json'):raise ValueError('Summary differs')
    return rows,read(folder/'summary.json')


def run(output,workers):
    output=private(output)
    if output.exists():raise FileExistsError('Use a fresh private output directory')
    if not 1<=workers<=3:raise ValueError('Use one to three workers')
    tasks,inputs,sources=planned_inputs();output.mkdir(parents=True)
    save(output/'configuration.json',read(CONFIG));save(output/'tasks.json',tasks)
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,source_sha256=sources,workers=workers))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows)
            print(f'Completed {len(rows)}/{len(tasks)} frozen-candidate calendar comparisons',flush=True)
    save(output/'rows.json',rows);save(output/'summary.json',summarise(rows))
    verify_mapping(inputs);verify_mapping(sources)
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs,source_sha256=sources,artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        empirical_calibration_accepted=False,publication_clearance=False))
    print(summarise(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();run(args.output,args.workers)
