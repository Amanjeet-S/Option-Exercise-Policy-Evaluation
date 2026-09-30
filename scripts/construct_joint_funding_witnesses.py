"""Construct and validate interior joint-quote fits without claiming identification."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict,replace
from datetime import datetime,timezone
from functools import lru_cache
from pathlib import Path
import time

from scipy.optimize import brentq

from cover_joint_cash_funding import verified_results as cash_results, model_at
from qf_research.market_data.settlement_comparison import restore_problem
from run_conditional_calibration import ROOT,PROJECT,private,read,save,sha,verify_mapping
from qf_research.computational.funded_kernel import NativeKernel,prepare,price_prepared
from qf_research.market_data.paired_volatility import call_quote_interval
from qf_research.optimal_stopping.funded_cash import DividendCap,european_reference,finite_difference_price
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance

CONFIG=PROJECT/'configs/joint_cash_funding_2026-09-20.json'


def planned_inputs(cover_run):
    cover_run=private(cover_run);rows,summary=cash_results(cover_run)
    protocol=read(CONFIG);manifest=read(cover_run/'manifest.json')
    if read(cover_run/'configuration.json')!=protocol:raise ValueError('Cover protocol changed')
    pairs=read(cover_run/'tasks.json')
    by_key={(p['cutoff'],p['strike']):p for p in pairs}
    tasks=[];missing=[]
    for record in summary['sessions']:
        day=record['date'];group=[by_key[(r['cutoff'],r['strike'])] for r in rows if r['date']==day]
        if not record['common_cash_set']:
            missing.append({'date':day,'reason':'empty_necessary_cash_set'});continue
        for i,(a,b) in enumerate(record['common_cash_set']):
            for fraction in protocol['candidate_fractions']:
                cash=a+(b-a)*fraction
                for j,pair in enumerate(group):
                    tasks.append(dict(pair,candidate_id=f'{day}:{i}:{fraction:g}',cash=cash,
                        state=sum(pair['observed']['underlying'][k] for k in ('BID','ASK'))/2,
                        component=[a,b],fraction=fraction,repeat_python=j in (0,len(group)-1),
                        expected_observations=len(group)))
    input_hashes=dict(manifest['input_sha256']);source_hashes=dict(manifest['source_sha256'])
    input_hashes.update({str((cover_run/n).relative_to(ROOT)):h for n,h in manifest['artifact_sha256'].items()})
    input_hashes[str((cover_run/'manifest.json').relative_to(ROOT))]=sha(cover_run/'manifest.json')
    source_hashes[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__))
    return tasks,missing,input_hashes,source_hashes


def evaluate(task):
    cfg=task['configuration'];protocol=task['protocol'];cash=task['cash'];spot=task['state']
    call=restore_problem(task['variants']['midpoint']['raw_call'])
    put=replace(call,option=replace(call.option,kind='put'),bid=task['observed']['put']['BID'],ask=task['observed']['put']['ASK'])
    base_model=model_at(task,'midpoint',cash,spot,call.model.volatility)
    condition=terminal_call_dominance(base_model,call.option,call.decisions)
    row={'candidate_id':task['candidate_id'],'date':task['date'],'strike':put.option.strike,
        'nominal_cash':cash,'fraction':task['fraction'],'common_component':task['component'],'state_at_cutoff':spot,'cutoff':task['cutoff'],
        'expected_observations':task['expected_observations'],'call_bid':call.bid,'call_ask':call.ask,
        'put_bid':put.bid,'put_ask':put.ask,'terminal_call_condition':asdict(condition)}
    if not condition.sufficient_condition_satisfied:return dict(row,status='unresolved_call_dominance')
    refs=[]
    @lru_cache(maxsize=None)
    def call_value(sigma):
        value=european_reference(replace(base_model,volatility=sigma),call.option,call.decisions[-1])
        refs.append(asdict(value));return value.premium_quote
    interval=call_quote_interval(call_value,call.bid,call.ask,tuple(cfg['volatility_bounds']),cfg['volatility_bracket_width'])
    row['call_interval']=interval
    if interval['status']!='candidate_interval':return dict(row,status='no_call_candidate_in_search')
    native=NativeKernel(Path(task['library']));search_grid=cfg['price_grids'][0]
    threshold=max(base_model.initial_state,*(put.option.strike*d.settlement_factor for d in put.decisions))+sum(d.discounted_cash for d in base_model.dividends)
    trials=[]
    @lru_cache(maxsize=None)
    def put_value(sigma):
        value=price_prepared(prepare(replace(base_model,volatility=sigma),put.option,search_grid['space_steps'],
            search_grid['time_steps'],cfg['domain_multiple']*threshold,put.decisions),native).premium_quote
        trials.append({'volatility':sigma,'search_grid_put':value});return value
    low,high=interval['lower'],interval['upper'];pl,pu=put_value(low),put_value(high)
    eta=cfg['numerical_allowance_fraction_of_full_put_spread']*(put.ask-put.bid)
    target_low=max(min(pl,pu),put.bid+2*eta);target_high=min(max(pl,pu),put.ask-2*eta)
    if target_low>target_high:
        return dict(row,status='candidate_does_not_supply_put_bracket',endpoint_puts=[pl,pu],trials=trials)
    target=(target_low+target_high)/2
    try:sigma=brentq(lambda s:put_value(s)-target,low,high,xtol=protocol['witness_root']['xtol'],maxiter=protocol['witness_root']['maxiter'])
    except (ValueError,RuntimeError) as exc:
        return dict(row,status='unresolved_put_root',error_type=type(exc).__name__,trials=trials)
    model=replace(base_model,volatility=sigma);fine_grid=cfg['price_grids'][1];wide_grid=cfg['domain_validation']
    coarse=put_value(sigma)
    fine=price_prepared(prepare(model,put.option,fine_grid['space_steps'],fine_grid['time_steps'],cfg['domain_multiple']*threshold,put.decisions),native).premium_quote
    wide=price_prepared(prepare(model,put.option,wide_grid['space_steps'],wide_grid['time_steps'],wide_grid['domain_multiple']*threshold,put.decisions),native).premium_quote
    c=call_value(sigma);tol=cfg['price_residual_tolerance']
    screens={'call_with_margin':call.bid<=c-tol and c+tol<=call.ask,
             'put_with_margin':put.bid<=fine-eta and fine+eta<=put.ask,
             'space_time':abs(fine-coarse)<=eta,'domain':abs(wide-fine)<=eta,
             'call_reference':max((v['quadrature_error_estimate']+v['omitted_tail_bound'])/model.premium_factor for v in refs)<=tol}
    repeat=None
    if all(screens.values()) and task['repeat_python']:
        reference=finite_difference_price(model,put.option,fine_grid['space_steps'],fine_grid['time_steps'],cfg['domain_multiple']*threshold,put.decisions).premium_quote
        tolerance=task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(reference)
        repeat={'original_python_price':reference,'native_price':fine,'difference':fine-reference,'tolerance':tolerance,'passed':abs(fine-reference)<=tolerance}
        screens['independent_python']=repeat['passed']
    return dict(row,status='conditional_numerical_witness' if all(screens.values()) else 'unresolved_validation',
        volatility=sigma,call_value=c,put_prices=[coarse,fine,wide],numerical_allowance=eta,call_tolerance=tol,
        put_target=target,trials=trials,numerical_screens=screens,python_repeat=repeat,
        call_margin=min(c-call.bid,call.ask-c),put_margin=min(fine-put.bid,put.ask-fine))


def summarise(rows,missing):
    candidates=[]
    for identity in dict.fromkeys(r['candidate_id'] for r in rows):
        group=[r for r in rows if r['candidate_id']==identity]
        if len(group)!=group[0]['expected_observations'] or len({(r['cutoff'],r['strike']) for r in group})!=len(group):
            raise ValueError('Candidate observation coverage differs')
        candidates.append({'candidate_id':identity,'date':group[0]['date'],'fraction':group[0]['fraction'],
            'nominal_cash':group[0]['nominal_cash'],'observations':len(group),
            'all_pairs_pass':all(r['status']=='conditional_numerical_witness' for r in group),
            'pair_statuses':[r['status'] for r in group]})
    return {'candidates':candidates,'unresolved_sessions':missing,
        'session_has_conditional_witness':{d:any(c['all_pairs_pass'] for c in candidates if c['date']==d) for d in sorted({c['date'] for c in candidates}|{m['date'] for m in missing})},
        'native_put_valuations':sum(len(r.get('trials',[]))+(2 if 'put_prices' in r else 0) for r in rows),
        'independent_python_prices':sum(bool(r.get('python_repeat')) for r in rows),
        'session_constant_cash_assumption':True,'same_state_across_distinct_cutoffs_imposed':False,
        'conditional_on_numerical_allowance':True,'dividend_identified':False,
        'empirical_calibration_accepted':False,'publication_clearance':False}


def verified_results(folder):
    folder=private(folder);m=read(folder/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(m[k])
    verify_mapping(m['artifact_sha256'],folder)
    rows=read(folder/'rows.json');summary=summarise(rows,read(folder/'missing.json'))
    if summary!=read(folder/'summary.json'):raise ValueError('Witness aggregation differs')
    return rows,summary


def run(cover_run,output,workers):
    output=private(output)
    if output.exists():raise FileExistsError('Use a fresh output directory')
    if not 1<=workers<=3:raise ValueError('Use one to three workers')
    tasks,missing,input_hashes,source_hashes=planned_inputs(cover_run);output.mkdir(parents=True)
    save(output/'configuration.json',read(CONFIG));save(output/'missing.json',missing);save(output/'tasks.json',tasks)
    save(output/'launch.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':input_hashes,'source_sha256':source_hashes,'workers':workers})
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows)
            if len(rows)%3==0:print(f'Completed {len(rows)}/{len(tasks)} candidate pair fits',flush=True)
    summary=summarise(rows,missing);save(output/'rows.json',rows);save(output/'summary.json',summary)
    verify_mapping(input_hashes);verify_mapping(source_hashes)
    save(output/'manifest.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':input_hashes,
        'source_sha256':source_hashes,'artifact_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        'elapsed_seconds':time.monotonic()-start,'publication_clearance':False,'full_empirical_acceptance':False})
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cover-run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args();run(args.cover_run,args.output,args.workers)
