"""Call-implied volatility intervals and conditional matched-put consistency."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, restore, verify_mapping
from summarise_conditional_calibration import verified_results
from qf_research.market_data.paired_volatility import (
    call_quote_interval, single_cash_cap_allowance, conditional_put_envelope,
)
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_cash import european_reference, finite_difference_price

CONFIG = PROJECT/'configs/paired_volatility_diagnostic_2026-09-19.json'
CODE = [Path(__file__).resolve(), ROOT/'src/qf_research/market_data/paired_volatility.py',
        ROOT/'src/qf_research/optimal_stopping/exercise_dominance.py',
        ROOT/'scripts/summarise_conditional_calibration.py']


def prepare_inputs(cfg):
    prior = read(ROOT/cfg['prior_verification'])
    verify_mapping(prior['source_sha256'])
    verify_mapping(prior['private_evidence_manifest_sha256'])
    if sha(ROOT/prior['notebook']) != prior['notebook_sha256']:
        raise ValueError('Earlier notebook differs')
    for name in prior['private_evidence_manifest_sha256']:
        p = ROOT/name
        manifest = read(p)
        for field in ('input_sha256','source_sha256'):verify_mapping(manifest[field])
        verify_mapping(manifest['artifact_sha256'],p.parent)
    sample,fit = ROOT/cfg['sample'], ROOT/cfg['fit']
    _,selected,_,_ = verified_results(sample,fit)
    by_id = {r['observation_id']:r for r in selected}
    raw = read(sample/'problems.json')
    return sample,fit,by_id,raw


def paired_inputs(raw, by_id):
    groups = {}
    for r in raw:
        source = by_id[r['observation_id']]
        key = (r['dividend_model'],source['observation_date'],source['expiry'],r['option']['strike'])
        group = groups.setdefault(key,{})
        side = r['option']['kind']
        if side in group:raise ValueError('Duplicate side in matched-contract group')
        group[side] = r
    pairs,unmatched = [],[]
    for key,group in sorted(groups.items()):
        if set(group) != {'call','put'}:
            unmatched.append({'dividend_model':key[0],'date':key[1],'expiry':key[2],
                              'strike':key[3],'retained_sides':sorted(group)})
            continue
        call,put = group['call'],group['put']
        if any(call[k] != put[k] for k in ('model','decisions','forecast','curve_date','premium_payment_date','split')):
            raise ValueError('Matched sides have different economic inputs')
        if call['option']['maturity'] != put['option']['maturity']:
            raise ValueError('Matched sides have different maturity times')
        a,b = (by_id[r['observation_id']] for r in (call,put))
        if any(a[k] != b[k] for k in ('cutoff','quoted_spot','underlying_bid','underlying_ask')):
            raise ValueError('Matched sides have different observed states or cutoffs')
        pairs.append((call,put))
    return pairs,unmatched


def invert_call(raw, source, cfg):
    record = restore(raw)
    if record.option.kind != 'call':raise ValueError('Invert calls only')
    dominance = terminal_call_dominance(record.model,record.option,record.decisions)
    if not dominance.sufficient_condition_satisfied:
        raise ValueError('European inversion is not justified by terminal dominance')
    references = []
    @lru_cache(maxsize=None)
    def price(sigma):
        ref = european_reference(replace(record.model,volatility=sigma),record.option,record.decisions[-1])
        references.append(asdict(ref))
        return ref.premium_quote
    interval = call_quote_interval(price,record.bid,record.ask,
                                   tuple(cfg['volatility_bounds']),cfg['volatility_bracket_width'])
    residuals = {name:max(abs(root[k]-target) for k in ('lower_price','upper_price'))
                 for name,target in [('bid',record.bid),('midpoint',record.midpoint),('ask',record.ask)]
                 if (root := interval['roots'][name])['status'] == 'bracketed'}
    fp = record.model.premium_factor
    quadrature = max(r['quadrature_error_estimate']/fp for r in references)
    tail = max(r['omitted_tail_bound']/fp for r in references)
    result = {'observation_id':record.observation_id,'dividend_model':raw['dividend_model'],
        'date':source['observation_date'],'expiry':source['expiry'],'strike':record.option.strike,
        'split':record.split,'bid':record.bid,'ask':record.ask,'interval':interval,
        'terminal_dominance':asdict(dominance),'root_endpoint_price_residuals':residuals,
        'unique_reference_evaluations':len(references),'maximum_quadrature_error_estimate_premium':quadrature,
        'maximum_omitted_tail_bound_premium':tail,
        'numerical_screens':{'root_residuals':all(v <= cfg['price_residual_tolerance'] for v in residuals.values()),
                             'reference_estimated_error':quadrature+tail <= cfg['price_residual_tolerance']}}
    if interval['status'] == 'candidate_interval':
        result['selected_call_prices'] = {name:price(sigma) for name,sigma in
             [('lower',interval['lower']),('midpoint',interval['midpoint_volatility']),('upper',interval['upper'])]}
    return result


def evaluate_pair(task):
    raw,call,cfg = task
    record = restore(raw)
    interval = call['interval']
    base = {'call_observation_id':call['observation_id'],'put_observation_id':record.observation_id,
        'date':call['date'],'expiry':call['expiry'],'strike':record.option.strike,
        'dividend_model':raw['dividend_model'],'split':record.split,'put_bid':record.bid,'put_ask':record.ask,
        'call_interval':interval}
    if interval['status'] != 'candidate_interval':
        return dict(base,classification='no_call_candidate_in_declared_range',numerical_screens={})
    threshold = max(record.model.initial_state,*(record.option.strike*d.settlement_factor for d in record.decisions))
    threshold += sum(e.discounted_cash for e in record.model.dividends if e.ex_time <= record.option.maturity)
    allowance = cfg['numerical_allowance_fraction_of_full_put_spread']*(record.ask-record.bid)
    values = {}
    for name,sigma in [('lower',interval['lower']),('midpoint',interval['midpoint_volatility']),('upper',interval['upper'])]:
        model = replace(record.model,volatility=sigma)
        prices = [finite_difference_price(model,record.option,g['space_steps'],g['time_steps'],
                    cfg['domain_multiple']*threshold,record.decisions).premium_quote for g in cfg['price_grids']]
        g = cfg['domain_validation']
        wide = finite_difference_price(model,record.option,g['space_steps'],g['time_steps'],
                                       g['domain_multiple']*threshold,record.decisions).premium_quote
        values[name] = {'volatility':sigma,'prices':prices,'wider_domain_price':wide,
                        'space_time_change':prices[1]-prices[0],'domain_change':wide-prices[1]}
    cap = single_cash_cap_allowance(record.model,record.option,interval['upper'])
    envelope = conditional_put_envelope(values['lower']['prices'][1],values['upper']['prices'][1],
                                        cap['premium_allowance'],allowance)
    screens = {**call['numerical_screens'],
        'space_time':all(abs(v['space_time_change']) <= allowance for v in values.values()),
        'domain':all(abs(v['domain_change']) <= allowance for v in values.values()),
        'midpoint_inside_comparison':envelope['lower'] <= values['midpoint']['prices'][1] <= envelope['upper']}
    if not all(screens.values()):classification = 'unresolved_numerical_screen'
    elif envelope['lower'] > record.ask:classification = 'conditionally_above_put_ask'
    elif envelope['upper'] < record.bid:classification = 'conditionally_below_put_bid'
    else:classification = 'not_separated_inconclusive'
    return dict(base,values=values,cap_allowance=cap,numerical_allowance=allowance,put_envelope=envelope,
                numerical_screens=screens,classification=classification,
                lower_minus_put_ask=envelope['lower']-record.ask,
                put_bid_minus_upper=record.bid-envelope['upper'])


def verified_diagnostic(folder):
    folder = private(folder)
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256','source_sha256'):verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'],folder)
    prepare_inputs(read(folder/'configuration.json'))
    return read(folder/'calls.json'),read(folder/'pairs.json'),read(folder/'audit_counts.json')


def run(output,workers):
    output = private(output)
    if output.exists():raise FileExistsError('Use a fresh output directory')
    if not 1 <= workers <= 3:raise ValueError('Use one to three worker processes')
    cfg = read(CONFIG)
    sample,fit,by_id,raw = prepare_inputs(cfg)
    pairs,unmatched = paired_inputs(raw,by_id)
    inputs = {str(p.relative_to(ROOT)):sha(p) for p in
              [CONFIG,ROOT/cfg['prior_verification'],sample/'manifest.json',fit/'manifest.json',
               sample/'problems.json',sample/'selected.json']}
    sources = dict(read(sample/'manifest.json')['source_sha256'])
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in CODE})
    output.mkdir(parents=True)
    save(output/'configuration.json',cfg)
    save(output/'launch.json',{'started_at_utc':datetime.now(timezone.utc).isoformat(),
         'input_sha256':inputs,'source_sha256':sources,'workers':workers,
         'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version()})
    start = time.monotonic()
    calls = [invert_call(r,by_id[r['observation_id']],cfg) for r in raw if r['option']['kind'] == 'call']
    save(output/'calls.json',calls)
    print(f'Completed {len(calls)} call inversions',flush=True)
    lookup = {(r['observation_id'],r['dividend_model']):r for r in calls}
    tasks = [(put,lookup[(call['observation_id'],call['dividend_model'])],cfg) for call,put in pairs]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate_pair,tasks,chunksize=1):
            results.append(row)
            if len(results)%3 == 0 or len(results) == len(tasks):
                save(output/'partial_pairs.json',results)
                print(f'Completed {len(results)}/{len(tasks)} matched-pair scenarios',flush=True)
    counts = {'observations':len(by_id),'cash_variants':len({r['dividend_model'] for r in raw}),
        'call_scenarios':len(calls),'matched_pair_scenarios':len(results),'unmatched_scenario_groups':len(unmatched),
        'call_interval_status':dict(Counter(r['interval']['status'] for r in calls)),
        'clipped_call_intervals':sum(r['interval'].get('clipped_to_volatility_range',False) for r in calls),
        'classification_counts':dict(Counter(r['classification'] for r in results)),
        'call_numerical_screen_failures':{k:sum(not r['numerical_screens'][k] for r in calls) for k in calls[0]['numerical_screens']},
        'pair_numerical_screen_failures':{k:sum(not r['numerical_screens'].get(k,False) for r in results)
            for k in ('root_residuals','reference_estimated_error','space_time','domain','midpoint_inside_comparison')},
        'pde_valuations':sum(3*len(r.get('values',{})) for r in results),
        'previously_held_out_quotes_reused_descriptively':True,'full_empirical_acceptance':False}
    for name,data in [('pairs.json',results),('unmatched.json',unmatched),('audit_counts.json',counts)]:
        save(output/name,data)
    report = ['# Paired volatility diagnostic','','Private conditional comparison under the frozen financial inputs.',
        'The quote-derived intervals reuse all retained dates; they are not predictive estimates.','',
        '| Date | Strike | Cash rule | Lower volatility | Upper volatility | Conditional lower put bound minus ask | Classification |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- |']
    for r in results:
        if 'values' in r:
            report.append(f'| {r["date"]} | {r["strike"]} | {r["dividend_model"]} | {r["call_interval"]["lower"]:.10f} | {r["call_interval"]["upper"]:.10f} | {r["lower_minus_put_ask"]:.9f} | {r["classification"]} |')
        else:report.append(f'| {r["date"]} | {r["strike"]} | {r["dividend_model"]} | | | | {r["classification"]} |')
    (output/'report.md').write_text('\n'.join(report)+'\n')
    verify_mapping(inputs);verify_mapping(sources)
    save(output/'manifest.json',{'completed_at_utc':datetime.now(timezone.utc).isoformat(),
         'input_sha256':inputs,'source_sha256':sources,
         'artifact_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
         'elapsed_seconds':time.monotonic()-start,'workers':workers,'full_empirical_acceptance':False})
    print(counts,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    run(args.output,args.workers)
