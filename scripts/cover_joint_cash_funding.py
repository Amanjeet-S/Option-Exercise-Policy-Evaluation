"""Cover cash values over all conditional term-funding nodes, preserving holds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
import math
import time

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from reprice_settlement_funding import verified_results as verified_prior
from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.settlement_comparison import restore_problem
from qf_research.market_data.settlement_funding import SettlementFunding
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.market_data.joint_cash_funding import uniform_call_margin, joint_call_cover, session_cash_sets
from qf_research.market_data.dividend_feasibility import merge_closed_intervals
from qf_research.market_data.paired_volatility import single_cash_cap_allowance, conditional_put_envelope
from qf_research.optimal_stopping.funded_cash import DividendCap, european_reference, finite_difference_price
from qf_research.computational.funded_kernel import NativeKernel, prepare, price_prepared

CONFIG = PROJECT / 'configs/joint_cash_funding_2026-09-20.json'


def planned_inputs():
    protocol = read(CONFIG)
    prior = ROOT / protocol['prior_run']
    verified_prior(prior)
    check = read(ROOT / protocol['prior_verification'])
    if check['experiment_manifest_sha256'] != sha(prior / 'manifest.json'):
        raise ValueError('Prior experiment differs from its verification')
    verify_mapping(check['source_sha256'])
    pm = read(prior / 'manifest.json')
    hashes = dict(pm['input_sha256'])
    hashes.update({str((prior/n).relative_to(ROOT)):h for n,h in pm['artifact_sha256'].items()})
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (CONFIG, prior/'manifest.json', ROOT/protocol['prior_verification'])})
    short = ROOT / protocol['short_inputs']
    sm = read(short / 'manifest.json')
    for name in ('input_sha256','source_sha256'):
        verify_mapping(sm[name]);hashes.update(sm[name])
    verify_mapping(sm['artifact_sha256'], short)
    hashes.update({str((short/n).relative_to(ROOT)):h for n,h in sm['artifact_sha256'].items()})
    hashes[str((short/'manifest.json').relative_to(ROOT))] = sha(short/'manifest.json')
    bands = {(r['cutoff'],r['strike']):r for r in read(short/'inputs.json')}
    old = read(prior/'tasks.json')
    by_key = {(t['cutoff'],t['strike'],t['dividend_model'],t['variant']):t for t in old}
    if len(by_key) != len(old):
        raise ValueError('Duplicate prior scenario')
    calendar = EventCalendar.from_config(read(ROOT/old[0]['protocol']['event_calendar']))
    planned = []
    for cutoff, strike in sorted({(t['cutoff'],t['strike']) for t in old}):
        base = by_key[(cutoff,strike,protocol['base_cash_rule'],'hybrid_midpoint')]
        forecasts = {t['dividend_model']:float(t['forecast']['regular_dividend_estimate'])
                     for t in old if (t['cutoff'],t['strike'],t['variant']) == (cutoff,strike,'hybrid_midpoint')}
        if set(forecasts) != set(base['protocol']['dividend_models']):
            raise ValueError('Nominal benchmark coverage differs')
        band = bands[(cutoff,strike)]
        s = date.fromisoformat(band['spot_date'])
        h = date.fromisoformat(base['premium_payment_date'])
        expiry = date.fromisoformat(base['protocol']['expiry'])
        stamps = calendar.research_decisions(datetime.fromisoformat(cutoff), expiry)
        deliveries = [calendar.settlement_after(t.date()).day.isoformat() for t in stamps]
        variants = {}
        for side in ('lower','midpoint','upper'):
            t = by_key[(cutoff,strike,protocol['base_cash_rule'],'hybrid_'+side)]
            raw = t['raw_call'];p = restore_problem(raw)
            if len(p.model.dividends) != 1 or p.model.premium_factor != 1. or p.model.initial_share_factor != 1.:
                raise ValueError('Require one future cash event in premium units')
            if p.option != restore_problem(base['raw_call']).option or t['observed'] != base['observed']:
                raise ValueError('Funding variants changed a quote or contract')
            ds = (s,)+tuple(date.fromisoformat(n['accrual_end']) for n in band['nodes'])
            curve = DiscountCurve(s,ds,(1.,)+tuple(n[side] for n in band['nodes']))
            funding = SettlementFunding(h,curve,t['settlement_to_spot'])
            payment = funding.factor(date.fromisoformat(t['forecast']['pay_date']))
            if p.model.dividends[0].discounted_cash != forecasts[protocol['base_cash_rule']]*payment:
                raise ValueError('Cash cap does not reconstruct exactly')
            if len(p.decisions)!=len(deliveries) or any(d.phase!='after' for d in p.decisions):
                raise ValueError('Unexpected exercise phase or coverage')
            for d, delivery in zip(p.decisions,deliveries):
                if d.settlement_factor != funding.factor(date.fromisoformat(delivery)):
                    raise ValueError('Dated factors differ from the recorded curve')
            variants[side] = dict(raw_call=raw,payment_factor=payment)
        lo, hi = [restore_problem(variants[side]['raw_call']) for side in ('lower','upper')]
        if [(d.time,d.phase) for d in lo.decisions] != [(d.time,d.phase) for d in hi.decisions] or lo.model.dividends[0].ex_time != hi.model.dividends[0].ex_time:
            raise ValueError('Funding endpoints changed clocks')
        dominance = uniform_call_margin(strike, deliveries, [d.settlement_factor for d in lo.decisions],
            [d.settlement_factor for d in hi.decisions], [d.time for d in lo.decisions],
            lo.model.dividends[0].ex_time, max(forecasts.values()), variants['upper']['payment_factor'])
        planned.append(dict(date=base['date'],cutoff=cutoff,strike=strike,observed=base['observed'],variants=variants,
            forecasts=forecasts,cash_upper=max(forecasts.values()),delivery_dates=deliveries,
            terminal_call_condition=dominance,configuration=base['configuration'],protocol=protocol,
            library=base['library'],native_tolerance=base['protocol']['native_price_tolerance'],repeat_python=False))
    if not planned:
        raise ValueError('No supported observations')
    planned[0]['repeat_python']=planned[-1]['repeat_python']=True
    sources = dict(pm['source_sha256'])
    for p in (Path(__file__).resolve(), ROOT/'src/qf_research/market_data/joint_cash_funding.py'):
        sources[str(p.relative_to(ROOT))]=sha(p)
    return planned, hashes, sources


def model_at(task, side, cash, spot, sigma):
    p=restore_problem(task['variants'][side]['raw_call'])
    return replace(p.model,quoted_spot=spot,volatility=sigma,
        dividends=(DividendCap(p.model.dividends[0].ex_time,cash*task['variants'][side]['payment_factor']),))


def evaluate(task):
    cfg,protocol=task['configuration'],task['protocol']
    low_problem=restore_problem(task['variants']['lower']['raw_call'])
    call=low_problem.option;put=replace(call,kind='put')
    sb,sa=[task['observed']['underlying'][k] for k in ('BID','ASK')]
    cb,ca=[task['observed']['call'][k] for k in ('BID','ASK')]
    pb,pa=[task['observed']['put'][k] for k in ('BID','ASK')]
    upper_cash=task['cash_upper']
    base={k:task[k] for k in ('date','cutoff','strike','forecasts','terminal_call_condition')}
    base.update(nominal_search=[0.,upper_cash],underlying_bid=sb,underlying_ask=sa,call_bid=cb,call_ask=ca,put_bid=pb,put_ask=pa)
    if not task['terminal_call_condition']['sufficient_condition_satisfied']:
        return dict(base,status='unresolved_terminal_call_condition',outer_cash_set=[[0.,upper_cash]],
                    unresolved_set=[[0.,upper_cash]],cells=[],endpoints=[],python_repeats=[],leaves=[])
    native=NativeKernel(Path(task['library']))
    rights={side:restore_problem(task['variants'][side]['raw_call']).decisions for side in ('lower','upper')}
    references=[]
    @lru_cache(maxsize=None)
    def call_price(side,spot,cash,sigma):
        ref=european_reference(model_at(task,side,cash,spot,sigma),call,rights[side][-1])
        references.append(ref)
        return ref.premium_quote
    endpoints=[]
    @lru_cache(maxsize=None)
    def endpoint(cash,side,sigma):
        spot=sa if side=='lower' else sb
        model=model_at(task,side,cash,spot,sigma)
        threshold=max(model.initial_state,*(put.strike*d.settlement_factor for d in rights[side]))+sum(e.discounted_cash for e in model.dividends)
        prices=[price_prepared(prepare(model,put,g['space_steps'],g['time_steps'],cfg['domain_multiple']*threshold,rights[side]),native).premium_quote for g in cfg['price_grids']]
        g=cfg['domain_validation']
        wide=price_prepared(prepare(model,put,g['space_steps'],g['time_steps'],g['domain_multiple']*threshold,rights[side]),native).premium_quote
        value=dict(id=len(endpoints),cash=cash,funding_side=side,spot=spot,volatility=sigma,prices=prices,
                   wider_domain_price=wide,space_time_change=prices[1]-prices[0],domain_change=wide-prices[1])
        endpoints.append(value)
        return value['id']
    allowance=cfg['numerical_allowance_fraction_of_full_put_spread']*(pa-pb)
    cells=[];pending=[(0.,upper_cash,0,None)];leaf_ids=[]
    while pending:
        low,high,depth,parent=pending.pop()
        cover=joint_call_cover(call_price,sb,sa,cb,ca,low,high,tuple(cfg['volatility_bounds']),cfg['volatility_bracket_width'])
        cell=dict(id=len(cells),parent=parent,depth=depth,cash_lower=low,cash_upper=high,cover=cover)
        cells.append(cell)
        reference_error=max(r.quadrature_error_estimate+r.omitted_tail_bound for r in references)
        if cover['status']=='no_call_candidate':
            roots=cover['roots'];distances=[]
            l=roots['lower_factor_lower_cash_call_bid'];u=roots['upper_factor_upper_cash_call_ask']
            if l['status']=='above_range':distances.append(cb-l['upper_price'])
            if u['status']=='below_range':distances.append(u['lower_price']-ca)
            screened=reference_error<=cfg['price_residual_tolerance'] and max(distances)>2*cfg['price_residual_tolerance']
            cell.update(call_exclusion_margin=max(distances),call_reference_error_estimate=reference_error)
            reason='excluded_call' if screened else 'unresolved_call_inversion'
        else:
            li=endpoint(low,'lower',cover['volatility_lower']);ui=endpoint(high,'upper',cover['volatility_upper'])
            left,right=endpoints[li],endpoints[ui]
            cap=single_cash_cap_allowance(model_at(task,'upper',high,sb,cover['volatility_upper']),put,cover['volatility_upper'])
            envelope=conditional_put_envelope(left['prices'][1],right['prices'][1],cap['premium_allowance'],allowance)
            targets={'lower_factor_lower_cash_call_bid':cb,'upper_factor_upper_cash_call_ask':ca}
            residuals={k:max(abs(r[e]-targets[k]) for e in ('lower_price','upper_price'))
                       for k,r in cover['roots'].items() if r['status']=='bracketed'}
            screens=dict(call_root=all(v<=cfg['price_residual_tolerance'] for v in residuals.values()),
                call_reference_estimate=reference_error<=cfg['price_residual_tolerance'],
                space_time=all(abs(r['space_time_change'])<=allowance for r in (left,right)),
                domain=all(abs(r['domain_change'])<=allowance for r in (left,right)))
            cell.update(endpoint_ids=[li,ui],cap_allowance=cap,numerical_allowance=allowance,put_envelope=envelope,
                        root_price_residuals=residuals,reference_error_estimate=reference_error,numerical_screens=screens)
            reason=('unresolved_numerical_screen' if not all(screens.values()) else
                    'excluded_above_put_ask' if envelope['lower']>pa else
                    'excluded_below_put_bid' if envelope['upper']<pb else 'not_excluded')
        if reason.startswith('excluded') or high-low<=protocol['cash_cell_maximum_width'] or depth>=protocol['maximum_bisection_depth']:
            cell.update(status=reason,leaf=True);leaf_ids.append(cell['id'])
        else:
            cell.update(status='subdivided',reason=reason,leaf=False)
            midpoint=(low+high)/2
            pending.extend([(midpoint,high,depth+1,cell['id']),(low,midpoint,depth+1,cell['id'])])
    leaves=[cells[i] for i in leaf_ids]
    if merge_closed_intervals([(c['cash_lower'],c['cash_upper']) for c in leaves]) != [[0.,upper_cash]]:
        raise ValueError('Leaf partition lost cash-domain coverage')
    kept=[c for c in leaves if not c['status'].startswith('excluded')]
    outer=merge_closed_intervals([(c['cash_lower'],c['cash_upper']) for c in kept])
    unresolved=merge_closed_intervals([(c['cash_lower'],c['cash_upper']) for c in kept if c['status']!='not_excluded'])
    repeats=[]
    if task['repeat_python']:
        available=[c for c in (kept or leaves) if 'endpoint_ids' in c]
        if available:
            boundaries=[min(available,key=lambda c:c['cash_lower']),max(available,key=lambda c:c['cash_upper'])]
            for i in sorted({i for c in boundaries for i in c['endpoint_ids']}):
                e=endpoints[i];side=e['funding_side'];model=model_at(task,side,e['cash'],e['spot'],e['volatility'])
                threshold=max(model.initial_state,*(put.strike*d.settlement_factor for d in rights[side]))+sum(d.discounted_cash for d in model.dividends)
                g=cfg['price_grids'][1]
                value=finite_difference_price(model,put,g['space_steps'],g['time_steps'],cfg['domain_multiple']*threshold,rights[side]).premium_quote
                tolerance=task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(value)
                if abs(e['prices'][1]-value)>tolerance:
                    raise ValueError('Python and native endpoint prices disagree')
                repeats.append(dict(endpoint_id=i,python_price=value,native_price=e['prices'][1],difference=e['prices'][1]-value,tolerance=tolerance))
    return dict(base,status='completed',outer_cash_set=outer,unresolved_set=unresolved,cells=cells,endpoints=endpoints,
        leaves=leaf_ids,python_repeats=repeats,leaf_status_counts=dict(Counter(c['status'] for c in leaves)),
        forecast_membership={name:any(a<=value<=b for a,b in outer) for name,value in task['forecasts'].items()},
        maximum_retained_cell_width=max((c['cash_upper']-c['cash_lower'] for c in kept),default=0.))


def summarise(rows):
    return dict(observations=len(rows),sessions=session_cash_sets(rows),
        cells=sum(len(r['cells']) for r in rows),native_put_valuations=sum(3*len(r['endpoints']) for r in rows),
        python_repeats=sum(len(r['python_repeats']) for r in rows),
        unresolved_observations=sum(bool(r['unresolved_set']) for r in rows),
        all_forecasts_excluded=all(not any(r.get('forecast_membership',{'unknown':True}).values()) for r in rows),
        empirical_calibration_accepted=False,publication_clearance=False)


def verified_results(folder):
    folder=private(folder);m=read(folder/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(m[k])
    verify_mapping(m['artifact_sha256'],folder)
    rows=read(folder/'rows.json')
    if summarise(rows)!=read(folder/'summary.json'):raise ValueError('Joint cover aggregation differs')
    return rows,read(folder/'summary.json')


def run(output,workers):
    output=private(output)
    if output.exists():raise FileExistsError('Use a fresh private output directory')
    if not 1<=workers<=3:raise ValueError('Use one to three workers')
    tasks,inputs,sources=planned_inputs()
    output.mkdir(parents=True)
    for name,value in [('configuration.json',read(CONFIG)),('tasks.json',tasks)]:save(output/name,value)
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,source_sha256=sources,workers=workers))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows)
            print(f'Completed {len(rows)}/{len(tasks)} joint cash/funding covers',flush=True)
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
