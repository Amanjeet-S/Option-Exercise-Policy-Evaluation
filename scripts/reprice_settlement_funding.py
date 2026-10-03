"""Recalibrate and reprice a fixed private sample under declared funding paths."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from build_valuation_inputs import curve_arguments
from qf_research.computational.funded_kernel import NativeKernel, prepare, price_prepared
from qf_research.market_data.distribution_announcements import reconstruct_benchmarks
from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.settlement_funding import build_premium_normalised_problem
from qf_research.market_data.settlement_comparison import funding_paths, check_normalised_equivalence, matched_changes, restore_problem
from qf_research.market_data.funding_sensitivity import lower_corner_components
from qf_research.market_data.valuation_inputs import inspect_sofr_curve
from qf_research.market_data.underlying_quote_interval import call_state_rectangle
from qf_research.market_data.paired_volatility import single_cash_cap_allowance, conditional_put_envelope
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_cash import european_reference, finite_difference_price

CONFIG = PROJECT / 'configs/settlement_repricing_2026-09-20.json'


def prepare_tasks(library):
    protocol = read(CONFIG)
    cfg = read(ROOT / protocol['numerical_protocol'])
    hashes = {}

    def track(path):
        path = path.resolve()
        hashes[str(path.relative_to(ROOT))] = sha(path)
        return read(path)

    def evidence(folder):
        folder = private(folder)
        m = track(folder / 'manifest.json')
        for key in ('input_sha256', 'source_sha256'):
            verify_mapping(m[key])
            hashes.update(m[key])
        verify_mapping(m['artifact_sha256'], folder)
        hashes.update({str((folder/n).relative_to(ROOT)): h for n,h in m['artifact_sha256'].items()})

    for name in ('front_inputs', 'short_inputs', 'quote_panel', 'declarations'):
        evidence(ROOT / protocol[name])
    track(CONFIG)
    track(ROOT / protocol['numerical_protocol'])
    calendar = EventCalendar.from_config(track(ROOT / protocol['event_calendar']))
    vc = track(ROOT / protocol['valuation_protocol'])
    folder = ROOT / protocol['retained_curves']
    cm = track(folder / 'manifest.json')
    verify_mapping({'config.json': cm['config_sha256'], 'acquisition_source.py': cm['acquisition_source_sha256']}, folder)
    hashes.update({str((folder/n).relative_to(ROOT)): sha(folder/n) for n in ('config.json', 'acquisition_source.py')})
    if cm['status'] != 'complete' or read(folder / 'config.json') != vc:
        raise ValueError('Retained curve acquisition differs')
    curves = {}
    for entry in cm['requests']:
        d = entry['curve_date']
        if entry['arguments'] != curve_arguments(vc, d):
            raise ValueError('Retained curve request differs')
        verify_mapping({entry['file']: entry['sha256']}, folder)
        curve, _ = inspect_sofr_curve(track(folder / entry['file']), d, vc['curve_id'])
        if d in curves:
            raise ValueError('Duplicate retained curve date')
        curves[d] = curve
    native_check = track(PROJECT / 'results/cpp_propagation/verification.json')
    verify_mapping(native_check['source_sha256'])
    hashes.update(native_check['source_sha256'])
    if sha(ROOT / native_check['notebook']) != native_check['notebook_sha256']:
        raise ValueError('Native verification notebook changed')
    hashes[native_check['notebook']] = native_check['notebook_sha256']
    library = library.resolve()
    build = track(library.parent / 'build.json')
    if library.name != build['library'] or sha(library) != build['library_sha256'] or sha(ROOT / build['source']) != build['source_sha256']:
        raise ValueError('Native library differs from its verified source or build')
    hashes[str(library.relative_to(ROOT))] = sha(library)
    NativeKernel(library)
    front = read(ROOT / protocol['front_inputs'] / 'inputs.json')
    short = read(ROOT / protocol['short_inputs'] / 'inputs.json')
    key = lambda r: (r['date'], r['cutoff'], r['strike'])
    short_map = {key(r): r for r in short}
    if len(short_map) != len(short) or len({key(r) for r in front}) != len(front) or set(short_map) != {key(r) for r in front}:
        raise ValueError('Funding sample identities differ')
    declarations = read(ROOT / protocol['declarations'] / 'declarations.json')
    tasks, excluded, selected = [], [], []
    for f in front:
        if not f['joint_age_checks'][str(protocol['funding_age_seconds'])]:
            excluded.append(dict(date=f['date'], cutoff=f['cutoff'], strike=f['strike'],
                reason='original_strict_funding_screen', checks=f['joint_age_checks']))
            continue
        quote_rows = read(ROOT / protocol['quote_panel'] / ('quotes_' + f['date'] + '.json'))
        matching = [r for r in quote_rows if (r['cutoff'], r['strike']) == (f['cutoff'], f['strike'])]
        if len(matching) != 1 or not matching[0]['unique_state_timing_checks'][str(protocol['option_age_seconds'])]:
            raise ValueError('Quote timing sample differs')
        quote = matching[0]
        selected.append(dict(date=f['date'], **quote))
        earlier_date = vc['curve_dates_by_observation'].get(f['date'])
        if earlier_date is None or earlier_date not in curves:
            raise ValueError('Required baseline curve is unavailable; no date fallback allowed')
        curve = curves[earlier_date]
        observation = datetime.fromisoformat(f['cutoff'])
        h, s = map(date.fromisoformat, (f['premium_date_assumption'], f['swap_spot_date_assumption']))
        if h != calendar.settlement_after(date.fromisoformat(f['date'])).day:
            raise ValueError('Premium-date assumption differs')
        paths = funding_paths(curve, h, s, f['candidate_settlement_to_spot'], short_map[key(f)]['nodes'])
        if set(paths) != set(protocol['variants']):
            raise ValueError('Funding variants differ')
        forecasts = reconstruct_benchmarks(declarations, f['cutoff'], [protocol['dividend_target']], protocol['last_completed_quarter'])
        if {r['model'] for r in forecasts} != set(protocol['dividend_models']):
            raise ValueError('Nominal cash rules differ')
        observed = {name: quote['quotes'][name]['price_consensus'] for name in ('call', 'put', 'underlying')}
        if any(not 0 < q['BID'] < q['ASK'] for q in observed.values()):
            raise ValueError('Require positive-width observed quote intervals')
        midpoint = (observed['underlying']['BID'] + observed['underlying']['ASK']) / 2
        for forecast in forecasts:
            dividends = (DatedDividend(date.fromisoformat(forecast['ex_date']), date.fromisoformat(forecast['pay_date']),
                float(forecast['regular_dividend_estimate']), 'Unchanged dated nominal benchmark; no extra payment assumed'),)
            common = (observation, date.fromisoformat(protocol['expiry']), midpoint, f['strike'], 0., 'call', dividends)
            full = build_dated_problem(calendar, curve, *common, premium_payment_date=h)
            for variant, funding in paths.items():
                problem = build_premium_normalised_problem(calendar, funding, *common, premium_payment_date=h)
                if problem.provisional_delivery:
                    raise ValueError('Provisional cash delivery is outside the comparison')
                error = check_normalised_equivalence(full, problem) if variant == 'retained_lagged' else None
                raw = dict(observation_id=f['cutoff'] + ':' + str(f['strike']),
                    bid=observed['call']['BID'], ask=observed['call']['ASK'], model=asdict(problem.model),
                    option=asdict(problem.option), decisions=[asdict(d) for d in problem.decisions])
                tasks.append(dict(date=f['date'], cutoff=f['cutoff'], strike=f['strike'], dividend_model=forecast['model'],
                    variant=variant, forecast=forecast, raw_call=raw, observed=observed,
                    premium_payment_date=h.isoformat(), baseline_curve_date=earlier_date,
                    funding_basis=problem.cash_discount_clock, settlement_to_spot=funding.settlement_to_spot,
                    normalisation_factor_error=error, configuration=cfg, protocol=protocol,
                    library=str(library), repeat_python=False))
                restore_problem(raw)
    if not tasks:
        raise ValueError('No supported strict-screen scenarios')
    tasks.sort(key=lambda t: (t['cutoff'], t['strike'], t['dividend_model'], protocol['variants'].index(t['variant'])))
    for variant in protocol['variants']:
        indices = [i for i,t in enumerate(tasks) if t['variant'] == variant]
        for i in {indices[0], indices[-1]}:
            tasks[i]['repeat_python'] = True
    return tasks, dict(selected_groups=selected, excluded_groups=excluded,
        scope='Primary strict funding screen; overlapping development observations',
        empirical_calibration_accepted=False), hashes


def decomposition(raw, row):
    call = restore_problem(raw)
    corner = row['corners']['lower']
    model = replace(call.model, quoted_spot=corner['quoted_spot'], volatility=corner['volatility'])
    ec = european_reference(model, call.option, call.decisions[-1])
    ep = european_reference(model, replace(call.option, kind='put'), call.decisions[-1])
    result = lower_corner_components(call_premium=ec.premium_quote,
        ownership_premium=model.initial_state/model.premium_factor,
        terminal_strike_premium=call.option.strike*call.decisions[-1].settlement_factor/model.premium_factor,
        expected_cash_premium=ec.expected_discounted_payment/model.premium_factor,
        put_premium=corner['prices'][1], cap_allowance=row['cap_allowance']['premium_allowance'],
        numerical_allowance=row['numerical_allowance'])
    result.update(call_reference=asdict(ec), put_reference=asdict(ep),
                  european_identity_residual=result['european_put_premium']-ep.premium_quote)
    if abs(result['conditional_lower_bound']-row['put_envelope']['lower']) > 1e-10 or abs(result['european_identity_residual']) > 2e-6:
        raise ValueError('Independent lower-bound accounting identity failed')
    return result


def evaluate(task):
    cfg, protocol = task['configuration'], task['protocol']
    call = restore_problem(task['raw_call'])
    put = replace(call, bid=task['observed']['put']['BID'], ask=task['observed']['put']['ASK'],
                  option=replace(call.option, kind='put'))
    sb, sa = (task['observed']['underlying'][k] for k in ('BID', 'ASK'))
    row = {k: task[k] for k in ('date', 'cutoff', 'strike', 'dividend_model', 'variant', 'baseline_curve_date',
                               'settlement_to_spot', 'normalisation_factor_error')}
    condition = terminal_call_dominance(call.model, call.option, call.decisions)
    row.update(underlying_bid=sb, underlying_ask=sa, call_bid=call.bid, call_ask=call.ask,
        put_bid=put.bid, put_ask=put.ask, terminal_call_condition=asdict(condition), numerical_screens={})
    if not condition.sufficient_condition_satisfied:
        return dict(row, classification='unresolved_terminal_call_condition')
    references = []
    @lru_cache(maxsize=None)
    def price(spot, sigma):
        r = european_reference(replace(call.model, quoted_spot=spot, volatility=sigma), call.option, call.decisions[-1])
        references.append(r)
        return r.premium_quote
    box = call_state_rectangle(price, sb, sa, call.bid, call.ask, tuple(cfg['volatility_bounds']), cfg['volatility_bracket_width'])
    row.update(rectangle=box, unique_reference_evaluations=len(references),
        maximum_reference_error_estimate=max(r.quadrature_error_estimate + r.omitted_tail_bound for r in references))
    targets = {'call_bid_at_underlying_ask':call.bid, 'call_ask_at_underlying_bid':call.ask}
    residuals = {name:max(abs(root[k]-targets[name]) for k in ('lower_price', 'upper_price'))
                 for name,root in box['roots'].items() if root['status'] == 'bracketed'}
    row['root_endpoint_price_residuals'] = residuals
    if box['status'] != 'candidate_rectangle':
        return dict(row, classification='no_call_candidate_in_declared_rectangle')
    allowance = cfg['numerical_allowance_fraction_of_full_put_spread'] * (put.ask - put.bid)
    native = NativeKernel(Path(task['library']))
    corners, repeats = {}, []
    for name, spot, sigma in [('lower', sa, box['volatility_lower']), ('upper', sb, box['volatility_upper'])]:
        model = replace(put.model, quoted_spot=spot, volatility=sigma)
        threshold = max(model.initial_state, *(put.option.strike*d.settlement_factor for d in put.decisions))
        threshold += sum(e.discounted_cash for e in model.dividends if e.ex_time <= put.option.maturity)
        values = [price_prepared(prepare(model, put.option, g['space_steps'], g['time_steps'],
            cfg['domain_multiple']*threshold, put.decisions), native).premium_quote for g in cfg['price_grids']]
        g = cfg['domain_validation']
        wide = price_prepared(prepare(model, put.option, g['space_steps'], g['time_steps'],
            g['domain_multiple']*threshold, put.decisions), native).premium_quote
        corners[name] = dict(quoted_spot=spot, volatility=sigma, prices=values, wider_domain_price=wide,
            space_time_change=values[1]-values[0], domain_change=wide-values[1])
        if task['repeat_python']:
            g = cfg['price_grids'][1]
            ref = finite_difference_price(model, put.option, g['space_steps'], g['time_steps'],
                cfg['domain_multiple']*threshold, put.decisions).premium_quote
            tolerance = protocol['native_price_tolerance']['absolute'] + protocol['native_price_tolerance']['relative']*abs(ref)
            repeats.append(dict(corner=name, python_price=ref, native_price=values[1], difference=values[1]-ref,
                                tolerance=tolerance, passed=abs(values[1]-ref) <= tolerance))
    cap = single_cash_cap_allowance(replace(put.model, quoted_spot=sb), put.option, box['volatility_upper'])
    envelope = conditional_put_envelope(corners['lower']['prices'][1], corners['upper']['prices'][1], cap['premium_allowance'], allowance)
    screens = dict(root_residuals=all(x <= cfg['price_residual_tolerance'] for x in residuals.values()),
        reference_estimated_error=row['maximum_reference_error_estimate'] <= cfg['price_residual_tolerance'],
        space_time=all(abs(c['space_time_change']) <= allowance for c in corners.values()),
        domain=all(abs(c['domain_change']) <= allowance for c in corners.values()))
    if repeats:
        screens['selected_python_repeat'] = all(r['passed'] for r in repeats)
    classification = ('unresolved_numerical_screen' if not all(screens.values()) else
        'conditionally_above_put_ask' if envelope['lower'] > put.ask else
        'conditionally_below_put_bid' if envelope['upper'] < put.bid else 'not_separated_inconclusive')
    row.update(corners=corners, cap_allowance=cap, numerical_allowance=allowance, put_envelope=envelope,
        numerical_screens=screens, python_repeats=repeats, classification=classification,
        lower_minus_put_ask=envelope['lower']-put.ask, put_bid_minus_upper=put.bid-envelope['upper'])
    row['decomposition'] = decomposition(task['raw_call'], row)
    return row


def summarise(rows, coverage):
    summary = dict(groups=len(coverage['selected_groups']), excluded_groups=len(coverage['excluded_groups']),
        dates=sorted({r['date'] for r in rows}), scenarios=len(rows),
        native_put_valuations=sum(3*len(r.get('corners', {})) for r in rows),
        python_repeats=sum(len(r.get('python_repeats', [])) for r in rows),
        variants={}, empirical_calibration_accepted=False, publication_clearance=False)
    for variant in dict.fromkeys(r['variant'] for r in rows):
        rs = [r for r in rows if r['variant'] == variant]
        gaps = [r['lower_minus_put_ask'] for r in rs if 'lower_minus_put_ask' in r]
        summary['variants'][variant] = dict(classifications=dict(Counter(r['classification'] for r in rs)),
            numerical_failures=dict(Counter(k for r in rs for k,v in r['numerical_screens'].items() if not v)),
            minimum_gap=min(gaps, default=None), maximum_gap=max(gaps, default=None))
    return summary


def verified_results(folder):
    folder = private(folder)
    manifest = read(folder / 'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    rows, coverage = read(folder / 'rows.json'), read(folder / 'coverage.json')
    if summarise(rows, coverage) != read(folder / 'summary.json') or matched_changes(rows) != read(folder / 'comparisons.json'):
        raise ValueError('Recomputed aggregation differs')
    return rows, coverage


def run(output, library, workers):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh experiment directory')
    if not 1 <= workers <= 3:
        raise ValueError('Use one to three workers')
    tasks, coverage, inputs = prepare_tasks(library)
    sources = {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT / 'src/qf_research').rglob('*.py'))}
    for name in ('reprice_settlement_funding.py', 'run_conditional_calibration.py',
                 'diagnose_curve_date_sensitivity.py', 'diagnose_underlying_quote_interval.py',
                 'diagnose_paired_volatility.py', 'review_lseg_pilot.py', 'build_valuation_inputs.py'):
        p = ROOT / 'scripts' / name
        sources[str(p.relative_to(ROOT))] = sha(p)
    output.mkdir(parents=True)
    for name, obj in [('configuration.json', read(CONFIG)), ('tasks.json', tasks), ('coverage.json', coverage)]:
        save(output / name, obj)
    save(output / 'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=sources, workers=workers,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 15 == 0 or len(rows) == len(tasks):
                save(output / 'partial_rows.json', rows)
                print(f'Completed {len(rows)}/{len(tasks)} conditional repricing scenarios', flush=True)
    for name, obj in [('rows.json', rows), ('summary.json', summarise(rows, coverage)),
                      ('comparisons.json', matched_changes(rows))]:
        save(output / name, obj)
    verify_mapping(inputs)
    verify_mapping(sources)
    save(output / 'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start, input_sha256=inputs, source_sha256=sources,
        artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        empirical_calibration_accepted=False, publication_clearance=False))
    print(summarise(rows, coverage), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.library, args.workers)
