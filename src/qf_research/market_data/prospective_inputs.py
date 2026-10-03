"""Fixed-clock observations and call-only inputs for the prospective study.

This module operates on supplied evidence. It neither retrieves data nor
certifies funding conventions. Missing observations remain explicit cases.
"""
from dataclasses import asdict, replace
from datetime import date, datetime
import math

from scipy.optimize import brentq

from .dense_quotes import inspect_funding_quote, merge_chunks, slice_window
from .front_funding import inspect_front_factor
from .occ_series import single_underlying
from .option_quotes import response_blocks
from .prospective_sample import request_plan
from .quote_review import review_quote_block, reviewed_pair
from .settlement_funding import SettlementFunding, build_premium_normalised_problem
from .short_ois import build_spot_band
from ..optimal_stopping.exercise_dominance import terminal_call_dominance
from ..optimal_stopping.funded_cash import european_reference


def contract_grid(rows, config):
    """Resolve identities from reference fields, without decoding invented RICs."""
    expected = {(float(k), s) for k in config['strikes'] for s in ('Call', 'Put')}
    result = {}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError('Incomplete contract reference grid')
    for row in rows:
        key = (float(row['StrikePrice']), row['CallPutOption'])
        if key not in expected or key in result:
            raise ValueError('Unexpected or duplicate contract identity')
        if (single_underlying(row['UnderlyingQuoteRIC']) != config['underlying'] or
                row['ExpiryDate'][:10] != config['expiry'] or
                row.get('ExerciseStyle') != 'A' or row.get('ExchangeCode') != 'OPQ' or
                row.get('Currency') != 'USD' or row.get('StrikeMultiplier') != 1):
            raise ValueError('Contract terms differ from the maintained specification')
        result[key] = row['RIC']
    if set(result.values()) != set(config['option_rics']):
        raise ValueError('Contract RIC coverage differs')
    return result


def review_day(config, slot, payloads, metadata, contracts):
    """Review all prescribed groups, including every missing or defective one.

    Request arguments and payload hashes must be checked by the IO layer first.
    None denotes a failed or missing response, never an empty successful quote.
    A missing chunk holds its instrument even if other chunks contain a quote.
    """
    grid = contract_grid(contracts, config)
    pieces, failures = {}, {}
    for request in request_plan(config, slot, config['option_rics']):
        args, name = request['arguments'], request['name']
        try:
            if payloads.get(name) is None:
                raise ValueError('Missing request response')
            blocks = response_blocks(payloads[name])
            rics = [b.get('universe', {}).get('ric') for b in blocks]
            if len(rics) != len(set(rics)) or set(rics) != set(args['universe']):
                raise ValueError('Response instrument coverage differs')
            for block in blocks:
                ric = block['universe']['ric']
                pieces.setdefault(ric, []).append(dict(start=args['start'], end=args['end'], block=block))
        except (ValueError, KeyError, TypeError) as exc:
            for ric in args['universe']:
                failures.setdefault(ric, []).append(dict(request=name, reason=str(exc)))
    joined = {}
    for ric, chunks in pieces.items():
        try:
            joined[ric] = merge_chunks(chunks, config['event_count_cap'])
        except (ValueError, KeyError, TypeError) as exc:
            failures.setdefault(ric, []).append(dict(reason=str(exc)))
    cutoff = slot['cutoff_utc']
    quotes, rates, front = {}, {}, {}
    quote_ages = [config['primary_quote_age_seconds'], *config['quote_sensitivity_ages_seconds']]
    funding_ages = [config['primary_funding_age_seconds'], *config['funding_sensitivity_ages_seconds']]
    for group, rics, target in (
            ('quotes', config['option_rics'] + [config['underlying']], quotes),
            ('term', config['term_rics'], rates), ('front', config['front_rics'], front)):
        for ric in rics:
            if ric in failures or ric not in joined:
                continue
            try:
                window = config['quote_lookback_seconds'] if group == 'quotes' else config['funding_lookback_seconds']
                block = slice_window(joined[ric], cutoff, window)
                # Each source cap was checked before merging. This is a local view.
                offline_cap = max(config['event_count_cap'], len(block['data'])) + 1
                if group == 'quotes':
                    target[ric] = review_quote_block(block, metadata[ric], cutoff, window, offline_cap)
                elif group == 'term':
                    target[ric] = inspect_funding_quote(block, cutoff, funding_ages)
                else:
                    target[ric] = inspect_front_factor(block, cutoff, funding_ages, count_cap=offline_cap)
            except (ValueError, KeyError, TypeError) as exc:
                failures.setdefault(ric, []).append(dict(reason=str(exc)))
    funding_checks = {str(age): all(r in rates and rates[r]['mechanical_age_checks'][str(age)]
        for r in config['term_rics']) and all(r in front and front[r]['mechanical_age_checks'][str(age)]
        for r in config['front_rics']) for age in funding_ages}
    groups = []
    for strike in config['strikes']:
        rics = dict(call=grid[(float(strike), 'Call')], put=grid[(float(strike), 'Put')],
                    underlying=config['underlying'])
        complete = all(ric in quotes for ric in rics.values())
        pairs = [] if not complete else [reviewed_pair(quotes[rics[side]], quotes[rics['underlying']], quote_ages)
                                        for side in ('call', 'put')]
        checks = {str(age): complete and all(p['unique_state_and_age_checks'][str(age)] for p in pairs)
                  for age in quote_ages}
        groups.append(dict(date=slot['date'], cutoff=cutoff, strike=strike,
            quotes={side: quotes.get(ric) for side, ric in rics.items()}, pairings=pairs,
            quote_age_checks=checks, funding_age_checks=funding_checks,
            primary_quotes_pass=checks[str(config['primary_quote_age_seconds'])],
            primary_funding_pass=funding_checks[str(config['primary_funding_age_seconds'])]))
    return dict(groups=groups, term=rates, front=front, failures=failures,
        source_coverage={ric: {k: v for k, v in row.items() if k not in ('block', 'times_ns')}
                         for ric, row in joined.items()})


def dated_funding(config, slot, audit, schedules, calendar):
    """Use dated accrual endpoints and historical rate midpoints only."""
    day = date.fromisoformat(slot['date'])
    if (not schedules or any(s.trade_date != day for s in schedules) or
            {s.ric for s in schedules} != set(config['term_rics'])):
        raise ValueError('Missing or misdated swap schedules')
    age = str(config['primary_funding_age_seconds'])
    if not all(g['funding_age_checks'][age] for g in audit['groups']):
        raise ValueError('Primary funding screen failed')
    h, s = calendar.settlement_after(day).day, calendar.settlement_after(day, lag=2).day
    if any(schedule.start != s for schedule in schedules):
        raise ValueError('Swap accrual start differs from candidate front date')
    quotes = {ric: tuple(audit['term'][ric]['latest_candidates'][0][side] for side in ('BID', 'ASK'))
              for ric in config['term_rics']}
    band = build_spot_band(schedules, quotes)
    factors = [audit['front'][ric]['factor'] for ric in config['front_rics']]
    funding = SettlementFunding(h, band.midpoint, factors[1] / factors[0])
    return funding, dict(premium_date=h.isoformat(), spot_date=s.isoformat(),
        candidate_front_ratio=funding.settlement_to_spot,
        nodes=[dict(date=d.isoformat(), lower=lo, midpoint=mid, upper=hi)
               for d, lo, mid, hi in zip(band.lower.dates, band.lower.factors,
                                       band.midpoint.factors, band.upper.factors)],
        schedules=[{k: v.isoformat() if isinstance(v, date) else v for k, v in asdict(x).items()}
                   for x in schedules], historical_conventions_verified=False)


def interval_residual(value, bid, ask):
    if not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (value, bid, ask)) or not 0 <= bid < ask:
        raise ValueError('Finite value and a positive-width quote interval required')
    return value - bid if value < bid else value - ask if value > ask else 0.


def calibrate_call(config, calendar, group, funding):
    """Fit one volatility with no put-price argument in the root objective."""
    if not group['primary_quotes_pass'] or not group['primary_funding_pass']:
        raise ValueError('Primary measurement screens failed')
    observed = {side: group['quotes'][side]['price_consensus'] for side in ('underlying', 'call', 'put')}
    midpoint = lambda side: (observed[side]['BID'] + observed[side]['ASK']) / 2
    low, high = config['calibration']['volatility_bracket']
    if not 0 < low < high or config['calibration']['cash'] != 0:
        raise ValueError('Require the declared no-cash volatility bracket')
    problem = build_premium_normalised_problem(calendar, funding, datetime.fromisoformat(group['cutoff']),
        date.fromisoformat(config['expiry']), midpoint('underlying'), group['strike'], low, 'call', (),
        premium_payment_date=funding.settlement_date)
    dominance = terminal_call_dominance(problem.model, problem.option, problem.decisions)
    base = dict(date=group['date'], cutoff=group['cutoff'], strike=group['strike'],
        observed=observed, call_midpoint=midpoint('call'), call_dominance=asdict(dominance))
    if not dominance.sufficient_condition_satisfied:
        return dict(base, status='terminal_call_dominance_failed')
    target = midpoint('call')
    def residual(volatility):
        model = replace(problem.model, volatility=volatility)
        return european_reference(model, problem.option, problem.decisions[-1]).premium_quote - target
    endpoints = [residual(low), residual(high)]
    base['root'] = dict(bracket=[low, high], endpoint_residuals=endpoints)
    if endpoints[0] > 0 or endpoints[1] < 0:
        return dict(base, status='call_root_unbracketed')
    volatility = brentq(residual, low, high, xtol=1e-14, rtol=1e-14)
    error = residual(volatility)
    base['root'].update(volatility=volatility, residual_quote=error)
    if abs(error) > config['computation']['call_root_tolerance_quote']:
        return dict(base, status='call_root_residual_failed')
    model = replace(problem.model, volatility=volatility)
    option = replace(problem.option, kind='put')
    return dict(base, status='call_calibrated', model=asdict(model), option=asdict(option),
        decisions=[asdict(r) for r in problem.decisions],
        deliveries=[dict(decision=t.isoformat(), payment=d.isoformat(), evidence=e)
                    for t, d, e in problem.deliveries],
        premium_date=problem.premium_payment_date.isoformat(),
        value_unit='USD per share in premium-settlement units', put_used_in_calibration=False)
