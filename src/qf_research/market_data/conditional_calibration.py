"""Selection and fitting for a narrowly defined retrospective model diagnostic.

This does not change the earlier empirical-acceptance decisions. Quote widths
scale the loss; they are not estimates of independent observation variance.
"""

from dataclasses import dataclass, replace
from datetime import datetime
import math
from zoneinfo import ZoneInfo

import numpy as np
from scipy.optimize import minimize_scalar

from ..optimal_stopping.contracts import Option, positive_integer
from ..optimal_stopping.funded_cash import FundedCashModel, FundedExercise, finite_difference_price


def select_october_pairs(review, references, config):
    train, held = set(config['train_dates']), set(config['heldout_dates'])
    if not train or not held or train & held or max(train) >= min(held):
        raise ValueError('Training dates must strictly precede distinct held-out dates')
    by_ric = {r['RIC']: r for r in references}
    if len(by_ric) != len(references):
        raise ValueError('Duplicate reference identity')
    events = {(e['ric'], e['cutoff_utc']): e for e in review['events']}
    if len(events) != len(review['events']):
        raise ValueError('Duplicate instrument/date event audit')
    selected, exclusions, seen = [], [], set()
    threshold = str(config['age_threshold_seconds'])
    for pair in review['pairs']:
        ric, cutoff = pair['option_ric'], pair['cutoff_utc']
        key = ric, cutoff
        if key in seen:
            raise ValueError('Duplicate option/date pair')
        seen.add(key)
        ref = by_ric[ric]
        when = datetime.fromisoformat(cutoff)
        if when.tzinfo is None:
            raise ValueError('Observation cutoff requires a timezone')
        day = when.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        if day not in train | held:
            raise ValueError('Observation date outside declared split')
        side = ref['CallPutOption'].lower()
        reasons = []
        if ref['ExpiryDate'][:10] != config['expiry']:
            reasons.append('outside_expiry_scope')
        elif ref['StrikePrice'] not in config['strikes'] or side not in config['kinds']:
            reasons.append('outside_contract_scope')
        else:
            option = events[key]
            underlying = events[(pair['underlying_ric'], cutoff)]
            if not pair['price_and_age_checks'][threshold]:
                reasons.append('price_or_dual_timestamp_screen_failed')
            if not option['quote_state_resolved'] or not underlying['quote_state_resolved']:
                reasons.append('nonunique_retained_quote_state')
            if not pair['unique_state_and_age_checks'][threshold] and not reasons:
                raise ValueError('Inconsistent reviewed pair status')
            if not reasons:
                bid, ask = [float(option['price_consensus'][x]) for x in ('BID', 'ASK')]
                sbid, sask = [float(underlying['price_consensus'][x]) for x in ('BID', 'ASK')]
                if not all(math.isfinite(x) for x in (bid, ask, sbid, sask)):
                    raise ValueError('Non-finite quote')
                if bid < 0 or ask < bid or sbid <= 0 or sask < sbid:
                    raise ValueError('Invalid quote state despite reviewed acceptance')
                if ask == bid:
                    reasons.append('zero_option_width_not_weightable')
        identity = {'observation_id': ric+'@'+cutoff, 'ric':ric, 'cutoff':cutoff,
                    'observation_date':day, 'split':'train' if day in train else 'heldout'}
        if reasons:
            exclusions.append({**identity, 'reasons': reasons})
            continue
        selected.append({**identity, 'expiry':config['expiry'], 'kind':side,
                         'strike':float(ref['StrikePrice']), 'bid':bid, 'ask':ask,
                         'underlying_bid':sbid, 'underlying_ask':sask,
                         'quoted_spot':(sbid+sask)/2,
                         'original_empirical_acceptance':pair['empirical_acceptance']})
    if not selected or {r['split'] for r in selected} != {'train', 'heldout'}:
        raise ValueError('Both training and held-out samples must survive selection')
    return sorted(selected, key=lambda r:r['observation_id']), sorted(exclusions, key=lambda r:r['observation_id'])


@dataclass(frozen=True)
class CalibrationQuote:
    observation_id: str
    split: str
    bid: float
    ask: float
    model: FundedCashModel
    option: Option
    decisions: tuple[FundedExercise, ...]

    def __post_init__(self):
        if self.split not in ('train', 'heldout') or not self.observation_id:
            raise ValueError('Explicit observation identity and split are required')
        if not math.isfinite(self.bid) or not math.isfinite(self.ask) or not 0 <= self.bid < self.ask:
            raise ValueError('Require finite non-negative bid and strictly positive width')

    @property
    def midpoint(self):
        return (self.bid+self.ask)/2

    @property
    def half_spread(self):
        return (self.ask-self.bid)/2


def price_quote(record, volatility, grid, domain_multiple):
    model = replace(record.model, volatility=float(volatility))
    threshold = max(model.initial_state, *(record.option.strike*d.settlement_factor for d in record.decisions))
    threshold += sum(e.discounted_cash for e in model.dividends if e.ex_time <= record.option.maturity)
    return finite_difference_price(model, record.option, grid['space_steps'], grid['time_steps'],
                                   domain_multiple*threshold, record.decisions).premium_quote


def scaled_midpoint_loss(records, prices):
    if not records or len(records) != len(prices):
        raise ValueError('Non-empty matched records and prices are required')
    values = np.asarray(prices, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Model prices must be finite and non-negative')
    residuals = [(p-r.midpoint)/r.half_spread for r,p in zip(records,values)]
    result = float(np.mean(np.square(residuals)))
    if not math.isfinite(result):
        raise ValueError('Non-finite scaled objective')
    return result


def fit_shared_volatility(records, config, grid, *, pricer=price_quote):
    """Fit training rows only, retaining the scan, refinements and boundaries.

    The deterministic scan plus bounded local searches is an optimisation
    diagnostic, not a proof of global optimality for a non-convex objective.
    No held-out observation is accepted by this function.
    """
    if not records or any(r.split != 'train' for r in records):
        raise ValueError('Calibration accepts training observations only')
    if len({r.observation_id for r in records}) != len(records):
        raise ValueError('Duplicate training identity')
    low, high = config['volatility_bounds']
    if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low < high:
        raise ValueError('Invalid volatility interval')
    count = positive_integer(config['search_grid_points'], 'search_grid_points', 3)
    tolerance = config['volatility_tolerance']
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive volatility tolerance required')
    cache = {}
    def objective(sigma):
        sigma = float(sigma)
        if sigma not in cache:
            cache[sigma] = scaled_midpoint_loss(records, [pricer(r,sigma,grid,config['domain_multiple']) for r in records])
        return cache[sigma]
    scan = [{'volatility':float(s), 'loss':objective(s)} for s in np.linspace(low,high,count)]
    candidates = [dict(scan[0], method='lower_boundary'), dict(scan[-1], method='upper_boundary')]
    candidates += [dict(s, method='grid') for s in scan[1:-1]]
    searches = []
    intervals = [(scan[0]['volatility'],scan[1]['volatility']), (scan[-2]['volatility'],scan[-1]['volatility'])]
    intervals += [(scan[i-1]['volatility'],scan[i+1]['volatility']) for i in range(1,count-1)
                  if scan[i]['loss'] <= min(scan[i-1]['loss'],scan[i+1]['loss'])]
    for a,b in intervals:
        result = minimize_scalar(objective, bounds=(a,b), method='bounded',
                                 options={'xatol':tolerance, 'maxiter':80})
        entry = {'left':a,'right':b,'volatility':float(result.x),'loss':float(result.fun),
                 'success':bool(result.success),'evaluations':int(result.nfev)}
        searches.append(entry)
        if result.success:
            candidates.append({'volatility':entry['volatility'],'loss':entry['loss'],'method':'bounded_local'})
    best = min(candidates, key=lambda c:(c['loss'],c['volatility']))
    return {'best':best, 'grid':grid, 'scan':scan, 'local_searches':searches,
            'unique_objective_evaluations':len(cache), 'training_count':len(records),
            'boundary_selected':best['volatility'] in (low,high),
            'all_local_searches_converged':all(s['success'] for s in searches),
            'global_optimality_certified':False}


def quote_diagnostics(record, price):
    scaled_midpoint_loss([record],[price])
    residual = price-record.midpoint
    outside = max(record.bid-price,0.,price-record.ask)
    return {'price':price,'midpoint_residual':residual,'scaled_midpoint_residual':residual/record.half_spread,
            'outside_quote_distance':outside,'outside_quote_distance_in_full_spreads':outside/(record.ask-record.bid),
            'inside_quote_interval':record.bid <= price <= record.ask}
