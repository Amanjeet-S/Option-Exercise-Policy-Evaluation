"""Cover joint quote-compatible state, volatility and deterministic cash inputs."""

import math

from .paired_volatility import increasing_price_bracket


def cash_state_volatility_cover(pricer, spot_bid, spot_ask, call_bid, call_ask,
                                cash_lower, cash_upper, volatility_bounds, width):
    """Necessary cover using call state/cash ordering and volatility monotonicity.

    The caller must establish terminal-call dominance throughout the cash cell.
    Numerical bisection is not interval arithmetic. Finite ordering checks are
    diagnostic; they cannot supply the underlying monotonicity proof.
    """
    values = (spot_bid,spot_ask,call_bid,call_ask,cash_lower,cash_upper)
    if any(isinstance(x,bool) or not math.isfinite(x) for x in values):
        raise ValueError('Finite quote and cash endpoints required')
    if not 0 < spot_bid <= spot_ask or not 0 <= call_bid < call_ask or not 0 <= cash_lower <= cash_upper:
        raise ValueError('Ordered positive state, positive option spread and non-negative cash required')
    for sigma in volatility_bounds:
        # The extremes must order every feasible call value in this cash cell.
        for cash in {cash_lower,cash_upper}:
            if pricer(spot_bid,cash,sigma) > pricer(spot_ask,cash,sigma)+1e-10:
                raise ValueError('Call values contradict increasing-state ordering')
        for spot in {spot_bid,spot_ask}:
            if pricer(spot,cash_lower,sigma)+1e-10 < pricer(spot,cash_upper,sigma):
                raise ValueError('Call values contradict decreasing-cash ordering')
    lower = increasing_price_bracket(lambda sigma:pricer(spot_ask,cash_lower,sigma),
                                    call_bid,*volatility_bounds,width)
    upper = increasing_price_bracket(lambda sigma:pricer(spot_bid,cash_upper,sigma),
                                    call_ask,*volatility_bounds,width)
    roots = {'lower_cash_call_bid_at_spot_ask':lower,'upper_cash_call_ask_at_spot_bid':upper}
    if lower['status']=='above_range' or upper['status']=='below_range':
        return {'status':'no_call_candidate','roots':roots}
    if lower['lower'] > upper['upper']:
        raise ValueError('Inconsistent covering volatility endpoints')
    return {'status':'candidate_cover','roots':roots,'cash_lower':cash_lower,'cash_upper':cash_upper,
            'spot_lower':spot_bid,'spot_upper':spot_ask,'volatility_lower':lower['lower'],
            'volatility_upper':upper['upper'],
            'clipped_to_volatility_range':lower['status']!='bracketed' or upper['status']!='bracketed'}


def merge_closed_intervals(intervals):
    """Retain a union, including isolated boundary points; never fill gaps."""
    validated = []
    for pair in intervals:
        if len(pair)!=2 or any(isinstance(x,bool) or not math.isfinite(x) for x in pair) or pair[0]>pair[1]:
            raise ValueError('Finite ordered closed intervals required')
        validated.append(tuple(pair))
    result = []
    for low,high in sorted(validated):
        if result and low <= result[-1][1]:result[-1][1]=max(result[-1][1],high)
        else:result.append([low,high])
    return result


def intersect_interval_unions(unions):
    """Intersect necessary sets, preserving disconnected and singleton pieces."""
    if not unions:raise ValueError('At least one interval union required')
    result = merge_closed_intervals(unions[0])
    for union in unions[1:]:
        other = merge_closed_intervals(union)
        result = merge_closed_intervals([[max(a,c),min(b,d)] for a,b in result for c,d in other
                                         if max(a,c)<=min(b,d)])
    return result
