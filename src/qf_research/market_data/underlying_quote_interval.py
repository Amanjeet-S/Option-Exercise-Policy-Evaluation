"""A covering volatility interval when an increasing call has uncertain state."""

import math

from .paired_volatility import increasing_price_bracket


def call_state_rectangle(pricer, spot_bid, spot_ask, call_bid, call_ask, bounds, width):
    """Cover all quote-compatible (spot, sigma) in the declared rectangle.

    The caller must establish that call price is increasing in state and
    strictly increasing in volatility. Root brackets are numerical, not
    certified integral enclosures. A covering rectangle need not consist
    entirely of jointly feasible points.
    """
    if not all(math.isfinite(x) for x in (spot_bid, spot_ask, call_bid, call_ask)):
        raise ValueError('Finite underlying and option quotes required')
    if not 0 < spot_bid <= spot_ask or not 0 <= call_bid < call_ask:
        raise ValueError('Positive ordered underlying quotes and a positive option spread required')

    lower = increasing_price_bracket(lambda sigma: pricer(spot_ask, sigma),
                                      call_bid, *bounds, width)
    upper = increasing_price_bracket(lambda sigma: pricer(spot_bid, sigma),
                                      call_ask, *bounds, width)
    # This finite check detects contradictions; monotonicity still needs proof.
    for sigma in bounds:
        if pricer(spot_bid, sigma) > pricer(spot_ask, sigma) + 1e-10:
            raise ValueError('Endpoint prices contradict increasing-state assumption')
    roots = {'call_bid_at_underlying_ask': lower, 'call_ask_at_underlying_bid': upper}
    if lower['status'] == 'above_range' or upper['status'] == 'below_range':
        return {'status': 'no_call_candidate_in_declared_rectangle', 'roots': roots}
    if lower['lower'] > upper['upper']:
        raise ValueError('Root brackets contradict the covering-interval ordering')
    return {'status': 'candidate_rectangle', 'spot_lower': spot_bid, 'spot_upper': spot_ask,
            'volatility_lower': lower['lower'], 'volatility_upper': upper['upper'],
            'clipped_to_volatility_range': lower['status'] != 'bracketed' or upper['status'] != 'bracketed',
            'roots': roots}
