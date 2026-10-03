"""Joint cash/funding covers in premium units with explicit dependence limits."""
from fractions import Fraction
import math

from .paired_volatility import increasing_price_bracket
from .dividend_feasibility import intersect_interval_unions


def uniform_call_margin(strike, delivery_dates, lower_factors, upper_factors,
                        decision_times, ex_time, cash_upper, payment_upper):
    """Conservative terminal-dominance bound over the entire node rectangle.

    Marginal extremes may be incompatible and therefore yield a loose bound.
    Identical delivery dates, especially the terminal right, cancel exactly.
    Arguments describe post-event exercise rights and one future cash event.
    """
    n = len(delivery_dates)
    if n < 1 or not all(len(v) == n for v in (lower_factors, upper_factors, decision_times)):
        raise ValueError('Inconsistent exercise-factor coverage')
    values = (strike, ex_time, cash_upper, payment_upper, *lower_factors, *upper_factors, *decision_times)
    if any(isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError('Finite model inputs required')
    if min(strike, ex_time, payment_upper, *lower_factors) <= 0 or cash_upper < 0:
        raise ValueError('Positive funding factors and non-negative cash required')
    if decision_times[0] < 0 or ex_time > decision_times[-1]:
        raise ValueError('Require non-negative exercise times and a cash event by maturity')
    if any(a > b for a,b in zip(lower_factors, upper_factors)) or any(
            b <= a for a,b in zip(decision_times, decision_times[1:])):
        raise ValueError('Reversed factor interval or unordered exercise times')
    if any(d > delivery_dates[-1] for d in delivery_dates):
        raise ValueError('Delivery after the terminal delivery is outside this bound')
    mapped = {}
    for d, lo, hi in zip(delivery_dates, lower_factors, upper_factors):
        if d in mapped and mapped[d] != (lo,hi):
            raise ValueError('Identical delivery dates require identical factor intervals')
        mapped[d] = (lo,hi)
    margins = []
    for d,lo,t in zip(delivery_dates,lower_factors,decision_times):
        difference = Fraction(0) if d == delivery_dates[-1] else Fraction(lo)-Fraction(upper_factors[-1])
        cash = Fraction(cash_upper)*Fraction(payment_upper) if ex_time > t else Fraction(0)
        value = Fraction(strike)*difference-cash
        approximate = float(value)
        margins.append(math.nextafter(approximate, -math.inf) if Fraction(approximate) > value else approximate)
    return dict(sufficient_condition_satisfied=all(v >= 0 for v in margins),
        minimum_margin_lower=min(margins), margins_lower=margins,
        tight_over_shared_nodes=False)


def joint_call_cover(pricer, spot_bid, spot_ask, call_bid, call_ask,
                     cash_lower, cash_upper, volatility_bounds, width):
    """Cover all state/cash/node combinations using ordered call envelopes.

    pricer(side, spot, cash, volatility) evaluates the lower or upper factor
    path. The caller must establish factor ordering and uniform call dominance.
    """
    values = (spot_bid,spot_ask,call_bid,call_ask,cash_lower,cash_upper)
    if any(isinstance(v,bool) or not math.isfinite(v) for v in values):
        raise ValueError('Finite quote and cash bounds required')
    if not 0 < spot_bid <= spot_ask or not 0 <= call_bid < call_ask or not 0 <= cash_lower <= cash_upper:
        raise ValueError('Invalid state, quote or cash interval')
    for sigma in volatility_bounds:
        if pricer('lower',spot_ask,cash_lower,sigma)+1e-10 < pricer('upper',spot_bid,cash_upper,sigma):
            raise ValueError('Call envelopes contradict the required ordering')
    lower = increasing_price_bracket(lambda s:pricer('lower',spot_ask,cash_lower,s),
                                     call_bid,*volatility_bounds,width)
    upper = increasing_price_bracket(lambda s:pricer('upper',spot_bid,cash_upper,s),
                                     call_ask,*volatility_bounds,width)
    roots = dict(lower_factor_lower_cash_call_bid=lower, upper_factor_upper_cash_call_ask=upper)
    if lower['status']=='above_range' or upper['status']=='below_range':
        return dict(status='no_call_candidate',roots=roots)
    if lower['lower'] > upper['upper']:
        raise ValueError('Reversed covering volatility endpoints')
    return dict(status='candidate_cover',roots=roots,volatility_lower=lower['lower'],volatility_upper=upper['upper'],
                clipped_to_volatility_range=lower['status']!='bracketed' or upper['status']!='bracketed')


def session_cash_sets(rows):
    """An additional constant-cash assumption; no common intraday state imposed."""
    if not rows or len({(r['cutoff'],r['strike']) for r in rows}) != len(rows):
        raise ValueError('Require unique observed cutoff/strike pairs')
    out = []
    for day in sorted({r['date'] for r in rows}):
        group = [r for r in rows if r['date']==day]
        common = intersect_interval_unions([r['outer_cash_set'] for r in group])
        out.append(dict(date=day, observation_pairs=len(group), distinct_cutoffs=len({r['cutoff'] for r in group}),
            common_cash_set=common, session_constant_cash_assumption=True,
            simultaneous_cross_strike_sample=len({r['cutoff'] for r in group})==1,
            excluded_within_declared_domain=not common, existence_of_joint_fit_established=False))
    return out
