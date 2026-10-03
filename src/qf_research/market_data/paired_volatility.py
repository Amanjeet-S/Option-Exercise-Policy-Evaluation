"""Numerical call inversion and a one-cash put volatility-comparison allowance."""

from fractions import Fraction
import math


def increasing_price_bracket(pricer, target, low, high, width):
    """Retain a bisection bracket for an assumed strictly increasing continuous price.

    The caller must justify monotonicity and audit the pricing error. This is
    numerical bracketing, not interval arithmetic or an error-certified root.
    """
    if not all(math.isfinite(x) for x in (target, low, high, width)) or target < 0 or not 0 <= low < high or width <= 0:
        raise ValueError('Finite target and increasing non-negative volatility bounds required')
    def value(sigma):
        result = float(pricer(sigma))
        if not math.isfinite(result) or result < 0:
            raise ValueError('Invalid non-negative pricing result')
        return result
    a, b = float(low), float(high)
    pa, pb = value(a), value(b)
    if pa >= pb:raise ValueError('Endpoint prices contradict strictly increasing-price assumption')
    if target < pa:return {'status':'below_range','lower':a,'upper':a,'lower_price':pa,'upper_price':pa,'iterations':0}
    if target > pb:return {'status':'above_range','lower':b,'upper':b,'lower_price':pb,'upper_price':pb,'iterations':0}
    if target == pa:return {'status':'bracketed','lower':a,'upper':a,'lower_price':pa,'upper_price':pa,'iterations':0}
    if target == pb:return {'status':'bracketed','lower':b,'upper':b,'lower_price':pb,'upper_price':pb,'iterations':0}
    iterations = 0
    while b-a > width:
        middle = (a+b)/2
        if middle in (a,b):raise ValueError('Requested volatility width is below floating-point resolution')
        pm = value(middle)
        if pm < pa-1e-10 or pm > pb+1e-10:
            raise ValueError('Evaluated prices contradict increasing-price assumption')
        if pm < target:a,pa = middle,pm
        else:b,pb = middle,pm
        iterations += 1
        if iterations > 200:raise ValueError('Bisection iteration limit exceeded')
    return {'status':'bracketed','lower':a,'upper':b,'lower_price':pa,'upper_price':pb,'iterations':iterations}


def call_quote_interval(pricer, bid, ask, bounds, width):
    if not math.isfinite(bid) or not math.isfinite(ask) or not 0 <= bid < ask:
        raise ValueError('Non-negative call bid and positive width required')
    roots = {name:increasing_price_bracket(pricer,target,*bounds,width)
             for name,target in [('bid',bid),('midpoint',(bid+ask)/2),('ask',ask)]}
    if roots['bid']['status'] == 'above_range' or roots['ask']['status'] == 'below_range':
        return {'status':'no_quote_compatible_volatility_in_declared_range','roots':roots}
    return {'status':'candidate_interval','roots':roots,'lower':roots['bid']['lower'],
            'upper':roots['ask']['upper'],
            'midpoint_volatility':sum(roots['midpoint'][key] for key in ('lower','upper'))/2,
            'midpoint_target_bracketed':roots['midpoint']['status']=='bracketed',
            'clipped_to_volatility_range':roots['bid']['status']!='bracketed' or roots['ask']['status']!='bracketed'}


def _up(value):
    result = float(value)
    if not math.isfinite(result):raise ValueError('Bound exceeds finite float range')
    return math.nextafter(result,math.inf) if Fraction(result) < value else result


def single_cash_cap_allowance(model, option, volatility_upper):
    """Upper bound on E[(c-Y_e)^+]/premium_factor for one future cash event.

    Before the sole event Y_e is driftless lognormal. Markov's inequality with
    the fourth inverse moment gives c^5/Y0^4 * exp(10*sigma^2*e). For x<1,
    exp(x)<=1/(1-x) yields an exact rational bound on binary inputs. The cap
    itself is a valid fallback. With no in-horizon event the allowance is zero.
    """
    if not math.isfinite(volatility_upper) or volatility_upper < 0:
        raise ValueError('Finite non-negative volatility upper bound required')
    relevant = [e for e in model.dividends if e.ex_time <= option.maturity]
    if len(relevant) > 1:raise ValueError('This comparison supports at most one in-horizon cash event')
    if not relevant or relevant[0].discounted_cash == 0:
        return {'premium_allowance':0.,'method':'no_in_horizon_cash','moment_exponent':0.}
    event = relevant[0]
    cash, initial = Fraction(event.discounted_cash), Fraction(model.initial_state)
    sigma, ex = Fraction(volatility_upper), Fraction(event.ex_time)
    exponent = 10*sigma*sigma*ex
    if sigma == 0:
        bound = max(cash-initial,Fraction(0)); method = 'deterministic_shortfall'
    elif exponent < 1:
        bound = min(cash,cash**5/initial**4/(1-exponent)); method = 'fourth_inverse_moment_rational_majorant'
    else:
        bound = cash; method = 'trivial_cash_cap'
    return {'premium_allowance':_up(bound/Fraction(model.premium_factor)),
            'method':method,'moment_exponent':float(exponent)}


def conditional_put_envelope(lower_price, upper_price, cap_allowance, numerical_allowance):
    """Enclose using assumed endpoint error allowances and the cap comparison.

    This function does not establish that a numerical error allowance is valid.
    Outward rounding only protects arithmetic on the supplied binary inputs.
    """
    values = (lower_price,upper_price,cap_allowance,numerical_allowance)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Finite non-negative prices and allowances required')
    error = Fraction(cap_allowance)+Fraction(numerical_allowance)
    lower = max(Fraction(0),Fraction(lower_price)-error)
    upper = Fraction(upper_price)+error
    if lower > upper:raise ValueError('Endpoint values contradict the supplied comparison allowances')
    return {'lower':-_up(-lower),'upper':_up(upper),
            'conditional_on_numerical_allowance':True}
