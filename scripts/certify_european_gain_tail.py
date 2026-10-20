"""Outward tail bounds for a restricted European-value max-call dual gain.

The finite-calendar envelope uses established comparison, scaling and
lognormal call identities. Conditional on fixed training information, the
selected stopping date must lie in dates 1,...,J. The bound concerns the ideal
European-value martingale, not a fitted polynomial or hinge martingale.

I bound the positive and negative parts of H_j-E[H_T|F_j] separately. A cap
split B=b_A+b_D then gives a sum of marginal lognormal call tails. Homogeneous
independent GBM, non-negative rates and dividend yields, and non-negative
volatility are required. Supplied ATM certificates are explicit assumptions;
the default route obtains them from the existing verified max-hinge routine.
All float parameters mean their exact binary values. Rational endpoints are
authoritative. Numerical tail enclosures do not certify sampling coverage.
"""
from collections.abc import Mapping, Sequence
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys

try:
    import flint
    from flint import arb, ctx, fmpq
except ImportError:
    flint = arb = ctx = fmpq = None

DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]


def exact(value, name='Parameter'):
    if isinstance(value, bool) or type(value) not in (int, float, Fraction):
        raise ValueError(f'{name} must be a finite exact rational or binary float')
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return Fraction(value)


def ball(value):
    value = exact(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoints(value):
    if not value.is_finite():
        raise ArithmeticError('A non-finite ball cannot certify a tail bound')
    lower, upper = value.lower().fmpq(), value.upper().fmpq()
    return Fraction(int(lower.p), int(lower.q)), Fraction(int(upper.p), int(upper.q))


def rational_record(value):
    value = exact(value)
    try:
        approximation = float(value)
    except OverflowError:
        approximation = None
    if approximation is not None and not math.isfinite(approximation):
        approximation = None
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=approximation)


def interval_record(lower, upper):
    if lower > upper:
        raise ArithmeticError('A tail interval cannot be empty')
    return dict(lower=rational_record(lower), upper=rational_record(upper), width=rational_record(upper-lower))


def _nonnegative_endpoints(value, name):
    lower, upper = endpoints(value)
    if upper < 0:
        raise ArithmeticError(name + ' conflicts with its mathematical non-negativity')
    return max(Fraction(0), lower), upper


def _positive_part_endpoints(value):
    lower, upper = endpoints(value)
    return max(Fraction(0), lower), max(Fraction(0), upper)


def _marginal_endpoints(value, precision):
    """Coarsen outwards on a dyadic grid without serialising tiny exponents.

    A positive tail below 2**(-precision) retains that positive upper bound.
    It is never rounded to an exact zero certificate. Each endpoint moves
    outwards by less than one grid unit; mathematical non-negativity supplies
    the lower clamp. The input ball remains the source enclosure.
    """
    if not value.is_finite():
        raise ArithmeticError('A non-finite marginal cannot certify a call moment')
    upper_ball = value.upper()
    if upper_ball < 0:
        raise ArithmeticError('A call moment conflicts with non-negativity')
    quantum = Fraction(1, 2**precision)
    if upper_ball <= ball(quantum):
        return Fraction(0), Fraction(0) if upper_ball == 0 else quantum
    lower, upper = endpoints(value)
    lower = max(Fraction(0), lower)
    scale = 2**precision
    return Fraction((lower*scale).numerator // (lower*scale).denominator, scale), \
        Fraction(-(-(upper*scale).numerator // (upper*scale).denominator), scale)


def _normal_cdf(value):
    return (-value / arb(2).sqrt()).erfc() / 2


def lognormal_call_moment(spot, time, threshold, *, rate, dividend, volatility, precision=96):
    """Enclose E[(S_time-threshold)+] for one exact-parameter GBM marginal.

    The zero-time and zero-volatility cases are evaluated deterministically.
    A zero threshold uses the first moment and never evaluates its logarithm.
    Non-trivial endpoints are coarsened outwards on the absolute 2**(-precision)
    dyadic grid; a small positive tail is not reported as exactly zero.
    """
    if flint is None:
        raise RuntimeError('Outward certification requires the separate python-flint environment')
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')
    spot, time, threshold, rate, dividend, volatility = (
        exact(value) for value in (spot, time, threshold, rate, dividend, volatility))
    if spot <= 0 or time < 0 or threshold < 0 or volatility < 0:
        raise ValueError('Call moments require a positive spot and non-negative time, threshold and volatility')
    with ctx.workprec(precision):
        if time == 0:
            payoff = max(spot-threshold, Fraction(0))
            return payoff, payoff
        first_moment = ball(spot) * (ball(rate-dividend) * ball(time)).exp()
        if threshold == 0:
            return _marginal_endpoints(first_moment, precision)
        if volatility == 0:
            raw = first_moment-ball(threshold)
            lower, upper = _positive_part_endpoints(raw)
            if upper == 0:
                return Fraction(0), Fraction(0)
            return _marginal_endpoints(raw, precision)
        sd = ball(volatility) * ball(time).sqrt()
        mean = (ball(rate-dividend)-ball(volatility)**2/2) * ball(time)
        if not sd.is_finite() or not sd > 0:
            raise ArithmeticError('The positive marginal standard deviation is not resolved')
        d2 = (ball(spot).log()-ball(threshold).log()+mean)/sd
        value = first_moment * _normal_cdf(d2+sd)-ball(threshold)*_normal_cdf(d2)
        return _marginal_endpoints(value, precision)


def _load_certifier(repository):
    source = Path(repository).resolve() / 'scripts/certify_max_hinge.py'
    spec = importlib.util.spec_from_file_location('european_tail_max_hinge', source)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.certify


def _decode_rational(record):
    if (not isinstance(record, Mapping) or set(record) != {'numerator', 'denominator', 'approximation'}
            or not isinstance(record['numerator'], str) or not isinstance(record['denominator'], str)):
        raise ValueError('Certificate endpoints require explicit exact rational records')
    value = Fraction(int(record['numerator']), int(record['denominator']))
    return value


def _decode_pair(pair):
    if isinstance(pair, Mapping):
        if set(pair) != {'lower', 'upper', 'width'}:
            raise ValueError('A supplied ATM certificate must include its exact width')
        lower, upper = _decode_rational(pair['lower']), _decode_rational(pair['upper'])
        if _decode_rational(pair['width']) != upper-lower:
            raise ValueError('A supplied ATM certificate changes its exact width')
    elif isinstance(pair, Sequence) and not isinstance(pair, (str, bytes)) and len(pair) == 2:
        lower, upper = (exact(value, 'ATM certificate endpoint') for value in pair)
    else:
        raise ValueError('A supplied ATM certificate requires two finite rational endpoints')
    if lower < 0 or upper < lower:
        raise ValueError('Discounted ATM call certificates must be non-negative and ordered')
    return lower, upper


def build_european_restricted_tail_bound(
    spots, exercise_times, *, strike=100, rate=.05, dividend=.10,
    volatility=.20, b_A=20, b_D=30, precision=96, correlation=0,
    atm_certificates=None, atm_certifier=None, repository=DEFAULT_REPOSITORY,
    cutoff_multiplier=4, absolute_tolerance=Fraction(1, 10**10),
    relative_tolerance=Fraction(1, 10**10), evaluation_limit=100000,
):
    """Build a clipping-tail bound without evaluating paths or fitting a policy.

    ``atm_certificates`` optionally maps each positive remaining horizon to
    an enclosure of the discounted ATM max-call C_h(K). Its validity is an
    explicit input premise. Without that mapping, the existing max-hinge API
    supplies undiscounted enclosures, which I discount outwards. A custom
    ``atm_certifier`` is an injectable replacement for that API.

    Coefficient intervals retain rounding uncertainty. Each marginal tail is
    evaluated with a rational upper slope and lower intercept. This gives an
    outward bound without dividing by a slope interval that contains zero.
    """
    if flint is None:
        raise RuntimeError('Outward certification requires the separate python-flint environment')
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')
    if type(evaluation_limit) is not int or evaluation_limit < 1:
        raise ValueError('A positive integer integration evaluation limit is required')
    if not isinstance(spots, Sequence) or isinstance(spots, (str, bytes)):
        raise ValueError('Spots require a non-empty indexed sequence')
    spots = tuple(exact(value, 'Spot') for value in spots)
    if not spots or any(value <= 0 for value in spots):
        raise ValueError('Every GBM spot must be strictly positive')
    if not isinstance(exercise_times, Sequence) or isinstance(exercise_times, (str, bytes)):
        raise ValueError('The full exercise calendar must be an indexed sequence')
    times = tuple(exact(value, 'Exercise date') for value in exercise_times)
    if len(times) < 2 or times[0] != 0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError('The full calendar must increase strictly from zero to a positive expiry')
    strike, rate, dividend, volatility, b_A, b_D, correlation, cutoff_multiplier, absolute_tolerance, relative_tolerance = (
        exact(value) for value in (strike, rate, dividend, volatility, b_A, b_D, correlation,
                                  cutoff_multiplier, absolute_tolerance, relative_tolerance))
    if strike <= 0 or rate < 0 or dividend < 0 or volatility < 0 or b_A <= 0 or b_D < 0:
        raise ValueError('Positive strike and b_A, with non-negative rate, dividend, volatility and b_D, are required')
    if correlation != 0:
        raise ValueError('The max-call envelope supports independent shocks only')
    if cutoff_multiplier <= 1 or absolute_tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError('The integration cutoff must exceed strike and tolerances must be positive')
    if atm_certificates is not None and atm_certifier is not None:
        raise ValueError('Supply ATM certificates or an ATM certifier, not both')
    remaining = tuple(times[-1]-instant for instant in times[1:])
    if atm_certificates is not None:
        if not isinstance(atm_certificates, Mapping):
            raise ValueError('ATM certificates must map exact remaining horizons to enclosures')
        supplied = {exact(key, 'ATM horizon'): _decode_pair(value) for key, value in atm_certificates.items()}
        required = set(remaining)-{Fraction(0)}
        if set(supplied)-{Fraction(0)} != required:
            raise ValueError('Every positive remaining horizon needs exactly one ATM certificate')
        if Fraction(0) in supplied and supplied[Fraction(0)] != (Fraction(0), Fraction(0)):
            raise ValueError('The terminal ATM call must equal zero exactly')
    else:
        supplied = None
    rows = []
    total_upper = Fraction(0)
    certifier = atm_certifier
    with ctx.workprec(precision):
        for index, (instant, horizon) in enumerate(zip(times[1:], remaining), 1):
            discount = _nonnegative_endpoints((-ball(rate)*ball(instant)).exp(), 'Discount factor')
            remaining_discount = _nonnegative_endpoints((-ball(rate)*ball(horizon)).exp(), 'Remaining discount')
            atm_source = None
            if horizon == 0:
                atm = (Fraction(0), Fraction(0))
            elif supplied is not None:
                atm = supplied[horizon]
            elif len(spots) == 1:
                marginal = lognormal_call_moment(strike, horizon, strike, rate=rate, dividend=dividend,
                    volatility=volatility, precision=precision)
                atm = (remaining_discount[0]*marginal[0], remaining_discount[1]*marginal[1])
            elif volatility == 0:
                deterministic = _positive_part_endpoints(ball(strike) * ((ball(rate-dividend)*ball(horizon)).exp()-1))
                atm = (remaining_discount[0]*deterministic[0], remaining_discount[1]*deterministic[1])
            else:
                if certifier is None:
                    certifier = _load_certifier(repository)
                atm_source = certifier([strike]*len(spots), strike=strike, rate=rate, dividend=dividend,
                    volatility=volatility, step=horizon, cutoff=strike*cutoff_multiplier,
                    precision=precision, absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance, correlation=0, evaluation_limit=evaluation_limit)
                raw = _decode_pair(atm_source['expectation'])
                atm = (remaining_discount[0]*raw[0], remaining_discount[1]*raw[1])
            Cbar = atm[1]
            a = _nonnegative_endpoints((-ball(rate)*ball(instant)).exp() *
                (1-(-ball(dividend)*ball(horizon)).exp()), 'Positive envelope slope') if dividend != 0 and horizon != 0 else (Fraction(0), Fraction(0))
            c = _nonnegative_endpoints((-ball(rate)*ball(instant)).exp() * ball(strike) *
                (1-(-ball(rate)*ball(horizon)).exp()), 'Positive envelope intercept') if rate != 0 and horizon != 0 else (Fraction(0), Fraction(0))
            # In one dimension the exact call recession coefficient is
            # exp(-q*h)<=1. The ATM secant coefficient is unnecessary.
            d = (Fraction(0), Fraction(0)) if horizon == 0 or len(spots) == 1 else \
                _positive_part_endpoints((-ball(rate)*ball(horizon)).exp()+ball(Cbar/strike)-1)
            gamma_upper = discount[1]*Cbar
            beta_upper = discount[1]*d[1]
            if b_D < gamma_upper:
                raise ValueError('The declared negative cap does not dominate every discounted ATM upper certificate')
            positive_calls = []
            negative_calls = []
            positive_upper = negative_upper = Fraction(0)
            positive_threshold = negative_threshold = None
            if a[1] != 0:
                positive_threshold = (c[0]+b_A)/a[1]
                positive_calls = [lognormal_call_moment(spot, instant, positive_threshold,
                    rate=rate, dividend=dividend, volatility=volatility, precision=precision) for spot in spots]
                positive_upper = a[1]*sum(upper for lower, upper in positive_calls)
            if beta_upper != 0:
                negative_threshold = strike+(b_D-gamma_upper)/beta_upper
                negative_calls = [lognormal_call_moment(spot, instant, negative_threshold,
                    rate=rate, dividend=dividend, volatility=volatility, precision=precision) for spot in spots]
                negative_upper = beta_upper*sum(upper for lower, upper in negative_calls)
            total_upper += positive_upper+negative_upper
            rows.append(dict(exercise_index=index, time=rational_record(instant), remaining_horizon=rational_record(horizon),
                discounted_atm_call=interval_record(*atm), atm_upper_used=rational_record(Cbar),
                discount=interval_record(*discount), a=interval_record(*a), c=interval_record(*c), d=interval_record(*d),
                recession_coefficient_method='exact_single_asset_dividend_discount' if len(spots)==1 else 'discounted_ATM_secant_bound',
                discounted_atm_intercept_upper=rational_record(gamma_upper), negative_slope_upper=rational_record(beta_upper),
                positive_threshold=None if positive_threshold is None else rational_record(positive_threshold),
                negative_threshold=None if negative_threshold is None else rational_record(negative_threshold),
                positive_marginal_calls=[interval_record(*value) for value in positive_calls],
                negative_marginal_calls=[interval_record(*value) for value in negative_calls],
                positive_tail_upper=rational_record(positive_upper), negative_tail_upper=rational_record(negative_upper),
                atm_integration=atm_source))
    return dict(schema_version=1, author='Amanjeet Singh', quantity='Upper bound for E[(G-B)+] for the restricted European-value dual gain',
        model=dict(spots=[rational_record(value) for value in spots], strike=rational_record(strike),
                   rate=rational_record(rate), dividend=rational_record(dividend), volatility=rational_record(volatility),
                   correlation=rational_record(correlation), exercise_times=[rational_record(value) for value in times]),
        cap_split=dict(b_A=rational_record(b_A), b_D=rational_record(b_D), B=rational_record(b_A+b_D)),
        precision_bits=precision, python_flint_version=flint.__version__, rows=rows,
        marginal_endpoint_grid=rational_record(Fraction(1,2**precision)),
        clipping_tail_allowance=interval_record(Fraction(0), total_upper),
        supplied_atm_certificates=atm_certificates is not None,
        injected_atm_certifier=atm_certifier is not None,
        atm_input_premise='Every discounted ATM upper endpoint dominates its ideal homogeneous independent-GBM ATM max-call.',
        ideal_model_tail_bound_certified=atm_certificates is None and atm_certifier is None,
        financial_decision_certified=False, sampling_coverage_certified=False,
        policy_fitting_executed=False, paths_generated=False,
        scope='Population clipping-tail allowance for the ideal European-value martingale and stopping dates 1,...,J. No result is asserted for fitted hinge or polynomial martingales. Rational upper slopes and lower intercepts preserve coefficient uncertainty; an unresolved positive slope is not discarded. The cap split is supplied before any evaluation sample. This component does not establish an exercise-policy tolerance decision.')
