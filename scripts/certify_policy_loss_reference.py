"""Enclose population shortfall of threshold policies in a two-right GBM call.

The policy exercises at the first right when S_1 > theta and otherwise waits;
thresholds theta >= K and an always-wait policy are supported. In the maintained
model r >= 0, q > 0 and sigma > 0, the discounted exercise advantage D(s)=H_1(s)
-C_1(s) is negative below K and strictly increasing above K. Its unique zero b
therefore defines the optimal first-right decision. A threshold policy's loss
is the integral of |D(s)| over the interval between theta and b. These are
standard conditional-expectation and optimal-stopping identities, not claims
of theoretical novelty.

Arb signs certify a rational root bracket [b_l,b_u]. Acb integrates only smooth
pieces outside this bracket. Its omitted strip is bounded using |D'| <= e^(-rt)
and the lognormal density bound 1/(s*sigma*sqrt(t)*sqrt(2*pi)). The always-wait
integral additionally has a Jensen/lognormal tail allowance. Endpoints are
exact rationals and enclose ideal-model population losses. This does not
certify the generation of simulation paths or their sampling coverage.

Float arguments denote their exact binary values. Sources: Black and Scholes
(1973), Journal of Political Economy 81(3), 637-654, doi:10.1086/260062;
Andersen and Broadie (2004), Management Science 50(9), 1222-1234,
doi:10.1287/mnsc.1040.0258 (process parameters; the two-right calendar is a
separate diagnostic); Johansson (2017), IEEE Transactions on Computers 66(8),
1281-1292, doi:10.1109/TC.2017.2690633 (Arb arithmetic). Acb integration API:
https://python-flint.readthedocs.io/en/latest/acb.html#flint.acb.integral
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter, process_time

try:
    import flint
    from flint import acb, arb, ctx, fmpq
except ImportError:
    flint = acb = arb = ctx = fmpq = None


def exact(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('A finite integer, binary float or Fraction is required')
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError('A finite scalar is required')
    return Fraction.from_float(value)


def ball(value):
    value = exact(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoints(value):
    if not value.is_finite():
        raise ArithmeticError('A finite real ball is required for certification')
    lo, hi = value.lower().fmpq(), value.upper().fmpq()
    return Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q))


def rational_record(value):
    value = exact(value)
    try:
        approximation = float(value)
    except OverflowError:
        approximation = None
    if approximation is not None and not math.isfinite(approximation):
        approximation = None
    return dict(numerator=str(value.numerator), denominator=str(value.denominator),
                approximation=approximation)


def interval_record(lo, hi):
    if lo > hi:
        raise ArithmeticError('An empty interval is not a certificate')
    return dict(lower=rational_record(lo), upper=rational_record(hi),
                width=rational_record(hi-lo))


def normal_cdf(value):
    return (-value/arb(2).sqrt()).erfc()/2


class TwoRightModel:
    """Ball evaluation for the stated continuous-dividend GBM model."""

    def __init__(self, *, spot, strike, rate, dividend_yield, volatility,
                 first_right, maturity):
        self.parameters = dict(spot=spot, strike=strike, rate=rate,
                               dividend_yield=dividend_yield, volatility=volatility,
                               first_right=first_right, maturity=maturity)
        self.spot, self.strike = ball(spot), ball(strike)
        self.rate, self.dividend = ball(rate), ball(dividend_yield)
        self.volatility, self.first, self.maturity = (
            ball(x) for x in (volatility, first_right, maturity))
        self.horizon = self.maturity-self.first
        self.first_sd = self.volatility*self.first.sqrt()
        self.future_sd = self.volatility*self.horizon.sqrt()
        self.mu = self.spot.log()+(self.rate-self.dividend-self.volatility**2/2)*self.first
        self.discount_first = (-self.rate*self.first).exp()
        self.discount_future = (-self.rate*self.horizon).exp()
        self.yield_future = (-self.dividend*self.horizon).exp()
        self.sqrt_two_pi = (2*arb.pi()).sqrt()
        # Jensen gives D(s)^+ <= (a*s-c)^+ above K.
        self.a = self.discount_first*(1-self.yield_future)
        self.c = self.discount_first*self.strike*(1-self.discount_future)
        for value in (self.first_sd, self.future_sd, self.discount_first, self.a):
            if not value.is_finite() or not value > 0:
                raise ArithmeticError('A required positive model quantity is not resolved')

    def exercise_advantage(self, state, *, analytic=False):
        """Smooth D(s) for s >= K, continued holomorphically by Acb."""
        log_ratio = (state/self.strike).log(analytic=analytic) if isinstance(state, acb) else (state/self.strike).log()
        d2 = (log_ratio+(self.rate-self.dividend-self.volatility**2/2)*self.horizon)/self.future_sd
        conditional = (state*self.yield_future*normal_cdf(d2+self.future_sd)
                       -self.strike*self.discount_future*normal_cdf(d2))
        return self.discount_first*(state-self.strike-conditional)

    def expiry_only_value(self):
        sd = self.volatility*self.maturity.sqrt()
        d2 = (self.spot.log()-self.strike.log()
              +(self.rate-self.dividend-self.volatility**2/2)*self.maturity)/sd
        return (self.spot*(-self.dividend*self.maturity).exp()*normal_cdf(d2+sd)
                -self.strike*(-self.rate*self.maturity).exp()*normal_cdf(d2))

    def tail_upper(self, cutoff):
        """Upper expectation of D(S_1)^+ 1{S_1 > cutoff}."""
        cutoff_ball = ball(cutoff)
        if not self.a*cutoff_ball-self.c >= 0:
            raise ArithmeticError('The cutoff must exceed the Jensen zero')
        z = (self.mu-cutoff_ball.log())/self.first_sd
        first_moment_tail = (self.mu+self.first_sd**2/2).exp()*normal_cdf(z+self.first_sd)
        tail = self.a*first_moment_tail-self.c*normal_cdf(z)
        lo, hi = endpoints(tail)
        if hi < 0:
            raise ArithmeticError('The analytic tail conflicts with non-negativity')
        return max(Fraction(0), hi)


class SignedLossIntegrand:
    def __init__(self, model, sign):
        self.model, self.sign = model, sign
        self.evaluations = 0
        self.nonfinite_evaluations = 0

    def __call__(self, state, analytic):
        self.evaluations += 1
        logarithm = state.log(analytic=analytic)
        if not logarithm.is_finite():
            self.nonfinite_evaluations += 1
            return acb('nan')
        z = (logarithm-self.model.mu)/self.model.first_sd
        density = (-z*z/2).exp()/(state*self.model.first_sd*self.model.sqrt_two_pi)
        result = self.sign*self.model.exercise_advantage(state, analytic=analytic)*density
        if not result.is_finite():
            self.nonfinite_evaluations += 1
        return result


def bracket_boundary(model, *, width, iteration_limit=512):
    """Strict certified signs, followed by rational bisection."""
    lower = model.parameters['strike']
    upper = 2*lower
    lower_value = model.exercise_advantage(ball(lower))
    if not lower_value < 0:
        raise ArithmeticError('The negative advantage at the strike is unresolved')
    expansions = 0
    upper_value = model.exercise_advantage(ball(upper))
    while not upper_value > 0:
        if not upper_value < 0:
            raise ArithmeticError('A boundary-bracketing sign is unresolved')
        expansions += 1
        if expansions >= iteration_limit:
            raise ArithmeticError('The exercise-boundary search exhausted its limit')
        upper *= 2
        upper_value = model.exercise_advantage(ball(upper))
    steps = 0
    while upper-lower > width:
        steps += 1
        if steps >= iteration_limit:
            raise ArithmeticError('The boundary bisection exhausted its limit')
        midpoint = (lower+upper)/2
        value = model.exercise_advantage(ball(midpoint))
        if value < 0:
            lower = midpoint
        elif value > 0:
            upper = midpoint
        else:
            raise ArithmeticError('The requested root width needs greater arithmetic precision')
    return lower, upper, dict(expansions=expansions, bisection_steps=steps,
                              lower_advantage=interval_record(*endpoints(model.exercise_advantage(ball(lower)))),
                              upper_advantage=interval_record(*endpoints(model.exercise_advantage(ball(upper)))))


def finite_integral(model, lower, upper, sign, *, absolute_tolerance,
                    relative_tolerance, evaluation_limit):
    if lower == upper:
        return Fraction(0), Fraction(0), dict(integrand_evaluations=0, nonfinite_integrand_evaluations=0)
    if lower < model.parameters['strike'] or upper < lower:
        raise ValueError('The smooth integration interval must lie above the strike')
    integrand = SignedLossIntegrand(model, sign)
    value = acb.integral(integrand, acb(ball(lower)), acb(ball(upper)),
                         abs_tol=ball(absolute_tolerance), rel_tol=ball(relative_tolerance),
                         eval_limit=evaluation_limit, use_heap=True)
    if not value.is_finite() or not value.imag.contains(0):
        raise ArithmeticError('The integral has no finite real enclosure')
    lo, hi = endpoints(value.real)
    if hi < 0:
        raise ArithmeticError('A non-negative loss integral has a negative upper endpoint')
    return max(Fraction(0), lo), hi, dict(
        integrand_evaluations=integrand.evaluations,
        nonfinite_integrand_evaluations=integrand.nonfinite_evaluations)


def threshold_gap_bound(model, threshold):
    """Enclose |D(theta)|, a pointwise bound for the exact-centred gap.

    With the exact terminal-call martingale, the realised policy gap is
    D(S_1)^+ - D(S_1) 1{S_1 > theta}. Since D is strictly increasing above K,
    this non-negative gap never exceeds |D(theta)|. No root search, simulation
    or loss integration is required to evaluate this deployment bound.
    """
    if threshold < model.parameters['strike']:
        raise ValueError('The threshold must be at or above the strike')
    lo, hi = endpoints(model.exercise_advantage(ball(threshold)))
    absolute_lo = min(abs(lo), abs(hi)) if lo*hi > 0 else Fraction(0)
    absolute_hi = max(abs(lo), abs(hi))
    return absolute_lo, absolute_hi


def classify(lo, hi, tolerance):
    if hi <= tolerance:
        return 'within_tolerance'
    if lo > tolerance:
        return 'exceeds_tolerance'
    return 'unresolved'


def certify(*, thresholds=(100, 110, 120, 130), tolerances=(.1, .25, .5),
            spot=100, strike=100, rate=.05, dividend_yield=.10, volatility=.20,
            first_right=1.5, maturity=3, cutoff=1000, precision=128,
            root_width=Fraction(1, 10**14), absolute_tolerance=1e-18,
            relative_tolerance=1e-18, evaluation_limit=100000):
    """Enclose threshold-policy losses and classify stated monetary tolerances.

    Accuracy requests control refinement, not validity. Reported rational
    intervals remain the certificate when their width exceeds a requested
    quadrature tolerance. Unresolved root signs and non-finite integration
    fail closed. Rates are non-negative, dividend yield strictly positive;
    zero volatility, thresholds below K and additional rights are unsupported.
    """
    wall_start, cpu_start = perf_counter(), process_time()
    if flint is None:
        raise RuntimeError('The separate python-flint environment is required')
    if type(precision) is not int or precision < 64:
        raise ValueError('At least 64 arithmetic precision bits are required')
    if type(evaluation_limit) is not int or evaluation_limit < 1:
        raise ValueError('A positive integer evaluation limit is required')
    parameters = {name: exact(value) for name, value in dict(
        spot=spot, strike=strike, rate=rate, dividend_yield=dividend_yield,
        volatility=volatility, first_right=first_right, maturity=maturity).items()}
    cutoff, root_width, absolute_tolerance, relative_tolerance = (
        exact(value) for value in (cutoff, root_width, absolute_tolerance, relative_tolerance))
    try:
        thresholds = tuple(exact(value) for value in thresholds)
        tolerances = tuple(exact(value) for value in tolerances)
    except TypeError as error:
        raise ValueError('Thresholds and tolerances must be finite scalar sequences') from error
    if min(parameters[name] for name in ('spot', 'strike', 'dividend_yield', 'volatility', 'first_right')) <= 0:
        raise ValueError('Positive spot, strike, dividend yield, volatility and first right are required')
    if parameters['rate'] < 0 or parameters['maturity'] <= parameters['first_right']:
        raise ValueError('A non-negative rate and ordered exercise rights are required')
    if not thresholds or any(value < parameters['strike'] for value in thresholds):
        raise ValueError('A non-empty threshold family at or above the strike is required')
    if len(set(thresholds)) != len(thresholds):
        raise ValueError('Thresholds must be distinct')
    if not tolerances or min(tolerances) <= 0 or len(set(tolerances)) != len(tolerances):
        raise ValueError('Distinct positive monetary tolerances are required')
    if min(root_width, absolute_tolerance, relative_tolerance) <= 0 or cutoff <= parameters['strike']:
        raise ValueError('Positive tolerances and a cutoff above the strike are required')
    with ctx.workprec(precision):
        model = TwoRightModel(**parameters)
        model_setup_work = dict(wall_seconds=perf_counter()-wall_start,
                               cpu_seconds=process_time()-cpu_start)
        b_lower, b_upper, boundary_work = bracket_boundary(model, width=root_width)
        if cutoff <= b_upper:
            raise ValueError('The tail cutoff must exceed the certified boundary bracket')
        # On the uncertain root strip, D vanishes at b and |D'|<=e^(-r*t).
        # The entire strip contribution is no larger than this positive bound.
        strip_ball = (model.discount_first*ball(b_upper-b_lower)**2
                      /(ball(b_lower)*model.first_sd*model.sqrt_two_pi))
        strip_upper = endpoints(strip_ball)[1]
        setup_work = dict(wall_seconds=perf_counter()-wall_start,
                          cpu_seconds=process_time()-cpu_start)
        policy_records = []
        quadrature_options = dict(absolute_tolerance=absolute_tolerance,
                                  relative_tolerance=relative_tolerance,
                                  evaluation_limit=evaluation_limit)
        for threshold in thresholds:
            policy_wall, policy_cpu = perf_counter(), process_time()
            bound_wall, bound_cpu = perf_counter(), process_time()
            gap_lower, gap_upper = threshold_gap_bound(model, threshold)
            gap_bound_work = dict(wall_seconds=perf_counter()-bound_wall,
                                  cpu_seconds=process_time()-bound_cpu)
            if threshold <= b_lower:
                lower, upper, sign = threshold, b_lower, -1
            elif threshold >= b_upper:
                lower, upper, sign = b_upper, threshold, 1
            else:
                lower, upper, sign = threshold, threshold, 1
            integral_wall, integral_cpu = perf_counter(), process_time()
            lo, hi, work = finite_integral(model, lower, upper, sign, **quadrature_options)
            work['wall_seconds'] = perf_counter()-integral_wall
            work['cpu_seconds'] = process_time()-integral_cpu
            loss_lo, loss_hi = lo, min(hi+strip_upper, gap_upper)
            if loss_lo > loss_hi:
                raise ArithmeticError('The integral and analytic gap bound disagree')
            policy_records.append(dict(
                policy='first_right_threshold', threshold=rational_record(threshold),
                exercise_instruction='Exercise at the first right iff S_1 > threshold; otherwise wait until maturity.',
                integrated_stock_interval=interval_record(lower, upper),
                integrand_sign=sign, finite_integral=interval_record(lo, hi),
                root_strip_allowance=interval_record(Fraction(0), strip_upper),
                tail_allowance=interval_record(Fraction(0), Fraction(0)),
                loss=interval_record(loss_lo, loss_hi),
                decisions=[dict(tolerance=rational_record(tolerance),
                                classification=classify(loss_lo, loss_hi, tolerance)) for tolerance in tolerances],
                deterministic_gap_upper=rational_record(gap_upper),
                analytic_gap_absolute_advantage=interval_record(gap_lower, gap_upper),
                analytic_bound_decisions=[dict(tolerance=rational_record(tolerance),
                                              classification='within_tolerance' if gap_upper <= tolerance else 'unresolved')
                                          for tolerance in tolerances],
                work=dict(standalone_analytic_bound_wall_seconds=model_setup_work['wall_seconds']+gap_bound_work['wall_seconds'],
                          standalone_analytic_bound_cpu_seconds=model_setup_work['cpu_seconds']+gap_bound_work['cpu_seconds'],
                          reference_body_wall_seconds=perf_counter()-policy_wall,
                          reference_body_cpu_seconds=process_time()-policy_cpu,
                          analytic_bound=gap_bound_work, quadrature=work)))
        wait_wall, wait_cpu = perf_counter(), process_time()
        integral_wall, integral_cpu = perf_counter(), process_time()
        lo, hi, work = finite_integral(model, b_upper, cutoff, 1, **quadrature_options)
        work['wall_seconds'] = perf_counter()-integral_wall
        work['cpu_seconds'] = process_time()-integral_cpu
        tail_upper = model.tail_upper(cutoff)
        wait_lo, wait_hi = lo, hi+strip_upper+tail_upper
        policy_records.append(dict(
            policy='always_wait', threshold=None,
            exercise_instruction='Wait until maturity on every path.',
            integrated_stock_interval=interval_record(b_upper, cutoff),
            integrand_sign=1, finite_integral=interval_record(lo, hi),
            root_strip_allowance=interval_record(Fraction(0), strip_upper),
            tail_allowance=interval_record(Fraction(0), tail_upper),
            loss=interval_record(wait_lo, wait_hi),
            decisions=[dict(tolerance=rational_record(tolerance),
                            classification=classify(wait_lo, wait_hi, tolerance)) for tolerance in tolerances],
            deterministic_gap_upper=None,
            analytic_gap_absolute_advantage=None,
            analytic_bound_decisions=[dict(tolerance=rational_record(tolerance), classification='unresolved')
                                      for tolerance in tolerances],
            work=dict(reference_body_wall_seconds=perf_counter()-wait_wall,
                      reference_body_cpu_seconds=process_time()-wait_cpu,
                      analytic_bound=None, quadrature=work)))
        expiry_lo, expiry_hi = endpoints(model.expiry_only_value())
        result = dict(
            schema_version=1, author='Amanjeet Singh',
            quantity='Expected discounted policy shortfall relative to the optimal two-right GBM call',
            ideal_model_population_loss_enclosed=True,
            realised_simulation_law_certified=False, statistical_coverage_certified=False,
            input_semantics='Float inputs denote their exact binary values; rational endpoints are authoritative.',
            model={name: rational_record(value) for name, value in parameters.items()},
            boundary=interval_record(b_lower, b_upper), boundary_work=boundary_work,
            shared_model_and_boundary_setup=setup_work,
            root_free_model_setup=model_setup_work,
            root_free_analytic_family_work=dict(
                wall_seconds=model_setup_work['wall_seconds']+sum(row['work']['analytic_bound']['wall_seconds']
                    for row in policy_records if row['work']['analytic_bound'] is not None),
                cpu_seconds=model_setup_work['cpu_seconds']+sum(row['work']['analytic_bound']['cpu_seconds']
                    for row in policy_records if row['work']['analytic_bound'] is not None)),
            expiry_only_value=interval_record(expiry_lo, expiry_hi),
            optimal_two_right_value=interval_record(expiry_lo+wait_lo, expiry_hi+wait_hi),
            policies=policy_records,
            arithmetic=dict(library='python-flint', version=flint.__version__, precision_bits=precision,
                            cutoff=rational_record(cutoff), root_width_requested=rational_record(root_width),
                            absolute_tolerance_requested=rational_record(absolute_tolerance),
                            relative_tolerance_requested=rational_record(relative_tolerance),
                            evaluation_limit=evaluation_limit,
                            branch_check='Acb log(analytic=analytic)'),
            work=dict(wall_seconds=perf_counter()-wall_start, cpu_seconds=process_time()-cpu_start,
                      cost_scope='Function validation and computation; process startup, imports and JSON serialisation are excluded.'),
            sources=[dict(authors='Black and Scholes', year=1973,
                          title='The Pricing of Options and Corporate Liabilities',
                          doi='10.1086/260062', role='Established conditional lognormal call formula'),
                     dict(authors='Andersen and Broadie', year=2004,
                          title='Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options',
                          doi='10.1287/mnsc.1040.0258', role='Published process parameters; separate two-right calendar'),
                     dict(authors='Johansson', year=2017,
                          title='Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic',
                          doi='10.1109/TC.2017.2690633', role='Established ball arithmetic software')])
    return result


def certify_analytic_bounds(*, thresholds=(100, 110, 120, 130), tolerances=(.1, .25, .5),
                            spot=100, strike=100, rate=.05, dividend_yield=.10,
                            volatility=.20, first_right=1.5, maturity=3, precision=128):
    """Certify root-free pointwise gap bounds and sufficient tolerance decisions.

    For each threshold the exact-centred gap lies in [0, |D(theta)|]. Its
    expected policy shortfall therefore lies in that interval as well. This
    is an upper-bound route: a bound above the tolerance yields an unresolved
    decision, never evidence that the policy's expected shortfall exceeds it.
    Only Arb formula evaluation is needed; no root search, finite integration,
    truncation choice or simulation is performed.
    """
    wall_start, cpu_start = perf_counter(), process_time()
    if flint is None:
        raise RuntimeError('The separate python-flint environment is required')
    if type(precision) is not int or precision < 64:
        raise ValueError('At least 64 arithmetic precision bits are required')
    parameters = {name: exact(value) for name, value in dict(
        spot=spot, strike=strike, rate=rate, dividend_yield=dividend_yield,
        volatility=volatility, first_right=first_right, maturity=maturity).items()}
    try:
        thresholds = tuple(exact(value) for value in thresholds)
        tolerances = tuple(exact(value) for value in tolerances)
    except TypeError as error:
        raise ValueError('Thresholds and tolerances must be finite scalar sequences') from error
    if min(parameters[name] for name in ('spot', 'strike', 'dividend_yield', 'volatility', 'first_right')) <= 0:
        raise ValueError('Positive spot, strike, dividend yield, volatility and first right are required')
    if parameters['rate'] < 0 or parameters['maturity'] <= parameters['first_right']:
        raise ValueError('A non-negative rate and ordered exercise rights are required')
    if not thresholds or any(value < parameters['strike'] for value in thresholds):
        raise ValueError('A non-empty threshold family at or above the strike is required')
    if len(set(thresholds)) != len(thresholds):
        raise ValueError('Thresholds must be distinct')
    if not tolerances or min(tolerances) <= 0 or len(set(tolerances)) != len(tolerances):
        raise ValueError('Distinct positive monetary tolerances are required')
    with ctx.workprec(precision):
        model = TwoRightModel(**parameters)
        setup_work = dict(wall_seconds=perf_counter()-wall_start,
                          cpu_seconds=process_time()-cpu_start)
        policies = []
        for threshold in thresholds:
            body_wall, body_cpu = perf_counter(), process_time()
            lower, upper = threshold_gap_bound(model, threshold)
            row = dict(
                policy='first_right_threshold', threshold=rational_record(threshold),
                exercise_instruction='Exercise at the first right iff S_1 > threshold; otherwise wait until maturity.',
                analytic_gap_absolute_advantage=interval_record(lower, upper),
                deterministic_gap_upper=rational_record(upper),
                loss=interval_record(Fraction(0), upper),
                decisions=[dict(tolerance=rational_record(tolerance),
                                classification='within_tolerance' if upper <= tolerance else 'unresolved')
                           for tolerance in tolerances])
            row['work'] = dict(wall_seconds=perf_counter()-body_wall,
                               cpu_seconds=process_time()-body_cpu)
            policies.append(row)
        result = dict(
            schema_version=1, author='Amanjeet Singh',
            quantity='Root-free deterministic upper bound on the exact-centred threshold-policy gap and its expected shortfall',
            ideal_model_population_loss_enclosed=True, population_loss_upper_only=True,
            boundary_search_executed=False, numerical_quadrature_executed=False,
            simulation_executed=False,
            realised_simulation_law_certified=False, statistical_coverage_certified=False,
            input_semantics='Float inputs denote their exact binary values; rational endpoints are authoritative.',
            model={name: rational_record(value) for name, value in parameters.items()},
            model_and_input_setup=setup_work, policies=policies,
            arithmetic=dict(library='python-flint', version=flint.__version__, precision_bits=precision),
            sources=[dict(authors='Black and Scholes', year=1973,
                          title='The Pricing of Options and Corporate Liabilities',
                          doi='10.1086/260062', role='Established conditional lognormal call formula'),
                     dict(authors='Andersen and Broadie', year=2004,
                          title='Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options',
                          doi='10.1287/mnsc.1040.0258', role='Published process parameters; separate two-right calendar'),
                     dict(authors='Johansson', year=2017,
                          title='Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval Arithmetic',
                          doi='10.1109/TC.2017.2690633', role='Established ball arithmetic software')])
    result['work'] = dict(wall_seconds=perf_counter()-wall_start,
                          cpu_seconds=process_time()-cpu_start,
                          cost_scope='Complete function validation, model construction, bound evaluation and result construction; process startup, imports and JSON serialisation are excluded.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analytic-only', action='store_true',
                        help='Evaluate sufficient analytic upper bounds without root search or quadrature')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[100, 110, 120, 130])
    parser.add_argument('--tolerances', type=float, nargs='+', default=[.1, .25, .5])
    parser.add_argument('--precision', type=int, default=128)
    parser.add_argument('--cutoff', type=float, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retained references must not be overwritten')
    if args.analytic_only:
        result = certify_analytic_bounds(thresholds=args.thresholds, tolerances=args.tolerances,
                                         precision=args.precision)
    else:
        result = certify(thresholds=args.thresholds, tolerances=args.tolerances,
                         precision=args.precision, cutoff=args.cutoff)
    result['created_at'] = datetime.now(timezone.utc).isoformat()
    result['environment'] = dict(python=platform.python_version(), python_flint=flint.__version__)
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for policy in result['policies']:
        loss = policy['loss']
        threshold = None if policy['threshold'] is None else policy['threshold']['approximation']
        print(policy['policy'], threshold, loss['lower']['approximation'], loss['upper']['approximation'])


if __name__ == '__main__':
    main()
