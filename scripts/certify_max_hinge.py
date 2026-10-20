"""Enclose an independent-GBM conditional max-hinge expectation.

This is a numerical component, with no claim of theoretical novelty. For
positive states and homogeneous positive volatility, conditional independence
gives P(max_i X_i <= z) = product_i Phi((log(z/x_i)-m)/s), where
m=(r-q-sigma**2/2)*dt and s=sigma*sqrt(dt). Integrating its survival function
from K to L gives the finite part of E[(max_i X_i-K)+]. The remaining tail is
non-negative and at most sum_i E[(X_i-L)+], by the pointwise inequality
(max_i X_i-L)+ <= sum_i (X_i-L)+. Each summand has the standard lognormal
call expectation. The product formula requires independent Brownian shocks.

All float inputs mean their exact binary values. Arb/Acb enclose special
functions, the finite integral and arithmetic. The logarithm forwards Acb's
analytic flag, as required by the official integration API. Non-finite results
fail closed. The tail allowance and finite integration width are reported
separately; neither is a statistical confidence radius. No claim is made about
path simulation, regression, martingale gains or confidence coverage.

The demonstration parameters follow Andersen and Broadie (2004), Management
Science 50(9), Table 2: five independent assets, K=100, r=.05, q=.10,
sigma=.20, T=3 and nine equal time intervals. The conditional step here is
binary64 1/3.
Free author PDF: https://business.columbia.edu/sites/default/files-efs/pubfiles/1234/1234.pdf
API: https://python-flint.readthedocs.io/en/latest/acb.html#flint.acb.integral
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
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
    """Convert finite binary64, integer or Fraction input to an exact rational."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('A finite scalar float, integer or Fraction is required')
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError('Inputs must be finite')
    return Fraction.from_float(value)


def ball(value):
    value = exact(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoints(value):
    """Exact rational outward endpoints, extracted at the current precision."""
    if not value.is_finite():
        raise ArithmeticError('A non-finite ball cannot certify an expectation')
    lo, hi = value.lower().fmpq(), value.upper().fmpq()
    return (Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q)))


def rational_record(value):
    """The rational is authoritative; approximation is only for display."""
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
        raise ArithmeticError('An empty enclosure cannot certify an expectation')
    return dict(lower=rational_record(lo), upper=rational_record(hi),
                width=rational_record(hi-lo))


def normal_cdf(value):
    return (-value/arb(2).sqrt()).erfc()/2


class SurvivalIntegrand:
    """Holomorphic extension with branch-domain checking delegated to Acb."""

    def __init__(self, log_states, mean, sd):
        self.log_states, self.mean, self.sd = log_states, mean, sd
        self.sqrt_two = arb(2).sqrt()
        self.evaluations = 0
        self.nonfinite_evaluations = 0

    def __call__(self, z, analytic):
        self.evaluations += 1
        log_z = z.log(analytic=analytic)
        if not log_z.is_finite():
            self.nonfinite_evaluations += 1
            return acb('nan')
        product = acb(1)
        for log_state in self.log_states:
            standardised = (log_z-log_state-self.mean)/self.sd
            product *= (-standardised/self.sqrt_two).erfc()/2
        result = 1-product
        if not result.is_finite():
            self.nonfinite_evaluations += 1
        return result


def certify(spots, *, strike=100, rate=.05, dividend=.10, volatility=.20,
            step=1/3, cutoff=200, precision=128, absolute_tolerance=1e-12,
            relative_tolerance=1e-12, correlation=0, evaluation_limit=100000):
    """Return an exact-endpoint certificate for one conditional expectation.

    Only homogeneous scalar drift/volatility and independent shocks are
    supported. States, strike, step and volatility must be strictly positive,
    and cutoff must exceed strike. Rates may be any finite scalar values.
    A returned finite enclosure remains valid if a requested accuracy is not
    reached; its actual width is the evidence. Missing flint is an error, never
    a fallback to an uncertified numerical estimate.
    """
    if flint is None:
        raise RuntimeError('Certification requires the optional python-flint dependency')
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')
    if type(evaluation_limit) is not int or evaluation_limit < 1:
        raise ValueError('A positive integer evaluation limit is required')
    try:
        states = tuple(exact(x) for x in spots)
    except TypeError as error:
        raise ValueError('A non-empty sequence of positive states is required') from error
    if not states or any(x <= 0 for x in states):
        raise ValueError('A non-empty sequence of strictly positive states is required')
    strike, rate, dividend, volatility, step, cutoff, absolute_tolerance, relative_tolerance, correlation = (
        exact(x) for x in (strike, rate, dividend, volatility, step, cutoff,
                          absolute_tolerance, relative_tolerance, correlation))
    if strike <= 0 or volatility <= 0 or step <= 0 or cutoff <= strike:
        raise ValueError('Positive strike, scalar volatility, step and cutoff > strike required')
    if correlation != 0:
        raise ValueError('Only independent shocks (correlation=0) are supported')
    if absolute_tolerance <= 0 or relative_tolerance <= 0:
        raise ValueError('Strictly positive integration tolerances are required')
    wall_start, cpu_start = perf_counter(), process_time()
    with ctx.workprec(precision):
        sd = ball(volatility)*ball(step).sqrt()
        mean = (ball(rate)-ball(dividend)-ball(volatility)**2/2)*ball(step)
        if not sd.is_finite() or not sd > 0 or not mean.is_finite():
            raise ArithmeticError('The transition parameters are not resolved and finite')
        log_states = tuple(ball(x).log() for x in states)
        if any(not x.is_finite() for x in log_states):
            raise ArithmeticError('The state logarithms are not finite')
        integrand = SurvivalIntegrand(log_states, mean, sd)
        finite = acb.integral(
            integrand, acb(ball(strike)), acb(ball(cutoff)),
            abs_tol=ball(absolute_tolerance), rel_tol=ball(relative_tolerance),
            eval_limit=evaluation_limit, use_heap=True)
        if not finite.is_finite() or not finite.imag.contains(0):
            raise ArithmeticError('The finite integral has no valid real enclosure')
        finite_lo, finite_hi = endpoints(finite.real)
        # The real survival function is between zero and one on [K,L].
        finite_lo = max(Fraction(0), finite_lo)
        finite_hi = min(cutoff-strike, finite_hi)
        individual_tails = []
        tail_upper = Fraction(0)
        for state, log_state in zip(states, log_states):
            d2 = (log_state-ball(cutoff).log()+mean)/sd
            call = (ball(state)*(ball(rate-dividend)*ball(step)).exp()
                    * normal_cdf(d2+sd)-ball(cutoff)*normal_cdf(d2))
            lo, hi = endpoints(call)
            if hi < 0:
                raise ArithmeticError('The analytic call tail conflicts with non-negativity')
            individual_tails.append(interval_record(max(Fraction(0), lo), hi))
            tail_upper += hi
        result = dict(
            schema_version=1,
            quantity='undiscounted conditional expectation of (max_i X_i - K)+',
            ideal_model_expectation_enclosed=True,
            realised_path_gains_certified=False,
            statistical_coverage_certified=False,
            input_semantics='Float inputs are exact binary64 values; rational endpoints are authoritative.',
            model=dict(spots=[rational_record(x) for x in states],
                       strike=rational_record(strike), rate=rational_record(rate),
                       dividend=rational_record(dividend), volatility=rational_record(volatility),
                       step=rational_record(step), cutoff=rational_record(cutoff),
                       correlation=rational_record(correlation), dimension=len(states)),
            arithmetic=dict(library='python-flint', version=flint.__version__,
                            precision_bits=precision, branch_check='Acb log(analytic=analytic)',
                            absolute_tolerance_requested=rational_record(absolute_tolerance),
                            relative_tolerance_requested=rational_record(relative_tolerance),
                            evaluation_limit=evaluation_limit),
            finite_integral=interval_record(finite_lo, finite_hi),
            tail_allowance=interval_record(Fraction(0), tail_upper),
            individual_call_tails=individual_tails,
            expectation=interval_record(finite_lo, finite_hi+tail_upper),
            work=dict(integrand_evaluations=integrand.evaluations,
                      nonfinite_integrand_evaluations=integrand.nonfinite_evaluations,
                      wall_seconds=perf_counter()-wall_start,
                      cpu_seconds=process_time()-cpu_start))
    return result


def demonstration():
    """Small conditional-state study, not a Bermudan price or policy bound."""
    rows = []
    for state in (90, 100, 110):
        for cutoff in (150, 200, 250):
            for precision, tolerance in ((64, 1e-8), (128, 1e-14)):
                rows.append(certify([state]*5, cutoff=cutoff, precision=precision,
                                    absolute_tolerance=tolerance,
                                    relative_tolerance=tolerance))
    return dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        scope='Numerical conditional max-hinge component only; no training, pricing or coverage study.',
        attribution='Andersen and Broadie (2004), Table 2; one-step conditional states are this study.',
        source_url='https://business.columbia.edu/sites/default/files-efs/pubfiles/1234/1234.pdf',
        environment=dict(python=platform.python_version(), python_flint=flint.__version__),
        runs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spots', type=float, nargs='+', default=[100]*5)
    parser.add_argument('--strike', type=float, default=100)
    parser.add_argument('--rate', type=float, default=.05)
    parser.add_argument('--dividend', type=float, default=.10)
    parser.add_argument('--volatility', type=float, default=.20)
    parser.add_argument('--step', type=float, default=1/3)
    parser.add_argument('--cutoff', type=float, default=200)
    parser.add_argument('--precision', type=int, default=128)
    parser.add_argument('--absolute-tolerance', type=float, default=1e-12)
    parser.add_argument('--relative-tolerance', type=float, default=1e-12)
    parser.add_argument('--evaluation-limit', type=int, default=100000)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.demo:
        result = demonstration()
    else:
        result = certify(args.spots, strike=args.strike, rate=args.rate,
                         dividend=args.dividend, volatility=args.volatility,
                         step=args.step, cutoff=args.cutoff, precision=args.precision,
                         absolute_tolerance=args.absolute_tolerance,
                         relative_tolerance=args.relative_tolerance,
                         evaluation_limit=args.evaluation_limit)
    content = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
