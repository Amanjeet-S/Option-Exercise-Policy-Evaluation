"""Outward clipping tails and interval empirical-Bernstein constructions.

I keep the zero-martingale population envelope separate from the existing
European-value allowance. Exact rational sufficient statistics retain the
uncertainty of every clipped gain interval. The variance upper bound follows
from the sample-centering projection and Minkowski's inequality; the sampling
radius uses Maurer and Pontil (2009), Theorem 4. These are established results.

The probability assertion requires fresh ideal iid observations conditional
on fixed information and a valid deterministic tail allowance. Numerical
construction certificates do not certify an ordinary floating-point sampler.
"""
from collections.abc import Mapping, Sequence
from fractions import Fraction
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys
import time

try:
    import flint
    from flint import arb, ctx, fmpq
except ImportError:
    flint = arb = ctx = fmpq = None

DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]


def exact(value, name='Parameter'):
    if isinstance(value, bool) or type(value) not in (int, float, Fraction):
        raise ValueError(name + ' must be an exact rational or finite binary float')
    if type(value) is float and not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return Fraction(value)


def ball(value):
    if flint is None:
        raise RuntimeError('Outward arithmetic requires the separate python-flint environment')
    value = exact(value)
    return arb(fmpq(value.numerator, value.denominator))


def endpoints(value):
    if not value.is_finite():
        raise ArithmeticError('A non-finite ball cannot supply a numerical certificate')
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
    return dict(numerator=str(value.numerator), denominator=str(value.denominator),
                approximation=approximation)


def interval_record(lower, upper):
    lower, upper = exact(lower), exact(upper)
    if lower > upper:
        raise ArithmeticError('A numerical certificate cannot have reversed endpoints')
    return dict(lower=rational_record(lower), upper=rational_record(upper),
                width=rational_record(upper-lower))


def decode_rational(record):
    if (not isinstance(record, Mapping)
            or set(record) != {'numerator', 'denominator', 'approximation'}
            or not isinstance(record['numerator'], str)
            or not isinstance(record['denominator'], str)):
        raise ValueError('An endpoint requires an explicit exact rational record')
    try:
        numerator, denominator = int(record['numerator']), int(record['denominator'])
    except (TypeError, ValueError) as exc:
        raise ValueError('An endpoint requires integer numerator and denominator strings') from exc
    if denominator <= 0:
        raise ValueError('An endpoint denominator must be positive')
    return Fraction(numerator, denominator)


def decode_interval(record):
    if not isinstance(record, Mapping) or set(record) != {'lower', 'upper', 'width'}:
        raise ValueError('A certificate requires explicit endpoints and exact width')
    lower, upper = decode_rational(record['lower']), decode_rational(record['upper'])
    if lower > upper or decode_rational(record['width']) != upper-lower:
        raise ValueError('Certificate endpoints and width must be ordered and consistent')
    return lower, upper


def _require_precision(precision):
    if flint is None:
        raise RuntimeError('Outward arithmetic requires the separate python-flint environment')
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')


def _load_marginal_module(repository):
    source = Path(repository).resolve() / 'scripts/certify_european_gain_tail.py'
    spec = importlib.util.spec_from_file_location('financial_decision_marginal_tail', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('The existing marginal tail source could not be loaded')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def build_zero_gain_tail_bound(spots, exercise_times, *, strike=100, rate=.05,
        dividend=.10, volatility=.20, cap=150, precision=96, correlation=0,
        repository=DEFAULT_REPOSITORY):
    """Bound E[(G_zero-cap)+] without drawing or evaluating assessment paths.

    For a stopping policy on positive dates, G_zero<=max_j H_j. A union of
    marginal call tails therefore suffices. For each dated term I use an
    outward upper discount and an outward lower strike threshold. The bound
    does not require independence among dates or assets. The supported model
    interface nevertheless retains the study's homogeneous independent GBM
    domain, with non-negative rates, yields and volatility.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    _require_precision(precision)
    if not isinstance(spots, Sequence) or isinstance(spots, (str, bytes)):
        raise ValueError('Spots require a non-empty indexed sequence')
    spots = tuple(exact(value, 'Spot') for value in spots)
    if not spots or any(value <= 0 for value in spots):
        raise ValueError('Every initial spot must be strictly positive')
    if not isinstance(exercise_times, Sequence) or isinstance(exercise_times, (str, bytes)):
        raise ValueError('The full calendar requires an indexed sequence')
    times = tuple(exact(value, 'Exercise date') for value in exercise_times)
    if (len(times) < 2 or times[0] != 0
            or any(right <= left for left, right in zip(times, times[1:]))):
        raise ValueError('The full calendar must increase strictly from zero to expiry')
    strike, rate, dividend, volatility, cap, correlation = (
        exact(value) for value in (strike, rate, dividend, volatility, cap, correlation))
    if strike <= 0 or rate < 0 or dividend < 0 or volatility < 0 or cap < 0:
        raise ValueError('Positive strike and non-negative rates, yield, volatility and cap are required')
    if correlation != 0:
        raise ValueError('The declared study interface supports independent coordinates only')
    rows, total_upper, calls = [], Fraction(0), 0
    sole_terminal_right = len(times) == 2
    marginal = None if sole_terminal_right else _load_marginal_module(repository)
    with ctx.workprec(precision):
        if not sole_terminal_right:
            for index, instant in enumerate(times[1:], 1):
                discount = endpoints((-ball(rate)*ball(instant)).exp())
                if discount[0] < 0 or discount[1] <= 0:
                    raise ArithmeticError('A positive discount factor was not certified')
                if cap == 0:
                    threshold = (strike, strike)
                else:
                    threshold = endpoints(ball(strike)+ball(cap)*(ball(rate)*ball(instant)).exp())
                    threshold = max(strike, threshold[0]), threshold[1]
                if threshold[1] < threshold[0]:
                    raise ArithmeticError('The clipping threshold conflicts with its strike lower bound')
                marginal_calls = [marginal.lognormal_call_moment(spot, instant, threshold[0],
                    rate=rate, dividend=dividend, volatility=volatility, precision=precision)
                    for spot in spots]
                calls += len(marginal_calls)
                row_upper = discount[1]*sum(upper for lower, upper in marginal_calls)
                total_upper += row_upper
                rows.append(dict(exercise_index=index, time=rational_record(instant),
                    discount=interval_record(*discount), threshold=interval_record(*threshold),
                    threshold_lower_used=rational_record(threshold[0]),
                    marginal_calls=[interval_record(*pair) for pair in marginal_calls],
                    tail_upper=rational_record(row_upper)))
    return dict(schema_version=1, author='Amanjeet Singh',
        quantity='Upper bound for E[(G_zero-B)+] on the retained positive exercise calendar',
        model=dict(spots=[rational_record(value) for value in spots],
            strike=rational_record(strike), rate=rational_record(rate),
            dividend=rational_record(dividend), volatility=rational_record(volatility),
            correlation=rational_record(correlation),
            exercise_times=[rational_record(value) for value in times]),
        cap=rational_record(cap), precision_bits=precision,
        python_flint_version=flint.__version__, rows=rows,
        sole_terminal_right_exact_zero=sole_terminal_right,
        clipping_tail_allowance=interval_record(Fraction(0), total_upper),
        work=dict(marginal_call_evaluations=calls, conditional_integral_calls=0,
            builder_cpu_seconds=time.process_time()-cpu,
            builder_wall_seconds=time.perf_counter()-wall),
        ideal_model_tail_bound_certified=True, paths_generated=False,
        sampling_coverage_certified=False, financial_decision_certified=False,
        scope='Population clipping allowance for the zero-martingale gain. The terminal date is retained. No result is asserted for European, stock, hinge or polynomial gains. Exact rational endpoints are authoritative; this calculation does not certify floating-point path generation.')


def evaluate_zero_gain(states, exercise_times, stopping_index, *, strike=100, rate=.05, precision=96):
    """Enclose a retained zero-martingale gain at exact supplied states.

    The selected candidate cancels exactly. A supplied index must arise from
    a separately checked adapted stopping rule; range validation alone does
    not establish admissibility under the probability model.
    """
    cpu, wall = time.process_time(), time.perf_counter()
    _require_precision(precision)
    if not isinstance(exercise_times, Sequence) or isinstance(exercise_times, (str, bytes)):
        raise ValueError('The full calendar requires an indexed sequence')
    times = tuple(exact(value, 'Exercise date') for value in exercise_times)
    if (len(times)<2 or times[0]!=0
            or any(right<=left for left,right in zip(times,times[1:]))):
        raise ValueError('The full calendar must increase strictly from zero to expiry')
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or len(states)!=len(times):
        raise ValueError('The path must contain one state vector for every calendar date')
    vectors=[]
    for state in states:
        if not isinstance(state, Sequence) or isinstance(state, (str, bytes)):
            raise ValueError('Each state must be an indexed positive vector')
        vector=tuple(exact(value,'State') for value in state)
        if not vector or any(value<=0 for value in vector):
            raise ValueError('Every state coordinate must be strictly positive')
        if vectors and len(vector)!=len(vectors[0]):
            raise ValueError('The state dimension must be constant over the calendar')
        vectors.append(vector)
    J=len(times)-1
    if type(stopping_index) is not int or not 1<=stopping_index<=J:
        raise ValueError('The retained stopping index must be an integer from one to expiry')
    strike,rate=exact(strike,'Strike'),exact(rate,'Rate')
    if strike<=0 or rate<0:
        raise ValueError('A positive strike and non-negative rate are required')
    rewards=[]
    with ctx.workprec(precision):
        for instant,vector in zip(times,vectors):
            payoff=max(Fraction(0),max(vector)-strike)
            if payoff==0 or rate==0 or instant==0:
                reward=(payoff,payoff)
            else:
                reward=endpoints(ball(payoff)*(-ball(rate)*ball(instant)).exp())
                reward=(max(Fraction(0),reward[0]),reward[1])
            rewards.append(reward)
    selected=rewards[stopping_index]
    candidates={str(j): (Fraction(0),Fraction(0)) if j==stopping_index else
                (rewards[j][0]-selected[1],rewards[j][1]-selected[0])
                for j in range(1,J+1)}
    gain=(max(Fraction(0),max(pair[0] for pair in candidates.values())),
          max(Fraction(0),max(pair[1] for pair in candidates.values())))
    digest=hashlib.sha256(json.dumps([[[str(x.numerator),str(x.denominator)] for x in vector]
        for vector in vectors],separators=(',',':')).encode()).hexdigest()
    return dict(schema_version=1,author='Amanjeet Singh',arm='zero_martingale',
        stopping_index=stopping_index,rights=J+1,dimension=len(vectors[0]),
        precision_bits=precision,state_sha256=digest,
        reward_certificates=[interval_record(*pair) for pair in rewards],
        candidate_certificates={j:interval_record(*pair) for j,pair in candidates.items()},
        gain_certificate=interval_record(*gain),completed=True,error=None,
        work=dict(conditional_integral_calls=0,marginal_call_evaluations=0,
                  reward_exponentials_evaluated=sum(rate!=0 and instant!=0 and max(vector)>strike
                      for instant,vector in zip(times,vectors))),
        cost=dict(assessment_cpu_seconds=time.process_time()-cpu,
                  assessment_wall_seconds=time.perf_counter()-wall),
        admissibility_input_premise='The supplied stopping index is produced by the frozen adapted policy on this retained calendar.',
        sampling_law_certified=False,
        scope='Ideal zero-martingale gain at exact supplied binary states; every positive exercise date is retained and the selected candidate is exactly zero.')


class IntervalClippedAccumulator:
    """Exact streaming statistics for uncertain non-negative gain observations.

    If Y_i belongs to clipped [l_i,u_i], its sample standard deviation is at
    most s_midpoint + sqrt(sum radius_i**2/(n-1)). Both the mean and the
    variance uncertainty are retained before the sampling inequality is used.
    """
    def __init__(self, cap):
        self.cap = exact(cap, 'Clipping cap')
        if self.cap < 0:
            raise ValueError('The clipping cap must be non-negative')
        self.count = 0
        self.sum_lower = Fraction(0)
        self.sum_upper = Fraction(0)
        self.sum_midpoint = Fraction(0)
        self.sum_midpoint_squared = Fraction(0)
        self.sum_radius_squared = Fraction(0)
        self.maximum_raw_width = Fraction(0)
        self.maximum_clipped_width = Fraction(0)

    def add_interval(self, lower, upper):
        lower, upper = exact(lower, 'Gain lower endpoint'), exact(upper, 'Gain upper endpoint')
        if lower > upper:
            raise ValueError('Gain endpoints must be ordered')
        if upper < 0:
            raise ArithmeticError('The gain certificate conflicts with non-negativity')
        clipped_lower = min(self.cap, max(Fraction(0), lower))
        clipped_upper = min(self.cap, max(Fraction(0), upper))
        midpoint = (clipped_lower+clipped_upper)/2
        radius = (clipped_upper-clipped_lower)/2
        self.count += 1
        self.sum_lower += clipped_lower
        self.sum_upper += clipped_upper
        self.sum_midpoint += midpoint
        self.sum_midpoint_squared += midpoint*midpoint
        self.sum_radius_squared += radius*radius
        self.maximum_raw_width = max(self.maximum_raw_width, upper-lower)
        self.maximum_clipped_width = max(self.maximum_clipped_width, 2*radius)

    def add_certificate(self, record):
        self.add_interval(*decode_interval(record))

    def snapshot(self):
        return dict(schema_version=1, author='Amanjeet Singh', count=self.count,
            cap=rational_record(self.cap),
            sum_lower=rational_record(self.sum_lower),
            sum_upper=rational_record(self.sum_upper),
            sum_midpoint=rational_record(self.sum_midpoint),
            sum_midpoint_squared=rational_record(self.sum_midpoint_squared),
            sum_radius_squared=rational_record(self.sum_radius_squared),
            maximum_raw_width=rational_record(self.maximum_raw_width),
            maximum_clipped_width=rational_record(self.maximum_clipped_width))

    def upper_limit(self, *, delta, tail_upper, precision=96):
        """Enclose a conditional upper-limit construction, given its premises.

        The returned authoritative upper_limit is the rational upper endpoint
        of the constructed expression, not a claim of certain coverage.
        The probability guarantee is conditional on valid observation and
        population-tail enclosures and fresh ideal iid observations. Distinct
        ideal quantities and fixed looks require an explicit union allocation.
        """
        cpu, wall = time.process_time(), time.perf_counter()
        _require_precision(precision)
        delta, tail_upper = exact(delta, 'Failure probability'), exact(tail_upper, 'Tail allowance')
        if not 0 < delta < 1:
            raise ValueError('The failure probability must lie strictly between zero and one')
        if tail_upper < 0:
            raise ValueError('The population tail allowance must be non-negative')
        base = dict(schema_version=1, author='Amanjeet Singh',
            construction='Clipped empirical Bernstein with midpoint-radius variance inflation',
            statistics=self.snapshot(), failure_probability=rational_record(delta),
            population_tail_upper=rational_record(tail_upper), precision_bits=precision,
            python_flint_version=flint.__version__,
            tail_input_premise='The supplied allowance dominates E[(G-B)+] under the fixed conditional ideal model.',
            observation_input_premise='Every clipped interval encloses its ideal gain observation.',
            sampling_input_premise='Fresh iid ideal observations conditional on the frozen information, including any cap-selection pilot.',
            confidence_inequality_source='Maurer and Pontil (2009), Theorem 4',
            sampling_law_certified=False, financial_decision_certified=False)
        if self.cap == 0:
            base.update(empirical_bernstein_applied=False,
                deterministic_zero_cap=True, mean_upper=rational_record(0),
                variance_upper=rational_record(0),
                upper_limit_certificate=interval_record(tail_upper, tail_upper),
                upper_limit=rational_record(tail_upper))
        else:
            if self.count < 2:
                raise ValueError('At least two observations are required for empirical Bernstein')
            n = self.count
            midpoint_variance = (self.sum_midpoint_squared-self.sum_midpoint**2/n)/(n-1)
            radius_squared = self.sum_radius_squared/(n-1)
            if midpoint_variance < 0 or radius_squared < 0:
                raise ArithmeticError('Exact sufficient statistics conflict with variance non-negativity')
            mean_upper = self.sum_upper/n
            with ctx.workprec(precision):
                variance_ball = (ball(midpoint_variance).sqrt()+ball(radius_squared).sqrt())**2
                variance_bounds = endpoints(variance_ball)
                variance_bounds = max(Fraction(0), variance_bounds[0]), variance_bounds[1]
                if variance_bounds[1] < 0:
                    raise ArithmeticError('The sample-variance bound conflicts with non-negativity')
                log_ball = ball(2/delta).log()
                log_bounds = endpoints(log_ball)
                if log_bounds[0] <= 0:
                    raise ArithmeticError('The positive confidence logarithm was not certified')
                radius_ball = (ball(2*variance_bounds[1]/n)*log_ball).sqrt() + \
                    ball(7*self.cap/Fraction(3*(n-1)))*log_ball
                radius_bounds = endpoints(radius_ball)
                radius_bounds = max(Fraction(0), radius_bounds[0]), radius_bounds[1]
                upper_bounds = endpoints(ball(mean_upper)+radius_ball+ball(tail_upper))
            base.update(empirical_bernstein_applied=True, deterministic_zero_cap=False,
                sample_midpoint_variance=rational_record(midpoint_variance),
                squared_radius_budget=rational_record(radius_squared),
                mean_upper=rational_record(mean_upper),
                variance_bound_certificate=interval_record(*variance_bounds),
                variance_upper=rational_record(variance_bounds[1]),
                log_two_over_delta_certificate=interval_record(*log_bounds),
                sampling_radius_certificate=interval_record(*radius_bounds),
                upper_limit_certificate=interval_record(*upper_bounds),
                upper_limit=rational_record(upper_bounds[1]))
        base['work'] = dict(bound_cpu_seconds=time.process_time()-cpu,
                            bound_wall_seconds=time.perf_counter()-wall)
        return base
