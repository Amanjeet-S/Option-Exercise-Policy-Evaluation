"""Cost of reaching a monetary loss tolerance in a public two-right model.

The reference and analytical bound are strong deployment comparators. Three
Monte Carlo assessors use valid martingales, independently frozen setup and
clipped empirical Bernstein limits. A Bonferroni allocation covers all looks,
policies and assessors. Model certification and numerical simulation are kept
separate; no market observations or claim of methodological priority is made.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qf_research.optimal_stopping.contracts import positive_integer
from validate_gap_coverage import GapProcess, conditional_terminal_reward, first_reward, wilson_interval
from validate_clipped_gap_coverage import analytic_tail_allowance, undiscounted_call

METHODS = ('zero', 'stock', 'terminal_call')
POLICIES = (100., 110., 120., 130., None)
COUNTS = (128, 512, 2048, 8192, 32768)
TOLERANCES = (.10, .25, .50)
CAPS = (.05, .1, .2, .5, 1., 2., 4., 8., 12., 20., 40., 80., 160., 320.)


def clocks():
    return time.process_time(), time.perf_counter()


def cost(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def label(threshold):
    return 'always_wait' if threshold is None else f'threshold_{threshold:g}'


def parameter_check(process, threshold, coefficient):
    if threshold is not None and (not math.isfinite(threshold) or threshold < process.strike):
        raise ValueError('A finite threshold at or above the strike is required')
    maximum = math.exp(-process.dividend_yield * process.maturity)
    if not math.isfinite(coefficient) or not 0 <= coefficient <= maximum:
        raise ValueError('The stock coefficient must lie in [0, exp(-q*T)]')


def simulate(first_rng, terminal_rng, count, process, *, terminal):
    drift = process.rate-process.dividend_yield-.5*process.volatility**2
    first = process.spot*np.exp(drift*process.first_right +
        process.volatility*math.sqrt(process.first_right)*first_rng.standard_normal(count))
    if not terminal:
        return first, None
    horizon = process.maturity-process.first_right
    last = first*np.exp(drift*horizon+process.volatility*math.sqrt(horizon)*terminal_rng.standard_normal(count))
    return first, last


def gains(first, terminal, process, threshold, method, coefficient=0.):
    parameter_check(process, threshold, coefficient)
    first = np.asarray(first, dtype=float)
    if first.ndim != 1 or np.any(~np.isfinite(first)) or np.any(first <= 0):
        raise ValueError('Finite positive first-right states are required')
    exercise = np.zeros(first.shape, dtype=bool) if threshold is None else first > threshold
    first_payoff = first_reward(first, process)
    if method == 'terminal_call':
        advantage = first_payoff-conditional_terminal_reward(first, process)
        if process.dividend_yield == 0:
            advantage = np.minimum(advantage, 0.)
    elif method in ('zero', 'stock'):
        terminal = np.asarray(terminal, dtype=float)
        if terminal.shape != first.shape or np.any(~np.isfinite(terminal)) or np.any(terminal <= 0):
            raise ValueError('Matching finite positive terminal states are required')
        advantage = first_payoff-math.exp(-process.rate*process.maturity)*np.maximum(terminal-process.strike, 0.)
        if method == 'stock':
            carry1 = first*math.exp(-(process.rate-process.dividend_yield)*process.first_right)
            carry2 = terminal*math.exp(-(process.rate-process.dividend_yield)*process.maturity)
            advantage = advantage+coefficient*(carry2-carry1)
    else:
        raise ValueError('Unknown assessor')
    return np.maximum(np.where(exercise, -advantage, advantage), 0.)


def stock_envelope(process, threshold, coefficient):
    """Pointwise envelope from the two affine branches of the terminal reward.

    For finite theta the waiting branch is bounded by W; the exercising branch
    is bounded by a constant and a positive affine function of S_T. For always
    wait, min f_T=-k*c_T*K gives an affine bound in S_1. These are specialised
    elementary inequalities, not a new general duality result.
    """
    parameter_check(process, threshold, coefficient)
    a1, a2 = (math.exp(-process.rate*t) for t in (process.first_right, process.maturity))
    c1, c2 = (math.exp(-(process.rate-process.dividend_yield)*t) for t in (process.first_right, process.maturity))
    if threshold is None:
        return dict(constant=coefficient*c2*process.strike, scale=a1-coefficient*c1,
                    shift=0., horizon=process.first_right)
    b = coefficient*c1*threshold-a1*(threshold-process.strike)
    waiting = max(0., -b)+coefficient*c2*process.strike
    scale = a2-coefficient*c2
    if scale < -4*math.ulp(a2):
        raise ArithmeticError('Negative terminal envelope slope')
    return dict(constant=max(waiting, b, 0.), scale=max(0., scale),
                shift=a2*process.strike-b, horizon=process.maturity)


def affine_call_mean(process, scale, strike, horizon):
    if scale < 0 or not all(math.isfinite(x) for x in (scale, strike, horizon)) or horizon <= 0:
        raise ValueError('Valid affine lognormal expectation required')
    if scale == 0:
        return max(-strike, 0.)
    if strike <= 0:
        return scale*process.spot*math.exp((process.rate-process.dividend_yield)*horizon)-strike
    return max(0., scale*float(undiscounted_call(process.spot, strike/scale, horizon, process)))


def tail(process, threshold, method, coefficient, cap):
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('A finite positive clipping cap is required')
    if method == 'terminal_call':
        if threshold is not None:
            raise ValueError('Finite threshold call gains use their deterministic bound')
        return analytic_tail_allowance(process, method, cap)
    if method not in ('zero', 'stock'):
        raise ValueError('Unknown assessor')
    envelope = stock_envelope(process, threshold, coefficient if method == 'stock' else 0.)
    if threshold is None:
        return affine_call_mean(process, envelope['scale'], cap-envelope['constant'], envelope['horizon'])
    return max(envelope['constant']-cap, 0.)+affine_call_mean(
        process, envelope['scale'], envelope['shift']+max(cap,envelope['constant']), envelope['horizon'])


def empirical_upper(samples, cap, allowance, alpha):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or np.any(~np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError('At least two finite non-negative gains are required')
    if not math.isfinite(cap) or cap < 0 or not math.isfinite(allowance) or allowance < 0 or not 0 < alpha < 1:
        raise ValueError('Valid clipping cap, tail allowance and probability required')
    clipped = np.minimum(samples, cap)
    logarithm = math.log(2/alpha)
    radius = math.sqrt(2*float(clipped.var(ddof=1))*logarithm/len(samples))
    radius += 7*cap*logarithm/(3*(len(samples)-1))
    return float(clipped.mean())+radius+allowance


def allocated_probability(budget,members):
    if not math.isfinite(budget) or not 0<budget<1 or type(members) is not int or members<1:
        raise ValueError('A positive family size and probability budget are required')
    alpha = math.nextafter(budget/members,0.)
    if alpha<=0 or Fraction.from_float(alpha)*members>Fraction.from_float(budget):
        raise ValueError('The allocated probability cannot respect the family budget')
    return alpha


def stock_coefficient(first, last, process):
    a1, a2 = first_reward(first, process), math.exp(-process.rate*process.maturity)*np.maximum(last-process.strike, 0.)
    x1 = first*math.exp(-(process.rate-process.dividend_yield)*process.first_right)-process.spot
    x2 = last*math.exp(-(process.rate-process.dividend_yield)*process.maturity)-process.spot
    maximum = np.nextafter(math.exp(-process.dividend_yield*process.maturity), 0.)
    objective = lambda k: float(np.maximum(a1-k*x1, a2-k*x2).mean())
    fit = minimize_scalar(objective, bounds=(0., maximum), method='bounded', options=dict(xatol=1e-9))
    if not fit.success or not math.isfinite(fit.fun):
        raise ArithmeticError('Independent stock-control fit did not converge')
    candidates = (0., float(fit.x), float(maximum))
    selected = min(candidates, key=lambda k: (objective(k), k))
    return selected, dict(objective='Independent pilot mean upper payoff; no reference value or assessment outcome used',
        bounded_interval=[0., float(maximum)], function_evaluations=int(fit.nfev)+len(candidates))


def freeze(process, threshold, method, seed, pilot_count, counts, alpha, deterministic_bound):
    start = clocks()
    if method == 'terminal_call' and threshold is not None:
        # The upper endpoint is exported by the separate certified reference.
        bound = float(Fraction(deterministic_bound['upper']))
        if Fraction.from_float(bound) < Fraction(deterministic_bound['upper']):
            bound = float(np.nextafter(bound, math.inf))
        return dict(coefficient=0., caps={str(n): dict(cap=bound, tail_allowance=0.) for n in counts},
            pilot_paths=0, setup_cost=cost(start), deterministic_bound_reused=True)
    first_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0, 0))))
    last_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0, 1))))
    first, last = simulate(first_rng, last_rng, pilot_count, process, terminal=method != 'terminal_call')
    coefficient, fit = stock_coefficient(first, last, process) if method == 'stock' else (0., None)
    pilot = gains(first, last, process, threshold, method, coefficient)
    statistics = []
    for cap in CAPS:
        clipped = np.minimum(pilot, cap)
        statistics.append(dict(cap=cap, tail_allowance=tail(process, threshold, method, coefficient, cap),
            mean=float(clipped.mean()), variance=float(clipped.var(ddof=1))))
    frozen = {}
    for n in counts:
        logarithm = math.log(2/alpha)
        forecast = lambda row: row['mean']+math.sqrt(2*row['variance']*logarithm/n)+7*row['cap']*logarithm/(3*(n-1))+row['tail_allowance']
        row = min(statistics, key=lambda row: (forecast(row), row['cap']))
        frozen[str(n)] = dict(cap=row['cap'], tail_allowance=row['tail_allowance'], pilot_forecast_upper=forecast(row))
    return dict(coefficient=coefficient, coefficient_fit=fit, caps=frozen, pilot_paths=pilot_count,
                setup_cost=cost(start), deterministic_bound_reused=False)


def affine_truncated_mean(state, intercept, slope, lower, upper, process):
    """Expectation of a positive affine payoff restricted to a lognormal interval."""
    if slope > 0:
        lower = max(lower, -intercept/slope)
    elif slope < 0:
        upper = min(upper, -intercept/slope)
    elif intercept <= 0:
        return 0.
    lower = max(0., lower)
    if upper <= lower:
        return 0.
    h = process.maturity-process.first_right
    sd = process.volatility*math.sqrt(h)
    mu = math.log(state)+(process.rate-process.dividend_yield-.5*process.volatility**2)*h
    zl = -math.inf if lower == 0 else (math.log(lower)-mu)/sd
    zu = math.inf if math.isinf(upper) else (math.log(upper)-mu)/sd
    interval_probability = lambda l,u: float(ndtr(-l)-ndtr(-u)) if l>=0 else float(ndtr(u)-ndtr(l))
    probability = interval_probability(zl,zu)
    first_moment = math.exp(mu+.5*sd**2)*interval_probability(zl-sd,zu-sd)
    return max(0., intercept*probability+slope*first_moment)


def mean_gap_reference(process, threshold, method, coefficient):
    """Independent outer quadrature of analytically integrated terminal branches.

    Error estimates are QUADPACK diagnostics, not integration certificates.
    The policy-loss reference is computed separately with interval arithmetic.
    """
    if method not in ('zero','stock'):
        raise ValueError('The conditional affine reference requires zero or stock centring')
    parameter_check(process,threshold,coefficient)
    started = clocks()
    a2 = math.exp(-process.rate*process.maturity)
    c1 = math.exp(-(process.rate-process.dividend_yield)*process.first_right)
    c2 = math.exp(-(process.rate-process.dividend_yield)*process.maturity)
    k = coefficient if method == 'stock' else 0.
    sd = process.volatility*math.sqrt(process.first_right)
    mu = math.log(process.spot)+(process.rate-process.dividend_yield-.5*process.volatility**2)*process.first_right
    def function(z):
        s = math.exp(mu+sd*z)
        d = float(first_reward(s, process))-k*c1*s
        exercise = threshold is not None and s > threshold
        sign = -1. if exercise else 1.
        value = affine_truncated_mean(s, sign*d, sign*k*c2, 0., process.strike, process)
        value += affine_truncated_mean(s, sign*(d+a2*process.strike), -sign*(a2-k*c2), process.strike, math.inf, process)
        return value*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    splits = [-12., 12.]
    for state in (process.strike, threshold):
        if state is not None:
            z = (math.log(state)-mu)/sd
            if -12 < z < 12:
                splits.append(z)
    splits = sorted(set(splits))
    results = [quad(function, left, right, epsabs=2e-10, epsrel=2e-10, limit=250)
               for left, right in zip(splits, splits[1:])]
    # A positive envelope controls the two omitted normal tails. Given S_1=s,
    # E[G|s] <= (a_1+k*c_1+a_2*exp((r-q)h)+k*c_2*exp((r-q)h))*s.
    # This deliberately broad bound covers both policy decisions and k=0.
    a1 = math.exp(-process.rate*process.first_right)
    carry = math.exp((process.rate-process.dividend_yield)*(process.maturity-process.first_right))
    slope = a1+k*c1+(a2+k*c2)*carry
    omitted = slope*math.exp(mu+.5*sd**2)*(float(ndtr(-12-sd))+float(ndtr(sd-12)))
    return dict(value=sum(v for v, _ in results), quadrature_error_estimate=sum(e for _, e in results),
                omitted_tail_upper=omitted, normal_integration_range=[-12, 12],
                rigorous_enclosure=False, cost=cost(started))


def rational_value(record):
    if not isinstance(record,dict) or set(record) != {'numerator','denominator','approximation'}:
        raise ValueError('A rational certificate endpoint is required')
    value = Fraction(int(record['numerator']),int(record['denominator']))
    if not math.isfinite(float(value)):
        raise ValueError('A finite rational endpoint is required')
    return value


def reference_lookup(reference,process):
    if reference.get('ideal_model_population_loss_enclosed') is not True:
        raise ValueError('Certified ideal-model population shortfalls are required')
    if set(reference['model']) != set(asdict(process)):
        raise ValueError('The reference model has different parameters')
    for key,value in asdict(process).items():
        if rational_value(reference['model'][key]) != Fraction.from_float(value):
            raise ValueError('The independent reference solves a different process')
    lookup = {}
    shared = reference['shared_model_and_boundary_setup']
    for row in reference['policies']:
        endpoint = None if row['threshold'] is None else rational_value(row['threshold'])
        permitted = {Fraction.from_float(t) for t in POLICIES if t is not None}
        if endpoint is not None and endpoint not in permitted:
            raise ValueError('The reference contains a different exact policy threshold')
        threshold = None if endpoint is None else float(endpoint)
        expected_kind = 'always_wait' if threshold is None else 'first_right_threshold'
        if row['policy'] != expected_kind:
            raise ValueError('The reference policy instruction is inconsistent')
        name = label(threshold)
        if name in lookup:
            raise ValueError('Duplicate reference policy')
        lo,hi = (rational_value(row['loss'][key]) for key in ('lower','upper'))
        if not 0 <= lo <= hi:
            raise ValueError('Invalid reference shortfall interval')
        endpoint = row['deterministic_gap_upper']
        bound = None if endpoint is None else rational_value(endpoint)
        if bound is not None and bound < hi:
            raise ValueError('The analytical bound is below the loss upper endpoint')
        work = row['work']
        lookup[name] = dict(loss_interval=dict(lower=str(lo),upper=str(hi)),
            deterministic_gap_upper=None if bound is None else dict(upper=str(bound)),
            deterministic_bound_cost=None if bound is None else {key:work['standalone_analytic_bound_'+key] for key in ('cpu_seconds','wall_seconds')},
            standalone_reference_cost={key:shared[key]+work['reference_body_'+key] for key in ('cpu_seconds','wall_seconds')})
    if set(lookup) != {label(t) for t in POLICIES}:
        raise ValueError('The reference must contain the prescribed policy family')
    return lookup


def write_new(path, record):
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


def run(args):
    root = (ROOT/'data/private').resolve()
    directory = args.output.resolve()
    if not directory.is_relative_to(root) or directory.exists():
        raise ValueError('A fresh private study directory is required')
    positive_integer(args.repetitions, 'repetitions', 2)
    positive_integer(args.pilot_paths, 'pilot_paths', 2)
    reference = json.loads(args.reference.read_text())
    analytic_reference = json.loads(args.analytic_reference.read_text())
    certificate_hash = hashlib.sha256((ROOT/'scripts/certify_policy_loss_reference.py').read_bytes()).hexdigest()
    for record in (reference,analytic_reference):
        if record.get('source_sha256') != certificate_hash:
            raise ValueError('The retained reference identifies a different certificate source')
    if any(analytic_reference.get(flag) is not False for flag in
           ('boundary_search_executed','numerical_quadrature_executed','simulation_executed')):
        raise ValueError('The analytical comparator must use the root-free route')
    if analytic_reference['model'] != reference['model']:
        raise ValueError('The analytical comparator solves a different model')
    process = GapProcess()
    lookup = reference_lookup(reference,process)
    analytic_thresholds = set()
    for row in analytic_reference['policies']:
        theta = rational_value(row['threshold'])
        if theta not in {Fraction.from_float(t) for t in POLICIES if t is not None} or theta in analytic_thresholds:
            raise ValueError('The analytical comparator contains a different threshold family')
        analytic_thresholds.add(theta)
        name = label(float(theta))
        if rational_value(row['deterministic_gap_upper']) != Fraction(lookup[name]['deterministic_gap_upper']['upper']):
            raise ValueError('The independent analytical bound endpoints disagree')
    if len(analytic_thresholds) != len(POLICIES)-1:
        raise ValueError('Incomplete analytical comparator family')
    # Charge the complete measured family computation to each standalone
    # deterministic decision, including validation and result construction.
    # This conservative accounting avoids mistaking an internal body timer
    # for the full cost of an executable comparator.
    for name,record in lookup.items():
        if name != 'always_wait':
            record['deterministic_bound_cost'] = {key:analytic_reference['work'][key] for key in ('cpu_seconds','wall_seconds')}
        record['standalone_reference_cost'] = {key:reference['work'][key] for key in ('cpu_seconds','wall_seconds')}
    family = len(POLICIES)*len(METHODS)*len(COUNTS)
    alpha = allocated_probability(.025,family)
    directory.mkdir(parents=True)
    design = dict(schema_version=1, author='Amanjeet Singh', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        process=asdict(process), thresholds=list(POLICIES), methods=list(METHODS), outer_counts=list(COUNTS),
        loss_tolerances=list(TOLERANCES), repetitions=args.repetitions, root_seed=args.seed, pilot_paths=args.pilot_paths,
        family_error_probability=.025, fixed_look_error_probability=alpha, family_members=family,
        confidence_scope='All five policies, three Monte Carlo assessors and five fixed looks jointly, conditional on frozen independent setup. One bound serves all tolerances.',
        cap_grid=list(CAPS), evaluation_streams='PCG64 SeedSequence(root_seed, spawn_key=(1,repetition,0/1)); shared innovations across policies and methods; nested sample prefixes.',
        pilot_streams='PCG64 SeedSequence(root_seed, spawn_key=(0,0/1)); independent of every evaluation prefix.',
        decision='Certify only if upper mean-gap limit <= monetary tolerance. Failure to certify is unresolved, not proof of unacceptable policy quality.',
        sampling_inequality='Maurer and Pontil (2009), COLT, Theorem 4; clipped iid gains with a deterministic tail allowance.',
        numerical_scope='Certified reference shortfalls and deterministic caps; Monte Carlo simulation, gain evaluations, tail allowances and confidence-limit arithmetic are floating-point computations.',
        cost_scope='Measured function work includes own pilot, coefficient/cap fitting, tail construction, evaluation and decisions. Imports, installation and file serialisation are excluded. Later validation looks do not enter an earlier decision cost.',
        standalone_cost_observations='Each observed assessment prefix adds one retained independent frozen setup measurement; setup is not independently rerun in each repetition.',
        deterministic_cost_scope='Each standalone decision is charged the full measured family computation, including validation and result construction. These single-pass timing observations are descriptive.',
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        analytic_reference_sha256=hashlib.sha256(args.analytic_reference.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), ROOT/'scripts/validate_gap_coverage.py', ROOT/'scripts/validate_clipped_gap_coverage.py')},
        market_validation_claim=False, new_general_theorem_claim=False, public_preregistration_claim=False)
    write_new(directory/'design.json', design)
    rows, setups, gap_references, deterministic = [], [], [], []
    family_false = np.zeros(args.repetitions,dtype=bool)
    whole = clocks()
    for threshold in POLICIES:
        policy = label(threshold)
        ref = lookup[policy]
        lo_fraction,hi_fraction = (Fraction(ref['loss_interval'][key]) for key in ('lower','upper'))
        lo,hi = float(lo_fraction),float(hi_fraction)
        for tolerance in TOLERANCES:
            exact_bound = ref.get('deterministic_gap_upper')
            bound = None if exact_bound is None else float(Fraction(exact_bound['upper']))
            tolerance_fraction = Fraction.from_float(tolerance)
            deterministic.append(dict(policy=policy, tolerance=tolerance,
                analytic_bound_certifies=exact_bound is not None and Fraction(exact_bound['upper']) <= tolerance_fraction,
                certified_reference_certifies=hi_fraction <= tolerance_fraction,
                certified_reference_excludes=lo_fraction > tolerance_fraction,
                loss_lower=lo, loss_upper=hi, analytic_gap_upper=bound,
                standalone_analytic_cost=ref['deterministic_bound_cost'],
                standalone_certified_quadrature_cost=ref['standalone_reference_cost']))
        for method in METHODS:
            setup = freeze(process, threshold, method, args.seed, args.pilot_paths, COUNTS, alpha,
                           ref.get('deterministic_gap_upper'))
            bound_cost = ref.get('deterministic_bound_cost', dict(cpu_seconds=0., wall_seconds=0.))
            if setup['deterministic_bound_reused']:
                setup['setup_cost'] = {key: setup['setup_cost'][key]+bound_cost[key] for key in ('cpu_seconds', 'wall_seconds')}
            setups.append(dict(policy=policy, method=method, **setup))
            gap_ref = dict(value=(lo+hi)/2, value_semantics='Display midpoint of the authoritative interval',
                rational_interval=ref['loss_interval'],rigorous_enclosure=True) if method == 'terminal_call' else mean_gap_reference(process, threshold, method, setup['coefficient'])
            gap_references.append(dict(policy=policy, method=method, expected_gap=gap_ref,
                                      reference_policy_loss=(lo+hi)/2, dual_slack_estimate=gap_ref['value']-(lo+hi)/2))
            outcomes = {epsilon: [] for epsilon in TOLERANCES}
            mean_records = {count: [] for count in COUNTS}
            for repetition in range(args.repetitions):
                start = clocks()
                first_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(args.seed, spawn_key=(1,repetition,0))))
                last_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(args.seed, spawn_key=(1,repetition,1))))
                samples = np.empty(0)
                selected = {}
                if method == 'terminal_call' and threshold is not None:
                    bound = setup['caps'][str(COUNTS[0])]['cap']
                    for tolerance in TOLERANCES:
                        if bound <= tolerance:
                            selected[tolerance] = dict(paths=0,certified=True,
                                assessment_cost=dict(cpu_seconds=0.,wall_seconds=0.),upper=bound)
                looks = []
                for count in COUNTS:
                    first, last = simulate(first_rng, last_rng, count-len(samples), process, terminal=method != 'terminal_call')
                    new = gains(first, last, process, threshold, method, setup['coefficient'])
                    samples = np.concatenate((samples, new))
                    frozen = setup['caps'][str(count)]
                    upper = empirical_upper(samples, frozen['cap'], frozen['tail_allowance'], alpha)
                    mean_records[count].append(dict(mean=float(samples.mean()), upper=upper))
                    for tolerance in TOLERANCES:
                        if tolerance not in selected and upper <= tolerance:
                            selected[tolerance] = dict(paths=count, certified=True, assessment_cost=cost(start), upper=upper)
                    looks.append(dict(paths=count, assessment_cost=cost(start), upper=upper))
                final = looks[-1]
                for tolerance in TOLERANCES:
                    chosen = selected.get(tolerance, dict(paths=final['paths'], certified=False,
                                                         assessment_cost=final['assessment_cost'], upper=final['upper']))
                    one_off = {key: chosen['assessment_cost'][key]+setup['setup_cost'][key] for key in ('cpu_seconds','wall_seconds')}
                    outcomes[tolerance].append(dict(**chosen, one_off_cost=one_off))
            for tolerance, trials in outcomes.items():
                successes = sum(row['certified'] for row in trials)
                truly_above = lo_fraction > Fraction.from_float(tolerance)
                if truly_above:
                    family_false |= np.array([row['certified'] for row in trials],dtype=bool)
                cpu = np.array([row['one_off_cost']['cpu_seconds'] for row in trials])
                assess = np.array([row['assessment_cost']['cpu_seconds'] for row in trials])
                successful_cpu = [row['one_off_cost']['cpu_seconds'] for row in trials if row['certified']]
                rows.append(dict(policy=policy, method=method, loss_tolerance=tolerance, repetitions=args.repetitions,
                    certifications=successes, certification_rate=successes/args.repetitions,
                    certification_rate_wilson_95=wilson_interval(successes,args.repetitions),
                    false_certifications=successes if truly_above else 0,
                    reference_classification='above_tolerance' if truly_above else 'within_tolerance' if hi_fraction <= Fraction.from_float(tolerance) else 'reference_unresolved',
                    mean_paths=float(np.mean([row['paths'] for row in trials])),
                    mean_one_off_cpu_seconds=float(cpu.mean()), p95_one_off_cpu_seconds=float(np.quantile(cpu,.95)),
                    mean_assessment_cpu_seconds=float(assess.mean()),
                    mean_one_off_cpu_seconds_given_success=None if not successful_cpu else float(np.mean(successful_cpu)),
                    setup_cpu_seconds=setup['setup_cost']['cpu_seconds']))
            write_new(directory/f'{policy}_{method}.json', dict(author='Amanjeet Singh', policy=policy,method=method,
                setup=setup, decisions={str(k): v for k,v in outcomes.items()},
                mean_upper_by_count={str(n):float(np.mean([row['upper'] for row in mean_records[n]])) for n in COUNTS},
                mean_gap_by_count={str(n):float(np.mean([row['mean'] for row in mean_records[n]])) for n in COUNTS}))
            print(f'Completed {policy}, {method}', flush=True)
    write_new(directory/'results.json', dict(schema_version=1, author='Amanjeet Singh',
        design_sha256=hashlib.sha256((directory/'design.json').read_bytes()).hexdigest(),
        process=asdict(process), reference=reference, analytical_reference=analytic_reference, deterministic_decisions=deterministic,
        setup=setups, expected_gap_references=gap_references, decisions=rows, actual_study_cost=cost(whole),
        actual_study_cost_scope='Observed validation workflow includes expected-gap reference integrations and per-arm evidence writing. It is not the standalone assessment phase cost. Final summary serialisation and process startup are excluded.',
        family_false_certification_repetitions=int(family_false.sum()),
        family_false_certification_rate=float(family_false.mean()),
        family_false_certification_rate_wilson_95=wilson_interval(int(family_false.sum()),args.repetitions),
        interpretation='Controlled two-right diagnostic. Direct reference integration and analytical decisions may dominate simulation. Success is conditional model certification, not market validation or a general multidate advantage.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--analytic-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=256)
    parser.add_argument('--pilot-paths', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=2026100409)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
