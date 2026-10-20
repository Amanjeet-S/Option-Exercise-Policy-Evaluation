"""Prepare and enclose a public-model multidimensional dual-gap experiment.

Preparation uses the ordinary research environment; assessment uses the
separate python-flint environment. Frozen value approximations combine
max-hinges with averaged individual hinges. Conditional independence gives
the maximum's product CDF; analytical lognormal call moments cover the
individual hinges. Both constructions are established, including the use of
max-call bases by Belomestny, Bender and Schoenmakers (2009).

I propagate verified integration and tail enclosures through stopping-time
differences, retaining shared-increment cancellation. The numerical
certificate concerns the ideal conditional functions and gap algebra at
the supplied binary states. It does not certify the distribution generated
by a floating-point simulator, confidence coverage or the model's fit to a
market. This is a research experiment, not a priority claim.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import time

from certify_max_hinge import ball, certify, ctx, endpoints, exact, normal_cdf, interval_record

ROOT = Path(__file__).resolve().parents[1]
KNOTS = (80., 100., 120., 140.)


def load_fraction(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def value_basis(states, knots=KNOTS):
    """A symmetric, payoff-aligned basis; it does not retain all asset identity."""
    import numpy as np
    states = np.asarray(states, dtype=float)
    columns = [np.ones(len(states)), states.mean(axis=1)/100.]
    columns += [np.maximum(states.max(axis=1)-k, 0.)/100. for k in knots]
    columns += [np.maximum(states-k, 0.).mean(axis=1)/100. for k in knots]
    return np.column_stack(columns)


def prepare(dimension=5, training_count=50000, value_count=50000,
            evaluation_count=64, seed=2026100407):
    import numpy as np
    from dataclasses import asdict
    from qf_research.optimal_stopping.multivariate_benchmark import (
        evaluate_max_call, fit_max_call_policy, published_max_call,
        simulate_max_call, sorted_polynomial_basis)

    if dimension not in (2, 3, 5) or min(training_count, value_count) < 64 or evaluation_count < 2:
        raise ValueError('Require dimension 2, 3 or 5, adequate fitting paths and two evaluation paths')
    wall, cpu = time.perf_counter(), time.process_time()
    model = published_max_call(dimension, 100.)
    streams = np.random.SeedSequence(seed).spawn(3)
    training = simulate_max_call(model, training_count, np.random.default_rng(streams[0]))
    policy = fit_max_call_policy(training, degree=2)
    del training
    policy_end_cpu, policy_end_wall = time.process_time(), time.perf_counter()
    target_cpu, target_wall = 0., 0.
    regression_cpu, regression_wall = 0., 0.
    value_paths = simulate_max_call(model, value_count, np.random.default_rng(streams[1]))
    cash = model.discounted_payoff(value_paths.states[:, -1], len(model.exercise_times)-1)
    values = [None]*len(model.exercise_times)
    diagnostics = []
    terminal = [0.]*10
    terminal[3] = 100.*math.exp(-model.rate*model.maturity)
    values[-1] = terminal
    target_cpu += time.process_time()-policy_end_cpu
    target_wall += time.perf_counter()-policy_end_wall
    for j in range(len(values)-2, 0, -1):
        target_start_cpu, target_start_wall = time.process_time(), time.perf_counter()
        rule = policy.rules[j]
        current = value_paths.states[:, j]
        immediate = model.discounted_payoff(current, j)
        if rule.coefficients:
            continuation = np.maximum(sorted_polynomial_basis(current, model.strike, policy.degree)
                                      @ rule.coefficients, 0.)
            exercise = immediate > continuation
            cash[exercise] = immediate[exercise]
        target_cpu += time.process_time()-target_start_cpu
        target_wall += time.perf_counter()-target_start_wall
        regression_start_cpu, regression_start_wall = time.process_time(), time.perf_counter()
        design = value_basis(current)
        coefficients, _, rank, singular = np.linalg.lstsq(design, cash, rcond=None)
        if not np.all(np.isfinite(coefficients)):
            raise ArithmeticError('Non-finite fitted value coefficients')
        values[j] = coefficients.tolist()
        diagnostics.append(dict(right=j, rank=int(rank), basis_dimension=design.shape[1],
                                condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else None))
        regression_cpu += time.process_time()-regression_start_cpu
        regression_wall += time.perf_counter()-regression_start_wall
    values[0] = [0.]*10
    del value_paths
    outer_start_cpu, outer_start_wall = time.process_time(), time.perf_counter()
    outer = simulate_max_call(model, evaluation_count, np.random.default_rng(streams[2]))
    exercise = evaluate_max_call(policy, outer)
    record = dict(schema_version=1, author='Amanjeet Singh',
                prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                source='Andersen and Broadie (2004), Table 2; mathematical inputs, not observations.',
                model=asdict(model), knots=list(KNOTS), coefficients=values,
                frozen_policy=[list(rule.coefficients) for rule in policy.rules],
                states=outer.states.tolist(), stopping_indices=exercise.stopping_indices.tolist(),
                design=dict(root_seed=seed, training_paths=training_count, value_paths=value_count,
                            outer_paths=evaluation_count, spawn_keys=[list(s.spawn_key) for s in streams],
                            policy_degree=2, policy_basis='Sorted full-state total-degree polynomial',
                            value_basis='Constant, mean stock, four max-hinges, four mean individual hinges',
                            evaluation_independent_of_both_fitting_streams=True),
                value_fit_diagnostics=diagnostics,
                approximation_scope='The value basis is exchangeable but reduced; the policy basis uses every state.')
    end_cpu, end_wall = time.process_time(), time.perf_counter()
    phases = dict(
        policy_training=dict(cpu_seconds=policy_end_cpu-cpu, wall_seconds=policy_end_wall-wall),
        hinge_value_stream_and_targets=dict(cpu_seconds=target_cpu, wall_seconds=target_wall),
        hinge_regression=dict(cpu_seconds=regression_cpu, wall_seconds=regression_wall),
        outer_evaluation_and_record_construction=dict(cpu_seconds=end_cpu-outer_start_cpu,
                                                      wall_seconds=end_wall-outer_start_wall))
    totals = dict(cpu_seconds=end_cpu-cpu, wall_seconds=end_wall-wall)
    phases['preparation_bookkeeping'] = {key:totals[key]-sum(row[key] for row in phases.values())
                                          for key in totals}
    if any(value < -1e-9 for row in phases.values() for value in row.values()):
        raise ArithmeticError('Preparation phase timers overlap')
    record['preparation_cost'] = totals
    record['preparation_phases'] = phases
    record['preparation_cost_scope'] = 'Exclusive measured policy, hinge-target, hinge-regression and outer-evaluation phases; bookkeeping completes their sum. JSON file writing and process start-up are excluded.'
    return record


def realised_value(states, coefficients, knots):
    """Exact rational evaluation of the declared frozen hinge function."""
    states, coefficients, knots = tuple(map(exact, states)), tuple(map(exact, coefficients)), tuple(map(exact, knots))
    if not states or any(s <= 0 for s in states) or len(coefficients) != 2+2*len(knots):
        raise ValueError('Invalid states or frozen coefficients')
    dimension = len(states)
    features = [Fraction(1), sum(states)/dimension/100]
    features += [max(max(states)-k, Fraction(0))/100 for k in knots]
    features += [sum(max(s-k, Fraction(0)) for s in states)/dimension/100 for k in knots]
    return sum(c*f for c, f in zip(coefficients, features))


def call_ball(state, strike, rate, dividend, volatility, step):
    sd = ball(volatility)*ball(step).sqrt()
    d2 = ((ball(state)/ball(strike)).log()
          +(ball(rate)-ball(dividend)-ball(volatility)**2/2)*ball(step))/sd
    return (ball(state)*((ball(rate)-ball(dividend))*ball(step)).exp()*normal_cdf(d2+sd)
            -ball(strike)*normal_cdf(d2))


def conditional_value(states, coefficients, knots, model, step, tolerance, *, precision=96):
    """Refine using the previous state only, preserving predictability.

    Every attempted integral is charged. The target is an actual enclosure
    width, including the analytic tail, rather than an integrator's tolerance.
    """
    states, coefficients, knots = tuple(map(exact, states)), tuple(map(exact, coefficients)), tuple(map(exact, knots))
    tolerance = exact(tolerance)
    if tolerance <= 0 or not states or len(coefficients) != 2+2*len(knots):
        raise ValueError('Require positive tolerance and a complete frozen value basis')
    rate, dividend, volatility = map(exact, (model['rate'], model['dividends'][0], model['volatilities'][0]))
    cutoff = max(max(knots)+1, max(states)*Fraction(5, 4))
    calls, evaluations = 0, 0
    with ctx.workprec(precision):
        base = ball(coefficients[0])
        base += ball(coefficients[1])*sum(ball(s) for s in states)/len(states)/100*((ball(rate)-ball(dividend))*ball(step)).exp()
        for c, knot in zip(coefficients[2+len(knots):], knots):
            if c:
                base += ball(c)*sum(call_ball(s, knot, rate, dividend, volatility, step) for s in states)/len(states)/100
        base_lower, base_upper = endpoints(base)
        for attempt in range(16):
            lower, upper = base_lower, base_upper
            for c, knot in zip(coefficients[2:2+len(knots)], knots):
                if not c:
                    continue
                record = certify(states, strike=knot, rate=rate, dividend=dividend,
                                 volatility=volatility, step=step, cutoff=cutoff, precision=precision,
                                 absolute_tolerance=Fraction(1, 10**12), relative_tolerance=Fraction(1, 10**12))
                lo = load_fraction(record['expectation']['lower'])
                hi = load_fraction(record['expectation']['upper'])
                lo, hi = c*lo/100, c*hi/100
                lower += min(lo, hi)
                upper += max(lo, hi)
                calls += 1
                evaluations += record['work']['integrand_evaluations']
            if upper-lower <= tolerance:
                return lower, upper, dict(integrals=calls, integrand_evaluations=evaluations,
                                         attempts=attempt+1, cutoff=float(cutoff), width=float(upper-lower))
            cutoff *= Fraction(13, 10)
    raise ArithmeticError('The actual conditional enclosure width exceeds its target after refinement')


def signed_gap(reward_intervals, penalty, drift_intervals, stopping):
    """Rational gain enclosure, with the selected-date contribution exactly zero."""
    rights = len(penalty)
    if (len(reward_intervals) != rights or len(drift_intervals) != rights or
            type(stopping) is not int or not 0 <= stopping < rights):
        raise ValueError('Require matching rights and a valid integer stopping index')
    if any(lo > hi for lo, hi in (*reward_intervals, *drift_intervals)):
        raise ValueError('Empty intervals are invalid')
    lower, upper = Fraction(0), Fraction(0)
    plain_lower, plain_upper = Fraction(0), Fraction(0)
    for j in range(rights):
        if j == stopping:
            continue
        blo = reward_intervals[j][0]-reward_intervals[stopping][1]-penalty[j]+penalty[stopping]
        bhi = reward_intervals[j][1]-reward_intervals[stopping][0]-penalty[j]+penalty[stopping]
        if j > stopping:
            dlo = sum(lo for lo, _ in drift_intervals[stopping+1:j+1])
            dhi = sum(hi for _, hi in drift_intervals[stopping+1:j+1])
        else:
            dlo = -sum(hi for _, hi in drift_intervals[j+1:stopping+1])
            dhi = -sum(lo for lo, _ in drift_intervals[j+1:stopping+1])
        lower, upper = max(lower, blo+dlo), max(upper, bhi+dhi)
        plain_lower, plain_upper = max(plain_lower, blo), max(plain_upper, bhi)
    absolute = sum(max(abs(lo), abs(hi)) for lo, hi in drift_intervals[1:])
    return lower, upper, max(Fraction(0), plain_lower-absolute), plain_upper+absolute


def check_inputs(record):
    model = record['model']
    states, stops, values = record['states'], record['stopping_indices'], record['coefficients']
    dimension, rights = len(model['spots']), len(model['exercise_times'])
    exact(model['rate'])
    if any(exact(s) <= 0 for s in model['spots']):
        raise ValueError('Initial spots must be finite and strictly positive')
    tuple(map(exact, (*model['dividends'], *model['volatilities'])))
    if (dimension not in (2, 3, 5) or rights < 2 or model['strike'] != 100 or len(values) != rights
            or len(states) != len(stops) or len(states) < 2
            or len(record['knots']) != 4 or any(exact(k) <= 0 for k in record['knots'])):
        raise ValueError('Invalid prepared design')
    if record.get('design', {}).get('outer_paths') != len(states):
        raise ValueError('The declared evaluation count differs from the complete paths')
    if (len(model['volatilities']) != dimension or len(model['dividends']) != dimension
            or len(model['correlation']) != dimension
            or len(set(model['volatilities'])) != 1 or model['volatilities'][0] <= 0
            or len(set(model['dividends'])) != 1
            or any(row != [int(i == j) for j in range(dimension)] for i, row in enumerate(model['correlation']))):
        raise ValueError('Conditional certification requires independent homogeneous positive-volatility assets')
    times = list(map(exact, model['exercise_times']))
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Require a strictly increasing calendar from zero')
    for path, stop in zip(states, stops):
        if len(path) != rights or type(stop) is not int or not 0 <= stop < rights:
            raise ValueError('Invalid path calendar or stopping index')
        if any(len(s) != dimension or any(exact(x) <= 0 for x in s) for s in path):
            raise ValueError('Require finite positive states with the declared dimension')
        if path[0] != model['spots']:
            raise ValueError('Every path must start at the declared spots')
    for coefficients in values:
        if len(coefficients) != 10:
            raise ValueError('Require every frozen coefficient')
        tuple(map(exact, coefficients))
    return model, states, stops, values, times


def check_policy(record, *, precision=192):
    """Check observed decisions against a frozen adapted polynomial rule.

    This verifies decisions at the supplied states. It establishes neither
    their sampling law nor accuracy of the policy away from those states.
    An unresolved comparison is an error; no ambiguous path is omitted.
    """
    model, paths, stops, _, times = check_inputs(record)
    rules = record.get('frozen_policy')
    dimension = len(model['spots'])
    indices = ((), *(p for degree in (1, 2) for p in combinations_with_replacement(range(dimension), degree)))
    if not isinstance(rules, list) or len(rules) != len(times) or rules[-1]:
        raise ValueError('A complete frozen policy with terminal exercise is required')
    if any(rule and len(rule) != len(indices) for rule in rules[:-1]):
        raise ValueError('The frozen policy basis dimension is invalid')
    coefficients = [tuple(map(exact, rule)) for rule in rules]
    with ctx.workprec(precision):
        for states, declared_stop in zip(paths, stops):
            realised_stop = len(times)-1
            for j, rule in enumerate(coefficients[:-1]):
                if not rule:
                    continue
                current = tuple(map(exact, states[j]))
                hinge = max(max(current)-100, Fraction(0))
                if hinge == 0:
                    continue
                ordered = [x/100-1 for x in sorted(current, reverse=True)]
                design = [math.prod(ordered[k] for k in p) for p in indices]
                continuation = max(sum(c*f for c, f in zip(rule, design)), Fraction(0))
                lower, upper = endpoints((-ball(model['rate'])*ball(times[j])).exp()*ball(hinge))
                if lower > continuation:
                    realised_stop = j
                    break
                if upper > continuation:
                    raise ArithmeticError('An exercise comparison is unresolved at the declared precision')
            if realised_stop != declared_stop:
                raise ArithmeticError('The supplied stopping index differs from the exact frozen rule')
    return True


def assess(record, tolerance, precision=96):
    model, paths, stops, values, times = check_inputs(record)
    wall, cpu = time.perf_counter(), time.process_time()
    policy_checked = check_policy(record)
    observations = []
    total_lower = total_upper = Fraction(0)
    for path_index, (states, stopping) in enumerate(zip(paths, stops)):
        rewards, penalty, drifts, work = [], [Fraction(0)], [(Fraction(0), Fraction(0))], []
        with ctx.workprec(precision):
            for state, instant in zip(states, times):
                rewards.append(endpoints((-ball(model['rate'])*ball(instant)).exp()
                                         *ball(max(max(map(exact, state))-100, 0))))
        for j in range(1, len(times)):
            lo, hi, effort = conditional_value(states[j-1], values[j], record['knots'], model,
                                               times[j]-times[j-1], tolerance, precision=precision)
            centre = (lo+hi)/2
            penalty.append(penalty[-1]+realised_value(states[j], values[j], record['knots'])-centre)
            drifts.append((lo-centre, hi-centre))
            work.append(effort)
        lo, hi, alo, ahi = signed_gap(rewards, penalty, drifts, stopping)
        total_lower += lo
        total_upper += hi
        observations.append(dict(path_index=path_index, stopping_index=stopping,
            signed_gap_certificate=interval_record(lo, hi),
            absolute_gap_certificate=interval_record(alo, ahi),
            signed_lower=float(lo), signed_upper=float(hi), signed_width=float(hi-lo),
            absolute_lower=float(alo), absolute_upper=float(ahi), absolute_width=float(ahi-alo),
            signed_inside_absolute=alo <= lo <= hi <= ahi,
            conditional_integrals=sum(w['integrals'] for w in work),
            integrand_evaluations=sum(w['integrand_evaluations'] for w in work),
            largest_conditional_width=max(w['width'] for w in work)))
    average = lambda key: sum(row[key] for row in observations)/len(observations)
    return dict(tolerance_per_increment=float(exact(tolerance)), precision_bits=precision,
                scope='Ideal conditional functions and real gain algebra at exact binary input states; simulation law and sampling coverage are not certified.',
                outer_paths=len(observations),
                observed_policy_decisions_checked=policy_checked,
                mean_signed_gap_certificate=interval_record(total_lower/len(observations), total_upper/len(observations)),
                mean_signed_gap_interval=[float(total_lower/len(observations)), float(total_upper/len(observations))],
                float_endpoint_semantics='Rounded display only; rational certificate endpoints are authoritative.',
                mean_signed_width=average('signed_width'), mean_absolute_width=average('absolute_width'),
                conditional_integrals=sum(row['conditional_integrals'] for row in observations),
                integrand_evaluations=sum(row['integrand_evaluations'] for row in observations),
                all_signed_inside_absolute=all(row['signed_inside_absolute'] for row in observations),
                cpu_seconds=time.process_time()-cpu, wall_seconds=time.perf_counter()-wall,
                observations=observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'assess'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dimension', type=int, default=5)
    parser.add_argument('--outer-paths', type=int, default=64)
    parser.add_argument('--tolerances', nargs='+', type=float, default=[1., .1, .01])
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/'data/private').resolve()) or destination.exists():
        raise ValueError('Use a new private output file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        result = prepare(dimension=args.dimension, evaluation_count=args.outer_paths)
    else:
        if args.input is None:
            raise ValueError('A frozen prepared input file is required')
        source = args.input.read_bytes()
        result = dict(schema_version=1, author='Amanjeet Singh', input_sha256=hashlib.sha256(source).hexdigest(),
                      assessed_at_utc=datetime.now(timezone.utc).isoformat(),
                      runs=[assess(json.loads(source), tolerance) for tolerance in args.tolerances])
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved {args.mode} experiment to {destination}', flush=True)


if __name__ == '__main__':
    main()
