"""Verified polynomial and exact-terminal regression-later dual comparators.

The polynomial conditional moments and regression-later construction are
established methods (Glasserman and Yu, 2004). The hybrid replaces only the
terminal fitted polynomial with the discounted max-call payoff. Its terminal
conditional expectation uses the independent-asset product-CDF integral.
There is no claim of methodological priority.

Frozen binary coefficients and states are treated as exact rationals. Arb
encloses conditional moments, discounts and terminal integration. Shared
increments cancel in each stopped-gap difference. The resulting intervals
enclose ideal path arithmetic, not the simulator law, a population mean or
statistical confidence coverage. Each degree/method is assessed independently;
no cached terminal work is omitted from its standalone assessment cost.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import statistics
import time

from certify_max_hinge import ball, certify, ctx, endpoints, exact, interval_record
from certify_multivariate_gap import check_inputs, check_policy, load_fraction, signed_gap

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('polynomial_all_dates', 'hybrid_exact_terminal')
SOURCE_FILES = ('scripts/certify_polynomial_dual.py', 'scripts/certify_multivariate_gap.py',
                'scripts/certify_max_hinge.py', 'scripts/validate_polynomial_dual.py')


def exponents(dimension, degree):
    if type(dimension) is not int or dimension < 1 or type(degree) is not int or not 0 <= degree <= 3:
        raise ValueError('A positive dimension and polynomial degree zero to three are required')
    result = [(0,)*dimension]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers = [0]*dimension
            for index in indices:
                powers[index] += 1
            result.append(tuple(powers))
    return tuple(result)


def validate_polynomial(states, coefficients, powers):
    states = tuple(map(exact, states))
    coefficients = tuple(map(exact, coefficients))
    if not states or any(s <= 0 for s in states) or len(coefficients) != len(powers):
        raise ValueError('Positive states and matching complete coefficients are required')
    if any(len(a) != len(states) or any(type(v) is not int or v < 0 for v in a) for a in powers):
        raise ValueError('Non-negative integer exponents with matching dimension are required')
    return states, coefficients


def realised_polynomial(states, coefficients, powers):
    states, coefficients = validate_polynomial(states, coefficients, powers)
    return sum((c*math.prod((s/100)**a for s, a in zip(states, powers_i))
                for c, powers_i in zip(coefficients, powers)), Fraction(0))


def conditional_polynomial(states, coefficients, powers, model, step, *, precision=96, factor_cache=None):
    """Enclose the analytical monomial moments under independent GBM."""
    if ctx is None:
        raise RuntimeError('The separate python-flint environment is required')
    states, coefficients = validate_polynomial(states, coefficients, powers)
    step = exact(step)
    dimension = len(states)
    if (step <= 0 or type(precision) is not int or precision < 32
            or len(model['dividends']) != dimension or len(model['volatilities']) != dimension
            or model['correlation'] != [[int(i == j) for j in range(dimension)] for i in range(dimension)]):
        raise ValueError('Independent GBM, matching parameters, positive step and adequate precision required')
    rate = exact(model['rate'])
    dividends, volatilities = tuple(map(exact, model['dividends'])), tuple(map(exact, model['volatilities']))
    if any(s <= 0 for s in volatilities):
        raise ValueError('Positive finite volatility is required')
    if not any(coefficients):
        return Fraction(0), Fraction(0), dict(polynomial_terms_evaluated=0,
            moment_factor_exponentials=0, moment_factor_cache_builds=0,
            moment_factor_cache_hits=0, moment_factors_reused=0)
    # These factors depend on the model, calendar increment and basis, never
    # on a realised state. Each assessment arm owns its cache. Precision and
    # exact model parameters enter the key so a changed model or retry cannot
    # reuse an incompatible enclosure.
    key = (precision, step, rate, dividends, volatilities, tuple(tuple(a) for a in powers))
    cached = factor_cache is not None and key in factor_cache
    exponentials = 0
    evaluated = 0
    with ctx.workprec(precision):
        if cached:
            factors = factor_cache[key]
        else:
            factors = []
            for powers_i in powers:
                growth = sum((a*(rate-q)+Fraction(a*(a-1), 2)*sigma**2
                              for a, q, sigma in zip(powers_i, dividends, volatilities)), Fraction(0))
                factors.append((ball(step*growth)).exp() if growth else ball(1))
                exponentials += bool(growth)
            factors = tuple(factors)
            if factor_cache is not None:
                factor_cache[key] = factors
        value = ball(0)
        for coefficient, powers_i, factor in zip(coefficients, powers, factors):
            if not coefficient:
                continue
            prefactor = coefficient*math.prod((s/100)**a for s, a in zip(states, powers_i))
            value += ball(prefactor)*factor
            evaluated += 1
        lower, upper = endpoints(value)
    return lower, upper, dict(polynomial_terms_evaluated=evaluated,
        moment_factor_exponentials=exponentials, moment_factor_cache_builds=int(not cached),
        moment_factor_cache_hits=int(cached), moment_factors_reused=evaluated if cached else 0)


def discounted_payoff(state, model, instant, *, precision=96):
    hinge = max(max(map(exact, state))-exact(model['strike']), Fraction(0))
    with ctx.workprec(precision):
        return endpoints((-ball(model['rate'])*ball(instant)).exp()*ball(hinge))


def conditional_terminal(states, model, step, target, *, precision=96, maximum_attempts=16):
    """Enclose the exact terminal payoff, including its exact model discount."""
    if ctx is None:
        raise RuntimeError('The separate python-flint environment is required')
    target, step = exact(target), exact(step)
    if (target <= 0 or step <= 0 or type(maximum_attempts) is not int or maximum_attempts < 1
            or type(precision) is not int or precision < 32):
        raise ValueError('Positive terminal width, step and attempt count are required')
    states = tuple(map(exact, states))
    if not states or any(s <= 0 for s in states):
        raise ValueError('Positive conditional states are required')
    dimension = len(states)
    if (len(model['volatilities']) != dimension or len(model['dividends']) != dimension
            or len(set(model['volatilities'])) != 1 or len(set(model['dividends'])) != 1
            or model['correlation'] != [[int(i == j) for j in range(dimension)] for i in range(dimension)]):
        raise ValueError('The terminal max-call integral requires independent homogeneous assets')
    cutoff = max(exact(model['strike'])+1, max(states)*Fraction(5, 4))
    integrals = evaluations = 0
    with ctx.workprec(precision):
        discount_lower, discount_upper = endpoints(
            (-ball(model['rate'])*ball(model['exercise_times'][-1])).exp())
    for attempt in range(maximum_attempts):
        result = certify(states, strike=model['strike'], rate=model['rate'],
            dividend=model['dividends'][0], volatility=model['volatilities'][0],
            step=step, cutoff=cutoff, precision=precision,
            absolute_tolerance=Fraction(1, 10**12), relative_tolerance=Fraction(1, 10**12))
        lower = discount_lower*load_fraction(result['expectation']['lower'])
        upper = discount_upper*load_fraction(result['expectation']['upper'])
        integrals += 1
        evaluations += result['work']['integrand_evaluations']
        if upper-lower <= target:
            return lower, upper, dict(conditional_integrals=integrals,
                integrand_evaluations=evaluations, attempts=attempt+1,
                actual_width=float(upper-lower), cutoff=float(cutoff))
        cutoff *= Fraction(13, 10)
    raise ArithmeticError('The terminal enclosure did not attain its actual width target')


def gain_enclosure(rewards, increments, stopping):
    """Use only the increment span between candidate and stopping dates."""
    if len(rewards) != len(increments) or any(lo > hi for lo, hi in increments):
        raise ValueError('Matching valid reward and martingale-increment intervals are required')
    # signed_gap adds the drift span. Negating the martingale increments
    # supplies precisely -(M_j-M_tau); the selected-date term is exactly zero.
    drifts = tuple((-upper, -lower) for lower, upper in increments)
    lower, upper, _, _ = signed_gap(rewards, (Fraction(0),)*len(rewards), drifts, stopping)
    return lower, upper


def validate_export(record, polynomial, degrees):
    model, paths, stops, _, times = check_inputs(record)
    if not isinstance(polynomial.get('degrees'), dict) or not polynomial['degrees']:
        raise ValueError('Exported per-degree fits are required')
    if (polynomial.get('root_seed') != record['design']['root_seed']
            or polynomial.get('sample_count') != len(paths)
            or polynomial.get('value_fitting_paths') != record['design']['value_paths']
            or polynomial.get('value_spawn_key') != [1]):
        raise ValueError('The polynomial fit metadata differs from the frozen input')
    dimension = len(model['spots'])
    frozen = {}
    for degree in degrees:
        if type(degree) is not int or degree not in (2, 3) or str(degree) not in polynomial['degrees']:
            raise ValueError('Exported degree-two or degree-three fits are required')
        exported = polynomial['degrees'][str(degree)]
        expected = exponents(dimension, degree)
        declared = exported.get('basis_exponents')
        if not isinstance(declared, list) or any(not isinstance(a, list) for a in declared):
            raise ValueError('The complete ordered exponent vectors must be exported')
        if tuple(tuple(a) for a in declared) != expected or any(type(a) is not int for row in declared for a in row):
            raise ValueError('The exported polynomial basis differs from the declared monomials')
        coefficients = exported.get('coefficients')
        if not isinstance(coefficients, list) or len(coefficients) != len(times):
            raise ValueError('Every exercise date must have complete frozen coefficients')
        frozen[degree] = tuple(validate_polynomial(model['spots'], row, expected)[1] for row in coefficients)
    return model, paths, stops, times, frozen


def assess_one(model, paths, stops, times, coefficients, degree, method, *, target, precision=96):
    """Independent full assessment of one degree/method, without shared caches."""
    if method not in METHODS:
        raise ValueError('Unknown verified comparator')
    cpu, wall = time.process_time(), time.perf_counter()
    powers = exponents(len(model['spots']), degree)
    increment_target = target/(len(times)-1)/2
    observations = []
    total_lower = total_upper = Fraction(0)
    terms = integrals = evaluations = precision_retries = 0
    factor_cache = {}
    total_exponentials = total_cache_builds = total_cache_hits = total_reused_factors = 0
    for path_index, (states, stopping) in enumerate(zip(paths, stops)):
        current_precision = precision
        attempted_integrals = attempted_evaluations = attempted_terms = 0
        attempted_exponentials = attempted_cache_builds = attempted_cache_hits = attempted_reused_factors = 0
        for retry in range(5):
            rewards = [discounted_payoff(state, model, instant, precision=current_precision)
                       for state, instant in zip(states, times)]
            increments = [(Fraction(0), Fraction(0))]
            terminal_work = None
            for j in range(1, len(times)):
                step = times[j]-times[j-1]
                if method == 'hybrid_exact_terminal' and j == len(times)-1:
                    realised_lower, realised_upper = rewards[-1]
                    conditional_lower, conditional_upper, terminal_work = conditional_terminal(
                        states[j-1], model, step, increment_target, precision=current_precision)
                    attempted_integrals += terminal_work['conditional_integrals']
                    attempted_evaluations += terminal_work['integrand_evaluations']
                else:
                    realised_lower = realised_upper = realised_polynomial(states[j], coefficients[j], powers)
                    conditional_lower, conditional_upper, work = conditional_polynomial(
                        states[j-1], coefficients[j], powers, model, step,
                        precision=current_precision, factor_cache=factor_cache)
                    attempted_terms += work['polynomial_terms_evaluated']
                    attempted_exponentials += work['moment_factor_exponentials']
                    attempted_cache_builds += work['moment_factor_cache_builds']
                    attempted_cache_hits += work['moment_factor_cache_hits']
                    attempted_reused_factors += work['moment_factors_reused']
                increments.append((realised_lower-conditional_upper, realised_upper-conditional_lower))
            lower, upper = gain_enclosure(rewards, increments, stopping)
            if upper-lower <= target:
                break
            current_precision *= 2
            precision_retries += 1
        else:
            raise ArithmeticError('The actual path-gap width exceeds its target after precision refinement')
        total_lower += lower
        total_upper += upper
        terms += attempted_terms
        integrals += attempted_integrals
        evaluations += attempted_evaluations
        total_exponentials += attempted_exponentials
        total_cache_builds += attempted_cache_builds
        total_cache_hits += attempted_cache_hits
        total_reused_factors += attempted_reused_factors
        observations.append(dict(path_index=path_index, stopping_index=stopping,
            signed_gap_certificate=interval_record(lower, upper),
            selected_date_gap_certificate=interval_record(Fraction(0), Fraction(0)),
            precision_bits_used=current_precision, precision_attempts=retry+1,
            conditional_integrals=attempted_integrals, integrand_evaluations=attempted_evaluations,
            polynomial_terms_evaluated=attempted_terms,
            moment_factor_exponentials=attempted_exponentials,
            moment_factor_cache_builds=attempted_cache_builds,
            moment_factor_cache_hits=attempted_cache_hits,
            moment_factors_reused=attempted_reused_factors, terminal_work=terminal_work))
    count = len(observations)
    widths = [load_fraction(row['signed_gap_certificate']['width']) for row in observations]
    midpoints = [float((load_fraction(row['signed_gap_certificate']['lower'])+
                        load_fraction(row['signed_gap_certificate']['upper']))/2) for row in observations]
    result = dict(method=method, degree=degree, outer_paths=count,
        mean_signed_gap_certificate=interval_record(total_lower/count, total_upper/count),
        approximate_midpoint_mean=statistics.mean(midpoints),
        approximate_midpoint_mean_standard_error=statistics.stdev(midpoints)/math.sqrt(count),
        mean_signed_width=float(sum(widths)/count), maximum_signed_width=float(max(widths)),
        midpoint_mean_numerical_error_bound=float(sum(widths)/count/2),
        all_targets_met=all(width <= target for width in widths),
        conditional_integrals=integrals, integrand_evaluations=evaluations,
        polynomial_terms_evaluated=terms, precision_refinements=precision_retries,
        moment_factor_exponentials=total_exponentials,
        moment_factor_cache_builds=total_cache_builds,
        moment_factor_cache_hits=total_cache_hits, moment_factors_reused=total_reused_factors,
        observations=observations)
    result.update(assessment_cpu_seconds=time.process_time()-cpu,
                  assessment_wall_seconds=time.perf_counter()-wall)
    return result


def assess(record, polynomial, *, target=Fraction(1, 100), precision=96, degrees=None, methods=METHODS):
    if ctx is None:
        raise RuntimeError('The separate python-flint environment is required')
    target = exact(target)
    if target <= 0 or type(precision) is not int or precision < 32:
        raise ValueError('A positive exact width target and adequate precision are required')
    if not methods or len(set(methods)) != len(methods) or any(name not in METHODS for name in methods):
        raise ValueError('Distinct supported comparators are required')
    if degrees is None:
        degrees = tuple(sorted(int(key) for key in polynomial.get('degrees', {})))
    if not degrees or len(set(degrees)) != len(degrees):
        raise ValueError('Distinct exported polynomial degrees are required')
    cpu, wall = time.process_time(), time.perf_counter()
    validation_cpu, validation_wall = time.process_time(), time.perf_counter()
    model, paths, stops, times, frozen = validate_export(record, polynomial, degrees)
    policy_checked = check_policy(record)
    validation_cpu = time.process_time()-validation_cpu
    validation_wall = time.perf_counter()-validation_wall
    output = {}
    for degree in degrees:
        exported = polynomial['degrees'][str(degree)]
        output[str(degree)] = dict(methods={})
        for method in methods:
            result = assess_one(model, paths, stops, times, frozen[degree], degree, method,
                                target=target, precision=precision)
            result['standalone_assessment_cpu_seconds_including_validation'] = validation_cpu+result['assessment_cpu_seconds']
            result['standalone_assessment_wall_seconds_including_validation'] = validation_wall+result['assessment_wall_seconds']
            setup = exported.get('hybrid_setup_pricing_cpu_seconds') if method == 'hybrid_exact_terminal' else (
                exported.get('standalone_polynomial_pricing_cpu_seconds', 0)-exported.get('dual_assessment_cpu_seconds', 0)
                if 'standalone_polynomial_pricing_cpu_seconds' in exported else None)
            if setup is not None:
                if not math.isfinite(setup) or setup < 0:
                    raise ValueError('A finite non-negative retained pricing setup cost is required')
                result['retained_setup_pricing_cpu_seconds'] = setup
                result['standalone_pricing_cpu_seconds'] = setup+result['assessment_cpu_seconds']
                result['standalone_verified_workflow_cpu_seconds'] = setup+result['assessment_cpu_seconds']+validation_cpu
            output[str(degree)]['methods'][method] = result
    return dict(schema_version=1, author='Amanjeet Singh', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Verified ideal conditional functions and stopped-gap arithmetic at exact binary input states. Neither simulator law, a population mean nor confidence coverage is enclosed.',
        construction='Established polynomial regression-later martingales; hybrid terminal function is the exact discounted max-call payoff.',
        root_seed=record['design']['root_seed'], outer_paths=len(paths), degrees=output,
        observed_policy_decisions_checked=policy_checked,
        path_gap_width_target=interval_record(target, target), precision_bits=precision,
        input_and_policy_validation_cpu_seconds=validation_cpu,
        input_and_policy_validation_wall_seconds=validation_wall,
        independent_arm_costs=True, exact_terminal_model_discount=True,
        numerical_endpoint_semantics='Rational endpoints are authoritative; all displayed floats are rounded.',
        statistical_scope='Ordinary within-fit midpoint standard errors are approximate conditional sampling diagnostics, not coverage certificates.',
        timing_scope='Each degree and method executes its own complete assessment. State-independent monomial growth factors are cached within each arm by exact model, step, basis and precision; all cache construction is charged. No moment or terminal-query cache is shared between arms. Validation is measured once and charged to each standalone verified workflow. Retained fitting setup costs are added phase measurements, not newly rerun fits. Process start-up, imports and JSON serialisation are excluded.',
        ideal_martingale_integrability='Finite coefficients and finite-degree monomials have all finite moments over the finite lognormal calendar; terminal max-call payoff is integrable.',
        new_general_theorem_claim=False, actual_assessment_cpu_seconds=time.process_time()-cpu,
        actual_assessment_wall_seconds=time.perf_counter()-wall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--polynomial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', default='1/100')
    parser.add_argument('--precision', type=int, default=96)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/'data/private').resolve()) or destination.exists():
        raise ValueError('A new private comparator output is required')
    source, polynomial_source = args.input.read_bytes(), args.polynomial.read_bytes()
    record, polynomial = json.loads(source), json.loads(polynomial_source)
    input_hash = hashlib.sha256(source).hexdigest()
    if polynomial.get('input_sha256') != input_hash:
        raise ValueError('The exported fits identify a different complete input record')
    result = assess(record, polynomial, target=Fraction(args.target), precision=args.precision)
    result.update(input_sha256=input_hash, polynomial_sha256=hashlib.sha256(polynomial_source).hexdigest(),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved verified polynomial and hybrid assessments to {destination}', flush=True)


if __name__ == '__main__':
    main()
