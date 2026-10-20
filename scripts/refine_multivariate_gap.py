"""Compare uniform and gap-directed verified refinement with fixed centres.

This experiment refines numerical enclosures of a fixed ideal martingale.
The frozen value functions and stopping policy are never refitted. Each
coarse centre q_j is fixed using the preceding state alone, before either
refinement strategy runs. Later choices can use the whole realised path
because they change only the accuracy of conditional expectation brackets,
not the centre, penalty, policy or ideal compensated martingale.

Outputs contain derived statistics and rational gap certificates only.
No path arrays or regression coefficients are copied into the output.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from certify_multivariate_gap import (
    ball, check_inputs, check_policy, conditional_value, ctx, endpoints, exact,
    interval_record, realised_value, signed_gap,
)

ROOT = Path(__file__).resolve().parents[1]


def intersect(previous, refined):
    """Two enclosures of the same real number must have nonempty overlap."""
    lo, hi = max(previous[0], refined[0]), min(previous[1], refined[1])
    if lo > hi:
        raise ArithmeticError('Conditional expectation certificates have empty intersection')
    return lo, hi


def drift_intervals(brackets, centres):
    if len(brackets) != len(centres):
        raise ValueError('Matching brackets and frozen centres are required')
    return tuple((lo-q, hi-q) for (lo, hi), q in zip(brackets, centres))


def candidate_bounds(rewards, penalty, drifts, stopping):
    """Individual date bounds, retaining shared increment cancellation."""
    # Validate the calendar, interval ordering and stopping index centrally.
    signed_gap(rewards, penalty, drifts, stopping)
    candidates = []
    for j in range(len(penalty)):
        if j == stopping:
            candidates.append((Fraction(0), Fraction(0)))
            continue
        lo = rewards[j][0]-rewards[stopping][1]-penalty[j]+penalty[stopping]
        hi = rewards[j][1]-rewards[stopping][0]-penalty[j]+penalty[stopping]
        span = range(min(j, stopping)+1, max(j, stopping)+1)
        if j > stopping:
            lo += sum(drifts[k][0] for k in span)
            hi += sum(drifts[k][1] for k in span)
        else:
            lo -= sum(drifts[k][1] for k in span)
            hi -= sum(drifts[k][0] for k in span)
        candidates.append((lo, hi))
    return tuple(candidates)


def select_increment(rewards, penalty, drifts, stopping):
    """Largest-width increment in the current maximum upper-date span.

    Ties choose the date and increment closest to the stopping date, then
    the smaller index. No future state enters the frozen coarse centres.
    """
    candidates = candidate_bounds(rewards, penalty, drifts, stopping)
    date = max(range(len(candidates)),
               key=lambda j: (candidates[j][1], -abs(j-stopping), -j))
    span = tuple(range(min(date, stopping)+1, max(date, stopping)+1))
    eligible = tuple(j for j in span if drifts[j][1] > drifts[j][0])
    if not eligible:
        eligible = tuple(j for j in range(1, len(drifts)) if drifts[j][1] > drifts[j][0])
    if not eligible:
        raise ArithmeticError('No conditional width remains available to refine')
    increment = max(eligible,
        key=lambda j: (drifts[j][1]-drifts[j][0], -abs(j-stopping), -j))
    return increment, date


def refine_gap(rewards, penalty, centres, brackets, stopping, refine,
               *, target=Fraction(1, 100), maximum_steps=256):
    """Refine a fixed path certificate, with an injectable verified oracle.

    ``refine(j, tolerance)`` returns (lo, hi, work) for the same frozen
    conditional function at the same preceding state. Its returned bracket
    is intersected with the previous bracket; an empty intersection fails.
    Penalty and centres are immutable exact rational tuples throughout.
    """
    target = exact(target)
    if target <= 0 or type(maximum_steps) is not int or maximum_steps < 1:
        raise ValueError('Positive target and maximum refinement count required')
    penalty, centres = tuple(penalty), tuple(centres)
    current = tuple(brackets)
    history = []
    for _ in range(maximum_steps+1):
        drifts = drift_intervals(current, centres)
        certificate = signed_gap(rewards, penalty, drifts, stopping)
        if certificate[1]-certificate[0] <= target:
            return current, certificate, history
        if len(history) >= maximum_steps:
            raise ArithmeticError('Gap target was not reached within the refinement limit')
        increment, date = select_increment(rewards, penalty, drifts, stopping)
        before = current[increment][1]-current[increment][0]
        requested = min(before/4, target/(len(current)-1))
        lo, hi, work = refine(increment, requested)
        tighter = intersect(current[increment], (lo, hi))
        after = tighter[1]-tighter[0]
        if after >= before:
            raise ArithmeticError('A refinement attempt did not narrow its conditional bracket')
        current = (*current[:increment], tighter, *current[increment+1:])
        history.append(dict(increment=increment, upper_date=date,
                            previous_width=float(before), refined_width=float(after),
                            requested_width=float(requested), work=work))
    raise AssertionError('Unreachable refinement state')


def frozen_hash(centres, penalty):
    representation = [[str(x.numerator), str(x.denominator)] for x in (*centres, *penalty)]
    return hashlib.sha256(json.dumps(representation, separators=(',', ':')).encode()).hexdigest()


def work_sum(work):
    return dict(conditional_integrals=sum(x['integrals'] for x in work),
                integrand_evaluations=sum(x['integrand_evaluations'] for x in work))


def summary(rows):
    count = len(rows)
    lo = sum(row['_certificate'][0] for row in rows)/count
    hi = sum(row['_certificate'][1] for row in rows)/count
    widths = [row['_certificate'][1]-row['_certificate'][0] for row in rows]
    midpoints = [float((row['_certificate'][1]+row['_certificate'][0])/2) for row in rows]
    return dict(mean_signed_gap_certificate=interval_record(lo, hi),
                mean_signed_gap_interval=[float(lo), float(hi)],
                mean_signed_width=float(sum(widths)/count),
                maximum_signed_width=float(max(widths)),
                approximate_midpoint_mean=statistics.mean(midpoints),
                approximate_midpoint_mean_standard_error=statistics.stdev(midpoints)/math.sqrt(count) if count > 1 else None,
                midpoint_mean_numerical_error_bound=float(sum(widths)/count/2),
                sampling_scope='Ordinary sample standard error of rounded interval midpoints; approximate statistical diagnostic, not computer-certified coverage.',
                mean_absolute_width=float(sum(row['_certificate'][3]-row['_certificate'][2]
                                               for row in rows)/count),
                conditional_integrals=sum(row['total_work']['conditional_integrals'] for row in rows),
                integrand_evaluations=sum(row['total_work']['integrand_evaluations'] for row in rows),
                refinement_integrals=sum(row['refinement_work']['conditional_integrals'] for row in rows),
                refinement_integrand_evaluations=sum(row['refinement_work']['integrand_evaluations'] for row in rows),
                cpu_seconds_including_shared_coarse=sum(row['cpu_seconds_including_shared_coarse'] for row in rows),
                refinement_cpu_seconds=sum(row['refinement_cpu_seconds'] for row in rows),
                all_targets_met=all(width <= rows[0]['_target'] for width in widths),
                all_signed_inside_absolute=all(a <= l <= h <= b
                    for l, h, a, b in (row['_certificate'] for row in rows)),
                refined_increments=sum(row['refined_increments'] for row in rows))


def compare(record, *, target=Fraction(1, 100), coarse_tolerance=Fraction(1), precision=96,
            include_uniform=True):
    if ctx is None:
        raise RuntimeError('The separate python-flint environment is required')
    target, coarse_tolerance = exact(target), exact(coarse_tolerance)
    if target <= 0 or coarse_tolerance <= 0 or type(precision) is not int or precision < 32:
        raise ValueError('Positive accuracy targets and at least 32 precision bits required')
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    validation_start = time.process_time()
    model, paths, stops, values, times = check_inputs(record)
    policy_checked = check_policy(record)
    validation_cpu = time.process_time()-validation_start
    uniform_tolerance = target/(len(times)-1)
    modes = dict(uniform=[], adaptive=[]) if include_uniform else dict(adaptive=[])
    coarse_integrals = coarse_evaluations = 0
    coarse_cpu = 0.
    for path_index, (states, stopping) in enumerate(zip(paths, stops)):
        reward_start = time.process_time()
        rewards = []
        with ctx.workprec(precision):
            for state, instant in zip(states, times):
                rewards.append(endpoints((-ball(model['rate'])*ball(instant)).exp()
                    *ball(max(max(map(exact, state))-100, Fraction(0)))))
        reward_cpu = time.process_time()-reward_start

        def oracle(j, tolerance):
            return conditional_value(states[j-1], values[j], record['knots'], model,
                                     times[j]-times[j-1], tolerance, precision=precision)

        cpu = time.process_time()
        brackets, centres, penalty, coarse_work = [(Fraction(0), Fraction(0))], [Fraction(0)], [Fraction(0)], []
        for j in range(1, len(times)):
            lo, hi, effort = oracle(j, coarse_tolerance)
            centre = (lo+hi)/2
            brackets.append((lo, hi)); centres.append(centre)
            penalty.append(penalty[-1]+realised_value(states[j], values[j], record['knots'])-centre)
            coarse_work.append(effort)
        brackets, centres, penalty = tuple(brackets), tuple(centres), tuple(penalty)
        fixed_hash = frozen_hash(centres, penalty)
        coarse_elapsed = time.process_time()-cpu
        coarse_cpu += coarse_elapsed
        initial_work = work_sum(coarse_work)
        coarse_integrals += initial_work['conditional_integrals']
        coarse_evaluations += initial_work['integrand_evaluations']

        for mode in modes:
            cpu = time.process_time()
            if mode == 'uniform':
                refined, effort = [brackets[0]], []
                for j in range(1, len(times)):
                    lo, hi, work = oracle(j, uniform_tolerance)
                    refined.append(intersect(brackets[j], (lo, hi)))
                    effort.append(work)
                certificate = signed_gap(rewards, penalty, drift_intervals(refined, centres), stopping)
                history = []
                refined_count = len(times)-1
            else:
                refined, certificate, history = refine_gap(rewards, penalty, centres, brackets,
                    stopping, oracle, target=target)
                effort = [step['work'] for step in history]
                refined_count = len(history)
            elapsed = time.process_time()-cpu
            if frozen_hash(centres, penalty) != fixed_hash:
                raise AssertionError('A strategy changed the frozen centres or penalty')
            lo, hi, alo, ahi = certificate
            if hi-lo > target:
                raise ArithmeticError('A strategy did not attain the path gap target')
            extra = work_sum(effort)
            total = {key: initial_work[key]+extra[key] for key in initial_work}
            modes[mode].append(dict(path_index=path_index, stopping_index=stopping,
                signed_gap_certificate=interval_record(lo, hi), absolute_gap_certificate=interval_record(alo, ahi),
                signed_width=float(hi-lo), coarse_work=initial_work, refinement_work=extra, total_work=total,
                coarse_cpu_seconds=coarse_elapsed, refinement_cpu_seconds=elapsed,
                reward_cpu_seconds=reward_cpu,
                cpu_seconds_including_shared_coarse=coarse_elapsed+elapsed,
                frozen_centres_and_penalty_sha256=fixed_hash, refined_increments=refined_count,
                refinement_history=history, _certificate=certificate, _target=target))
        # Both methods enclose the identical ideal martingale path gap.
        if include_uniform:
            intersect(modes['uniform'][-1]['_certificate'][:2], modes['adaptive'][-1]['_certificate'][:2])

    result = dict(schema_version=1, author='Amanjeet Singh', outer_paths=len(paths),
        assessed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Ideal conditional functions and real gain algebra at exact binary input states; neither simulator law nor sampling coverage is certified.',
        observed_policy_decisions_checked=policy_checked,
        input_and_policy_validation_cpu_seconds=validation_cpu,
        frozen_value_function_and_policy=True, fixed_predictable_coarse_centres=True,
        coarse_width_target=interval_record(coarse_tolerance, coarse_tolerance),
        path_gap_width_target=interval_record(target, target),
        uniform_increment_width_target=interval_record(uniform_tolerance, uniform_tolerance), precision_bits=precision,
        selection_rule='Largest conditional width between stopping and current maximum upper date; ties nearest stopping then smallest index.',
        accounting='Each strategy is charged every coarse integral plus its own refined integrals. Shared coarse CPU is included in each strategy total; actual experiment CPU counts it once. Uniform refinement is also reported separately.',
        float_endpoint_semantics='Rounded display only; rational certificate endpoints are authoritative.',
        coarse_work=dict(conditional_integrals=coarse_integrals, integrand_evaluations=coarse_evaluations,
                         cpu_seconds=coarse_cpu),
        strategies={mode: summary(rows) for mode, rows in modes.items()},
        all_method_certificates_overlap=True,
        actual_experiment_cpu_seconds=time.process_time()-cpu_start,
        actual_experiment_wall_seconds=time.perf_counter()-wall_start)
    for rows in modes.values():
        for row in rows:
            del row['_certificate'], row['_target']
    result['observations'] = modes
    return result


def cold_uniform(record, *, target=Fraction(1, 100), precision=96):
    """Full uniform certificate from fine brackets, with no coarse phase.

    Its fine predictable centres differ from the adaptive coarse centres.
    Algebraically both compensate the same v_j(S_j)-E[v_j(S_j)|S_{j-1}]
    increments, hence enclose the same ideal path gap.
    """
    if ctx is None:
        raise RuntimeError('The separate python-flint environment is required')
    target = exact(target)
    if target <= 0 or type(precision) is not int or precision < 32:
        raise ValueError('Positive accuracy target and at least 32 precision bits required')
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    validation_start = time.process_time()
    model, paths, stops, values, times = check_inputs(record)
    policy_checked = check_policy(record)
    validation_cpu = time.process_time()-validation_start
    tolerance = target/(len(times)-1)
    rows = []
    for path_index, (states, stopping) in enumerate(zip(paths, stops)):
        cpu = time.process_time()
        rewards = []
        with ctx.workprec(precision):
            for state, instant in zip(states, times):
                rewards.append(endpoints((-ball(model['rate'])*ball(instant)).exp()
                    *ball(max(max(map(exact, state))-100, Fraction(0)))))
        penalty, drifts, centres = [Fraction(0)], [(Fraction(0), Fraction(0))], [Fraction(0)]
        effort = []
        for j in range(1, len(times)):
            lo, hi, work = conditional_value(states[j-1], values[j], record['knots'], model,
                times[j]-times[j-1], tolerance, precision=precision)
            centre = (lo+hi)/2
            centres.append(centre)
            penalty.append(penalty[-1]+realised_value(states[j], values[j], record['knots'])-centre)
            drifts.append((lo-centre, hi-centre))
            effort.append(work)
        certificate = signed_gap(rewards, penalty, drifts, stopping)
        elapsed = time.process_time()-cpu
        lo, hi, alo, ahi = certificate
        if hi-lo > target:
            raise ArithmeticError('Cold uniform did not attain the path gap target')
        total = work_sum(effort)
        rows.append(dict(path_index=path_index, stopping_index=stopping,
            signed_gap_certificate=interval_record(lo, hi), absolute_gap_certificate=interval_record(alo, ahi),
            signed_width=float(hi-lo), coarse_work=work_sum([]), refinement_work=total, total_work=total,
            coarse_cpu_seconds=0., refinement_cpu_seconds=elapsed,
            cpu_seconds_including_shared_coarse=elapsed,
            frozen_centres_and_penalty_sha256=frozen_hash(centres, penalty), refined_increments=len(times)-1,
            _certificate=certificate, _target=target))
    aggregate = summary(rows)
    for row in rows:
        del row['_certificate'], row['_target']
    return dict(schema_version=1, author='Amanjeet Singh', outer_paths=len(paths),
        assessed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Ideal conditional functions and real gap algebra at exact binary input states; neither simulator law nor sampling coverage is certified.',
        observed_policy_decisions_checked=policy_checked,
        input_and_policy_validation_cpu_seconds=validation_cpu,
        coarse_initialisation_performed=False,
        path_gap_width_target=interval_record(target, target),
        uniform_increment_width_target=interval_record(tolerance, tolerance), precision_bits=precision,
        float_endpoint_semantics='Rounded display only; rational certificate endpoints are authoritative.',
        strategy=aggregate, observations=rows,
        actual_assessment_cpu_seconds=time.process_time()-cpu_start,
        actual_assessment_wall_seconds=time.perf_counter()-wall_start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=str, default='1/100')
    parser.add_argument('--coarse-tolerance', type=str, default='1')
    parser.add_argument('--precision', type=int, default=96)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/'data/private').resolve()) or destination.exists():
        raise ValueError('Use a new private output file')
    source = args.input.read_bytes()
    result = compare(json.loads(source), target=Fraction(args.target),
                     coarse_tolerance=Fraction(args.coarse_tolerance), precision=args.precision)
    result['input_sha256'] = hashlib.sha256(source).hexdigest()
    result['source_sha256'] = {name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
        for name in ('refine_multivariate_gap.py', 'certify_multivariate_gap.py', 'certify_max_hinge.py')}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved fixed-centre gap refinement comparison to {destination}', flush=True)


if __name__ == '__main__':
    main()
