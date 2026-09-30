"""Measure retrospective bounded-gain interval widths without new simulation.

The mathematical envelope applies to exact conditional integration of the
retained constant-tail put functions. Floating-point reconstruction is not an
interval-arithmetic certificate. All numerical outputs remain private because
the envelopes contain contract and funding quantities.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import product
import math
import os
from pathlib import Path
import platform

import numpy as np

from audit_training_allocation_main import verify as verify_main
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_training_allocation import THREADS, checked
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import build_approximation


MAIN = ROOT / 'data/private/training_allocation/2026-09-27/main_run_v1'
REFERENCE_REPETITIONS = 64
ALPHA = .05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_positive(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0, f'{name} must be finite and positive')
    return value


def positive_count(value, name):
    require(type(value) is int and value > 0, f'{name} must be a positive integer')
    return value


def mixture_halfwidth(bound, repetitions, *, alpha=ALPHA,
                      reference_repetitions=REFERENCE_REPETITIONS):
    """Evaluate the two-sided Gaussian-mixture boundary with rho=L^2 R0.

    This algebraic form avoids squaring the envelope in the width calculation.
    One alpha already covers both signs through the squared mixture statistic.
    """
    finite_positive(bound, 'gain envelope')
    repetitions = positive_count(repetitions, 'repetitions')
    reference_repetitions = positive_count(reference_repetitions, 'reference count')
    require(not isinstance(alpha, bool) and isinstance(alpha, (int, float))
            and math.isfinite(alpha) and 0 < alpha < 1, 'alpha must lie between zero and one')
    combined = reference_repetitions + repetitions
    width = bound * math.sqrt(combined * (
        2 * math.log(1 / alpha) + math.log1p(repetitions / reference_repetitions))) / repetitions
    finite_positive(width, 'mixture half-width')
    return width


def exact_fixtures():
    """Enumerate finite stipulated laws, using no random-number generator."""
    stopped = []
    for first, second in product((-1, 1), repeat=2):
        count = 1 if first == 1 else 2
        total = first if count == 1 else first + second
        stopped.append(dict(signs=[first, second], count=count, total=total,
                            mean=Fraction(total, count), probability=Fraction(1, 4)))
    stopped_sum = sum((x['probability'] * x['total'] for x in stopped), Fraction())
    stopped_mean = sum((x['probability'] * x['mean'] for x in stopped), Fraction())
    require(stopped_sum == 0 and stopped_mean == Fraction(1, 4), 'CPU-stop enumeration differs')

    branches = []
    for repetitions, paths in ((2, 1), (3, 2)):
        count = repetitions * (paths + 1)
        outcomes = []
        for signs in product((-1, 1), repeat=count):
            rows = []
            for r in range(repetitions):
                begin = r * (paths + 1)
                training = signs[begin]
                evaluation = signs[begin + 1:begin + 1 + paths]
                rows.append(1 + training + Fraction(2 * sum(evaluation), paths))
            outcomes.append(sum(rows, Fraction()) / repetitions)
        mean = sum(outcomes, Fraction()) / len(outcomes)
        variance = sum(((value - mean) ** 2 for value in outcomes), Fraction()) / len(outcomes)
        formula = Fraction(1, repetitions) + Fraction(4, repetitions * paths)
        require(mean == 1 and variance == formula, 'Random-pilot conditional enumeration differs')
        branches.append(dict(repetitions=repetitions, evaluation_paths=paths,
            probability=Fraction(1, 2), enumerated_states=len(outcomes), mean=mean,
            variance=variance))
    exact_variance = sum((x['probability'] * x['variance'] for x in branches), Fraction())
    mean_r = sum((x['probability'] * x['repetitions'] for x in branches), Fraction())
    mean_rn = sum((x['probability'] * x['repetitions'] * x['evaluation_paths'] for x in branches), Fraction())
    substitution = 1 / mean_r + 4 / mean_rn
    require(exact_variance == Fraction(7, 4) and substitution == Fraction(7, 5),
            'Inverse-count and mean-count comparison differs')

    # Inversion at the boundary must give the single two-sided crossing level.
    width = mixture_halfwidth(2., 64)
    rho, variance_clock = 2. ** 2 * 64, 2. ** 2 * 64
    log_mixture = .5 * math.log(rho / (rho + variance_clock)) + (64 * width) ** 2 / (2 * (rho + variance_clock))
    require(math.isclose(log_mixture, math.log(1 / ALPHA), rel_tol=1e-13),
            'Gaussian-mixture inversion constant differs')
    for count in (True, 0, -1, 2.5, math.inf):
        try:
            mixture_halfwidth(1., count)
        except ValueError:
            pass
        else:
            raise ValueError('An invalid repetition count was accepted')

    def serialise(value):
        if isinstance(value, Fraction):
            return str(value)
        if isinstance(value, list):
            return [serialise(x) for x in value]
        if isinstance(value, dict):
            return {key: serialise(x) for key, x in value.items()}
        return value
    return serialise(dict(
        classification='Exact enumeration of stipulated probability laws, not market observations',
        cpu_stopping=dict(cost_rule='C_i=(3+X_i)/2; stop on reaching two cost units',
            rows=stopped, expected_stopped_sum=stopped_sum, expected_stopped_mean=stopped_mean),
        random_pilot=dict(gain_rule='Z=1+D+epsilon, D is an equiprobable sign, epsilon is an independent equiprobable signed two',
            training_variance=1, evaluation_variance=4, branches=branches,
            unconditional_variance=exact_variance, inverse_mean_count_substitution=substitution,
            expected_R=mean_r, expected_RN=mean_rn),
        mixture_boundary_single_alpha_checked=True, invalid_counts_rejected=True))


def gain_envelope(approximation):
    """Bound precisely the interpolants used by value_martingale's increments."""
    require(approximation.option.kind == 'put', 'The bounded construction requires a put')
    nodes = approximation.nodes
    require(nodes and nodes[0].time == 0., 'Value construction must start at time zero')
    require(all(right.time > 0 for right in approximation.decisions),
            'The retained exercise calendar must exclude time zero')
    previous = 0.
    increments = []
    for node in nodes:
        if node.time == 0.:
            continue
        require(math.isfinite(node.time) and node.time > previous,
                'Propagation nodes must be finite and increase strictly')
        value = node.before if node.before is not None else node.after
        require(value.tail_slope == 0., 'A non-constant upper tail invalidates the bounded envelope')
        require(value.knots[0] == 0. and all(math.isfinite(x) for x in value.values),
                'Require finite interpolation values on the non-negative state domain')
        lower, upper = min(value.values), max(value.values)
        oscillation = upper - lower
        require(math.isfinite(oscillation) and oscillation >= 0, 'Invalid interpolant oscillation')
        increments.append(dict(time=node.time, previous_time=previous,
            endpoint_phase='before' if node.before is not None else 'after',
            knots=len(value.knots), minimum=lower, maximum=upper, oscillation=oscillation,
            tail_slope=value.tail_slope, deterministic_cash_jump_increment=0.))
        previous = node.time
    require(previous == approximation.option.maturity, 'The final value node must be expiry')
    oscillation_sum = math.fsum(row['oscillation'] for row in increments)
    maximum_reward = max(approximation.option.strike * right.settlement_factor
                         for right in approximation.decisions)
    factor = finite_positive(approximation.model.premium_factor, 'premium factor')
    bound = finite_positive((maximum_reward + oscillation_sum) / factor, 'gain envelope')
    target_bound = finite_positive(maximum_reward / factor, 'target mean bound')
    return dict(approximation_fingerprint=approximation.fingerprint,
        stochastic_transition_nodes=len(increments), increments=increments,
        martingale_oscillation_sum=oscillation_sum, maximum_discounted_put_reward=maximum_reward,
        premium_factor=factor, gain_absolute_bound=bound, mean_gain_absolute_bound=target_bound,
        mathematical_basis='Put rewards lie in [0,H*]; the martingale difference between stopping indices is bounded by the sum of interpolant oscillations',
        exact_integration_required=True, floating_point_bound_certified=False)


def report(summary):
    rows = summary['arms']
    lines = [
        '# Retrospective width of bounded-gain sequential intervals', '',
        '## 1. Question and interpretation', '',
        'I evaluate whether the conservative Gaussian-mixture construction in the conditional-allocation note would give useful interval widths at the retained allocation counts. I reconstruct the thirteen deterministic put value approximations and use the already completed training-pair estimates. No policy is refitted and no Monte Carlo path is generated. This is a retrospective width diagnostic, not a prospectively specified coverage experiment or a replacement for the retained Student intervals.', '',
        'The calculation uses alpha = 0.05 and the illustrative tuning choice rho = L² times 64, where 64 is the original reference pair count. I apply the same rule in every case and both arms. Its selection is retrospective; I do not describe it as pre-specified in the original experiment. The time-uniform mathematical construction applies to each fixed financial specification separately. No simultaneous confidence claim across the thirteen models or twenty-six arms is made.', '',
        '## 2. Mathematical envelope and numerical scope', '',
        'I follow the stochastic increments in the retained value-martingale implementation, using the before-event interpolant at a cash event and the after-event interpolant otherwise. Deterministic cash transitions add no martingale increment. Every retained function has constant tails. Its finite knot-value range therefore bounds its full-support value, and the sum of these oscillations bounds the difference between any two stopped martingale values. Adding the maximum put reward and dividing by the premium-payment factor gives the pathwise adjusted-gain envelope L.', '',
        'The proof concerns exact conditional integration of the retained functions. Their reconstruction and the width calculation use floating-point arithmetic and do not provide a certified rounding-error enclosure. Matching fingerprints verifies that the reconstructed functions coincide with the retained numerical constructions; it does not certify the martingale centring in exact arithmetic.', '',
        'The target is a difference between two expected put rewards, so its absolute value is also bounded by the maximum put reward in quoted premium units. I intersect the illustrative interval with that narrower deterministic target range. Exact envelopes and target ranges contain contract and funding quantities and remain private.', '',
        '## 3. Widths at the retained counts', '',
        '| Case | Arm | Pairs | Evaluation paths | Mean gain | Student half-width | Mixture half-width | Width ratio | Entire target range retained |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        lines.append(f"| {row['case']} | {row['arm']} | {row['repetitions']} | {row['evaluation_paths']} | {row['mean_gain']:.6f} | {row['student_halfwidth']:.6f} | {row['mixture_halfwidth']:.3f} | {row['mixture_to_student_halfwidth_ratio']:.0f} | {row['full_target_range_retained']} |")
    aggregate = summary['aggregate']
    lines += ['',
        f"The mixture half-widths range from {aggregate['mixture_halfwidth_range'][0]:.3f} to {aggregate['mixture_halfwidth_range'][1]:.3f} USD per share in premium-quote units. Their ratios to the retained Student half-widths range from {aggregate['width_ratio_range'][0]:.0f} to {aggregate['width_ratio_range'][1]:.0f}. All {aggregate['arms']} intervals include zero, and {aggregate['entire_target_range_retained']} retain the entire deterministic range of the target after intersection.", '',
        'These widths show the cost of using a global pathwise envelope that accumulates full-support value-function oscillations. The calculation establishes that this particular bound is uninformative at the retained counts. It does not show that every sequential construction would be uninformative, that the Student intervals have their stated finite-sample coverage, or that the underlying mean gains are zero. A sharper conditional-variance or self-normalised construction would require its own checked assumptions and practical validation.', '',
        '## 4. Exact finite probability examples', '',
        'I enumerate the CPU-stopping example with exact rational arithmetic. The stopped sum has expectation zero, while its outcome-dependent sample mean has expectation one quarter. This separates optional sampling for the sum from unbiasedness of its random-denominator average.', '',
        'A second stipulated model uses Z = 1 + D + epsilon, with independent symmetric D in {-1,1} and epsilon in {-2,2}. An independent pilot chooses (R,N) = (2,1) or (3,2), each with probability one half. Exact enumeration gives mean one in both branches and unconditional estimator variance 7/4. Substituting the arithmetic mean counts gives 1/E[R] + 4/E[RN] = 7/5 instead. The discrepancy illustrates why the conditional variance identity must be averaged using inverse counts. These finite examples are mathematical fixtures and contain no fabricated market observations.', '',
        '## 5. Evidence', '',
        'The diagnostic checks the completed allocation evidence, rebuilds every deterministic approximation and verifies all twenty-six retained fingerprints. The summary, per-node oscillations, exact examples and manifest are retained together. Existing experiment records, source files and figure assets are checked before and after the calculation. No original numerical result is overwritten. The complete output remains private because its envelopes and target ranges can disclose financial inputs.', '',
    ]
    return '\n'.join(lines)


def run(main, output):
    main, output = private(main), private(output)
    require(not output.exists(), 'Use a fresh diagnostic output directory')
    require(all(os.environ.get(key) == '1' for key in THREADS),
            'Use the recorded numerical-library thread settings')
    fixtures = exact_fixtures()
    manifest, cfg, tasks, reconstructed = verify_main(main)
    require(cfg['reference_repetitions'] == REFERENCE_REPETITIONS, 'Reference pair count differs')
    require(cfg['approximation'] == dict(core_steps=800, wing_steps=40, tolerance=.001),
            'Retained fine approximation differs')
    require([task['case'] for task in tasks] == list(range(13)), 'Thirteen ordered cases are required')
    registration = ROOT / manifest['registration']
    inputs = dict(manifest['input_sha256'])
    for path in (main / 'manifest.json', main / 'summary.json', registration / 'manifest.json',
                 registration / 'tasks.json', registration / 'allocations.json'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    sources = dict(manifest['source_sha256'])
    for relative in ('scripts/check_allocation_inference.py', 'scripts/audit_training_allocation.py',
                     'scripts/audit_training_allocation_main.py'):
        sources[relative] = sha(ROOT / relative)
    # The diagnostic neither regenerates nor alters any existing figure.
    protected_figures = {}
    for folder in (main.parent / 'main_report_v1',
                   ROOT / 'data/private/policy_paper/2026-09-27/full_latex_v7/figures'):
        require(folder.is_dir(), f'Missing retained figure directory: {folder.name}')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                protected_figures[str(path.relative_to(ROOT))] = sha(path)
    require(protected_figures, 'Retained figure assets are required')
    verify_mapping(inputs)
    verify_mapping(sources)
    verify_mapping(protected_figures)
    output.mkdir(parents=True)
    try:
        save(output / 'exact_fixtures.json', fixtures)
        models, arms = [], []
        for task, model in zip(tasks, reconstructed['models']):
            require(task['case'] == model['case'], 'Reconstructed case identity differs')
            base = load_frozen(task['source_policy'])[0]
            approximation, setup = build_approximation(base, **cfg['approximation'])
            envelope = gain_envelope(approximation)
            envelope['case'] = task['case']
            envelope['reconstruction_setup'] = setup
            L = envelope['gain_absolute_bound']
            target_bound = envelope['mean_gain_absolute_bound']
            for arm in ('reference', 'selected'):
                retained = model['arms'][arm]
                require(retained['setup']['approximation_fingerprint'] == approximation.fingerprint,
                        'Reconstructed approximation differs from a retained arm')
                allocation, estimate = retained['allocation'], retained['inference']['estimate']
                count = positive_count(allocation['repetitions'], 'retained pair count')
                paths = positive_count(allocation['evaluation_paths'], 'retained evaluation count')
                require(estimate['observations'] == count and estimate['confidence'] == .95,
                        'Retained inference count or confidence differs')
                width = mixture_halfwidth(L, count)
                mean = estimate['value']
                student_width = finite_positive((estimate['upper'] - estimate['lower']) / 2,
                                                'retained Student half-width')
                require(math.isclose((estimate['upper'] + estimate['lower']) / 2, mean,
                                     rel_tol=1e-12, abs_tol=1e-14), 'Student interval is not centred')
                require(math.isfinite(mean) and abs(mean) <= L, 'Observed pair mean exceeds the envelope')
                lower, upper = mean - width, mean + width
                clipped = [max(lower, -target_bound), min(upper, target_bound)]
                require(clipped[0] <= clipped[1], 'Illustrative interval has empty target intersection')
                arms.append(dict(case=task['case'], arm=arm, repetitions=count,
                    evaluation_paths=paths, mean_gain=mean, student_halfwidth=student_width,
                    mixture_halfwidth=width, mixture_to_student_halfwidth_ratio=width / student_width,
                    alpha=ALPHA, mixture_precision=L * L * REFERENCE_REPETITIONS,
                    untruncated_interval=[lower, upper], deterministic_target_range=[-target_bound, target_bound],
                    target_intersected_interval=clipped,
                    full_target_range_retained=lower <= -target_bound and upper >= target_bound,
                    contains_zero=clipped[0] <= 0 <= clipped[1]))
            models.append(envelope)
            print(f'Checked case {task["case"]}: deterministic approximation and both retained arms', flush=True)
        require(len(models) == 13 and len(arms) == 26, 'Incomplete diagnostic coverage')
        aggregate = dict(cases=len(models), arms=len(arms),
            retained_pairs=sum(row['repetitions'] for row in arms),
            mixture_halfwidth_range=[min(x['mixture_halfwidth'] for x in arms), max(x['mixture_halfwidth'] for x in arms)],
            student_halfwidth_range=[min(x['student_halfwidth'] for x in arms), max(x['student_halfwidth'] for x in arms)],
            width_ratio_range=[min(x['mixture_to_student_halfwidth_ratio'] for x in arms), max(x['mixture_to_student_halfwidth_ratio'] for x in arms)],
            intervals_containing_zero=sum(x['contains_zero'] for x in arms),
            entire_target_range_retained=sum(x['full_target_range_retained'] for x in arms))
        require(aggregate['retained_pairs'] == 5030, 'Retained pair denominator differs')
        summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
            scope='Retrospective mathematical width diagnostic using retained mean estimates; no new simulation',
            alpha=ALPHA, reference_repetitions=REFERENCE_REPETITIONS,
            tuning='rho=L^2 times the original reference count 64; illustrative retrospective choice',
            models=models, arms=arms, aggregate=aggregate,
            source_mathematical_envelope=True, floating_point_bound_certified=False,
            prospectively_registered_interval=False, interval_coverage_validated=False,
            simultaneous_inference_across_models=False, monte_carlo_paths_generated=0,
            policies_fitted=0, deterministic_approximations_rebuilt=13,
            publication_clearance=False)
        verify_mapping(inputs)
        verify_mapping(sources)
        verify_mapping(protected_figures)
        checked(main)
        save(output / 'summary.json', summary)
        (output / 'findings.md').write_text(report(summary))
        artifacts = {str(path.relative_to(output)): sha(path)
                     for path in sorted(output.rglob('*')) if path.is_file()}
        save(output / 'manifest.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
            complete=True, input_sha256=inputs, source_sha256=sources,
            protected_figure_artifact_sha256=protected_figures, artifact_sha256=artifacts,
            python=platform.python_version(), numpy=np.__version__,
            thread_environment={key: os.environ[key] for key in THREADS},
            no_original_evidence_modified=True, publication_clearance=False))
        return aggregate
    except Exception as exc:
        save(output / 'incomplete.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            complete=False, exception_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=MAIN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixtures-only', action='store_true')
    args = parser.parse_args()
    if args.fixtures_only:
        exact_fixtures()
        print('Exact finite-law enumeration and mixture-boundary checks passed; no market data read.')
    else:
        if args.output is None:
            parser.error('--output is required unless --fixtures-only is supplied')
        result = run(args.main, args.output)
        print(f'Completed {result["cases"]} cases and {result["arms"]} retrospective width comparisons; all outputs remain private.')
