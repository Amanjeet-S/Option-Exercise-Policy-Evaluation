"""Reassess retained interval widths using full-domain backward residuals.

This is a retrospective mathematical/numerical diagnostic. It generates no
simulation paths, fits no policy and does not establish empirical coverage.
Exact financial inputs and their reversible summaries remain private.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from audit_training_allocation_main import verify as verify_main
from check_allocation_inference import MAIN, gain_envelope, mixture_halfwidth
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_training_allocation import THREADS, checked
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import build_approximation
from qf_research.optimal_stopping.residual_envelope import stopping_gain_envelope
from qf_research.optimal_stopping.sequential_inference import empirical_bernstein_sequence


PREVIOUS = ROOT / 'data/private/cpu_budget_transfer/2026-09-27/inference_diagnostic_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def report(summary):
    a = summary['aggregate']
    lines = ['# Residual bounds and variance-adaptive sequential inference', '',
        '## 1. Research question', '',
        'I test whether preserving the cancellation between successive approximate values improves the previously uninformative sequential intervals. I apply the backward-residual identity to the same thirteen retained put constructions and replay the ordered pair means from the completed allocation study. The financial specifications, training procedures, evaluation counts and original estimates remain fixed. No policy is refitted and no simulation path is generated.', '',
        'The mathematical envelope covers all exercise rights and the entire non-negative state domain. Continuation residuals include the economic early-exercise compensator as well as approximation effects. The calculation does not identify all residuals as numerical error. Cash-event edges are represented explicitly, and a common history cancels between two stopping times.', '',
        '## 2. Range computation and inference', '',
        'I calculate the residual envelope at tolerances 0.01 and 0.001 in discounted reward units. Monotone interval bounds and a curvature bound cover finite state cells; a separate constant-tail argument covers the unbounded remainder. Cash residuals and exercise defects use the union of their piecewise-linear knots. These are numerical evaluations of exact mathematical enclosures. Normal distribution evaluations and arithmetic are not outward-rounded, so this diagnostic does not certify every floating-point error.', '',
        'I retain the earlier Gaussian-mixture choice rho = 64 times the squared envelope. I also apply the empirical-Bernstein construction in Howard et al. (2021), Theorem 4 and equation (4.1), with the stitched boundary in Theorem 1 and equation (3.1). Each arm uses confidence 0.95, hence one-sided crossing probability 0.025. The support diameter is twice the residual envelope. The stitching choices are eta = 2, s = 1.4 and m equal to the squared support diameter. The first predictor is zero; subsequent predictors are preceding-pair means. The current pair never enters its own predictor.', '',
        'The squared prediction errors are calculated from completed training-pair means, not from within-pair standard errors. This preserves the distinction between uncertainty about a training procedure and uncertainty conditional on a fitted policy. The settings are retrospective diagnostic choices. No prospective coverage validation, joint coverage across arms or superiority after selecting the narrower construction is claimed.', '',
        '## 3. Findings', '',
        f"The residual envelope is smaller than the original oscillation envelope by factors from {a['envelope_reduction_range'][0]:.1f} to {a['envelope_reduction_range'][1]:.1f}. All thirteen fine calculations meet their declared optimisation tolerance. Refining the tolerance changes the envelope by at most {a['maximum_relative_envelope_change']:.6f} in relative terms. This is a refinement comparison, not a certificate for rounding or approximation error.", '',
        '| Case | Arm | Pairs | Student half-width | Residual mixture half-width | Empirical-Bernstein half-width | Empirical-Bernstein interval contains zero |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for r in summary['arms']:
        lines.append(f"| {r['case']} | {r['arm']} | {r['repetitions']} | {r['student_halfwidth']:.6f} | {r['residual_mixture_halfwidth']:.6f} | {r['empirical_bernstein_halfwidth']:.6f} | {r['empirical_bernstein_contains_zero']} |")
    lines += ['',
        f"The residual Gaussian-mixture half-widths range from {a['mixture_halfwidth_range'][0]:.6f} to {a['mixture_halfwidth_range'][1]:.6f} USD per share in premium-quote units. The empirical-Bernstein half-widths range from {a['empirical_bernstein_halfwidth_range'][0]:.6f} to {a['empirical_bernstein_halfwidth_range'][1]:.6f}. The respective numbers of final intervals containing zero are {a['mixture_intervals_containing_zero']} and {a['empirical_bernstein_intervals_containing_zero']} out of 26.", '',
        'A smaller envelope is a substantive numerical improvement, but the inference question is whether the remaining width resolves the training gain. The retained Student intervals remain asymptotic comparisons; their shorter widths do not establish finite-sample coverage. Conversely, a conservative interval containing zero does not establish a zero effect.', '',
        '## 4. Scope and next question', '',
        'The residual support is uniform over possible training realisations. The empirical-Bernstein construction can exploit small observed variability, but its finite-count width also depends on the deterministic support diameter. That term can remain large relative to the mean gain. Further tightening would require a justified restriction on admissible stopping decisions, a more informative conditional envelope, or additional independent observations under a prospectively specified design.', '',
        'Exact financial quantities and envelope decompositions remain private. The input manifest verifies the completed allocation evidence and the preceding diagnostic. Approximation fingerprints match both retained arms. Every retained mean is read in its recorded repetition order and reproduces the existing final mean. No simulation, timing experiment or original evidence file is changed.', '',
        '## 5. Sources', '',
        'The [residual derivation](../residual_theory_v1/residual_envelope.md) attributes the approximate-value and residual-reward representation to Haugh and Kogan (2004), equations (8)–(10), and derives the cash-phase and pairwise range calculation. Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021), [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://doi.org/10.1214/20-AOS1991), *The Annals of Statistics*, 49(2), 1055–1080, supplies the sequential boundaries. The [dated reading record](../../../literature/residual_envelope_2026-09-27/source_record.json) specifies the published sections consulted. No new confidence-sequence theorem or general methodological priority is claimed.', '']
    return '\n'.join(lines)


def run(main, output):
    main, output = private(main), private(output)
    require(not output.exists(), 'Use a fresh output directory')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'Recorded thread settings are required')
    manifest, cfg, tasks, reconstructed = verify_main(main)
    old_manifest = read(PREVIOUS / 'manifest.json')
    for key in ('input_sha256', 'source_sha256', 'protected_figure_artifact_sha256'):
        verify_mapping(old_manifest[key])
    verify_mapping({str((PREVIOUS/name).relative_to(ROOT)): digest
                    for name, digest in old_manifest['artifact_sha256'].items()})
    inputs = dict(manifest['input_sha256'])
    inputs.update({str((main/name).relative_to(ROOT)): digest
                   for name, digest in manifest['artifact_sha256'].items()})
    for path in (main/'manifest.json', PREVIOUS/'manifest.json', PREVIOUS/'summary.json',
                 ROOT/'data/private/literature/residual_envelope_2026-09-27/source_record.json'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    sources = dict(old_manifest['source_sha256'])
    for name in ('scripts/check_residual_inference.py',
                 'src/qf_research/optimal_stopping/residual_envelope.py',
                 'src/qf_research/optimal_stopping/sequential_inference.py',
                 'tests/test_residual_envelope.py', 'tests/test_sequential_inference.py'):
        sources[name] = sha(ROOT/name)
    verify_mapping(inputs)
    verify_mapping(sources)
    output.mkdir(parents=True)
    try:
        models, arms, sequences, certification_cases = [], [], [], []
        for task, model in zip(tasks, reconstructed['models']):
            case = task['case']
            require(case == model['case'], 'Case identity differs')
            approximation, _ = build_approximation(load_frozen(task['source_policy'])[0], **cfg['approximation'])
            coarse = stopping_gain_envelope(approximation, tolerance=.01)
            fine = stopping_gain_envelope(approximation, tolerance=.001)
            require(coarse['all_refinements_converged'] and fine['all_refinements_converged'],
                    'A state-domain refinement did not converge')
            original = gain_envelope(approximation)
            bound = fine['gain_absolute_bound']
            require(math.isfinite(bound) and 0 < bound < original['gain_absolute_bound'],
                    'A positive improving residual envelope is required')
            model_row = dict(case=case, coarse=coarse, fine=fine,
                previous_envelope=original['gain_absolute_bound'],
                reduction_factor=original['gain_absolute_bound']/bound,
                relative_refinement_change=abs(bound-coarse['gain_absolute_bound'])/bound)
            models.append(model_row)
            # Selected primitive checks use no simulated state. Every coordinate
            # is a retained knot or the retained initial state, including zero.
            for node_index in (1, len(approximation.nodes)//2, len(approximation.nodes)-1):
                node, previous = approximation.nodes[node_index], approximation.nodes[node_index-1]
                value = node.before if node.before is not None else node.after
                sd = approximation.model.volatility*math.sqrt(node.time-previous.time)
                states = sorted({0., approximation.model.initial_state,
                                 value.knots[len(value.knots)//2], value.knots[-1]})
                certification_cases.append(dict(case=case, node_index=node_index,
                    knots=list(value.knots), values=list(value.values), tail_slope=value.tail_slope,
                    sd=sd, states=states, scipy_expectations=value.expectation(np.array(states), sd).tolist()))
            for arm in ('reference', 'selected'):
                retained = model['arms'][arm]
                require(retained['setup']['approximation_fingerprint'] == approximation.fingerprint,
                        'Retained approximation fingerprint differs')
                count = retained['allocation']['repetitions']
                values = [read(main/'blocks'/f'case_{case:03d}_{arm}_rep_{r:04d}.json')['evaluation']['gain']['estimate']['value']
                          for r in range(count)]
                sequence = empirical_bernstein_sequence(values, -bound, bound)
                last = sequence['rows'][-1]
                estimate = retained['inference']['estimate']
                require(math.isclose(last['mean'], estimate['value'], rel_tol=1e-12, abs_tol=1e-12),
                        'Ordered pair means do not reconstruct the retained mean')
                mixture = mixture_halfwidth(bound, count)
                mean = estimate['value']
                arms.append(dict(case=case, arm=arm, repetitions=count,
                    evaluation_paths=retained['allocation']['evaluation_paths'], mean_gain=mean,
                    student_halfwidth=(estimate['upper']-estimate['lower'])/2,
                    original_mixture_halfwidth=mixture_halfwidth(original['gain_absolute_bound'], count),
                    residual_mixture_halfwidth=mixture,
                    residual_mixture_contains_zero=mean-mixture <= 0 <= mean+mixture,
                    empirical_bernstein_halfwidth=last['halfwidth'],
                    empirical_bernstein_contains_zero=last['lower'] <= 0 <= last['upper'],
                    empirical_bernstein_intrinsic_variance=last['intrinsic_variance'],
                    empirical_bernstein_variance_floor=sequence['settings']['m'],
                    empirical_bernstein_interval=[last['lower'], last['upper']]))
                sequences.append(dict(case=case, arm=arm, **sequence))
            print(f'Case {case}: refined global residual envelope and replayed both arms', flush=True)
        require(len(models) == 13 and len(arms) == 26, 'Incomplete retained comparison')
        aggregate = dict(cases=13, arms=26, retained_pairs=sum(r['repetitions'] for r in arms),
            envelope_reduction_range=[min(r['reduction_factor'] for r in models),max(r['reduction_factor'] for r in models)],
            maximum_relative_envelope_change=max(r['relative_refinement_change'] for r in models),
            mixture_halfwidth_range=[min(r['residual_mixture_halfwidth'] for r in arms),max(r['residual_mixture_halfwidth'] for r in arms)],
            empirical_bernstein_halfwidth_range=[min(r['empirical_bernstein_halfwidth'] for r in arms),max(r['empirical_bernstein_halfwidth'] for r in arms)],
            mixture_intervals_containing_zero=sum(r['residual_mixture_contains_zero'] for r in arms),
            empirical_bernstein_intervals_containing_zero=sum(r['empirical_bernstein_contains_zero'] for r in arms))
        require(aggregate['retained_pairs'] == 5030, 'Retained pair count differs')
        summary = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), models=models,
            arms=arms, aggregate=aggregate, floating_point_bound_certified=False,
            prospectively_registered_interval=False, interval_coverage_validated=False,
            simultaneous_inference_across_models=False, monte_carlo_paths_generated=0,
            policies_fitted=0, publication_clearance=False)
        save(output/'summary.json', summary)
        save(output/'sequences.json', sequences)
        save(output/'certification_inputs.json', dict(scope='Selected conditional-expectation primitive checks, not a global certificate', cases=certification_cases))
        (output/'findings.md').write_text(report(summary))
        verify_mapping(inputs)
        verify_mapping(sources)
        verify_mapping(old_manifest['protected_figure_artifact_sha256'])
        checked(main)
        save(output/'manifest.json', dict(complete=True,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            input_sha256=inputs, source_sha256=sources,
            artifact_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.iterdir()) if p.is_file()},
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            original_evidence_unchanged=True, publication_clearance=False))
        return aggregate
    except Exception as exc:
        save(output/'incomplete.json',dict(complete=False,exception_type=type(exc).__name__,reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=MAIN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.main, args.output))
