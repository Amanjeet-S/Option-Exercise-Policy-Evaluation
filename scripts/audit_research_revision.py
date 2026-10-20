"""Reconstruct private, derived evidence for the manuscript revision.

This audit reads retained outcomes, checks the bytes of consumed artifacts and
exports only explicitly selected research summaries. It does not simulate new
outcomes, certify a PDE enclosure, validate interval coverage or publish data.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ((5000, 2), (5000, 4), (20000, 2), (20000, 4))


def distribution(values):
    """Descriptive summaries of every supplied outcome, including negatives."""
    values = [float(x) for x in values]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Require nonempty finite outcomes')
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values),
                negative=sum(x < 0 for x in values), zero=sum(x == 0 for x in values),
                positive=sum(x > 0 for x in values))


def interval_summary(estimates, factor=1.):
    """Describe individual Monte Carlo intervals without pooling their errors."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Require a positive premium factor')
    records = [{k: float(e[k])/factor for k in ('value', 'standard_error', 'lower', 'upper')}
               for e in estimates]
    if any(e['standard_error'] < 0 or e['lower'] > e['value'] or e['value'] > e['upper']
           for e in records):
        raise ValueError('Invalid retained interval')
    return dict(**{k: distribution(e[k] for e in records)
                   for k in ('value', 'standard_error', 'lower', 'upper')},
                positive_intervals=sum(e['lower'] > 0 for e in records),
                negative_intervals=sum(e['upper'] < 0 for e in records),
                intervals_including_zero=sum(e['lower'] <= 0 <= e['upper'] for e in records))


def family_calibration(critical=3.25, contrasts=12):
    if not math.isfinite(critical) or critical <= 0 or type(contrasts) is not int or contrasts < 1:
        raise ValueError('Require a positive cutoff and family size')
    pointwise_error = float(2*norm.sf(critical))
    family_bound = min(1., contrasts*pointwise_error)
    return dict(critical_value=critical, contrasts=contrasts,
                asymptotic_normal_two_sided_error=pointwise_error,
                asymptotic_union_bound_family_error=family_bound,
                asymptotic_union_bound_coverage_lower=1-family_bound,
                five_per_cent_bonferroni_normal_cutoff=float(norm.isf(.05/(2*contrasts))),
                independence_between_contrasts_required=False,
                finite_sample_coverage_established=False,
                component_ratio_sets_are_a_separate_inference_family=True)


class Archive:
    """Read only manifest-bound inputs, retaining hashes rather than their data."""

    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.inputs = {}

    def _path(self, relative):
        path = (self.root/relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError('Input leaves the retained archive')
        return path

    def manifest(self, folder):
        relative = str(Path(folder)/'manifest.json')
        path = self._path(relative)
        payload = path.read_bytes()
        self.inputs[relative] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    def read(self, folder, name):
        manifest = self.manifest(folder)
        relative = str(Path(folder)/name)
        payload = self._path(relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if manifest['artifact_sha256'].get(name) != digest:
            raise ValueError(f'Retained artifact hash differs: {relative}')
        self.inputs[relative] = digest
        return json.loads(payload)


def measurement_evidence(rows, configuration):
    individual, groups, case_reductions = [], [], []
    for row in rows:
        factor = row['premium_factor']
        for p in row['policies']:
            individual.append(dict(case=row['case'], repetition=p['repetition'],
                training_paths=p['training_paths'], degree=p['degree'],
                methods={name: dict(mean_gap=r['gap_bound']['value']/factor,
                    standard_error=r['gap_bound']['standard_error']/factor,
                    upper_loss_limit_97_5=r['loss_upper_97_5_quote'],
                    gap_zero_fraction=r['sample_zero_gap_fraction'],
                    numerical_pde_minus_adjusted_policy=row['pde_quote']-r['policy_value']['value']/factor,
                    adjusted_upper_minus_numerical_pde=r['upper_value']['value']/factor-row['pde_quote'])
                    for name, r in p['results'].items()},
                paired_gap_comparisons={name: {k: e[k] for k in
                    ('value', 'standard_error', 'lower', 'upper', 'observations')}
                    for name, e in p['paired_gap_comparisons'].items()}))
        for n, d in VARIANTS:
            ps = [p for p in row['policies'] if (p['training_paths'], p['degree']) == (n, d)]
            baseline = statistics.mean(p['results']['baseline']['gap_bound']['value'] for p in ps)
            fine = statistics.mean(p['results']['fine']['gap_bound']['value'] for p in ps)
            case_reductions.append(1-fine/baseline)
    for n, d in VARIANTS:
        records = [p for p in individual if (p['training_paths'], p['degree']) == (n, d)]
        methods = {name: {k: distribution(p['methods'][name][k] for p in records)
                         for k in records[0]['methods'][name]}
                   for name in ('baseline', 'coarse', 'fine')}
        reduction = 1-methods['fine']['mean_gap']['mean']/methods['baseline']['mean_gap']['mean']
        groups.append(dict(training_paths=n, degree=d, policies=len(records),
                           methods=methods, reduction_of_configuration_average=reduction))
    return dict(outer_paths_per_case=configuration['evaluation_paths'],
        construction_settings=configuration['approximations'],
        configurations=groups, individual_policies=individual,
        configuration_average_reductions=distribution(g['reduction_of_configuration_average'] for g in groups),
        case_configuration_average_reductions=distribution(case_reductions),
        individual_policy_reductions=distribution(1-p['methods']['fine']['mean_gap']/
            p['methods']['baseline']['mean_gap'] for p in individual),
        aggregation_inference='Configuration averages are descriptive. All policies in a case share outer paths. '
            'The retained scalar records omit covariance between their gap vectors, so their '
            'configuration-average Monte Carlo standard errors cannot be reconstructed from rows.json.',
        numerical_reference_status='PDE differences are comparisons with a retained numerical reference, '
            'not certified policy losses or dual slacks. Agreement under refinement is not an error enclosure.')


def financial_scale(rows):
    cases = {}
    for row in rows:
        current = (row['european_quote'], row['pde_quote'], row['model_exercise_premium'])
        if row['case'] in cases and cases[row['case']] != current:
            raise ValueError('Case pricing references differ between repetitions')
        if not math.isclose(current[1]-current[0], current[2], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Numerical exercise-premium identity differs')
        cases[row['case']] = current
    groups = []
    for n, d in VARIANTS:
        entries = [p for row in rows for p in row['policies']
                   if (p['training_paths'], p['degree']) == (n, d)]
        groups.append(dict(training_paths=n, degree=d, policies=len(entries),
            gain_over_terminal_waiting=interval_summary([p['gain']['controlled_quote'] for p in entries]),
            numerical_pde_shortfall=distribution(p['pde_shortfall'] for p in entries),
            fraction_of_numerical_exercise_premium_captured=distribution(p['captured_fraction'] for p in entries),
            early_exercise_fraction=distribution(p['early_exercise_fraction'] for p in entries)))
    return dict(cases=len(cases),
        terminal_waiting_model_price=distribution(x[0] for x in cases.values()),
        numerical_pde_model_price=distribution(x[1] for x in cases.values()),
        numerical_exercise_premium=distribution(x[2] for x in cases.values()),
        configurations=groups,
        interpretation='All prices are model-derived outputs, not provider quotes. Controlled gains '
            'compare the frozen policy with terminal exercise on common paths. The numerical PDE '
            'price and its exercise premium have no certified error enclosure.')


def retraining_evidence(rows, extension_seconds=5400):
    records = []
    for row in rows:
        scenario = next(s for s in row['scenarios'] if s['extension_seconds'] == extension_seconds)
        for p in scenario['policies']:
            records.append(dict(case=row['case'], repetition=p['repetition'],
                training_paths=p['training_paths'], degree=p['degree'],
                **{k: p['paired_gain'][k] for k in ('value', 'standard_error', 'lower', 'upper')},
                changed_stop_fraction=p['changed_stops']/row['stream']['paths']))
    groups = []
    for n, d in VARIANTS:
        ps = [p for p in records if (p['training_paths'], p['degree']) == (n, d)]
        groups.append(dict(training_paths=n, degree=d,
                          paired_gain=interval_summary(ps),
                          changed_stop_fraction=distribution(p['changed_stop_fraction'] for p in ps)))
    return dict(extension_seconds=extension_seconds,
        sign_convention='Retrained reward minus terminal-only-adjusted reward, divided by the premium '
            'factor. Equivalently, terminal-only gap minus retrained gap under their common upper payoff. '
            'Negative values mean that retraining worsens the expected reward.',
        configurations=groups, pooled=interval_summary(records), individual_policies=records,
        aggregation_inference='These averages and sign counts describe dependent retained policies; '
            'individual pointwise intervals do not establish a simultaneous ranking or a '
            'confidence interval for the pooled average.')


def cost_accounting(pilot_manifest, main_manifest, main, timing, prospective, cpu):
    arm_totals = {arm: sum(x['arms'][arm]['standalone_cpu_seconds'] for x in main['models'])
                  for arm in ('reference', 'selected')}
    for arm in arm_totals:
        timing_total = sum(timing['totals'][arm]['actual_cpu_seconds'].values())
        if not math.isclose(arm_totals[arm], timing_total, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('Standalone cost and retained component sums differ')
    pilot_cost = pilot_manifest['process_cpu_seconds']
    selected_cost = pilot_cost+arm_totals['selected']
    per_case = []
    for model in main['models']:
        r, s = (model['arms'][a] for a in ('reference', 'selected'))
        per_case.append(dict(case=model['case'], reference_cpu=r['standalone_cpu_seconds'],
            selected_cpu=s['standalone_cpu_seconds'], selected_to_reference_cpu=s['standalone_cpu_seconds']/r['standalone_cpu_seconds'],
            reference_pairs=r['allocation']['repetitions'], selected_pairs=s['allocation']['repetitions'],
            selected_paths=s['allocation']['evaluation_paths'],
            selected_to_reference_estimator_variance=model['comparison']['estimated_variance_ratio'],
            between_pair_dominance_prediction=r['allocation']['repetitions']/s['allocation']['repetitions']))
    return dict(pilot_process_cpu=pilot_cost, pilot_elapsed=pilot_manifest['elapsed_seconds'],
        main_process_cpu=main_manifest['process_cpu_seconds'], standalone_arm_cpu=arm_totals,
        one_off_pilot_plus_selected_cpu=selected_cost,
        pilot_plus_both_standalone_arms_cpu=pilot_cost+sum(arm_totals.values()),
        pilot_plus_selected_to_reference_cpu=selected_cost/arm_totals['reference'],
        simple_reference_variance_counterfactual_ratio=arm_totals['reference']/selected_cost,
        counterfactual_assumptions='Illustrative 1/m scaling at unchanged reference pair cost and '
            'negligible setup. No additional reference run was observed at that enlarged cost.',
        per_case=per_case,
        selected_to_reference_actual_cpu=distribution(x['selected_to_reference_cpu'] for x in per_case),
        prospective_primary_numerical_cpu=prospective['primary_numerical_cpu_seconds'],
        local_cpu_numerical_total=float(Fraction(cpu['numerical_totals']['cpu'])),
        local_cpu_process_total=cpu['process_cpu_seconds'],
        equal_observed_cost_comparison=False,
        pilot_expense_is_a_separate_one_off_charge=True)


def prospective_diagnostics(arms):
    result = []
    for arm in arms:
        values = arm['statistics']
        variance = float(Fraction(values['variance_of_mean']))
        influence = float(Fraction(values['largest_squared_variance_influence_share']))
        leave = values['leave_one_variance_at_fixed_count']
        result.append(dict(case=arm['case'], evaluation_paths=arm['evaluation_paths'],
            pairs=values['pairs'], mean_gain=float(Fraction(values['mean'])),
            estimated_variance_of_mean=variance,
            numerical_cpu_seconds=float(Fraction(arm['exact_numerical_cpu'])),
            actual_to_forecast=float(Fraction(arm['actual_to_forecast'])),
            largest_squared_variance_influence_share=influence,
            delete_one_at_fixed_count_relative_variance_range={key: float(Fraction(leave[key]))/variance
                                                               for key in ('lower', 'upper')},
            most_variance_reducing_deleted_repetition=leave['lower_row']))
    comparisons = []
    for case in sorted({row['case'] for row in result}):
        reference = next(row for row in result if (row['case'], row['evaluation_paths']) == (case, 4096))
        for target in (row for row in result if row['case'] == case and row['evaluation_paths'] != 4096):
            cpu_ratio = target['numerical_cpu_seconds']/reference['numerical_cpu_seconds']
            variance_ratio = target['estimated_variance_of_mean']/reference['estimated_variance_of_mean']
            comparisons.append(dict(case=case, evaluation_paths=target['evaluation_paths'],
                selected_to_reference_cpu=cpu_ratio, selected_to_reference_estimator_variance=variance_ratio,
                variance_times_cpu_ratio=cpu_ratio*variance_ratio,
                between_pair_dominance_prediction=reference['pairs']/target['pairs']))
    return dict(arms=result, comparisons=comparisons,
        primary_numerical_cpu_seconds=sum(row['numerical_cpu_seconds'] for row in result),
        interpretation='The influence shares concern uncertainty about '
        'estimated variance. They are not shares of the estimator variance. Delete-one ranges '
        'are descriptive sensitivity diagnostics; all primary pairs remain retained.')


def audit(archive_root):
    archive = Archive(archive_root)
    value = 'data/private/value_martingale/2026-09-20/run_v1'
    fitted = 'data/private/fitted_policy_gain/2026-09-20/run_v1'
    terminal = 'data/private/terminal_training/2026-09-20/run_v1'
    main = 'data/private/training_allocation/2026-09-27/main_run_v1'
    pilot = 'data/private/training_allocation/2026-09-27/pilot_v1'
    timing = 'data/private/training_allocation/2026-09-27/timing_review_v1'
    prospective = 'data/private/prospective_allocation/2026-09-28/run_v1'
    cpu = 'data/private/cpu_prediction/2026-09-28/run_v1'
    value_rows = archive.read(value, 'rows.json')
    if [r['case'] for r in value_rows] != list(range(13)):
        raise ValueError('Require all thirteen retained cases')
    arms = [archive.read(prospective, f'arm_{case:02d}_{paths}.json')
            for case in (5, 6, 10) for paths in (128, 256, 4096)]
    diagnostics = prospective_diagnostics(arms)
    contrasts = archive.read(prospective, 'comparisons.json')
    if contrasts['family_size'] != 2*len(contrasts['comparisons']):
        raise ValueError('Retained contrast family size differs')
    report = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Private reconstruction of retained research outputs for revision. No new simulation '
            'result, market validation, provider observations or publication permission.',
        monetary_unit='USD per share in premium-quote units',
        measurement=measurement_evidence(value_rows, archive.read(value, 'configuration.json')),
        financial_scale=financial_scale(archive.read(fitted, 'rows.json')),
        retraining=retraining_evidence(archive.read(terminal, 'rows.json')),
        twelve_contrast_calibration=family_calibration(
            float(Fraction(contrasts['critical_value'])), contrasts['family_size']),
        cost_accounting=cost_accounting(archive.manifest(pilot), archive.manifest(main),
            archive.read(main, 'summary.json'), archive.read(timing, 'diagnostic.json'),
            diagnostics, archive.read(cpu, 'summary.json')),
        prospective_variance_diagnostics=diagnostics,
        record_availability=dict(
            path_vectors_retained=False,
            available='Per-policy means, standard errors, limits and paired differences; complete '
                'training-pair means, within-pair errors and covariance summaries; frozen policies, '
                'value constructions and stream identities for numerical regeneration.',
            table_one=value+'/rows.json',
            terminal_waiting_comparison=fitted+'/rows.json',
            retraining=terminal+'/rows.json',
            eight_pair_gains='data/private/training_evaluation/2026-09-26/run_v1/blocks/',
            allocation_pair_gains=main+'/blocks/',
            prospective_pair_gains=prospective+'/rows/',
            coverage_experiments='No interval-coverage experiment is reconstructed. Retained '
                'means may support empirical resampling diagnostics, but cannot establish '
                'coverage for the original population. New public stipulated-model experiments '
                'must identify their sampling law and independently resolved target.'))
    report['consumed_artifact_sha256'] = archive.inputs
    report['integrity_scope'] = ('Consumed artifacts checked against their retained manifests. '
        'The complete historical source/input closure and financial validity are not re-audited.')
    return report


def write_report(report, output, private_root=None):
    private_root = (ROOT/'data/private' if private_root is None else Path(private_root)).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(private_root):
        raise ValueError('Derived evidence must remain in the private output tree')
    if output.exists():
        raise FileExistsError('Use a fresh private output directory')
    output.mkdir(parents=True)
    path = output/'derived_evidence.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    print('Reading retained outcome records; provider inputs will not be printed.', flush=True)
    result = audit(arguments.archive_root)
    destination = write_report(result, arguments.output)
    print(f'Derived private evidence saved to {destination}')
