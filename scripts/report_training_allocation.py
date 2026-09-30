"""Report a completed, audited allocation comparison without rerunning numerics.

All dependent findings, numerical values and plots remain in the private data
tree. Pilot predictions and main estimated variance ratios are descriptive;
this report supplies neither ratio confidence intervals nor an MSE benchmark.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping


def ratio(numerator, denominator):
    """Retain an undefined ratio when the reference quantity is zero or absent."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    result = numerator/denominator
    if not math.isfinite(result):
        raise ValueError('A reported ratio is not finite')
    return result


def extent(values):
    finite = [value for value in values if value is not None]
    return dict(minimum=min(finite), maximum=max(finite), observations=len(finite)) if finite else None


def report_values(summary, configuration):
    """Extract comparable quantities from the audited, frozen main summary."""
    models = []
    for model in summary['models']:
        arms = {}
        reference_budget = model['arms']['reference']['allocation']['total_cpu_seconds']
        for name in ('reference', 'selected'):
            retained = model['arms'][name]
            allocation = retained['allocation']
            estimate = retained['inference']['estimate']
            arms[name] = dict(
                repetitions=allocation['repetitions'],
                evaluation_paths=allocation['evaluation_paths'],
                gain_mean=estimate['value'], gain_standard_error=estimate['standard_error'],
                gain_lower=estimate['lower'], gain_upper=estimate['upper'],
                confidence=estimate['confidence'],
                estimated_variance=estimate['standard_error']**2,
                pilot_predicted_variance=allocation['predicted_variance'],
                pilot_predicted_cpu_seconds=allocation['total_cpu_seconds'],
                predicted_reference_budget_cpu_seconds=reference_budget,
                actual_cpu_seconds=retained['standalone_cpu_seconds'],
                actual_elapsed_seconds=retained['standalone_elapsed_seconds'],
                actual_to_budget_ratio=retained['actual_to_budget_ratio'],
                material_budget_mismatch=retained['material_budget_mismatch'],
                zero_gain_sample_pairs=retained['zero_gain_sample_pairs'])
        reference, selected = arms['reference'], arms['selected']
        comparison = dict(
            pilot_predicted_variance_ratio=ratio(selected['pilot_predicted_variance'],
                                                 reference['pilot_predicted_variance']),
            estimated_variance_ratio=ratio(selected['estimated_variance'], reference['estimated_variance']),
            actual_cpu_ratio=ratio(selected['actual_cpu_seconds'], reference['actual_cpu_seconds']),
            estimated_variance_cpu_ratio=ratio(
                selected['estimated_variance']*selected['actual_cpu_seconds'],
                reference['estimated_variance']*reference['actual_cpu_seconds']))
        models.append(dict(case=model['case'], arms=arms, comparison=comparison))
    if not models or len({model['case'] for model in models}) != len(models):
        raise ValueError('Require nonempty, distinct audited model records')
    confidence = {arm['confidence'] for model in models for arm in model['arms'].values()}
    if len(confidence) != 1:
        raise ValueError('The report requires one common interval confidence level')
    models.sort(key=lambda model: model['case'])
    totals = {name: dict(
        repetitions=sum(model['arms'][name]['repetitions'] for model in models),
        actual_cpu_seconds=sum(model['arms'][name]['actual_cpu_seconds'] for model in models),
        actual_elapsed_seconds=sum(model['arms'][name]['actual_elapsed_seconds'] for model in models),
        pilot_predicted_cpu_seconds=sum(model['arms'][name]['pilot_predicted_cpu_seconds'] for model in models),
        material_budget_mismatches=sum(model['arms'][name]['material_budget_mismatch'] for model in models))
        for name in ('reference', 'selected')}
    ranges = {name: extent([model['comparison'][name] for model in models]) for name in
              ('pilot_predicted_variance_ratio', 'estimated_variance_ratio',
               'actual_cpu_ratio', 'estimated_variance_cpu_ratio')}
    descriptive_counts = {}
    for field in ('estimated_variance_ratio', 'estimated_variance_cpu_ratio'):
        defined = [model['comparison'][field] for model in models
                   if model['comparison'][field] is not None]
        descriptive_counts[field] = dict(below_one=sum(value < 1 for value in defined),
                                        defined_comparisons=len(defined))
    descriptive_counts['positive_pointwise_lower_bounds'] = {
        name: dict(positive=sum(model['arms'][name]['gain_lower'] > 0 for model in models),
                   model_count=len(models)) for name in ('reference', 'selected')}
    lower, upper = configuration['budget_ratio_limits']

    def budget_counts(ratios):
        return dict(actual_to_budget_ratio=extent(ratios), arm_count=len(ratios),
                    below_budget=sum(value < 1 for value in ratios),
                    above_budget=sum(value > 1 for value in ratios),
                    equal_to_budget=sum(value == 1 for value in ratios),
                    below_lower_limit=sum(value < lower for value in ratios),
                    above_upper_limit=sum(value > upper for value in ratios),
                    within_limits=sum(lower <= value <= upper for value in ratios))

    budget_assessment = budget_counts([arm['actual_to_budget_ratio']
        for model in models for arm in model['arms'].values()])
    budget_assessment['by_arm'] = {
        name: budget_counts([model['arms'][name]['actual_to_budget_ratio'] for model in models])
        for name in ('reference', 'selected')}
    budget_assessment.update(registered_limits=[lower, upper], fixed_counts_preserved=True,
                             validated_equal_budget=False, global_optimum_claim=False)
    return dict(models=models, totals=totals, ranges=ranges,
                descriptive_counts=descriptive_counts,
                budget_assessment=budget_assessment,
                model_count=len(models), confidence=confidence.pop(),
                training_paths=configuration['training_paths'],
                requested_regression_degree=configuration['degree'],
                budget_ratio_limits=configuration['budget_ratio_limits'],
                main_outcomes=True, descriptive_ratio_estimates=True,
                ratio_confidence_intervals=False, measured_mean_squared_error=False,
                validated_interval_coverage=False, independent_market_validation=False,
                validated_equal_budget=False, global_optimum_claim=False,
                publication_clearance=False)


def number(value, places=3):
    return 'not defined' if value is None else f'{value:.{places}f}'


def range_text(record):
    return ('not defined' if record is None else
            f"{number(record['minimum'])} to {number(record['maximum'])}")


def markdown(values, main, audit, output):
    """Write research prose from retained quantities, without ranking markets."""
    count = values['model_count']
    low, high = values['training_paths']
    confidence = 100*values['confidence']
    reference, selected = values['totals']['reference'], values['totals']['selected']
    positive = values['descriptive_counts']['positive_pointwise_lower_bounds']
    budget = values['budget_assessment']
    limits = values['budget_ratio_limits']
    main_link = Path(os.path.relpath(main/'manifest.json', output)).as_posix()
    audit_link = Path(os.path.relpath(audit/'verification.json', output)).as_posix()
    methods_link = Path(os.path.relpath(
        ROOT/'projects/01_optimal_stopping/reports/training_allocation.md', output)).as_posix()
    mismatches = budget['below_lower_limit']+budget['above_upper_limit']
    calibration_statement = (
        f"The pilot timing calibration did not meet its registered tolerance for {mismatches} "
        f"of {budget['arm_count']} workloads. " if mismatches else
        'All observed workload costs lie within the registered calibration tolerance. ')
    lines = [
        '# Independent main comparison of computational allocations', '',
        '## 1. Question and completed design', '',
        f'I compare two prospectively frozen allocations for estimating the mean quoted policy-value '
        f'gain from training the fixed exercise-rule procedure on {high:,} rather than {low:,} '
        f'independent simulated paths. The requested regression degree is '
        f"{values['requested_regression_degree']}. The reference and selected allocations target "
        'the same procedure-level mean within each financial specification. Changing the allocation '
        'changes its assessment, not the training procedures or the reward of a particular frozen '
        f'policy. The [methods note]({methods_link}) defines the adjusted estimator, gives the '
        'elementary variance-allocation derivation and attributes the established regression '
        'and martingale constructions.', '',
        f'Both arms completed their prescribed counts across {count} retained conditional '
        f'specifications. The reference comprises {reference["repetitions"]:,} independently trained '
        f'pairs and the selected allocation {selected["repetitions"]:,}. Each pair uses independent '
        'training samples for the two procedures and a fresh common evaluation sample for their '
        'comparison. Pilot and main draws, and the two main arms, use distinct streams. The counts '
        'were not changed in response to observed gains, interval widths or execution times.', '',
        '## 2. Budget calibration and observed computational costs', '',
        f"Across all {budget['arm_count']} model-arm workloads, actual CPU cost divided by the "
        f"pilot-predicted reference budget ranges from "
        f"{range_text(budget['actual_to_budget_ratio'])}. Actual cost is below the predicted "
        f"budget in {budget['below_budget']} workloads and above it in {budget['above_budget']}. "
        f"Of all workloads, "
        f"{budget['below_lower_limit']} fall below the registered lower limit of "
        f"{number(limits[0], 1)}, {budget['above_upper_limit']} exceed the upper limit of "
        f"{number(limits[1], 1)}, and {budget['within_limits']} lie within the tolerance band. "
        f'{calibration_statement}I preserve every frozen '
        'sample count despite the realised budget differences. The experiment does not validate '
        'an exact or equal-budget comparison, or a globally optimal allocation.', '',
        'Each arm pays for its own value-approximation construction, both training procedures and '
        'the shared evaluation workload, including the retained dual assessments. The following '
        'CPU and elapsed charges cover these standalone numerical workloads. Pilot work, initial '
        'warm-up, source checks and output handling are additional research expenses. These are '
        'observations on the recorded host, not hardware-independent performance guarantees.', '',
        '| Model | Reference R / N | Selected R / N | Reference CPU seconds | Selected CPU seconds | Reference elapsed seconds | Selected elapsed seconds | Budget mismatch |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for model in values['models']:
        ref, sel = model['arms']['reference'], model['arms']['selected']
        mismatch = ', '.join(name for name in ('reference', 'selected')
                             if model['arms'][name]['material_budget_mismatch']) or 'none'
        lines.append(f"| {model['case']} | {ref['repetitions']:,} / {ref['evaluation_paths']:,} "
                     f"| {sel['repetitions']:,} / {sel['evaluation_paths']:,} "
                     f"| {number(ref['actual_cpu_seconds'])} | {number(sel['actual_cpu_seconds'])} "
                     f"| {number(ref['actual_elapsed_seconds'])} | {number(sel['actual_elapsed_seconds'])} | {mismatch} |")
    lines += ['',
        f"The summed standalone CPU charges are {number(reference['actual_cpu_seconds'])} "
        f"seconds for the reference and {number(selected['actual_cpu_seconds'])} seconds for the "
        f"selected allocation. Their pilot-predicted charges were "
        f"{number(reference['pilot_predicted_cpu_seconds'])} and "
        f"{number(selected['pilot_predicted_cpu_seconds'])} seconds. These sums describe research "
        'expenditure; they do not pool the specifications for financial inference.', '',
        f"The registered mismatch rule compares each arm's actual CPU charge with its model-specific "
        f"reference budget, using limits {number(limits[0], 1)} and {number(limits[1], 1)}. "
        f"It flags {reference['material_budget_mismatches']} reference arms and "
        f"{selected['material_budget_mismatches']} selected arms. A flag is retained alongside the "
        'fixed counts. Predicted budget matching does not establish equal realised computation.', '',
        '## 3. Estimates of the common procedure-level gain', '',
        f'I report approximate, pointwise {confidence:g} per cent Student intervals across the '
        'independently trained pairs. Gains and interval endpoints are in quoted premium units. '
        'The two arms estimate the same mean; their realised means need not coincide. These '
        'intervals provide no simultaneous conclusion across the retained specifications.', '',
        '| Model | Reference mean gain | Reference interval | Selected mean gain | Selected interval |',
        '|---|---:|---|---:|---|']
    for model in values['models']:
        ref, sel = model['arms']['reference'], model['arms']['selected']
        lines.append(f"| {model['case']} | {number(ref['gain_mean'], 5)} "
                     f"| [{number(ref['gain_lower'], 5)}, {number(ref['gain_upper'], 5)}] "
                     f"| {number(sel['gain_mean'], 5)} "
                     f"| [{number(sel['gain_lower'], 5)}, {number(sel['gain_upper'], 5)}] |")
    lines += ['',
        f"The reference arm's lower interval bound is positive in "
        f"{positive['reference']['positive']} of {positive['reference']['model_count']} "
        f"specifications; the selected arm's bound is positive in "
        f"{positive['selected']['positive']} of {positive['selected']['model_count']}. "
        'These counts describe realised pointwise intervals. They are not a simultaneous '
        'superiority claim or a count of independent market confirmations.', '',
        '## 4. Descriptive variance and cost comparisons', '',
        'Every ratio below divides the selected quantity by its reference counterpart within the '
        'same model. Estimated variance is the square of the standard error across independent '
        'pair means. The variance-times-CPU ratio multiplies that estimated variance by the '
        'observed standalone CPU cost before comparison. It accounts descriptively for realised '
        'cost differences; it is not a confidence statement about efficiency.', '',
        '| Model | Pilot-predicted variance ratio | Main estimated variance ratio | Actual CPU ratio | Estimated variance-times-CPU ratio |',
        '|---|---:|---:|---:|---:|']
    for model in values['models']:
        comparison = model['comparison']
        lines.append(f"| {model['case']} | "+' | '.join(number(comparison[name]) for name in
                     ('pilot_predicted_variance_ratio', 'estimated_variance_ratio',
                      'actual_cpu_ratio', 'estimated_variance_cpu_ratio'))+' |')
    ranges = values['ranges']
    variance_counts = values['descriptive_counts']['estimated_variance_ratio']
    efficiency_counts = values['descriptive_counts']['estimated_variance_cpu_ratio']
    lines += ['',
        f"The selected arm has a smaller estimated variance in {variance_counts['below_one']} "
        f"of {variance_counts['defined_comparisons']} defined model-specific comparisons. "
        f"Its estimated variance-times-CPU product is smaller in "
        f"{efficiency_counts['below_one']} of {efficiency_counts['defined_comparisons']} "
        'defined comparisons. These are descriptive counts across related conditional '
        'models, not confidence statements about the ratios.', '',
        f"Across defined model-specific comparisons, main estimated variance ratios range from "
        f"{range_text(ranges['estimated_variance_ratio'])}, actual CPU ratios from "
        f"{range_text(ranges['actual_cpu_ratio'])}, and estimated variance-times-CPU ratios from "
        f"{range_text(ranges['estimated_variance_cpu_ratio'])}. The pilot-predicted variance ratios "
        f"range from {range_text(ranges['pilot_predicted_variance_ratio'])}. The pilot selected "
        'counts using estimated components and measured costs; its minimised prediction is not '
        'an independent demonstration of efficiency. The main estimates can differ through '
        'sampling variation and changes in realised computational cost.', '',
        '![Descriptive variance, CPU and budget-calibration ratios](allocation_comparison.png)', '',
        'The left panel separates pilot predictions from main estimated variance ratios. The '
        'middle panel shows selected-to-reference actual CPU ratios. The right panel shows '
        "each arm's actual CPU cost divided by its model's pilot-predicted reference budget, "
        f"with the registered {number(limits[0], 1)} to {number(limits[1], 1)} tolerance band. "
        'The horizontal lines mark a ratio of one. The budget panel makes calibration '
        'departures visible even when the selected-to-reference CPU ratio is near one. Points '
        'have no ratio confidence intervals; differences between models are not a ranking of '
        'independent markets. Variance ratios are not defined where the reference variance is '
        'zero or a pilot prediction was unresolved.', '',
        '## 5. Interpretation and limits', '',
        'I interpret these findings as a comparison of numerical allocations for one stated '
        'estimand under each retained financial specification. Estimated precision and observed '
        'computational cost are distinct from realised squared error. The population mean is '
        'unknown, so this experiment does not measure mean squared error against a known target '
        'or validate nominal interval coverage. It does not establish a continuous or universal '
        'optimum, an improvement in every fitted policy, or the best allocation for valuing a '
        'single frozen policy or tightening a dual upper bound.', '',
        'The specifications are related conditional models rather than independent market '
        'observations. Fresh simulated paths do not remove the historical information-set, '
        'contract-measurement or calibration limitations of the development evidence. No '
        'market-profitability or independent market-validation conclusion follows.', '',
        '## 6. Evidence', '',
        f'This report uses the [completed main record]({main_link}) and its '
        f'[separate verification record]({audit_link}). The accompanying values file retains '
        'the exact quantities used for the tables and figure, including model-specific budgets '
        'and mismatch flags. Licensed inputs, fitted policies and dependent numerical findings '
        'remain private.', '']
    return '\n'.join(lines)


def plot_comparison(values, output):
    """Draw descriptive points without confidence intervals or pooled models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    models = values['models']
    positions = list(range(len(models)))
    labels = [str(model['case']) for model in models]
    figure, axes = plt.subplots(1, 3, figsize=(max(16.5, min(26., len(models)*1.2)), 4.8))
    for field, label, colour, offset, marker in (
            ('pilot_predicted_variance_ratio', 'Pilot-predicted variance', '#D55E00', -.12, 's'),
            ('estimated_variance_ratio', 'Main estimated variance', '#0072B2', .12, 'o')):
        pairs = [(position+offset, model['comparison'][field])
                 for position, model in zip(positions, models) if model['comparison'][field] is not None]
        if pairs:
            axes[0].scatter([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                            label=label, color=colour, marker=marker, s=35, zorder=3)
    cpu_pairs = [(position, model['comparison']['actual_cpu_ratio'])
                 for position, model in zip(positions, models)
                 if model['comparison']['actual_cpu_ratio'] is not None]
    if cpu_pairs:
        axes[1].scatter([pair[0] for pair in cpu_pairs], [pair[1] for pair in cpu_pairs],
                        color='#009E73', marker='o', s=35, zorder=3)
    lower, upper = values['budget_ratio_limits']
    axes[2].axhspan(lower, upper, color='#777777', alpha=.16,
                    label=f'Registered band [{lower:g}, {upper:g}]')
    for name, colour, offset, marker in (
            ('reference', '#0072B2', -.12, 'o'),
            ('selected', '#D55E00', .12, 's')):
        axes[2].scatter([position+offset for position in positions],
                        [model['arms'][name]['actual_to_budget_ratio'] for model in models],
                        label=name.capitalize(), color=colour, marker=marker, s=35, zorder=3)
    for axis in axes:
        axis.axhline(1., color='#555555', linestyle='--', linewidth=1)
        axis.set_xticks(positions, labels)
        axis.set_xlabel('Retained conditional model')
        axis.grid(axis='y', alpha=.25)
        axis.set_ylim(bottom=0.)
    axes[0].set_title('Variance ratios: predictions and estimates')
    axes[0].set_ylabel('Selected / reference variance')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, fontsize=9)
    else:
        axes[0].text(.5, .5, 'No defined variance ratios', transform=axes[0].transAxes,
                     ha='center', va='center')
    axes[1].set_title('Observed computational cost ratio')
    axes[1].set_ylabel('Selected / reference process CPU')
    axes[2].set_title('Pilot budget calibration: both arms')
    axes[2].set_ylabel('Actual process CPU / predicted budget')
    axes[2].legend(fontsize=9)
    figure.suptitle('Descriptive comparisons; no confidence intervals for ratios', fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(figure)
    return matplotlib.__version__


def preflight(main, audit):
    """Bind a successful audit to completed evidence before creating output.

    The supplemental auditor deliberately retains its additional source closure
    separately from the original registered sources. Both closures are checked;
    the report does not amend any frozen registration or execute its audit reruns.
    """
    import audit_training_allocation_main as main_audit

    audit_manifest = read(audit/'manifest.json')
    if (audit_manifest.get('complete') is not True
            or audit_manifest.get('registration_unchanged') is not True):
        raise ValueError('Require a successful audit with an unchanged registration')
    for group in ('input_sha256', 'registered_source_sha256',
                  'additional_audit_source_sha256', 'audit_source_sha256'):
        verify_mapping(audit_manifest[group])
    verify_mapping(audit_manifest['artifact_sha256'], audit)
    if 'verification.json' not in audit_manifest['artifact_sha256']:
        raise ValueError('Audit verification record is not bound by its manifest')
    auditor_path = Path(main_audit.__file__).resolve()
    auditor_name = str(auditor_path.relative_to(ROOT))
    if audit_manifest['audit_source_sha256'].get(auditor_name) != sha(auditor_path):
        raise ValueError('The current main verifier is not bound by the completed audit')
    verification = read(audit/'verification.json')
    main_name = str(main.relative_to(ROOT))
    main_hash = sha(main/'manifest.json')
    if (verification.get('main_run') != main_name
            or verification.get('main_manifest_sha256') != main_hash
            or audit_manifest['input_sha256'].get(f'{main_name}/manifest.json') != main_hash):
        raise ValueError('Audit does not identify the supplied completed main run')

    main_manifest, configuration, _, checks = main_audit.verify(main)
    if main_manifest.get('complete') is not True:
        raise ValueError('Require a completed main run')
    normalised_checks = json.loads(json.dumps(checks, allow_nan=False))
    if normalised_checks != verification['checks']:
        raise ValueError('Current verification differs from the retained successful audit')
    summary = read(main/'summary.json')
    if 'summary.json' not in main_manifest['artifact_sha256']:
        raise ValueError('The main summary is not bound by the completed manifest')
    values = report_values(summary, configuration)
    values['main_stage_execution'] = dict(
        process_cpu_seconds=main_manifest['process_cpu_seconds'],
        elapsed_seconds=main_manifest['elapsed_seconds'])
    values['evidence'] = dict(main_run=main_name, main_manifest_sha256=main_hash,
        audit=str(audit.relative_to(ROOT)), audit_manifest_sha256=sha(audit/'manifest.json'))
    sources = dict(audit_manifest['audit_source_sha256'])
    report_source = Path(__file__).resolve()
    sources[str(report_source.relative_to(ROOT))] = sha(report_source)
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in
              (main/'manifest.json', main/'summary.json', audit/'manifest.json',
               audit/'verification.json')}
    verify_mapping(inputs)
    verify_mapping(sources)
    return main_manifest, audit_manifest, values, inputs, sources


def report(main, audit, output):
    """Create private findings from a complete audit, with no numerical reruns."""
    main, audit, output = map(private, map(Path, (main, audit, output)))
    if output.exists():
        raise FileExistsError('Use a fresh private report directory')
    main_manifest, audit_manifest, values, inputs, sources = preflight(main, audit)
    text = markdown(values, main, audit, output)
    # All input, source and retained-artifact checks above precede output creation.
    output.mkdir(parents=True)
    save(output/'values.json', values)
    (output/'main_findings.md').write_text(text)
    matplotlib_version = plot_comparison(values, output/'allocation_comparison.png')
    # Prevent a completed report from silently depending on changed evidence.
    verify_mapping(inputs)
    verify_mapping(sources)
    verify_mapping(main_manifest['artifact_sha256'], main)
    verify_mapping(audit_manifest['artifact_sha256'], audit)
    names = ('main_findings.md', 'allocation_comparison.png', 'values.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        complete=True, main_run=str(main.relative_to(ROOT)), audit=str(audit.relative_to(ROOT)),
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={name: sha(output/name) for name in names},
        python=platform.python_version(), matplotlib=matplotlib_version,
        numerical_reruns=False, descriptive_ratio_estimates=True,
        ratio_confidence_intervals=False, measured_mean_squared_error=False,
        validated_equal_budget=False, global_optimum_claim=False,
        validated_interval_coverage=False, publication_clearance=False))
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'audit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    arguments = parser.parse_args()
    result = report(arguments.main, arguments.audit, arguments.output)
    print(f"Reported {result['model_count']} audited conditional models; findings remain private.")
