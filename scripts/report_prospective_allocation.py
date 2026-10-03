"""Report a completed and independently audited prospective allocation study.

Rounded displays preserve the exact source decisions. Variance ratios are
descriptive; separate variance-difference intervals use asymptotic calibration.
"""

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'prospective_allocation/2026-09-28'
CASES = (5, 6, 10)
GRID = (128, 256, 4096)
COLOURS = {128: '#a55b23', 256: '#187c6a', 4096: '#456b94'}
MARKERS = {128: 'o', 256: 's', 4096: 'D'}
OFFSETS = {128: -.23, 256: 0, 4096: .23}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(items):
        result = {}
        for name, value in items:
            require(name not in result, 'Duplicate JSON field')
            result[name] = value
        return result

    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)

    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(value):
    require(type(value) is str, 'Require a retained exact rational string')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational string')
    return result


def number(value):
    if value is None:
        return None
    result = float(rational(value) if isinstance(value, str) else value)
    require(math.isfinite(result), 'Display conversion is not finite')
    return result


def display(value):
    return 'undefined' if value is None else f'{number(value):.7g}'


def encoded(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    return value


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(encoded(value), indent=2, allow_nan=False) + '\n')


def add(evidence, mapping, anchor=ROOT):
    anchor = anchor.resolve()
    require(isinstance(mapping, dict) and mapping, 'Missing evidence hash mapping')
    for name, digest in mapping.items():
        require(isinstance(name, str) and isinstance(digest, str), 'Malformed evidence mapping')
        path = (anchor / name).resolve()
        require(path.is_relative_to(anchor) and path.is_file(), 'Unsafe or absent evidence: ' + name)
        require(sha(path) == digest, 'Changed evidence: ' + name)
        relative = str(path.relative_to(ROOT))
        require(relative not in evidence or evidence[relative] == digest, 'Conflicting evidence: ' + relative)
        evidence[relative] = digest


def verify_stage(evidence, folder):
    path = folder / 'manifest.json'
    manifest = read(path)
    require(manifest.get('complete') is True, 'An incomplete stage cannot supply a report')
    require(not (folder / 'failure.json').exists(), 'A failed execution cannot supply a report')
    add(evidence, manifest['evidence_sha256'])
    add(evidence, manifest['artifact_sha256'], folder)
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != 'manifest.json'}
    require(actual == set(manifest['artifact_sha256']), 'A stage contains unbound or missing artifacts')
    evidence[str(path.relative_to(ROOT))] = sha(path)
    return manifest


def gather(registration, run, audit):
    evidence = {}
    registered = verify_stage(evidence, registration)
    executed = verify_stage(evidence, run)
    require(registered.get('stage') == 'prospective_allocation_registration', 'Incorrect registration stage')
    require(executed['registration_sha256'] == sha(registration / 'manifest.json'),
            'Execution used a different registration')
    checks = read(audit)
    require(checks.get('complete') is True, 'A complete independent audit is required')
    audited = checks['evidence_sha256']
    for folder in (registration, run):
        for path in folder.rglob('*'):
            if path.is_file():
                name = str(path.relative_to(ROOT))
                require(audited.get(name) == sha(path), 'The audit does not bind a consumed stage artifact: ' + name)
    add(evidence, audited)
    evidence[str(audit.relative_to(ROOT))] = sha(audit)
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).resolve())
    configuration = read(registration / 'configuration.json')
    allocations = read(registration / 'allocations.json')
    summary = read(run / 'summary.json')
    comparison_document = read(run / 'comparisons.json')
    require(configuration['cases'] == list(CASES) and configuration['evaluation_grid'] == list(GRID),
            'Unexpected case or allocation coverage')
    require(configuration['critical_value'] == '13/4' and configuration['contrast_family_size'] == 12,
            'Unexpected contrast calibration')
    require(summary['all_registered_counts_observed'] is True
            and summary['completion_status_source'] == 'manifest.json'
            and summary['arms'] == 9 and summary['primary_pairs'] == 3978,
            'Require all nine complete arms and their fixed pair counts')
    require(comparison_document['family_size'] == 12 and comparison_document['critical_value'] == '13/4',
            'Comparison family differs')
    for key in ('finite_sample_coverage_certified', 'measured_mean_squared_error', 'numerical_summary_errors_certified'):
        require(comparison_document[key] is False, 'Unexpected inference or numerical certification claim')
    model_map = {row['case']: row for row in allocations['models']}
    require(set(model_map) == set(CASES) and len(allocations['models']) == 3, 'Incomplete registered models')
    arms = []
    for case in CASES:
        for paths in GRID:
            arm = read(run / f'arm_{case:02d}_{paths}.json')
            require((arm['case'], arm['evaluation_paths'], arm['complete']) == (case, paths, True),
                    'Arm identity or completion differs')
            candidate = next(r for r in model_map[case]['candidates'] if r['evaluation_paths'] == paths)
            require(arm['repetitions'] == candidate['repetitions'] == arm['statistics']['pairs'],
                    'A completed count differs from registration')
            require(arm['budget'] == model_map[case]['budget'] and arm['planning_charge'] == candidate['planning_charge'],
                    'A reported cost comparator differs from registration')
            arms.append(arm)
    comparisons = comparison_document['comparisons']
    require(len(comparisons) == 6 and
            {(r['case'], r['evaluation_paths']) for r in comparisons} == {(c, n) for c in CASES for n in (128, 256)},
            'Incomplete or duplicated comparisons')
    for row in comparisons:
        require(row['reference_paths'] == 4096 and row['ratio_has_confidence_interval'] is False
                and row['critical_value'] == '13/4', 'Comparison interpretation differs')
        for key in ('mean_contrast', 'variance_contrast'):
            require(row[key]['asymptotic_only'] is True and row[key]['normal_interval_status'] in
                    ('positive', 'negative', 'includes_zero', 'degenerate'), 'Unexpected contrast status')
    require(sum(row['repetitions'] for row in arms) == summary['primary_pairs'], 'Pair count differs')
    return evidence, configuration, model_map, summary, arms, comparisons


def approximate_interval(contrast):
    """Only the display uses square roots; exact source statuses are retained."""
    difference = rational(contrast['difference'])
    radius_squared = rational(contrast['radius_squared'])
    require(radius_squared >= 0, 'Negative squared interval radius')
    if contrast['normal_interval_status'] == 'degenerate':
        return None, None
    radius = math.sqrt(number(radius_squared))
    centre = number(difference)
    return centre-radius, centre+radius


def arm_table(arms, models):
    rows = []
    for arm in arms:
        statistics, case, paths = arm['statistics'], arm['case'], arm['evaluation_paths']
        model = models[case]
        rows.append(dict(case=case, evaluation_paths=paths, repetitions=arm['repetitions'],
            mean_exact=statistics['mean'], mean_display=display(statistics['mean']),
            standard_error_display=display(math.sqrt(number(statistics['variance_of_mean']))),
            variance_of_mean_exact=statistics['variance_of_mean'],
            variance_of_mean_display=display(statistics['variance_of_mean']),
            numerical_cpu_seconds_exact=arm['exact_numerical_cpu'],
            numerical_cpu_seconds_display=display(arm['exact_numerical_cpu']),
            forecast_seconds_exact=arm['planning_charge'], forecast_seconds_display=display(arm['planning_charge']),
            budget_seconds_exact=arm['budget'], budget_seconds_display=display(arm['budget']),
            actual_to_forecast_exact=arm['actual_to_forecast'], actual_to_forecast_display=display(arm['actual_to_forecast']),
            actual_to_budget_exact=arm['actual_to_budget'], actual_to_budget_display=display(arm['actual_to_budget']),
            over_budget=arm['over_budget'], within_forecast_tolerance=arm['within_forecast_tolerance'],
            registered_nominal_choice=paths == model['nominal']['evaluation_paths'],
            registered_minimax_choice=paths == model['minimax']['selected']['evaluation_paths'],
            zero_gain_pairs=arm['zero_gain_pairs'],
            mean_variance_degenerate=statistics['mean_variance_degenerate'],
            variance_influence_degenerate=statistics['variance_influence_degenerate']))
    return rows


def comparison_table(comparisons):
    rows = []
    for record in sorted(comparisons, key=lambda r: (r['case'], r['evaluation_paths'])):
        row = dict(case=record['case'], evaluation_paths=record['evaluation_paths'], reference_paths=4096,
                   estimated_variance_ratio_exact=record['estimated_variance_ratio'],
                   estimated_variance_ratio_display=display(record['estimated_variance_ratio']),
                   actual_cpu_ratio_exact=record['actual_cpu_ratio'], actual_cpu_ratio_display=display(record['actual_cpu_ratio']),
                   estimated_variance_cpu_ratio_exact=record['estimated_variance_cpu_ratio'],
                   estimated_variance_cpu_ratio_display=display(record['estimated_variance_cpu_ratio']))
        for name in ('mean', 'variance'):
            contrast = record[name+'_contrast']
            low, high = approximate_interval(contrast)
            row.update({name+'_difference_exact': contrast['difference'],
                        name+'_difference_display': display(contrast['difference']),
                        name+'_sampling_variance_exact': contrast['estimated_sampling_variance'],
                        name+'_radius_squared_exact': contrast['radius_squared'],
                        name+'_interval_lower_display': display(low),
                        name+'_interval_upper_display': display(high),
                        name+'_interval_status_exact': contrast['normal_interval_status']})
        rows.append(row)
    return rows


def csv_table(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(arm_rows, comparison_rows, output):
    text = [
        '# Prospective comparison of computational allocations', '',
        '## 1. Design and estimand', '',
        'I compare three fixed allocations within each of retained cases 5, 6 and 10. '
        'Each independent row retrains the two exercise procedures on 5,000 and 20,000 paths '
        'and evaluates their paired gain on a fresh common sample. The inner evaluation counts '
        'are 128, 256 and 4,096; the number of independent training pairs is fixed separately '
        'for each arm. I selected these cases and counts from earlier evidence and froze the '
        'new design before observing its outcomes. The comparison is prospective conditional '
        'on that design and the retained financial specifications.', '',
        'The standard error is the square root of the sample variance of the independent '
        'row means divided by their registered count. It estimates uncertainty in a common '
        'procedure-level mean. These samples do not reveal that population mean, supply '
        'independent market observations or measure mean squared error.', '',
        '## 2. Complete arm estimates and computational expenditure', '',
        'I charge each arm for its own approximation construction, both training procedures '
        'and its retained evaluation work. The forecast is the largest of the retained '
        'planning costs used in registration; the budget is the corresponding case-specific '
        'CPU allowance. The ratio to the forecast and the ratio to the budget therefore '
        'answer different questions. The 0.9 to 1.1 tolerance applies to the former. An '
        'over-budget flag records actual numerical CPU greater than the allowance. Counts '
        'remain fixed when realised expenditure differs from its forecast.', '',
        '| Case | N | R | Mean gain | Standard error | Estimated variance | Numerical CPU, s | Forecast, s | Budget, s | CPU / forecast | CPU / budget | Over budget |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |',
    ]
    for row in arm_rows:
        fields = [row['case'], row['evaluation_paths'], row['repetitions'], row['mean_display'],
                  row['standard_error_display'], row['variance_of_mean_display'], row['numerical_cpu_seconds_display'],
                  row['forecast_seconds_display'], row['budget_seconds_display'], row['actual_to_forecast_display'],
                  row['actual_to_budget_display'], 'yes' if row['over_budget'] else 'no']
        text.append('| ' + ' | '.join(map(str, fields)) + ' |')
    text += [
        '', 'The CPU observations describe this execution on the recorded host. They do not '
        'guarantee future expenditure or establish an equal realised budget. Warm-up, previous '
        'studies, provenance checks, file handling and statistical reporting are additional '
        'research costs outside the arm-level numerical charge.', '',
        '## 3. Independent-arm contrasts', '',
        'I subtract the 4,096-path reference from each alternative arm within the same case. '
        'The twelve registered contrasts comprise six mean differences and six differences '
        'between estimated variances of the fixed-count mean. Normal intervals use the '
        'cutoff 13/4 and plug-in sampling variances. Their joint calibration is asymptotic, '
        'conditional on the frozen design; finite-sample coverage and the rounding error '
        'in the original numerical summaries remain unverified.', '',
        'I retain the interval status from exact rational source comparisons. Displayed '
        'endpoints use floating-point square roots and rounding. A degenerate status means '
        'that the estimated contrast sampling variance is zero, so I do not display an '
        'inferential interval. The estimated variance ratios are descriptive and have no '
        'confidence intervals.', '',
        '| Case | N | Estimated variance ratio | Mean difference | Mean interval, approximate | Mean status | Variance difference | Variance interval, approximate | Variance status |',
        '| ---: | ---: | ---: | ---: | :--- | :--- | ---: | :--- | :--- |',
    ]
    for row in comparison_rows:
        intervals = {}
        for name in ('mean', 'variance'):
            intervals[name] = ('undefined' if row[name+'_interval_status_exact'] == 'degenerate' else
                               '['+row[name+'_interval_lower_display']+', '+row[name+'_interval_upper_display']+']')
        fields = [row['case'], row['evaluation_paths'], row['estimated_variance_ratio_display'],
                  row['mean_difference_display'], intervals['mean'], row['mean_interval_status_exact'],
                  row['variance_difference_display'], intervals['variance'], row['variance_interval_status_exact']]
        text.append('| ' + ' | '.join(map(str, fields)) + ' |')
    text += [
        '', '## 4. Graphical comparison', '',
        '![Prospective variance and CPU comparisons](prospective_allocation.png)', '',
        'The left panel displays estimated variance divided by the within-case 4,096-path '
        'reference. Its markers have no error bars; the reference value of one is a '
        'normalisation when its estimated variance is positive. The middle panel displays '
        'actual numerical CPU divided by the case budget, with one marking the allowance. '
        'The right panel displays approximate intervals for variance differences in squared '
        'quoted premium units. Its vertical zero line and separate horizontal scale prevent '
        'reading those intervals as uncertainty bounds for the ratios. Open reference '
        'markers in the first two panels identify the 4,096-path arm.', '',
        'The accompanying CSV files retain exact source quantities alongside rounded '
        'displays. The report manifest binds the completed registration, execution and '
        'independent audit. I keep these derived numerical findings in private storage '
        'pending publication review.', '',
    ]
    with (output / 'tables.md').open('x') as stream:
        stream.write('\n'.join(text))


def render(arms, comparisons, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1, 1.25]}, layout='constrained')
    references = {row['case']: row for row in arms if row['evaluation_paths'] == 4096}
    ratios = []
    cpu_ratios = []
    for arm in arms:
        case, paths = arm['case'], arm['evaluation_paths']
        y = CASES.index(case)+OFFSETS[paths]
        baseline = rational(references[case]['statistics']['variance_of_mean'])
        variance = rational(arm['statistics']['variance_of_mean'])
        variance_ratio = variance/baseline if baseline > 0 else None
        cpu_ratio = rational(arm['actual_to_budget'])
        for ax, value in ((axes[0], variance_ratio), (axes[1], cpu_ratio)):
            if value is None:
                ax.text(.02, y, f'N={paths}: undefined', transform=ax.get_yaxis_transform(),
                        va='center', fontsize=8, color=COLOURS[paths])
                continue
            ax.scatter([number(value)], [y], color=COLOURS[paths], marker=MARKERS[paths], s=43,
                       facecolors='white' if paths == 4096 else COLOURS[paths], linewidths=1.3, zorder=3)
        if variance_ratio is not None:
            ratios.append(number(variance_ratio))
        cpu_ratios.append(number(cpu_ratio))
    for record in comparisons:
        paths, case = record['evaluation_paths'], record['case']
        y = CASES.index(case)+OFFSETS[paths]
        centre = number(record['variance_contrast']['difference'])
        low, high = approximate_interval(record['variance_contrast'])
        if low is None:
            axes[2].scatter([centre], [y], marker='x', color=COLOURS[paths], s=43)
            axes[2].annotate('degenerate', (centre, y), xytext=(5, 3), textcoords='offset points', fontsize=8)
        else:
            axes[2].hlines(y, low, high, color=COLOURS[paths], linewidth=1.8)
            axes[2].scatter([low, high], [y, y], marker='|', color=COLOURS[paths], s=34)
            axes[2].scatter([centre], [y], marker=MARKERS[paths], color=COLOURS[paths], s=32, zorder=3)
    for ax in axes[:2]:
        ax.axvline(1, color='#78828b', linestyle=':', linewidth=1, zorder=0)
        ax.set_xlim(left=0)
    if ratios:
        axes[0].set_xlim(0, 1.13*max(1, max(ratios)))
    axes[1].set_xlim(0, 1.13*max(1, max(cpu_ratios)))
    axes[2].axvline(0, color='#78828b', linestyle=':', linewidth=1, zorder=0)
    axes[2].ticklabel_format(axis='x', style='sci', scilimits=(-2, 2), useMathText=True)
    axes[0].set_title('Estimated variance ratio', loc='left', pad=10)
    axes[1].set_title('Observed CPU against budget', loc='left', pad=10)
    axes[2].set_title('Variance difference and interval', loc='left', pad=10)
    axes[0].set_xlabel('Estimated variance / 4,096-path reference')
    axes[1].set_xlabel('Numerical CPU / registered budget')
    axes[2].set_xlabel('Variance minus reference (squared premium units)')
    axes[0].set_yticks(range(len(CASES)), [f'Case {case}' for case in CASES])
    axes[0].set_ylim(len(CASES)-.5, -.7)
    for ax in axes:
        ax.grid(axis='x', alpha=.19)
        for boundary in (.5, 1.5):
            ax.axhline(boundary, color='#d6d6d6', linewidth=.6, zorder=0)
    handles = [Line2D([0], [0], marker=MARKERS[n], color=COLOURS[n], linestyle='none',
                       markerfacecolor='white' if n == 4096 else COLOURS[n], markersize=6,
                       label=f'N = {n:,}') for n in GRID]
    fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Fixed-count prospective allocations in three retained financial cases', fontsize=13)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'prospective_allocation.{suffix}', dpi=180)
    plt.close(fig)
    return dict(matplotlib_version=matplotlib.__version__, ratio_error_bars=False,
                variance_difference_endpoints='rounded floating-point display only')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=BASE / 'registration_v1')
    parser.add_argument('--run', type=Path, default=BASE / 'run_v1')
    parser.add_argument('--audit', type=Path, default=BASE / 'audit_v1/checks.json')
    parser.add_argument('--output', type=Path, default=BASE / 'report_v1')
    args = parser.parse_args()
    registration, run, audit, output = (p.resolve() for p in (args.registration, args.run, args.audit, args.output))
    require(all(p.is_relative_to(PRIVATE.resolve()) for p in (registration, run, audit, output)),
            'Require private numerical inputs and outputs')
    require(not output.exists(), 'Use a fresh report directory')
    evidence, configuration, models, summary, arms, comparisons = gather(registration, run, audit)
    arm_rows, comparison_rows = arm_table(arms, models), comparison_table(comparisons)
    add({}, evidence)
    output.mkdir(parents=True)
    csv_table(output / 'arms.csv', arm_rows)
    csv_table(output / 'comparisons.csv', comparison_rows)
    markdown(arm_rows, comparison_rows, output)
    plotting = render(arms, comparisons, output)
    save(output / 'summary.json', dict(complete=True, cases=list(CASES), arms=len(arms),
         comparisons=len(comparisons), primary_pairs=summary['primary_pairs'],
         over_budget_arms=summary['over_budget_arms'], within_forecast_tolerance=summary['within_forecast_tolerance'],
         critical_value=configuration['critical_value'], contrast_family_size=configuration['contrast_family_size'],
         ratio_confidence_intervals=False, finite_sample_coverage_certified=False,
         numerical_summary_errors_certified=False, cpu_guarantee=False,
         measured_mean_squared_error=False, independent_audit_complete=True, publication_clearance=False))
    add({}, evidence)
    save(output / 'manifest.json', dict(complete=True, recorded_at_utc=datetime.now(timezone.utc).isoformat(),
         registration_sha256=sha(registration / 'manifest.json'), run_sha256=sha(run / 'manifest.json'),
         audit_sha256=sha(audit), source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
         evidence_sha256=evidence, artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
         environment=dict(python_version=platform.python_version(), **plotting), publication_clearance=False))
    print(json.dumps(dict(report=str(output.relative_to(ROOT)), arms=len(arms), comparisons=len(comparisons),
                          independent_audit_complete=True)))


if __name__ == '__main__':
    main()
