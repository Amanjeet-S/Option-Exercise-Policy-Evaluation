"""Display retained ratio sets, allocation resolution and influence concentration.

Endpoint midpoints are used for display only. Exact quadratic sets and their
allocation mappings remain in the verified numerical records.
"""

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/private/component_ratio/2026-09-28'
SOURCES = ('pilot', 'reference', 'selected')
LABELS = {'pilot': 'Original pilot', 'reference': 'Main reference', 'selected': 'Main selected'}
COLOURS = {'pilot': '#9a571f', 'reference': '#376d9a', 'selected': '#17816b'}
MARKERS = {'pilot': 'o', 'reference': 's', 'selected': 'D'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(mapping, anchor=ROOT):
    anchor = anchor.resolve()
    for name, digest in mapping.items():
        path = (anchor / name).resolve()
        require(path.is_relative_to(anchor) and sha(path) == digest,
                'Changed or unsafe evidence: ' + name)


def add(evidence, mapping, anchor=ROOT):
    verify(mapping, anchor)
    for name, digest in mapping.items():
        relative = str((anchor / name).resolve().relative_to(ROOT))
        require(relative not in evidence or evidence[relative] == digest,
                'Conflicting evidence: ' + relative)
        evidence[relative] = digest


def midpoint(endpoint):
    lower, upper = Fraction(endpoint['lower']), Fraction(endpoint['upper'])
    require(lower <= upper, 'Reversed endpoint enclosure')
    return (lower + upper) / 2


def number(value):
    return float(Fraction(value))


def display(value):
    return 'undefined' if value is None else f'{number(value):.6g}'


def interval_text(parts, kind):
    if kind == 'empty':
        return 'empty'
    if kind == 'full_domain':
        return '[0, 1] (full domain)'
    return ' union '.join(f'[{display(left)}, {display(right)}]' for left, right in parts)


def gather(run, audit):
    evidence = {}
    inputs = BASE / 'inputs_v1'
    for directory in (inputs, run):
        manifest = read(directory / 'manifest.json')
        require(manifest['complete'] is True, 'Incomplete numerical evidence')
        add(evidence, manifest['evidence_sha256'])
        add(evidence, manifest['artifact_sha256'], directory)
        evidence[str((directory / 'manifest.json').relative_to(ROOT))] = sha(directory / 'manifest.json')
    audited = audit.is_file()
    if audited:
        checks = read(audit)
        require(checks['complete'] is True, 'The available independent audit is incomplete')
        for path in run.glob('*.json'):
            require(checks['evidence_sha256'].get(str(path.relative_to(ROOT))) == sha(path),
                    'Result lacks coverage in the available independent audit')
        add(evidence, checks['evidence_sha256'])
        evidence[str(audit.relative_to(ROOT))] = sha(audit)
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    summary, enclosures = read(run / 'summary.json'), read(run / 'enclosures.json')
    require(summary['complete'] is True and summary['component_sets'] == 39,
            'Require the complete 39-set numerical result')
    require(enclosures['input_sha256'] == sha(run / 'endpoint_input.json'),
            'Endpoint enclosure input differs')
    endpoint_rows = {(r['case'], r['source']): r for r in enclosures['sets']}
    require(len(endpoint_rows) == 39, 'Endpoint identities are incomplete or duplicated')
    records = []
    for case in range(13):
        for source in SOURCES:
            result = read(run / f'case_{case:02d}_{source}.json')
            require((result['case'], result['source']) == (case, source), 'Record identity differs')
            enclosed = endpoint_rows[case, source]['components']
            components = result['region']['components']
            require(len(enclosed) == len(components), 'Enclosed component count differs')
            parts = []
            for exact, rounded in zip(components, enclosed):
                require(all(rounded[key]['spec'] == exact[key] for key in ('lower', 'upper')),
                        'Enclosed and exact endpoint specifications differ')
                lower, upper = midpoint(rounded['lower']), midpoint(rounded['upper'])
                require(0 <= lower <= upper <= 1, 'Displayed physical interval differs')
                parts.append((lower, upper))
            moments = result['moments']
            require(len(result['regimes']) == 6, 'Require all six fixed cost regimes')
            records.append(dict(case=case, source=source, moments=moments, parts=parts,
                physical_kind=result['region']['physical_kind'],
                unrestricted_kind=result['region']['unrestricted_kind'],
                uniquely_resolved_regimes=sum(r['uniformly_selected_size'] is not None for r in result['regimes']),
                selected_size_counts=[len(r['selected_sizes']) for r in result['regimes']]))
    verify(evidence)
    return evidence, records, audited


def write_tables(records, output):
    fields = ['case', 'source', 'pairs', 'inner_paths', 'point_share', 'point_is_admissible',
              'physical_set_kind', 'unrestricted_set_kind', 'approximate_ratio_set',
              'covariance_rank', 'largest_ratio_contrast_squared_share',
              'equal_magnitude_squared_share', 'uniquely_resolved_regimes_out_of_six']
    rows = []
    for row in records:
        moments = row['moments']
        rows.append(dict(zip(fields, [row['case'], row['source'], moments['pairs'], moments['paths'],
            display(moments['point_share']), moments['point_in_parameter_domain'],
            row['physical_kind'], row['unrestricted_kind'], interval_text(row['parts'], row['physical_kind']),
            moments['covariance_rank'], display(moments['largest_squared_influence_share']['ratio_contrast']),
            display(Fraction(1, moments['pairs'])), row['uniquely_resolved_regimes']])))
    with (output / 'component_ratio_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# Retrospective component-ratio comparisons', '',
        '## 1. Interpretation', '',
        'I report the three retained sources separately. The ratio is the evaluation-variance contribution '
        'at 4,096 paths divided by the total variance at that count. The sets use the registered normal '
        'cutoff 13/4 and joint moment covariance. Their family calibration is asymptotic; neither the '
        'numerical enclosures nor agreement between sources establishes finite-sample coverage.', '',
        'I display rounded midpoints of the rigorous endpoint enclosures. Exact set topology and '
        'allocation decisions remain in the numerical records. Empty sets, full-domain sets and '
        'disconnected components retain their separate meanings. A uniquely resolved regime selects '
        'one evaluation size throughout its nonempty accepted set, conditional on the retained costs.', '',
        'The influence column reports the largest squared ratio-contrast contribution divided by '
        'the sum of squared contributions. The equal-magnitude reference is 1/m for m retained pairs; '
        'it is not a normality threshold. All displayed quantities are derived results.', '',
        '## 2. Source comparisons', '',
        '| Case | Source | Pairs m | Inner paths | Point ratio | Accepted ratio set, approximate | Covariance rank | Largest squared influence share | Resolved regimes / 6 |',
        '| ---: | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |',
    ]
    for row in rows:
        point = row['point_share'] + ('' if row['point_is_admissible'] else ' (inadmissible or undefined)')
        text.append(f"| {row['case']} | {LABELS[row['source']]} | {row['pairs']} | {row['inner_paths']} | "
                    f"{point} | {row['approximate_ratio_set']} | {row['covariance_rank']} | "
                    f"{row['largest_ratio_contrast_squared_share']} | {row['uniquely_resolved_regimes_out_of_six']} |")
    text.extend(['', '## 3. Figure', '',
        '![Ratio sets and influence concentration](component_ratio.png)', '',
        'I preserve separate interval components in the left panel. Filled markers show admissible '
        'point estimates. The right panel compares observed concentration with the open-marker '
        'equal-magnitude reference. The sources have different pair counts and, for the main selected '
        'sample, different inner evaluation counts. These distinctions preclude reading a narrower '
        'set as a controlled sample-size experiment.', ''])
    (output / 'component_ratio_table.md').write_text('\n'.join(text))
    return rows


def render(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.8), sharey=True,
                             gridspec_kw={'width_ratios': [1.65, 1]}, layout='constrained')
    endpoints = [float(x) for row in records for part in row['parts'] for x in part]
    admissible_points = [number(row['moments']['point_share']) for row in records
                         if row['moments']['point_in_parameter_domain']]
    points = endpoints + admissible_points
    positive = [x for x in points if x > 0]
    has_zero = any(x == 0 for x in points)
    if points and not has_zero:
        axes[0].set_xscale('log')
        axes[0].set_xlim(min(positive)*0.77, 1.13)
        scale_label = 'log scale'
        axes[0].xaxis.set_major_locator(FixedLocator([.005, .01, .02, .05, .1, .2, .5, 1]))
    else:
        threshold = min(positive)/10 if positive else .001
        axes[0].set_xscale('symlog', linthresh=threshold)
        axes[0].set_xlim(0, 1.13)
        scale_label = f'symmetric log scale; linear below {threshold:.2g}'
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{100*value:g}'))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{100*value:g}'))
    for row in records:
        source, moments = row['source'], row['moments']
        y = row['case'] + {'pilot': -.23, 'reference': 0, 'selected': .23}[source]
        colour, marker = COLOURS[source], MARKERS[source]
        for lower, upper in row['parts']:
            axes[0].hlines(y, float(lower), float(upper), color=colour, linewidth=2)
            axes[0].scatter([float(lower), float(upper)], [y, y], marker='|', color=colour, s=30)
        if row['physical_kind'] in ('empty', 'full_domain'):
            axes[0].text(.995, y, row['physical_kind'].replace('_', ' '),
                         transform=axes[0].get_yaxis_transform(), ha='right', va='center', fontsize=7, color=colour)
        point = moments['point_share']
        if moments['point_in_parameter_domain']:
            axes[0].scatter([number(point)], [y], color=colour, marker=marker, s=23, zorder=4)
        elif point is None:
            axes[0].text(.01, y, 'undefined point', transform=axes[0].get_yaxis_transform(),
                         fontsize=7, color=colour, va='center')
        else:
            negative = Fraction(point) < 0
            boundary = axes[0].get_xlim()[0] if negative else 1
            axes[0].scatter([boundary], [y], marker='<' if negative else '>', color=colour, s=35, clip_on=False)
            axes[0].annotate(f'point {display(point)}', (boundary, y), xytext=(5 if negative else -5, 4),
                             textcoords='offset points', ha='left' if negative else 'right', fontsize=7, color=colour)
        influence = moments['largest_squared_influence_share']['ratio_contrast']
        if influence is not None:
            benchmark = 1/moments['pairs']
            axes[1].hlines(y, benchmark, number(influence), color=colour, alpha=.3, linewidth=1)
            axes[1].scatter([benchmark], [y], color=colour, facecolor='white', marker=marker, s=20, zorder=3)
            axes[1].scatter([number(influence)], [y], color=colour, marker=marker, s=23, zorder=4)
        else:
            axes[1].text(.01, y, 'undefined contrast', fontsize=7, va='center', color=colour)
    handles = [Line2D([0], [0], color=COLOURS[source], marker=MARKERS[source],
                      linewidth=2, markersize=5, label=LABELS[source]) for source in SOURCES]
    axes[0].legend(handles=handles, loc='upper right', frameon=False, fontsize=9)
    axes[1].legend(handles=[Line2D([0], [0], marker='o', color='#677078', linestyle='none',
                      markerfacecolor='white', markersize=5, label='Equal-magnitude reference, 1/m')],
                   loc='upper right', frameon=False, fontsize=9)
    axes[0].set_title('Accepted ratio sets and point estimates', loc='left', pad=11)
    axes[1].set_title('Concentration of estimated uncertainty', loc='left', pad=11)
    axes[0].set_xlabel('Evaluation share at 4,096 paths, w (per cent; ' + scale_label + ')')
    axes[1].set_xlabel('Largest squared ratio-contrast influence share (per cent)')
    axes[0].set_ylabel('Retained financial case')
    axes[0].set_ylim(12.65, -2.1)
    axes[0].set_yticks(range(13))
    axes[1].set_xlim(-.012, 1)
    axes[1].xaxis.set_major_locator(FixedLocator([0, .2, .4, .6, .8, 1]))
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        for boundary in range(12):
            ax.axhline(boundary+.5, color='#dddddd', linewidth=.45, zorder=0)
    fig.suptitle('Joint component uncertainty in three retained training-pair samples', fontsize=13)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'component_ratio.{extension}', dpi=180)
    plt.close(fig)
    return scale_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE / 'run_v1')
    parser.add_argument('--audit', type=Path, default=BASE / 'audit_v1/checks.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, audit, output = (p.resolve() for p in (args.run, args.audit, args.output))
    require(all(p.is_relative_to(ROOT / 'data/private') for p in (run, audit, output)),
            'Numerical inputs and outputs must remain private')
    require(not output.exists(), 'Require a fresh private output directory')
    evidence, records, audited = gather(run, audit)
    output.mkdir(parents=True)
    rows = write_tables(records, output)
    scale = render(records, output)
    summary = dict(complete=True, cases=13, sources=3, comparison_rows=len(rows),
        fixed_cost_comparisons=sum(len(row['selected_size_counts']) for row in records),
        source_summaries=[dict(source=source,
            uniquely_resolved_regimes=sum(row['uniquely_resolved_regimes'] for row in records if row['source'] == source),
            comparisons=78) for source in SOURCES],
        horizontal_scale=scale, endpoint_display='Midpoints of retained rigorous enclosures',
        independent_audit_included=audited, finite_sample_coverage_certified=False,
        publication_clearance=False, new_simulations=0)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    verify(evidence)
    manifest = dict(complete=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        evidence_sha256=evidence, artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
        independent_audit_included=audited, finite_sample_coverage_certified=False,
        publication_clearance=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
