"""Produce tables and figures from audited complete-arm CPU prediction results."""

import argparse
import csv
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from study_prospective_allocation import ROOT, private, read, save, sha, utc, verify_manifest
from study_cpu_prediction import BASE


def report(run, audit, output):
    run, audit, output = map(private, (run, audit, output))
    if output.exists():
        raise FileExistsError('Use a fresh report directory')
    evidence = verify_manifest(audit/'manifest.json')
    results = [read(p) for p in sorted(run.glob('pass_*/result_*.json'))]
    results.sort(key=lambda r: (r['pass_id'], r['position']))
    if len(results) != 18:
        raise ValueError('Eighteen audited arms are required')
    records = []
    for result in results:
        row = {k: result[k] for k in ('pass_id', 'position', 'case', 'evaluation_paths', 'repetitions')}
        row.update(actual_cpu=float(Fraction(result['actual']['cpu'])),
            actual_elapsed=float(Fraction(result['actual']['elapsed'])),
            calibration_cpu=float(Fraction(result['calibration_charge']['cpu'])),
            charged_cpu=float(Fraction(result['charged_numerical_cost']['cpu'])),
            original_budget=float(Fraction(result['budget'])),
            charged_affordable=result['charged_cpu_affordable'])
        for name in ('historical', 'pass_start', 'immediate'):
            for clock, comparison in result['comparisons'][name].items():
                row[f'{name}_{clock}_ratio'] = float(Fraction(comparison['ratio']))
                row[f'{name}_{clock}_predicted'] = float(Fraction(comparison['predicted']))
                row[f'{name}_{clock}_within_tolerance'] = comparison['within_tolerance']
        records.append(row)
    output.mkdir(parents=True)
    with (output/'arms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6), sharex=True)
    colours = dict(historical='#666666', pass_start='#0b7285', immediate='#b65e18')
    labels = dict(historical='Historical', pass_start='Pass-start calibration', immediate='Immediate calibration')
    for pass_id, ax in enumerate(axes):
        selected = [r for r in records if r['pass_id'] == pass_id]
        ax.axvspan(.9, 1.1, color='#e6f0e6', zorder=0)
        ax.axvline(1, color='#555555', lw=.7, zorder=0)
        for name in colours:
            ax.plot([r[f'{name}_cpu_ratio'] for r in selected], range(9), 'o',
                    label=labels[name], color=colours[name], markersize=5)
        ax.set_yticks(range(9), [f"Case {r['case']}, N={r['evaluation_paths']:,}" for r in selected])
        ax.invert_yaxis()
        ax.set_title(f'Pass {pass_id+1}: execution order')
        ax.set_xlabel('Actual / predicted numerical CPU')
        ax.grid(axis='x', alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(.5, .01))
    fig.suptitle('Complete-arm CPU forecasts under local calibration', fontsize=13)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output/'cpu_prediction.png', dpi=170)
    fig.savefig(output/'cpu_prediction.svg')
    plt.close(fig)
    summary = read(run/'summary.json')
    lines = ['# Complete-arm CPU prediction', '',
        '| Predictor | Within 10 per cent | Within upper allowance | Mean absolute relative error | Ratio range |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name, value in summary['comparisons'].items():
        lines.append(f"| {labels[name]} | {value['within_tolerance']}/18 | {value['within_allowance']}/18 | "
            f"{100*float(Fraction(value['mean_absolute_relative_error'])):.3f}% | "
            f"{float(Fraction(value['ratio_min'])):.4f} to {float(Fraction(value['ratio_max'])):.4f} |")
    lines += ['', 'The upper allowance is 1.1 times the forecast. It is a planning margin without a coverage interpretation.', '']
    (output/'tables.md').write_text('\n'.join(lines))
    tex = [r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\caption{Complete-arm CPU prediction across eighteen fixed validation arms.}',
        r'\label{tab:cpu-prediction}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Predictor & Within 10\% & Above allowance & MARE (\%) & Actual/forecast \\',
        r'\midrule']
    for name, value in summary['comparisons'].items():
        tex.append(f"{labels[name]} & {value['within_tolerance']}/18 & "
            f"{18-value['within_allowance']}/18 & "
            f"{100*float(Fraction(value['mean_absolute_relative_error'])):.3f} & "
            f"{float(Fraction(value['ratio_min'])):.4f}--{float(Fraction(value['ratio_max'])):.4f} \\\\")
    tex += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip',
        r'\begin{minipage}{0.96\textwidth}\footnotesize',
        r'MARE is the equally weighted mean of $|C-\widehat C|/\widehat C$ across complete arms. '
        r'The allowance is $1.1\widehat C$, a descriptive planning margin. '
        r'Charges include a fresh validation setup and the fixed number of complete training and evaluation pairs. '
        r'Calibration expenditure is reported separately. The two passes assess local transfer and do not establish prediction coverage.',
        r'\end{minipage}', r'\end{table}', '']
    (output/'tables.tex').write_text('\n'.join(tex))
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    save(output/'manifest.json', dict(complete=True, recorded_at_utc=utc(), evidence_sha256=evidence,
        artifact_sha256={p.name: sha(p) for p in output.iterdir()}, publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--audit', type=Path, default=BASE/'audit_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.audit, args.output)
