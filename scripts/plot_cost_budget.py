"""Private runtime and loss-bound diagnostics for the retained workload."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_conditional_calibration import private, read, save, sha, verify_mapping


def plot(folder, audit, output):
    folder, audit, output = private(folder), private(audit), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private figure directory')
    manifest, check = read(folder/'manifest.json'), read(audit/'verification.json')
    verify_mapping(manifest['artifact_sha256'], folder)
    if check['experiment_manifest_sha256'] != sha(folder/'manifest.json'):
        raise ValueError('Audit belongs to another experiment')
    rows = read(folder/'rows.json')
    methods = ('baseline', 'coarse', 'fine')
    labels = ('Predictable slope', 'Coarse value', 'Fine value')
    colours = ('#6C7788', '#117C83', '#C16B27')
    budgets = sorted({r['budget_seconds'] for r in rows})
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(budgets), 3, figsize=(13, 7.5), squeeze=False)
    plotted = []
    for i, budget in enumerate(budgets):
        ratios, gaps, radii = [], [], []
        for method in methods:
            selected = [r for r in rows if r['method'] == method and r['budget_seconds'] == budget]
            ratios.append([r['elapsed_seconds']/budget for r in selected])
            # One point per workload; do not present shared-path policies as independent replications.
            gaps.append([float(np.median([p['estimates_quote']['gap_bound']['upper'] for p in r['policies']]))
                         for r in selected])
            radii.append([float(np.median([p['estimates_quote']['gap_bound']['upper']-
                                          p['estimates_quote']['gap_bound']['value'] for p in r['policies']]))
                          for r in selected])
            for r, x, g, radius in zip(selected, ratios[-1], gaps[-1], radii[-1]):
                plotted.append(dict(case=r['case'], method=method, target_seconds=budget,
                    repetition=r['evaluation_repetition'], elapsed_ratio=x,
                    within_workload_median_upper_quote=g, within_workload_median_radius_quote=radius))
        for ax, values, title in zip(axes[i], (ratios, gaps, radii),
                ('Actual time / target time', 'Median upper loss limit', 'Median sampling radius')):
            bp = ax.boxplot(values, positions=range(3), widths=.5, patch_artist=True,
                            showfliers=False, medianprops={'color': '#111111', 'linewidth': 1.5})
            for patch, colour in zip(bp['boxes'], colours):
                patch.set_facecolor(colour)
                patch.set_alpha(.2)
            for j, (v, colour) in enumerate(zip(values, colours)):
                ax.scatter(j+np.linspace(-.12, .12, len(v)), v, s=12, alpha=.55,
                           color=colour, linewidths=0)
            ax.set_xticks(range(3), labels, rotation=12)
            ax.set_title(f'{budget:g} s target | {title}')
            ax.grid(axis='y', alpha=.15)
        axes[i, 0].axhline(1., color='#222222', linestyle='--', linewidth=1)
        axes[i, 0].set_ylabel('Ratio')
        for ax in axes[i, 1:]:
            ax.set_ylabel('Quoted-premium units')
    fig.suptitle('Computational budgets for fixed-policy loss measurement', fontsize=16, y=.985)
    fig.text(.02, .016,
        'Each point is one model and evaluation repetition. Bounds are conditional and pointwise. '
        'Warm-up and allocation-pilot costs are additional.', fontsize=9)
    fig.tight_layout(rect=(0., .055, 1., .95))
    output.mkdir(parents=True)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'cost_and_loss_bounds.{suffix}', dpi=170)
    plt.close(fig)
    save(output/'plotted_values.json', plotted)
    save(output/'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(folder/'manifest.json'), audit_sha256=sha(audit/'verification.json'),
        source_sha256=sha(Path(__file__)), publication_clearance=False,
        artifact_sha256={n: sha(output/n) for n in
                        ('cost_and_loss_bounds.png', 'cost_and_loss_bounds.svg', 'plotted_values.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.experiment, args.audit, args.output)
