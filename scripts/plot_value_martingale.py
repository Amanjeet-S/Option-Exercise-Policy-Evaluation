"""Private comparisons of martingale tightness and shared evaluation cost."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, StrMethodFormatter
import numpy as np

from run_conditional_calibration import private, save, sha
from study_value_martingale import verified_results


def plot(folder, output):
    folder, output = private(folder), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private figure directory')
    rows, summary = verified_results(folder)
    output.mkdir(parents=True)
    variants = [(v['training_paths'], v['degree']) for v in summary['variants']]
    labels = [f'{n//1000}k, degree {d}' for n, d in variants]
    colours = {'baseline': '#b56b40', 'coarse': '#7b8191', 'fine': '#246b8e'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for method in colours:
        axes[0].plot(range(4), [v['methods'][method]['mean_gap_quote'] for v in summary['variants']],
                     'o-', color=colours[method], label=method, lw=1.6)
    axes[0].set(yscale='log', xticks=range(4), xticklabels=labels,
        ylabel='Mean dual gap (USD per share)', title='Same policies and shared evaluation paths')
    axes[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    axes[0].legend()
    for j, variant in enumerate(variants):
        selected = [(r, p) for r in rows for p in r['policies'] if (p['training_paths'], p['degree']) == variant]
        ratio = [p['results']['fine']['loss_upper_97_5_quote']/r['observed_spread'] for r, p in selected]
        axes[1].scatter(j+np.linspace(-.16, .16, len(ratio)), ratio, color=colours['fine'], s=17, alpha=.65)
    axes[1].axhline(1., color='#222222', ls='--', lw=1, label='One observed spread')
    axes[1].set(xticks=range(4), xticklabels=labels, ylabel='Fine loss upper bound / observed spread',
        title='Every approximate pointwise 97.5% bound')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        ax.grid(alpha=.18)
    fig.suptitle('Analytically centred value martingales | conditional fitted models', fontsize=13)
    fig.savefig(output/'bounds_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for j, variant in enumerate(variants):
        selected = [p['paired_gap_comparisons']['fine_minus_coarse'] for r in rows for p in r['policies']
                    if (p['training_paths'], p['degree']) == variant]
        x = j+np.linspace(-.16, .16, len(selected))
        axes[0].scatter(x, [c['value']*1e4 for c in selected], color=colours['fine'], s=13, alpha=.6)
        axes[0].vlines(x, [c['lower']*1e4 for c in selected], [c['upper']*1e4 for c in selected],
                      color=colours['fine'], alpha=.25, lw=.6)
    axes[0].axhline(0., color='#222222', lw=1)
    axes[0].set(xticks=range(4), xticklabels=labels, ylabel='Fine minus coarse gap (0.0001 USD per share)',
                title='Paired refinement effect with 95% intervals')
    cases = [r['case']+1 for r in rows]
    for method in ('coarse', 'fine'):
        axes[1].plot(cases, [r['shared_costs'][method]['martingale_seconds'] for r in rows],
                     'o-', label=method, color=colours[method])
    axes[1].set(xlabel='Retained model', ylabel='Shared martingale construction (seconds)',
                title='One construction serves all frozen policies')
    axes[1].legend()
    axes[0].tick_params(axis='x', rotation=15)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('Resolution and computational cost | no simultaneous coverage claim', fontsize=13)
    fig.savefig(output/'resolution_and_cost.png', dpi=180)
    plt.close(fig)
    save(output/'manifest.json', dict(experiment_manifest_sha256=sha(folder/'manifest.json'),
        source_sha256=sha(Path(__file__).resolve()),
        figures={p.name: sha(p) for p in sorted(output.iterdir())}, publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)
