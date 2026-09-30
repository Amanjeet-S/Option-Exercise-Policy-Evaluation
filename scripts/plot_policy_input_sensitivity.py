"""Private scenario coverage, paired gap changes and finite-family limits."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_conditional_calibration import private, save, sha
from study_policy_input_sensitivity import verified_results


def plot(folder, output):
    folder, output = private(folder), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private figure directory')
    rows, summary = verified_results(folder)
    output.mkdir(parents=True)
    variants = [(f['training_paths'], f['degree']) for f in summary['families']]
    labels = [f'{n//1000}k, degree {d}' for n, d in variants]
    scenarios = summary['scenarios']
    names = [s['scenario'] for s in scenarios]
    scenario_labels = [name.replace('cash_', 'Cash fraction ').replace('_funding_', ', ') for name in names]
    counts = np.array([[next((v['pointwise_bounds_below_spread'] for v in s['variants']
        if (v['training_paths'], v['degree']) == variant), np.nan) for variant in variants] for s in scenarios])
    changes = np.array([[next((v['mean_paired_gap_change']*1e4 for v in s['variants']
        if (v['training_paths'], v['degree']) == variant), np.nan) for variant in variants] for s in scenarios])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7), constrained_layout=True)
    first = axes[0].imshow(counts, vmin=0, vmax=max(f['policies'] for f in summary['families']), cmap='Blues', aspect='auto')
    span = max(float(np.nanmax(np.abs(changes))), 1e-8)
    second = axes[1].imshow(changes, vmin=-span, vmax=span, cmap='RdBu_r', aspect='auto')
    for ax, values, label in zip(axes, (counts, changes), ('d', '.2f')):
        ax.set(xticks=range(len(labels)), xticklabels=labels, yticks=range(len(names)),
            yticklabels=[f'{s} ({c["evaluated_models"]}/{c["total_models"]})' for s,c in zip(scenario_labels, scenarios)])
        ax.tick_params(axis='x', rotation=15)
        for i in range(len(names)):
            for j in range(len(labels)):
                value = values[i, j]
                text = 'Missing' if np.isnan(value) else (str(int(value)) if label == 'd' else f'{value:.2f}')
                dark = (value > np.nanmax(values)*.6 if label == 'd' else abs(value) > span*.65)
                ax.text(j, i, text, ha='center', va='center', color='white' if dark else '#172531', fontsize=9)
    axes[0].set_title('Pointwise 97.5% upper limits below the spread')
    axes[1].set_title('Mean paired change in dual gap from baseline')
    fig.colorbar(first, ax=axes[0], shrink=.7, label='Number of frozen policies')
    fig.colorbar(second, ax=axes[1], shrink=.7, label='0.0001 USD per share')
    fig.suptitle('Local cash and diagonal funding scenarios | model coverage in brackets', fontsize=13)
    fig.savefig(output/'scenario_comparison.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for j, variant in enumerate(variants):
        selected = [(r, i, f) for r in rows for i, f in enumerate(r['families'])
                    if (f['training_paths'], f['degree']) == variant]
        x = j+np.linspace(-.17, .17, len(selected))
        baseline = [r['scenarios'][0]['policies'][i]['family_adjusted_upper_quote']/f['observed_spread'] for r,i,f in selected]
        maximum = [f['maximum_family_adjusted_upper_quote']/f['observed_spread'] for _,_,f in selected]
        axes[0].vlines(x, baseline, maximum, color='#7194a5', lw=.8, alpha=.6)
        axes[0].scatter(x, baseline, s=13, color='#8d9aa5', alpha=.65, label='Baseline, same tail allocation' if j == 0 else None)
        axes[0].scatter(x, maximum, s=20, color='#246b8e', alpha=.8, label='Maximum across evaluated scenarios' if j == 0 else None)
        all_changes = [p['paired_gap_change'] for r in rows for s in r['scenarios'][1:] for p in s['policies']
                       if (p['training_paths'], p['degree']) == variant]
        z = j+np.linspace(-.18, .18, len(all_changes))
        axes[1].vlines(z, [e['lower']*1e4 for e in all_changes], [e['upper']*1e4 for e in all_changes],
                       lw=.5, alpha=.2, color='#246b8e')
        axes[1].scatter(z, [e['value']*1e4 for e in all_changes], s=6, alpha=.35, color='#246b8e')
    axes[0].axhline(1., ls='--', color='#222222', lw=1)
    axes[0].set(title='Finite-family limits for each original policy', ylabel='Quoted upper limit / observed spread')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[1].axhline(0., color='#222222', lw=1)
    axes[1].set(title='Every non-baseline paired comparison, 95% intervals', ylabel='Gap change (0.0001 USD per share)')
    for ax in axes:
        ax.set(xticks=range(len(labels)), xticklabels=labels)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(alpha=.15)
    complete = sum(f['full_family'] for r in rows for f in r['families'])
    total = sum(len(r['families']) for r in rows)
    fig.suptitle(f'Conditional inference | {complete}/{total} complete prescribed policy families', fontsize=13)
    fig.savefig(output/'family_bounds.png', dpi=180)
    plt.close(fig)
    save(output/'manifest.json', dict(experiment_manifest_sha256=sha(folder/'manifest.json'),
        source_sha256=sha(Path(__file__).resolve()), figures={p.name: sha(p) for p in sorted(output.iterdir())},
        publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.run, args.output)
