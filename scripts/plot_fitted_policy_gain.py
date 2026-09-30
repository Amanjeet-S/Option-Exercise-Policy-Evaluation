"""Private policy-gain and loss-bound figures from verified retained results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_conditional_calibration import private, save, sha
from study_fitted_policy_gain import verified_results


def plot(folder, output):
    folder, output = private(folder), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh figure directory')
    rows, summary = verified_results(folder)
    output.mkdir(parents=True)
    cases = sorted({r['case'] for r in rows})
    keys = [(p['training_paths'], p['degree']) for p in rows[0]['policies']]
    labels = [f'{n//1000}k, degree {d}' for n, d in keys]
    colours = ['#246086', '#cb713c', '#498852', '#8955a1']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), constrained_layout=True)
    reference = [next(r['model_exercise_premium'] for r in rows if r['case'] == c) for c in cases]
    axes[0].plot(np.array(cases)+1, reference, color='#222222', marker='x', label='Numerical optimum')
    for j, key in enumerate(keys):
        values = [next(v['mean_gain'] for v in summary['by_case'] if (v['case'], v['training_paths'], v['degree']) == (c, *key)) for c in cases]
        axes[0].plot(np.array(cases)+1, values, marker='o', ms=4, lw=1, color=colours[j], label=labels[j])
        entries = [(r, r['policies'][j]) for r in rows]
        ratio = [p['duals']['policy_slope']['loss_upper_97_5_quote']/r['observed_spread'] for r, p in entries]
        offset = np.linspace(-.14, .14, len(ratio))
        axes[1].scatter(j+offset, ratio, color=colours[j], alpha=.65, s=14)
    axes[0].set(xlabel='Retained model, ordered by quote cutoff and strike', ylabel='Exercise gain (USD per share)',
                title='Policy gains: means across four repetitions', xticks=np.array(cases)[::2]+1)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(1., color='#222222', ls='--', lw=1, label='One observed bid-ask spread')
    axes[1].set(xticks=range(len(keys)), xticklabels=labels, yscale='log',
                ylabel='Loss upper bound / observed spread', title='Slope-martingale bound: every frozen policy')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle('Conditional fitted models | independent policy evaluation', fontsize=13)
    fig.savefig(output/'policy_gain_and_bounds.png', dpi=180)
    fig.savefig(output/'policy_gain_and_bounds.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pairs = [(c['left'], c['right']) for c in rows[0]['contrasts']]
    for j, pair in enumerate(pairs):
        entries = [next(c['controlled'] for c in r['contrasts'] if (c['left'], c['right']) == pair) for r in rows]
        x = j+np.linspace(-.2, .2, len(entries))
        ax.scatter(x, [c['value'] for c in entries], s=13, alpha=.55, color='#246086')
        # Display every pointwise interval, without selecting significant results.
        ax.vlines(x, [c['lower'] for c in entries], [c['upper'] for c in entries], lw=.5, alpha=.2, color='#246086')
    ax.axhline(0., color='#222222', lw=1)
    ax.set(xticks=range(len(pairs)), xticklabels=[f'{labels[b]}\nminus {labels[a]}' for a, b in pairs],
           ylabel='Paired exercise-gain difference (USD per share)',
           title='Every prescribed policy contrast\nApproximate 95% pointwise sampling intervals; no simultaneous claim')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'paired_policy_comparisons.png', dpi=180)
    fig.savefig(output/'paired_policy_comparisons.pdf')
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
