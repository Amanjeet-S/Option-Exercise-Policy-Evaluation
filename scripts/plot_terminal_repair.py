"""Private figures separating a policy improvement from conditional loss bounds."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_conditional_calibration import private, save, sha
from study_terminal_repair import verified_results


def plot(folder, output):
    folder, output = private(folder), private(output)
    if output.exists(): raise FileExistsError('Use a fresh private figure directory')
    rows, summary = verified_results(folder); output.mkdir(parents=True)
    variants = [(f['training_paths'], f['degree']) for f in summary['families']]
    labels = [f'{n//1000}k, degree {d}' for n, d in variants]
    colours = ['#8899a6', '#b95e3c', '#2984a5']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    selections = [(0, 'original', 'Original horizon'), (2, 'original', '+90 min, unchanged policy'),
                  (2, 'repaired', '+90 min, terminal adjustment')]
    for j, (scenario, name, label) in enumerate(selections):
        axes[0].bar(np.arange(4)+(j-1)*.25, [v['mean_gap_quote'][name] for v in summary['scenarios'][scenario]['variants']],
            width=.24, color=colours[j], label=label)
    axes[0].set(xticks=range(4), xticklabels=labels, ylabel='Mean dual gap (USD per share)', title='Loss before and after the terminal adjustment')
    axes[0].legend(fontsize=8)
    for j, variant in enumerate(variants):
        for seconds, offset, colour in [(900, -.13, '#2984a5'), (5400, .13, '#b95e3c')]:
            points = [p['direct_to_integrated_variance_ratio'] for r in rows for s in r['scenarios'] if s['extension_seconds'] == seconds
                      for p in s['policies'] if (p['training_paths'], p['degree']) == variant]
            axes[1].scatter(j+offset+np.linspace(-.07,.07,len(points)), points, color=colour, s=15, alpha=.65,
                label=f'+{seconds//60} minutes' if j == 0 else None)
    axes[1].set(xticks=range(4), xticklabels=labels, yscale='log', ylabel='Direct / integrated gain variance', title='Conditioning out the added diffusion')
    axes[1].legend(fontsize=9)
    for ax in axes: ax.grid(axis='y', alpha=.18)
    fig.suptitle('Terminal policy adjustment under unchanged financial inputs', fontsize=13)
    fig.savefig(output/'terminal_gain.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), constrained_layout=True)
    palette = ['#8899a6', '#b95e3c', '#2984a5', '#2e7c69']
    for j, variant in enumerate(variants):
        selected = [(r,f) for r in rows for f in r['families'] if (f['training_paths'],f['degree']) == variant]
        x = [f['maximum_upper_quote']['original']/r['spread'] for r,f in selected]
        y = [f['maximum_upper_quote']['repaired']/r['spread'] for r,f in selected]
        axes[0].scatter(x, y, color=palette[j], s=22, alpha=.65, label=labels[j])
        ratios = [p['gain_to_pde_increment'] for r in rows for s in r['scenarios'] if s['extension_seconds'] == 5400
                  for p in s['policies'] if (p['training_paths'],p['degree']) == variant]
        axes[1].scatter(j+np.linspace(-.15,.15,len(ratios)), ratios, color=palette[j], s=18, alpha=.65)
    limits = max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]); axes[0].plot([0,limits],[0,limits],ls=':',color='#555555',lw=1)
    axes[0].axhline(1,ls='--',color='#555555',lw=1); axes[0].axvline(1,ls='--',color='#555555',lw=1)
    axes[0].set(xlabel='Original family limit / observed spread', ylabel='Adjusted family limit / observed spread', title='All original policies and all three endpoints')
    axes[0].legend(fontsize=8)
    axes[1].axhline(1,ls='--',color='#555555',lw=1)
    axes[1].set(xticks=range(4),xticklabels=labels,ylabel='Integrated gain / PDE calendar increment',title='+90 minutes: a descriptive value comparison')
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle('Residual loss and the value of the added opportunity',fontsize=13)
    fig.savefig(output/'terminal_bounds.png',dpi=180);plt.close(fig)
    save(output/'manifest.json',dict(experiment_manifest_sha256=sha(folder/'manifest.json'),
        source_sha256=sha(Path(__file__).resolve()),figures={p.name:sha(p) for p in sorted(output.iterdir())},publication_clearance=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.run,args.output)
