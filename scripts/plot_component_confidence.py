"""Plot the scale and allocation implications of an audited component region."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/private/component_confidence/2026-09-28'
PREVIOUS = ROOT/'data/private/allocation_variance_sensitivity/2026-09-28'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=BASE/'audit_v1/checks.json')
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data/private') or output.exists():
        raise ValueError('Require a fresh private output directory')
    audit = read(args.audit)
    if audit['complete'] is not True:
        raise ValueError('A completed independent audit is required')
    evidence = dict(audit['evidence_sha256'])
    evidence[str(args.audit.resolve().relative_to(ROOT))] = sha(args.audit)
    for path in args.run.glob('*.json'):
        key = str(path.resolve().relative_to(ROOT))
        if key not in evidence or sha(path) != evidence[key]:
            raise ValueError('Result lacks audit coverage: '+key)
    previous = PREVIOUS/'run_v1/summary.json'
    if str(previous.relative_to(ROOT)) not in evidence:
        raise ValueError('Influence summary lacks audit coverage')
    for name, digest in evidence.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError('Changed evidence: '+name)
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    summary, influence = read(args.run/'summary.json'), read(previous)
    rows, old = summary['models'], influence['models']
    if [r['case'] for r in rows] != list(range(13)) or [r['case'] for r in old] != list(range(13)):
        raise ValueError('Case coverage differs')
    number = lambda value: float(Fraction(value))
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True, constrained_layout=True)
    y = np.arange(13)
    axes[0].scatter([number(r['variance_radius_to_observed']) for r in rows], y,
                    color='#294b70', marker='o', label='Pair-mean variance')
    axes[0].scatter([number(r['evaluation_radius_to_observed']) for r in rows], y,
                    color='#b76d28', marker='s', label='Mean conditional variance')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Upper-limit allowance / observed moment')
    axes[0].set_ylabel('Conditional model case')
    axes[0].set_title('Scale of the concentration allowances', loc='left')
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    axes[0].set_ylim(13.7, -.7)
    axes[0].set_yticks(y)
    lower = [number(r['share_bounds']['lower']) for r in rows]
    upper = [number(r['share_bounds']['upper']) for r in rows]
    axes[1].hlines(y, lower, upper, color='#9da5ad', linewidth=2,
                   label='Fixed-sample region, zero summary error')
    axes[1].hlines(y, [number(r['diagnostic_share_range'][0]) for r in old],
                   [number(r['diagnostic_share_range'][1]) for r in old],
                   color='#127f86', linewidth=5, label='Observed influence interval')
    axes[1].scatter([number(r['retained_share']) for r in old], y, color='#14383a', s=16, zorder=3)
    axes[1].set_xscale('log')
    axes[1].set_xlim(1e-11, 1.6)
    axes[1].set_xlabel('Evaluation share at 4,096 paths, w (log scale)')
    axes[1].set_title('Component ranges and observed influence', loc='left')
    axes[1].legend(loc='lower left', frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(output/f'component_uncertainty.{extension}', dpi=180)
    plt.close(fig)
    (output/'manifest.json').write_text(json.dumps(dict(complete=True, evidence_sha256=evidence,
         artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
         complete_statistical_coverage_certified=False, publication_clearance=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
