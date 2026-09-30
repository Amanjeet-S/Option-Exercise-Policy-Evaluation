"""Render observed component influence and allocation changes from audited files."""

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
BASE = ROOT/'data/private/allocation_variance_sensitivity/2026-09-28'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--audit', type=Path, default=BASE/'audit_v1/checks.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data/private') or output.exists():
        raise ValueError('Require a fresh private output directory')
    audit = read(args.audit)
    if audit['complete'] is not True:
        raise ValueError('Require a completed independent audit')
    evidence = dict(audit['evidence_sha256'])
    evidence[str(args.audit.resolve().relative_to(ROOT))] = sha(args.audit)
    for p in args.run.glob('*.json'):
        key = str(p.resolve().relative_to(ROOT))
        if key not in evidence or sha(p) != evidence[key]:
            raise ValueError('Run artifact lacks audit coverage: '+key)
    for name, digest in evidence.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != digest:
            raise ValueError('Changed evidence: '+name)
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    data = read(args.run/'summary.json')
    cases = data['models']
    output.mkdir(parents=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.5), constrained_layout=True)
    for ax, account, title in zip(axes, ('marginal', 'screening_charged'),
                                  ('Existing cost information', 'Three-size timing pilot charged')):
        counts = np.array([[next(r['changed_deletions'] for r in c['regimes']
                                if r['scope'] == scope and r['accounting'] == account)
                            for c in cases] for scope in ('pilot', 'validation', 'joint')])
        im = ax.imshow(counts, vmin=0, vmax=32, cmap='Blues', aspect='auto')
        for i in range(3):
            for j in range(13):
                ax.text(j, i, str(counts[i, j]), ha='center', va='center',
                        color='white' if counts[i, j] >= 18 else '#182b40')
        ax.set_xticks(range(13), [str(c['case']) for c in cases])
        ax.set_yticks(range(3), ['Pilot costs', 'Validation costs', 'Joint costs'])
        ax.set_title(title, loc='left')
        ax.set_xlabel('Conditional model case')
    fig.colorbar(im, ax=axes, label='Changed choices among 32 whole-pair deletions', shrink=.85)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'deletion_choices.{suffix}', dpi=180)
    plt.close(fig)

    full, lows, highs = [], [], []
    for c in cases:
        row = read(args.run/f"case_{c['case']:02d}.json")
        valid = [100*float(Fraction(d['share'])) for d in row['deletions'] if d['share'] is not None]
        if not valid:
            raise ValueError('Influence plot needs at least one admissible deletion')
        full.append(100*float(Fraction(row['retained_share'])))
        lows.append(min(valid)); highs.append(max(valid))
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    y = np.arange(len(cases))
    ax.hlines(y, lows, highs, linewidth=2.2, color='#2a6f97', label='Range over 32 deletions')
    ax.scatter(full, y, color='#172f43', s=27, zorder=3, label='Retained full pilot')
    ax.set_yticks(y, [str(c['case']) for c in cases])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel('Evaluation contribution at 4,096 paths (per cent)')
    ax.set_ylabel('Conditional model case')
    ax.set_title('Influence of individual fitted-policy pairs', loc='left')
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='best', frameon=False)
    fig.text(.99, -.015, 'Deletion ranges are descriptive, not confidence intervals.',
             ha='right', fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(output/f'component_influence.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    artifacts = {p.name: sha(p) for p in sorted(output.iterdir())}
    (output/'manifest.json').write_text(json.dumps(dict(complete=True, evidence_sha256=evidence,
         artifact_sha256=artifacts, publication_clearance=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
