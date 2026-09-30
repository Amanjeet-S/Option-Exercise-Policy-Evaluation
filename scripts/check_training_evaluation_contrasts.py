"""Link paired moments to the workloads whose financial quantities they compare."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from audit_training_evaluation import close, verify
from run_conditional_calibration import ROOT, private, save, sha


def compare_components(paired, right, left):
    """Reconstruct component moments from independently retained estimates."""
    count = paired['estimate']['observations']
    if right['observations'] != count or left['observations'] != count:
        raise ValueError('Component counts do not match the paired contrast')
    for side, estimate in (('right', right), ('left', left)):
        close(paired[f'{side}_mean'], estimate['value'])
        close(paired[f'{side}_variance'], estimate['standard_error']**2*count)
    close(paired['estimate']['value'], right['value']-left['value'])


def check(run, output):
    run, output = private(run), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh contrast verification directory')
    blocks, config, _, _ = verify(run)
    low, high = config['training_paths']
    counts = dict(policy_gains=0, evaluation_effects=0, gain_consistency=0)
    for block in blocks:
        workloads = {(w['training_paths'], w['method'], w['evaluation_paths']):
                     w['estimates_quote'] for w in block['workloads']}
        gains = {(g['method'], g['evaluation_paths']): g['estimate']
                 for g in block['policy_gains']}
        for paired in block['policy_gains']:
            method, size = paired['method'], paired['evaluation_paths']
            right, left = workloads[high, method, size], workloads[low, method, size]
            if right['upper_value'] != left['upper_value']:
                raise ValueError('The policies do not share the recorded dual upper estimate')
            compare_components(paired, right['policy_value'], left['policy_value'])
            close(paired['estimate']['value'], left['gap_bound']['value']-right['gap_bound']['value'])
            counts['policy_gains'] += 1
        for paired in block['evaluation_effects']:
            training, size = paired['training_paths'], paired['evaluation_paths']
            compare_components(paired, workloads[training, 'fine', size]['gap_bound'],
                               workloads[training, 'coarse', size]['gap_bound'])
            counts['evaluation_effects'] += 1
        for paired in block['gain_consistency']:
            size = paired['evaluation_paths']
            compare_components(paired, gains['fine', size], gains['coarse', size])
            counts['gain_consistency'] += 1
    output.mkdir(parents=True)
    save(output/'verification.json', dict(verified_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(run/'manifest.json'),
        source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        linked_contrasts=counts, numerical_paths_rerun=False, publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.run, args.output)
