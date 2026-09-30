"""Repeat the fixed block selection with JSON-normalised numerical records.

The registered auditor compares in-memory tuples with saved JSON lists. This
supplement preserves that registered source and corrects only the comparison
representation. Every numerical value, stream identity and policy is retained.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_training_evaluation import verify
from run_conditional_calibration import ROOT, private, read, save, sha
from study_training_evaluation import execute_block, numerical_record


def canonical_numerical_record(block):
    """Use the storage representation without rounding or numerical tolerance."""
    return json.loads(json.dumps(numerical_record(block), allow_nan=False))


def reproduce(run, output):
    run, output = private(run), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh reproduction directory')
    blocks, config, tasks, checks = verify(run)
    output.mkdir(parents=True)
    repeated = []
    for task in (tasks[0], tasks[-1]):
        for repetition in (0, config['training_repetitions']-1):
            row = execute_block(task, config, repetition)
            old = next(b for b in blocks if (b['case'], b['repetition']) ==
                       (task['case'], repetition))
            if canonical_numerical_record(row) != canonical_numerical_record(old):
                save(output/'differing_block.json', row)
                raise ValueError('Numerical rerun differs after JSON normalisation')
            name = f'repeated_case_{task["case"]}_rep_{repetition}.json'
            save(output/name, row)
            repeated.append([task['case'], repetition])
            print(f'Reproduced model {task["case"]}, repetition {repetition+1}', flush=True)
    save(output/'verification.json', dict(verified_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(run/'manifest.json'), checks=checks,
        repeated_blocks=repeated, full_numerical_rerun=False,
        comparison='Exact numerical equality after JSON list/tuple normalisation; timings excluded',
        numerical_tolerance=None, registered_sources_unchanged=True,
        registered_source_sha256=read(run/'manifest.json')['source_sha256'],
        reproduction_source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        artifact_sha256={p.name: sha(p) for p in output.glob('repeated_*.json')},
        publication_clearance=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.run, args.output)
