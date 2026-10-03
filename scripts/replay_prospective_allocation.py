"""Replay three registered financial rows after primary timing has completed."""

import argparse
import os
from pathlib import Path
import time

from study_prospective_allocation import (
    BASE, ROOT, EXPECTED, THREADS, build_approximation, load_frozen, merge, pair,
    private, read, save, sha, utc, verify_manifest,
)


def financial(value):
    if isinstance(value, dict):
        return {k: financial(v) for k, v in value.items()
                if not any(word in k for word in ('cpu_seconds', 'elapsed_seconds', 'simulation_seconds', 'fitting_seconds'))}
    if isinstance(value, list):
        return [financial(v) for v in value]
    return value


def replay(registration, run, output):
    registration, run, output = map(private, (registration, run, output))
    if output.exists():
        raise FileExistsError('Use a fresh replay directory')
    evidence = verify_manifest(run/'manifest.json')
    merge(evidence, verify_manifest(registration/'manifest.json'))
    if any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Use the registered numerical thread settings')
    if read(registration/'configuration.json') != EXPECTED:
        raise ValueError('Replay configuration differs')
    tasks = {t['case']: t for t in read(registration/'tasks.json')}
    forbidden = set(read(registration/'allocations.json')['forbidden_stream_ids'])
    output.mkdir(parents=True)
    cpu, wall = time.process_time(), time.perf_counter()
    checks = []
    for case in EXPECTED['cases']:
        base = load_frozen(tasks[case]['source_policy'])[0]
        approximation, setup = build_approximation(base, **EXPECTED['approximation'])
        row = pair(base, approximation, 1, case, 128, 0, forbidden)
        original = run/'rows'/f'case_{case:02d}_n_0128_rep_0000.json'
        if financial(row) != financial(read(original)):
            raise ArithmeticError('Replayed financial row differs: '+str(case))
        saved_setup = read(run/f'setup_{case:02d}_128.json')
        if financial(setup) != financial(saved_setup):
            raise ArithmeticError('Replayed approximation differs')
        save(output/f'case_{case:02d}.json', dict(row=row, setup=setup))
        checks.append(dict(case=case, evaluation_paths=128, repetition=0,
            original_row=str(original.relative_to(ROOT)), original_sha256=sha(original),
            financial_record_exactly_reproduced=True, approximation_exactly_reproduced=True))
    evidence[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    merge({}, evidence)
    save(output/'checks.json', dict(complete=True, recorded_at_utc=utc(), checks=checks,
        replayed_rows=3, full_numerical_rerun=False, new_independent_observations=0,
        timing_fields_compared=False, started_after_primary_manifest=True,
        process_cpu_seconds=time.process_time()-cpu, elapsed_seconds=time.perf_counter()-wall,
        publication_clearance=False))
    save(output/'manifest.json', dict(complete=True, evidence_sha256=evidence,
        artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir())}, publication_clearance=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=BASE/'registration_v1')
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.registration, args.run, args.output)


if __name__ == '__main__':
    main()
