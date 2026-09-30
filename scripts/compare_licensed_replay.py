"""Compare complete historical study records with fresh private executions.

This is a numerical comparator, not a financial-data acquisition check. Every
manifested artifact is checked, classified and included. Observed timings,
derived timing quantities and execution metadata have explicit exclusions.
The complete exclusion ledger and every failed comparison remain private.
Both run directories and the new report directory must be beneath this
checkout's data/private directory; symlinked ancestors are not accepted.
Use the licensed-input reproduction guide to assemble an authorised archive.
A subscription alone does not provide the original historical inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


STUDIES = {
    'value_martingale': 'value_martingale/2026-09-20/run_v1',
    'terminal_repair': 'terminal_repair/2026-09-20/run_v1',
    'terminal_training': 'terminal_training/2026-09-20/run_v1',
    'training_evaluation': 'training_evaluation/2026-09-26/run_v1',
    'allocation_pilot': 'training_allocation/2026-09-27/pilot_v1',
    'training_allocation': 'training_allocation/2026-09-27/main_run_v1',
    'prospective_allocation': 'prospective_allocation/2026-09-28/run_v1',
    'cpu_prediction': 'cpu_prediction/2026-09-28/run_v1',
}
ALLOCATIONS = {'allocation_pilot', 'training_allocation'}
TIMING_KEYS = {
    'elapsed_seconds', 'cpu_seconds', 'simulation_seconds', 'fitting_seconds',
    'simulation_cpu_seconds', 'fitting_cpu_seconds', 'build_seconds',
    'martingale_seconds', 'policy_evaluation_seconds', 'evaluation_seconds',
    'policy_and_estimation_seconds', 'total_elapsed_seconds', 'total_cpu_seconds',
    'additional_training_seconds', 'training_seconds', 'training_cpu_seconds',
    'evaluation_cpu_seconds', 'training_elapsed_seconds', 'evaluation_elapsed_seconds',
    'standalone_cpu_seconds', 'standalone_elapsed_seconds', 'outer_cpu_seconds',
    'outer_elapsed_seconds', 'process_cpu_seconds', 'process_elapsed_seconds',
}
ALLOCATION_DERIVED = {
    'actual_to_budget_ratio', 'material_budget_mismatch', 'actual_cpu_ratio',
    'estimated_variance_cpu_ratio',
}
PROSPECTIVE_DERIVED = {
    'exact_numerical_cpu', 'exact_numerical_elapsed', 'actual_to_budget',
    'over_budget', 'actual_to_forecast', 'within_forecast_tolerance',
    'cpu_quartiles', 'numerical_cpu', 'warmup_numerical_cpu', 'total_numerical_cpu',
    'over_budget_arms', 'actual_cpu_ratio', 'estimated_variance_cpu_ratio',
}
CPU_DERIVED = {
    'actual', 'comparisons', 'calibration_charge', 'charged_numerical_cost',
    'charged_cpu_affordable', 'first_quarter_pair_cpu', 'last_quarter_pair_cpu',
    'allowance_predictions', 'first_half_pair_mean', 'last_half_pair_mean',
    'numerical_validation_cost', 'calibration_cost', 'charged_arms_affordable',
    'numerical_totals', 'warmup_charge', 'forecast_sha256',
    'prediction_sha256_before_validation', 'cpu_since_launch',
    'elapsed_since_launch', 'numerical_charge',
}
DATE_KEYS = {'started_at', 'started_at_utc', 'recorded_at_utc', 'ended_at_utc',
             'completed_at', 'completed_at_utc'}
LAUNCH_ENVIRONMENT = {
    'python', 'numpy', 'scipy', 'platform', 'processor', 'environment', 'numpy_build',
    'thread_environment', 'threads', 'python_workers', 'single_worker',
    'effective_native_thread_count', 'workers', 'process_id', 'telemetry',
}


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('Duplicate JSON key: ' + key)
        out[key] = value
    return out


def read(path):
    return json.loads(path.read_text(), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError('Nonfinite JSON constant: ' + value)))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def private(path):
    path = path.absolute()
    private_root = ROOT / 'data/private'
    if '..' in path.parts or path.resolve() != path:
        raise ValueError('Noncanonical or symlink paths are not permitted')
    if not path.is_relative_to(private_root):
        raise ValueError('All inputs and output must be beneath the repository data/private directory')
    return path


def member(folder, name):
    part = PurePosixPath(name)
    if (part.is_absolute() or '..' in part.parts or str(part) != name
            or '\\' in name or not name.endswith('.json')):
        raise ValueError('Unsafe or non-JSON artifact path: ' + name)
    path = folder / name
    current = folder
    for segment in part.parts:
        current = current / segment
        if current.is_symlink():
            raise ValueError('Symlink artifact: ' + name)
    return path


def checked_manifest(folder):
    manifest = read(member(folder, 'manifest.json'))
    if manifest.get('complete', True) is not True:
        raise ValueError('Incomplete study manifest')
    if (folder / 'incomplete.json').exists() or (folder / 'failure.json').exists():
        raise ValueError('A failure record is present')
    artifacts = manifest.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Manifest must contain a nonempty artifact mapping')
    for name, expected in artifacts.items():
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid artifact digest: ' + name)
        if digest(member(folder, name)) != expected:
            raise ValueError('Artifact hash mismatch: ' + name)
    for path in folder.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink inside study directory')
        if not path.is_file():
            continue
        name = str(path.relative_to(folder))
        if name in artifacts or name == 'manifest.json':
            continue
        # The first three original runners leave this checkpoint outside their
        # final manifest. An archive may omit it. Accept only an EXACT duplicate
        # of the manifested final rows, never an additional observation.
        if name == 'partial_rows.json' and 'rows.json' in artifacts:
            if digest(path) != digest(member(folder, 'rows.json')):
                raise ValueError('Unmanifested checkpoint differs from final rows')
            continue
        raise ValueError('Unmanifested artifact: ' + name)
    return manifest


def exclusion(study, file, path, key, value):
    """Return a specific reason, with immutable financial choices preserved."""
    if file in {'configuration.json', 'tasks.json'}:
        return None
    if file == 'manifest.json':
        if key == 'artifact_sha256':
            return 'Per-run artifact hashes verified independently before comparison'
        if key in DATE_KEYS or key in TIMING_KEYS:
            return 'Observed execution timestamp or duration'
        return None
    if file == 'launch.json' and key in LAUNCH_ENVIRONMENT:
        return 'Execution environment recorded separately without requiring identity'
    if key in DATE_KEYS:
        return 'Observed execution timestamp'
    if key in {'telemetry', 'telemetry_before', 'telemetry_after'}:
        return 'Observed runtime telemetry'
    # Main allocation prices and counts are fixed by the ORIGINAL pilot and must
    # compare, including the old CPU budget used to choose those counts.
    if study == 'training_allocation' and 'allocation' in path:
        return None
    if study == 'allocation_pilot' and key == 'allocation':
        return 'Fresh pilot allocation depends on new CPU observations; original main counts remain frozen'
    if key in TIMING_KEYS:
        return 'Observed workload duration'
    if study == 'value_martingale' and key == 'shared_costs':
        return 'Observed shared build and martingale durations'
    if study in ALLOCATIONS and key in ALLOCATION_DERIVED:
        return 'Quantity computed from newly observed workload durations'
    if study == 'prospective_allocation' and key in PROSPECTIVE_DERIVED:
        return 'Quantity computed from newly observed workload durations'
    if study == 'cpu_prediction':
        if key == 'complete_financial_record_sha256':
            return 'Original digest includes unsaved financial fields AND observed timings; exact financial-value replay cannot be established from this digest'
        if key in CPU_DERIVED:
            return 'CPU forecast, accounting result or digest depending on new timing observations'
        if key == 'predictions' and isinstance(value, dict) and set(value) <= {'cpu', 'elapsed'}:
            return 'Clock-specific cost forecast'
    return None


class Comparison:
    def __init__(self, study, out):
        self.study, self.out = study, out
        self.counts = Counter()
        self.reasons = Counter()
        self.files = []
        self.exceptions = gzip.open(out / 'differences.jsonl.gz', 'wt', compresslevel=1)
        self.exclusions = gzip.open(out / 'exclusions.jsonl.gz', 'wt', compresslevel=1)
        self.roundoff = gzip.open(out / 'roundoff_candidates.jsonl.gz', 'wt', compresslevel=1)

    def difference(self, file, path, expected, actual, reason):
        self.counts['failed_fields'] += 1
        self.exceptions.write(json.dumps(dict(file=file, path=path, expected=expected,
            actual=actual, reason=reason), allow_nan=False) + '\n')

    def project(self, value, file, side, path=()):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                reason = exclusion(self.study, file, path, key, item)
                if reason:
                    self.counts[side + '_excluded_fields'] += 1
                    self.reasons[reason] += 1
                    self.exclusions.write(json.dumps(dict(side=side, file=file,
                        path=list(path + (key,)), reason=reason,
                        value_sha256=json_digest(item)), allow_nan=False) + '\n')
                    out[key] = {'__declared_observational_exclusion__': reason}
                else:
                    out[key] = self.project(item, file, side, path + (key,))
            return out
        if isinstance(value, list):
            return [self.project(item, file, side, path + (index,))
                    for index, item in enumerate(value)]
        return value

    def compare(self, expected, actual, file, path=()):
        self.counts['compared_nodes'] += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.difference(file, path, type(expected).__name__, type(actual).__name__, 'Type differs')
                return
            if set(expected) != set(actual):
                self.difference(file, path, sorted(expected), sorted(actual), 'Object keys differ')
            for key in expected.keys() & actual.keys():
                self.compare(expected[key], actual[key], file, path + (key,))
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                self.difference(file, path, type(expected).__name__, type(actual).__name__, 'Type differs')
                return
            if len(expected) != len(actual):
                self.difference(file, path, len(expected), len(actual), 'List lengths differ')
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.compare(left, right, file, path + (index,))
        else:
            self.counts['compared_leaves'] += 1
            if type(expected) is type(actual) and expected == actual:
                self.counts['exact_leaves'] += 1
                return
            # A near-equal value remains a FAILED exact replay comparison. The
            # prior allocation audit's summary tolerances are diagnostic only.
            if (self.study in ALLOCATIONS and isinstance(expected, float)
                    and type(actual) in (float, int)
                    and math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14)):
                self.counts['roundoff_candidate_leaves'] += 1
                self.roundoff.write(json.dumps(dict(file=file, path=path,
                    expected=expected, actual=actual, absolute_difference=abs(expected-actual)),
                    allow_nan=False) + '\n')
            self.difference(file, path, expected, actual, 'Value differs')

    def file(self, name, left, right):
        before = self.counts['failed_fields']
        expected, actual = read(left), read(right)
        self.compare(self.project(expected, name, 'original'),
                     self.project(actual, name, 'fresh'), name)
        self.files.append(dict(file=name, original_sha256=digest(left), fresh_sha256=digest(right),
                               equal_after_declared_exclusions=self.counts['failed_fields'] == before))

    def close(self):
        self.exceptions.close(); self.exclusions.close(); self.roundoff.close()


def coverage(study, folder, artifacts):
    """Count all primary records, plus retained warm-up and calibration work."""
    if study in {'value_martingale', 'terminal_repair', 'terminal_training'}:
        rows = read(folder / 'rows.json')
        result = dict(model_rows=len(rows), cases=[row['case'] for row in rows])
        if study == 'value_martingale':
            result['policy_records'] = sum(len(row['policies']) for row in rows)
        else:
            result['scenario_records'] = sum(len(row['scenarios']) for row in rows)
            result['policy_scenario_records'] = sum(len(s['policies']) for row in rows for s in row['scenarios'])
        return result
    if study == 'training_evaluation':
        blocks = [read(folder / name) for name in artifacts if name.startswith('blocks/')]
        return dict(blocks=len(blocks), workloads=sum(len(b['workloads']) for b in blocks),
                    cases=sorted({b['case'] for b in blocks}), warmup_records=1)
    if study == 'allocation_pilot':
        blocks = [read(folder / name) for name in artifacts if name.startswith('blocks/')]
        return dict(model_blocks=len(blocks), statistical_pairs=sum(len(b['statistical']) for b in blocks),
                    timing_evaluations=sum(len(b['timing']) for b in blocks),
                    setups=sum(len(b['setup']) for b in blocks), warmup_records=1)
    if study == 'training_allocation':
        return dict(main_pairs=sum(name.startswith('blocks/') for name in artifacts),
                    model_summaries=sum(name.startswith('model_') for name in artifacts), warmup_records=1)
    if study == 'prospective_allocation':
        return dict(primary_pairs=sum(name.startswith('rows/') for name in artifacts),
                    arms=sum(name.startswith('arm_') for name in artifacts),
                    setup_records=sum(name.startswith('setup_') for name in artifacts),
                    warmup_pairs=len(read(folder / 'warmup.json')['rows']))
    counts = Counter()
    for name in artifacts:
        if re.search(r'/row_\d+\.json$', name):
            row = read(folder / name)
            counts[row['phase'] + '_pairs'] += 1
        if name.endswith('/setup.json'):
            counts['setup_records'] += 1
    counts['financial_estimate_values_retained'] = 0
    return dict(counts)


def run(study, original, fresh, output):
    original, fresh, output = map(private, (original, fresh, output))
    if output.exists():
        raise FileExistsError('Use a fresh comparison destination')
    if output == original or output == fresh or original in output.parents or fresh in output.parents:
        raise ValueError('Comparison output must be outside both study directories')
    output.mkdir(parents=True)
    record = dict(study=study, original=str(original), fresh=str(fresh),
                  recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  complete=False, numerical_agreement=False, publication_clearance=False,
                  financial_acquisition_replayed=False, timing_values_expected_to_match=False,
                  comparator_sha256=digest(Path(__file__)))
    cmp = Comparison(study, output)
    try:
        original_manifest_hash = digest(member(original, 'manifest.json'))
        fresh_manifest_hash = digest(member(fresh, 'manifest.json'))
        lm, rm = checked_manifest(original), checked_manifest(fresh)
        left, right = set(lm['artifact_sha256']), set(rm['artifact_sha256'])
        record['original_artifact_count'], record['fresh_artifact_count'] = len(left), len(right)
        record['unmanifested_duplicate_checkpoint'] = {
            'original': (original / 'partial_rows.json').exists(),
            'fresh': (fresh / 'partial_rows.json').exists(),
            'acceptance_rule': 'If present, partial_rows.json must be byte-identical to manifested rows.json',
        }
        record['artifact_coverage_equal'] = left == right
        if left != right:
            cmp.difference('manifest.json', ('artifact_sha256',), sorted(left), sorted(right),
                           'Manifest artifact coverage differs')
        cmp.file('manifest.json', original / 'manifest.json', fresh / 'manifest.json')
        # Every declared artifact is included; no wildcard only sampling route.
        for name in sorted(left & right):
            cmp.file(name, member(original, name), member(fresh, name))
        record['original_coverage'] = coverage(study, original, left)
        record['fresh_coverage'] = coverage(study, fresh, right)
        cmp.compare(record['original_coverage'], record['fresh_coverage'], 'coverage')
        save(output / 'execution_metadata.json', dict(original=read(original / 'launch.json'),
                                                     fresh=read(fresh / 'launch.json')))
        # Bind the end of comparison to the same complete artifact inventories.
        # A changed file or a simultaneously replaced manifest invalidates it.
        checked_manifest(original)
        checked_manifest(fresh)
        if (digest(member(original, 'manifest.json')) != original_manifest_hash
                or digest(member(fresh, 'manifest.json')) != fresh_manifest_hash):
            raise ValueError('A manifest changed during comparison')
        record['original_manifest_sha256'] = original_manifest_hash
        record['fresh_manifest_sha256'] = fresh_manifest_hash
        record['artifact_integrity_checked_before_and_after'] = True
        record['complete'] = True
        record['numerical_agreement'] = cmp.counts['failed_fields'] == 0
        record['exact_retained_numerical_equality'] = record['numerical_agreement']
        record['same_run_directories'] = original.resolve() == fresh.resolve()
        record['distinct_run_manifests'] = original_manifest_hash != fresh_manifest_hash
        record['execution_provenance_independently_required'] = True
        record['comparison_rule'] = 'Exact JSON numerical and structural equality; near-equality is diagnostic only and does not pass'
        record['scope_limitations'] = [
            'Execution proves equality only for retained numerical outcomes under supplied model inputs',
            'Historical processor observations and their derived ratios do not recur by specification',
            'Manifest byte verification does not establish the financial validity of supplied inputs',
        ]
        if study == 'cpu_prediction':
            record['scope_limitations'].append(
                'CPU record retains policy, path and approximation fingerprints and stream identities, '
                'but omits financial estimates; their original digest includes timing values and '
                'cannot establish replay equality of those omitted estimates')
        if study == 'allocation_pilot':
            record['scope_limitations'].append(
                'New pilot timings may select new counts; those derived allocations are logged as '
                'excluded and do not replace the original main registration')
    except Exception as exc:
        record['error'] = dict(type=type(exc).__name__, message=str(exc))
    finally:
        cmp.close()
        record['counts'] = dict(cmp.counts)
        record['exclusion_reasons'] = dict(cmp.reasons)
        save(output / 'files.json', cmp.files)
        save(output / 'comparison.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=sorted(STUDIES), required=True)
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--fresh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.archive_root is None) == (args.original is None):
        parser.error('Specify exactly one of --archive-root and --original')
    original = args.original if args.original else args.archive_root / 'data/private' / STUDIES[args.study]
    result = run(args.study, original, args.fresh, args.output)
    print(json.dumps({key: result[key] for key in ('study', 'complete', 'numerical_agreement', 'counts')},
                     sort_keys=True))
    if 'error' in result:
        print(json.dumps(result['error']), file=sys.stderr)
    return 0 if result['complete'] and result['numerical_agreement'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
