"""Export the thirteen retained interpolants for independent certification.

Only deterministic approximation reconstruction is performed. The export
contains licensed financial inputs and must remain in the private data tree.
It does not certify the global residual bound or any simulation estimate.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform

import numpy as np
import scipy

from audit_training_allocation_main import verify as verify_main
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_training_allocation import THREADS
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import build_approximation


BASE = ROOT / 'data/private/cpu_budget_transfer/2026-09-27'
MAIN = ROOT / 'data/private/training_allocation/2026-09-27/main_run_v1'
RESIDUAL = BASE / 'residual_inference_v1'
COMPLETION = BASE / 'residual_completion_v1.json'
EXPECTED_CASES = list(range(13))
SETTINGS = dict(core_steps=800, wing_steps=40, tolerance=.001)
APPROXIMATION_FIELDS = ('model', 'option', 'decisions', 'nodes', 'core_steps',
                        'wing_steps', 'compression_tolerance', 'initial_value')
CONVENTION = {
    'scalars': 'Every exported floating-point value denotes its exact binary64 real value.',
    'verification': 'The parallel hexadecimal map must equal float.hex() for every floating-point leaf.',
    'interpolants': 'Linear slopes and hinge coefficients are recomputed from the exact exported knots and values, without prior binary64 rounding.',
    'transition': 'sd = exact(volatility) * sqrt(exact(next_time) - exact(previous_time)).',
    'reward': 'The settlement strike is exact(strike) * exact(settlement_factor).',
    'cash_event': 'The state map is max(y - exact(discounted_cash), 0).',
    'premium_conversion': 'Divide discounted reward quantities by exact(premium_factor).',
    'distinction_from_retained_computation': 'The existing numerical routine rounds the time subtraction, square root, multiplication, slopes and normal evaluations. Those rounded intermediate expressions are not the exact-arithmetic model specified here.',
    'scope': 'The retained interpolants are fixed functions. A later certificate can enclose their residuals in this exact-arithmetic model without certifying their distance from the optimal value, the original calendar conversion, or the retained simulation computation.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_hashes(*mappings):
    """Reject conflicting provenance rather than silently replacing a hash."""
    result = {}
    for mapping in mappings:
        require(isinstance(mapping, dict), 'A checksum mapping is required')
        for name, digest in mapping.items():
            require(isinstance(name, str) and isinstance(digest, str)
                    and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                    'Malformed checksum entry')
            require(name not in result or result[name] == digest,
                    f'Conflicting evidence checksums: {name}')
            result[name] = digest
    return result


def relative_artifacts(folder, mapping):
    result = {}
    for name, digest in mapping.items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()), 'Artifact escapes its evidence directory')
        result[str(path.relative_to(ROOT))] = digest
    return result


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def hexadecimal_leaves(value, prefix=''):
    """Use JSON-pointer paths to retain the binary64 identity of every float."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            escaped = key.replace('~', '~0').replace('/', '~1')
            result.update(hexadecimal_leaves(item, prefix + '/' + escaped))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(hexadecimal_leaves(item, prefix + '/' + str(index)))
        return result
    if type(value) is float:
        require(math.isfinite(value), 'Non-finite binary64 input')
        return {prefix: value.hex()}
    return {}


def finite(value, name, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'Invalid {name}')


def case_coverage(rows, name):
    require(isinstance(rows, list) and rows
            and all(isinstance(row, dict) and type(row.get('case')) is int for row in rows)
            and [row['case'] for row in rows] == EXPECTED_CASES,
            f'{name} must contain each of the thirteen cases in order')


def validate_value(value):
    require(isinstance(value, dict) and set(value) == {'knots', 'values', 'tail_slope'},
            'Malformed linear value')
    x, y = value['knots'], value['values']
    require(isinstance(x, list) and isinstance(y, list) and len(x) == len(y) >= 2,
            'At least two matched knots and values are required')
    require(all(type(v) in (float, int) and math.isfinite(v) for v in x + y),
            'Finite knots and values are required')
    require(x[0] == 0 and all(b > a for a, b in zip(x, x[1:])),
            'Knots must increase strictly from zero')
    require(type(value['tail_slope']) in (float, int) and value['tail_slope'] == 0,
            'The retained certificate scope requires a constant tail')


def validate_document(document):
    """Validate identities and model coverage without the certification package."""
    require(document.get('schema_version') == 1 and document.get('exact_input_convention') == CONVENTION,
            'Unknown export schema or exact-input convention')
    rows = document['models']
    case_coverage(rows, 'Export')
    require(document['approximation_fields'] == list(APPROXIMATION_FIELDS),
            'Approximation fingerprint field coverage differs')
    for row in rows:
        spec = {key: row[key] for key in APPROXIMATION_FIELDS}
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(digest == row['approximation_fingerprint'], 'Exported approximation fingerprint differs')
        require(row['matching_arm_fingerprints'] == {'reference': digest, 'selected': digest}
                and row['residual_fine_fingerprint'] == digest, 'Retained approximation identity differs')
        require(row['exact_binary64_hex'] == hexadecimal_leaves(spec), 'Binary64 hexadecimal identity differs')
        require((row['core_steps'], row['wing_steps'], row['compression_tolerance']) ==
                (SETTINGS['core_steps'], SETTINGS['wing_steps'], SETTINGS['tolerance']),
                'Retained approximation settings differ')
        option, model = row['option'], row['model']
        require(option['kind'] == 'put', 'Only the retained put constructions are covered')
        finite(option['strike'], 'strike', positive=True)
        finite(option['maturity'], 'maturity', positive=True)
        for key in ('quoted_spot', 'initial_share_factor', 'premium_factor'):
            finite(model[key], key, positive=True)
        finite(model['volatility'], 'volatility')
        require(isinstance(model['dividends'], list), 'Dividend schedule must be a list')
        cash_times = []
        for event in model['dividends']:
            finite(event['ex_time'], 'cash-event time', positive=True)
            finite(event['discounted_cash'], 'cash cap')
            cash_times.append(event['ex_time'])
        require(all(b > a for a, b in zip(cash_times, cash_times[1:])),
                'Cash-event times must increase strictly')
        nodes, rights = row['nodes'], row['decisions']
        require(isinstance(nodes, list) and len(nodes) >= 2 and isinstance(rights, list) and rights,
                'Full node and exercise calendars are required')
        times = [node['time'] for node in nodes]
        for time in times:
            finite(time, 'node time')
        require(times[0] == 0 and times[-1] == option['maturity']
                and all(b > a for a, b in zip(times, times[1:])), 'Invalid node time coverage')
        events = {time for time in cash_times if time <= option['maturity']}
        require(events.issubset(set(times)), 'A cash event is absent from the node calendar')
        for node in nodes:
            require((node['before'] is not None) == (node['time'] in events),
                    'Cash-phase coverage differs')
            validate_value(node['after'])
            if node['before'] is not None:
                validate_value(node['before'])
        keys = []
        for right in rights:
            finite(right['time'], 'exercise time')
            finite(right['settlement_factor'], 'settlement factor', positive=True)
            require(right['time'] in times and right['phase'] in ('before', 'after'),
                    'Invalid exercise time or phase')
            require(right['phase'] != 'before' or right['time'] in events,
                    'A before-event right must coincide with a cash event')
            keys.append((right['time'], right['phase']))
        require(len(set(keys)) == len(keys) and keys == sorted(keys, key=lambda k: (k[0], k[1] == 'after'))
                and (option['maturity'], 'after') in keys, 'Invalid exercise calendar')


def run(main, residual, completion, output):
    main, residual, completion, output = map(private, (main, residual, completion, output))
    require(not output.exists(), 'Use a fresh output directory')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'Recorded thread settings are required')
    main_manifest, cfg, tasks, reconstructed = verify_main(main)
    require(cfg['approximation'] == SETTINGS, 'The frozen fine approximation settings are required')
    case_coverage(tasks, 'Main tasks')
    case_coverage(reconstructed['models'], 'Reconstructed main evidence')
    previous = read(residual / 'manifest.json')
    complete = read(completion)
    require(previous.get('complete') is True and previous.get('original_evidence_unchanged') is True,
            'Completed residual evidence is required')
    require(complete.get('independent_numerical_audit_completed') is True
            and complete.get('global_residual_bound_rounding_certified') is False,
            'The previous uncertified residual completion is required')
    explicit = {str(path.relative_to(ROOT)): sha(path) for path in
                (main / 'manifest.json', residual / 'manifest.json', completion)}
    inputs = merge_hashes(main_manifest['input_sha256'], previous['input_sha256'],
        relative_artifacts(main, main_manifest['artifact_sha256']),
        relative_artifacts(residual, previous['artifact_sha256']), complete['evidence_sha256'], explicit)
    sources = merge_hashes(main_manifest['source_sha256'], previous['source_sha256'],
        {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))},
        {'tests/test_export_residual_certificate.py': sha(ROOT / 'tests/test_export_residual_certificate.py')})
    closure = merge_hashes(inputs, sources)
    verify_mapping(closure)
    previous_summary = read(residual / 'summary.json')
    case_coverage(previous_summary['models'], 'Residual evidence')
    require(reconstructed['blocks'] == 5030, 'Main block coverage differs')
    output.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        for task, retained, prior in zip(tasks, reconstructed['models'], previous_summary['models']):
            approximation, _ = build_approximation(load_frozen(task['source_policy'])[0], **cfg['approximation'])
            spec = native(asdict(approximation))
            fingerprint = approximation.fingerprint
            arm_fingerprints = {arm: retained['arms'][arm]['setup']['approximation_fingerprint']
                                for arm in ('reference', 'selected')}
            require(all(value == fingerprint for value in arm_fingerprints.values())
                    and prior['fine']['approximation_fingerprint'] == fingerprint,
                    'Reconstruction differs from a retained arm or residual approximation')
            rows.append(dict(case=task['case'], **spec, approximation_fingerprint=fingerprint,
                matching_arm_fingerprints=arm_fingerprints,
                residual_fine_fingerprint=prior['fine']['approximation_fingerprint'],
                exact_binary64_hex=hexadecimal_leaves(spec)))
            print(f"Case {task['case']}: full interpolants reconstructed and fingerprints matched", flush=True)
        document = dict(schema_version=1, exact_input_convention=CONVENTION,
            approximation_fields=list(APPROXIMATION_FIELDS), models=rows)
        validate_document(document)
        save(output / 'models.json', document)
        validate_document(read(output / 'models.json'))
        counts = dict(models=len(rows), matched_main_arms=2*len(rows),
            matched_residual_approximations=len(rows), retained_training_pairs=reconstructed['blocks'],
            nodes=sum(len(row['nodes']) for row in rows),
            linear_functions=sum(1 + (node['before'] is not None) for row in rows for node in row['nodes']),
            exact_binary64_leaves=sum(len(row['exact_binary64_hex']) for row in rows),
            monte_carlo_paths_generated=0, policies_fitted=0)
        save(output / 'verification.json', dict(**counts,
            json_roundtrip_and_fingerprints_verified=True, global_bound_certified=False,
            original_evidence_unchanged=True, exact_input_convention=CONVENTION))
        verify_mapping(closure)
        distributions = sorted(({'name': dist.metadata['Name'], 'version': dist.version}
                                for dist in metadata.distributions()), key=lambda d: d['name'].lower())
        save(output / 'manifest.json', dict(complete=True,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            scope='Private full-interpolant inputs for an independent residual certificate; no certificate is produced by this export.',
            input_sha256=inputs, source_sha256=sources,
            artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
            exact_input_convention=CONVENTION, approximation_settings=SETTINGS,
            python=platform.python_version(), python_implementation=platform.python_implementation(),
            python_build=list(platform.python_build()), numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), machine=platform.machine(),
            thread_environment={key: os.environ[key] for key in THREADS},
            installed_distributions=distributions, counts=counts,
            original_evidence_unchanged=True, publication_clearance=False))
        verify_mapping(closure)
        verify_mapping(read(output / 'manifest.json')['artifact_sha256'], base=output)
        return counts
    except Exception as exc:
        save(output / 'incomplete.json', dict(complete=False, exception_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=MAIN)
    parser.add_argument('--residual', type=Path, default=RESIDUAL)
    parser.add_argument('--completion', type=Path, default=COMPLETION)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.main, args.residual, args.completion, args.output), sort_keys=True))
