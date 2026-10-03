"""Check an authorised original archive before historical reproduction.

This standard-library command verifies evidence bytes without executing financial
workloads, copying market observations or acquiring data. A licence alone does not
supply the original archive. The report remains under the local data/private tree.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT/'replication/licensed_requirements.json'
_spec = importlib.util.spec_from_file_location('licensed_source_paths',
    ROOT/'replication/verify_sources.py')
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
relative_path, checked_file, read_json_bytes = (
    _helpers.relative_path, _helpers.checked_file, _helpers.read_json_bytes)
ROOT_MAPS = frozenset(('input_sha256', 'source_sha256', 'evidence_sha256',
    'source_hashes', 'reference_sha256', 'registered_source_sha256',
    'additional_audit_source_sha256', 'audit_source_sha256',
    'protected_figure_artifact_sha256'))
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_FILES = 100_000


def digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Invalid SHA-256 digest in evidence metadata')
    return value


def plain_directory(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink directory is not an archive or output location')
    return path.resolve(strict=True)


def metadata(path):
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError('Evidence metadata exceeds the size limit')
    raw = path.read_bytes()
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('Evidence metadata exceeds the size limit')
    value = read_json_bytes(raw)
    if not isinstance(value, dict):
        raise ValueError('Evidence metadata must be a JSON object')
    return raw, value


def requirements(path=REQUIREMENTS):
    raw, document = metadata(Path(path))
    if (set(document) != {'schema_version', 'title', 'scope', 'roots',
                          'additional_archive_files', 'limitations'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
        raise ValueError('Unexpected requirements schema')
    if any(not isinstance(document[k], str) or not document[k].strip() for k in ('title', 'scope')):
        raise ValueError('Requirements descriptions must be nonempty strings')
    if (not isinstance(document['limitations'], list) or not document['limitations']
            or any(not isinstance(v, str) or not v.strip() for v in document['limitations'])):
        raise ValueError('Requirements limitations must be nonempty strings')
    roots, extra = document['roots'], document['additional_archive_files']
    if not isinstance(roots, list) or not roots or not isinstance(extra, list):
        raise ValueError('Invalid requirements entry lists')
    paths, identifiers, tables = set(), set(), set()
    for entry in roots + extra:
        is_root = entry in roots
        fields = {'id', 'tables', 'path', 'sha256'} if is_root else {'path', 'sha256', 'role'}
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError('Invalid requirements entry')
        name = str(relative_path(entry['path']))
        if name in paths:
            raise ValueError('Duplicate requirements path')
        paths.add(name)
        digest(entry['sha256'])
        if is_root:
            if (not name.startswith('data/private/') or not name.endswith('/manifest.json')
                    or not isinstance(entry['id'], str)
                    or re.fullmatch('[a-z][a-z0-9_]*', entry['id']) is None
                    or entry['id'] in identifiers):
                raise ValueError('Invalid or duplicate experiment root')
            identifiers.add(entry['id'])
            if (not isinstance(entry['tables'], list)
                    or any(type(t) is not int or not 1 <= t <= 9 for t in entry['tables'])
                    or len(set(entry['tables'])) != len(entry['tables'])
                    or tables.intersection(entry['tables'])):
                raise ValueError('Invalid or overlapping table coverage')
            tables.update(entry['tables'])
        elif not isinstance(entry['role'], str) or re.fullmatch('[a-z][a-z0-9_]*', entry['role']) is None:
            raise ValueError('Invalid additional archive role')
    return raw, document


def fingerprint(path):
    before = path.stat()
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    after = path.stat()
    if signature(before) != signature(after):
        raise ValueError('An evidence file changed while it was checked')
    return value.hexdigest(), signature(after)


def signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def check_archive(archive_root=ROOT, requirements_path=REQUIREMENTS):
    """Verify the explicit, root-bound dependency closure; return metadata only."""
    root = plain_directory(archive_root)
    raw, contract = requirements(requirements_path)
    pending, expected, verified, states = deque(), {}, {}, {}
    issues, manifests, opaque, scalar_fields = [], [], [], []

    def add(name, sha):
        name = str(relative_path(name))
        digest(sha)
        if name in expected and expected[name] != sha:
            raise ValueError('Conflicting evidence hashes for ' + name)
        if name not in expected:
            if len(expected) >= MAX_FILES:
                raise ValueError('Evidence closure exceeds the file limit')
            expected[name] = sha
            pending.append(name)

    for entry in contract['roots'] + contract['additional_archive_files']:
        add(entry['path'], entry['sha256'])
    while pending:
        name = pending.popleft()
        try:
            path = checked_file(root, name)
        except ValueError:
            # A missing file is an incomplete archive; an unsafe alias is invalid.
            candidate = root
            for part in relative_path(name).parts:
                candidate /= part
                if candidate.is_symlink():
                    raise ValueError('Symlink component in evidence path: ' + name)
            issues.append(dict(path=name, status='missing_or_non_regular'))
            continue
        sha, state = fingerprint(path)
        states[name] = state
        if sha != expected[name]:
            issues.append(dict(path=name, status='digest_mismatch'))
            continue
        verified[name] = sha
        if not path.name.endswith('manifest.json'):
            continue
        manifest_raw, manifest = metadata(path)
        if hashlib.sha256(manifest_raw).hexdigest() != sha:
            raise ValueError('A manifest changed during dependency expansion')
        manifests.append(name)
        supported = []
        for key, mapping in manifest.items():
            if key not in ROOT_MAPS and key != 'artifact_sha256':
                if isinstance(mapping, dict) and (key.endswith('_sha256') or key.endswith('_hashes')):
                    raise ValueError('Unsupported path-to-digest map in ' + name)
                continue
            # Acquisition records sometimes use a scalar digest without a path.
            if key in ROOT_MAPS and isinstance(mapping, str):
                digest(mapping)
                scalar_fields.append(dict(manifest=name, field=key))
                continue
            if not isinstance(mapping, dict):
                raise ValueError('Evidence hash map is not an object: ' + name + ' ' + key)
            supported.append(key)
            for child, wanted in mapping.items():
                relative_path(child)
                destination = (str(PurePosixPath(name).parent / child)
                               if key == 'artifact_sha256' else child)
                add(destination, wanted)
        if not supported:
            opaque.append(name)
    for name, state in states.items():
        if signature(checked_file(root, name).stat()) != state:
            raise ValueError('An evidence file changed during the readiness check')
    if Path(requirements_path).read_bytes() != raw:
        raise ValueError('Requirements changed during the readiness check')
    return dict(schema_version=1, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        scope=contract['scope'], evidence_ready=not issues, financial_workloads_replayed=False,
        original_cpu_measurements_recreated=False, licence_or_transfer_rights_verified=False,
        computational_environment_verified=False, market_observations_exported=False,
        requirements_sha256=hashlib.sha256(raw).hexdigest(),
        roots=contract['roots'], dependency_files_declared=len(expected),
        dependency_files_verified=len(verified), manifests_expanded=len(manifests),
        outside_private_files=sum(not n.startswith('data/private/') for n in expected),
        additional_archive_files=contract['additional_archive_files'],
        evidence_files=expected, issues=issues, opaque_manifest_leaves=opaque,
        scalar_digest_fields_not_expanded=scalar_fields,
        limitations=contract['limitations'])


def output_directory(value):
    root = plain_directory(ROOT)
    path = Path(value)
    path = path if path.is_absolute() else root/path
    if '..' in path.parts or not path.is_relative_to(root/'data/private'):
        raise ValueError('Output must be a fresh directory under data/private')
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Symlink output locations are not permitted')
    if path.exists():
        raise FileExistsError('Use a fresh output directory')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=ROOT,
                        help='Checkout containing the authorised historical archive at its original relative paths')
    parser.add_argument('--output', type=Path, default=Path('data/private/reproduction/licensed_readiness'),
                        help='Fresh local directory under data/private; no observations are copied')
    args = parser.parse_args()
    try:
        output = output_directory(args.output)
        report = check_archive(args.archive_root)
        output.mkdir(parents=True)
        (output/'readiness.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit('Readiness check rejected invalid or unreadable metadata (' + type(error).__name__ + ')') from error
    print(json.dumps({k:report[k] for k in ('evidence_ready', 'dependency_files_declared',
        'dependency_files_verified', 'manifests_expanded', 'financial_workloads_replayed')}))
    raise SystemExit(0 if report['evidence_ready'] else 2)


if __name__ == '__main__':
    main()
