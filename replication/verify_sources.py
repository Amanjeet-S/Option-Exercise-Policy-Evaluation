"""Check listed source bytes against the local replication manifest.

This is a consistency check, not proof of authorship, authenticity, complete
dependency coverage or correctness. The manifest is not an external trust
anchor. Only the standard library is needed for the standalone command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'replication/source_manifest.json'
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MANIFEST_FIELDS = {'schema_version', 'title', 'scope', 'dependency_scope',
                   'files', 'external_imports'}
ENTRY_FIELDS = {'path', 'category', 'sha256', 'bytes'}
CHECK_SCOPE = ('Consistency of listed source files and the supplied local manifest; '
               'not authenticity, completeness, mathematical correctness or market validation')


def read_json_bytes(raw):
    """Reject duplicate object keys and non-finite JSON constants at every level."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError('JSON input must not contain non-finite numbers: ' + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject)


def relative_path(value):
    """Require a canonical relative POSIX path, without traversal or aliases."""
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('Manifest paths must be canonical relative POSIX paths')
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in ('.', '..') for part in value.split('/'))):
        raise ValueError('Manifest paths must be canonical relative POSIX paths')
    return path


def checked_file(root, relative):
    path = root
    for part in relative_path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('Symlink component in manifest path: ' + relative)
    if not path.is_file():
        raise ValueError('Missing or non-regular manifest file: ' + relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Manifest path escapes the checkout: ' + relative)
    return path


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(document, manifest_relative):
    """Check the declared manifest schema before opening any listed source."""
    if (not isinstance(document, dict) or set(document) != MANIFEST_FIELDS
            or type(document['schema_version']) is not int
            or document['schema_version'] != 1):
        raise ValueError('Unexpected manifest fields or schema version')
    if any(not _text(document[name]) for name in ('title', 'scope', 'dependency_scope')):
        raise ValueError('Manifest descriptions must be nonempty strings')
    entries = document['files']
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest files must be a nonempty list')
    paths = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
            raise ValueError('Unexpected manifest entry fields')
        relative_path(entry['path'])
        if entry['path'] in paths:
            raise ValueError('Duplicate manifest path: ' + entry['path'])
        if entry['path'] == manifest_relative:
            raise ValueError('The manifest cannot list its own hash')
        paths.add(entry['path'])
        if (not isinstance(entry['category'], str)
                or re.fullmatch(r'[a-z][a-z0-9_]*', entry['category']) is None):
            raise ValueError('Invalid manifest category')
        if (not isinstance(entry['sha256'], str)
                or re.fullmatch(r'[0-9a-f]{64}', entry['sha256']) is None):
            raise ValueError('Invalid manifest SHA-256 digest')
        if type(entry['bytes']) is not int or entry['bytes'] < 0:
            raise ValueError('Manifest byte counts must be non-negative integers')
    imports = document['external_imports']
    if not isinstance(imports, dict):
        raise ValueError('Manifest external_imports must be an object')
    for name, importers in imports.items():
        if (re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', name, re.ASCII) is None
                or not isinstance(importers, list) or not importers):
            raise ValueError('Invalid external import entry')
        if any(not isinstance(p, str) or p not in paths for p in importers):
            raise ValueError('External import refers to an unlisted source')
        if len(set(importers)) != len(importers):
            raise ValueError('Duplicate external import source')
    return entries


def verify_sources(root=ROOT, manifest_relative=MANIFEST, *, required_paths=()):
    """Verify every declared byte count and digest; do not change the manifest."""
    root = Path(root).resolve(strict=True)
    manifest_path = checked_file(root, manifest_relative)
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('Source manifest exceeds the size limit')
    with manifest_path.open('rb') as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError('Source manifest exceeds the size limit')
    entries = validate_manifest(read_json_bytes(raw), manifest_relative)
    hashes = {}
    for entry in entries:
        path = checked_file(root, entry['path'])
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Source byte count differs: ' + entry['path'])
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                count += len(block)
                digest.update(block)
        if count != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Source SHA-256 or byte count differs: ' + entry['path'])
        hashes[entry['path']] = digest.hexdigest()
    if any(p not in hashes for p in required_paths):
        raise ValueError('A required execution source is absent from the manifest')
    if manifest_path.read_bytes() != raw:
        raise ValueError('Source manifest changed during verification')
    return dict(source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                verified_source_files=len(hashes), file_sha256=hashes,
                scope=CHECK_SCOPE)


def check_import_origins(verification, root=ROOT, *, modules=None):
    """Bind loaded project modules to listed files in the verified checkout.

    Numerical dependencies retain their separate version records. Runtime
    monkeypatches and in-memory code replacement are outside this check.
    """
    root = Path(root).resolve(strict=True)
    modules = sys.modules if modules is None else modules
    origins = {}
    for name, module in tuple(modules.items()):
        if name != 'qf_research' and not name.startswith('qf_research.'):
            continue
        if not isinstance(module, ModuleType):
            raise ValueError('Invalid loaded project module: ' + name)
        origin = getattr(module, '__file__', None)
        declared = getattr(getattr(module, '__spec__', None), 'origin', None)
        if not isinstance(origin, str) or not isinstance(declared, str):
            raise ValueError('Missing project module origin: ' + name)
        path, spec_path = Path(origin).absolute(), Path(declared).absolute()
        if path != spec_path or not path.is_relative_to(root / 'src'):
            raise ValueError('Project module is outside the verified checkout: ' + name)
        relative = path.relative_to(root).as_posix()
        if relative not in verification['file_sha256']:
            raise ValueError('Project module is not a listed source: ' + name)
        checked_file(root, relative)
        search = getattr(module, '__path__', None)
        if search is not None and tuple(Path(p).absolute() for p in search) != (path.parent,):
            raise ValueError('Project package search path differs from the verified checkout: ' + name)
        origins[name] = relative
    if 'qf_research' not in origins:
        raise ValueError('The project package has not been loaded')
    return dict(sorted(origins.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', default=MANIFEST,
                        help='Canonical path relative to the chosen checkout root')
    args = parser.parse_args()
    try:
        result = verify_sources(args.root, args.manifest)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({key: result[key] for key in
        ('source_manifest_sha256', 'verified_source_files', 'scope')}, indent=2))


if __name__ == '__main__':
    main()
