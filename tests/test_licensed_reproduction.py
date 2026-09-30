"""Declared archive fixtures test provenance checks without market observations."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/check_licensed_reproduction.py'
_spec = importlib.util.spec_from_file_location('licensed_readiness_tests', SCRIPT)
readiness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readiness)


def write(root, name, content):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = content if isinstance(content, bytes) else json.dumps(content).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def archive(tmp_path, *, root_mapping=None, extra=()):
    root = tmp_path/'archive'
    root.mkdir()
    source = 'scripts/declared_model.py'
    source_hash = write(root, source, b'# Declared mathematical fixture.\n')
    leaf = 'data/private/fixture/child/manifest.json'
    observation = write(root, 'data/private/fixture/child/example.txt',
                        b'Declared content sentinel: never copy this into the report.\n')
    leaf_hash = write(root, leaf, dict(source_sha256={source:source_hash},
        artifact_sha256={'example.txt':observation}))
    name = 'data/private/fixture/run/manifest.json'
    manifest = dict(evidence_sha256={leaf:leaf_hash}) if root_mapping is None else root_mapping
    sha = write(root, name, manifest)
    contract = tmp_path/'requirements.json'
    contract.write_text(json.dumps(dict(schema_version=1, title='Declared test fixture',
        scope='Explicit fixture dependencies only',
        roots=[dict(id='fixture', tables=[1], path=name, sha256=sha)],
        additional_archive_files=list(extra), limitations=['No financial data or simulation.'])))
    return root, contract, name, leaf, source


def test_transitive_relative_artifacts_and_outside_private_source_are_verified(tmp_path):
    root, contract, name, leaf, source = archive(tmp_path)
    report = readiness.check_archive(root, contract)
    assert report['evidence_ready'] is True
    assert report['dependency_files_verified'] == 4
    assert report['manifests_expanded'] == 2
    assert report['outside_private_files'] == 1
    assert report['financial_workloads_replayed'] is False
    assert report['computational_environment_verified'] is False
    assert set(report['evidence_files']) == {name, leaf, source, 'data/private/fixture/child/example.txt'}
    assert 'Declared content sentinel' not in json.dumps(report)


@pytest.mark.parametrize('action,status', [('remove','missing_or_non_regular'), ('change','digest_mismatch')])
def test_absent_or_mutated_required_source_prevents_readiness(tmp_path, action, status):
    root, contract, _, _, source = archive(tmp_path)
    if action == 'remove':
        (root/source).unlink()
    else:
        (root/source).write_text('# Different declared fixture.\n')
    report = readiness.check_archive(root, contract)
    assert report['evidence_ready'] is False
    assert report['issues'] == [dict(path=source, status=status)]


def test_changed_manifest_is_not_used_to_expand_untrusted_dependencies(tmp_path):
    root, contract, _, leaf, _ = archive(tmp_path)
    (root/leaf).write_text(json.dumps({'source_sha256':{'../../outside':'0'*64}}))
    report = readiness.check_archive(root, contract)
    assert report['issues'] == [dict(path=leaf, status='digest_mismatch')]
    assert report['manifests_expanded'] == 1
    assert '../../outside' not in report['evidence_files']


@pytest.mark.parametrize('unsafe', ['../outside', '/absolute', 'x/../y', 'x//y', 'x\\y'])
@pytest.mark.parametrize('mapping', ['input_sha256', 'artifact_sha256'])
def test_bound_manifest_cannot_escape_or_alias_archive_paths(tmp_path, unsafe, mapping):
    root, contract, *_ = archive(tmp_path, root_mapping={mapping:{unsafe:'0'*64}})
    with pytest.raises(ValueError, match='canonical relative'):
        readiness.check_archive(root, contract)


def test_conflicting_transitive_hashes_are_rejected(tmp_path):
    root, contract, name, leaf, source = archive(tmp_path)
    data = json.loads((root/name).read_text())
    data['source_sha256'] = {source:'0'*64}
    new_hash = write(root, name, data)
    specification = json.loads(contract.read_text())
    specification['roots'][0]['sha256'] = new_hash
    contract.write_text(json.dumps(specification))
    with pytest.raises(ValueError, match='Conflicting evidence hashes'):
        readiness.check_archive(root, contract)


@pytest.mark.parametrize('location', ['requirements', 'nested_manifest'])
def test_duplicate_json_keys_cannot_change_the_evidence_contract(tmp_path, location):
    root, contract, name, *_ = archive(tmp_path)
    if location == 'requirements':
        contract.write_text(contract.read_text().replace('"schema_version": 1',
                            '"schema_version": 1, "schema_version": 1'))
    else:
        new_hash = write(root, name, b'{"input_sha256": {}, "input_sha256": {}}')
        data = json.loads(contract.read_text())
        data['roots'][0]['sha256'] = new_hash
        contract.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        readiness.check_archive(root, contract)


@pytest.mark.parametrize('target', ['file', 'directory', 'archive_root'])
def test_symlink_inputs_are_rejected_even_when_bytes_match(tmp_path, target):
    root, contract, _, _, source = archive(tmp_path)
    if target == 'file':
        original = root/source
        relocated = root/'retained_fixture.py'
    elif target == 'directory':
        original = root/'scripts'
        relocated = root/'retained_scripts'
    else:
        original = root
        relocated = tmp_path/'retained_archive'
    original.rename(relocated)
    original.symlink_to(relocated, target_is_directory=target != 'file')
    with pytest.raises(ValueError, match='Symlink'):
        readiness.check_archive(root, contract)


def test_source_changed_later_in_pass_is_rejected(tmp_path, monkeypatch):
    root, contract, _, _, source = archive(tmp_path)
    original = readiness.fingerprint

    def mutate(path):
        result = original(path)
        if path.name == 'example.txt':
            (root/source).write_text('# Changed after its earlier verification.\n')
        return result

    monkeypatch.setattr(readiness, 'fingerprint', mutate)
    with pytest.raises(ValueError, match='changed during the readiness'):
        readiness.check_archive(root, contract)


def test_unsupported_maps_fail_and_pathless_scalar_digests_are_disclosed(tmp_path):
    root, contract, name, *_ = archive(tmp_path,
        root_mapping={'unknown_sha256':{'scripts/declared_model.py':'0'*64}})
    with pytest.raises(ValueError, match='Unsupported path-to-digest map'):
        readiness.check_archive(root, contract)
    sha = write(root, name, {'source_sha256':'0'*64})
    data = json.loads(contract.read_text())
    data['roots'][0]['sha256'] = sha
    contract.write_text(json.dumps(data))
    report = readiness.check_archive(root, contract)
    assert report['opaque_manifest_leaves'] == [name]
    assert report['scalar_digest_fields_not_expanded'] == [dict(manifest=name, field='source_sha256')]


def test_output_rejects_existing_public_or_symlink_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, 'ROOT', tmp_path)
    output = tmp_path/'data/private/existing'
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        readiness.output_directory(output)
    with pytest.raises(ValueError, match='under data/private'):
        readiness.output_directory(tmp_path/'published')
    linked = tmp_path/'data/private/alias'
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        readiness.output_directory(linked/'new')
    assert readiness.output_directory('data/private/fresh') == tmp_path/'data/private/fresh'


def test_help_runs_without_scientific_packages():
    result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--archive-root' in result.stdout and '--output' in result.stdout
    assert 'without executing financial' in result.stdout
