"""Source-consistency checks using declared temporary file fixtures."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'projects/01_optimal_stopping/replication/verify_sources.py'
spec = importlib.util.spec_from_file_location('source_consistency_checks', PATH)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def tree(tmp_path):
    files = {
        'src/qf_research/__init__.py': b'"""Declared test package."""\n',
        'src/qf_research/example.py': b'VALUE = 2\n',
    }
    entries = []
    for name, data in files.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(dict(path=name, category='test_fixture', bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    document = dict(schema_version=1, title='Declared source fixture',
        scope='No financial observations', dependency_scope='Two text files',
        files=entries, external_imports={})
    (tmp_path/'manifest.json').write_text(json.dumps(document))
    return tmp_path, document


def write(tree, document):
    (tree[0]/'manifest.json').write_text(json.dumps(document))


def verify(tree):
    return verifier.verify_sources(tree[0], 'manifest.json')


def module(name, path, *, package=False):
    result = ModuleType(name)
    result.__file__ = str(path)
    result.__spec__ = importlib.util.spec_from_file_location(name, path)
    if package:
        result.__path__ = [str(path.parent)]
    return result


def test_standalone_check_needs_no_scientific_packages(tree):
    result = subprocess.run([sys.executable, '-S', str(PATH), '--root', str(tree[0]),
        '--manifest', 'manifest.json'], capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record['verified_source_files'] == 2
    assert record['source_manifest_sha256'] == hashlib.sha256(
        (tree[0]/'manifest.json').read_bytes()).hexdigest()
    assert 'not authenticity' in record['scope']


@pytest.mark.parametrize('change', ['same_length', 'different_length', 'missing'])
def test_changed_or_missing_source_is_rejected(tree, change):
    path = tree[0]/'src/qf_research/example.py'
    if change == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'VALUE = 3\n' if change == 'same_length' else b'changed\n')
    with pytest.raises(ValueError, match='Source|Missing'):
        verify(tree)


@pytest.mark.parametrize('path', [
    '/tmp/outside.py', '../outside.py', 'src/../outside.py', './source.py',
    'src//source.py', 'src/source.py/', 'C:/outside.py', 'src\\source.py',
    'src/\x00source.py', '.', '',
])
def test_noncanonical_or_escaping_paths_rejected_before_file_access(tree, path):
    document = copy.deepcopy(tree[1])
    document['files'][0]['path'] = path
    write(tree, document)
    with pytest.raises(ValueError, match='canonical relative POSIX'):
        verify(tree)


@pytest.mark.parametrize('change', [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(extra='not in the schema'),
    lambda d: d.update(files=[]),
    lambda d: d['files'][0].update(extra=1),
    lambda d: d['files'][0].update(bytes=True),
    lambda d: d['files'][0].update(bytes=-1),
    lambda d: d['files'][0].update(bytes=3.0),
    lambda d: d['files'][0].update(sha256='0'*63),
    lambda d: d['files'][0].update(category=''),
    lambda d: d.update(external_imports={'numpy':['unlisted.py']}),
    lambda d: d.update(external_imports={'numpy':[]}),
    lambda d: d.update(external_imports={'numpy':[d['files'][0]['path']]*2}),
])
def test_manifest_schema_has_no_silent_coercions(tree, change):
    document = copy.deepcopy(tree[1])
    change(document)
    write(tree, document)
    with pytest.raises(ValueError):
        verify(tree)


@pytest.mark.parametrize('raw', [
    '{"schema_version":1,"schema_version":1}',
    '{"files":[{"path":"a","path":"b"}]}',
    '{"external_imports":{"numpy":[],"numpy":[]}}',
])
def test_duplicate_json_keys_are_not_overwritten(tree, raw):
    (tree[0]/'manifest.json').write_text(raw)
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        verify(tree)


def test_duplicate_file_entries_and_manifest_self_hash_are_rejected(tree):
    document = copy.deepcopy(tree[1])
    document['files'].append(copy.deepcopy(document['files'][0]))
    write(tree, document)
    with pytest.raises(ValueError, match='Duplicate manifest path'):
        verify(tree)
    document['files'][-1]['path'] = 'manifest.json'
    write(tree, document)
    with pytest.raises(ValueError, match='cannot list its own hash'):
        verify(tree)


@pytest.mark.parametrize('location', ['source', 'directory', 'manifest'])
def test_symlink_components_rejected_even_when_the_target_is_inside(tree, location):
    root, document = tree
    if location == 'source':
        path = root/'src/qf_research/example.py'
        path.unlink()
        path.symlink_to(root/'src/qf_research/__init__.py')
    elif location == 'directory':
        (root/'alias').symlink_to(root/'src', target_is_directory=True)
        document['files'][0]['path'] = 'alias/qf_research/__init__.py'
        write(tree, document)
    else:
        (root/'manifest.json').rename(root/'real_manifest.json')
        (root/'manifest.json').symlink_to(root/'real_manifest.json')
    with pytest.raises(ValueError, match='Symlink component'):
        verify(tree)


def test_required_execution_file_cannot_be_omitted(tree):
    with pytest.raises(ValueError, match='required execution source'):
        verifier.verify_sources(tree[0], 'manifest.json', required_paths=('run.py',))


def test_loaded_modules_bind_to_listed_local_sources(tree):
    root, _ = tree
    modules = {'qf_research':module('qf_research', root/'src/qf_research/__init__.py', package=True),
        'qf_research.example':module('qf_research.example', root/'src/qf_research/example.py')}
    result = verifier.check_import_origins(verify(tree), root, modules=modules)
    assert result == {name:Path(m.__file__).relative_to(root).as_posix()
                      for name,m in modules.items()}


def test_same_version_foreign_package_cannot_use_local_source_verification(tree, tmp_path):
    imported = module('qf_research', tmp_path/'installed/qf_research/__init__.py', package=True)
    imported.__version__ = '0.1.0'
    with pytest.raises(ValueError, match='outside the verified checkout'):
        verifier.check_import_origins(verify(tree), tree[0], modules={'qf_research':imported})


def test_unlisted_module_and_extended_package_path_are_rejected(tree):
    root, _ = tree
    package = module('qf_research', root/'src/qf_research/__init__.py', package=True)
    other = module('qf_research.unlisted', root/'src/qf_research/unlisted.py')
    with pytest.raises(ValueError, match='not a listed source'):
        verifier.check_import_origins(verify(tree), root,
            modules={'qf_research':package,'qf_research.unlisted':other})
    package.__path__.append(str(root/'elsewhere'))
    with pytest.raises(ValueError, match='package search path differs'):
        verifier.check_import_origins(verify(tree), root, modules={'qf_research':package})
