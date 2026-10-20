"""Review-scope and byte-consistency checks for the current inventory builder."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('current_inventory_builder', ROOT/'scripts/build_current_inventory.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def checkout(tmp_path):
    (tmp_path/'replication').mkdir()
    shutil.copyfile(ROOT/'replication/verify_sources.py', tmp_path/'replication/verify_sources.py')
    files = {
        'README.md': b'# Declared mathematical fixture\n',
        'src/qf_research/__init__.py': b'from .example import VALUE\n',
        'src/qf_research/example.py': b'import fractions\nVALUE = 2\n',
        'scripts/example.py': b'from qf_research.example import VALUE\nimport numpy as np\n',
    }
    for name,data in files.items():
        path=tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    groups={
        'reader_documentation':('README.md',),
        'verification':('replication/verify_sources.py',),
        'numerical_source':('src/qf_research/__init__.py','src/qf_research/example.py'),
        'research_script':('scripts/example.py',),
    }
    return tmp_path,groups


def build(checkout):
    root,groups=checkout
    return builder.build_manifest(root,groups=groups)


def test_generation_is_deterministic_and_has_no_default_write(checkout):
    root,groups=checkout
    first=builder.encoded_manifest(build(checkout))
    reordered={key:tuple(reversed(value)) for key,value in reversed(list(groups.items()))}
    assert first==builder.encoded_manifest(builder.build_manifest(root,groups=reordered))
    assert not (root/builder.MANIFEST).exists()
    record=json.loads(first)
    assert record['external_imports']=={'numpy':['scripts/example.py']}
    assert [row['path'] for row in record['files']]==sorted(row['path'] for row in record['files'])


def test_public_inventory_passes_existing_verifier_then_detects_change(checkout):
    root,_=checkout
    builder.write_manifest(root,build(checkout))
    verifier=builder.load_verifier(root)
    result=verifier.verify_sources(root,builder.MANIFEST)
    assert result['verified_source_files']==5
    (root/'src/qf_research/example.py').write_bytes(b'import fractions\nVALUE = 3\n')
    with pytest.raises(ValueError,match='Source SHA-256'):
        verifier.verify_sources(root,builder.MANIFEST)


def test_standalone_verifier_checks_current_schema_without_packages(checkout):
    root,_=checkout
    builder.write_manifest(root,build(checkout))
    result=subprocess.run([sys.executable,'-S',str(root/'replication/verify_sources.py'),
        '--root',str(root),'--manifest',builder.MANIFEST],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['verified_source_files']==5


@pytest.mark.parametrize('name',[
    'data/private/provider.csv','data/public/quotes.csv','results/provider.json',
    '.git/config','.venv/bin/python','paper/figures/generated.pdf',
    'src/qf_research/__pycache__/example.py','src/qf_research/raw/input.py',
])
def test_private_raw_and_generated_files_cannot_enter_any_category(checkout,name):
    root,groups=checkout
    for category in builder.REVIEWED_GROUPS:
        amended=copy.deepcopy(groups)
        amended[category]=(name,)
        with pytest.raises(ValueError,match='outside the reviewed public source scope'):
            builder.build_manifest(root,groups=amended)


@pytest.mark.parametrize('name',['../secret.py','/tmp/secret.py','scripts/../secret.py',
    './scripts/example.py','scripts//example.py','scripts\\example.py'])
def test_noncanonical_paths_are_rejected_before_access(checkout,name):
    root,groups=checkout
    groups['research_script']=(name,)
    with pytest.raises(ValueError,match='canonical relative POSIX'):
        builder.build_manifest(root,groups=groups)


def test_unlisted_private_file_does_not_change_inventory(checkout):
    root,_=checkout
    first=builder.encoded_manifest(build(checkout))
    private=root/'data/private/secret.csv'
    private.parent.mkdir(parents=True)
    private.write_text('not part of public model inputs\n')
    assert builder.encoded_manifest(build(checkout))==first
    assert b'secret' not in first


def test_duplicate_selection_rejected(checkout):
    root,groups=checkout
    groups['research_script']*=2
    with pytest.raises(ValueError,match='Duplicate reviewed source'):
        builder.build_manifest(root,groups=groups)


@pytest.mark.parametrize('name',[builder.MANIFEST,'replication/source_manifest.json'])
def test_neither_inventory_can_hash_itself_or_the_historical_inventory(checkout,name):
    root,groups=checkout
    groups['paper_source']=(name,)
    with pytest.raises(ValueError,match='Inventory manifests are outside'):
        builder.build_manifest(root,groups=groups)


@pytest.mark.parametrize('inside',[True,False])
def test_symlink_source_rejected_for_internal_or_external_target(checkout,tmp_path,inside):
    root,_=checkout
    target=root/'README.md' if inside else tmp_path.parent/'outside_source.py'
    if not inside:target.write_text('VALUE=7\n')
    path=root/'scripts/example.py'
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError,match='Symlink component'):
        build(checkout)


def test_symlink_parent_rejected(checkout):
    root,_=checkout
    actual=root/'renamed_scripts'
    (root/'scripts').rename(actual)
    (root/'scripts').symlink_to(actual,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink component'):
        build(checkout)


def test_missing_selection_fails_instead_of_silently_omitting_it(checkout):
    root,_=checkout
    (root/'README.md').unlink()
    with pytest.raises(ValueError,match='Missing or non-regular'):
        build(checkout)


def test_static_project_import_requires_listed_source(checkout):
    root,groups=checkout
    groups['numerical_source']=('src/qf_research/__init__.py',)
    with pytest.raises(ValueError,match='Unlisted static project import'):
        builder.build_manifest(root,groups=groups)


def test_extra_schema_field_is_rejected_before_output_creation(checkout):
    root,_=checkout
    document=build(checkout)
    document['not_in_schema']=True
    with pytest.raises(ValueError,match='Unexpected manifest fields'):
        builder.write_manifest(root,document)
    assert not (root/builder.MANIFEST).exists()


def test_symlinked_output_is_not_overwritten(checkout):
    root,_=checkout
    document=build(checkout)
    (root/builder.MANIFEST).symlink_to(root/'README.md')
    original=(root/'README.md').read_bytes()
    with pytest.raises(ValueError,match='regular local path'):
        builder.write_manifest(root,document)
    assert (root/'README.md').read_bytes()==original


def test_changed_source_before_write_is_not_sealed(checkout):
    root,_=checkout
    document=build(checkout)
    (root/'README.md').write_text('# Changed after review\n')
    with pytest.raises(ValueError,match='changed before inventory write'):
        builder.write_manifest(root,document)
    assert not (root/builder.MANIFEST).exists()


def test_parent_package_must_also_be_listed(checkout):
    root,groups=checkout
    groups['numerical_source']=('src/qf_research/example.py',)
    with pytest.raises(ValueError,match='Unlisted static project import qf_research'):
        builder.build_manifest(root,groups=groups)


def test_static_sibling_script_cannot_be_misreported_as_external(checkout):
    root,_=checkout
    (root/'scripts/helper.py').write_text('VALUE=1\n')
    (root/'scripts/example.py').write_text('from helper import VALUE\n')
    with pytest.raises(ValueError,match='Unlisted static sibling import helper'):
        build(checkout)



def test_reader_paths_resolve_names_and_starred_test_sequence_without_execution():
    source = b"NOTEBOOK='notebooks/example.ipynb'\nCURRENT_TESTS=('tests/example.py',)\nCORE_SOURCES=('scripts/reader.py', NOTEBOOK, *CURRENT_TESTS)\nraise RuntimeError('not executed')\n"
    assert builder.declared_reader_paths(source) == (
        'scripts/reader.py', 'notebooks/example.ipynb', 'tests/example.py')


@pytest.mark.parametrize('source', [
    b"CORE_SOURCES=discover_sources()\n",
    b"CORE_SOURCES=('scripts/reader.py', *'tests/example.py')\n",
    b"CORE_SOURCES=()\n",
])
def test_reader_paths_reject_undeclared_or_executable_selection(source):
    with pytest.raises(ValueError, match='Reader|reader'):
        builder.declared_reader_paths(source)


def test_reader_selection_drift_requires_an_explicit_inventory_edit(checkout):
    root, groups = checkout
    reader = root/'scripts/reproduce_current_study.py'
    reader.write_text("CORE_SOURCES=('tests/new_reader_check.py',)\n")
    groups['research_script'] += ('scripts/reproduce_current_study.py',)
    with pytest.raises(ValueError, match='Unlisted declared source tests/new_reader_check.py'):
        builder.build_manifest(root, groups=groups)
    test = root/'tests/new_reader_check.py'
    test.parent.mkdir()
    test.write_text('assert True\n')
    groups['test_source'] = ('tests/new_reader_check.py',)
    assert builder.build_manifest(root, groups=groups)['files']


def test_ci_entry_points_include_literal_unittest_discovery():
    source = b"run: python replication/verify_sources.py --manifest replication/current_source_manifest.json\nrun: |\n  python -m pip install -r requirements.lock\n  python scripts/reproduce_current_study.py --output data/private/current\n  python -B -m unittest discover -s tests -p test_certificate.py -v\n  python scripts/test_certificate.py\n"
    assert builder.declared_ci_paths(source) == (
        'replication/verify_sources.py', 'scripts/reproduce_current_study.py',
        'scripts/test_certificate.py', 'tests/test_certificate.py')


@pytest.mark.parametrize('command', [
    'python -m unittest discover -s tests -p test_*.py',
    'python -m unittest discover',
    'python -m pytest',
    'python -m unreviewed_module',
])
def test_unsupported_ci_python_selection_requires_review(command):
    with pytest.raises(ValueError, match='Workflow|workflow'):
        builder.declared_ci_paths(('run: '+command+'\n').encode())


def test_ci_entry_point_drift_requires_the_called_source(checkout):
    root, groups = checkout
    workflow = root/'.github/workflows/ci.yml'
    workflow.parent.mkdir(parents=True)
    workflow.write_text('run: python scripts/new_interval_check.py\n')
    groups['environment'] = ('.github/workflows/ci.yml',)
    with pytest.raises(ValueError, match='Unlisted declared source scripts/new_interval_check.py'):
        builder.build_manifest(root, groups=groups)


def test_known_dynamic_test_load_must_be_selected(checkout):
    root, groups = checkout
    test = root/'tests/test_reproduce_paper.py'
    test.parent.mkdir()
    test.write_text("# This name has a separately reviewed dynamic source relationship.\n")
    groups['test_source'] = ('tests/test_reproduce_paper.py',)
    with pytest.raises(ValueError, match='Unlisted declared source scripts/reproduce_paper.py'):
        builder.build_manifest(root, groups=groups)


def test_reviewed_selection_covers_current_reader_and_ci_sources():
    verifier = builder.load_verifier(ROOT)
    selection = builder.reviewed_selection(builder.REVIEWED_GROUPS, verifier)
    # This reads only public source bytes. It does not run either workflow.
    blobs = {name: verifier.checked_file(ROOT, name).read_bytes() for name in selection}
    builder.check_declared_entrypoints(blobs)
    expected_interval = {
        'tests/test_certify_gain_replay.py', 'scripts/certify_gain_replay.py',
        'scripts/certify_residual_envelope.py', 'scripts/test_certified_residual.py',
    }
    assert expected_interval <= set(selection)
    assert set(builder.declared_reader_paths(blobs['scripts/reproduce_current_study.py'])) <= set(selection)
    assert set(builder.declared_ci_paths(blobs['.github/workflows/ci.yml'])) <= set(selection)



def test_reader_notebook_and_native_path_constants_are_both_required():
    source = b"NOTEBOOK_SOURCES=('src/qf_research/example.py',)\nNATIVE_SOURCES=('cpp/example.cpp', 'scripts/check_native.py')\nCURRENT_TESTS=('tests/example.py',)\nCORE_SOURCES=(*NOTEBOOK_SOURCES, *CURRENT_TESTS)\n"
    assert builder.declared_reader_paths(source) == (
        'src/qf_research/example.py', 'tests/example.py',
        'cpp/example.cpp', 'scripts/check_native.py')


def test_undeclared_native_reader_load_is_rejected():
    source = b"NATIVE_SOURCES=discover_native_sources()\nCORE_SOURCES=('scripts/reader.py',)\n"
    with pytest.raises(ValueError, match='Reader source constants need explicit review'):
        builder.declared_reader_paths(source)
