"""Public-only selection, interpreter isolation and honest failure records."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/reproduce_paper.py'
spec = importlib.util.spec_from_file_location('reproduce_public_paper', SCRIPT)
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


def test_help_needs_no_scientific_environment_and_offers_no_financial_input():
    process = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'],
                             capture_output=True, text=True, check=True)
    assert '--output' in process.stdout and '--full-tests' in process.stdout
    assert '--config' not in process.stdout
    assert 'not regenerated here' in process.stdout


def test_existing_or_public_output_is_rejected_without_touching_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduction, 'ROOT', tmp_path)
    existing = tmp_path/'data/private/existing'
    existing.mkdir(parents=True)
    (existing/'retained.txt').write_text('preserve')
    with pytest.raises(FileExistsError, match='never merged'):
        reproduction.reproduce(existing)
    with pytest.raises(ValueError, match='under data/private'):
        reproduction.reproduce(tmp_path/'public')
    assert (existing/'retained.txt').read_text() == 'preserve'
    assert not (tmp_path/'public').exists()


def test_children_do_not_inherit_foreign_python_search_or_optimisation(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduction, 'ROOT', tmp_path)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
        monkeypatch.setenv(key, '/declared/foreign/test/path')
    environment = reproduction.child_environment()
    assert environment['PYTHONPATH'] == str(tmp_path/'src')
    assert not any(key in environment for key in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'))
    assert environment['PYTHONNOUSERSITE'] == '1'
    assert all(environment[key] == '1' for key in reproduction.THREAD_VARIABLES)


def test_failed_public_check_retains_incomplete_record_and_stops(tmp_path, monkeypatch):
    verifier = reproduction.source_checker()
    monkeypatch.setattr(reproduction, 'ROOT', tmp_path)
    monkeypatch.setattr(reproduction, 'source_checker', lambda:verifier)
    first = reproduction.VERIFICATION + '/numerical_results.py'
    files = {'scripts/reproduce_paper.py':b'# Declared file-inventory fixture.\n',
             first:b'print("Declared failed check")\nraise SystemExit(7)\n'}
    entries = []
    for name,data in files.items():
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entries.append(dict(path=name, category='test_fixture', bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest()))
    manifest = tmp_path/verifier.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(schema_version=1, title='Declared test fixture',
        scope='No financial data', dependency_scope='Failure propagation check',
        files=entries, external_imports={})))
    monkeypatch.setattr(reproduction, 'PUBLIC_INPUTS', (first,))
    output = tmp_path/'data/private/new_run'
    with pytest.raises(RuntimeError, match='numerical_results failed'):
        reproduction.reproduce(output)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert record['steps'] == [dict(name='numerical_results', returncode=7,
                                  log='numerical_results.log')]
    assert record['financial_simulation_tables_replayed'] is False
    assert record['published_cpu_workloads_retimed'] is False
    assert record['raw_market_data_read_or_exported'] is False
    assert not (output/'fixture.json').exists()
    assert not (output/'notebooks').exists()


def test_only_named_notebooks_execute_in_invoking_environment(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    pytest.importorskip('nbclient')
    monkeypatch.setattr(reproduction, 'ROOT', tmp_path)
    package = tmp_path/'src/declared_fixture_package'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('ORIGIN = "verified local fixture"\n')
    foreign = tmp_path/'foreign/declared_fixture_package'
    foreign.mkdir(parents=True)
    (foreign/'__init__.py').write_text('raise AssertionError("Foreign fixture imported")\n')
    monkeypatch.setenv('PYTHONPATH', str(foreign.parent))
    code = ('import sys\nimport declared_fixture_package as fixture\n'
            f'assert sys.executable == {sys.executable!r}\n'
            'assert fixture.ORIGIN == "verified local fixture"\n'
            'print("Fresh declared calculation")')
    original = {}
    for name in reproduction.NOTEBOOKS:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code,
            execution_count=99, outputs=[nbformat.v4.new_output('stream',
                name='stdout', text='Stale saved output')])])
        nbformat.write(notebook, path)
        original[name] = path.read_bytes()
    other = tmp_path/Path(reproduction.NOTEBOOKS[0]).parent/'99_unpublished.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        'raise AssertionError("Unpublished notebook must not execute")')]), other)
    output = tmp_path/'data/private/fresh_notebooks'
    records = reproduction.execute_notebooks(output)
    assert [record['source'] for record in records] == list(reproduction.NOTEBOOKS)
    assert len(list(output.glob('*.ipynb'))) == 2
    for name,raw in original.items():
        assert (tmp_path/name).read_bytes() == raw
        result = nbformat.read(output/Path(name).name, as_version=4)
        assert result.cells[0].execution_count == 1
        assert result.cells[0].outputs[0].text == 'Fresh declared calculation\n'


def test_release_runner_ignores_foreign_default_kernel_and_python_path(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    pytest.importorskip('nbclient')
    script = ROOT/'scripts/execute_release_notebooks.py'
    spec = importlib.util.spec_from_file_location('released_notebook_execution', script)
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    monkeypatch.setattr(release, 'ROOT', tmp_path)
    monkeypatch.setattr(release, 'NOTEBOOKS', ('notebooks/declared.ipynb',))

    kernel = tmp_path/'foreign_kernels/kernels/python3'
    kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-c', 'raise RuntimeError("Foreign default kernel selected")',
              '{connection_file}'], display_name='Foreign fixture', language='python')))
    monkeypatch.setenv('JUPYTER_PATH', str(kernel.parent.parent))

    package = tmp_path/'src/declared_release_package'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('ORIGIN = "local fixture"\n')
    foreign = tmp_path/'foreign_imports/declared_release_package'
    foreign.mkdir(parents=True)
    (foreign/'__init__.py').write_text('raise AssertionError("Foreign package selected")\n')
    monkeypatch.setenv('PYTHONPATH', str(foreign.parent))

    notebook = tmp_path/release.NOTEBOOKS[0]
    notebook.parent.mkdir()
    code = ('import sys\nimport declared_release_package as fixture\n'
            f'assert sys.executable == {sys.executable!r}\n'
            'assert fixture.ORIGIN == "local fixture"\n'
            'print("Fresh release calculation")')
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)]), notebook)
    original = notebook.read_bytes()
    records = release.execute()
    assert records[0]['code_cells_executed'] == 1
    assert records[0]['saved_outputs'] is False
    assert notebook.read_bytes() == original
