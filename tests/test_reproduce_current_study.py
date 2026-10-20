"""Current public reproduction must retain real failures and preserve sources."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'scripts/reproduce_current_study.py'
SPEC = importlib.util.spec_from_file_location('reproduce_current_study_under_test', SCRIPT)
REPRODUCTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRODUCTION)


def test_help_uses_standard_library_and_declares_limited_scope():
    result = subprocess.run([sys.executable, '-S', str(SCRIPT), '--help'],
                            capture_output=True, text=True, check=True)
    assert '--output' in result.stdout and '--with-native' in result.stdout
    assert '--compiler' in result.stdout
    assert 'not rerun' in result.stdout
    assert '--input' not in result.stdout and '--config' not in result.stdout


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    verifier = REPRODUCTION.source_checker()
    monkeypatch.setattr(REPRODUCTION, 'ROOT', tmp_path)
    monkeypatch.setattr(REPRODUCTION, 'source_checker', lambda: verifier)
    source = tmp_path/'scripts/reproduce_current_study.py'
    source.parent.mkdir()
    source.write_text('# Declared orchestration fixture.\n')
    notebook = tmp_path/REPRODUCTION.NOTEBOOK
    notebook.parent.mkdir()
    numerical_sources = []
    for name in REPRODUCTION.NOTEBOOK_SOURCES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Declared numerical source fixture.\n')
        numerical_sources.append(path)
    notebook.write_text(json.dumps(dict(nbformat=4, nbformat_minor=5,
        metadata=dict(author='Amanjeet Singh', title='Declared computation',
            language_info=dict(name='python', version='3.13.9'),
            simulation_design=dict(training_paths=4, evaluation_paths=8),
            source_sha256={path.relative_to(tmp_path).as_posix():
                           hashlib.sha256(path.read_bytes()).hexdigest() for path in numerical_sources}),
        cells=[dict(cell_type='markdown', id='question', metadata={}, source='A declared calculation.'),
               dict(cell_type='code', id='first', metadata={}, source='answer = 2+2',
                    execution_count=None, outputs=[]),
               dict(cell_type='code', id='second', metadata={}, source='assert answer == 4',
                    execution_count=None, outputs=[])])))
    monkeypatch.setattr(REPRODUCTION, 'CORE_SOURCES',
                        ('scripts/reproduce_current_study.py', REPRODUCTION.NOTEBOOK))
    monkeypatch.setattr(REPRODUCTION, 'CURRENT_TESTS', ('tests/declared_fixture.py',))
    def manifest(extra=()):
        entries = []
        for path in dict.fromkeys((source, notebook, *numerical_sources, *extra)):
            payload = path.read_bytes()
            entries.append(dict(path=path.relative_to(tmp_path).as_posix(),
                category='test_fixture', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
        path = tmp_path/REPRODUCTION.MANIFEST
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(dict(schema_version=1, title='Declared orchestration test',
            scope='No provider input', dependency_scope='Test fixtures only',
            files=entries, external_imports={})))
    manifest()
    return tmp_path, manifest


def successful_stub(name, arguments, output, records):
    records.append(dict(name=name, arguments=arguments, returncode=0))
    (output/(name+'.log')).write_text('Declared successful fixture.\n')
    if name == 'current_notebook':
        target = output/'notebooks/pricing_and_stopping.ipynb'
        target.parent.mkdir()
        notebook = json.loads((REPRODUCTION.ROOT/REPRODUCTION.NOTEBOOK).read_text())
        count = 0
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                count += 1
                cell['execution_count'] = count
                cell['outputs'] = []
        notebook['metadata']['execution'] = dict(fresh_kernel=True,
            executed_code_cells=count, figures=0, native_cpp_executed=False,
            python_version='3.13.9', date='2026-10-19',
            started_utc='2026-10-19T10:00:00+00:00', completed_utc='2026-10-19T10:01:00+00:00')
        target.write_text(json.dumps(notebook))


def test_success_requires_all_default_steps_and_a_separate_executed_copy(fixture, monkeypatch):
    root, _ = fixture
    monkeypatch.setattr(REPRODUCTION, 'step', successful_stub)
    original = (root/REPRODUCTION.NOTEBOOK).read_bytes()
    output = root/'data/private/current_run'
    record = REPRODUCTION.reproduce(output)
    assert record['complete'] is True
    assert [step['name'] for step in record['steps']] == [
        'current_aggregates', 'current_tests', 'current_notebook']
    assert record['source_sha256_before'] == record['source_sha256_after']
    assert record['notebook']['execution']['fresh_kernel'] is True
    assert record['native_verification_requested'] is False
    assert not record['research_experiments_replayed']
    assert not record['provider_inputs_read_or_exported']
    assert (root/REPRODUCTION.NOTEBOOK).read_bytes() == original
    assert 'notebooks/pricing_and_stopping.ipynb' in record['artifact_sha256']


@pytest.mark.parametrize('failed_step', ['current_aggregates', 'current_tests', 'current_notebook'])
def test_failure_stops_at_requested_step_and_retains_incomplete_record(fixture, monkeypatch, failed_step):
    root, _ = fixture
    def failing(name, arguments, output, records):
        if name == failed_step:
            records.append(dict(name=name, returncode=7))
            (output/(name+'.log')).write_text('Declared failed fixture.\n')
            raise RuntimeError(name+' failed')
        successful_stub(name, arguments, output, records)
    monkeypatch.setattr(REPRODUCTION, 'step', failing)
    output = root/'data/private/failed_run'
    with pytest.raises(RuntimeError, match=failed_step+' failed'):
        REPRODUCTION.reproduce(output)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert record['steps'][-1]['name'] == failed_step
    assert record['steps'][-1]['returncode'] == 7
    assert failed_step+'.log' in record['artifact_sha256']


def test_changed_source_after_execution_cannot_pass(fixture, monkeypatch):
    root, _ = fixture
    def mutation(name, arguments, output, records):
        successful_stub(name, arguments, output, records)
        if name == 'current_notebook':
            path = root/'scripts/reproduce_current_study.py'
            path.write_text(path.read_text()+'# Declared concurrent edit.\n')
    monkeypatch.setattr(REPRODUCTION, 'step', mutation)
    output = root/'data/private/changed_run'
    with pytest.raises(ValueError, match='byte count differs'):
        REPRODUCTION.reproduce(output)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert 'concurrent edit' in (root/'scripts/reproduce_current_study.py').read_text()


def test_changed_manifest_with_updated_hashes_also_cannot_pass(fixture, monkeypatch):
    root, manifest = fixture
    def mutation(name, arguments, output, records):
        successful_stub(name, arguments, output, records)
        if name == 'current_notebook':
            path = root/'scripts/reproduce_current_study.py'
            path.write_text(path.read_text()+'# Declared concurrent edit.\n')
            manifest()
    monkeypatch.setattr(REPRODUCTION, 'step', mutation)
    output = root/'data/private/changed_manifest'
    with pytest.raises(ValueError, match='changed during reproduction'):
        REPRODUCTION.reproduce(output)
    assert json.loads((output/'verification.json').read_text())['complete'] is False


@pytest.mark.parametrize('selection', ['public', 'data/private/existing',
                                      'data/private/../outside', 'data/private/link/new'])
def test_invalid_output_rejected_before_any_verification_or_execution(fixture, monkeypatch, selection):
    root, _ = fixture
    retained = root/'data/private/existing'
    retained.mkdir(parents=True)
    (retained/'keep').write_text('Retained result')
    (root/'data/private/link').symlink_to(root/'notebooks', target_is_directory=True)
    def unexpected():
        raise AssertionError('Output must be checked before reading inventory')
    monkeypatch.setattr(REPRODUCTION, 'source_checker', unexpected)
    with pytest.raises((ValueError, FileExistsError)):
        REPRODUCTION.reproduce(root/selection)
    assert (retained/'keep').read_text() == 'Retained result'
    assert not (root/'notebooks/new').exists()


def test_manifest_rejects_private_inputs_before_reading_them(fixture, monkeypatch):
    root, manifest = fixture
    private = root/'data/private/provider.csv'
    private.parent.mkdir(parents=True)
    private.write_text('Declared forbidden fixture')
    manifest((private,))
    def unexpected(*args, **kwargs):
        raise AssertionError('No substep should run')
    monkeypatch.setattr(REPRODUCTION, 'step', unexpected)
    output = root/'data/private/forbidden_run'
    with pytest.raises(ValueError, match='must not select private or provider inputs'):
        REPRODUCTION.reproduce(output)
    assert json.loads((output/'verification.json').read_text())['complete'] is False


def test_successful_process_without_executed_notebook_cannot_pass(fixture, monkeypatch):
    root, _ = fixture
    def incomplete(name, arguments, output, records):
        records.append(dict(name=name, returncode=0))
    monkeypatch.setattr(REPRODUCTION, 'step', incomplete)
    output = root/'data/private/missing_copy'
    with pytest.raises(ValueError, match='notebook copy is missing'):
        REPRODUCTION.reproduce(output)
    assert json.loads((output/'verification.json').read_text())['complete'] is False


def test_native_request_is_not_successful_without_passed_native_ledger(fixture, monkeypatch):
    root, manifest = fixture
    driver = root/'scripts/verify_bermudan_tree.py'
    native_source = root/'cpp/bermudan_tree.cpp'
    native_source.parent.mkdir(exist_ok=True)
    driver.write_text('# Declared native driver fixture.\n')
    manifest((driver, native_source))
    def bad_native(name, arguments, output, records):
        if name != 'native_pricer':
            return successful_stub(name, arguments, output, records)
        records.append(dict(name=name, returncode=0))
        directory = output/'native'
        directory.mkdir()
        (directory/'verification.json').write_text(json.dumps(dict(
            status='failed', checks=2, passed_checks=1, failures=['Declared failure'])))
    monkeypatch.setattr(REPRODUCTION, 'step', bad_native)
    output = root/'data/private/native_failure'
    with pytest.raises(ValueError, match='ledger or executable is missing'):
        REPRODUCTION.reproduce(output, with_native=True, compiler='declared-compiler')
    assert json.loads((output/'verification.json').read_text())['complete'] is False


def test_actual_child_exit_and_arguments_are_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(REPRODUCTION, 'ROOT', tmp_path)
    output = tmp_path/'logs'
    output.mkdir()
    records = []
    with pytest.raises(RuntimeError, match='declared_step failed'):
        REPRODUCTION.step('declared_step', ['-c', 'print("Declared failure"); raise SystemExit(9)'],
                          output, records)
    assert records[0]['returncode'] == 9
    assert records[0]['command'][:2] == [sys.executable, '-B']
    assert records[0]['cwd'] == str(tmp_path)
    assert 'Declared failure' in (output/'declared_step.log').read_text()
    assert records[0]['finished_utc'] >= records[0]['started_utc']


def test_child_environment_uses_this_checkout_and_one_numerical_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(REPRODUCTION, 'ROOT', tmp_path)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
        monkeypatch.setenv(name, '/declared/foreign/path')
    environment = REPRODUCTION.child_environment()
    assert environment['PYTHONPATH'] == str(tmp_path/'src')
    assert not any(name in environment for name in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'))
    assert all(environment[name] == '1' for name in REPRODUCTION.THREAD_VARIABLES)


def test_failed_child_and_source_mutation_are_both_retained(fixture, monkeypatch):
    root, _ = fixture
    def changed_failure(name, arguments, output, records):
        path = root/'scripts/reproduce_current_study.py'
        path.write_text(path.read_text()+'# Declared interrupted edit.\n')
        records.append(dict(name=name, returncode=3))
        raise RuntimeError('Declared simultaneous failure')
    monkeypatch.setattr(REPRODUCTION, 'step', changed_failure)
    output = root/'data/private/changed_failure'
    with pytest.raises(RuntimeError, match='Declared simultaneous failure'):
        REPRODUCTION.reproduce(output)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert record['failure']['message'] == 'Declared simultaneous failure'
    assert 'byte count differs' in record['source_verification_after_failure']['message']


def test_requested_native_success_is_recorded_only_after_native_ledger(fixture, monkeypatch):
    root, manifest = fixture
    driver = root/'scripts/verify_bermudan_tree.py'
    native_source = root/'cpp/bermudan_tree.cpp'
    native_source.parent.mkdir(exist_ok=True)
    driver.write_text('# Declared native driver fixture.\n')
    manifest((driver, native_source))
    def native_pass(name, arguments, output, records):
        if name != 'native_pricer':
            return successful_stub(name, arguments, output, records)
        records.append(dict(name=name, arguments=arguments, returncode=0))
        successful_native_fixture(output)
    monkeypatch.setattr(REPRODUCTION, 'step', native_pass)
    record = REPRODUCTION.reproduce(root/'data/private/native_success',
                                    with_native=True, compiler='declared-compiler')
    assert record['complete'] is True
    assert record['native_checks'] == 2
    assert record['compiler'] == 'declared-compiler'
    assert record['steps'][-1]['name'] == 'native_pricer'


def successful_native_fixture(output):
    directory = output/'native'
    directory.mkdir()
    executable = directory/'bermudan_tree'
    executable.write_bytes(b'Declared executable fixture')
    executions = []
    for returncode, ok in ((0, True), (2, False)):
        result = dict(author='Amanjeet Singh', ok=ok)
        executions.append(dict(arguments=['declared'], returncode=returncode,
                               stdout=json.dumps(result), stderr='', result=result))
    native = dict(author='Amanjeet Singh', status='passed', checks=2, passed_checks=2,
        failures=[], cases=[dict(name='Declared '+str(index), passed=True, **invocation)
                           for index, invocation in enumerate(executions)],
        source_sha256={str(REPRODUCTION.ROOT/name):REPRODUCTION.digest(REPRODUCTION.ROOT/name)
                       for name in REPRODUCTION.NATIVE_SOURCES},
        executable_sha256=REPRODUCTION.digest(executable), compiler='/declared/compiler',
        compiler_version='Declared compiler version', compiler_version_returncode=0,
        flags=REPRODUCTION.NATIVE_FLAGS, compilation_returncode=0,
        command=['/declared/compiler', *REPRODUCTION.NATIVE_FLAGS,
                 str(REPRODUCTION.ROOT/'cpp/bermudan_tree.cpp'), '-o', str(executable)],
        native_executions=executions)
    (directory/'verification.json').write_text(json.dumps(native))
    return native


@pytest.mark.parametrize('change', ['code', 'markdown', 'cell_order', 'author', 'design', 'source_hash',
                                    'duplicate_count', 'reordered_count', 'boolean_count',
                                    'error_output', 'interpreter', 'backward_time', 'figure_count'])
def test_plausible_notebook_metadata_cannot_hide_changed_source_or_failed_execution(fixture, monkeypatch, change):
    root, _ = fixture
    def tamper(name, arguments, output, records):
        successful_stub(name, arguments, output, records)
        if name != 'current_notebook':
            return
        path = output/'notebooks/pricing_and_stopping.ipynb'
        notebook = json.loads(path.read_text())
        metadata, cells = notebook['metadata'], notebook['cells']
        if change == 'code': cells[1]['source'] = 'answer = 0'
        if change == 'markdown': cells[0]['source'] = 'An unrelated claim.'
        if change == 'cell_order': cells[0], cells[1] = cells[1], cells[0]
        if change == 'author': metadata['author'] = 'Another author'
        if change == 'design': metadata['simulation_design']['evaluation_paths'] = 99
        if change == 'source_hash': metadata['source_sha256'][REPRODUCTION.NOTEBOOK_SOURCES[0]] = '0'*64
        if change == 'duplicate_count': cells[2]['execution_count'] = 1
        if change == 'reordered_count': cells[1]['execution_count'], cells[2]['execution_count'] = 2, 1
        if change == 'boolean_count': cells[1]['execution_count'] = True
        if change == 'error_output': cells[2]['outputs'] = [dict(output_type='error', ename='AssertionError')]
        if change == 'interpreter': metadata['execution']['python_version'] = '0.0.0'
        if change == 'backward_time': metadata['execution']['completed_utc'] = '2026-10-19T09:59:00+00:00'
        if change == 'figure_count': metadata['execution']['figures'] = 2
        path.write_text(json.dumps(notebook))
    monkeypatch.setattr(REPRODUCTION, 'step', tamper)
    output = root/'data/private/tampered_notebook'
    with pytest.raises(ValueError):
        REPRODUCTION.reproduce(output)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert record['steps'][-1]['name'] == 'current_notebook'
    assert record['source_sha256_before'] == record['source_sha256_after']
    assert 'notebooks/pricing_and_stopping.ipynb' in record['artifact_sha256']


def test_stale_numerical_hash_in_declared_notebook_cannot_pass(fixture, monkeypatch):
    root, manifest = fixture
    source = root/REPRODUCTION.NOTEBOOK
    notebook = json.loads(source.read_text())
    notebook['metadata']['source_sha256'][REPRODUCTION.NOTEBOOK_SOURCES[0]] = '1'*64
    source.write_text(json.dumps(notebook))
    manifest()
    monkeypatch.setattr(REPRODUCTION, 'step', successful_stub)
    with pytest.raises(ValueError, match='numerical source identities differ'):
        REPRODUCTION.reproduce(root/'data/private/stale_notebook')


@pytest.mark.parametrize('change', ['counts_only', 'missing_case', 'false_case', 'boolean_case',
    'duplicate_case', 'foreign_source', 'stale_executable', 'failed_compile', 'flags',
    'missing_invocations', 'missing_case_invocation', 'changed_invocation', 'boolean_count'])
def test_native_pass_label_cannot_replace_case_build_and_source_evidence(fixture, monkeypatch, change):
    root, manifest = fixture
    driver = root/'scripts/verify_bermudan_tree.py'
    driver.write_text('# Declared native driver fixture.\n')
    manifest((driver,))
    def tamper(name, arguments, output, records):
        if name != 'native_pricer':
            return successful_stub(name, arguments, output, records)
        records.append(dict(name=name, returncode=0))
        native = successful_native_fixture(output)
        if change == 'counts_only': native.pop('cases')
        if change == 'missing_case': native['cases'].pop()
        if change == 'false_case': native['cases'][0]['passed'] = False
        if change == 'boolean_case': native['cases'][0]['passed'] = 1
        if change == 'duplicate_case': native['cases'][1]['name'] = native['cases'][0]['name']
        if change == 'foreign_source': native['source_sha256'][str(root/REPRODUCTION.NATIVE_SOURCES[0])] = '0'*64
        if change == 'stale_executable': (output/'native/bermudan_tree').write_bytes(b'Altered executable')
        if change == 'failed_compile': native['compilation_returncode'] = 1
        if change == 'flags': native['flags'] = ['-ffast-math']
        if change == 'missing_invocations': native['native_executions'] = []
        if change == 'missing_case_invocation': native['cases'][0].pop('result')
        if change == 'changed_invocation': native['native_executions'][0]['stdout'] = '{}'
        if change == 'boolean_count': native['checks'] = True
        (output/'native/verification.json').write_text(json.dumps(native))
    monkeypatch.setattr(REPRODUCTION, 'step', tamper)
    output = root/'data/private/tampered_native'
    with pytest.raises(ValueError):
        REPRODUCTION.reproduce(output, with_native=True)
    record = json.loads((output/'verification.json').read_text())
    assert record['complete'] is False
    assert record['steps'][-1]['name'] == 'native_pricer'
    assert 'native/verification.json' in record['artifact_sha256']
    assert record['source_sha256_before'] == record['source_sha256_after']
