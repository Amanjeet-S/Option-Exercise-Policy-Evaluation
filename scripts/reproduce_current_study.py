"""Reproduce the current public-model checks and pricing notebook.

I check retained aggregate arithmetic and manuscript transcription, run selected
numerical tests, and execute the current notebook in a fresh kernel. Larger
research experiments, historical licensed-data replays and published CPU studies
are not rerun. No provider input, private archive or network request is selected.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'replication/current_source_manifest.json'
NOTEBOOK = 'notebooks/pricing_and_stopping.ipynb'
NOTEBOOK_SOURCES = (
    'cpp/bermudan_tree.cpp',
    'src/qf_research/optimal_stopping/analytical.py',
    'src/qf_research/optimal_stopping/contracts.py',
    'src/qf_research/optimal_stopping/finite_difference.py',
    'src/qf_research/optimal_stopping/least_squares.py',
    'src/qf_research/optimal_stopping/simulation.py',
    'src/qf_research/optimal_stopping/tree.py',
)
NATIVE_SOURCES = (
    'cpp/bermudan_tree.cpp',
    'scripts/verify_bermudan_tree.py',
    'src/qf_research/optimal_stopping/tree.py',
    'src/qf_research/optimal_stopping/contracts.py',
    'src/qf_research/optimal_stopping/analytical.py',
)
NATIVE_FLAGS = ['-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off']
CURRENT_TESTS = (
    'paper/verification/test_revision_results.py',
    'tests/test_optimal_stopping.py',
    'tests/test_multivariate_benchmark.py',
    'tests/test_reference_policy.py',
    'tests/test_policy_gain.py',
    'tests/test_gap_coverage.py',
    'tests/test_clipped_gap_coverage.py',
    'tests/test_policy_certification.py',
    'tests/test_sequential_inference.py',
    'tests/test_notebook_execution_metadata.py',
    'tests/test_reproduce_paper.py',
    'tests/test_reproduce_current_study.py',
    'tests/test_current_inventory.py',
)
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
CORE_SOURCES = (
    'scripts/reproduce_current_study.py', 'replication/verify_sources.py',
    'scripts/execute_release_notebooks.py', NOTEBOOK,
    'paper/verification/revision_results.py',
    'paper/revision_results.json', 'paper/main.tex', *NOTEBOOK_SOURCES, *CURRENT_TESTS,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_checker():
    path = ROOT/'replication/verify_sources.py'
    spec = importlib.util.spec_from_file_location('current_reproduction_sources', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_environment():
    environment = dict(os.environ)
    for name in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
        environment.pop(name, None)
    environment.update(PYTHONPATH=str(ROOT/'src'), PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    environment.update({name: '1' for name in THREAD_VARIABLES})
    return environment


def private_output(value):
    """Reject existing destinations, traversal and any symlink component."""
    root = ROOT.resolve(strict=True)
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Output must not contain parent traversal')
    path = path if path.is_absolute() else root/path
    if not path.is_relative_to(root/'data/private') or path == root/'data/private':
        raise ValueError('Output must be a fresh directory beneath data/private')
    current = root
    for part in path.relative_to(root).parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('Output must not contain symlink components')
        if current.exists() and not current.is_dir():
            raise ValueError('Output parent must be a directory')
    if path.exists():
        raise FileExistsError('Use a fresh output directory; previous runs are never replaced')
    return path


def inventory(verifier, *, required_paths):
    """Inspect the declared path set before reading any listed source bytes."""
    root = ROOT.resolve(strict=True)
    path = verifier.checked_file(root, MANIFEST)
    if path.stat().st_size > verifier.MAX_MANIFEST_BYTES:
        raise ValueError('Current source manifest exceeds the size limit')
    raw = path.read_bytes()
    entries = verifier.validate_manifest(verifier.read_json_bytes(raw), MANIFEST)
    for entry in entries:
        name = entry['path']
        if (name.startswith('.git/') or name == '.git'
                or (name.startswith('data/') and name != 'data/README.md')):
            raise ValueError('Current public inventory must not select private or provider inputs')
    result = verifier.verify_sources(root, MANIFEST, required_paths=required_paths)
    if path.read_bytes() != raw:
        raise ValueError('Current source manifest changed during verification')
    return result


def step(name, arguments, output, records):
    command = [sys.executable, '-B', *arguments]
    environment = child_environment()
    record = dict(name=name, command=command, cwd=str(ROOT), log=name+'.log',
                  started_utc=utc_now(), returncode=None,
                  environment={key: environment[key] for key in
                      ('PYTHONPATH', 'PYTHONNOUSERSITE', 'PYTHONDONTWRITEBYTECODE',
                       'MPLBACKEND', *THREAD_VARIABLES)})
    records.append(record)
    print('Running '+name, flush=True)
    try:
        with (output/record['log']).open('x', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, text=True)
        record['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError(name+' failed; inspect '+record['log'])
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record['finished_utc'] = utc_now()


def notebook_source_cells(notebook):
    """Compare every cell except outputs, counts and transient execution timings."""
    cells = notebook.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('The notebook must contain source cells')
    result = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get('cell_type') not in ('code', 'markdown', 'raw'):
            raise ValueError('The notebook contains an invalid source cell')
        source = cell.get('source')
        if isinstance(source, list) and all(isinstance(item, str) for item in source):
            source = ''.join(source)
        if not isinstance(source, str):
            raise ValueError('The notebook source cell must contain text')
        static = {key: value for key, value in cell.items()
                  if key not in ('outputs', 'execution_count', 'metadata', 'source')}
        metadata = cell.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('The notebook cell metadata are invalid')
        static.update(source=source, metadata={key: value for key, value in metadata.items()
                                              if key != 'execution'})
        result.append(static)
    return result


def checked_notebook_copy(output, before):
    path = output/'notebooks'/Path(NOTEBOOK).name
    if path.is_symlink() or not path.is_file():
        raise ValueError('The executed notebook copy is missing or is a symlink')
    source_path = ROOT/NOTEBOOK
    if digest(source_path) != before['file_sha256'][NOTEBOOK]:
        raise ValueError('The declared notebook source changed before copy validation')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    notebook = json.loads(path.read_text(encoding='utf-8'))
    if notebook_source_cells(notebook) != notebook_source_cells(source):
        raise ValueError('The executed notebook cells differ from the declared source')
    metadata = notebook.get('metadata', {})
    source_metadata = source.get('metadata', {})
    static = lambda value: {key: item for key, item in value.items()
                            if key not in ('execution', 'language_info')}
    if (static(metadata) != static(source_metadata)
            or metadata.get('author') != 'Amanjeet Singh'):
        raise ValueError('The executed notebook metadata differ from the declared source')
    expected_sources = {name: before['file_sha256'][name] for name in NOTEBOOK_SOURCES}
    if metadata.get('source_sha256') != expected_sources:
        raise ValueError('The notebook numerical source identities differ from the current inventory')
    execution = metadata.get('execution', {})
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    counts = [cell.get('execution_count') for cell in cells]
    figures = sum('image/png' in item.get('data', {})
                  for cell in cells for item in cell.get('outputs', []))
    if (execution.get('fresh_kernel') is not True or not cells
            or type(execution.get('executed_code_cells')) is not int
            or execution['executed_code_cells'] != len(cells)
            or any(type(count) is not int for count in counts)
            or counts != list(range(1, len(cells)+1))
            or any(item.get('output_type') == 'error'
                   for cell in cells for item in cell.get('outputs', []))
            or type(execution.get('native_cpp_executed')) is not bool
            or type(execution.get('figures')) is not int or execution['figures'] != figures):
        raise ValueError('The notebook copy lacks a successful sequential fresh-kernel execution')
    language = metadata.get('language_info', {})
    if (language.get('name') != 'python' or not isinstance(language.get('version'), str)
            or not language['version'] or execution.get('python_version') != language['version']):
        raise ValueError('The notebook execution interpreter metadata are inconsistent')
    try:
        started = datetime.fromisoformat(execution['started_utc'])
        finished = datetime.fromisoformat(execution['completed_utc'])
        if (started.utcoffset() is None or finished.utcoffset() is None or finished < started
                or execution.get('date') != finished.date().isoformat()):
            raise ValueError('Execution times differ')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('The notebook execution timestamps are invalid') from error
    return dict(source=NOTEBOOK, source_sha256=before['file_sha256'][NOTEBOOK],
                output=path.relative_to(output).as_posix(), sha256=digest(path),
                execution=execution)


def checked_native_result(output, before):
    """Bind the native driver's actual cases, invocations and build to this run."""
    directory = output/'native'
    path, executable = directory/'verification.json', directory/'bermudan_tree'
    if any(item.is_symlink() or not item.is_file() for item in (path, executable)):
        raise ValueError('The native verification ledger or executable is missing or is a symlink')
    native = json.loads(path.read_text(encoding='utf-8'))
    cases, checks = native.get('cases'), native.get('checks')
    if (native.get('status') != 'passed' or native.get('author') != 'Amanjeet Singh'
            or type(checks) is not int or checks <= 0
            or type(native.get('passed_checks')) is not int or native['passed_checks'] != checks
            or not isinstance(cases, list) or len(cases) != checks
            or any(not isinstance(case, dict) or case.get('passed') is not True for case in cases)
            or native.get('failures') != []):
        raise ValueError('The requested native verification did not pass every recorded check')
    names = [case.get('name') for case in cases]
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('The native verification case identities are missing or duplicated')
    expected_sources = {str(ROOT/name): before['file_sha256'][name] for name in NATIVE_SOURCES}
    if (native.get('source_sha256') != expected_sources
            or native.get('executable_sha256') != digest(executable)):
        raise ValueError('The native verification source or executable identities differ')
    compiler = native.get('compiler')
    if (not isinstance(compiler, str) or not compiler
            or native.get('command') != [compiler, *NATIVE_FLAGS,
                                        str(ROOT/'cpp/bermudan_tree.cpp'), '-o', str(executable)]
            or native.get('flags') != NATIVE_FLAGS
            or type(native.get('compilation_returncode')) is not int
            or native['compilation_returncode'] != 0
            or type(native.get('compiler_version_returncode')) is not int
            or native['compiler_version_returncode'] != 0
            or not isinstance(native.get('compiler_version'), str) or not native['compiler_version']):
        raise ValueError('The native verification lacks a successful declared build')
    invocations = native.get('native_executions')
    if not isinstance(invocations, list) or not invocations:
        raise ValueError('The native verification lacks invocation records')
    keys = ('arguments', 'returncode', 'stdout', 'stderr', 'result')
    def identity(item):
        if not isinstance(item, dict) or any(key not in item for key in keys):
            raise ValueError('The native invocation record is incomplete')
        return json.dumps({key: item[key] for key in keys}, sort_keys=True, allow_nan=False)
    for item in invocations:
        identity(item)
        if (not isinstance(item['arguments'], list)
                or not all(isinstance(argument, str) for argument in item['arguments'])
                or type(item['returncode']) is not int
                or not isinstance(item['stdout'], str) or not isinstance(item['stderr'], str)
                or not isinstance(item['result'], dict)
                or json.loads(item['stdout']) != item['result']
                or item['result'].get('author') != 'Amanjeet Singh'
                or (item['returncode'], item['result'].get('ok')) not in ((0, True), (2, False))
                or type(item['result'].get('ok')) is not bool):
            raise ValueError('The native invocation outcome is inconsistent')
    def case_invocations(value):
        if isinstance(value, dict):
            if all(key in value for key in keys):
                yield identity(value)
            else:
                for child in value.values():
                    yield from case_invocations(child)
        elif isinstance(value, list):
            for child in value:
                yield from case_invocations(child)
    recorded = sorted(identity(item) for item in invocations)
    if sorted(case_invocations(cases)) != recorded:
        raise ValueError('The native cases and invocation ledger differ')
    return dict(checks=checks, native_executions=len(invocations),
                executable_sha256=native['executable_sha256'], ledger_sha256=digest(path))


def artifact_inventory(output):
    records = {}
    for path in sorted(output.rglob('*')):
        if path.is_symlink():
            raise ValueError('A reproduction artifact is a symlink')
        if path.is_file() and path != output/'verification.json':
            records[path.relative_to(output).as_posix()] = digest(path)
    return records


def reproduce(output, *, with_native=False, compiler='clang++'):
    """Retain each requested outcome; complete means every requested step passed."""
    output = private_output(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema_version=1, author='Amanjeet Singh', complete=False,
        scope='Current public aggregate checks, selected numerical tests and fresh pricing notebook execution',
        research_experiments_replayed=False, historical_licensed_replays_executed=False,
        provider_inputs_read_or_exported=False, published_cpu_studies_retimed=False,
        network_data_requested=False, native_verification_requested=with_native,
        compiler=compiler if with_native else None, started_utc=utc_now(),
        python=platform.python_version(), platform=platform.platform(),
        dependencies={}, steps=[])
    for name in ('numpy', 'scipy', 'matplotlib', 'pytest', 'nbformat', 'nbclient', 'ipykernel'):
        try:
            record['dependencies'][name] = version(name)
        except PackageNotFoundError:
            record['dependencies'][name] = None
    before = None
    try:
        verifier = source_checker()
        required = list(CORE_SOURCES)
        if with_native:
            required += ['scripts/verify_bermudan_tree.py', 'cpp/bermudan_tree.cpp']
        before = inventory(verifier, required_paths=required)
        record['source_manifest_sha256'] = before['source_manifest_sha256']
        record['verified_source_files'] = before['verified_source_files']
        record['source_sha256_before'] = before['file_sha256']
        step('current_aggregates', ['paper/verification/revision_results.py'], output, record['steps'])
        step('current_tests', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *CURRENT_TESTS],
             output, record['steps'])
        step('current_notebook', ['scripts/execute_release_notebooks.py', '--foundations',
             '--output-dir', str(output/'notebooks')], output, record['steps'])
        record['notebook'] = checked_notebook_copy(output, before)
        if with_native:
            step('native_pricer', ['scripts/verify_bermudan_tree.py', '--output',
                 str(output/'native'), '--compiler', compiler], output, record['steps'])
            native = checked_native_result(output, before)
            record['native_checks'] = native['checks']
            record['native_verification'] = native
        after = inventory(verifier, required_paths=required)
        record['source_sha256_after'] = after['file_sha256']
        if after != before:
            raise ValueError('A listed source or the current inventory changed during reproduction')
        record['artifact_sha256'] = artifact_inventory(output)
        record['complete'] = True
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if before is not None and 'source_sha256_after' not in record:
            try:
                after = inventory(verifier, required_paths=required)
                record['source_sha256_after'] = after['file_sha256']
                if after != before:
                    raise ValueError('A listed source or inventory changed during the failed run')
            except (OSError, ValueError) as error:
                record['source_verification_after_failure'] = dict(
                    type=type(error).__name__, message=str(error))
        record['finished_utc'] = utc_now()
        artifact_failure = None
        try:
            record['artifact_sha256'] = artifact_inventory(output)
        except (OSError, ValueError) as error:
            record['complete'] = False
            artifact_failure = error
            record['artifact_inventory_failure'] = dict(type=type(error).__name__, message=str(error))
        (output/'verification.json').write_text(
            json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if artifact_failure is not None and sys.exc_info()[0] is None:
            raise ValueError('The reproduction artifact inventory failed') from artifact_failure
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh directory beneath data/private; existing runs are retained')
    parser.add_argument('--with-native', action='store_true',
                        help='Also build and independently verify the native C++17 pricer')
    parser.add_argument('--compiler', default='clang++',
                        help='Existing C++17 compiler used when --with-native is selected')
    args = parser.parse_args()
    reproduce(args.output, with_native=args.with_native, compiler=args.compiler)
    print('Completed current public-model reproduction; see verification.json in the selected directory.')


if __name__ == '__main__':
    main()
