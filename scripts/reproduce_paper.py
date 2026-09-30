"""Reproduce the public checks and mathematical example for the stopping paper.

Only released summaries, retained CPU measurements and stipulated mathematical
inputs are used. Financial simulation tables require the separate licensed-input
archive and are not regenerated here. No restricted raw financial data are read
or exported; no data are downloaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'projects/01_optimal_stopping'
PAPER = PROJECT + '/paper'
VERIFICATION = PAPER + '/verification'
NOTEBOOKS = (
    PROJECT + '/notebooks/01_numerical_validation.ipynb',
    PROJECT + '/notebooks/27_conditional_research_replication.ipynb',
)
ANALYTICAL_TESTS = ('test_allocation_probability', 'test_conditional_drift',
                    'test_cpu_timings', 'test_numerical_results', 'test_paper_results')
PUBLIC_INPUTS = (*NOTEBOOKS,
    *(VERIFICATION + '/' + name + '.py' for name in ANALYTICAL_TESTS),
    *(VERIFICATION + '/' + name + '.py' for name in
      ('numerical_results', 'paper_results', 'cpu_timings', 'allocation_probability', 'conditional_drift')),
    PAPER + '/main.tex', PAPER + '/numerical_results.json', PAPER + '/numerical_results.md',
    PAPER + '/cpu_timings.json', PAPER + '/cpu_timings.md',
    PROJECT + '/replication/configs/stipulated_fixture.json',
    PROJECT + '/configs/benchmark.json', PROJECT + '/results/baseline/manifest.json',
    PROJECT + '/results/baseline/grid_convergence.csv')
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
NOTEBOOK_WORKER = """
import json
from pathlib import Path
import runpy
import sys
module = runpy.run_path(sys.argv[1])
print(json.dumps(module['execute_notebooks'](Path(sys.argv[2])), indent=2))
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_checker():
    path = ROOT/PROJECT/'replication/verify_sources.py'
    spec = importlib.util.spec_from_file_location('public_reproduction_sources', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_environment():
    environment = dict(os.environ)
    for name in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
        environment.pop(name, None)
    environment.update(PYTHONPATH=str(ROOT/'src'), PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    environment.update({name:'1' for name in THREAD_VARIABLES})
    return environment


def private_output(path):
    path = Path(path)
    path = (path if path.is_absolute() else ROOT/path).resolve()
    if not path.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Output must be a fresh directory under data/private')
    if path.exists():
        raise FileExistsError('Use a fresh output directory; existing results are never merged')
    return path


def execute_notebooks(output):
    """Execute only the two named public notebooks and retain separate copies."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

    class InvokingInterpreter(KernelSpecManager):
        def get_kernel_spec(self, kernel_name):
            return KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher',
                                    '-f', '{connection_file}'],
                              display_name='Public research reproduction', language='python', env={})

    output = private_output(output)
    output.mkdir(parents=True)
    records = []
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT/name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop('execution', None)
        nbformat.validate(notebook)
        manager = AsyncKernelManager(kernel_name='public-reproduction',
                                     kernel_spec_manager=InvokingInterpreter())
        client = NotebookClient(notebook, km=manager, timeout=240,
                                resources={'metadata': {'path': str(ROOT)}})
        client.execute(env=child_environment(), cleanup_kc=True)
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
        nbformat.validate(notebook)
        outputs = json.dumps([cell.get('outputs', []) for cell in notebook.cells])
        if any(marker in outputs for marker in
               ('/Users/', '/home/', '/var/folders/', '/tmp/ipykernel_', 'data/private/',
                'Traceback (most recent call last)')):
            raise ValueError('Unexpected local path or failed execution in ' + name)
        destination = output/Path(name).name
        nbformat.write(notebook, destination)
        records.append(dict(source=name, output=destination.name,
            code_cells_executed=sum(c.cell_type == 'code' for c in notebook.cells),
            sha256=digest(destination), kernel_interpreter='invoking Python interpreter'))
    return records


def step(name, arguments, output, records, *, cwd=None):
    log = output/(name + '.log')
    print('Running ' + name, flush=True)
    with log.open('x') as stream:
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT if cwd is None else cwd,
            env=child_environment(), stdout=stream, stderr=subprocess.STDOUT, text=True)
    records.append(dict(name=name, returncode=result.returncode, log=log.name))
    if result.returncode:
        raise RuntimeError(name + ' failed; inspect ' + log.name)


def reproduce(output, *, full_tests=False):
    """Create a fresh, source-bound record without opening private inputs."""
    output = private_output(output)
    verifier = source_checker()
    before = verifier.verify_sources(ROOT, required_paths=('scripts/reproduce_paper.py',))
    inputs = {name:digest(verifier.checked_file(ROOT.resolve(), name)) for name in PUBLIC_INPUTS}
    output.mkdir(parents=True)
    record = dict(schema_version=1, complete=False,
        scope='Public summary checks, retained CPU arithmetic and stipulated mathematical simulations only',
        financial_simulation_tables_replayed=False, raw_market_data_read_or_exported=False,
        published_cpu_workloads_retimed=False, fixture_classification='stipulated_mathematical_fixture',
        exact_financial_replay='Requires the separate authorised original input and evidence archive',
        started_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        source_manifest_sha256=before['source_manifest_sha256'],
        verified_source_files=before['verified_source_files'], public_input_sha256=inputs,
        full_tests_requested=full_tests, steps=[])
    try:
        for name in ('numerical_results', 'paper_results', 'cpu_timings',
                     'allocation_probability', 'conditional_drift'):
            step(name, [VERIFICATION + '/' + name + '.py'], output, record['steps'])
        step('analytical_tests', ['-m', 'unittest', '-v', *ANALYTICAL_TESTS],
             output, record['steps'], cwd=ROOT/VERIFICATION)
        step('mathematical_fixture', [PROJECT + '/replication/run.py',
             '--output', str(output/'fixture.json')], output, record['steps'])
        fixture = json.loads((output/'fixture.json').read_text())
        if (fixture['classification'] != 'stipulated_mathematical_fixture'
                or fixture['provenance']['source_manifest_sha256'] != before['source_manifest_sha256']):
            raise ValueError('Fixture classification or source provenance differs')
        record['fixture_numerical_sha256'] = fixture['numerical_sha256']
        record['fixture_input_sha256'] = fixture['provenance']['input_specification_sha256']
        step('notebooks', ['-c', NOTEBOOK_WORKER, str(Path(__file__).resolve()),
             str(output/'notebooks')], output, record['steps'])
        if full_tests:
            tests = sorted(name for name in before['file_sha256']
                           if name.startswith('tests/test_') and name.endswith('.py'))
            if not tests:
                raise ValueError('No published test files in the verified source inventory')
            step('published_tests', ['-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests],
                 output, record['steps'])
        after = verifier.verify_sources(ROOT)
        if after != before or any(digest(verifier.checked_file(ROOT.resolve(), name)) != value
                                  for name,value in inputs.items()):
            raise ValueError('A verified source or public input changed during reproduction')
        record['complete'] = True
    except BaseException as error:
        record['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        record['artifact_sha256'] = {path.relative_to(output).as_posix():digest(path)
            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'verification.json'}
        (output/'verification.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/private/reproduction/paper_run'),
                        help='Fresh directory beneath data/private; existing results are never replaced')
    parser.add_argument('--full-tests', action='store_true',
                        help='Also run the published source-inventory tests; optional tool checks may skip')
    args = parser.parse_args()
    reproduce(args.output, full_tests=args.full_tests)
    print('Completed public reproduction; see verification.json in the selected output directory.')


if __name__ == '__main__':
    main()
