"""Execute the historical notebooks or the current pricing computation from fresh kernels.

The selection is explicit. Local investigative notebooks are not silently added
to the release route. --write retains successful outputs only after the selected
notebooks complete. Outputs are mathematical fixtures or released summaries. The path check is an
execution guard, not a general publication or licensing review.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    'notebooks/numerical_validation.ipynb',
    'notebooks/conditional_research_replication.ipynb',
)

FOUNDATIONS = ('notebooks/pricing_and_stopping.ipynb',)


class InvokingInterpreter(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher',
                                '-f', '{connection_file}'],
                          display_name='Research notebook evaluation',
                          language='python', env={})


def checked_copy_directory(value):
    """Retain executed copies only in a fresh local private directory."""
    root = ROOT.resolve(strict=True)
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Notebook output must not contain parent traversal')
    path = path if path.is_absolute() else root/path
    if not path.is_relative_to(root/'data/private') or path == root/'data/private':
        raise ValueError('Notebook output must be a fresh directory beneath data/private')
    current = root
    for part in path.relative_to(root).parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('Notebook output must not contain symlink components')
        if current.exists() and not current.is_dir():
            raise ValueError('Notebook output parent must be a directory')
    if path.exists():
        raise FileExistsError('Existing notebook output is never replaced')
    return path


def execute(*, write=False, foundations=False, output_dir=None):
    if write and output_dir is not None:
        raise ValueError('Choose source updates or a separate executed copy, not both')
    destination = None if output_dir is None else checked_copy_directory(output_dir)
    completed = []
    for name in (FOUNDATIONS if foundations else NOTEBOOKS):
        path = ROOT/name
        source_bytes = path.read_bytes()
        source_identity = hashlib.sha256(source_bytes).hexdigest()
        notebook = nbformat.reads(source_bytes.decode('utf-8'), as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop('execution', None)
        started_at = datetime.now(timezone.utc)
        runtime_state = {}
        code_indices = [index for index, cell in enumerate(notebook.cells)
                        if cell.cell_type == 'code']
        if not code_indices:
            raise ValueError(f'No code cells in {name}')

        async def capture_runtime_state(cell, cell_index, execute_reply):
            if cell_index != code_indices[-1] or not foundations:
                return
            if execute_reply['content']['status'] != 'ok':
                return
            message_id = client.kc.execute(
                '', silent=True, store_history=False,
                user_expressions={'native_cpp': "globals().get('cpp_evaluated')"})
            reply = await client.async_wait_for_reply(message_id)
            if reply is None or reply['content']['status'] != 'ok':
                raise RuntimeError('The notebook execution state was not returned')
            value = reply['content']['user_expressions']['native_cpp']
            if value['status'] != 'ok':
                raise RuntimeError('The native pricing execution state is unavailable')
            native_cpp = ast.literal_eval(value['data']['text/plain'])
            if type(native_cpp) is not bool:
                raise ValueError('The native pricing execution state must be Boolean')
            runtime_state['native_cpp_executed'] = native_cpp

        manager = AsyncKernelManager(kernel_name='research-evaluation',
                                     kernel_spec_manager=InvokingInterpreter())
        client = NotebookClient(notebook, timeout=240, km=manager,
                                resources={'metadata': {'path': str(ROOT)}},
                                on_cell_executed=capture_runtime_state)
        environment = dict(os.environ)
        for variable in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
            environment.pop(variable, None)
        environment['PYTHONPATH'] = str(ROOT/'src')
        environment['PYTHONNOUSERSITE'] = '1'
        environment['MPLBACKEND'] = ('module://matplotlib_inline.backend_inline'
                                     if foundations else 'Agg')
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment[variable] = '1'
        client.execute(env=environment, cleanup_kc=True)
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
        completed_at = datetime.now(timezone.utc)
        execution = dict(
            date=completed_at.date().isoformat(),
            started_utc=started_at.isoformat(), completed_utc=completed_at.isoformat(),
            fresh_kernel=True,
            python_version=notebook.metadata['language_info']['version'],
            executed_code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
            figures=sum('image/png' in output.get('data', {})
                        for cell in notebook.cells for output in cell.get('outputs', [])),
            **runtime_state)
        notebook.metadata['execution'] = execution
        nbformat.validate(notebook)
        outputs = json.dumps([cell.get('outputs', []) for cell in notebook.cells])
        if any(marker in outputs for marker in ('/Users/', '/var/folders/', '/tmp/ipykernel_', 'data/private/',
                                                    'Traceback (most recent call last)')):
            raise ValueError(f'Unexpected local path or failed execution in {name}')
        completed.append((path, notebook, source_identity))
    for path, notebook, source_identity in completed:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_identity:
            raise ValueError(f'Notebook source changed during execution: {path.name}')
    if destination is not None:
        checked_copy_directory(destination)
        destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path, notebook, source_identity in completed:
        if write:
            nbformat.write(notebook, path)
        if destination is not None:
            nbformat.write(notebook, destination/path.name)
        records.append({'notebook': path.relative_to(ROOT).as_posix(),
                        'code_cells_executed': sum(c.cell_type == 'code' for c in notebook.cells),
                        'saved_outputs': write or destination is not None,
                        'source_updated': write,
                        'executed_copy': (None if destination is None else
                                          (destination/path.name).relative_to(ROOT).as_posix()),
                        'source_sha256': source_identity,
                        'execution': dict(notebook.metadata['execution']),
                        'executed_notebook_sha256': hashlib.sha256(
                            nbformat.writes(notebook).encode()).hexdigest(),
                        'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='Retain outputs after all selected notebooks execute successfully')
    parser.add_argument('--foundations', action='store_true',
                        help='Execute only the current pricing and stopping notebook')
    parser.add_argument('--output-dir', type=Path,
                        help='Retain executed copies in a fresh directory beneath data/private')
    args = parser.parse_args()
    print(json.dumps(execute(write=args.write, foundations=args.foundations,
                             output_dir=args.output_dir), indent=2))


if __name__ == '__main__':
    main()
