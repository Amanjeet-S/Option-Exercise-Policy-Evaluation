"""Execute the two released optimal-stopping notebooks from fresh kernels.

The selection is explicit. Local investigative notebooks are not silently added
to the release route. --write retains successful outputs only after both selected
notebooks complete. Outputs are mathematical fixtures or released summaries. The path check is an
execution guard, not a general publication or licensing review.
"""

import argparse
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

class InvokingInterpreter(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        return KernelSpec(argv=[sys.executable, '-m', 'ipykernel_launcher',
                                '-f', '{connection_file}'],
                          display_name='Research notebook evaluation',
                          language='python', env={})


def execute(*, write=False):
    completed = []
    for name in NOTEBOOKS:
        path = ROOT/name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop('execution', None)
        manager = AsyncKernelManager(kernel_name='research-evaluation',
                                     kernel_spec_manager=InvokingInterpreter())
        client = NotebookClient(notebook, timeout=240, km=manager,
                                resources={'metadata': {'path': str(ROOT)}})
        environment = dict(os.environ)
        for variable in ('PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONOPTIMIZE'):
            environment.pop(variable, None)
        environment['PYTHONPATH'] = str(ROOT/'src')
        environment['PYTHONNOUSERSITE'] = '1'
        environment['MPLBACKEND'] = 'Agg'
        for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment[variable] = '1'
        client.execute(env=environment, cleanup_kc=True)
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
        nbformat.validate(notebook)
        outputs = json.dumps([cell.get('outputs', []) for cell in notebook.cells])
        if any(marker in outputs for marker in ('/Users/', '/var/folders/', '/tmp/ipykernel_', 'data/private/',
                                                    'Traceback (most recent call last)')):
            raise ValueError(f'Unexpected local path or failed execution in {name}')
        completed.append((path, notebook))
    records = []
    for path, notebook in completed:
        if write:
            nbformat.write(notebook, path)
        records.append({'notebook': path.relative_to(ROOT).as_posix(),
                        'code_cells_executed': sum(c.cell_type == 'code' for c in notebook.cells),
                        'saved_outputs': write,
                        'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='Retain outputs after both notebooks execute successfully')
    args = parser.parse_args()
    print(json.dumps(execute(write=args.write), indent=2))


if __name__ == '__main__':
    main()
