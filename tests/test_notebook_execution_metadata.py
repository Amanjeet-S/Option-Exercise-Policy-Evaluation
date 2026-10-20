"""Fresh execution records must describe the current run, including skipped C++."""
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import pytest

nbformat = pytest.importorskip('nbformat')
pytest.importorskip('nbclient')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'notebook_metadata_runner', ROOT/'scripts/execute_release_notebooks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'FOUNDATIONS', ('notebooks/current.ipynb',))
    return module


def fixture_notebook(runner, source):
    path = runner.ROOT/runner.FOUNDATIONS[0]
    path.parent.mkdir()
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(source, execution_count=57,
            outputs=[nbformat.v4.new_output('stream', name='stdout', text='Old output')])],
        metadata={'author':'Amanjeet Singh', 'date':'2001-01-01',
                  'source_sha256':{'declared_source':'retained identity'},
                  'simulation_design':{'seed':17,'paths':4},
                  'execution':{'date':'2001-01-01','native_cpp_executed':True,
                               'python_version':'0.0','executed_code_cells':99}})
    nbformat.write(notebook,path)
    return path


@pytest.mark.parametrize('native_cpp', [False, True])
@pytest.mark.parametrize('write', [False, True])
def test_record_uses_actual_kernel_outcome(runner, native_cpp, write):
    path=fixture_notebook(runner, f'cpp_evaluated = {native_cpp!r}\nprint("Current computation")')
    original=path.read_bytes()
    before=datetime.now(timezone.utc)
    records=runner.execute(foundations=True,write=write)
    after=datetime.now(timezone.utc)
    execution=records[0]['execution']
    assert execution['native_cpp_executed'] is native_cpp
    assert execution['executed_code_cells']==1
    assert execution['python_version']==sys.version.split()[0]
    assert before<=datetime.fromisoformat(execution['completed_utc'])<=after
    assert execution['date']==datetime.fromisoformat(execution['completed_utc']).date().isoformat()
    notebook=nbformat.read(path,as_version=4)
    if write:
        assert notebook.metadata.execution==execution
        assert notebook.cells[0].execution_count==1
        assert notebook.cells[0].outputs[0].text=='Current computation\n'
    else:
        assert path.read_bytes()==original
    assert notebook.metadata.author=='Amanjeet Singh'
    assert notebook.metadata.date=='2001-01-01'
    assert notebook.metadata.simulation_design=={'seed':17,'paths':4}
    assert notebook.metadata.source_sha256=={'declared_source':'retained identity'}


@pytest.mark.parametrize('source', ['print("No native execution state")',
                                    'cpp_evaluated = "unknown"'])
def test_unreported_or_invalid_native_state_does_not_overwrite(runner, source):
    path=fixture_notebook(runner,source)
    original=path.read_bytes()
    with pytest.raises(ValueError,match='must be Boolean'):
        runner.execute(foundations=True,write=True)
    assert path.read_bytes()==original


def test_failed_calculation_does_not_replace_saved_success(runner):
    path=fixture_notebook(runner,'cpp_evaluated=False\nraise ArithmeticError("Declared failed calculation")')
    original=path.read_bytes()
    with pytest.raises(Exception,match='Declared failed calculation'):
        runner.execute(foundations=True,write=True)
    assert path.read_bytes()==original


def test_concurrent_source_edit_is_preserved(runner):
    path=runner.ROOT/runner.FOUNDATIONS[0]
    source=('cpp_evaluated=False\n'
            'from pathlib import Path\n'
            f'p=Path({str(path)!r})\n'
            'p.write_text(p.read_text()+"\\n")\n')
    path=fixture_notebook(runner,source)
    original=path.read_bytes()
    with pytest.raises(ValueError,match='source changed during execution'):
        runner.execute(foundations=True,write=True)
    assert path.read_bytes()==original+b'\n'


def test_executed_copy_preserves_source_and_records_new_runtime(runner):
    path = fixture_notebook(runner, 'cpp_evaluated=False\nprint("Separate current output")')
    original = path.read_bytes()
    output = runner.ROOT/'data/private/executed'
    records = runner.execute(foundations=True, output_dir=output)
    copy = output/path.name
    assert path.read_bytes() == original
    retained = nbformat.read(copy, as_version=4)
    assert retained.cells[0].outputs[0].text == 'Separate current output\n'
    assert retained.metadata.execution.native_cpp_executed is False
    assert records[0]['saved_outputs'] is True
    assert records[0]['source_updated'] is False
    assert records[0]['executed_copy'] == 'data/private/executed/current.ipynb'


@pytest.mark.parametrize('selection', ['public', 'data/private/existing',
                                      'data/private/link/new', 'data/private/../outside'])
def test_invalid_copy_destination_is_rejected_before_kernel(runner, selection):
    fixture_notebook(runner, 'raise AssertionError("The kernel must not execute")')
    (runner.ROOT/'data/private/existing').mkdir(parents=True)
    (runner.ROOT/'data/private/link').symlink_to(runner.ROOT/'notebooks', target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        runner.execute(foundations=True, output_dir=runner.ROOT/selection)


def test_source_write_and_separate_copy_are_mutually_exclusive(runner):
    path = fixture_notebook(runner, 'raise AssertionError("The kernel must not execute")')
    original = path.read_bytes()
    with pytest.raises(ValueError, match='not both'):
        runner.execute(foundations=True, write=True, output_dir=runner.ROOT/'data/private/new')
    assert path.read_bytes() == original


def test_failed_kernel_does_not_create_successful_executed_copy(runner):
    path = fixture_notebook(runner, 'cpp_evaluated=False\nraise ArithmeticError("Declared copy failure")')
    original = path.read_bytes()
    output = runner.ROOT/'data/private/executed'
    with pytest.raises(Exception, match='Declared copy failure'):
        runner.execute(foundations=True, output_dir=output)
    assert path.read_bytes() == original
    assert not output.exists()


def test_concurrent_source_edit_prevents_executed_copy_retention(runner):
    path = runner.ROOT/runner.FOUNDATIONS[0]
    source = ('cpp_evaluated=False\nfrom pathlib import Path\n'
              f'p=Path({str(path)!r})\n'
              'p.write_text(p.read_text()+"\\n")\n')
    path = fixture_notebook(runner, source)
    original = path.read_bytes()
    output = runner.ROOT/'data/private/concurrent'
    with pytest.raises(ValueError, match='source changed during execution'):
        runner.execute(foundations=True, output_dir=output)
    assert path.read_bytes() == original+b'\n'
    assert not output.exists()
