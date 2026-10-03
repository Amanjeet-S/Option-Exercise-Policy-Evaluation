"""Input classification, common-upper identities and fresh fixture repetition."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'replication/run.py'
spec = importlib.util.spec_from_file_location('replication_runner', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    return runner.load_specification(runner.FIXTURE)


def test_fixture_numerics_are_reproducible_and_common_upper_cancels():
    first, second = runner.run(fixture()), runner.run(fixture())
    assert first['numerics'] == second['numerics']
    assert first['numerical_sha256'] == second['numerical_sha256']
    provenance = first['provenance']
    assert provenance['input_specification_sha256'] == hashlib.sha256(
        json.dumps(fixture(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = ROOT/'replication/source_manifest.json'
    assert provenance['source_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert provenance['verified_source_files'] == len(json.loads(manifest.read_text())['files'])
    assert provenance['project_module_origins']['qf_research'] == 'src/qf_research/__init__.py'
    assert len(first['numerics']['pair_results']) == 8
    assert first['numerics']['independent_stream_identifiers'] == 17
    terminal = first['numerics']['terminal_adjustment']
    assert terminal['minimum_integrated_gain'] >= 0
    assert terminal['maximum_common_upper_identity_error'] < 1e-12
    assert first['costs']['selected_counts_executed'] is False
    for row in first['numerics']['pair_results']:
        assert min(row['minimum_gap_quote']) >= 0
    for row in first['numerics']['training_evaluation']:
        assert row['training_variance_moment_estimate'] == pytest.approx(
            row['repetition_estimate_variance']-row['mean_conditional_evaluation_variance'])


def test_changed_observations_cannot_use_the_public_fixture_label():
    changed = fixture()
    changed['model']['quoted_spot'] += 1
    with pytest.raises(ValueError, match='fixture label'):
        runner.validate(changed)
    changed['classification'] = 'authorised_model'
    model, _, _ = runner.validate(changed)
    assert model.quoted_spot == 101


@pytest.mark.parametrize('alter', [
    lambda s: s.update(schema_version=True),
    lambda s: s['option'].update(strike=True),
    lambda s: s['option'].update(maturity=True),
    lambda s: s['option'].update(strike=0),
    lambda s: s['decisions'][0].pop('phase'),
    lambda s: s['decisions'][0].update(extra=1),
    lambda s: s['experiment'].update(terminal_extension=0),
    lambda s: s['experiment'].update(root_seed=True),
    lambda s: s['experiment'].update(training_paths=[512, 512]),
    lambda s: s['experiment'].update(evaluation_paths=[1024, 256]),
    lambda s: s['experiment'].update(repetitions=1),
    lambda s: s['experiment'].update(terminal_extension=-.1),
    lambda s: s['experiment']['approximation'].update(tolerance=float('nan')),
    lambda s: s['decisions'].reverse(),
    lambda s: s.update(unreviewed_quotes=[99, 101]),
])
def test_invalid_model_contract_rejected_before_execution(alter):
    changed = fixture()
    changed['classification'] = 'authorised_model'
    alter(changed)
    with pytest.raises((ValueError, TypeError)):
        runner.validate(changed)


def test_non_finite_json_cannot_enter_input_contract(tmp_path):
    path = tmp_path/'invalid.json'
    path.write_text('{"x":NaN}')
    with pytest.raises(ValueError, match='non-finite'):
        runner.read_json(path)


def test_duplicate_input_keys_cannot_silently_select_a_specification(tmp_path):
    path = tmp_path/'ambiguous.json'
    path.write_text('{"experiment":{"root_seed":1,"root_seed":2}}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        runner.read_json(path)


def test_cli_rejects_unreviewed_public_output(tmp_path):
    output = tmp_path/'results.json'
    completed = subprocess.run([sys.executable, str(PATH), '--validate-only',
                                '--output', str(output)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'under data/private' in completed.stderr
    assert not output.exists()


def copy_checkout(tmp_path):
    """Copy only public manifest entries, without observations or prior outputs."""
    checkout = tmp_path/'checkout'
    manifest = Path('replication/source_manifest.json')
    entries = json.loads((ROOT/manifest).read_text())['files']
    for relative in [manifest, *(Path(row['path']) for row in entries)]:
        destination = checkout/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, destination)
    return checkout


def copied_command(checkout, code):
    environment = dict(os.environ, PYTHONPATH=str(checkout/'src'), PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-c', code], cwd=checkout, env=environment,
                          capture_output=True, text=True)


def test_validate_only_records_verified_sources_and_input_without_simulation(tmp_path):
    checkout = copy_checkout(tmp_path)
    output = checkout/'data/private/validated.json'
    completed = copied_command(checkout, """
import importlib.util, sys
from pathlib import Path
p = Path('replication/run.py')
s = importlib.util.spec_from_file_location('isolated_runner', p)
r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
def expensive(_):
    raise AssertionError('Validation must not run numerical experiments')
r._run_numerics = expensive
sys.argv = [str(p), '--validate-only', '--output', 'data/private/validated.json']
r.main()
""")
    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text())
    assert record['valid'] is True
    assert 'numerics' not in record
    assert record['provenance']['source_manifest_sha256'] == hashlib.sha256(
        (checkout/'replication/source_manifest.json').read_bytes()).hexdigest()
    assert record['provenance']['input_specification_sha256'] == runner.canonical_digest(fixture())


@pytest.mark.parametrize('damage', ['before', 'after', 'late_import'])
def test_failed_execution_provenance_leaves_no_output(tmp_path, damage):
    checkout = copy_checkout(tmp_path)
    completed = copied_command(checkout, """
import importlib.util, sys, types
from pathlib import Path
p = Path('replication/run.py')
s = importlib.util.spec_from_file_location('isolated_runner', p)
r = importlib.util.module_from_spec(s); s.loader.exec_module(r)
damage = DAMAGE
def changed(_):
    if damage == 'late_import':
        foreign = types.ModuleType('qf_research.late')
        foreign.__file__ = str(r.ROOT.parent/'installed/qf_research/late.py')
        foreign.__spec__ = importlib.util.spec_from_file_location(foreign.__name__, foreign.__file__)
        sys.modules[foreign.__name__] = foreign
    else:
        r.FIXTURE.write_bytes(r.FIXTURE.read_bytes()+b'\\n')
    return {}
r._run_numerics = changed
if damage == 'before':
    r.FIXTURE.write_bytes(r.FIXTURE.read_bytes()+b'\\n')
sys.argv = [str(p), '--output', 'data/private/failed.json']
r.main()
""".replace('DAMAGE', repr(damage)))
    assert completed.returncode != 0
    expected = 'outside the verified checkout' if damage == 'late_import' else 'Source byte count differs'
    assert expected in completed.stderr
    assert not (checkout/'data/private/failed.json').exists()
