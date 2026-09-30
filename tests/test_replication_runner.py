"""Input classification, common-upper identities and fresh fixture repetition."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'projects/01_optimal_stopping/replication/run.py'
spec = importlib.util.spec_from_file_location('replication_runner', PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    return runner.load_specification(runner.FIXTURE)


def test_fixture_numerics_are_reproducible_and_common_upper_cancels():
    first, second = runner.run(fixture()), runner.run(fixture())
    assert first['numerics'] == second['numerics']
    assert first['numerical_sha256'] == second['numerical_sha256']
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


def test_cli_rejects_unreviewed_public_output(tmp_path):
    output = tmp_path/'results.json'
    completed = subprocess.run([sys.executable, str(PATH), '--validate-only',
                                '--output', str(output)], capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'under data/private' in completed.stderr
    assert not output.exists()
