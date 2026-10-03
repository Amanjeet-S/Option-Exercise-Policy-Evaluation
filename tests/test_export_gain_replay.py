"""Replay-integrity checks using generator and record fixtures, not market data."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_gain_replay import CapturingGenerator, exact_record_match, selection_rule


def allocations():
    return [dict(case=i, arms={
        'reference': dict(allocation=dict(evaluation_paths=4096, repetitions=64)),
        'selected': dict(allocation=dict(evaluation_paths=128 if i == 12 else 256, repetitions=2)),
    }) for i in range(13)]


def test_selection_uses_count_and_first_repetition_only():
    rows = allocations()
    for row in rows:
        row['gain'] = 'This field must not enter the selection rule'
    selection = selection_rule(rows)
    assert (selection['case'], selection['arm'], selection['repetition']) == (12, 'selected', 0)


def test_ties_use_case_then_fixed_arm_order():
    rows = allocations()
    rows[0]['arms']['reference']['allocation']['evaluation_paths'] = 128
    rows[0]['arms']['selected']['allocation']['evaluation_paths'] = 128
    selection = selection_rule(rows)
    assert (selection['case'], selection['arm']) == (0, 'reference')


@pytest.mark.parametrize('rows', [[], allocations()[:-1], list(reversed(allocations()))])
def test_incomplete_selection_universe_rejected(rows):
    with pytest.raises(ValueError, match='thirteen'):
        selection_rule(rows)


def test_capture_preserves_normal_draws_and_generator_state():
    # This checks a deterministic pseudorandom stream, not a research sample.
    seed = np.random.SeedSequence(917, spawn_key=(4, 2), pool_size=4)
    captured = CapturingGenerator(np.random.PCG64(seed))
    reference = np.random.Generator(np.random.PCG64(seed))
    for count in (3, 8, 3):
        assert np.array_equal(captured.standard_normal(count), reference.standard_normal(count))
    assert captured.bit_generator.state == reference.bit_generator.state
    assert len(captured.calls) == 3
    captured.calls[0][0] = 0.
    assert captured.bit_generator.state == reference.bit_generator.state


def test_binary64_record_comparison_rejects_one_ulp_and_signed_zero():
    exact_record_match({'a': [1., -0.]}, {'a': [1., -0.]})
    with pytest.raises(ValueError, match='Binary64'):
        exact_record_match({'a': float(np.nextafter(1., 2.))}, {'a': 1.})
    with pytest.raises(ValueError, match='Binary64'):
        exact_record_match({'a': 0.}, {'a': -0.})


def test_replay_record_omission_is_rejected():
    with pytest.raises(ValueError, match='keys'):
        exact_record_match({'paths': 128}, {'paths': 128, 'mean': 0.})
