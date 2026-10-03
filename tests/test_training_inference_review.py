"""Deterministic mathematical fixtures for repetition influence diagnostics."""

import math
from pathlib import Path
import sys

import pytest
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from review_training_inference import delete_one, summary


def records(values):
    return [dict(repetition=i, value=v, standard_error=.1) for i, v in enumerate(values)]


def test_student_interval_matches_separate_scipy_reference():
    values = [-.5, .2, .4, .7, 1.1, 1.8, 3.2, 4.]
    result = summary(records(values))
    reference = ttest_1samp(values, 0.).confidence_interval(.95)
    assert result['lower'] == pytest.approx(reference.low)
    assert result['upper'] == pytest.approx(reference.high)


def test_every_deletion_is_retained_and_order_does_not_select_results():
    rows = records([1., 2., 3., 4.])
    result = delete_one(rows)
    assert delete_one(list(reversed(rows))) == result
    assert [r['omitted_repetition'] for r in result['deletions']] == [0, 1, 2, 3]
    assert [r['mean'] for r in result['deletions']] == [3., 8/3, 7/3, 2.]
    assert result['maximum_absolute_mean_change'] == .5


def test_zero_between_repetition_variance_does_not_create_a_variance_share():
    result = delete_one(records([1., 1., 1., 1.]))
    for row in [result['primary'], *result['deletions']]:
        assert row['evaluation_share_percent'] is None
        assert row['training_variance_moment_estimate'] == pytest.approx(-.01)


def test_large_conditional_noise_is_not_truncated():
    result = summary(records([1., 1.01, .99]))
    assert result['evaluation_share_percent'] > 100
    assert result['training_variance_moment_estimate'] < 0


@pytest.mark.parametrize('bad', [math.nan, math.inf, -1.])
def test_invalid_conditional_error_is_rejected(bad):
    rows = records([1., 2., 3.]); rows[0]['standard_error'] = bad
    with pytest.raises(ValueError):
        delete_one(rows)


def test_duplicate_identifiers_and_insufficient_repetitions_are_rejected():
    rows = records([1., 2., 3.]); rows[1]['repetition'] = 0
    with pytest.raises(ValueError):
        delete_one(rows)
    with pytest.raises(ValueError):
        delete_one(records([1., 2.]))


@pytest.mark.parametrize('confidence', [0., 1., math.nan])
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        delete_one(records([1., 2., 3.]), confidence)
