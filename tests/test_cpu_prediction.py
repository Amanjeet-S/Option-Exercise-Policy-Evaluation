"""Exact accounting fixtures, not observations or financial simulation data."""

from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from study_cpu_prediction import (
    EXPECTED, aggregate, check_rows, costs, forecast, fraction, schedule,
    seed_sequence, summarise,
)
from qf_research.optimal_stopping.funded_policy import _digest

ITEM = dict(pass_id=0, position=0, case=5, evaluation_paths=128, repetitions=64, budget='700')


def rows(phase, count, charge=8., item=None):
    item = ITEM if item is None else item
    return [dict(pass_id=item['pass_id'], case=item['case'], evaluation_paths=item['evaluation_paths'],
        phase=phase, repetition=rep, training=dict(cpu_seconds=charge/2, elapsed_seconds=charge),
        evaluation=dict(cpu_seconds=charge/2, elapsed_seconds=charge)) for rep in range(count)]


def setup(cpu=8.):
    return dict(cpu_seconds=cpu, elapsed_seconds=2*cpu)


def predicted(item=None):
    item = ITEM if item is None else item
    return forecast(item, rows('local_calibration', 16, item=item), setup(), 'local_calibration')


def test_schedule_is_a_complete_counterbalanced_pair_of_orders():
    result = schedule()
    assert result == schedule()
    first, second = result[:9], result[9:]
    assert [(r['case'], r['evaluation_paths']) for r in first] == list(reversed([
        (r['case'], r['evaluation_paths']) for r in second]))
    assert {(r['case'], r['evaluation_paths']) for r in first} == {
        (case, n) for case in (5, 6, 10) for n in (128, 256, 4096)}
    assert [r['position'] for r in first] == list(range(9))
    assert [r['position'] for r in second] == list(range(9))


def test_stream_namespaces_separate_all_phases_passes_cases_counts_and_roles():
    identifiers = set()
    keys = set()
    for item in schedule():
        for phase in ('warmup', 'pass_calibration', 'local_calibration', 'validation'):
            for rep in range(16):
                for role in range(3):
                    s = seed_sequence(phase, item['pass_id'], item['case'], item['evaluation_paths'], rep, role)
                    identity = _digest(np.random.PCG64(s).state)
                    assert s.spawn_key not in keys and identity not in identifiers
                    keys.add(s.spawn_key)
                    identifiers.add(identity)
    assert len(identifiers) == 18*4*16*3


def test_prediction_extrapolates_the_mean_and_excludes_calibration_from_target():
    pred = predicted()
    assert pred['predictions'] == dict(cpu=Fraction(520), elapsed=Fraction(1040))
    assert pred['allowance_predictions']['cpu'] == 572
    assert pred['calibration_charge']['cpu'] == 136
    assert not pred['includes_calibration_cost_in_validation_prediction']
    assert not pred['prediction_coverage_claim']


def test_validation_charges_calibration_and_reports_budget_decisions():
    result = summarise(ITEM, rows('validation', 64), setup(), dict(immediate=predicted()),
                       dict(cpu=272, elapsed=544))
    comparison = result['comparisons']['immediate']['cpu']
    assert comparison['ratio'] == 1
    assert comparison['forecast_affordable'] and comparison['actual_affordable']
    assert result['charged_numerical_cost']['cpu'] == 792
    assert not result['charged_cpu_affordable']
    assert result['first_quarter_pair_cpu'] == result['last_quarter_pair_cpu'] == 8


@pytest.mark.parametrize('factor, expected', [(Fraction(9, 10), True), (Fraction(11, 10), True),
    (Fraction(8999, 10000), False), (Fraction(11001, 10000), False)])
def test_forecast_tolerance_uses_exact_inclusive_boundaries(factor, expected):
    pred = predicted()
    # Stipulated rational timers make both boundaries exact.
    result = summarise(ITEM, rows('validation', 64, charge=8*factor),
                       setup(8*factor), dict(immediate=pred), dict(cpu=0, elapsed=0))
    assert result['comparisons']['immediate']['cpu']['ratio'] == factor
    assert result['comparisons']['immediate']['cpu']['within_tolerance'] == expected


def test_false_affordability_and_allowance_are_distinct():
    item = dict(ITEM, budget='600')
    pred = predicted(item)
    result = summarise(item, rows('validation', 64, charge=10, item=item), setup(10),
                       dict(immediate=pred), dict(cpu=0, elapsed=0))
    comp = result['comparisons']['immediate']['cpu']
    assert comp['falsely_forecast_affordable'] and comp['falsely_allowance_affordable']
    assert not comp['within_allowance']
    assert comp['actual'] == 650


@pytest.mark.parametrize('phase', ['local_calibration', 'validation'])
@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reordered', 'case', 'pass_id', 'evaluation_paths', 'phase'])
def test_incomplete_or_misidentified_blocks_rejected(phase, mutation):
    count = 16 if phase == 'local_calibration' else 64
    sample = rows(phase, count)
    if mutation == 'missing':
        sample.pop()
    elif mutation == 'duplicate':
        sample[-1]['repetition'] = 0
    elif mutation == 'reordered':
        sample.reverse()
    else:
        sample[0][mutation] = 'wrong'
    with pytest.raises(ValueError):
        check_rows(ITEM, sample, phase, count)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1., '-1/2'])
def test_invalid_timer_rejected(invalid):
    with pytest.raises(ValueError):
        fraction(invalid)


def test_timer_addition_preserves_both_stored_binary64_numbers():
    row = rows('validation', 1)[0]
    row['training']['cpu_seconds'] = .1
    row['evaluation']['cpu_seconds'] = .2
    assert costs(row) == Fraction(.1)+Fraction(.2)
    assert costs(row) != Fraction(.1+.2)


@pytest.mark.parametrize('field', ['pass_id', 'case', 'evaluation_paths', 'repetitions'])
def test_forecast_identity_must_match(field):
    pred = predicted()
    pred[field] += 1
    with pytest.raises(ValueError, match='identity'):
        summarise(ITEM, rows('validation', 64), setup(), dict(immediate=pred), dict(cpu=0, elapsed=0))


def test_incomplete_study_cannot_supply_aggregate():
    with pytest.raises(ValueError, match='eighteen'):
        aggregate([])


def test_zero_cost_pair_rejected():
    with pytest.raises(ValueError, match='positive'):
        check_rows(ITEM, rows('validation', 64, charge=0.), 'validation', 64)


def test_warmup_cannot_supply_prediction():
    with pytest.raises(ValueError, match='calibration'):
        forecast(ITEM, rows('warmup', 16), setup(), 'warmup')
