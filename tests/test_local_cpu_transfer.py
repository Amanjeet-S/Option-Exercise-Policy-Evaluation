"""Analytical timer fixtures for local CPU accounting, not financial data.

No fitting, path simulation, numerical workload or wall-clock measurement is
performed. Generator states are inspected only to verify stream namespaces.
"""

from copy import deepcopy
from fractions import Fraction
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from study_local_cpu_transfer import (
    EXPECTED, forecast, phase_costs, reserve_streams, schedule, stream, summarise_cell,
)


ITEM = dict(position=0, case=0, evaluation_paths=128)


def timer_row(phase, repetition, cost=9., *, item=None):
    """A stipulated additive charge with exactly representable phase shares."""
    item = ITEM if item is None else item
    training = cost / 2
    return dict(**item, phase=phase, repetition=repetition,
        training=dict(cpu_seconds=training, phases=dict(
            simulation_cpu_seconds=training / 4, fitting_cpu_seconds=training / 2)),
        evaluation=dict(cpu_seconds=cost / 2), pair_cpu_seconds=cost)


def pilot(cost=9.):
    return [timer_row('pilot', rep, cost) for rep in range(8)]


def validation(cost=9.):
    return [timer_row('validation', rep, cost) for rep in range(64)]


def prediction():
    return forecast(ITEM, pilot(), dict(cpu_seconds=64.))


def test_schedule_covers_each_case_and_count_once_in_fixed_order():
    rows = schedule()
    assert rows == schedule()
    assert [r['position'] for r in rows] == list(range(39))
    actual = [(r['case'], r['evaluation_paths']) for r in rows]
    assert len(actual) == len(set(actual)) == 39
    assert set(actual) == {(case, count) for case in range(13) for count in (128, 256, 4096)}
    for case in range(13):
        assert sorted(r['evaluation_paths'] for r in rows if r['case'] == case) == [128, 256, 4096]


def test_all_registered_stream_roles_are_distinct_and_reserved():
    seen = set()
    keys = set()
    for repetition in range(4):
        seeds = tuple(stream('warmup', 0, repetition, 0, 4096, role) for role in range(3))
        keys.update(seed.spawn_key for seed in seeds)
        reserve_streams(seeds, seen)
    for cell in schedule():
        for phase, repetitions in (('pilot', 8), ('validation', 64)):
            for repetition in range(repetitions):
                seeds = tuple(stream(phase, cell['position'], repetition, cell['case'],
                                     cell['evaluation_paths'], role) for role in range(3))
                assert all(seed.entropy == EXPECTED['root_seed'] for seed in seeds)
                assert all(seed.spawn_key not in keys for seed in seeds)
                keys.update(seed.spawn_key for seed in seeds)
                reserve_streams(seeds, seen)
    assert len(keys) == len(seen) == 3 * (4 + 39 * (8 + 64))


def test_reservation_rejects_previously_used_stream_without_mutating_seen():
    seeds = tuple(stream('pilot', 0, 0, 0, 128, role) for role in range(3))
    seen = set()
    identifiers = reserve_streams(seeds, seen)
    assert set(identifiers) == seen
    snapshot = set(seen)
    with pytest.raises(ValueError, match='reused'):
        reserve_streams(seeds, seen)
    assert seen == snapshot


@pytest.mark.parametrize('roles', [(0, 0, 2), (0, 1), (0, 1, 2, 3)])
def test_reservation_rejects_duplicate_or_wrong_role_count(roles):
    seeds = tuple(stream('pilot', 0, 0, 0, 128, role) for role in roles)
    seen = set()
    with pytest.raises(ValueError, match='reused'):
        reserve_streams(seeds, seen)
    assert not seen


def test_forecast_uses_all_pilot_costs_and_keeps_pilot_charge_separate():
    rows = [timer_row('pilot', rep, float(rep + 1)) for rep in range(8)]
    result = forecast(ITEM, rows, dict(cpu_seconds=7.))
    # Eight pilot charges sum to 36; the 64-pair prediction scales by eight.
    assert Fraction(result['exact_predicted_block_cpu']) == 7 + 8 * 36
    assert result['predicted_block_cpu'] == 295.
    assert result['pilot_numerical_cpu'] == 43.
    assert result['includes_pilot_charge_in_validation_prediction'] is False
    assert sum(map(Fraction, result['exact_phase_predictions_cpu'].values()), Fraction(0)) == 295
    assert result['phase_predictions_cpu']['setup'] == 7.


def test_phase_identity_retains_binary64_timer_addition_rounding():
    row = timer_row('pilot', 0)
    row['training'] = dict(cpu_seconds=.1, phases=dict(simulation_cpu_seconds=.025, fitting_cpu_seconds=.05))
    row['evaluation']['cpu_seconds'] = .2
    row['pair_cpu_seconds'] = .1 + .2
    phases = phase_costs(row)
    expected_rounding = Fraction(.1 + .2) - Fraction(.1) - Fraction(.2)
    assert expected_rounding != 0
    assert phases['timer_sum_rounding'] == expected_rounding
    assert sum(phases.values(), Fraction(0)) == Fraction(row['pair_cpu_seconds'])
    rows = [dict(deepcopy(row), repetition=rep) for rep in range(8)]
    result = forecast(ITEM, rows, dict(cpu_seconds=.125))
    assert Fraction(result['exact_phase_predictions_cpu']['timer_sum_rounding']) == 64 * expected_rounding
    assert Fraction(result['exact_predicted_block_cpu']) == Fraction(.125) + 64 * Fraction(.1 + .2)


def test_validation_charge_and_phase_changes_close_without_pilot_cost():
    predicted = prediction()
    result = summarise_cell(ITEM, validation(), dict(cpu_seconds=32.), predicted)
    assert Fraction(result['exact_predicted_block_cpu']) == 640
    assert Fraction(result['exact_actual_block_cpu']) == 608
    assert Fraction(result['exact_ratio']) == Fraction(19, 20)
    assert result['pilot_numerical_cpu'] == 136.
    assert result['phase_changes_cpu']['setup'] == -32.
    assert all(value == 0. for key, value in result['phase_changes_cpu'].items() if key != 'setup')
    assert sum(map(Fraction, result['exact_phase_actual_cpu'].values()), Fraction(0)) == 608
    assert sum(Fraction(result['exact_phase_actual_cpu'][key]) -
               Fraction(predicted['exact_phase_predictions_cpu'][key])
               for key in result['exact_phase_actual_cpu']) == -32


@pytest.mark.parametrize('pair_cost, expected_ratio', [(8., Fraction(9, 10)), (10., Fraction(11, 10))])
def test_both_exact_tolerance_boundaries_are_inclusive(pair_cost, expected_ratio):
    result = summarise_cell(ITEM, validation(pair_cost), dict(cpu_seconds=64.), prediction())
    assert Fraction(result['exact_ratio']) == expected_ratio
    assert result['within_registered_tolerance'] is True


@pytest.mark.parametrize('pair_cost', [math.nextafter(8., -math.inf), math.nextafter(10., math.inf)])
def test_one_ulp_outside_each_tolerance_boundary_is_rejected(pair_cost):
    result = summarise_cell(ITEM, validation(pair_cost), dict(cpu_seconds=64.), prediction())
    assert result['within_registered_tolerance'] is False


def test_within_block_drift_is_retained_despite_exact_aggregate_prediction():
    rows = [timer_row('validation', rep, 8. if rep < 32 else 10.) for rep in range(64)]
    result = summarise_cell(ITEM, rows, dict(cpu_seconds=64.), prediction())
    assert result['exact_ratio'] == '1'
    assert result['first_half_mean_pair_cpu'] == 8.
    assert result['second_half_mean_pair_cpu'] == 10.


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reordered'])
@pytest.mark.parametrize('phase', ['pilot', 'validation'])
def test_incomplete_duplicate_or_reordered_repetitions_rejected(phase, mutation):
    rows = pilot() if phase == 'pilot' else validation()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1]['repetition'] = rows[-2]['repetition']
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match='complete ordered'):
        if phase == 'pilot':
            forecast(ITEM, rows, dict(cpu_seconds=64.))
        else:
            summarise_cell(ITEM, rows, dict(cpu_seconds=64.), prediction())


@pytest.mark.parametrize('key, wrong', [('case', 1), ('position', 1), ('evaluation_paths', 256), ('phase', 'warmup')])
@pytest.mark.parametrize('phase', ['pilot', 'validation'])
def test_mismatched_phase_or_scheduled_identity_rejected(phase, key, wrong):
    rows = pilot() if phase == 'pilot' else validation()
    rows[3][key] = wrong
    with pytest.raises(ValueError, match='identity'):
        if phase == 'pilot':
            forecast(ITEM, rows, dict(cpu_seconds=64.))
        else:
            summarise_cell(ITEM, rows, dict(cpu_seconds=64.), prediction())


@pytest.mark.parametrize('key, wrong', [('case', 1), ('position', 1), ('evaluation_paths', 256), ('validation_pairs', 63)])
def test_forecast_cannot_be_reused_for_another_validation_cell(key, wrong):
    predicted = prediction()
    predicted[key] = wrong
    with pytest.raises(ValueError, match='Forecast identity'):
        summarise_cell(ITEM, validation(), dict(cpu_seconds=64.), predicted)


def test_inconsistent_phase_forecast_rejected():
    predicted = prediction()
    predicted['exact_phase_predictions_cpu']['evaluation'] = '1'
    with pytest.raises(ValueError, match='does not close'):
        summarise_cell(ITEM, validation(), dict(cpu_seconds=64.), predicted)


def test_nonpositive_total_forecast_rejected():
    with pytest.raises(ValueError, match='positive'):
        forecast(ITEM, pilot(0.), dict(cpu_seconds=0.))
