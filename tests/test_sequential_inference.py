"""Predictability, scale and analytical-constant checks for sequential widths."""

import json
import math

import numpy as np
import pytest

from qf_research.optimal_stopping.sequential_inference import (
    empirical_bernstein_halfwidth, empirical_bernstein_sequence,
    stitched_empirical_bernstein_boundary,
)


def test_closed_form_boundary_calibration_uses_two_tail_error_budget():
    # eta=4 gives k1=k2=3/2; this confidence choice makes ell(m)=4.
    # The resulting sum boundary is exactly 6+3*sqrt(5).
    confidence = 1-2*(math.pi**2/6)*math.exp(-4)
    boundary = stitched_empirical_bernstein_boundary(
        1., 1., confidence=confidence, eta=4., s=2., m=1.)
    assert boundary == pytest.approx(6+3*math.sqrt(5), rel=2e-15)
    assert empirical_bernstein_halfwidth(
        1., 7, 1., confidence=confidence, eta=4., s=2., m=1.) == pytest.approx(boundary/7)


def test_predictions_precede_observations_and_variance_uses_their_errors():
    result = empirical_bernstein_sequence([0., 1., 0.], 0., 1.)
    rows = result['rows']
    assert [r['prediction'] for r in rows] == [0.5, 0., 0.5]
    assert [r['intrinsic_variance'] for r in rows] == [0.25, 1.25, 1.5]
    assert [r['mean'] for r in rows] == pytest.approx([0., .5, 1/3])
    changed = empirical_bernstein_sequence([0., 1., 1.], 0., 1.)
    assert changed['rows'][:2] == rows[:2]
    assert changed['rows'][2]['prediction'] == rows[2]['prediction']
    assert result['settings']['alpha_side'] == pytest.approx(.025)


def test_appending_rows_does_not_change_previous_prefixes():
    data = [-.5, .1, .8]
    prefix = empirical_bernstein_sequence(data, -1., 1., initial_prediction=.2)
    longer = empirical_bernstein_sequence(data+[-1., 1., .7], -1., 1., initial_prediction=.2)
    assert longer['rows'][:len(data)] == prefix['rows']
    assert prefix['rows'][0]['prediction'] == .2
    assert prefix['rows'][0]['intrinsic_variance'] == pytest.approx(.49)


def test_default_construction_is_equivariant_under_units_and_location():
    data = np.array([-.8, .1, .4, -.2, .7])
    original = empirical_bernstein_sequence(data, -1., 1.)
    scale, shift = 3.7, -5.2
    transformed = empirical_bernstein_sequence(scale*data+shift, -scale+shift, scale+shift)
    assert transformed['settings']['m'] == pytest.approx(scale**2*original['settings']['m'])
    for old, new in zip(original['rows'], transformed['rows']):
        for key in ('mean', 'prediction', 'lower', 'upper', 'untruncated_lower', 'untruncated_upper'):
            assert new[key] == pytest.approx(scale*old[key]+shift)
        assert new['intrinsic_variance'] == pytest.approx(scale**2*old['intrinsic_variance'])
        assert new['halfwidth'] == pytest.approx(scale*old['halfwidth'])


def test_one_observation_at_midpoint_retains_positive_uncertainty():
    result = empirical_bernstein_sequence([2.], 1., 3.)
    row = result['rows'][0]
    assert row['intrinsic_variance'] == 0
    assert row['halfwidth'] > 0
    assert (row['lower'], row['upper']) == (1., 3.)
    assert row['halfwidth'] == empirical_bernstein_halfwidth(0., 1, 2.)
    assert result['floating_point_bound_certified'] is False
    assert result['coverage_validated'] is False
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_support_endpoints_are_valid_and_constant_samples_do_not_imply_zero_width():
    endpoints = empirical_bernstein_sequence([-1., 1., -1., 1.], -1., 1.)
    assert len(endpoints['rows']) == 4
    constant = empirical_bernstein_sequence([0.]*100, -1., 1.)
    assert all(row['intrinsic_variance'] == 0 for row in constant['rows'])
    assert 0 < constant['rows'][-1]['halfwidth'] < constant['rows'][0]['halfwidth']


def test_endpoint_routine_accepts_final_variance_without_retaining_the_sequence():
    result = empirical_bernstein_sequence([.1, -.2, .2, .3], -.5, .5, m=.2, eta=3., s=1.7)
    last = result['rows'][-1]
    assert last['halfwidth'] == empirical_bernstein_halfwidth(
        last['intrinsic_variance'], 4, 1., m=.2, eta=3., s=1.7)


@pytest.mark.parametrize('arguments', [
    ([], -1., 1.), ([[0., 1.]], -1., 1.), (0., -1., 1.),
    ([0., math.nan], -1., 1.), ([math.inf], -1., 1.),
    (['0'], -1., 1.), ([True], -1., 1.), ([True, .1], -1., 1.), ([1j], -1., 1.),
    ([0.], 0., 0.), ([0.], 1., -1.), ([0.], -math.inf, 1.),
    ([0.], -1., math.nan), ([2.], -1., 1.), ([0.], True, 2.),
    ([0.], -1e308, 1e308),
])
def test_invalid_sequences_or_support_are_rejected(arguments):
    with pytest.raises(ValueError):
        empirical_bernstein_sequence(*arguments)


@pytest.mark.parametrize('options', [
    {'confidence': 0}, {'confidence': 1}, {'confidence': True},
    {'confidence': math.nan}, {'eta': 1}, {'eta': math.inf}, {'eta': False},
    {'s': 1}, {'s': -2}, {'s': math.inf}, {'m': 0}, {'m': -1},
    {'m': math.inf}, {'m': True}, {'initial_prediction': 2},
    {'initial_prediction': math.nan}, {'initial_prediction': True},
])
def test_invalid_tuning_and_unbounded_initial_predictions_are_rejected(options):
    with pytest.raises(ValueError):
        empirical_bernstein_sequence([0., .1], -1., 1., **options)


@pytest.mark.parametrize('variance,width', [
    (-1., 1.), (math.inf, 1.), (math.nan, 1.), (True, 1.),
    (0., 0.), (0., -1.), (0., math.inf), (0., True),
    (0., 1e308), (0., 1e-300),
])
def test_invalid_or_unrepresentable_boundary_inputs_are_rejected(variance, width):
    with pytest.raises(ValueError):
        stitched_empirical_bernstein_boundary(variance, width)


@pytest.mark.parametrize('count', [True, 0, -1, 2.5, math.inf, '2'])
def test_invalid_prefix_lengths_are_rejected(count):
    with pytest.raises(ValueError):
        empirical_bernstein_halfwidth(1., count, 1.)


def test_unrepresentable_prediction_error_with_explicit_scale_is_rejected():
    with pytest.raises(ValueError):
        empirical_bernstein_sequence([-1e200, 1e200], -1e200, 1e200, m=1.)
