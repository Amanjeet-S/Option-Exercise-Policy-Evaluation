"""Analytical moment, information-order and unclipped dual baseline checks."""

from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from qf_research.optimal_stopping.multivariate_benchmark import (
    published_max_call, simulate_max_call,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_polynomial_dual import (
    conditional_basis, dual_paths, exponent_vectors, fit_values, paired_comparison,
    policy_targets, polynomial_basis,
)


def test_complete_unsorted_basis_preserves_asset_labels_and_order():
    assert len(exponent_vectors(5, 2)) == 21
    assert len(exponent_vectors(5, 3)) == 56
    actual = polynomial_basis([[110., 90.]], 2)
    assert actual[0] == pytest.approx([1., 1.1, .9, 1.21, .99, .81])
    assert not np.array_equal(actual, polynomial_basis([[90., 110.]], 2))
    for states in ([[0., 100.]], [[np.nan, 100.]]):
        with pytest.raises(ValueError):
            polynomial_basis(states, 2)
    with pytest.raises(ValueError):
        exponent_vectors(2, 4)


def test_correlated_lognormal_moments_match_closed_forms_through_degree_three():
    model = replace(published_max_call(2, correlation=-.4), volatilities=(.2, .3), dividends=(.1, .02))
    states, step = np.array([[123., 80.]]), .4
    result = conditional_basis(states, model, step, 3)[0]
    for observed, (a, b) in zip(result, exponent_vectors(2, 3)):
        exponent = (a*(.05-.1)+b*(.05-.02)
                    +.5*a*(a-1)*.2**2+.5*b*(b-1)*.3**2+a*b*.2*.3*(-.4))
        expected = (123/100)**a*(80/100)**b*math.exp(step*exponent)
        assert observed == pytest.approx(expected, rel=2e-14)
    with pytest.raises(ValueError):
        conditional_basis(states, model, 0., 3)


def test_conditional_cubic_moments_match_independent_transition_sampling():
    model = replace(published_max_call(2, correlation=.6), spots=(105., 95.),
                    exercise_times=(0., .4))
    paths = simulate_max_call(model, 100000, np.random.default_rng(6281))
    sample = polynomial_basis(paths.states[:, 1], 3)
    expected = conditional_basis([model.spots], model, .4, 3)[0]
    means = sample.mean(axis=0)
    standard_errors = sample.std(axis=0, ddof=1)/np.sqrt(len(sample))
    assert np.all(np.abs(means[1:]-expected[1:]) < 6*standard_errors[1:])
    assert means[0] == expected[0] == 1.


def test_zero_volatility_and_singular_correlation_moments_are_supported():
    model = replace(published_max_call(2, correlation=1., volatility=0.), spots=(120., 90.),
                    exercise_times=(0., 1.))
    paths = simulate_max_call(model, 2, np.random.default_rng(913))
    assert conditional_basis([model.spots], model, 1., 3)[0] == pytest.approx(
        polynomial_basis(paths.states[:, 1], 3)[0], rel=2e-14)


def test_targets_use_frozen_policy_and_terminal_polynomial_fit_is_unconditional():
    model = replace(published_max_call(2), exercise_times=(0., 1., 2.))
    states = np.array([[[100., 100.], [110., 95.], [10000., 10000.]],
                       [[100., 100.], [100., 95.], [120., 90.]]])
    targets = policy_targets(states, model, [[], [5., 0., 0., 0., 0., 0.], []])
    assert targets[1][0] == pytest.approx(10*math.exp(-.05))
    assert targets[1][1] == pytest.approx(20*math.exp(-.1))
    assert np.array_equal(targets[0], targets[1])
    rng = np.random.default_rng(214)
    fitting_states = rng.lognormal(math.log(100), .3, size=(500, 3, 2))
    true = np.array([-3., 2., -1., .4, .8, -.2])
    polynomial_targets = [polynomial_basis(fitting_states[:, j], 2) @ true for j in range(3)]
    fitted, diagnostics = fit_values(fitting_states, polynomial_targets, 2)
    assert np.allclose(fitted[-1], true, rtol=1e-11, atol=1e-11)
    assert all(d['rank'] == 6 for d in diagnostics)


def test_negative_fitted_values_are_not_clipped_and_future_states_do_not_change_current_m():
    model = replace(published_max_call(2), rate=0., dividends=(0., 0.),
                    exercise_times=(0., 1., 2.))
    states = np.array([[[100., 100.], [120., 100.], [130., 110.]],
                       [[100., 100.], [90., 100.], [110., 80.]]])
    coefficients = [np.zeros(6), np.array([-5., -2., 0., 0., 0., 0.]), np.zeros(6)]
    result = dual_paths(states, model, [1, 1], coefficients, 2)
    assert result['martingale'][:, 1] == pytest.approx([-.4, .2], abs=1e-14)
    assert np.all(result['gap'] >= 0)
    assert result['gap'] == pytest.approx(result['dual']-result['policy_reward']+result['stopped_martingale'])
    changed = states.copy(); changed[:, -1] *= 10
    other = dual_paths(changed, model, [1, 1], coefficients, 2)
    assert np.array_equal(result['martingale'][:, 1], other['martingale'][:, 1])
    with pytest.raises(ValueError):
        dual_paths(states, model, [True, True], coefficients, 2)


def test_zero_functions_give_pathwise_lookback_dual_without_terminal_equality():
    model = replace(published_max_call(2), exercise_times=(0., 1., 2.))
    states = np.array([[[100., 100.], [120., 100.], [110., 100.]],
                       [[100., 100.], [100., 100.], [130., 100.]]])
    result = dual_paths(states, model, [2, 2], [np.zeros(6)]*3, 2)
    assert np.array_equal(result['martingale'], np.zeros((2, 3)))
    assert result['gap'] == pytest.approx([20*math.exp(-.05)-10*math.exp(-.1), 0.])


def test_paired_difference_standard_error_accounts_for_common_paths():
    def scalar(n): return dict(numerator=str(n), denominator='100')
    rows = [dict(path_index=i, signed_gap_certificate=dict(lower=scalar(100*(i+1)-1),
                                                          upper=scalar(100*(i+1)+1))) for i in range(4)]
    result = paired_comparison(np.array([2., 4., 6., 8.]), rows)
    expected = np.array([1., 2., 3., 4.])
    assert result['mean'] == expected.mean()
    assert result['standard_error'] == pytest.approx(expected.std(ddof=1)/2)
    assert result['mean_difference_range_from_hinge_brackets'] == pytest.approx([2.49, 2.51])


def test_preparation_phases_cover_the_same_three_stream_work():
    from certify_multivariate_gap import prepare
    record = prepare(dimension=2,training_count=256,value_count=256,evaluation_count=8,seed=99241)
    assert record['design']['spawn_keys']==[[0],[1],[2]]
    assert record['design']['evaluation_independent_of_both_fitting_streams'] is True
    expected = {'policy_training','hinge_value_stream_and_targets','hinge_regression',
                'outer_evaluation_and_record_construction','preparation_bookkeeping'}
    assert set(record['preparation_phases'])==expected
    for key in ('cpu_seconds','wall_seconds'):
        costs = [phase[key] for phase in record['preparation_phases'].values()]
        assert min(costs)>=-1e-9
        assert sum(costs)==pytest.approx(record['preparation_cost'][key],abs=1e-12)
    assert record['model']['spots']==(100.,100.)
    assert len(record['states'])==8


def test_polynomial_fit_timers_cover_each_date_without_changing_the_basis_fit():
    rng = np.random.default_rng(31916)
    states = rng.lognormal(math.log(100.),.3,size=(200,4,2))
    coefficients = np.array([-.2,.5,.9,-.1,.4,.7])
    targets = [polynomial_basis(states[:,j],2)@coefficients for j in range(4)]
    fitted,diagnostics = fit_values(states,targets,2)
    assert [row['date_index'] for row in diagnostics]==[1,2,3]
    assert all(row['fit_cpu_seconds']>=0 for row in diagnostics)
    for row in fitted[1:]:
        np.testing.assert_allclose(row,coefficients,atol=1e-10)
