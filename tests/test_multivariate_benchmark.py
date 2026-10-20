"""Analytical and information-order checks for the public multiasset baseline."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import ttest_1samp

from qf_research.optimal_stopping.multivariate_benchmark import (
    MaxCallPaths, MaxCallPolicy, MaxCallRule, correlation_factor, evaluate_max_call,
    fit_max_call_policy, independent_european_max_call, monomial_indices,
    published_max_call, simulate_max_call, sorted_polynomial_basis,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_multivariate_benchmark import Moments
from study_multivariate_matrix import exercise_margin_grid


def test_covariance_factor_handles_singular_psd_and_rejects_invalid_correlations():
    matrix = np.array([[1., -.5, -.5], [-.5, 1., -.5], [-.5, -.5, 1.]])
    model = published_max_call(3, correlation=-.5)
    factor = correlation_factor(model.correlation)
    assert np.allclose(factor @ factor.T, matrix, atol=1e-14)
    assert simulate_max_call(model, 20, np.random.default_rng(1)).states.shape == (20, 10, 3)
    with pytest.raises(ValueError, match='positive-semidefinite'):
        published_max_call(3, correlation=-.6)
    with pytest.raises(ValueError):
        replace(model, correlation=((1., np.nan, 0.), (np.nan, 1., 0.), (0., 0., 1.)))
    with pytest.raises(ValueError, match='finite symmetric'):
        correlation_factor([[1., np.nan], [np.nan, 1.]])


def test_correlated_log_returns_have_the_declared_covariance():
    model = published_max_call(3, correlation=.6)
    paths = simulate_max_call(model, 80000, np.random.default_rng(402))
    dt = model.exercise_times[1]
    returns = np.log(paths.states[:, 1]/paths.states[:, 0])
    actual = np.cov(returns.T)
    expected = np.asarray(model.correlation)*.2**2*dt
    assert np.allclose(actual, expected, atol=2e-4, rtol=0.)
    with pytest.raises(ValueError, match='independent assets'):
        independent_european_max_call(model)


def test_full_state_polynomials_are_permutation_invariant_and_use_lower_ranked_assets():
    states = np.array([[110., 100., 90., 80., 70.], [102., 101., 99., 98., 96.]])
    basis = sorted_polynomial_basis(states, 100., 3)
    assert basis.shape == (2, 56)
    assert len(monomial_indices(5, 2)) == 21
    assert np.array_equal(basis, sorted_polynomial_basis(states[:, [4, 1, 0, 3, 2]], 100., 3))
    changed = states.copy(); changed[:, -1] -= 1.
    assert not np.array_equal(basis, sorted_polynomial_basis(changed, 100., 3))


def test_expiry_only_policy_and_zero_volatility_reference_are_exact():
    model = published_max_call(2, spot=110., volatility=0.)
    paths = simulate_max_call(model, 10, np.random.default_rng(9))
    continuation = tuple(MaxCallRule((), 0, 0, None) for _ in model.exercise_times)
    policy = MaxCallPolicy(model, 2, continuation, 'another-stream', 10)
    result = evaluate_max_call(policy, paths)
    assert np.array_equal(result.matched_gains, np.zeros(10))
    assert np.array_equal(result.stopping_indices, np.full(10, 9))
    value = np.exp(-.05*3)*max(110*np.exp(-.05*3)-100, 0.)
    assert independent_european_max_call(model)['value'] == pytest.approx(value, abs=1e-12)
    fitted = fit_max_call_policy(paths, 2)
    fresh = simulate_max_call(model, 10, np.random.default_rng(10))
    realised = evaluate_max_call(fitted, fresh)
    assert np.all(realised.stopping_indices == 0)
    assert np.all(realised.discounted_rewards == 10.)


def test_training_stream_cannot_evaluate_its_policy_and_future_state_cannot_change_current_decision():
    model = published_max_call(2, spot=100.)
    training = simulate_max_call(model, 300, np.random.default_rng(401))
    fitted = fit_max_call_policy(training, 2)
    with pytest.raises(ValueError, match='training stream'):
        evaluate_max_call(fitted, training)
    repeated_stream = simulate_max_call(model, 20, np.random.default_rng(401))
    with pytest.raises(ValueError, match='training stream'):
        evaluate_max_call(fitted, repeated_stream)
    states = np.full((2, 10, 2), 100.)
    states[:, 1] = 110.
    coefficient = (5.,)+(0.,)*5
    rules = (MaxCallRule((), 0, 0, None), MaxCallRule(coefficient, 100, 6, 1.),
             *(MaxCallRule((), 0, 0, None) for _ in range(8)))
    policy = MaxCallPolicy(model, 2, rules, 'training', 100)
    first = evaluate_max_call(policy, MaxCallPaths(model, states, 'fresh-one'))
    changed = states.copy(); changed[:, 2:] = 10000.
    second = evaluate_max_call(policy, MaxCallPaths(model, changed, 'fresh-two'))
    assert np.all(first.stopping_indices == 1)
    assert np.array_equal(first.stopping_indices, second.stopping_indices)
    assert np.array_equal(first.discounted_rewards, second.discounted_rewards)


def test_terminal_monte_carlo_matches_independent_product_cdf_integral():
    model = published_max_call(5, spot=100.)
    paths = simulate_max_call(model, 150000, np.random.default_rng(202611))
    payoffs = model.discounted_payoff(paths.states[:, -1], 9)
    reference = independent_european_max_call(model)
    assert abs(payoffs.mean()-reference['value']) < 5*payoffs.std(ddof=1)/np.sqrt(len(payoffs))
    assert reference['quadrature_error_estimate'] < 1e-6
    assert reference['certified_error_enclosure'] is False


def test_chunked_moments_match_an_independent_student_interval():
    values = np.array([-2., .2, 1., 3., 8., 20., -1.])
    moments = Moments()
    for chunk in [values[:2], values[2:3], values[3:]]:
        moments.add(chunk)
    summary = moments.summary()
    reference = ttest_1samp(values, 0.).confidence_interval(.95)
    assert summary['mean'] == pytest.approx(values.mean())
    assert summary['standard_error'] == pytest.approx(values.std(ddof=1)/np.sqrt(len(values)))
    assert summary['pointwise_student_interval'] == pytest.approx([reference.low, reference.high])


def test_scarce_high_degree_training_uses_the_declared_continue_fallback():
    model = published_max_call(5, spot=100.)
    training = simulate_max_call(model, 40, np.random.default_rng(611))
    policy = fit_max_call_policy(training, 3)
    assert not any(rule.coefficients for rule in policy.rules)
    evaluation = evaluate_max_call(policy, simulate_max_call(model, 100, np.random.default_rng(612)))
    assert np.all(evaluation.stopping_indices == 9)
    assert np.array_equal(evaluation.matched_gains, np.zeros(100))


def test_perfectly_correlated_assets_expose_rank_deficiency_without_invalid_regression():
    model = published_max_call(2, correlation=1.)
    training = simulate_max_call(model, 500, np.random.default_rng(613))
    assert np.array_equal(training.states[:, :, 0], training.states[:, :, 1])
    policy = fit_max_call_policy(training, 2)
    deficient = [rule for rule in policy.rules[1:-1] if rule.training_itm_paths >= 64]
    assert deficient and all(0 < rule.rank < 6 for rule in deficient)
    assert not any(rule.coefficients for rule in policy.rules)
    evaluation = evaluate_max_call(policy, simulate_max_call(model, 100, np.random.default_rng(614)))
    assert np.array_equal(evaluation.matched_gains, np.zeros(100))


def test_mixed_zero_volatility_reference_matches_a_call_above_a_deterministic_floor():
    from qf_research.optimal_stopping.analytical import black_scholes
    from qf_research.optimal_stopping.contracts import Market, Option
    model = replace(published_max_call(2), spots=(200., 100.), volatilities=(0., .2))
    floor = 200*np.exp(-.05*3)
    expected = np.exp(-.05*3)*(floor-100)+black_scholes(Market(100., .05, .2, .10),
        Option(float(floor), 3., 'call'))
    assert independent_european_max_call(model)['value'] == pytest.approx(expected, abs=1e-9)


def test_state_margin_grid_retains_both_assets_and_uses_strict_exercise_ties():
    model = published_max_call(2)
    coefficient = (5.,)+(0.,)*5
    rules = tuple(MaxCallRule(coefficient if index == 3 else (), 100, 6, 1.)
                  for index in range(10))
    policy = MaxCallPolicy(model, 2, rules, 'training', 100)
    rows, metadata = exercise_margin_grid(policy, lower=90., upper=110., points=3)
    assert len(rows) == 9
    assert metadata['time_index'] == 3 and metadata['time'] == 1.
    assert metadata['permutation_invariance_maximum_absolute_discrepancy'] == 0.
    lookup = {(row['asset_1'], row['asset_2']): row for row in rows}
    assert lookup[(90., 110.)]['exercise_margin'] == lookup[(110., 90.)]['exercise_margin']
    assert lookup[(90., 90.)]['exercise_selected'] is False
    assert lookup[(110., 110.)]['exercise_selected'] is True
