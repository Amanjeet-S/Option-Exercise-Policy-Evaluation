"""Independent identities and quadrature for declared mathematical examples."""

from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap, european_reference
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy
from qf_research.optimal_stopping.value_martingale import (
    LinearValue, compress, jump_value, build_values, observation_rights, value_martingale, evaluate_value_dual,
)


@pytest.mark.parametrize('state', [0., 1., 70., 100., 300.])
@pytest.mark.parametrize('sd', [0., .15, .8])
def test_piecewise_expectation_against_independent_segment_quadrature(state, sd):
    value = LinearValue((0., 30., 85., 110., 180.), (80., 65., 22., 10., 4.), -.01)
    if state == 0 or sd == 0:
        expected = float(value(state))
    else:
        breaks = [-np.inf, *[(math.log(x/state)+.5*sd*sd)/sd for x in value.knots[1:]], np.inf]
        def integrand(z):
            if abs(z) > 38:
                return 0.
            return float(value(state*math.exp(-.5*sd*sd+sd*z)))*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
        expected = sum(quad(integrand, a, b, epsabs=2e-10, epsrel=2e-12)[0] for a, b in zip(breaks, breaks[1:]))
    assert float(value.expectation(state, sd)) == pytest.approx(expected, abs=3e-9)


def test_affine_and_put_expectations_have_independent_analytical_values():
    affine = LinearValue((0., 100., 200.), (3., 203., 403.), 2.)
    states = np.array([0., .001, 50., 100., 1e6])
    np.testing.assert_allclose(affine.expectation(states, .6), 3.+2*states, atol=1e-10)
    value = LinearValue((0., 100., 400.), (100., 0., 0.))
    model = FundedCashModel(103., .23, 1., 1.)
    option = Option(100., .7, 'put')
    reference = european_reference(model, option, FundedExercise(.7, 1.))
    assert float(value.expectation(103., .23*math.sqrt(.7))) == pytest.approx(reference.present_value, abs=1e-11)


def test_compression_has_uniform_error_bound_and_preserves_tail():
    grid = np.linspace(0., 500., 1001)
    values = 80*np.exp(-grid/60.)+2.
    original = LinearValue(tuple(grid), tuple(values), .1)
    smaller = compress(original, .01)
    dense = np.linspace(0., 800., 8001)
    assert len(smaller.knots) < len(original.knots)/4
    assert np.max(np.abs(original(dense)-smaller(dense))) <= .01000000001
    np.testing.assert_allclose(smaller(dense[dense >= 500]), original(dense[dense >= 500]), atol=1e-12)


def test_exact_jump_composition_includes_liquidation_plateau():
    value = LinearValue((0., 80., 110., 400.), (95., 15., 0., 0.))
    shifted = jump_value(value, 50.)
    states = np.linspace(0., 1000., 2001)
    np.testing.assert_allclose(shifted(states), value(np.maximum(states-50., 0.)), atol=1e-13)
    assert float(shifted(0.)) == float(shifted(50.)) == 95.
    assert jump_value(value, 0.) is value


def fixture(sigma=.3):
    model = FundedCashModel(100., sigma, 1., .97, (DividendCap(.4, 30.), DividendCap(.8, 40.)))
    option = Option(100., 1., 'put')
    rights = (FundedExercise(0., 1.), FundedExercise(.2, .99),
              FundedExercise(.4, .98, 'before'), FundedExercise(.4, .98),
              FundedExercise(.7, .97), FundedExercise(1., .96))
    return model, option, rights


def test_added_event_observations_preserve_every_original_path_and_cash_payment():
    model, option, rights = fixture()
    approx = build_values(model, option, rights, core_steps=80, wing_steps=12, tolerance=.01)
    old = simulate_paths(model, option, rights, 500, np.random.default_rng(98))
    expanded = simulate_paths(model, option, observation_rights(approx), 500, np.random.default_rng(98))
    sample = value_martingale(approx, expanded)
    np.testing.assert_array_equal(sample.paths.states, old.states)
    np.testing.assert_array_equal(sample.paths.detached_payments, old.detached_payments)
    np.testing.assert_array_equal(sample.values[:, 2], sample.values[:, 3])
    np.testing.assert_array_equal(sample.values[:, 0], np.zeros(500))


def test_european_martingale_recovers_constant_analytical_value():
    model = FundedCashModel(101., .2, 1., .98)
    option, rights = Option(100., 1., 'put'), (FundedExercise(1., .97),)
    approx = build_values(model, option, rights, core_steps=40, tolerance=.001)
    train = simulate_paths(model, option, rights, 50, np.random.default_rng(1))
    policy = fit_policy(option, train, 2)
    obs = simulate_paths(model, option, observation_rights(approx), 2000, np.random.default_rng(2))
    result = evaluate_value_dual(policy, value_martingale(approx, obs))
    reference = european_reference(model, option, rights[-1]).present_value
    np.testing.assert_allclose(result.upper_payoffs, reference, atol=3e-12)
    assert result.gap_bound.value == result.gap_bound.standard_error == 0.


def test_stopped_martingale_mean_with_multiple_cash_caps_and_before_rights():
    model, option, rights = fixture()
    approx = build_values(model, option, rights, core_steps=80, wing_steps=12, tolerance=.02)
    train = simulate_paths(model, option, rights, 2000, np.random.default_rng(16))
    policy = fit_policy(option, train, 2)
    observations = simulate_paths(model, option, observation_rights(approx), 30000, np.random.default_rng(17))
    result = evaluate_value_dual(policy, value_martingale(approx, observations))
    for est in (result.stopped_martingale, result.terminal_martingale):
        assert abs(est.value) < 5*est.standard_error+1e-11
    assert np.min(result.gap_payoffs) >= 0.


def test_zero_volatility_martingale_is_zero_even_with_coarse_values():
    model, option, rights = fixture(0.)
    approx = build_values(model, option, rights, core_steps=20, wing_steps=2, tolerance=10.)
    obs = simulate_paths(model, option, observation_rights(approx), 20, np.random.default_rng(1))
    np.testing.assert_allclose(value_martingale(approx, obs).values, 0., atol=1e-12)


def test_future_observation_changes_do_not_change_earlier_martingale_values():
    model, option, rights = fixture()
    approx = build_values(model, option, rights, core_steps=40, tolerance=.1)
    obs = simulate_paths(model, option, observation_rights(approx), 30, np.random.default_rng(31))
    original = value_martingale(approx, obs)
    states = obs.states.copy()
    states[:, -1] *= 1.1
    changed = value_martingale(approx, replace(obs, states=states))
    np.testing.assert_array_equal(original.values[:, :-1], changed.values[:, :-1])


def test_role_and_problem_mismatch_rejected():
    model, option, rights = fixture()
    approx = build_values(model, option, rights, core_steps=40, tolerance=.1)
    train = simulate_paths(model, option, rights, 100, np.random.default_rng(7))
    policy = fit_policy(option, train, 2)
    obs = simulate_paths(model, option, observation_rights(approx), 100, np.random.default_rng(7))
    with pytest.raises(ValueError, match='Training stream'):
        evaluate_value_dual(policy, value_martingale(approx, obs))
    with pytest.raises(ValueError, match='match the value'):
        value_martingale(approx, replace(obs, model=replace(model, volatility=.4)))
    with pytest.raises(ValueError, match='puts only'):
        build_values(model, replace(option, kind='call'), rights)


@pytest.mark.parametrize('value', [float('nan'), -.1])
def test_invalid_tolerances_and_transition_inputs(value):
    function = LinearValue((0., 1.), (1., 0.))
    with pytest.raises(ValueError):
        compress(function, value)
    with pytest.raises(ValueError):
        function.expectation(np.array([1.]), value)
