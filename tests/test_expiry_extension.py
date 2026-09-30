"""Analytical terminal identities and adaptation of embedded stopping rules."""

from dataclasses import replace
import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.computational.funded_kernel import prepare, price_prepared
from qf_research.optimal_stopping.funded_policy import fit_policy, simulate_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale, evaluate_value_dual
from qf_research.optimal_stopping.expiry_extension import (
    tail_put, price_terminal_extension, extended_calendar, restrict_paths,
    evaluate_embedded_policy, european_increment,
)


def fixture():
    model = FundedCashModel(100., .2, 1., 1.)
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.5, 1.), FundedExercise(1., 1.))
    return model, option, rights


def test_conditional_tail_matches_black_scholes_and_zero_state():
    states = np.array([0., 80., 100., 120.])
    values = tail_put(states, 100., .3, .1)
    assert values[0] == 100.
    for s,v in zip(states[1:], values[1:]):
        assert v == pytest.approx(black_scholes(Market(s, 0., .3), Option(100., .1, 'put')), abs=1e-12)


def test_time_value_is_non_negative_and_maximal_at_strike():
    states = np.r_[np.linspace(0., 200., 1001), 100.]
    gain = tail_put(states, 100., .2, .1)-np.maximum(100.-states, 0.)
    assert gain.min() >= -1e-12
    assert gain.max() == pytest.approx(gain[-1], abs=1e-12)


@pytest.mark.parametrize('sigma,extra', [(.2, 0.), (0., .1), (0., 0.)])
def test_deterministic_or_zero_extension_has_no_extra_time_value(sigma, extra):
    states = np.array([0., 50., 100., 150.])
    np.testing.assert_array_equal(tail_put(states, 100., sigma, extra), np.maximum(100.-states, 0.))


def test_zero_extension_preserves_prepared_solver_exactly():
    model, option, rights = fixture()
    p = prepare(model, option, 500, 100, 400., rights)
    assert price_terminal_extension(p, 0.) == price_prepared(p)


def test_terminal_replacement_matches_independent_longer_european_value():
    model, option, rights = fixture()
    p = prepare(model, option, 1600, 600, 400., (rights[-1],))
    actual = price_terminal_extension(p, .1).premium_quote
    expected = black_scholes(Market(100., 0., .2), replace(option, maturity=1.1))
    assert actual == pytest.approx(expected, abs=6e-4)


def test_early_exercise_calendar_increment_is_below_european_increment():
    model, option, _ = fixture()
    rights = (FundedExercise(.25, 1.), FundedExercise(.5, .99), FundedExercise(1., .97))
    p = prepare(model, option, 1600, 600, 400., rights)
    increment = price_terminal_extension(p, .1).premium_quote-price_terminal_extension(p, 0.).premium_quote
    bound = european_increment(model, option, rights[-1], .1)['increment_quote']
    assert 0 <= increment <= bound+6e-4


def test_new_cash_event_and_expiry_cash_event_are_rejected():
    model, option, rights = fixture()
    with pytest.raises(ValueError, match='no cash transfer'):
        extended_calendar(replace(model, dividends=(DividendCap(1.01, 1.),)), option, rights, (.02,))
    p = prepare(replace(model, dividends=(DividendCap(1., 1.),)), option, 50, 10, 400., rights)
    with pytest.raises(ValueError, match='at the original expiry'):
        price_terminal_extension(p, .02)


@pytest.mark.parametrize('extensions', [(), (.1, .05), (0.,), (-.1,)])
def test_non_nested_extension_lists_are_rejected(extensions):
    model, option, rights = fixture()
    with pytest.raises(ValueError):
        extended_calendar(model, option, rights, extensions)


def test_embedded_policy_ignores_future_information_and_baseline_dual_matches():
    model, option, rights = fixture()
    training = simulate_paths(model, option, rights, 500, np.random.default_rng(1))
    policy = fit_policy(option, training, 2)
    longer, union = extended_calendar(model, option, rights, (.1, .2))
    approx = build_values(model, longer, union, core_steps=80)
    draws = simulate_paths(model, longer, observation_rights(approx), 2000, np.random.default_rng(2))
    sample = value_martingale(approx, draws)
    bound, original = evaluate_embedded_policy(policy, sample)
    states = sample.paths.states.copy(); states[:, len(rights):] *= 2
    changed = replace(sample, paths=replace(sample.paths, states=states))
    _, repeated = evaluate_embedded_policy(policy, changed)
    np.testing.assert_array_equal(original.stopping_indices, repeated.stopping_indices)
    np.testing.assert_array_equal(original.discounted_payoffs, repeated.discounted_payoffs)
    assert bound.gap_payoffs.min() >= 0
    base = build_values(model, option, rights, core_steps=80)
    restricted = restrict_paths(draws, option, observation_rights(base))
    base_sample = value_martingale(base, restricted)
    normal = evaluate_value_dual(policy, base_sample)
    embedded,_ = evaluate_embedded_policy(policy, base_sample)
    np.testing.assert_array_equal(normal.gap_payoffs, embedded.gap_payoffs)


def test_cash_columns_are_restricted_with_horizon_and_funding_changes_rejected():
    model, option, rights = fixture()
    model = replace(model, dividends=(DividendCap(.75, 1.), DividendCap(1.1, 2.)))
    longer = replace(option, maturity=1.2)
    all_rights = rights+(FundedExercise(1.2, 1.),)
    draws = simulate_paths(model, longer, all_rights, 20, np.random.default_rng(7))
    short = restrict_paths(draws, option, rights)
    assert short.detached_payments.shape == (20, 1)
    np.testing.assert_array_equal(short.states, draws.states[:, :2])
    with pytest.raises(ValueError, match='unchanged terms'):
        restrict_paths(draws, option, (rights[0], replace(rights[-1], settlement_factor=.99)))
