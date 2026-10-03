"""Martingale predictability, event accounting and finite-calendar duality."""

from dataclasses import replace
import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap, deterministic_value
from qf_research.optimal_stopping.funded_policy import (
    ContinuationRule, FundedPaths, simulate_paths, fit_policy, calibrate_control, realise_policy,
)
from qf_research.optimal_stopping.funded_dual import gain_at_right, hedge_weights, martingale_values, evaluate_dual


def fixture(kind="put", sigma=.3, n=20000):
    m = FundedCashModel(100., sigma, .99, .98, (DividendCap(.4, 30.), DividendCap(.8, 40.)))
    o = Option(100., 1., kind)
    ds = (FundedExercise(.2, .99), FundedExercise(.4, .98, "before"), FundedExercise(.4, .98),
          FundedExercise(.7, .97), FundedExercise(1., .96))
    samples = [simulate_paths(m, o, ds, k, np.random.default_rng(seed)) for k, seed in ((5000, 44), (5000, 45), (n, 46))]
    policy = fit_policy(o, samples[0], 2)
    return policy, calibrate_control(policy, samples[1]), samples[2]


def test_total_gains_do_not_jump_when_ownership_becomes_a_receivable():
    policy, control, sample = fixture()
    np.testing.assert_allclose(gain_at_right(sample, 1), gain_at_right(sample, 2), atol=2e-14)
    np.testing.assert_allclose(gain_at_right(sample, 4), sample.total_gain, rtol=0., atol=3e-14)
    values = martingale_values(policy, sample, control.coefficient)
    np.testing.assert_allclose(values[:, 1], values[:, 2], atol=2e-14)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("hedge", ["zero", "constant", "policy_slope"])
def test_dual_inequality_and_stopped_martingale_mean(kind, hedge):
    policy, control, sample = fixture(kind, n=80000)
    result = evaluate_dual(policy, sample, control, hedge=hedge)
    assert np.all(result.gap_payoffs >= 0.)
    assert result.upper_value.value >= result.policy_value.value
    assert result.pricing_lower <= result.pricing_upper
    assert result.gap_bound.value == pytest.approx(result.upper_value.value-result.policy_value.value)
    for estimate in (result.stopped_martingale, result.terminal_martingale):
        assert abs(estimate.value) <= 5*estimate.standard_error+1e-12


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_volatility_dual_is_exact_for_deterministic_policy(kind):
    policy, control, sample = fixture(kind, sigma=0., n=100)
    result = evaluate_dual(policy, sample, control)
    expected = deterministic_value(policy.model, policy.option, policy.decisions)
    assert result.policy_value.value == pytest.approx(expected)
    assert result.upper_value.value == pytest.approx(expected)
    assert result.gap_bound.value == 0.


def test_current_increment_cannot_use_a_future_state_in_its_weight():
    policy, control, sample = fixture(n=1000)
    original = martingale_values(policy, sample, control.coefficient)
    changed = sample.states.copy()
    changed[:, -1] *= 2.
    other = martingale_values(policy, replace(sample, states=changed), control.coefficient)
    np.testing.assert_array_equal(original[:, :-1], other[:, :-1])
    first_gain = gain_at_right(sample, 0)
    np.testing.assert_allclose(original[:, 0], np.clip(control.coefficient, -1., 0.)*(first_gain-99.))
    prev_weight = hedge_weights(policy, 3, sample.states[:, 3], control.coefficient)
    np.testing.assert_allclose(other[:, -1]-original[:, -1], prev_weight*(changed[:, -1]-sample.states[:, -1]), atol=1e-12)


def test_constant_hedge_telescopes_exactly():
    policy, control, sample = fixture()
    values = martingale_values(policy, sample, -.4, "constant")
    for j in range(len(policy.decisions)):
        np.testing.assert_allclose(values[:, j], -.4*(gain_at_right(sample, j)-99.), atol=1e-13)


def test_exhaustive_finite_support_martingale_has_known_snell_value():
    # Four equally likely paths are an algebraic martingale fixture, not GBM draws.
    model, option = FundedCashModel(100., .2, 1., 1.), Option(100., 1., "put")
    rights = (FundedExercise(.5, 1.1), FundedExercise(1., .8))
    train = FundedPaths(model, 1., rights, np.array([[80., 40.], [80., 120.], [120., 60.], [120., 180.]]), np.empty((4, 0)), "train")
    policy = fit_policy(option, train, 1, minimum_itm_paths=2)
    rule = ContinuationRule(0., 1., (21.,), 4, 1, 1., 0., 200., "declared_fixture")
    policy = replace(policy, rules=(rule, policy.rules[-1]))
    pilot = replace(train, stream_id="pilot")
    control = calibrate_control(policy, pilot)
    evaluation = replace(train, stream_id="evaluation")
    result = evaluate_dual(policy, evaluation, control)
    # First-node max(30,20); second-node max(0,10); equal node probabilities.
    exact_snell = .5*30+.5*10
    assert result.policy_value.value == pytest.approx(exact_snell)
    assert result.upper_value.value >= exact_snell
    assert result.stopped_martingale.value == pytest.approx(0., abs=1e-13)


def test_european_restriction_has_zero_dual_gap_for_any_hedge():
    policy, control, sample = fixture()
    rights = (policy.decisions[-1],)
    model, option = policy.model, policy.option
    train = simulate_paths(model, option, rights, 500, np.random.default_rng(100))
    policy = fit_policy(option, train, 2)
    pilot = simulate_paths(model, option, rights, 500, np.random.default_rng(101))
    control = calibrate_control(policy, pilot)
    evaluation = simulate_paths(model, option, rights, 1000, np.random.default_rng(102))
    result = evaluate_dual(policy, evaluation, control)
    assert result.gap_bound.value == result.gap_bound.standard_error == 0.


def test_hedge_and_sample_role_validation():
    policy, control, sample = fixture(n=100)
    for h in ("unknown",):
        with pytest.raises(ValueError):
            martingale_values(policy, sample, .1, h)
    with pytest.raises(ValueError, match="finite"):
        martingale_values(policy, sample, float("nan"))
    with pytest.raises(ValueError, match="pilot"):
        evaluate_dual(policy, replace(sample, stream_id=control.pilot_stream_id), control)
    with pytest.raises(ValueError, match="frozen policy"):
        evaluate_dual(policy, sample, replace(control, policy_fingerprint="wrong"))
