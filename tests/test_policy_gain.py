"""Finite-support mathematical examples, not observations or fitted market data."""

from dataclasses import replace

import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise
from qf_research.optimal_stopping.funded_policy import (
    ContinuationRule, FundedPaths, FundedPolicy, GainControl, realise_policy,
)
from qf_research.optimal_stopping.policy_gain import calibrate_gain_control, evaluate_gain
from qf_research.optimal_stopping.simulation import estimate_samples


def example():
    # Four equally likely leaves of a two-step martingale tree.
    model = FundedCashModel(100., .2, 1., .97)
    option = Option(105., 1., 'put')
    rights = (FundedExercise(.5, 1.), FundedExercise(1., .98))
    states = np.array([[90., 80.], [90., 100.], [110., 100.], [110., 120.]])
    paths = FundedPaths(model, 1., rights, states, np.empty((4, 0)), 'evaluation')
    rules = (ContinuationRule(100., 1., (10.,), 4, 1, 1., 90., 110., 'constant'),
             ContinuationRule(0., 1., (), 0, 0, None, None, None, 'terminal'))
    policy = FundedPolicy(model, option, rights, rules, 0, 4, 2, 1e8, 'training')
    return policy, paths


def test_gain_expectation_on_complete_martingale_tree():
    policy, paths = example()
    result = evaluate_gain(policy, paths)
    # Early exercise gives 15 on each down branch; otherwise wait to maturity.
    np.testing.assert_allclose(result.raw_gains, [-7.9, 12.1, 0., 0.], atol=1e-13)
    assert result.raw_pv.value == pytest.approx(1.05)
    assert result.raw_quote.value == pytest.approx(1.05/.97)
    assert result.control_mean.value == 0.
    assert not result.raw_gains.flags.writeable


def test_control_preserves_exact_mean_without_european_reference():
    policy, paths = example()
    control = calibrate_gain_control(policy, replace(paths, stream_id='pilot'))
    result = evaluate_gain(policy, paths, control=control)
    assert control.coefficient == pytest.approx(.1975)
    assert result.controlled_pv.value == pytest.approx(1.05)
    assert result.controlled_pv.standard_error < result.raw_pv.standard_error
    assert not result.controlled_gains.flags.writeable


def test_terminal_rule_has_exactly_zero_gain():
    policy, paths = example()
    rule = replace(policy.rules[0], coefficients=(), status='continue_insufficient_itm')
    policy = replace(policy, rules=(rule, policy.rules[-1]))
    control = calibrate_gain_control(policy, replace(paths, stream_id='pilot'))
    result = evaluate_gain(policy, paths, control=control)
    assert control.coefficient == 0.
    assert result.controlled_pv.value == result.controlled_pv.standard_error == 0.


def test_negative_gain_is_not_clipped():
    policy, paths = example()
    # Same martingale, but rising settlement value makes waiting preferable.
    rights = (policy.decisions[0], FundedExercise(1., 1.1))
    policy, paths = replace(policy, decisions=rights), replace(paths, decisions=rights)
    assert evaluate_gain(policy, paths).raw_pv.value < 0.


def test_pairing_preserves_covariance_and_cancels_terminal_payoff():
    policy, paths = example()
    terminal = replace(policy, rules=(replace(policy.rules[0], coefficients=()), policy.rules[-1]))
    a, b = evaluate_gain(policy, paths), evaluate_gain(terminal, paths)
    pay_a, pay_b = realise_policy(policy, paths), realise_policy(terminal, paths)
    np.testing.assert_allclose(a.raw_gains-b.raw_gains, pay_a.discounted_payoffs-pay_b.discounted_payoffs)
    covariance = np.cov(a.raw_gains, b.raw_gains, ddof=1)
    expected = (covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1])/4
    assert estimate_samples(a.raw_gains-b.raw_gains).standard_error**2 == pytest.approx(expected)


def test_independent_training_and_pilot_roles():
    policy, paths = example()
    with pytest.raises(ValueError, match='Training stream'):
        evaluate_gain(policy, replace(paths, stream_id='training'))
    with pytest.raises(ValueError, match='Training stream'):
        calibrate_gain_control(policy, replace(paths, stream_id='training'))
    control = calibrate_gain_control(policy, replace(paths, stream_id='pilot'))
    with pytest.raises(ValueError, match='pilot cannot'):
        evaluate_gain(policy, replace(paths, stream_id='pilot'), control=control)
    with pytest.raises(ValueError, match='another frozen policy'):
        evaluate_gain(replace(policy, requested_degree=1), paths, control=control)
    with pytest.raises(TypeError, match='exercise gains'):
        evaluate_gain(policy, paths, control=GainControl(0., policy.fingerprint, 'pilot', 4))


def test_zero_variance_control_uses_zero_coefficient():
    policy, paths = example()
    states = paths.states.copy()
    states[:, -1] = 100.
    paths = replace(paths, states=states)
    control = calibrate_gain_control(policy, replace(paths, stream_id='pilot'))
    assert control.coefficient == 0.
    assert evaluate_gain(policy, paths, control=control).control_mean.standard_error == 0.


@pytest.mark.parametrize('coefficient', [float('nan'), float('inf')])
def test_nonfinite_control_rejected(coefficient):
    policy, paths = example()
    control = calibrate_gain_control(policy, replace(paths, stream_id='pilot'))
    with pytest.raises(ValueError, match='must be finite'):
        evaluate_gain(policy, paths, control=replace(control, coefficient=coefficient))
