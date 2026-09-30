"""Mathematical fixtures for combined policy transfer and calendar embedding."""

from dataclasses import replace

import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.expiry_extension import extended_calendar, evaluate_embedded_policy
from qf_research.optimal_stopping.policy_transfer import evaluate_transferred_dual
from qf_research.optimal_stopping.joint_input_expiry import evaluate_joint_policy


def fixture(changed=True, extended=True):
    original = FundedCashModel(100., .24, 1., .97, (DividendCap(.4, 5.),))
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.2, 1.), FundedExercise(.4, .99, 'before'),
              FundedExercise(.4, .99), FundedExercise(.7, .98), FundedExercise(1., .97))
    policy = fit_policy(option, simulate_paths(original, option, rights, 2000, np.random.default_rng(910)), 2)
    model = replace(original, initial_share_factor=.99, volatility=.27, premium_factor=.96,
                    dividends=(DividendCap(.4, 6.),)) if changed else original
    scenario_rights = tuple(replace(d, settlement_factor=d.settlement_factor*.99) for d in rights) if changed else rights
    contract, calendar = extended_calendar(model, option, scenario_rights, (.01, .05)) if extended else (option, scenario_rights)
    approximation = build_values(model, contract, calendar, core_steps=80, wing_steps=12, tolerance=.01)
    paths = simulate_paths(model, contract, observation_rights(approximation), 1200, np.random.default_rng(911))
    return policy, scenario_rights, value_martingale(approximation, paths)


@pytest.mark.parametrize('changed,extended', [(False, False), (True, False), (False, True)])
def test_joint_method_reduces_exactly_to_existing_special_cases(changed, extended):
    source, rights, sample = fixture(changed, extended)
    result, realised = evaluate_joint_policy(source, sample, rights)
    old, previous = evaluate_embedded_policy(source, sample) if extended else evaluate_transferred_dual(source, sample)
    for field in ('adjusted_policy_payoffs', 'upper_payoffs', 'gap_payoffs'):
        np.testing.assert_array_equal(getattr(result, field), getattr(old, field))
    np.testing.assert_array_equal(realised.stopping_indices, previous.stopping_indices)
    assert result.gap_bound == old.gap_bound


def test_joint_decisions_ignore_later_states_and_match_manual_quote_transfer():
    source, rights, sample = fixture()
    result, realised = evaluate_joint_policy(source, sample, rights)
    positions = [sample.paths.decisions.index(d) for d in rights]
    # Translate the same observed quotes to original state coordinates, then
    # exercise the original policy. There is no reuse of the transfer routine.
    states = sample.paths.states[:, positions].copy()
    factors = np.array([old.settlement_factor/new.settlement_factor for old, new in zip(source.decisions, rights)])
    original_states = states*factors
    # Original cash-cap validation would be inappropriate for transferred
    # histories, so reconstruct the stopping predicate explicitly.
    manual = np.full(len(states), len(rights)-1)
    alive = np.ones(len(states), dtype=bool)
    for j, rule in enumerate(source.rules[:-1]):
        immediate = source.model.reward(source.option, source.decisions[j], original_states[:, j])
        if rule.coefficients:
            take = alive & (immediate > 0) & (immediate > np.maximum(rule.raw_prediction(original_states[:, j]), 0.))
            manual[take] = j
            alive[take] = False
    np.testing.assert_array_equal(manual, realised.stopping_indices)
    expected = np.array([sample.paths.model.reward(source.option, rights[j], states[i, j]) for i, j in enumerate(manual)])
    np.testing.assert_array_equal(expected, realised.discounted_payoffs)
    altered = sample.paths.states.copy()
    altered[:, [i for i, d in enumerate(sample.paths.decisions) if d.time > source.option.maturity]] *= .7
    _, again = evaluate_joint_policy(source, replace(sample, paths=replace(sample.paths, states=altered)), rights)
    np.testing.assert_array_equal(realised.stopping_indices, again.stopping_indices)
    np.testing.assert_array_equal(realised.discounted_payoffs, again.discounted_payoffs)
    assert result.gap_payoffs.min() >= 0


def test_extended_upper_problem_contains_each_frozen_stopping_reward():
    source, rights, sample = fixture()
    result, realised = evaluate_joint_policy(source, sample, rights)
    indices = np.array([sample.paths.decisions.index(d) for d in rights])[realised.stopping_indices]
    stopped = sample.values[np.arange(len(indices)), indices]
    np.testing.assert_allclose(result.gap_payoffs,
        result.upper_payoffs-realised.discounted_payoffs+stopped, rtol=0., atol=2e-14)
    assert result.gap_bound.value == pytest.approx(result.upper_value.value-result.policy_value.value)


@pytest.mark.parametrize('corruption', ['drop', 'shift', 'phase', 'factor'])
def test_missing_or_changed_scenario_rights_rejected(corruption):
    source, rights, sample = fixture()
    changed = list(rights)
    if corruption == 'drop': changed.pop(0)
    if corruption == 'shift': changed[0] = replace(changed[0], time=.21)
    if corruption == 'phase': changed[0] = replace(changed[0], phase='before')
    if corruption == 'factor': changed[0] = replace(changed[0], settlement_factor=.8)
    with pytest.raises(ValueError):
        evaluate_joint_policy(source, sample, tuple(changed))


def test_nonfinite_martingale_and_training_stream_rejected():
    source, rights, sample = fixture()
    bad = sample.values.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError): evaluate_joint_policy(source, replace(sample, values=bad), rights)
    with pytest.raises(ValueError):
        evaluate_joint_policy(source, replace(sample, paths=replace(sample.paths, stream_id=source.training_stream_id)), rights)

