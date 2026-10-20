"""Independent reward, calendar and policy-versus-assessment checks."""

from dataclasses import replace

import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, deterministic_value, european_reference,
)
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.reference_policy import (
    ReferencePolicy, continuation_values, realise_reference_policy, evaluate_reference_policy,
)
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale,
)


def test_terminal_only_reward_and_centred_price_match_independent_formula():
    model = FundedCashModel(101., .2, 1., .98)
    option = Option(100., .7, "put")
    rights = (FundedExercise(.7, .97),)
    approximation = build_values(model, option, rights, core_steps=40)
    policy = ReferencePolicy(approximation)
    observations = simulate_paths(model, option, observation_rights(approximation),
                                  1000, np.random.default_rng(51))
    sample = value_martingale(approximation, observations)
    realised = realise_reference_policy(policy, sample.paths)
    np.testing.assert_array_equal(realised.discounted_payoffs,
                                  np.maximum(97.-sample.paths.states[:, 0], 0.))
    np.testing.assert_array_equal(realised.stopping_indices, np.zeros(1000, dtype=int))
    result = evaluate_reference_policy(policy, sample)
    oracle = european_reference(model, option, rights[-1]).present_value
    np.testing.assert_allclose(result.adjusted_policy_payoffs, oracle, atol=3e-12)
    assert result.gap_bound.value == result.gap_bound.standard_error == 0.


@pytest.mark.parametrize("after_factor, expected_index, expected_reward", [
    (.95, 1, 20.), (.70, 0, 15.),
])
def test_genuine_before_and_after_rights_use_the_correct_cash_phase(
        after_factor, expected_index, expected_reward):
    model = FundedCashModel(85., 0., 1., 1., (DividendCap(.5, 10.),))
    option = Option(100., 1., "put")
    rights = (FundedExercise(.5, 1., "before"), FundedExercise(.5, after_factor),
              FundedExercise(1., .8))
    approximation = build_values(model, option, rights, core_steps=80, tolerance=0.)
    policy = ReferencePolicy(approximation)
    paths = simulate_paths(model, option, rights, 8, np.random.default_rng(52))
    realised = realise_reference_policy(policy, paths)
    np.testing.assert_array_equal(realised.stopping_indices, np.full(8, expected_index))
    np.testing.assert_allclose(realised.discounted_payoffs, expected_reward, atol=1e-12)
    assert expected_reward == deterministic_value(model, option, rights)


def test_cash_event_observations_do_not_grant_additional_exercise_rights():
    model = FundedCashModel(85., 0., 1., 1., (DividendCap(.5, 10.),))
    option = Option(100., 1., "put")
    rights = (FundedExercise(.25, 1.), FundedExercise(1., .8))
    approximation = build_values(model, option, rights, core_steps=80, tolerance=0.)
    policy = ReferencePolicy(approximation)
    observations = simulate_paths(model, option, observation_rights(approximation),
                                  8, np.random.default_rng(53))
    sample = value_martingale(approximation, observations)
    realised = realise_reference_policy(policy, sample.paths)
    # The ungranted event reward would be 25, whereas the best genuine reward is 15.
    np.testing.assert_array_equal(realised.stopping_indices, np.zeros(8, dtype=int))
    np.testing.assert_allclose(realised.discounted_payoffs, 15.)
    with pytest.raises(ValueError, match="original rights"):
        realise_reference_policy(policy, observations)


def test_zero_volatility_rewards_follow_enumerated_optimal_stopping():
    model = FundedCashModel(100., 0., 1., .99, (DividendCap(.4, 20.), DividendCap(.8, 60.)))
    option = Option(100., 1., "put")
    rights = (FundedExercise(0., 1.), FundedExercise(.3, .98),
              FundedExercise(.6, .95), FundedExercise(1., .9))
    approximation = build_values(model, option, rights, core_steps=80, tolerance=0.)
    policy = ReferencePolicy(approximation)
    observations = simulate_paths(model, option, observation_rights(approximation),
                                  10, np.random.default_rng(54))
    sample = value_martingale(approximation, observations)
    result = evaluate_reference_policy(policy, sample)
    np.testing.assert_allclose(result.adjusted_policy_payoffs,
                               deterministic_value(model, option, rights), atol=1e-12)
    np.testing.assert_allclose(sample.values, 0., atol=1e-12)


def test_assessment_refinement_leaves_policy_rewards_and_stopping_unchanged():
    model = FundedCashModel(100., .25, 1., 1., (DividendCap(.4, 3.),))
    option = Option(100., 1., "put")
    rights = (FundedExercise(.2, .99), FundedExercise(.6, .97), FundedExercise(1., .95))
    coarse = build_values(model, option, rights, core_steps=40, tolerance=.05)
    fine = build_values(model, option, rights, core_steps=160, tolerance=.001)
    policy = ReferencePolicy(coarse)
    observations = simulate_paths(model, option, observation_rights(coarse),
                                  2000, np.random.default_rng(55))
    samples = [value_martingale(a, observations) for a in (coarse, fine)]
    realised = [realise_reference_policy(policy, s.paths) for s in samples]
    np.testing.assert_array_equal(realised[0].discounted_payoffs, realised[1].discounted_payoffs)
    np.testing.assert_array_equal(realised[0].stopping_indices, realised[1].stopping_indices)
    for sample in samples:
        result = evaluate_reference_policy(policy, sample)
        assert result.gap_payoffs.min() >= 0.
        stopped = sample.values[np.arange(len(sample.paths.states)), realised[0].stopping_indices]
        np.testing.assert_allclose(result.adjusted_policy_payoffs+stopped,
                                   realised[0].discounted_payoffs, atol=1e-12)


def test_initial_grid_value_is_never_substituted_for_forward_reward():
    model = FundedCashModel(100., .2, 1., 1.)
    option = Option(100., 1., "put")
    rights = (FundedExercise(.5, .97), FundedExercise(1., .94))
    approximation = build_values(model, option, rights, core_steps=40)
    paths = simulate_paths(model, option, rights, 200, np.random.default_rng(56))
    original = realise_reference_policy(ReferencePolicy(approximation), paths)
    changed = realise_reference_policy(ReferencePolicy(replace(approximation, initial_value=1e9)), paths)
    np.testing.assert_array_equal(original.discounted_payoffs, changed.discounted_payoffs)
    np.testing.assert_array_equal(original.stopping_indices, changed.stopping_indices)
    with pytest.raises(ValueError, match="terminal right"):
        continuation_values(ReferencePolicy(approximation), 1, np.array([100.]))


def test_better_policy_means_higher_reward_against_expiry_only():
    model = FundedCashModel(80., 0., 1., 1.)
    option = Option(100., 1., "put")
    rights = (FundedExercise(.5, 1.), FundedExercise(1., .9))
    policy = ReferencePolicy(build_values(model, option, rights, core_steps=40, tolerance=0.))
    paths = simulate_paths(model, option, rights, 10, np.random.default_rng(58))
    realised = realise_reference_policy(policy, paths)
    expiry = model.reward(option, rights[-1], paths.states[:, -1])
    # An independent deterministic comparison gives 20 at the earlier right and 10 at expiry.
    np.testing.assert_allclose(realised.discounted_payoffs-expiry, 10., atol=1e-12)
    assert realised.discounted_payoffs.mean() > expiry.mean()


def test_future_path_changes_cannot_change_earlier_exercise_decisions():
    model = FundedCashModel(80., .2, 1., 1.)
    option = Option(100., 1., "put")
    rights = (FundedExercise(.25, 1.), FundedExercise(.5, .95), FundedExercise(1., .9))
    policy = ReferencePolicy(build_values(model, option, rights, core_steps=80))
    paths = simulate_paths(model, option, rights, 200, np.random.default_rng(57))
    states = paths.states.copy()
    states[:, 1:] *= 1.1
    original = realise_reference_policy(policy, paths)
    changed = realise_reference_policy(policy, replace(paths, states=states))
    np.testing.assert_array_equal(original.stopping_indices == 0, changed.stopping_indices == 0)
