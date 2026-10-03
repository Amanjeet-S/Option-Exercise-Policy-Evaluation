"""Mathematical tests of quote-space invariance and finite-family inference."""

from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import t as student_t

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise
from qf_research.optimal_stopping.funded_policy import (
    FundedPaths, FundedPolicy, ContinuationRule, realise_policy, simulate_paths, fit_policy,
)
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale, evaluate_value_dual
from qf_research.optimal_stopping.policy_transfer import realise_transferred_policy, evaluate_transferred_dual, family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples


def example():
    model = FundedCashModel(100., .2, 1., 1.)
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.5, 1.), FundedExercise(1., .98))
    rules = (ContinuationRule(100., 1., (9.,), 4, 1, 1., 80., 120., 'constant'),
             ContinuationRule(0., 1., (), 0, 0, None, None, None, 'terminal'))
    policy = FundedPolicy(model, option, rights, rules, 0, 4, 2, 1e8, 'training')
    states = np.array([[85., 80.], [89., 90.], [95., 95.], [110., 105.]])
    paths = FundedPaths(model, 1., rights, states, np.empty((4, 0)), 'evaluation')
    return policy, paths


def test_baseline_transfer_reproduces_original_stops_and_rewards_exactly():
    policy, paths = example()
    original, transferred = realise_policy(policy, paths), realise_transferred_policy(policy, paths)
    np.testing.assert_array_equal(original.stopping_indices, transferred.stopping_indices)
    np.testing.assert_array_equal(original.discounted_payoffs, transferred.discounted_payoffs)


def test_same_quote_history_preserves_decisions_while_scenario_rewards_change():
    policy, paths = example()
    factors = (.5, .7)
    rights = tuple(replace(d, settlement_factor=f) for d, f in zip(paths.decisions, factors))
    target = replace(paths, decisions=rights,
        states=paths.states*np.array([b.settlement_factor/a.settlement_factor for a,b in zip(paths.decisions,rights)]))
    baseline = realise_policy(policy, paths)
    transferred = realise_transferred_policy(policy, target)
    np.testing.assert_array_equal(baseline.stopping_indices, transferred.stopping_indices)
    # First two paths still exercise, although their scenario PV reward is halved.
    np.testing.assert_allclose(transferred.discounted_payoffs[:2], baseline.discounted_payoffs[:2]/2)
    # Naively replacing the original factors changes the predicate and misses them.
    naive = realise_policy(replace(policy, decisions=rights), target)
    assert not np.array_equal(naive.stopping_indices, transferred.stopping_indices)


def test_ties_continue_and_insufficient_fits_do_not_exercise():
    policy, paths = example()
    policy = replace(policy, rules=(replace(policy.rules[0], coefficients=(15.,)), policy.rules[-1]))
    assert realise_transferred_policy(policy, paths).stopping_indices[0] == 1
    policy = replace(policy, rules=(replace(policy.rules[0], coefficients=()), policy.rules[-1]))
    assert np.all(realise_transferred_policy(policy, paths).stopping_indices == 1)


def test_changed_clocks_or_training_stream_are_rejected():
    policy, paths = example()
    with pytest.raises(ValueError, match='identical decision clocks'):
        realise_transferred_policy(policy, replace(paths, decisions=(FundedExercise(.4,1.),paths.decisions[-1])))
    with pytest.raises(ValueError, match='Training stream'):
        realise_transferred_policy(policy, replace(paths, stream_id='training'))


def test_future_states_do_not_change_prior_exercise_choices():
    policy, paths = example()
    original = realise_transferred_policy(policy, paths)
    states = paths.states.copy(); states[:, -1] *= 3.
    changed = realise_transferred_policy(policy, replace(paths, states=states))
    np.testing.assert_array_equal(original.stopping_indices, changed.stopping_indices)


def test_transferred_dual_matches_existing_dual_on_baseline():
    model = FundedCashModel(100., .2, 1., 1.)
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.5,.99),FundedExercise(1.,.98))
    training = simulate_paths(model,option,rights,500,np.random.default_rng(1))
    policy = fit_policy(option,training,2)
    values = build_values(model,option,rights,core_steps=40,tolerance=.01)
    paths = simulate_paths(model,option,observation_rights(values),2000,np.random.default_rng(2))
    sample = value_martingale(values,paths)
    old = evaluate_value_dual(policy,sample)
    new,_ = evaluate_transferred_dual(policy,sample)
    np.testing.assert_array_equal(old.gap_payoffs,new.gap_payoffs)
    np.testing.assert_array_equal(old.adjusted_policy_payoffs,new.adjusted_policy_payoffs)


def test_bonferroni_tail_allocation_and_monotonicity():
    estimate = estimate_samples(np.array([1.,2.,3.,4.]))
    bound = family_upper_limit(estimate,9,.05)
    quantile = (bound-estimate.value)/estimate.standard_error
    assert 9*student_t.sf(quantile,3) == pytest.approx(.05)
    assert bound > family_upper_limit(estimate,1,.05)
    assert bound > estimate.upper


def test_deterministic_family_limit_does_not_add_sampling_noise():
    estimate = estimate_samples(np.ones(20)*.02)
    assert family_upper_limit(estimate,9) == pytest.approx(.02,abs=1e-15)


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_family_size_rejected(size):
    with pytest.raises(ValueError,match='positive integer'):
        family_upper_limit(estimate_samples(np.array([1.,2.])),size)
