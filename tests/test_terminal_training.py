"""Matched training and admissible evaluation of a changed terminal target."""

from dataclasses import replace
import numpy as np
import pytest
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy
from qf_research.optimal_stopping.terminal_training import fit_terminal_policy
from qf_research.optimal_stopping.terminal_repair import evaluate_terminal_repair
from qf_research.optimal_stopping.expiry_extension import extended_calendar
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale


def fixture(volatility=.2, terminal_only=False):
    model=FundedCashModel(100.,volatility,1.,1.)
    option=Option(100.,1.,'put')
    rights=(FundedExercise(1.,1.),) if terminal_only else (FundedExercise(.5,1.01),FundedExercise(1.,1.))
    paths=simulate_paths(model,option,rights,2000,np.random.default_rng(41))
    return model,option,rights,paths


@pytest.mark.parametrize('degree',[0,2,4])
def test_zero_extension_reproduces_original_policy_fingerprint(degree):
    _,option,_,paths=fixture()
    old=fit_policy(option,paths,degree)
    new=fit_terminal_policy(option,paths,degree,0.)
    assert new.policy == old
    assert new.policy.fingerprint == old.fingerprint


def test_positive_extension_changes_earlier_regressions_on_unchanged_paths():
    _,option,_,paths=fixture()
    frozen=paths.states.copy()
    old=fit_terminal_policy(option,paths,2,0.)
    new=fit_terminal_policy(option,paths,2,.1)
    assert new.policy.rules[0].coefficients != old.policy.rules[0].coefficients
    assert new.policy.training_stream_id == old.policy.training_stream_id
    assert new.policy.decisions == old.policy.decisions
    np.testing.assert_array_equal(paths.states,frozen)


def test_zero_volatility_leaves_the_regressions_unchanged():
    _,option,_,paths=fixture(volatility=0.)
    assert fit_terminal_policy(option,paths,2,.1).policy == fit_policy(option,paths,2)


def test_terminal_only_instructions_are_identical_but_horizon_fingerprint_differs():
    _,option,_,paths=fixture(terminal_only=True)
    old=fit_terminal_policy(option,paths,2,0.)
    new=fit_terminal_policy(option,paths,2,.1)
    assert old.policy == new.policy
    assert old.fingerprint != new.fingerprint


@pytest.mark.parametrize('extension',[-.1,float('nan'),True])
def test_invalid_extension_rejected(extension):
    _,option,_,paths=fixture()
    with pytest.raises(ValueError): fit_terminal_policy(option,paths,2,extension)


def test_new_cash_event_rejected():
    model,option,_,paths=fixture()
    changed=replace(paths,model=replace(model,dividends=(DividendCap(1.05,1.),)))
    with pytest.raises(ValueError,match='no cash transfer'):
        fit_terminal_policy(option,changed,2,.1)


def test_training_and_evaluation_roles_remain_separate():
    model,option,rights,paths=fixture()
    trained=fit_terminal_policy(option,paths,2,.1)
    longer,calendar=extended_calendar(model,option,rights,(.1,))
    values=build_values(model,longer,calendar,core_steps=80)
    draws=simulate_paths(model,longer,observation_rights(values),1000,np.random.default_rng(42))
    sample=value_martingale(values,draws)
    result=evaluate_terminal_repair(trained.policy,sample,trained.extension)
    assert result.repaired.gap_payoffs.min()>=0
    changed=replace(sample,paths=replace(sample.paths,stream_id=trained.policy.training_stream_id))
    with pytest.raises(ValueError,match='Training'):
        evaluate_terminal_repair(trained.policy,changed,trained.extension)


def test_final_draw_does_not_change_retrained_earlier_decisions():
    model,option,rights,paths=fixture()
    trained=fit_terminal_policy(option,paths,2,.1)
    longer,calendar=extended_calendar(model,option,rights,(.1,))
    values=build_values(model,longer,calendar,core_steps=80)
    draws=simulate_paths(model,longer,observation_rights(values),1000,np.random.default_rng(42))
    sample=value_martingale(values,draws)
    first=evaluate_terminal_repair(trained.policy,sample,.1)
    states=sample.paths.states.copy();states[:,-1]*=2
    second=evaluate_terminal_repair(trained.policy,replace(sample,paths=replace(sample.paths,states=states)),.1)
    np.testing.assert_array_equal(first.stopping_indices,second.stopping_indices)
    np.testing.assert_array_equal(first.repaired.gap_payoffs,second.repaired.gap_payoffs)
