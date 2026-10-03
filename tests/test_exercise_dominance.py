"""Analytical fixtures for a sufficient condition, not observed option prices."""

from dataclasses import replace
import math

import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    FundedCashModel, FundedExercise, DividendCap, deterministic_value, european_reference,
)
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance, coarsening_dominance


def boundary_case(cash=1., volatility=0.):
    model = FundedCashModel(100., volatility, 1., .99, (DividendCap(.5,cash),))
    option = Option(8.,1.,'call')
    rights = (FundedExercise(.25,1.125),FundedExercise(.75,1.),FundedExercise(1.,1.))
    return model, option, rights


def test_exact_sufficient_boundary_and_deterministic_value():
    model, option, rights = boundary_case()
    check = terminal_call_dominance(model,option,rights)
    assert check.sufficient_condition_satisfied and check.failed_conditions == 0
    assert check.minimum_margin_lower == 0.
    assert deterministic_value(model,option,rights) == deterministic_value(model,option,(rights[-1],))


def test_one_ulp_condition_failure_is_not_rounded_into_acceptance():
    model, option, rights = boundary_case(math.nextafter(1.,math.inf))
    check = terminal_call_dominance(model,option,rights)
    assert not check.sufficient_condition_satisfied and check.failed_conditions == 1
    assert check.margins_lower[0] < 0


@pytest.mark.parametrize('sigma',[0.,.2,1.])
def test_condition_is_independent_of_volatility(sigma):
    model, option, rights = boundary_case(volatility=sigma)
    assert terminal_call_dominance(model,option,rights).sufficient_condition_satisfied


def test_failure_does_not_imply_early_exercise_is_optimal():
    model, option, rights = boundary_case(2.)
    model = replace(model,quoted_spot=1.)
    check = terminal_call_dominance(model,option,rights)
    assert not check.sufficient_condition_satisfied
    assert deterministic_value(model,option,rights) == 0.
    assert deterministic_value(model,option,(rights[-1],)) == 0.


def test_before_and_after_event_have_different_remaining_cash():
    model, option, rights = boundary_case(1.)
    rights = (FundedExercise(.5,1.,'before'),FundedExercise(.5,1.,'after'),rights[-1])
    check = terminal_call_dominance(model,option,rights)
    assert check.margins_lower == (-1.,0.,0.)
    assert check.failed_conditions == 1


def test_post_expiry_cash_does_not_enter_the_condition():
    model, option, rights = boundary_case()
    alternative = replace(model,dividends=model.dividends+(DividendCap(2.,1000.),))
    assert terminal_call_dominance(model,option,rights) == terminal_call_dominance(alternative,option,rights)


def test_negative_rate_ordering_can_fail_even_without_dividends():
    model, option, rights = boundary_case(0.)
    rights = (FundedExercise(.25,.9),rights[-1])
    assert not terminal_call_dominance(model,option,rights).sufficient_condition_satisfied


def test_put_and_missing_terminal_right_are_rejected():
    model, option, rights = boundary_case()
    with pytest.raises(ValueError,match='calls only'):
        terminal_call_dominance(model,replace(option,kind='put'),rights)
    with pytest.raises(ValueError,match='terminal'):
        terminal_call_dominance(model,option,rights[:-1])


@pytest.mark.parametrize('kind',['call','put'])
def test_identical_factors_without_intervening_cash_allow_coarsening(kind):
    model = FundedCashModel(100.,.2,1.,1.,(DividendCap(.5,2.),))
    option = Option(100.,1.,kind)
    coarse = (FundedExercise(.4,.99),FundedExercise(1.,.98))
    fine = (FundedExercise(.2,.99),coarse[0],FundedExercise(.5,.98),FundedExercise(.8,.98),coarse[1])
    result = coarsening_dominance(model,option,coarse,fine)
    assert result.sufficient_condition_satisfied and result.added_decisions == 3
    # The event at the added opening right has already occurred; it is not a
    # jump between that post-event state and the following close.
    assert result.intervals_crossing_cash_events == 0


def test_jump_between_added_right_and_later_close_is_not_ignored():
    model, option, rights = boundary_case()
    coarse = (rights[-1],)
    fine = (FundedExercise(.25,1.),rights[-1])
    result = coarsening_dominance(model,option,coarse,fine)
    assert not result.sufficient_condition_satisfied
    assert result.intervals_crossing_cash_events == 1


def test_varying_funding_can_make_earlier_exercise_strictly_better():
    model = FundedCashModel(100.,0.,1.,1.)
    option = Option(100.,1.,'call')
    coarse = (FundedExercise(1.,1.),)
    fine = (FundedExercise(.5,.9),coarse[0])
    result = coarsening_dominance(model,option,coarse,fine)
    assert not result.sufficient_condition_satisfied and result.factor_mismatches == 1
    assert deterministic_value(model,option,fine) > deterministic_value(model,option,coarse)


def test_non_nested_or_pre_event_sets_are_rejected():
    model, option, rights = boundary_case()
    with pytest.raises(ValueError,match='subset'):
        coarsening_dominance(model,option,rights,(rights[-1],))
    before = (FundedExercise(.5,1.,'before'),rights[-1])
    with pytest.raises(ValueError,match='post-event'):
        coarsening_dominance(model,option,before,before)
