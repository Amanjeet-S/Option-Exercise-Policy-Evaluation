"""Sufficient waiting conditions on declared mathematical exercise problems."""
import pytest
from qf_research.optimal_stopping.exercise_redundancy import same_settlement_redundancy
from qf_research.optimal_stopping.funded_cash import FundedCashModel,FundedExercise,DividendCap


def model(events=()):
    return FundedCashModel(100.,.2,1.,1.,events)


def test_equal_settlement_and_no_cash_certify_both_rewards():
    base=(FundedExercise(.5,.99),FundedExercise(1.,.98))
    added=(FundedExercise(.25,.99),base[0],FundedExercise(.75,.98),base[1])
    result=same_settlement_redundancy(model(),base,added)
    assert result['sufficient_condition_satisfied'] and result['added_rights']==2
    assert result['applies_to']==['call','put']


def test_post_event_opening_does_not_include_transfer_already_detached():
    base=(FundedExercise(.5,.99),);added=(FundedExercise(.2,.99),*base)
    assert same_settlement_redundancy(model((DividendCap(.2,2.),)),base,added)['sufficient_condition_satisfied']


@pytest.mark.parametrize('event_time',[.3,.5])
def test_transfer_before_or_at_target_close_leaves_claim_unresolved(event_time):
    base=(FundedExercise(.5,.99),);added=(FundedExercise(.2,.99),*base)
    result=same_settlement_redundancy(model((DividendCap(event_time,2.),)),base,added)
    assert not result['sufficient_condition_satisfied']
    assert result['checks'][0]['same_settlement_factor']


def test_zero_cash_transfer_is_irrelevant():
    base=(FundedExercise(.5,.99),);added=(FundedExercise(.2,.99),*base)
    assert same_settlement_redundancy(model((DividendCap(.3,0.),)),base,added)['sufficient_condition_satisfied']


def test_changed_delivery_factor_does_not_receive_certificate():
    base=(FundedExercise(.5,.99),);added=(FundedExercise(.2,1.),*base)
    assert not same_settlement_redundancy(model(),base,added)['sufficient_condition_satisfied']


@pytest.mark.parametrize('case',['before','duplicate','missing','later_terminal'])
def test_invalid_or_changed_original_schedule_rejected(case):
    base=(FundedExercise(.5,.99),FundedExercise(1.,.98))
    added=(FundedExercise(.2,.99),*base)
    if case=='before':added=(FundedExercise(.2,.99,'before'),*base)
    if case=='duplicate':added=(added[0],added[0],*base)
    if case=='missing':added=(added[0],base[-1])
    if case=='later_terminal':added=(*added,FundedExercise(1.1,.98))
    with pytest.raises(ValueError):same_settlement_redundancy(model(),base,added)


def test_no_added_rights_is_exactly_the_same_problem():
    base=(FundedExercise(.5,.99),)
    r=same_settlement_redundancy(model(),base,base)
    assert r['sufficient_condition_satisfied'] and r['added_rights']==0
