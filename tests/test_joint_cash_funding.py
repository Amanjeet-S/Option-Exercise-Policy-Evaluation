"""Original mathematical fixtures, not sampled or fabricated market observations."""
from itertools import product
import pytest

from qf_research.market_data.joint_cash_funding import joint_call_cover, uniform_call_margin, session_cash_sets
from qf_research.market_data.dividend_feasibility import cash_state_volatility_cover
from qf_research.market_data.paired_volatility import call_quote_interval, single_cash_cap_allowance, conditional_put_envelope
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise, european_reference


def price(factors, spot, cash, sigma, kind='call'):
    model = FundedCashModel(spot, sigma, 1., 1., (DividendCap(.1, cash*factors[0]),))
    return european_reference(model, Option(100., .5, kind), FundedExercise(.5, factors[1])).premium_quote


def test_collapsed_funding_band_recovers_cash_and_state_cover():
    fixed = lambda s,d,v: price((.99,.98), s,d,v)
    args = (99.,101.,5.,6.,.2,1.8,(.01,1.),1e-9)
    old = cash_state_volatility_cover(fixed, *args)
    new = joint_call_cover(lambda side,s,d,v: fixed(s,d,v), *args)
    assert new['volatility_lower'] == old['volatility_lower']
    assert new['volatility_upper'] == old['volatility_upper']


def test_independent_funding_nodes_are_inside_call_and_put_covers():
    endpoints = {'lower':(.985,.970), 'upper':(.995,.990)}
    pricer = lambda side,s,d,v: price(endpoints[side],s,d,v)
    bid,ask = price((.99,.98),100.,1.,.2),price((.99,.98),100.,1.,.24)
    cover = joint_call_cover(pricer,99.,101.,bid,ask,.2,1.8,(.01,1.),1e-9)
    lo,hi = cover['volatility_lower'], cover['volatility_upper']
    model = FundedCashModel(99.,hi,1.,1.,(DividendCap(.1,1.8*.995),))
    cap = single_cash_cap_allowance(model,Option(100.,.5,'put'),hi)['premium_allowance']
    envelope = conditional_put_envelope(price(endpoints['lower'],101.,.2,lo,'put'),
        price(endpoints['upper'],99.,1.8,hi,'put'),cap,1e-8)
    # The two mixed corners exercise independent payment and terminal nodes.
    for factors in product((.985,.99,.995),(.970,.98,.990)):
        for spot,cash in product((99.,101.),(.2,1.8)):
            interval = call_quote_interval(lambda v:price(factors,spot,cash,v),bid,ask,(.01,1.),1e-9)
            assert lo <= interval['lower'] <= interval['upper'] <= hi
            for sigma in (interval['lower'],interval['upper']):
                assert envelope['lower'] <= price(factors,spot,cash,sigma,'put') <= envelope['upper']


def test_terminal_delivery_cancels_its_own_uncertainty_exactly():
    result = uniform_call_margin(100.,('a','b','b'),(1.,.97,.97),(1.,.98,.98),(.1,.3,.5),.2,1.,1.)
    assert result['sufficient_condition_satisfied']
    assert result['margins_lower'][1:] == [0.,0.]


def test_failed_uniform_bound_does_not_assert_exercise_advantage():
    result = uniform_call_margin(100.,('a','b'),(.97,.97),(.99,.99),(.1,.5),.2,0.,1.)
    assert not result['sufficient_condition_satisfied']
    # A flat curve within the rectangle has zero strike loss and no cash.
    flat = uniform_call_margin(100.,('a','b'),(.98,.98),(.98,.98),(.1,.5),.2,0.,1.)
    assert flat['sufficient_condition_satisfied']


def test_uniform_margin_bounds_mixed_factor_configurations():
    bound = uniform_call_margin(100.,('a','b'),(.995,.97),(1.,.98),(.1,.5),.2,1.,.997)
    for first,last,payment,cash in product((.995,1.),(.97,.98),(.99,.997),(0.,1.)):
        actual = 100*(first-last)-cash*payment
        assert bound['margins_lower'][0] <= actual+1e-14


@pytest.mark.parametrize('times,ex', [((-.1,.5),.2),((.1,.5),.6),((.5,.1),.2)])
def test_invalid_calendar_rejected(times,ex):
    with pytest.raises(ValueError):
        uniform_call_margin(100.,('a','b'),(1.,.98),(1.,.99),times,ex,1.,1.)


def test_identical_delivery_dates_cannot_have_different_factor_intervals():
    with pytest.raises(ValueError,match='Identical delivery'):
        uniform_call_margin(100.,('a','a'),(1.,.98),(1.,.99),(.1,.5),.2,1.,1.)


@pytest.mark.parametrize('low,high',[(-1.,1.),(2.,1.),(0.,float('nan'))])
def test_invalid_cash_cell_rejected(low,high):
    with pytest.raises(ValueError):
        joint_call_cover(lambda side,s,d,v:s-d+v,99.,101.,5.,6.,low,high,(.01,1.),1e-9)


def test_reversed_funding_envelope_rejected():
    with pytest.raises(ValueError,match='ordering'):
        joint_call_cover(lambda side,s,d,v:s-d+v+(10 if side=='upper' else 0),99.,101.,5.,6.,0.,1.,(.01,1.),1e-9)


def test_empty_endpoint_slices_do_not_exclude_interior_joint_fit():
    result = joint_call_cover(lambda side,s,d,v:s-d+v,10.,10.,5.,6.,0.,9.,(.1,1.),1e-8)
    assert result['status']=='candidate_cover' and result['clipped_to_volatility_range']


def test_session_intersection_preserves_gaps_and_distinct_cutoffs():
    rows = [dict(date='day',cutoff='first',strike=100.,outer_cash_set=[[0.,1.],[2.,3.]]),
            dict(date='day',cutoff='second',strike=100.,outer_cash_set=[[1.,2.]])]
    result = session_cash_sets(rows)[0]
    assert result['common_cash_set']==[[1.,1.],[2.,2.]]
    assert result['distinct_cutoffs']==2
    assert not result['simultaneous_cross_strike_sample']
    assert not result['existence_of_joint_fit_established']


def test_empty_observation_set_excludes_constant_session_cash():
    rows = [dict(date='day',cutoff='first',strike=100.,outer_cash_set=[]),
            dict(date='day',cutoff='second',strike=101.,outer_cash_set=[[0.,2.]])]
    assert session_cash_sets(rows)[0]['excluded_within_declared_domain']
    with pytest.raises(ValueError):session_cash_sets([rows[0],rows[0]])


def test_witness_summary_rejects_incomplete_candidate_coverage(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from construct_joint_funding_witnesses import summarise
    row=dict(candidate_id='fixture',date='day',fraction=.5,nominal_cash=1.,cutoff='a',strike=100.,
             expected_observations=2,status='conditional_numerical_witness')
    with pytest.raises(ValueError,match='coverage'):summarise([row],[])
    with pytest.raises(ValueError,match='coverage'):summarise([row,row],[])


def test_witness_summary_requires_every_observation_to_pass(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from construct_joint_funding_witnesses import summarise
    row=dict(candidate_id='fixture',date='day',fraction=.5,nominal_cash=1.,cutoff='a',strike=100.,
             expected_observations=2,status='conditional_numerical_witness')
    second=dict(row,cutoff='b',status='candidate_does_not_supply_put_bracket')
    result=summarise([row,second],[])
    assert not result['session_has_conditional_witness']['day']
    assert not result['same_state_across_distinct_cutoffs_imposed']
