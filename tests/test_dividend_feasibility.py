"""Cash/state covering arguments on labelled mathematical fixtures."""

from dataclasses import replace
import pytest

from qf_research.market_data.dividend_feasibility import (
    cash_state_volatility_cover, merge_closed_intervals, intersect_interval_unions,
)
from qf_research.market_data.paired_volatility import call_quote_interval, single_cash_cap_allowance, conditional_put_envelope
from qf_research.market_data.underlying_quote_interval import call_state_rectangle
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise, european_reference


def fixture_price(spot,cash,sigma,kind='call'):
    model=FundedCashModel(spot,sigma,.999,.998,(DividendCap(.1,cash),))
    return european_reference(model,Option(100.,.5,kind),FundedExercise(.5,.98)).premium_quote


def test_zero_cash_width_reduces_to_existing_state_cover():
    bid,ask=fixture_price(100.,1.,.2),fixture_price(100.,1.,.24)
    cover=cash_state_volatility_cover(fixture_price,99.,101.,bid,ask,1.,1.,(.01,1.),1e-9)
    earlier=call_state_rectangle(lambda s,v:fixture_price(s,1.,v),99.,101.,bid,ask,(.01,1.),1e-9)
    assert (cover['volatility_lower'],cover['volatility_upper'])==(earlier['volatility_lower'],earlier['volatility_upper'])


def test_joint_cover_contains_all_cash_and_state_slices():
    bid,ask=fixture_price(100.,1.,.2),fixture_price(100.,1.,.24)
    cover=cash_state_volatility_cover(fixture_price,99.,101.,bid,ask,.2,1.8,(.01,1.),1e-9)
    for cash in (.2,1.,1.8):
        for spot in (99.,100.,101.):
            interval=call_quote_interval(lambda v:fixture_price(spot,cash,v),bid,ask,(.01,1.),1e-9)
            assert cover['volatility_lower']<=interval['lower']<=interval['upper']<=cover['volatility_upper']
    model=FundedCashModel(99.,cover['volatility_upper'],.999,.998,(DividendCap(.1,1.8),))
    allowance=single_cash_cap_allowance(model,Option(100.,.5,'put'),cover['volatility_upper'])['premium_allowance']
    envelope=conditional_put_envelope(fixture_price(101.,.2,cover['volatility_lower'],'put'),
        fixture_price(99.,1.8,cover['volatility_upper'],'put'),allowance,1e-8)
    for cash in (.2,1.,1.8):
        for spot in (99.,100.,101.):
            interval=call_quote_interval(lambda v:fixture_price(spot,cash,v),bid,ask,(.01,1.),1e-9)
            for sigma in (interval['lower'],interval['midpoint_volatility'],interval['upper']):
                assert envelope['lower']<=fixture_price(spot,cash,sigma,'put')<=envelope['upper']


def test_wider_cash_range_cannot_shrink_cover():
    args=(fixture_price,99.,101.,5.,6.)
    narrow=cash_state_volatility_cover(*args,.5,1.,(.01,1.),1e-9)
    wide=cash_state_volatility_cover(*args,0.,2.,(.01,1.),1e-9)
    assert wide['volatility_lower']<=narrow['volatility_lower']<=narrow['volatility_upper']<=wide['volatility_upper']


def test_empty_endpoint_slice_does_not_exclude_interior_cash():
    # At either cash endpoint no sigma fits, while interior cash can fit.
    c=cash_state_volatility_cover(lambda s,d,v:s-d+v,10.,10.,5.,6.,0.,9.,(.1,1.),1e-8)
    assert c['status']=='candidate_cover' and c['clipped_to_volatility_range']


@pytest.mark.parametrize('bid,ask',[(20.,21.),(.01,.02)])
def test_call_exclusion_uses_the_entire_joint_domain(bid,ask):
    c=cash_state_volatility_cover(lambda s,d,v:s-d+v,10.,11.,bid,ask,0.,1.,(.1,1.),1e-8)
    assert c['status']=='no_call_candidate'


@pytest.mark.parametrize('low,high',[(-1.,1.),(2.,1.),(0.,float('nan'))])
def test_invalid_cash_domain_rejected(low,high):
    with pytest.raises(ValueError):cash_state_volatility_cover(fixture_price,99.,101.,5.,6.,low,high,(.01,1.),1e-9)


def test_wrong_cash_ordering_is_rejected():
    with pytest.raises(ValueError,match='decreasing-cash'):
        cash_state_volatility_cover(lambda s,d,v:s+d+v,99.,101.,100.,102.,0.,2.,(.1,1.),1e-9)


def test_union_does_not_fill_gaps_or_drop_boundary_points():
    assert merge_closed_intervals([(1.,2.),(0.,1.),(3.,3.),(4.,5.)])==[[0.,2.],[3.,3.],[4.,5.]]
    assert intersect_interval_unions([[(0.,1.),(2.,3.)],[(1.,2.)]])==[[1.,1.],[2.,2.]]


def test_one_empty_strike_set_excludes_common_cash():
    assert intersect_interval_unions([[(0.,2.)],[],[(1.,3.)]])==[]


@pytest.mark.parametrize('unions',[[], [[(2.,1.)]], [[(0.,float('inf'))]]])
def test_invalid_intersection_input_rejected(unions):
    with pytest.raises(ValueError):intersect_interval_unions(unions)
