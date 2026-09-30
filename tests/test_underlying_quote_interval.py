"""Analytical fixtures for a covering set, not observed quote data."""

from dataclasses import replace

import pytest

from qf_research.market_data.underlying_quote_interval import call_state_rectangle
from qf_research.market_data.paired_volatility import (
    call_quote_interval, conditional_put_envelope, single_cash_cap_allowance,
)
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.funded_cash import (
    FundedCashModel, FundedExercise, DividendCap, european_reference,
)


def price(spot, sigma):
    return black_scholes(Market(spot=spot, rate=.03, volatility=sigma), Option(100., .5, 'call'))


def test_covering_rectangle_contains_all_analytical_state_slices():
    bid, ask = price(100., .2), price(100., .24)
    box = call_state_rectangle(price, 99., 101., bid, ask, (.01, 1.), 1e-9)
    assert box['status'] == 'candidate_rectangle'
    for state in [99., 99.5, 100., 100.5, 101.]:
        interval = call_quote_interval(lambda s: price(state, s), bid, ask, (.01, 1.), 1e-9)
        assert box['volatility_lower'] <= interval['lower']
        assert box['volatility_upper'] >= interval['upper']


def test_zero_underlying_spread_reduces_to_fixed_state_inversion():
    bid, ask = price(100., .2), price(100., .24)
    box = call_state_rectangle(price, 100., 100., bid, ask, (.01, 1.), 1e-9)
    interval = call_quote_interval(lambda s: price(100., s), bid, ask, (.01, 1.), 1e-9)
    assert (box['volatility_lower'], box['volatility_upper']) == (interval['lower'], interval['upper'])


@pytest.mark.parametrize('bid,ask', [(20., 21.), (1., 2.)])
def test_no_state_can_recover_out_of_range_option_quotes(bid, ask):
    box = call_state_rectangle(lambda y,s: y+s, 10., 11., bid, ask, (.1, 1.), 1e-9)
    assert box['status'] == 'no_call_candidate_in_declared_rectangle'


def test_no_endpoint_fit_does_not_imply_no_interior_state_fit():
    # At low state all prices are below the quote; at high state all are above.
    box = call_state_rectangle(lambda y,s: y+s, 10., 20., 15., 16., (.1, 1.), 1e-9)
    assert box['status'] == 'candidate_rectangle' and box['clipped_to_volatility_range']
    assert (box['volatility_lower'], box['volatility_upper']) == (.1, 1.)


@pytest.mark.parametrize('sb,sa,cb,ca', [(0.,1.,1.,2.),(2.,1.,1.,2.),(1.,2.,2.,1.),(1.,2.,1.,1.),(1.,float('nan'),1.,2.)])
def test_invalid_quote_intervals(sb,sa,cb,ca):
    with pytest.raises(ValueError):call_state_rectangle(price,sb,sa,cb,ca,(.01,1.),1e-9)


def test_decreasing_state_pricer_is_rejected():
    with pytest.raises(ValueError,match='increasing-state'):
        call_state_rectangle(lambda y,s:30.-y+s,10.,11.,19.,20.,(.1,1.),1e-9)


def test_flat_volatility_pricer_is_rejected():
    with pytest.raises(ValueError,match='strictly increasing'):
        call_state_rectangle(lambda y,s:y,10.,11.,10.1,10.2,(.1,1.),1e-9)


def test_cash_model_rectangle_put_bound_covers_feasible_state_slices():
    model = FundedCashModel(100., .2, .999, .998, (DividendCap(.1, 2.),))
    call, put = Option(100., .5, 'call'), Option(100., .5, 'put')
    terminal = FundedExercise(.5, .985)
    def quoted(kind, state, sigma):
        return european_reference(replace(model,quoted_spot=state,volatility=sigma),kind,terminal).premium_quote
    bid, ask = quoted(call,100.,.2), quoted(call,100.,.24)
    box = call_state_rectangle(lambda y,s:quoted(call,y,s),99.,101.,bid,ask,(.01,1.),1e-9)
    lo,hi = box['volatility_lower'],box['volatility_upper']
    cap = single_cash_cap_allowance(replace(model,quoted_spot=99.),put,hi)['premium_allowance']
    envelope = conditional_put_envelope(quoted(put,101.,lo),quoted(put,99.,hi),cap,1e-8)
    for state in [99.,100.,101.]:
        interval = call_quote_interval(lambda s:quoted(call,state,s),bid,ask,(.01,1.),1e-9)
        for sigma in [interval['lower'],interval['midpoint_volatility'],interval['upper']]:
            assert envelope['lower'] <= quoted(put,state,sigma) <= envelope['upper']


def test_enlarging_observed_state_range_cannot_shrink_the_cover():
    bid,ask = price(100.,.2),price(100.,.24)
    narrow = call_state_rectangle(price,99.9,100.1,bid,ask,(.01,1.),1e-9)
    wide = call_state_rectangle(price,99.,101.,bid,ask,(.01,1.),1e-9)
    assert wide['volatility_lower'] <= narrow['volatility_lower']
    assert wide['volatility_upper'] >= narrow['volatility_upper']
