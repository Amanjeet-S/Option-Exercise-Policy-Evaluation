"""Analytical fixtures for inversion and comparison bounds, not market observations."""

from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path
import sys

import pytest

from qf_research.market_data.paired_volatility import (
    increasing_price_bracket, call_quote_interval, single_cash_cap_allowance,
    conditional_put_envelope,
)
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.funded_cash import (
    FundedCashModel, DividendCap, FundedExercise, european_reference,
)


def analytical_call(sigma):
    return black_scholes(Market(spot=100., rate=.03, dividend=.01, volatility=sigma), Option(102., .7, 'call'))


def test_analytical_parameter_recovery_retains_outer_bracket():
    sigma = .237
    target = analytical_call(sigma)
    root = increasing_price_bracket(analytical_call, target, .01, 1., 1e-9)
    assert root['lower'] <= sigma <= root['upper']
    assert root['upper']-root['lower'] <= 1e-9
    assert root['lower_price'] <= target <= root['upper_price']


def test_quote_interval_contains_entire_analytical_compatible_range():
    result = call_quote_interval(analytical_call, analytical_call(.2), analytical_call(.25), (.01,1.), 1e-9)
    assert result['lower'] <= .2 < .25 <= result['upper']
    assert result['midpoint_target_bracketed'] and not result['clipped_to_volatility_range']
    assert set(result['roots']) == {'bid','midpoint','ask'}


@pytest.mark.parametrize('bid,ask,status',[(.1,.2,'below_range'),(3.,4.,'above_range')])
def test_nonexistent_interval_is_not_reported_as_a_fit(bid,ask,status):
    result = call_quote_interval(lambda x:x, bid,ask,(.5,2.),1e-9)
    assert result['status'] == 'no_quote_compatible_volatility_in_declared_range'
    assert result['roots']['midpoint']['status'] == status


def test_partly_clipped_interval_retains_boundary_and_midpoint_status():
    result = call_quote_interval(lambda x:x, .1,.7,(.5,2.),1e-9)
    assert result['lower'] == .5 and result['upper'] >= .7
    assert result['clipped_to_volatility_range'] and not result['midpoint_target_bracketed']


@pytest.mark.parametrize('pricer',[lambda x:1.,lambda x:2.-x,lambda x:float('nan'),lambda x:-1.])
def test_flat_decreasing_or_invalid_prices_are_rejected(pricer):
    with pytest.raises(ValueError):
        increasing_price_bracket(pricer,.5,.1,1.,1e-9)


def test_interior_contradiction_is_rejected():
    with pytest.raises(ValueError,match='Evaluated prices'):
        increasing_price_bracket(lambda x:5. if x == .5 else x,.4,0.,1.,1e-9)


@pytest.mark.parametrize('target,low,high,width',[(float('nan'),0.,1.,1e-8),(-1.,0.,1.,1e-8),(.5,1.,0.,1e-8),(.5,0.,1.,0.)])
def test_invalid_inversion_arguments(target,low,high,width):
    with pytest.raises(ValueError):increasing_price_bracket(lambda x:x,target,low,high,width)


@pytest.mark.parametrize('bid,ask',[(1.,1.),(-1.,2.),(1.,float('inf'))])
def test_invalid_quote_interval(bid,ask):
    with pytest.raises(ValueError):call_quote_interval(lambda x:x,bid,ask,(0.,1.),1e-8)


@pytest.mark.parametrize('spot,cash,sigma,time',[(100.,2.,.2,.1),(100.,80.,.35,.2),(1.,2.,.4,.25),(100.,90.,2.,1.)])
def test_cap_bound_dominates_exact_lognormal_shortfall(spot,cash,sigma,time):
    model = FundedCashModel(spot,sigma,1.,.99,(DividendCap(time,cash),))
    result = single_cash_cap_allowance(model,Option(100.,time+1.,'put'),sigma)
    exact = black_scholes(Market(spot=spot,rate=0.,dividend=0.,volatility=sigma),Option(cash,time,'put'))/.99
    assert exact <= result['premium_allowance']+1e-13
    assert result['premium_allowance'] <= math.nextafter(cash/.99,math.inf)


def test_rational_bound_is_rounded_outwards_in_premium_units():
    model = FundedCashModel(103.,.2,.999,.998,(DividendCap(.07,3.1),))
    result = single_cash_cap_allowance(model,Option(100.,.3,'put'),.2)
    c,y,e,s,fp = map(Fraction,(3.1,model.initial_state,.07,.2,.998))
    rational = c**5/y**4/(1-10*s*s*e)/fp
    assert Fraction(result['premium_allowance']) >= rational
    assert Fraction(math.nextafter(result['premium_allowance'],-math.inf)) < rational


def test_zero_volatility_and_no_cash_limits():
    model = FundedCashModel(1.,0.,1.,.5,(DividendCap(.5,3.),))
    option = Option(1.,1.,'put')
    assert single_cash_cap_allowance(model,option,0.)['premium_allowance'] == 4.
    assert single_cash_cap_allowance(replace(model,dividends=()),option,1.)['premium_allowance'] == 0.
    assert single_cash_cap_allowance(replace(model,dividends=(DividendCap(2.,3.),)),option,1.)['premium_allowance'] == 0.


def test_multiple_cash_events_and_invalid_volatility_are_rejected():
    model = FundedCashModel(100.,.2,1.,1.,(DividendCap(.2,1.),DividendCap(.4,1.)))
    option = Option(100.,1.,'put')
    with pytest.raises(ValueError,match='at most one'):single_cash_cap_allowance(model,option,.2)
    with pytest.raises(ValueError,match='volatility'):single_cash_cap_allowance(model,option,-.2)


def test_uncapped_european_comparison_under_material_cap_risk():
    # The auxiliary negative-state model has the same call payoff pathwise.
    # Its put follows from linear expectation and European parity.
    model = FundedCashModel(80.,.7,1.,.99,(DividendCap(.5,90.),))
    call,put = Option(100.,1.,'call'),Option(100.,1.,'put')
    terminal = FundedExercise(1.,.98)
    c,p = (european_reference(model,o,terminal).premium_quote for o in (call,put))
    auxiliary_put = c+(-model.initial_state+90.+100.*.98)/.99
    shortfall = black_scholes(Market(spot=80.,rate=0.,dividend=0.,volatility=.7),Option(90.,.5,'put'))/.99
    assert auxiliary_put-p == pytest.approx(shortfall,abs=2e-9)
    assert auxiliary_put >= p
    assert auxiliary_put-p <= single_cash_cap_allowance(model,put,.7)['premium_allowance']


def test_envelope_contains_independent_intermediate_european_values():
    model = FundedCashModel(100.,.2,1.,.99,(DividendCap(.1,2.),))
    option,terminal = Option(101.,.7,'put'),FundedExercise(.7,.98)
    price = lambda s:european_reference(replace(model,volatility=s),option,terminal).premium_quote
    cap = single_cash_cap_allowance(model,option,.4)['premium_allowance']
    envelope = conditional_put_envelope(price(.1),price(.4),cap,1e-8)
    for sigma in [.1,.15,.25,.35,.4]:
        assert envelope['lower'] <= price(sigma) <= envelope['upper']


def test_envelope_outward_rounding_and_zero_floor():
    result = conditional_put_envelope(.1,.2,.003,.004)
    exact_error = Fraction(.003)+Fraction(.004)
    assert Fraction(result['lower']) <= Fraction(.1)-exact_error
    assert Fraction(result['upper']) >= Fraction(.2)+exact_error
    assert conditional_put_envelope(.001,.002,.1,.1)['lower'] == 0.


@pytest.mark.parametrize('values',[(1.,2.,-.1,.1),(1.,2.,.1,float('nan')),(3.,1.,.1,.1)])
def test_invalid_or_contradictory_envelope(values):
    with pytest.raises(ValueError):conditional_put_envelope(*values)


def pairing_fixture():
    """Minimal schema fixture; no market observation or calibrated value."""
    base = {'dividend_model':'mathematical_case','model':{'state':100.},'decisions':[],
            'forecast':{},'curve_date':'fixture','premium_payment_date':'fixture','split':'fixture'}
    rows = [dict(base,observation_id=side,option={'kind':side,'strike':100.,'maturity':1.})
            for side in ('call','put')]
    source = {'observation_date':'fixture','expiry':'fixture','cutoff':'fixture',
              'quoted_spot':100.,'underlying_bid':99.,'underlying_ask':101.}
    return rows,{side:dict(source) for side in ('call','put')}


def pair_records(rows,sources):
    scripts = str(Path(__file__).resolve().parents[1]/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    from diagnose_paired_volatility import paired_inputs
    return paired_inputs(rows,sources)


def test_matching_retains_missing_sides():
    rows,sources = pairing_fixture()
    pairs,unmatched = pair_records(rows,sources)
    assert len(pairs) == 1 and not unmatched
    pairs,unmatched = pair_records(rows[:1],sources)
    assert not pairs and unmatched[0]['retained_sides'] == ['call']


@pytest.mark.parametrize('field,value',[('model',{'state':101.}),('decisions',[1]),('forecast',{'cash':2.}),('split','different')])
def test_different_economic_inputs_cannot_be_paired(field,value):
    rows,sources = pairing_fixture()
    rows[1][field] = value
    with pytest.raises(ValueError,match='economic inputs'):pair_records(rows,sources)


def test_different_observed_states_and_duplicate_sides_are_rejected():
    rows,sources = pairing_fixture()
    sources['put']['quoted_spot'] = 101.
    with pytest.raises(ValueError,match='observed states'):pair_records(rows,sources)
    rows,sources = pairing_fixture()
    with pytest.raises(ValueError,match='Duplicate'):pair_records(rows+rows[:1],sources)
