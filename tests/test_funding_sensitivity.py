"""Declared funding fixtures and a holiday-free test calendar, not market data."""

from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
import json
import math

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.funding_sensitivity import rebuild_funding, lower_corner_components, component_changes
from qf_research.market_data.valuation_inputs import DiscountCurve


def curve(rate=.04, anchor=date(2026,9,9)):
    end=date(2027,9,9)
    return DiscountCurve(anchor,(anchor,end),(1.,math.exp(-rate*(end-anchor).days/365)))


@pytest.fixture
def fixture():
    cal=EventCalendar(date(2026,9,1),date(2027,9,9),date(2027,9,9),frozenset(),frozenset(),())
    observation=datetime(2026,9,10,15,45,tzinfo=NEW_YORK)
    source={'observation_id':'mathematical_fixture','cutoff':observation.isoformat(),
            'observation_date':'2026-09-10','expiry':'2026-10-16','quoted_spot':100.,'strike':103.,'kind':'put'}
    dated=build_dated_problem(cal,curve(),observation,date(2026,10,16),100.,103.,.2,'put',
        (DatedDividend(date(2026,9,18),date(2026,10,30),2.,'Mathematical fixture'),),premium_payment_date=date(2026,9,11))
    raw={'observation_id':source['observation_id'],'dividend_model':'fixture','split':'train','bid':2.,'ask':2.1,
         'model':asdict(dated.model),'option':asdict(dated.option),'decisions':[asdict(d) for d in dated.decisions],
         'forecast':{'as_of':source['cutoff'],'currency':'USD','latest_input_publication_bound':'2026-09-01T00:00:00-04:00',
                     'ex_date':'2026-09-18','pay_date':'2026-10-30','regular_dividend_estimate':'2.0'},
         'curve_date':'2026-09-09','premium_payment_date':'2026-09-11'}
    return json.loads(json.dumps(raw)),source,cal


def test_original_curve_reconstructs_all_inputs_exactly(fixture):
    raw,source,cal=fixture
    assert rebuild_funding(raw,source,cal,curve())==raw


@pytest.mark.parametrize('rate',[.09,-.03])
def test_curve_changes_every_funding_component_and_preserves_nominal_inputs(fixture,rate):
    raw,source,cal=fixture; original=deepcopy(raw);new=curve(rate)
    changed=rebuild_funding(raw,source,cal,new)
    assert changed['model']['initial_share_factor']==pytest.approx(math.exp(-rate/365))
    assert changed['model']['premium_factor']==pytest.approx(math.exp(-rate/365))
    assert changed['model']['dividends'][0]['discounted_cash']==pytest.approx(2*math.exp(-rate*50/365))
    assert changed['decisions'][-1]['settlement_factor']==pytest.approx(new.between(date(2026,9,10),date(2026,10,19)))
    for key in ('bid','ask','option','forecast','premium_payment_date','split'):assert changed[key]==raw[key]
    assert raw==original


def test_observation_date_curve_needs_explicit_opt_in(fixture):
    raw,source,cal=fixture;new=curve(anchor=date(2026,9,10))
    with pytest.raises(ValueError,match='retrospective'):rebuild_funding(raw,source,cal,new)
    assert rebuild_funding(raw,source,cal,new,allow_observation_date=True)['curve_date']=='2026-09-10'


@pytest.mark.parametrize('change',['identity','nominal_event_date','future_publication','exercise_time'])
def test_mismatched_contract_or_information_cannot_enter_comparison(fixture,change):
    raw,source,cal=fixture
    if change=='identity':source['strike']+=1
    elif change=='nominal_event_date':raw['forecast']['ex_date']='2026-09-21'
    elif change=='future_publication':raw['forecast']['latest_input_publication_bound']='2026-09-11T00:00:00-04:00'
    else:raw['decisions'][0]['time']+=.001
    with pytest.raises(ValueError):rebuild_funding(raw,source,cal,curve())


def test_future_curve_rejected_even_with_retrospective_opt_in(fixture):
    with pytest.raises(ValueError):rebuild_funding(*fixture,curve(anchor=date(2026,9,11)),allow_observation_date=True)


def test_curve_cannot_extrapolate_to_a_missing_payment_date(fixture):
    short=DiscountCurve(date(2026,9,9),(date(2026,9,9),date(2026,10,20)),(1.,.99))
    with pytest.raises(ValueError,match='extrapolation'):rebuild_funding(*fixture,short)


def parts(strike=99.,put=3.5,cap=.02,allowance=.08):
    return lower_corner_components(call_premium=2.,ownership_premium=100.,terminal_strike_premium=strike,
        expected_cash_premium=2.,put_premium=put,cap_allowance=cap,numerical_allowance=allowance)


def test_funding_decomposition_identifies_an_exact_strike_factor_change():
    result=component_changes(parts(),parts(strike=98.,put=2.5))
    assert result['lower_bound_change']==pytest.approx(-1.)
    assert result['changes']['terminal_strike_funding']==-1.
    assert result['changes']['numerical_put_exercise_premium']==0.
    assert abs(result['identity_residual'])<1e-14


def test_zero_floor_is_retained_in_the_identity():
    target=parts(put=.02,cap=.03,allowance=.08)
    assert target['conditional_lower_bound']==0.
    assert target['components']['zero_floor']==pytest.approx(.09)
    assert sum(target['components'].values())==pytest.approx(0.,abs=1e-14)
    assert abs(component_changes(parts(),target)['identity_residual'])<1e-14


def test_numerical_early_premium_is_not_silently_clamped():
    assert parts(put=2.9)['components']['numerical_put_exercise_premium']<0


def test_nonfinite_decomposition_rejected():
    with pytest.raises(ValueError):parts(put=float('nan'))


def test_altered_component_identity_is_rejected():
    changed=parts();changed['components']['terminal_strike_funding']+=1
    with pytest.raises(ValueError,match='does not close'):component_changes(parts(),changed)
