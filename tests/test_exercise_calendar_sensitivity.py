"""Actual calendar dates with explicitly mathematical cash and price fixtures."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone, timedelta
import json
import math
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK, act365f
from qf_research.market_data.exercise_calendar_sensitivity import (
    core_session_schedules, descriptive_strike_slope, paired_quote_decomposition,
)
from qf_research.market_data.funded_inputs import build_dated_problem, DatedDividend
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.funded_cash import finite_difference_price, european_reference

ROOT = Path(__file__).resolve().parents[1]


def fixture(day=date(2026, 9, 17), expiry=date(2026, 9, 21), kind='put', dividends=()):
    calendar = EventCalendar.from_config(json.loads((ROOT/'projects/01_optimal_stopping/configs/us_event_calendar_2026-09-16.json').read_text()))
    observation = datetime(day.year, day.month, day.day, 12, 45, tzinfo=NEW_YORK)
    curve = DiscountCurve(day, (day, date(2027, 9, 17)), (1., .95))
    problem = build_dated_problem(calendar, curve, observation, expiry, 100., 100., .2, kind,
                                  dividends, premium_payment_date=calendar.settlement_after(day).day)
    return calendar, observation, expiry, problem


def test_nesting_preserves_factors_and_every_original_right():
    cal, obs, expiry, p = fixture()
    result = core_session_schedules(cal, obs, expiry, p.decisions)
    assert set(result['terminal_only']) < set(result['core_close']) < set(result['core_60_minutes']) < set(result['core_30_minutes'])
    by_date = {close.date(): r.settlement_factor for close, r in zip(cal.research_decisions(obs, expiry), p.decisions)}
    for r in result['core_30_minutes']:
        stamp = (obs.astimezone(timezone.utc)+timedelta(seconds=round(r.time*365*86400))).astimezone(NEW_YORK)
        session = cal.session(stamp.date())
        assert obs < stamp <= session.closes_at and stamp >= session.opens_at
        assert r.settlement_factor == by_date[stamp.date()]
    assert result['core_close'] is p.decisions


def test_no_pre_dividend_opening_right_and_overnight_gap_preserved():
    div = DatedDividend(date(2026, 9, 18), date(2026, 10, 30), 2., 'Mathematical validation cash')
    cal, obs, expiry, p = fixture(dividends=(div,))
    rights = core_session_schedules(cal, obs, expiry, p.decisions)['core_30_minutes']
    event = p.model.dividends[0].ex_time
    assert next(r for r in rights if r.time == event).phase == 'after'
    assert all(r.phase == 'after' for r in rights)
    previous = max(r.time for r in rights if r.time < event)
    assert (event-previous)*365*24 == pytest.approx(17.5)


def test_early_close_and_dst_use_source_dates_and_elapsed_utc_time():
    cal, obs, expiry, p = fixture(day=date(2026, 10, 30), expiry=date(2026, 11, 27))
    rights = core_session_schedules(cal, obs, expiry, p.decisions)['core_30_minutes']
    terminal = cal.session(expiry).closes_at
    assert terminal.hour == 13 and rights[-1].time == act365f(obs, terminal)
    friday = next(r for r in p.decisions if r.time == act365f(obs, cal.session(date(2026, 10, 30)).closes_at))
    monday = next(r for r in p.decisions if r.time == act365f(obs, cal.session(date(2026, 11, 2)).closes_at))
    assert (monday.time-friday.time)*365*24 == pytest.approx(73.)


@pytest.mark.parametrize('intervals', [(), (True,), (0,), (-1,), (30, 60), (60, 60), (60, 25), (60.0, 30)])
def test_invalid_or_nonnested_intraday_intervals_rejected(intervals):
    cal, obs, expiry, p = fixture()
    with pytest.raises(ValueError):core_session_schedules(cal, obs, expiry, p.decisions, intervals)


@pytest.mark.parametrize('change', ['missing', 'shift', 'phase'])
def test_reject_baseline_that_is_not_the_dated_core_close_schedule(change):
    cal, obs, expiry, p = fixture()
    rights = p.decisions
    if change == 'missing':rights = rights[1:]
    elif change == 'shift':rights = (replace(rights[0], time=rights[0].time+1e-6),)+rights[1:]
    else:rights = (replace(rights[0], phase='before'),)+rights[1:]
    with pytest.raises(ValueError):core_session_schedules(cal, obs, expiry, rights)


def test_nested_calendar_numerical_order_and_european_reference():
    cal, obs, expiry, p = fixture()
    schedules = core_session_schedules(cal, obs, expiry, p.decisions)
    anchors = tuple(r.time for r in schedules['core_30_minutes'])
    prices = [finite_difference_price(p.model, p.option, 1600, 256, 400., rights,
                                     propagation_times=anchors).premium_quote for rights in schedules.values()]
    assert all(b >= a-1e-10 for a,b in zip(prices,prices[1:]))
    reference = european_reference(p.model, p.option, p.decisions[-1])
    assert abs(prices[0]-reference.premium_quote) < .005


def test_strike_slope_has_units_and_is_invariant_to_residual_level():
    assert descriptive_strike_slope([90.,100.,110.],[-1.,1.,3.]) == pytest.approx(.2)
    assert descriptive_strike_slope([90.,100.,110.],[8.,10.,12.]) == pytest.approx(.2)
    with pytest.raises(ValueError):descriptive_strike_slope([100.,100.],[1.,2.])


def pair_fixture():
    common = dict(date='2026-09-17', strike=100., expiry='2026-10-16', dividend_model='mathematical_fixture',
                  volatility=.2, quoted_spot=100., initial_share_factor=1., premium_factor=1.,
                  terminal_settlement_factor=.99, dividend_caps=((.01,2.),))
    call = dict(common,kind='call',bid=3.,ask=5.,model_price=4.5,european_price=4.)
    put = dict(common,kind='put',bid=4.,ask=6.,model_price=5.7,european_price=5.)
    return call, put


def test_paired_decomposition_preserves_bid_ask_difference_and_premia():
    call, put = pair_fixture(); before = deepcopy((call,put))
    result = paired_quote_decomposition(call,put)
    assert result['observed_difference_lower'] == -3.
    assert result['observed_difference_upper'] == 1.
    assert result['model_minus_observed_difference'] == pytest.approx(-.2)
    assert result['early_exercise_premium_difference'] == pytest.approx(-.2)
    assert abs(result['decomposition_rounding_residual']) < 1e-14
    assert (call,put) == before


@pytest.mark.parametrize('field,value', [('volatility',.3), ('quoted_spot',101.), ('premium_factor',.99),
                                       ('terminal_settlement_factor',.98), ('dividend_caps',()), ('kind','call')])
def test_unmatched_economic_specification_cannot_be_paired(field,value):
    call, put = pair_fixture();put[field] = value
    with pytest.raises(ValueError):paired_quote_decomposition(call,put)
