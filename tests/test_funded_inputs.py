"""Actual calendar dates combined with declared mathematical funding fixtures."""

from datetime import date, datetime
import json
import math
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK, act365f
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.funded_cash import european_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    config = json.loads((ROOT / "configs/us_event_calendar_2026-09-16.json").read_text())
    calendar = EventCalendar.from_config(config)
    curve = DiscountCurve(date(2026, 9, 9), (date(2026, 9, 9), date(2027, 9, 9)), (1., math.exp(-.05)))
    observation = datetime(2026, 9, 9, 15, 45, tzinfo=NEW_YORK)
    return calendar, curve, observation


def build(inputs, *, expiry=date(2026, 10, 16), dividends=(), **kwargs):
    return build_dated_problem(*inputs, expiry, 100., 100., .2, "call", dividends,
                               premium_payment_date=date(2026, 9, 10), **kwargs)


def test_dated_funding_ratios_and_exact_decision_clock(inputs):
    problem = build(inputs)
    calendar, curve, observation = inputs
    assert len(problem.decisions) == 28
    assert problem.share_payment_date == date(2026, 9, 10)
    assert problem.model.initial_state == pytest.approx(100.*math.exp(-.05/365))
    assert problem.decisions[0].time*365*86400 == pytest.approx(900.)
    assert problem.option.maturity == act365f(observation, calendar.session(date(2026, 10, 16)).closes_at)
    assert problem.deliveries[-1][1] == date(2026, 10, 19)
    assert problem.decisions[-1].settlement_factor == curve.between(observation.date(), date(2026, 10, 19))
    assert not problem.provisional_delivery and not problem.empirical_calibration_accepted


def test_dividend_gap_payment_lag_and_expiry_order(inputs):
    dividend = DatedDividend(date(2026, 12, 18), date(2027, 1, 29), 4., "Analytical fixture, not an observation")
    problem = build(inputs, expiry=date(2026, 12, 18), dividends=(dividend,))
    event = problem.model.dividends[0]
    previous = problem.decisions[-2]
    assert (event.ex_time-previous.time)*365*24 == pytest.approx(17.5)
    assert event.ex_time < problem.option.maturity
    assert event.discounted_cash == pytest.approx(4.*inputs[1].between(inputs[2].date(), dividend.pay_date))
    assert all(d.phase == "after" for d in problem.decisions)


def test_unknown_amount_never_becomes_zero(inputs):
    dividend = DatedDividend(date(2026, 9, 18), date(2026, 10, 30), None, "Unresolved")
    with pytest.raises(ValueError, match="Unresolved amount"):
        build(inputs, dividends=(dividend,))


def test_unknown_event_occurrence_and_explicit_exclusion(inputs):
    event = DatedDividend(date(2026, 12, 31), date(2027, 1, 29), 0., "Specified amount cannot resolve occurrence", None)
    with pytest.raises(ValueError, match="occurrence"):
        build(inputs, expiry=date(2027, 1, 15), dividends=(event,), allow_provisional=True)
    excluded = DatedDividend(event.ex_date, event.pay_date, None, "Explicit no-event scenario", False)
    problem = build(inputs, expiry=date(2027, 1, 15), dividends=(excluded,), allow_provisional=True)
    assert not problem.model.dividends and problem.dividend_inputs == (excluded,)
    assert problem.provisional_delivery


def test_january_requires_explicit_calendar_opt_in(inputs):
    with pytest.raises(ValueError, match="provisional"):
        build(inputs, expiry=date(2027, 1, 15))


def test_post_expiry_unknown_dividend_does_not_block_cash_model(inputs):
    event = DatedDividend(date(2026, 12, 31), date(2027, 1, 29), None, "Unresolved", None)
    assert not build(inputs, dividends=(event,)).model.dividends


def test_premium_date_is_not_automatically_set_to_share_delivery(inputs):
    calendar, curve, _ = inputs
    observation = datetime(2026, 10, 9, 15, 45, tzinfo=NEW_YORK)
    problem = build_dated_problem(calendar, curve, observation, date(2026, 10, 16), 100., 100., .2,
                                  "call", (), premium_payment_date=date(2026, 10, 12))
    assert problem.share_payment_date == date(2026, 10, 13)
    assert problem.premium_payment_date == date(2026, 10, 12)
    assert problem.model.premium_factor != problem.model.initial_share_factor
    # The supplied premium date is a scenario, not an account-level certification.
    assert not problem.premium_date_verified


def test_curve_payment_coverage_required(inputs):
    dividend = DatedDividend(date(2026, 9, 18), date(2028, 1, 1), 4., "Analytical fixture")
    with pytest.raises(ValueError, match="extrapolation"):
        build(inputs, dividends=(dividend,))


def test_funded_calendar_reaches_analytical_reference(inputs):
    problem = build(inputs)
    reference = european_reference(problem.model, problem.option, problem.decisions[-1])
    assert reference.present_value > 0.
    assert reference.premium_quote == reference.present_value/problem.model.premium_factor


@pytest.mark.parametrize("cash", [float("nan"), -1., True])
def test_bad_cash_rejected(cash):
    with pytest.raises(ValueError):
        DatedDividend(date(2026, 9, 18), date(2026, 10, 30), cash, "Fixture")
