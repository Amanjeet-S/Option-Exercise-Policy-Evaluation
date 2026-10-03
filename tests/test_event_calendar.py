"""Historical calendar facts and declared timing conventions, without quotes."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK, act365f, prior_ex_date_test

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def calendar():
    path = ROOT / "configs/us_event_calendar_2026-09-16.json"
    return EventCalendar.from_config(json.loads(path.read_text()))


@pytest.mark.parametrize("instruction,expected", [
    ("2026-09-11", "2026-09-14"),
    ("2026-10-09", "2026-10-13"),
    ("2026-10-12", "2026-10-13"),
    ("2026-11-10", "2026-11-12"),
    ("2026-11-11", "2026-11-12"),
    ("2026-11-25", "2026-11-27"),
    ("2026-12-24", "2026-12-28"),
])
def test_distinct_trading_and_settlement_days(calendar, instruction, expected):
    d = date.fromisoformat(instruction)
    assert calendar.is_trading_day(d)
    result = calendar.settlement_after(d)
    assert result.day == date.fromisoformat(expected)
    assert result.evidence == "published_2026_dtc_schedule"


@pytest.mark.parametrize("day", [date(2026, 10, 12), date(2026, 11, 11)])
def test_trading_without_settlement(calendar, day):
    assert calendar.is_trading_day(day)
    assert not calendar.is_settlement_day(day)


def test_early_closes_and_dst(calendar):
    assert calendar.session(date(2026, 11, 27)).closes_at.hour == 13
    assert calendar.session(date(2026, 12, 24)).closes_at.hour == 13
    before = calendar.session(date(2026, 10, 30)).closes_at
    after = calendar.session(date(2026, 11, 2)).closes_at
    assert before.astimezone(timezone.utc).hour == 20
    assert after.astimezone(timezone.utc).hour == 21
    assert act365f(before, after) * 365 * 24 == pytest.approx(73)


def test_provisional_requires_opt_in_including_year_boundary(calendar):
    for instruction, expected in [(date(2026, 12, 31), date(2027, 1, 4)),
                                  (date(2027, 1, 15), date(2027, 1, 19))]:
        with pytest.raises(ValueError, match="provisional"):
            calendar.settlement_after(instruction)
        projected = calendar.settlement_after(instruction, allow_provisional=True)
        assert projected.day == expected
        assert projected.evidence == "provisional_2027_no_dtc_schedule"


def test_prior_day_gap_and_expiry_day_entitlement(calendar):
    ex = date(2026, 12, 18)
    event = calendar.model_ex_event(ex)
    prior = calendar.previous_session(ex).closes_at
    assert act365f(prior, event) * 365 * 24 == pytest.approx(17.5)
    assert prior_ex_date_test(prior, ex)
    assert not prior_ex_date_test(datetime(2026, 12, 18, 0, 1, tzinfo=NEW_YORK), ex)
    assert not prior_ex_date_test(calendar.session(ex).closes_at, ex)
    # Date condition is based on New York, not the input timestamp's zone.
    assert prior_ex_date_test(datetime(2026, 12, 18, 1, tzinfo=timezone.utc), ex)


def test_research_decisions_end_at_core_close(calendar):
    observation = datetime(2026, 11, 25, 15, 45, tzinfo=NEW_YORK)
    rights = calendar.research_decisions(observation, date(2026, 11, 30))
    assert [t.day for t in rights] == [25, 27, 30]
    assert [t.hour for t in rights] == [16, 13, 16]
    assert rights == calendar.research_decisions(observation.astimezone(timezone.utc), date(2026, 11, 30))
    assert act365f(observation, rights[0]) * 365 * 86400 == pytest.approx(900)


@pytest.mark.parametrize("day", [date(2026, 8, 31), date(2027, 2, 2)])
def test_no_extrapolation(calendar, day):
    with pytest.raises(ValueError, match="coverage"):
        calendar.is_trading_day(day)


def test_settlement_cannot_run_outside_coverage(calendar):
    with pytest.raises(ValueError, match="coverage"):
        calendar.settlement_after(date(2027, 2, 1), allow_provisional=True)


@pytest.mark.parametrize("lag", [0, -1, 1.5, True])
def test_invalid_lags(calendar, lag):
    with pytest.raises(ValueError, match="lag"):
        calendar.settlement_after(date(2026, 9, 9), lag=lag)


def test_reject_ambiguous_or_closed_session_inputs(calendar):
    with pytest.raises(ValueError, match="zoned"):
        calendar.research_decisions(datetime(2026, 9, 9, 15, 45), date(2026, 10, 16))
    with pytest.raises(ValueError, match="outside"):
        calendar.research_decisions(datetime(2026, 11, 27, 15, 45, tzinfo=NEW_YORK), date(2026, 12, 18))
    with pytest.raises(ValueError, match="No core"):
        calendar.session(date(2026, 11, 26))
    with pytest.raises(ValueError, match="precedes"):
        calendar.research_decisions(datetime(2026, 9, 11, 15, 45, tzinfo=NEW_YORK), date(2026, 9, 10))
    with pytest.raises(ValueError, match="calendar date"):
        calendar.session(datetime(2026, 9, 9))


def test_malformed_early_close_rejected(calendar):
    with pytest.raises(ValueError, match="early close"):
        replace(calendar, early_closes=((date(2026, 11, 26), time(13)),))
    with pytest.raises(ValueError, match="Duplicate"):
        replace(calendar, early_closes=calendar.early_closes * 2)


def test_negative_interval_rejected():
    start = datetime(2026, 9, 9, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="Negative"):
        act365f(start, start - timedelta(seconds=1))
