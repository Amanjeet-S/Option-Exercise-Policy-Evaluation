"""Mechanical unit fixtures only; these are not observations or forecast results."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from html import escape
import pytest

from qf_research.market_data.distribution_announcements import (
    parse_announcement, reconcile_announcements, reconstruct_benchmarks,
)


def html_fixture(**overrides):
    fields = {
        "Issuer/ Manager": "SPDR S&P 500 ETF TRUST",
        "Security": "SPDR S&P 500 ETF TRUST - US78462F1030 - S27",
        "Announcement Title": "Mandatory Cash Dividend/ Distribution",
        "Date &Time of Broadcast": "18-Jun-2026 07:54:55", "Status": "New",
        "Corporate Action Reference": "SG260618DVCA0000",
        "Dividend/ Distribution Type": "Interim",
        "Declared Dividend/ Distribution Rate (Per Share/ Unit)": "USD 2.000000",
        "Record Date": "18/06/2026", "Ex Date": "17/06/2026",
        "Payment Type": "Payment Rate in Gross", "Gross Rate (Per Share)": "USD 2.000000",
        "Net Rate (Per Share)": "USD 2.000000", "Pay Date": "31/07/2026",
        "Gross Rate Status": "Actual Rate", "Related Announcements": "",
    }
    fields.update(overrides)
    text = '<meta name="Date" content="20260618T07:54:55Z">'
    text += "<h1>Cash Dividend/ Distribution::Mandatory</h1><dl>"
    text += "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in fields.items())
    return (text + "</dl>").encode()


def parsed(body=None, retrieval="2026-09-17T12:00:00+00:00"):
    return parse_announcement(html_fixture() if body is None else body,
                              source_url="https://example.test/unit-fixture", retrieved_at=retrieval)


def declarations():
    # Algebraic fixture with four distinct amounts tests horizon indexing.
    base = parsed()
    result = []
    for q, amount, record, pay in (
        ("2025Q3", "1", "2025-09-19", "2025-10-31"),
        ("2025Q4", "3", "2025-12-19", "2026-01-30"),
        ("2026Q1", "5", "2026-03-20", "2026-04-30"),
        ("2026Q2", "7", "2026-06-18", "2026-07-31"),
    ):
        result.append(dict(base, quarter=q, reference=f"fixture-{q}", gross_per_unit=amount,
                           record_date=record, pay_date=pay))
    return result


ORIGIN = "2026-09-09T15:45:00-04:00"
TARGETS = [{"ex_date": "2026-09-18", "pay_date": "2026-10-30"},
           {"ex_date": "2026-12-18", "pay_date": "2027-01-29"}]


def forecast(records=None, origin=ORIGIN, targets=None):
    return reconstruct_benchmarks(declarations() if records is None else records,
                                  origin, TARGETS if targets is None else targets, "2026Q2")


def test_source_clock_is_not_silently_converted_and_ex_date_is_separate():
    r = parsed()
    assert r["publication_bound_at"] == "2026-06-20T00:00:00+00:00"
    assert r["metadata_date_label"] == "20260618T07:54:55Z"
    assert not r["broadcast_timezone_verified"]
    assert r["retrieved_at"] == "2026-09-17T12:00:00+00:00"
    assert r["sgx_ex_date"] == "2026-06-17"
    assert "us_ex_date" not in r and "available_at" not in r
    assert r["gross_per_unit"] == "2.000000"


def test_bound_follows_end_of_displayed_date_in_extreme_timezones():
    bound = datetime.fromisoformat(parsed()["publication_bound_at"])
    for offset in (-12, 14):
        last_in_date = datetime(2026, 6, 18, 23, 59, 59, tzinfo=timezone(timedelta(hours=offset)))
        assert bound > last_in_date


@pytest.mark.parametrize("change", [
    {"Status": "Replacement"}, {"Gross Rate Status": "Estimated Rate"},
    {"Security": "ANOTHER FUND - US0000000000 - S27"},
    {"Issuer/ Manager": "ANOTHER FUND"}, {"Gross Rate (Per Share)": "SGD 2.000000"},
    {"Gross Rate (Per Share)": "USD NaN"}, {"Net Rate (Per Share)": "USD 1.900000"},
    {"Related Announcements": "Replacement filing"},
    {"Pay Date": "17/06/2026"}, {"Record Date": "06/18/2026"},
    {"Corporate Action Reference": "SG260619DVCA0000"},
])
def test_unreviewed_fields_and_dates_rejected(change):
    with pytest.raises(ValueError):
        parsed(html_fixture(**change))


@pytest.mark.parametrize("addition", [
    b"<dt>Status</dt><dd>New</dd>",
    b'<meta name="Date" content="20260618T07:54:55Z">',
    b"<h1>Cash Dividend/ Distribution::Mandatory</h1>",
])
def test_ambiguous_html_rejected(addition):
    with pytest.raises(ValueError):
        parsed(html_fixture() + addition)


def test_blank_related_link_is_still_a_relationship():
    body = html_fixture().replace(b"<dt>Related Announcements</dt><dd></dd>",
                                  b'<dt>Related Announcements</dt><dd><a href="replacement"></a></dd>')
    with pytest.raises(ValueError, match="Related announcement"):
        parsed(body)


def test_recent_or_metadata_disagreeing_declarations_rejected():
    with pytest.raises(ValueError, match="too recent"):
        parsed(retrieval="2026-06-19T23:59:59+00:00")
    with pytest.raises(ValueError, match="metadata"):
        parsed(html_fixture().replace(b"20260618T07:54:55Z", b"20260617T23:54:55Z"))


def test_zero_is_reported_cash_not_missing():
    keys = ["Declared Dividend/ Distribution Rate (Per Share/ Unit)",
            "Gross Rate (Per Share)", "Net Rate (Per Share)"]
    assert parsed(html_fixture(**{k: "USD 0" for k in keys}))["gross_per_unit"] == "0"


def issuer_fixture():
    return dict(source_row=99, instrument="SPY", cusip="78462F103", currency="USD",
                record_date="2026-06-18", ex_date="2026-06-18", pay_date="2026-07-31",
                dividend="2", date_issues=[])


def test_reconciliation_uses_record_date_not_sgx_ex_date():
    r = reconcile_announcements([parsed()], [issuer_fixture()])[0]
    assert r["all_checks_passed"]
    assert r["sgx_ex_date"] != r["us_ex_date"]


@pytest.mark.parametrize("change", [
    {"dividend": "2.000001"}, {"dividend": None}, {"cusip": "000000000"},
    {"currency": "SGD"}, {"pay_date": "2026-08-01"}, {"date_issues": ["bad_date"]},
])
def test_discrepancies_are_not_repaired(change):
    issuer = dict(issuer_fixture(), **change)
    copy = deepcopy(issuer)
    assert not reconcile_announcements([parsed()], [issuer])[0]["all_checks_passed"]
    assert issuer == copy


def test_ambiguous_or_missing_matches_rejected():
    for records in ([], [issuer_fixture(), issuer_fixture()]):
        with pytest.raises(ValueError, match="unique"):
            reconcile_announcements([parsed()], records)
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_announcements([parsed(), parsed()], [issuer_fixture()])


def test_all_models_and_two_horizons_use_only_observed_declarations():
    rows = forecast()
    assert [r["regular_dividend_estimate"] for r in rows] == ["7", "1", "4", "7", "3", "4"]
    assert all(r["calculation_type"] == "retrospective_public_announcement_reconstruction" for r in rows)
    assert not any(r["risk_neutral_expectation_established"] for r in rows)
    assert not any(r["complete_cash_schedule"] for r in rows)
    assert rows[1]["input_references"] == ["fixture-2025Q3"]
    assert rows[4]["input_references"] == ["fixture-2025Q4"]


def test_future_publication_and_unpaid_cash_rejected():
    for changes in ({"publication_bound_at": "2026-09-10T00:00:00+00:00"},
                    {"pay_date": "2026-09-09"}):
        records = declarations()
        records[-1].update(changes)
        with pytest.raises(ValueError):
            forecast(records)


def test_missing_duplicate_and_unknown_cash_rejected():
    with pytest.raises(ValueError, match="Missing declaration"):
        forecast(declarations()[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        forecast(declarations() + declarations()[-1:])
    records = declarations()
    records[-1]["gross_per_unit"] = None
    with pytest.raises(ValueError, match="Missing declaration amount"):
        forecast(records)


@pytest.mark.parametrize("targets", [
    [], [TARGETS[0], TARGETS[0]],
    [dict(TARGETS[0], ex_date="2026-09-19"), TARGETS[0]],
    [dict(TARGETS[0], pay_date="2026-09-17")],
    [dict(TARGETS[0], ex_date="2026-08-18")],
    [dict(TARGETS[0], ex_date="2027-03-19", pay_date="2027-04-30")],
])
def test_bad_targets_rejected(targets):
    with pytest.raises(ValueError):
        forecast(targets=targets)


def test_new_quarter_requires_new_information_set():
    with pytest.raises(ValueError, match="last completed quarter"):
        forecast(origin="2026-10-01T15:45:00-04:00")
