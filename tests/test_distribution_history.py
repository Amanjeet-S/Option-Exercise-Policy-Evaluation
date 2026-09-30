"""Declared table fixtures, not observed distributions or forecast results."""

from copy import deepcopy
from datetime import datetime

import pytest

from qf_research.market_data.distribution_history import (
    ISSUER_HEADER, coverage, parse_issuer_rows, prospective_benchmarks, reconcile,
)

RETRIEVED = "2026-09-17T10:00:00Z"
ORIGIN = "2026-09-17T11:00:00Z"
TARGETS = [dict(ex_date="2026-09-18", pay_date="2026-10-30"),
           dict(ex_date="2026-12-18", pay_date="2027-01-29")]


def source_row(ex, pay, amount, short="", long=""):
    return ["Declared test fund", "SPY", "78462F103", ex, ex, pay, amount, short, long, "Quarterly"]


def fixture():
    return [ISSUER_HEADER,
            source_row("09/19/2025", "10/31/2025", "1.000000"),
            source_row("12/19/2025", "01/30/2026", "2.000000"),
            source_row("03/20/2026", "04/30/2026", "3.000000"),
            source_row("06/18/2026", "07/31/2026", "4.000000")]


def records():
    return parse_issuer_rows(fixture(), RETRIEVED)


def forecast(rows=None, origin=ORIGIN, targets=TARGETS):
    return prospective_benchmarks(records() if rows is None else rows, origin, targets, "2026Q2")


def test_decimal_precision_zero_and_blank_components_remain_distinct():
    table = [ISSUER_HEADER, source_row("06/18/2026", "07/31/2026", "0.000000", "0.000000", "")]
    r = parse_issuer_rows(table, RETRIEVED)[0]
    assert r["dividend"] == "0.000000" and r["short_gain"] == "0.000000"
    assert r["long_gain"] is None and r["total_cash_if_all_components_reported"] is None
    table[1][8] = "0.120000"
    assert parse_issuer_rows(table, RETRIEVED)[0]["total_cash_if_all_components_reported"] == "0.120000"


def test_unsorted_rows_keep_sheet_locators_and_inputs_unchanged():
    table = [ISSUER_HEADER, *reversed(fixture()[1:])]
    original = deepcopy(table)
    r = parse_issuer_rows(table, RETRIEVED)
    assert [x["source_row"] for x in r] == [5, 4, 3, 2]
    assert table == original
    assert all(x["available_at"] == "2026-09-17T10:00:00+00:00" for x in r)


@pytest.mark.parametrize("column,value", [(2, "wrong CUSIP"), (6, "=1+1"), (6, "NaN"),
                                          (6, "-1"), (6, True), (3, "2026-06-18"),
                                          (5, "02/30/2025"), (9, "Monthly")])
def test_ambiguous_identity_cash_formula_or_dates_rejected(column, value):
    table = fixture(); table[1][column] = value
    with pytest.raises(ValueError):
        parse_issuer_rows(table, RETRIEVED)


def test_changed_layout_empty_and_duplicate_ex_dates_rejected():
    for table in ([], [ISSUER_HEADER], [tuple(reversed(ISSUER_HEADER)), *fixture()[1:]],
                  [*fixture(), fixture()[1]]):
        with pytest.raises(ValueError):
            parse_issuer_rows(table, RETRIEVED)


def test_excel_typed_calendar_dates_supported_without_guessing_serials():
    table = fixture()
    table[1][3:6] = [datetime(2025, 9, 19), datetime(2025, 9, 19), datetime(2025, 10, 31)]
    assert parse_issuer_rows(table, RETRIEVED)[0]["ex_date"] == "2025-09-19"
    table[1][3] = 45919
    with pytest.raises(ValueError, match="date format"):
        parse_issuer_rows(table, RETRIEVED)


def test_extra_event_is_preserved_and_prevents_single_payment_forecast():
    table = fixture() + [source_row("11/15/2025", "12/02/2025", "0.350000")]
    r = parse_issuer_rows(table, RETRIEVED)
    c = coverage(r, "2025Q3", "2026Q2")
    assert c["expected_quarters"] == 4 and not c["missing_quarters"]
    assert c["multiple_event_quarters"] == {"2025Q4": [6, 3]}
    assert c["off_cycle_rows"] == [6]
    with pytest.raises(ValueError, match="exactly one event"):
        forecast(r)


def test_missing_quarter_cannot_be_replaced_by_an_older_payment():
    r = records()[1:]
    assert coverage(r, "2025Q3", "2026Q2")["missing_quarters"] == ["2025Q3"]
    with pytest.raises(ValueError, match="exactly one event"):
        forecast(r)


def test_source_date_anomaly_is_retained_but_cannot_enter_forecast():
    table = fixture(); table[1][5] = "09/01/2025"
    r = parse_issuer_rows(table, RETRIEVED)
    assert r[0]["pay_date"] == "2025-09-01"
    assert coverage(r, "2025Q3", "2026Q2")["date_issue_rows"] == [2]
    with pytest.raises(ValueError, match="date inconsistency"):
        forecast(r)


def test_two_horizon_benchmarks_use_frozen_known_inputs_without_recursion():
    r = records(); original = deepcopy(r)
    f = forecast(r)
    assert [x["forecast_amount"] for x in f] == ["4.000000", "1.000000", "2.500000",
                                               "4.000000", "2.000000", "2.500000"]
    assert f[1]["source_ex_dates"] == ["2025-09-19"]
    assert f[4]["source_ex_dates"] == ["2025-12-19"]
    assert r == original
    assert all(not x["risk_neutral_expectation_verified"] and not x["complete_future_schedule_verified"] for x in f)


def test_information_origin_and_backdating_are_enforced():
    with pytest.raises(ValueError, match="unavailable"):
        forecast(origin="2026-09-15T19:45:00Z")
    r = records(); r[0]["available_at"] = "2025-09-19T00:00:00Z"
    with pytest.raises(ValueError, match="backdated"):
        forecast(r)
    with pytest.raises(ValueError, match="zoned"):
        forecast(origin="2026-09-17")


def test_zero_forecast_and_payment_time_restriction():
    r = records(); r[-1]["dividend"] = "0.000000"
    assert forecast(r)[0]["forecast_amount"] == "0.000000"
    r[-1]["pay_date"] = "2026-09-17"
    with pytest.raises(ValueError, match="not in the past"):
        forecast(r)


def test_realised_target_or_uncompleted_reference_quarter_rejected():
    with pytest.raises(ValueError, match="future regular"):
        forecast(targets=[dict(ex_date="2026-06-18", pay_date="2026-07-31")])
    with pytest.raises(ValueError, match="last completed"):
        prospective_benchmarks(records(), ORIGIN, TARGETS, "2026Q3")


def test_reconciliation_does_not_merge_ambiguous_or_unplaced_events():
    r = records()
    def vendor(i, amount="1.0000004"):
        return dict(record_id=i, ex_date=r[0]["ex_date"], amount=amount, currency="USD",
                    record_date=r[0]["record_date"], pay_date=r[0]["pay_date"],
                    payment_type="Cash Dividend", dividend_type="Interim")
    v = vendor("a")
    compared = reconcile(r, [v], "0.0000005")
    assert compared["rows"][0]["amount_within_tolerance"]
    assert not compared["rows"][0]["amount_exact"]
    assert compared["rows"][1]["status"] == "unmatched"
    unplaced = dict(v, record_id="c", ex_date=None)
    ambiguous = reconcile(r, [v, vendor("b"), unplaced], "0.0000005")
    assert ambiguous["rows"][0]["status"] == "ambiguous"
    assert ambiguous["vendor_unplaced_rows"] == ["c"]
    missing = reconcile(r, [vendor("d", None)], "0.0000005")
    assert missing["rows"][0]["amount_difference"] is None
    assert not missing["rows"][0]["amount_within_tolerance"]


def test_component_diagnostic_does_not_silently_redefine_dividend():
    table = [ISSUER_HEADER, source_row("09/19/2025", "10/31/2025", "1.000000", "0.100000", "0.040000")]
    r = parse_issuer_rows(table, RETRIEVED)
    v = dict(record_id="v", ex_date="2025-09-19", record_date="2025-09-19",
             pay_date="2025-10-31", amount="1.100000", currency="USD",
             payment_type="Cash Dividend", dividend_type="Interim")
    row = reconcile(r, [v], "0.0000005")["rows"][0]
    assert row["vendor_equals_dividend_plus_reported_short_gain"]
    assert not row["amount_within_tolerance"]
    assert row["amount_difference"] == "-0.100000"
    assert r[0]["total_cash_if_all_components_reported"] == "1.140000"
