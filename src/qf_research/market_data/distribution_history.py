"""Current-vintage issuer distributions and prospective regular-cash benchmarks.

An ex-date does not prove when a particular historical amount became available.
Blank capital-gain cells remain unknown. These forecasts are not risk-neutral
expectations or a complete future distribution schedule.
"""

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from zoneinfo import ZoneInfo

from .dividends import zoned_time


ISSUER_HEADER = (
    "FUND NAME", "TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE",
    "DIVIDEND ($)", "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)",
    "FREQUENCY",
)
MODELS = ("last_regular", "same_quarter_previous_year", "last_four_regular_mean")


def cash(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean cash amount")
    text = str(value).strip()
    if not text:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise ValueError("Expected a non-negative decimal cash amount")
    try:
        return str(Decimal(text))
    except InvalidOperation as exc:
        raise ValueError("Invalid cash amount") from exc


def issuer_date(value):
    if isinstance(value, datetime):
        if value.time().isoformat() != "00:00:00" or value.tzinfo is not None:
            raise ValueError("Expected an unzoned calendar date")
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        raise ValueError("Unsupported issuer date format")
    return datetime.strptime(value, "%m/%d/%Y").date().isoformat()


def quarter_number(day):
    d = date.fromisoformat(day)
    return 4 * d.year + (d.month - 1) // 3


def quarter_label(number):
    year, q = divmod(number, 4)
    return f"{year}Q{q + 1}"


def parse_quarter(label):
    if not re.fullmatch(r"\d{4}Q[1-4]", label):
        raise ValueError("Invalid quarter")
    return int(label[:4]) * 4 + int(label[-1]) - 1


def parse_issuer_rows(rows, retrieved_at):
    """Parse the inspected dividend worksheet; retain Excel row locators.

    The caller reads formulas as formulas, not cached numbers. No workbook is
    edited. USD refers to this US-listed SPY distribution table and identity.
    """
    rows = iter(rows)
    if tuple(next(rows, ())) != ISSUER_HEADER:
        raise ValueError("Issuer worksheet header differs from reviewed layout")
    available = zoned_time(retrieved_at).isoformat()
    result, identities = [], set()
    for index, row in enumerate(rows, 2):
        if len(row) > 1 and isinstance(row[1], str) and row[1].startswith("="):
            raise ValueError("Formula-based ticker cannot be classified")
        if len(row) < 2 or row[1] != "SPY":
            continue
        if len(row) != len(ISSUER_HEADER):
            raise ValueError("Issuer row width differs")
        if any(isinstance(v, str) and v.startswith("=") for v in row):
            raise ValueError("Formula in issuer SPY row requires separate review")
        if row[2] != "78462F103" or row[9] != "Quarterly":
            raise ValueError("Unexpected issuer identity or frequency")
        ex, record, pay = (issuer_date(v) for v in row[3:6])
        date_issues = [] if ex <= record <= pay else ["dates_not_ex_record_payment_order"]
        if ex in identities:
            raise ValueError("Duplicate issuer ex-date requires event-level review")
        identities.add(ex)
        values = [cash(v) for v in row[6:9]]
        total = str(sum(map(Decimal, values), Decimal(0))) if all(v is not None for v in values) else None
        result.append(dict(source_row=index, instrument="SPY", cusip=row[2], currency="USD",
                           ex_date=ex, record_date=record, pay_date=pay,
                           dividend=values[0], short_gain=values[1], long_gain=values[2],
                           total_cash_if_all_components_reported=total,
                           date_issues=date_issues,
                           available_at=available, retrieved_at=available,
                           availability_basis="current_workbook_retrieval"))
    if not result:
        raise ValueError("No issuer SPY rows")
    return sorted(result, key=lambda r: r["ex_date"])


def coverage(records, first_quarter, last_quarter):
    lo, hi = parse_quarter(first_quarter), parse_quarter(last_quarter)
    if lo > hi:
        raise ValueError("Reversed quarter range")
    groups = defaultdict(list)
    for r in records:
        groups[quarter_number(r["ex_date"])].append(r)
    expected = range(lo, hi + 1)
    return dict(
        expected_quarters=hi - lo + 1,
        missing_quarters=[quarter_label(q) for q in expected if not groups[q]],
        multiple_event_quarters={quarter_label(q): [r["source_row"] for r in groups[q]]
                                 for q in expected if len(groups[q]) > 1},
        outside_quarter_range=[r["source_row"] for q, rs in groups.items() if not lo <= q <= hi for r in rs],
        off_cycle_rows=[r["source_row"] for r in records if date.fromisoformat(r["ex_date"]).month not in (3, 6, 9, 12)],
        missing_dividend_rows=[r["source_row"] for r in records if r["dividend"] is None],
        date_issue_rows=[r["source_row"] for r in records if r["date_issues"]],
        nonzero_capital_gain_rows=[r["source_row"] for r in records if any(
            r[k] is not None and Decimal(r[k]) != 0 for k in ("short_gain", "long_gain"))],
        unknown_component_total_rows=sum(r["total_cash_if_all_components_reported"] is None for r in records),
        historical_vintage_verified=False,
    )


def reconcile(issuer, vendor, tolerance):
    """Join exactly on ex-date; preserve unmatched and ambiguous records."""
    tol = Decimal(tolerance)
    if not tol.is_finite() or tol < 0:
        raise ValueError("Invalid reconciliation tolerance")
    group = defaultdict(list)
    unplaced = []
    for row in vendor:
        if row["ex_date"] is None:
            unplaced.append(row["record_id"])
        else:
            group[row["ex_date"]].append(row)
    result = []
    for row in issuer:
        matches = group.get(row["ex_date"], [])
        item = dict(issuer_row=row["source_row"], ex_date=row["ex_date"],
                    vendor_rows=[v["record_id"] for v in matches])
        if len(matches) != 1:
            item["status"] = "unmatched" if not matches else "ambiguous"
        else:
            v = matches[0]
            diff = None if row["dividend"] is None or v["amount"] is None else Decimal(row["dividend"]) - Decimal(v["amount"])
            item.update(status="compared", amount_difference=None if diff is None else str(diff),
                        amount_exact=False if diff is None else diff == 0,
                        amount_within_tolerance=False if diff is None else abs(diff) <= tol,
                        record_date_equal=row["record_date"] == v["record_date"],
                        pay_date_equal=row["pay_date"] == v["pay_date"],
                        currency_equal=row["currency"] == v["currency"],
                        cash_type_verified=v["payment_type"] == "Cash Dividend",
                        vendor_dividend_type=v["dividend_type"])
            # Component equality is a diagnostic, not permission to redefine
            # either provider's measure or to treat a blank component as zero.
            if row["dividend"] is not None and row["short_gain"] is not None and v["amount"] is not None:
                combined = Decimal(row["dividend"]) + Decimal(row["short_gain"])
                item["vendor_equals_dividend_plus_reported_short_gain"] = Decimal(v["amount"]) == combined
            else:
                item["vendor_equals_dividend_plus_reported_short_gain"] = None
        result.append(item)
    dates = {r["ex_date"] for r in issuer}
    return dict(rows=result, vendor_unplaced_rows=unplaced,
                vendor_only_rows=[r["record_id"] for r in vendor if r["ex_date"] is not None and r["ex_date"] not in dates],
                complete_future_schedule_verified=False)


def prospective_benchmarks(records, as_of, targets, last_completed_quarter):
    """Emit all three frozen benchmarks, requiring actual source availability.

    Targets must be in an uncompleted quarter. The model uses one observed
    regular dividend for each required reference quarter and never substitutes
    future realised amounts, silently drops an extra event or fills a gap.
    """
    origin = zoned_time(as_of)
    local_day = origin.astimezone(ZoneInfo("America/New_York")).date()
    last = parse_quarter(last_completed_quarter)
    if last != quarter_number(local_day.isoformat()) - 1:
        raise ValueError("Reference end must be the last completed calendar quarter")
    if not records:
        raise ValueError("No issuer records")
    groups = defaultdict(list)
    seen = set()
    for r in records:
        if zoned_time(r["available_at"]) > origin:
            raise ValueError("Source record unavailable at forecast origin")
        if zoned_time(r["available_at"]) != zoned_time(r["retrieved_at"]):
            raise ValueError("Current workbook availability cannot be backdated")
        if r["availability_basis"] != "current_workbook_retrieval":
            raise ValueError("Unreviewed availability basis")
        if r["ex_date"] in seen:
            raise ValueError("Duplicate source event")
        seen.add(r["ex_date"])
        groups[quarter_number(r["ex_date"])].append(r)

    def required(q):
        rs = groups[q]
        if len(rs) != 1:
            raise ValueError(f"Expected exactly one event in {quarter_label(q)}")
        r = rs[0]
        if r["date_issues"]:
            raise ValueError("Unresolved reference-date inconsistency")
        if r["currency"] != "USD" or r["dividend"] is None:
            raise ValueError("Unknown cash dividend or currency")
        if date.fromisoformat(r["pay_date"]) >= local_day:
            raise ValueError("Reference cash payment is not in the past")
        if date.fromisoformat(r["ex_date"]).month not in (3, 6, 9, 12):
            raise ValueError("Reference event has an off-cycle ex-month")
        amount = cash(r["dividend"])
        return r, Decimal(amount)

    trailing = [required(q) for q in range(last - 3, last + 1)]
    results, seen_targets = [], set()
    for target in targets:
        ex, pay = date.fromisoformat(target["ex_date"]), date.fromisoformat(target["pay_date"])
        q = quarter_number(ex.isoformat())
        if ex <= local_day or pay < ex or q not in (last + 1, last + 2) or q in seen_targets:
            raise ValueError("Targets must be unique future regular dates in the next two quarters")
        if ex.month not in (3, 6, 9, 12):
            raise ValueError("Off-cycle target")
        seen_targets.add(q)
        seasonal = required(q - 4)
        specifications = (
            (MODELS[0], trailing[-1][1], [trailing[-1][0]]),
            (MODELS[1], seasonal[1], [seasonal[0]]),
            (MODELS[2], sum(v for _, v in trailing) / Decimal(4), [r for r, _ in trailing]),
        )
        for model, amount, inputs in specifications:
            results.append(dict(model=model, **target, forecast_amount=str(amount), currency="USD",
                                forecast_origin=origin.isoformat(),
                                source_rows=[r["source_row"] for r in inputs],
                                source_ex_dates=[r["ex_date"] for r in inputs],
                                last_input_available_at=max(r["available_at"] for r in inputs),
                                scope="regular dividend component only",
                                made_before_amount_announcement_verified=False,
                                risk_neutral_expectation_verified=False,
                                complete_future_schedule_verified=False))
    if not results:
        raise ValueError("No forecast targets")
    return results
