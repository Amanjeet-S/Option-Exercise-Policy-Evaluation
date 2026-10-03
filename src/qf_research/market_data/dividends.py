"""Inspect current dividend records without inventing historical information.

An event announcement date is not an amount-publication timestamp. Availability
evidence must refer to the particular record version and value being used.
"""

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from zoneinfo import ZoneInfo


POLICY_VERSION = "dividend_information_2026-09-16_v1"


def zoned_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("An explicitly zoned availability timestamp is required")
    return result.astimezone(timezone.utc)


def optional_date(value):
    if value is None or value == "":
        return None
    return date.fromisoformat(value).isoformat()


def optional_amount(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("A Boolean is not a cash amount")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid cash amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Cash dividends must be finite and non-negative")
    # Decimal strings preserve the returned decimal amount without float maths.
    return str(amount)


def parse_dividend_response(payload, requested_fields, retrieved_at):
    """Normalise an SDK-decoded fundamental/reference response; retain gaps.

    Current records are conservatively available at retrieval. An older event
    announcement, or a past ex-date, does not backdate the amount's availability.
    Missing requested columns are reported separately from blank returned cells.
    """
    retrieved_at = zoned_time(retrieved_at).isoformat()
    if payload.get("headerOrientation") != "horizontal" or len(payload.get("headers", [])) != 1:
        raise ValueError("Unsupported dividend table layout")
    columns = [h.get("field", h.get("displayName", "")).upper() for h in payload["headers"][0]]
    if not columns or columns[0] != "INSTRUMENT" or len(set(columns)) != len(columns):
        raise ValueError("Missing instrument or ambiguous dividend columns")
    required = {"TR.DIVANNOUNCEMENTDATE", "TR.DIVEXDATE", "TR.DIVRECORDDATE",
                "TR.DIVPAYDATE", "TR.DIVUNADJUSTEDGROSS", "TR.DIVORIGINALCURR",
                "TR.DIVTYPE", "TR.DIVPAYMENTTYPE"}
    if not required.issubset(columns):
        raise ValueError("Missing essential dividend columns")
    records = []
    for index, values in enumerate(payload.get("data", [])):
        if len(values) != len(columns):
            raise ValueError("Dividend row width differs from its header")
        row = dict(zip(columns, values))
        if row["INSTRUMENT"] != "SPY":
            raise ValueError("This acquisition is scoped to SPY")
        amount = optional_amount(row["TR.DIVUNADJUSTEDGROSS"])
        records.append({
            "record_id": f"response_row_{index:03d}",  # Local locator, not a provider event ID.
            "instrument": row["INSTRUMENT"],
            "provider_event_id": row.get("TR.CAEVENTID") or None,
            "announcement_date": optional_date(row["TR.DIVANNOUNCEMENTDATE"]),
            "ex_date": optional_date(row["TR.DIVEXDATE"]),
            "record_date": optional_date(row["TR.DIVRECORDDATE"]),
            "pay_date": optional_date(row["TR.DIVPAYDATE"]),
            "amount": amount,
            "currency": row["TR.DIVORIGINALCURR"] or None,
            "dividend_type": row["TR.DIVTYPE"],
            "payment_type": row["TR.DIVPAYMENTTYPE"],
            "date_status": row.get("TR.CADATESTATUS"),
            "rescinded": row.get("TR.CAISRESCINDED"),
            "schedule_available_at": retrieved_at,
            "amount_available_at": retrieved_at if amount is not None else None,
            "availability_basis": "current_record_retrieved",
            "amount_status": "unverified",
        })
    return {"records": records,
            "missing_requested_fields": sorted(set(f.upper() for f in requested_fields) - set(columns)),
            "returned_columns": columns}


def assess_cash_dividend(record, cutoff):
    """Audit an independently evidenced dividend version at an exact cutoff.

    Callers supplying archived publication evidence must establish that it refers
    to this version's dates, amount, currency and status. This function enforces
    time restrictions, not the authenticity of an external source.
    """
    cutoff = zoned_time(cutoff)
    issues = []
    for name in ("schedule_available_at", "amount_available_at"):
        value = record.get(name)
        if value is None:
            issues.append(f"{name}_missing")
        elif zoned_time(value) > cutoff:
            issues.append(f"{name}_after_cutoff")
    ex_date = optional_date(record.get("ex_date"))
    pay_date = optional_date(record.get("pay_date"))
    if ex_date is None:
        issues.append("ex_date_missing")
    if pay_date is None:
        issues.append("pay_date_missing")
    if ex_date and pay_date and pay_date < ex_date:
        issues.append("payment_precedes_ex_date")
    amount = optional_amount(record.get("amount"))
    if amount is None:
        issues.append("amount_missing")
    if record.get("currency") != "USD":
        issues.append("currency_missing_or_not_usd")
    if record.get("payment_type") != "Cash Dividend":
        issues.append("payment_type_not_verified_cash")
    if record.get("amount_status") != "confirmed":
        issues.append("amount_not_confirmed")
    if record.get("rescinded") is not False:
        issues.append("rescinded_or_rescission_status_unknown")
    return {"record_id": record["record_id"], "issues": issues,
            "cash_input_eligible": not issues,
            "ex_to_pay_calendar_days": ((date.fromisoformat(pay_date) - date.fromisoformat(ex_date)).days
                                        if ex_date and pay_date else None)}


def horizon_exposure(records, cutoff, expiry_date, *, coverage_verified=False):
    """Calendar exposure, including ex-dates on expiry; not a pricing schedule.

    Equal valuation/ex dates require an intraday entitlement convention. Unknown
    ex-dates remain visible and block completeness. Absence of returned events
    does not establish a zero-dividend horizon.
    """
    if not isinstance(coverage_verified, bool):
        raise ValueError("Coverage verification must be explicit Boolean evidence")
    today = zoned_time(cutoff).astimezone(ZoneInfo("America/New_York")).date()
    expiry = date.fromisoformat(expiry_date)
    if expiry < today:
        raise ValueError("Expiry precedes valuation")
    ids = [r["record_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate local dividend record IDs")
    candidates, unplaced = [], []
    for record in records:
        ex = optional_date(record.get("ex_date"))
        if ex is None:
            unplaced.append(record["record_id"])
            continue
        ex_day = date.fromisoformat(ex)
        if today <= ex_day <= expiry:
            result = assess_cash_dividend(record, cutoff)
            result.update(ex_date=ex, pay_date=record.get("pay_date"),
                          dividend_type=record.get("dividend_type"),
                          ex_date_equals_expiry=ex_day == expiry,
                          ex_date_equals_valuation=ex_day == today,
                          payment_after_expiry=bool(record.get("pay_date") and date.fromisoformat(record["pay_date"]) > expiry))
            if ex_day == today:
                result["issues"].append("valuation_day_entitlement_timing_unresolved")
                result["cash_input_eligible"] = False
            candidates.append(result)
    # Two economic rows on the same ex-date may be distinct dividends or a
    # duplicate/revision. Do not sum or pick one without verified identities.
    candidates.sort(key=lambda r: (r["ex_date"], r["record_id"]))
    ex_dates = [r["ex_date"] for r in candidates]
    ambiguous = sorted(d for d in set(ex_dates) if ex_dates.count(d) > 1)
    return {"cutoff": cutoff, "expiry_date": expiry_date, "events": candidates,
            "unplaced_records": unplaced, "ambiguous_ex_dates": ambiguous,
            "coverage_verified": coverage_verified,
            "cash_inputs_ready": bool(coverage_verified and not unplaced and not ambiguous
                                      and all(r["cash_input_eligible"] for r in candidates))}


def reconcile_calendar(records, calendar):
    """Compare dates, preserving a conditional event as conditional.

    Agreement between current sources does not backdate either source's values.
    """
    if calendar["instrument"] != "SPY":
        raise ValueError("Expected the SPY distribution calendar")
    results, seen = [], set()
    for event in calendar["events"]:
        ex = optional_date(event["ex_date"])
        if ex is None or ex in seen or not isinstance(event["conditional"], bool):
            raise ValueError("Incomplete or ambiguous issuer calendar")
        seen.add(ex)
        matches = [r for r in records if r["instrument"] == "SPY" and r["ex_date"] == ex]
        date_agreement = len(matches) == 1 and all(
            optional_date(event[k]) == matches[0][k] for k in ("record_date", "pay_date"))
        results.append({"ex_date": ex, "matching_provider_records": len(matches),
                        "dates_agree": date_agreement, "conditional": event["conditional"],
                        "historical_availability_established": False})
    return results


def compare_contract_terms(rows, reference):
    """Cross-check current desktop fields with the retained discovery records.

    Agreement is corroboration within one provider. Blank second deliverables
    are not evidence that the contract has no adjustment or cash component.
    """
    expected = {r["RIC"]: r for r in reference}
    if len(expected) != len(reference):
        raise ValueError("Duplicate reference identities")
    seen, results = set(), []
    for row in rows:
        ric = row.get("Instrument")
        if ric not in expected or ric in seen:
            raise ValueError("Unexpected or duplicate contract")
        seen.add(ric)
        ref = expected[ric]
        kind = ref["CallPutOption"].upper()
        checks = {
            "strike": optional_amount(row.get("STRIKE_PRC")) is not None and Decimal(str(row["STRIKE_PRC"])) == Decimal(str(ref["StrikePrice"])),
            "expiry": str(row.get("EXPIR_DATE", ""))[:10] == ref["ExpiryDate"][:10],
            "option_type": str(row.get("PUTCALLIND", "")).strip() == kind,
            "american_exercise": row.get("PUT_CALL") == {"CALL": "C-AM", "PUT": "P-AM"}[kind],
            "currency": row.get("CURRENCY") == ref["Currency"] == "USD",
            "underlying": row.get("STOCK_RIC") == "SPY",
            "lot_size": optional_amount(row.get("LOT_SIZE_A")) is not None and Decimal(str(row["LOT_SIZE_A"])) == Decimal(str(ref["LotSize"])) == 100,
            "lot_unit": row.get("LOTSZUNITS") == "SHARE",
            "trade_price_multiplier": optional_amount(row.get("MULTIPLIER")) is not None and Decimal(str(row["MULTIPLIER"])) == 100,
        }
        results.append({"ric": ric, "checks": checks, "reference_agreement": all(checks.values()),
                        "missing_returned_fields": sorted(k for k, v in row.items() if v is None),
                        "contract_deliverables_certified": False,
                        "historical_terms_certified": False})
    if seen != set(expected):
        raise ValueError("Incomplete contract coverage")
    return results
