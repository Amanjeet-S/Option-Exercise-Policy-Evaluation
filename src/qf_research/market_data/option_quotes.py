"""Offline, conservative inspection of LSEG historical quote-event responses.

Original response order and missing observations are never rewritten. Selection
uses DATE_TIME, retains SOURCE_DATETIME and does not certify feed eligibility,
NBBO semantics, historical availability or contract deliverables. Test fixtures
for this parser are not empirical market data.
"""

from datetime import datetime, timezone
import json
import math
import re


_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def timestamp_ns(value: str) -> int:
    """Parse an explicitly zoned timestamp without rounding nanoseconds."""
    match = _STAMP.fullmatch(value)
    if not match:
        raise ValueError(f"Timestamp must include an explicit timezone: {value!r}")
    base, fraction, offset = match.groups()
    moment = datetime.fromisoformat(base + ("+00:00" if offset == "Z" else offset))
    delta = moment.astimezone(timezone.utc) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 10**9 + int((fraction or "").ljust(9, "0"))


def response_blocks(payload: dict | list) -> list[dict]:
    """Normalise the documented single/multiple instrument response shapes."""
    blocks = [payload] if isinstance(payload, dict) else payload
    if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
        raise ValueError("Unexpected historical-pricing payload shape")
    return blocks


def quote_issues(row: dict, cutoff_ns: int) -> list[str]:
    """Report mechanical defects, without treating a zero bid as missing."""
    issues = []
    for name in ("BID", "ASK", "BIDSIZE", "ASKSIZE"):
        value = row.get(name)
        if value is None:
            issues.append(f"missing_{name.lower()}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(f"invalid_{name.lower()}")
        elif value < 0 or (name != "BID" and value == 0):
            issues.append(f"non_positive_{name.lower()}")
    if not any(i.endswith(("_bid", "_ask")) for i in issues) and row["BID"] > row["ASK"]:
        issues.append("crossed_market")
    source = row.get("SOURCE_DATETIME")
    if source:
        try:
            source_ns = timestamp_ns(source)
            if source_ns > cutoff_ns:
                issues.append("source_timestamp_after_cutoff")
            if source_ns > timestamp_ns(row["DATE_TIME"]):
                issues.append("source_timestamp_after_event")
        except (ValueError, TypeError):
            issues.append("invalid_source_timestamp")
    return issues


def inspect_quote_block(block: dict, cutoff: str, lookback_seconds: int, count_cap: int) -> dict:
    """Inspect all returned events; choose the latest event at/before cutoff.

    Never search backwards for a more attractive or complete quote. A malformed
    response, provider status, saturated count limit or conflicting latest
    timestamp prevents mechanical acceptance. No partial-field forward fill is
    performed. The 60-second acquisition window is not a liquidity criterion.
    """
    if lookback_seconds <= 0 or count_cap <= 0:
        raise ValueError("Positive lookback and response count cap required")
    cutoff_ns = timestamp_ns(cutoff)
    start_ns = cutoff_ns - lookback_seconds * 10**9
    ric = block.get("universe", {}).get("ric")
    if not ric:
        raise ValueError("Missing response instrument")
    headers = [h["name"] for h in block.get("headers", [])]
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate response field names")
    data = block.get("data", [])
    if any(len(row) != len(headers) for row in data):
        raise ValueError("Response row width differs from its schema")
    result = {
        "ric": ric, "cutoff_utc": cutoff, "returned_events": len(data),
        "returned_fields": headers, "provider_status": block.get("status"),
        "adjustments": block.get("adjustments"), "qos": block.get("qos"),
        "events_after_cutoff": 0, "events_before_window": 0,
        "non_quote_events": 0, "invalid_event_timestamps": 0,
        "duplicate_event_timestamps": 0, "latest_timestamp_rows": 0,
        "selected": None, "event_age_seconds": None, "issues": [],
        "mechanical_pass": False,
    }
    issues = result["issues"]
    if block.get("status"):
        issues.append("provider_status_requires_review")
    if len(data) >= count_cap:
        issues.append("possible_count_truncation")
    if block.get("adjustments") != ["unadjusted"]:
        issues.append("unexpected_adjustments")
    missing = set(("DATE_TIME", "EVENT_TYPE", "BID", "ASK", "BIDSIZE", "ASKSIZE")) - set(headers)
    if missing:
        issues.append("missing_fields:" + ",".join(sorted(missing)))
    events = []
    seen = set()
    for cells in data:
        row = dict(zip(headers, cells))
        if row.get("EVENT_TYPE") != "quote":
            result["non_quote_events"] += 1
            continue
        try:
            stamp = timestamp_ns(row["DATE_TIME"])
        except (KeyError, ValueError, TypeError):
            result["invalid_event_timestamps"] += 1
            continue
        if stamp in seen:
            result["duplicate_event_timestamps"] += 1
        seen.add(stamp)
        if stamp > cutoff_ns:
            result["events_after_cutoff"] += 1
        elif stamp < start_ns:
            result["events_before_window"] += 1
        else:
            events.append((stamp, row))
    if result["invalid_event_timestamps"]:
        issues.append("invalid_event_timestamps")
    if result["events_after_cutoff"] or result["events_before_window"]:
        issues.append("response_outside_requested_window")
    if result["non_quote_events"]:
        issues.append("unexpected_non_quote_events")
    if not events:
        issues.append("no_quote_in_window")
        return result
    latest = max(stamp for stamp, _ in events)
    candidates = [row for stamp, row in events if stamp == latest]
    result["latest_timestamp_rows"] = len(candidates)
    # A sequence field cannot break ties until its ordering semantics are known.
    state_fields = ("BID", "ASK", "BIDSIZE", "ASKSIZE", "SOURCE_DATETIME", "QUALIFIERS")
    states = {json.dumps({k: row.get(k) for k in state_fields}, sort_keys=True) for row in candidates}
    if len(states) != 1:
        issues.append("ambiguous_latest_timestamp")
        return result
    selected = candidates[0]
    result["selected"] = selected
    result["event_age_seconds"] = (cutoff_ns - latest) / 10**9
    issues.extend(quote_issues(selected, cutoff_ns))
    if selected.get("QUALIFIERS"):
        issues.append("quote_qualifiers_require_review")
    result["mechanical_pass"] = not issues
    return result


def pair_diagnostics(option: dict, underlying: dict, thresholds: list[int]) -> dict:
    """Report bounded-age pairing, without claiming simultaneous observations."""
    if option["cutoff_utc"] != underlying["cutoff_utc"]:
        raise ValueError("Option and underlying cutoffs must match")
    left, right = option["selected"], underlying["selected"]
    skew = None
    if left is not None and right is not None:
        skew = abs(timestamp_ns(left["DATE_TIME"]) - timestamp_ns(right["DATE_TIME"])) / 10**9
    passes = {
        str(threshold): bool(
            option["mechanical_pass"] and underlying["mechanical_pass"]
            and option["event_age_seconds"] <= threshold
            and underlying["event_age_seconds"] <= threshold
            and skew <= threshold
        ) for threshold in thresholds
    }
    return {
        "option_ric": option["ric"], "cutoff_utc": option["cutoff_utc"],
        "underlying_ric": underlying["ric"], "timestamp_skew_seconds": skew,
        "option_event_age_seconds": option["event_age_seconds"],
        "underlying_event_age_seconds": underlying["event_age_seconds"],
        "passes_age_checks": passes,
    }
