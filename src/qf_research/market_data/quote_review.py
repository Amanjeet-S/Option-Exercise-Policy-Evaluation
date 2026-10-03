"""Versioned, offline interpretation of the SPY feasibility pilot.

This policy uses retained instrument metadata, not a universal interpretation of
Reuters condition codes. Source: Option Exercise-Policy Evaluation's lseg_field_review.md. It identifies
price agreement without inventing an ordering or a unique size for tied events.
"""

from collections import Counter
import json
import re

from .option_quotes import inspect_quote_block, quote_issues, timestamp_ns


POLICY_VERSION = "spy_quote_review_2026-09-16_v1"
_TOKEN = re.compile(r"([^\[\]]*)\[([A-Z0-9_]+)\]")


def metadata_context(ric: str, metadata: dict) -> dict:
    """Validate the narrow instrument rule set actually reviewed for this pilot."""
    rule = metadata.get("MarketRulesData", {})
    result = {"ric": ric, "rule_name": rule.get("ruleName"),
              "rule_version": rule.get("ruleVersionId"), "kind": None}
    qualifiers = list(metadata.get("Qualifiers", {}).values())
    if ric == "SPY" and rule.get("ruleName") == "US_PCQ_EQ_112_1_1":
        required = [
            {"code": "R", "fid": "118", "exchcode": "60", "desc": "Regular - Resume"},
            {"code": "BOLO", "desc": "Best Odd Lot Quote", "fact": "REU_QUAL"},
        ]
        if not all(any(all(q.get(k) == v for k, v in wanted.items()) for q in qualifiers)
                   for wanted in required):
            raise ValueError("SPY qualifier definitions differ from the reviewed policy")
        result["kind"] = "spy_consolidated"
    elif ric.startswith("SPY") and rule.get("ruleName") == "US_OPQ_EQ_115_1_1":
        result["kind"] = "spy_opra_option"
    else:
        raise ValueError(f"Unreviewed instrument metadata context: {ric}")
    return result


def qualifier_class(value, context: dict) -> str:
    """Classify whole observed tokens, leaving unknown or mixed strings on hold.

    No qualifier reported is distinct from an explicit normal-market code.
    An empty-valued PRC_QL_CD token is not automatically approved by this policy.
    """
    if value is None or value == "":
        return "no_qualifier_reported"
    if not isinstance(value, str):
        return "unreviewed"
    token = _TOKEN.fullmatch(value.strip())
    if token and context["kind"] == "spy_consolidated":
        code, field = token.groups()
        if field == "PRC_QL_CD" and code.strip() == "R":
            return "regular_cqs"
        if field == "USER" and code.strip() == "BOLO":
            return "odd_lot_stream"
    return "unreviewed"


def review_quote_block(block: dict, metadata: dict, cutoff: str,
                       lookback_seconds: int, count_cap: int) -> dict:
    """Inspect the ordinary quote stream, preserving a tied candidate set.

    Known BOLO records belong to another stream and are counted separately.
    Every unknown latest condition remains blocking. No malformed, stale,
    crossed or otherwise undesirable ordinary quote is skipped to select an
    earlier one. All tied candidates must pass before price agreement is used.
    """
    original = inspect_quote_block(block, cutoff, lookback_seconds, count_cap)
    context = metadata_context(original["ric"], metadata)
    headers = original["returned_fields"]
    rows = [dict(zip(headers, cells)) for cells in block.get("data", [])]
    classes = [qualifier_class(row.get("QUALIFIERS"), context) for row in rows]
    keep = [not (kind == "odd_lot_stream" and row.get("EVENT_TYPE") == "quote")
            for row, kind in zip(rows, classes)]
    filtered = dict(block, data=[cells for cells, chosen in zip(block.get("data", []), keep) if chosen])
    base = inspect_quote_block(filtered, cutoff, lookback_seconds, count_cap)
    # Structural request defects must not disappear when separating quote streams.
    deferred = {"ambiguous_latest_timestamp", "quote_qualifiers_require_review"}
    structural = {"possible_count_truncation", "response_outside_requested_window",
                  "invalid_event_timestamps", "unexpected_non_quote_events"}
    issues = set(base["issues"]) - deferred
    issues.update(set(original["issues"]) & structural)
    for name in ("SOURCE_DATETIME", "QUALIFIERS"):
        if name not in headers:
            issues.add("missing_review_field:" + name)
    report = {
        "policy_version": POLICY_VERSION, "ric": original["ric"], "cutoff_utc": cutoff,
        "metadata_context": context, "original_returned_events": len(rows),
        "qualifier_class_counts": dict(sorted(Counter(classes).items())),
        "excluded_odd_lot_events": len(rows) - sum(keep),
        "latest_candidates": [], "price_consensus": None,
        "quote_state_resolved": False, "price_screen_pass": False,
        "event_age_seconds": None, "source_age_seconds_max": None,
        "source_time_ns_range": None, "latest_event_time_ns": None,
        "candidate_size_ranges": None, "issues": [],
        "empirical_acceptance": False,
    }
    cutoff_ns = timestamp_ns(cutoff)
    start_ns = cutoff_ns - lookback_seconds * 10**9
    in_window = []
    for row, chosen in zip(rows, keep):
        if not chosen or row.get("EVENT_TYPE") != "quote":
            continue
        try:
            t = timestamp_ns(row["DATE_TIME"])
        except (KeyError, TypeError, ValueError):
            continue
        if start_ns <= t <= cutoff_ns:
            in_window.append((t, row))
    if not in_window:
        report["issues"] = sorted(issues | {"no_ordinary_quote_in_window"})
        return report
    latest = max(t for t, _ in in_window)
    candidates = [row for t, row in in_window if t == latest]
    report["latest_candidates"] = candidates
    report["latest_event_time_ns"] = latest
    report["event_age_seconds"] = (cutoff_ns - latest) / 10**9
    source_times = []
    for row in candidates:
        issues.update(quote_issues(row, cutoff_ns))
        if qualifier_class(row.get("QUALIFIERS"), context) == "unreviewed":
            issues.add("latest_quote_qualifier_unreviewed")
        if not row.get("SOURCE_DATETIME"):
            issues.add("missing_source_timestamp")
        else:
            try:
                source_times.append(timestamp_ns(row["SOURCE_DATETIME"]))
            except (TypeError, ValueError):
                issues.add("invalid_source_timestamp")
    prices_agree = all(row.get("BID") == candidates[0].get("BID")
                       and row.get("ASK") == candidates[0].get("ASK") for row in candidates)
    if not prices_agree:
        issues.add("conflicting_latest_prices")
    elif not issues:
        # Equality is exact. No averaging, rounding or arbitrary tie breaking.
        report["price_consensus"] = {k: candidates[0][k] for k in ("BID", "ASK")}
    state_fields = ("BID", "ASK", "BIDSIZE", "ASKSIZE", "SOURCE_DATETIME", "QUALIFIERS")
    states = {json.dumps({k: row.get(k) for k in state_fields}, sort_keys=True) for row in candidates}
    report["quote_state_resolved"] = len(states) == 1
    if len(source_times) == len(candidates):
        report["source_time_ns_range"] = [min(source_times), max(source_times)]
        report["source_age_seconds_max"] = (cutoff_ns - min(source_times)) / 10**9
    if not issues:
        report["candidate_size_ranges"] = {
            k: [min(row[k] for row in candidates), max(row[k] for row in candidates)]
            for k in ("BIDSIZE", "ASKSIZE")
        }
    report["issues"] = sorted(issues)
    report["price_screen_pass"] = not issues
    return report


def reviewed_pair(option: dict, underlying: dict, thresholds: list[int]) -> dict:
    """Require both timestamp ages and both cross-instrument skews to pass."""
    if option["cutoff_utc"] != underlying["cutoff_utc"]:
        raise ValueError("Quote cutoffs differ")
    if not thresholds or any(isinstance(t, bool) or not isinstance(t, int) or t <= 0 for t in thresholds):
        raise ValueError("Positive integer age thresholds required")
    event_skew = source_skew = None
    if option["latest_event_time_ns"] is not None and underlying["latest_event_time_ns"] is not None:
        event_skew = abs(option["latest_event_time_ns"] - underlying["latest_event_time_ns"]) / 10**9
    left, right = option["source_time_ns_range"], underlying["source_time_ns_range"]
    if left is not None and right is not None:
        source_skew = max(abs(left[0] - right[1]), abs(left[1] - right[0])) / 10**9
    measures = [event_skew, source_skew, option["event_age_seconds"], underlying["event_age_seconds"],
                option["source_age_seconds_max"], underlying["source_age_seconds_max"]]
    passed = {
        str(t): bool(option["price_screen_pass"] and underlying["price_screen_pass"]
                     and all(v is not None and 0 <= v <= t for v in measures))
        for t in thresholds
    }
    return {
        "option_ric": option["ric"], "underlying_ric": underlying["ric"],
        "cutoff_utc": option["cutoff_utc"], "event_skew_seconds": event_skew,
        "source_skew_seconds_max": source_skew, "price_and_age_checks": passed,
        "unique_state_and_age_checks": {
            str(t): bool(passed[str(t)] and option["quote_state_resolved"]
                         and underlying["quote_state_resolved"]) for t in thresholds},
        "empirical_acceptance": False,
    }
