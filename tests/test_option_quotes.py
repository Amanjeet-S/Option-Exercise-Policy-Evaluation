"""Hand-written parser fixtures, not observed quotes or empirical evidence."""

import copy

import pytest

from qf_research.market_data.option_quotes import (
    inspect_quote_block, pair_diagnostics, response_blocks, timestamp_ns,
)


CUTOFF = "2026-09-15T19:45:00Z"
FIELDS = ["DATE_TIME", "EVENT_TYPE", "BID", "ASK", "BIDSIZE", "ASKSIZE", "SOURCE_DATETIME", "QUALIFIERS"]


def block(rows):
    return {"universe": {"ric": "TEST_ONLY"}, "adjustments": ["unadjusted"],
            "headers": [{"name": name} for name in FIELDS], "data": rows}


def event(stamp="2026-09-15T19:44:59Z", bid=1.0, ask=1.1):
    return [stamp, "quote", bid, ask, 1, 2, None, None]


def inspect(rows, **kwargs):
    return inspect_quote_block(block(rows), CUTOFF, 60, **{"count_cap": 100, **kwargs})


def test_nanosecond_cutoff_and_explicit_timezone():
    assert timestamp_ns("2026-09-15T15:45:00-04:00") == timestamp_ns(CUTOFF)
    assert timestamp_ns("2026-09-15T19:45:00.000000001Z") - timestamp_ns(CUTOFF) == 1
    with pytest.raises(ValueError, match="timezone"):
        timestamp_ns("2026-09-15T19:45:00")


def test_future_event_never_selected_even_one_nanosecond_later():
    result = inspect([event("2026-09-15T19:45:00.000000001Z"), event()])
    assert result["selected"]["DATE_TIME"] == "2026-09-15T19:44:59Z"
    assert result["events_after_cutoff"] == 1
    assert not result["mechanical_pass"]


def test_selection_is_latest_not_response_order_and_does_not_mutate_input():
    payload = block([event("2026-09-15T19:44:57Z"), event(), event("2026-09-15T19:44:58Z")])
    original = copy.deepcopy(payload)
    result = inspect_quote_block(payload, CUTOFF, 60, 100)
    assert result["event_age_seconds"] == 1
    assert result["mechanical_pass"]
    assert payload == original


def test_missing_latest_bid_is_not_replaced_by_an_earlier_quote():
    result = inspect([event(bid=None), event("2026-09-15T19:44:58Z")])
    assert result["selected"]["BID"] is None
    assert "missing_bid" in result["issues"]
    assert not result["mechanical_pass"]


def test_zero_bid_is_distinct_from_missing_bid():
    assert inspect([event(bid=0)])["mechanical_pass"]
    assert not inspect([event(bid=None)])["mechanical_pass"]


@pytest.mark.parametrize("bid,ask,issue", [(2, 1, "crossed_market"), (-1, 1, "non_positive_bid"),
                                          (0, 0, "non_positive_ask"), (float("inf"), 1, "invalid_bid")])
def test_invalid_quotes_are_flagged(bid, ask, issue):
    result = inspect([event(bid=bid, ask=ask)])
    assert issue in result["issues"]
    assert not result["mechanical_pass"]


def test_ambiguous_latest_timestamp_is_not_arbitrarily_resolved():
    result = inspect([event(bid=0.9), event(bid=1)])
    assert result["selected"] is None
    assert "ambiguous_latest_timestamp" in result["issues"]


def test_identical_tied_quotes_remain_counted():
    result = inspect([event(), event()])
    assert result["mechanical_pass"]
    assert result["returned_events"] == 2
    assert result["duplicate_event_timestamps"] == 1
    assert result["latest_timestamp_rows"] == 2


def test_count_cap_and_provider_status_prevent_acceptance():
    assert "possible_count_truncation" in inspect([event()], count_cap=1)["issues"]
    payload = block([event()])
    payload["status"] = {"code": "partial_response"}
    result = inspect_quote_block(payload, CUTOFF, 60, 100)
    assert not result["mechanical_pass"]


def test_source_time_after_cutoff_and_qualifier_are_retained():
    row = event()
    row[-2:] = ["2026-09-15T19:45:01Z", "TEST_CONDITION"]
    result = inspect([row])
    assert "source_timestamp_after_cutoff" in result["issues"]
    assert "quote_qualifiers_require_review" in result["issues"]
    assert result["selected"]["QUALIFIERS"] == "TEST_CONDITION"


def test_window_boundary_and_empty_window():
    result = inspect([event("2026-09-15T19:44:00Z")])
    assert result["event_age_seconds"] == 60
    assert inspect([])["selected"] is None
    assert "no_quote_in_window" in inspect([event("2026-09-15T19:43:59Z")])["issues"]


def test_pair_requires_both_ages_and_shared_cutoff():
    option = inspect([event("2026-09-15T19:44:55Z")])
    underlying = inspect([event()])
    pair = pair_diagnostics(option, underlying, [1, 5, 15])
    assert pair["timestamp_skew_seconds"] == 4
    assert pair["passes_age_checks"] == {"1": False, "5": True, "15": True}
    underlying["cutoff_utc"] = "2026-09-15T19:44:00Z"
    with pytest.raises(ValueError, match="cutoffs"):
        pair_diagnostics(option, underlying, [1, 5])


def test_malformed_response_schema_is_rejected():
    payload = block([event()[:-1]])
    with pytest.raises(ValueError, match="width"):
        inspect_quote_block(payload, CUTOFF, 60, 100)
    payload = block([event()])
    payload["headers"][1] = payload["headers"][0]
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_quote_block(payload, CUTOFF, 60, 100)
    assert response_blocks({"universe": {}}) == [{"universe": {}}]
    with pytest.raises(ValueError, match="shape"):
        response_blocks("unexpected")
