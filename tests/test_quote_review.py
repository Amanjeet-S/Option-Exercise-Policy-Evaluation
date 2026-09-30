"""Declared, deterministic test fixtures, not empirical market observations."""

import copy

import pytest

from qf_research.market_data.option_quotes import inspect_quote_block
from qf_research.market_data.quote_review import metadata_context, review_quote_block, reviewed_pair


CUTOFF = "2026-09-15T19:45:00Z"
FIELDS = ["DATE_TIME", "EVENT_TYPE", "BID", "ASK", "BIDSIZE", "ASKSIZE",
          "SOURCE_DATETIME", "RTL", "QUALIFIERS"]


def metadata(ric="SPY"):
    if ric != "SPY":
        return {"MarketRulesData": {"ruleName": "US_OPQ_EQ_115_1_1", "ruleVersionId": "fixture"}}
    return {"MarketRulesData": {"ruleName": "US_PCQ_EQ_112_1_1", "ruleVersionId": "fixture"},
            "Qualifiers": {
                "regular": {"code": "R", "fid": "118", "exchcode": "60", "desc": "Regular - Resume"},
                "odd_lot": {"code": "BOLO", "desc": "Best Odd Lot Quote", "fact": "REU_QUAL"}}}


def event(stamp="2026-09-15T19:44:59Z", *, source=None, bid=1.0, ask=1.1,
          size=100, rtl=16, qualifier="R  [PRC_QL_CD]"):
    return [stamp, "quote", bid, ask, size, 200, source or stamp, rtl, qualifier]


def block(rows, ric="SPY"):
    return {"universe": {"ric": ric}, "adjustments": ["unadjusted"],
            "headers": [{"name": f} for f in FIELDS], "data": rows}


def review(rows, ric="SPY", cap=100):
    return review_quote_block(block(rows, ric), metadata(ric), CUTOFF, 60, cap)


def test_scoped_regular_code_is_supported_without_changing_original_audit():
    payload = block([event()])
    assert not inspect_quote_block(payload, CUTOFF, 60, 100)["mechanical_pass"]
    assert review([event()])["price_screen_pass"]
    # The same displayed letter is not granted the same meaning for other instruments.
    assert not review([event()], "SPY_OPTION_TEST")["price_screen_pass"]
    assert review([event(qualifier=None)], "SPY_OPTION_TEST")["price_screen_pass"]


def test_conflicting_size_and_source_time_do_not_imply_conflicting_prices():
    rows = [event(size=100, rtl=65520), event(size=200, rtl=0, source="2026-09-15T19:44:58.999Z")]
    result = review(rows)
    assert result["price_screen_pass"]
    assert result["price_consensus"] == {"BID": 1.0, "ASK": 1.1}
    assert not result["quote_state_resolved"]
    assert result["candidate_size_ranges"]["BIDSIZE"] == [100, 200]
    assert len(result["latest_candidates"]) == 2
    assert result["source_age_seconds_max"] == pytest.approx(1.001)
    assert review(list(reversed(rows)))["price_consensus"] == result["price_consensus"]


def test_no_rtl_ranking_or_price_averaging():
    result = review([event(bid=1, rtl=32), event(bid=0.99, rtl=16)])
    assert "conflicting_latest_prices" in result["issues"]
    assert result["price_consensus"] is None
    assert not result["price_screen_pass"]


def test_numerically_equal_integer_and_float_prices_agree():
    assert review([event(bid=1), event(bid=1.0)])["price_screen_pass"]


def test_all_tied_candidates_must_be_valid():
    result = review([event(), event(size=0)])
    assert not result["price_screen_pass"]
    assert "non_positive_bidsize" in result["issues"]
    assert result["candidate_size_ranges"] is None


def test_known_odd_lot_stream_is_separate_and_original_is_unchanged():
    payload = block([event(qualifier="BOLO[USER]"), event("2026-09-15T19:44:58Z")])
    original = copy.deepcopy(payload)
    result = review_quote_block(payload, metadata(), CUTOFF, 60, 100)
    assert result["excluded_odd_lot_events"] == 1
    assert result["event_age_seconds"] == 2
    assert result["price_screen_pass"]
    assert payload == original


@pytest.mark.parametrize("qualifier", ["HALT[PRC_QL_CD]", "   [PRC_QL_CD]", "R[OTHER]",
                                       "R[PRC_QL_CD];BOLO[USER]", "unexpected", 42])
def test_unknown_latest_condition_cannot_be_skipped(qualifier):
    result = review([event(qualifier=qualifier), event("2026-09-15T19:44:58Z")])
    assert result["event_age_seconds"] == 1
    assert "latest_quote_qualifier_unreviewed" in result["issues"]
    assert not result["price_screen_pass"]


def test_count_cap_is_checked_before_stream_separation():
    result = review([event(qualifier="BOLO[USER]"), event("2026-09-15T19:44:58Z")], cap=2)
    assert "possible_count_truncation" in result["issues"]
    assert not result["price_screen_pass"]


def test_missing_source_time_or_qualifier_column_is_not_treated_as_good_data():
    row = event()
    row[6] = None
    assert "missing_source_timestamp" in review([row])["issues"]
    payload = block([event()])
    payload["headers"].pop()
    payload["data"][0].pop()
    result = review_quote_block(payload, metadata(), CUTOFF, 60, 100)
    assert "missing_review_field:QUALIFIERS" in result["issues"]


def test_dual_timestamp_age_rule_rejects_recent_record_with_old_source_time():
    option = review([event(source="2026-09-15T19:44:54Z", qualifier=None)], "SPY_OPTION_TEST")
    underlying = review([event()])
    result = reviewed_pair(option, underlying, [1, 5, 15])
    assert result["price_and_age_checks"] == {"1": False, "5": False, "15": True}


def test_price_pass_is_distinct_from_unique_quote_state_and_empirical_acceptance():
    option = review([event(qualifier=None)], "SPY_OPTION_TEST")
    underlying = review([event(size=100), event(size=200)])
    result = reviewed_pair(option, underlying, [5])
    assert result["price_and_age_checks"]["5"]
    assert not result["unique_state_and_age_checks"]["5"]
    assert result["empirical_acceptance"] is False


def test_metadata_drift_is_not_silently_accepted():
    changed = metadata()
    changed["Qualifiers"]["regular"]["desc"] = "Different meaning"
    with pytest.raises(ValueError, match="definitions differ"):
        metadata_context("SPY", changed)
    with pytest.raises(ValueError, match="Unreviewed"):
        metadata_context("OTHER", metadata())


def test_only_odd_lot_events_leave_no_ordinary_candidate():
    result = review([event(qualifier="BOLO[USER]")])
    assert not result["price_screen_pass"]
    assert result["price_consensus"] is None
    assert "no_ordinary_quote_in_window" in result["issues"]
