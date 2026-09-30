"""Declared deterministic quote/rate fixtures, not market observations."""

import copy

import pytest

from qf_research.market_data.dense_quotes import inspect_funding_quote, merge_chunks, slice_window
from test_quote_review import CUTOFF, block, event


def stamp(second):
    return f'2026-09-15T19:44:{second:02d}Z'


def chunk(start, end, rows):
    return dict(start=stamp(start), end=stamp(end), block=block(rows))


def test_boundary_deduplication_preserves_distinct_ties_and_multiplicity():
    tie = event(stamp(30))
    other = event(stamp(30), size=300)
    chunks = [chunk(0, 30, [tie, tie, other]), chunk(30, 59, [tie, tie, other, event(stamp(59))])]
    original = copy.deepcopy(chunks)
    merged = merge_chunks(chunks, 100)
    assert len(merged['block']['data']) == 4
    assert merged['exact_boundary_duplicates_removed'] == 3
    assert not merged['issues'] and chunks == original


def test_conflicting_boundary_retains_both_states_and_holds_coverage():
    chunks = [chunk(0, 30, [event(stamp(30))]),
              chunk(30, 59, [event(stamp(30), bid=0.8)])]
    merged = merge_chunks(chunks, 100)
    assert len(merged['block']['data']) == 2
    assert merged['issues'][0]['reason'] == 'boundary_response_disagreement'
    assert slice_window(merged, stamp(40), 20)['status']
    assert 'status' not in slice_window(merged, stamp(59), 10)


def test_source_cap_remains_blocking_after_filtering_and_merging():
    chunks = [chunk(0, 30, [event(stamp(10)), event(stamp(20))]),
              chunk(30, 59, [event(stamp(50))])]
    merged = merge_chunks(chunks, 2)
    assert slice_window(merged, stamp(25), 10)['status']
    assert 'status' not in slice_window(merged, stamp(59), 10)


def test_joined_window_can_exceed_individual_source_cap_without_truncation():
    chunks = [chunk(0, 30, [event(stamp(i)) for i in (5, 10, 20)]),
              chunk(30, 59, [event(stamp(i)) for i in (35, 40, 50)])]
    merged = merge_chunks(chunks, 4)
    assert len(merged['block']['data']) == 6 and not merged['issues']


def test_missing_chunk_is_not_hidden_by_an_otherwise_complete_latest_state():
    chunks = [chunk(0, 20, [event(stamp(10))]), chunk(30, 59, [event(stamp(50))])]
    merged = merge_chunks(chunks, 100)
    assert slice_window(merged, stamp(40), 25)['status']


def test_rolling_slice_never_uses_future_quotes():
    merged = merge_chunks([chunk(0, 59, [event(stamp(10)), event(stamp(40))])], 100)
    sliced = slice_window(merged, stamp(30), 30)
    assert len(sliced['data']) == 1 and sliced['data'][0][0] == stamp(10)
    with pytest.raises(ValueError, match='coverage'):
        slice_window(merged, CUTOFF, 60)


def test_overlapping_interiors_schema_changes_and_wrong_instruments_fail():
    with pytest.raises(ValueError, match='overlap'):
        merge_chunks([chunk(0, 40, []), chunk(30, 59, [])], 100)
    c = chunk(30, 59, [])
    c['block']['universe']['ric'] = 'OTHER'
    with pytest.raises(ValueError, match='instrument'):
        merge_chunks([chunk(0, 30, []), c], 100)
    c = chunk(30, 59, [])
    c['block']['headers'].pop()
    with pytest.raises(ValueError, match='schema'):
        merge_chunks([chunk(0, 30, []), c], 100)


def rate_block(rows):
    fields = ['DATE_TIME', 'EVENT_TYPE', 'BID', 'ASK', 'SOURCE_DATETIME', 'QUALIFIERS']
    return dict(universe={'ric': 'FIXTURE_OIS'}, headers=[dict(name=k) for k in fields], data=rows)


def rate_row(bid=-0.2, ask=-0.1, when=None, source=None):
    when = when or stamp(59)
    return [when, 'quote', bid, ask, source or when, None]


def test_negative_and_zero_rates_do_not_need_equity_sizes():
    for bid, ask in [(-0.2, -0.1), (0, 0)]:
        result = inspect_funding_quote(rate_block([rate_row(bid, ask)]), CUTOFF, [60])
        assert not result['issues'] and result['mechanical_age_checks']['60']
        assert not result['discount_curve_constructed']


def test_latest_missing_or_reversed_rate_cannot_be_replaced_by_older_quote():
    for bid, ask in [(None, 1), (2, 1), (float('nan'), 1)]:
        rows = [rate_row(when=stamp(58)), rate_row(bid, ask)]
        result = inspect_funding_quote(rate_block(rows), CUTOFF, [60])
        assert not result['mechanical_age_checks']['60']
        assert result['latest_candidates'][0]['DATE_TIME'] == stamp(59)


def test_source_clock_anomaly_is_reported_without_arbitrary_tolerance():
    result = inspect_funding_quote(rate_block([rate_row(source='2026-09-15T19:44:59.001Z')]), CUTOFF, [60])
    assert 'source_after_event_requires_review' in result['issues']
    assert not result['mechanical_age_checks']['60']


def test_empty_tied_and_qualified_funding_states_are_held():
    empty = inspect_funding_quote(rate_block([]), CUTOFF, [60])
    assert 'no_historical_quote' in empty['issues']
    tied = inspect_funding_quote(rate_block([rate_row(), rate_row(-0.3, -0.1)]), CUTOFF, [60])
    assert 'unresolved_latest_state' in tied['issues']
    row = rate_row()
    row[-1] = 'UNREVIEWED'
    assert 'unreviewed_qualifier' in inspect_funding_quote(rate_block([row]), CUTOFF, [60])['issues']
