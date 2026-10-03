"""Chunk coverage and fixed-clock observations for a development quote panel.

There is no price-outcome selection. A joined response has no independent API
row cap: completeness is checked on each original request before slicing.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
import json
import math

from .option_quotes import timestamp_ns


def merge_chunks(chunks, count_cap):
    """Join adjacent inclusive requests, preserving ties and source defects."""
    if not chunks or count_cap <= 0:
        raise ValueError('Nonempty chunks and a positive source cap required')
    ordered = sorted(chunks, key=lambda x: timestamp_ns(x['start']))
    first = ordered[0]['block']
    headers = [h['name'] for h in first.get('headers', [])]
    if len(headers) != len(set(headers)) or 'DATE_TIME' not in headers:
        raise ValueError('Invalid chunk schema')
    ric = first.get('universe', {}).get('ric')
    if not ric:
        raise ValueError('Missing instrument identity')
    time_index = headers.index('DATE_TIME')
    rows, issues, counts = [], [], []
    previous_end = None
    duplicate_rows = 0
    for chunk in ordered:
        start, end = timestamp_ns(chunk['start']), timestamp_ns(chunk['end'])
        block = chunk['block']
        if start >= end or block.get('universe', {}).get('ric') != ric:
            raise ValueError('Wrong instrument or invalid chunk interval')
        if [h['name'] for h in block.get('headers', [])] != headers:
            raise ValueError('Chunk schema changed')
        if previous_end is not None:
            if start < previous_end:
                raise ValueError('Chunk interiors overlap')
            if start > previous_end:
                issues.append(dict(start_ns=previous_end, end_ns=start, reason='missing_chunk_interval'))
        local_issues = []
        data = block.get('data', [])
        if len(data) >= count_cap:
            local_issues.append('source_count_cap_reached')
        if block.get('status'):
            local_issues.append('provider_status_requires_review')
        if block.get('adjustments') != ['unadjusted']:
            local_issues.append('unexpected_adjustments')
        incoming = []
        for cells in data:
            if len(cells) != len(headers):
                raise ValueError('Chunk row width differs from schema')
            try:
                stamp = timestamp_ns(cells[time_index])
            except (TypeError, ValueError):
                local_issues.append('invalid_event_timestamp')
                continue
            if not start <= stamp <= end:
                local_issues.append('event_outside_requested_chunk')
                continue
            incoming.append((stamp, cells))
        if previous_end == start:
            before = Counter(json.dumps(cells) for t, cells in rows if t == start)
            after = Counter(json.dumps(cells) for t, cells in incoming if t == start)
            if before != after:
                issues.append(dict(start_ns=start, end_ns=start, reason='boundary_response_disagreement'))
            available = before.copy()
            kept = []
            for stamp, cells in incoming:
                key = json.dumps(cells)
                if stamp == start and available[key]:
                    available[key] -= 1
                    duplicate_rows += 1
                else:
                    kept.append((stamp, cells))
            incoming = kept
        rows.extend(incoming)
        issues.extend(dict(start_ns=start, end_ns=end, reason=r) for r in sorted(set(local_issues)))
        counts.append(dict(start_ns=start, end_ns=end, source_rows=len(data)))
        previous_end = end
    rows.sort(key=lambda x: x[0])
    return dict(block=dict(universe={'ric': ric}, headers=first['headers'], adjustments=['unadjusted'],
                           data=[cells for _, cells in rows]),
                times_ns=[t for t, _ in rows], issues=issues, source_chunks=counts,
                exact_boundary_duplicates_removed=duplicate_rows,
                start_ns=timestamp_ns(ordered[0]['start']), end_ns=previous_end)


def slice_window(joined, cutoff, lookback_seconds):
    """Return a historical slice plus all coverage defects intersecting it."""
    end = timestamp_ns(cutoff)
    start = end - lookback_seconds * 10**9
    if start < joined['start_ns'] or end > joined['end_ns']:
        raise ValueError('Rolling window is outside acquired coverage')
    lo, hi = bisect_left(joined['times_ns'], start), bisect_right(joined['times_ns'], end)
    block = dict(joined['block'], data=joined['block']['data'][lo:hi])
    problems = [r for r in joined['issues'] if r['start_ns'] <= end and r['end_ns'] >= start]
    if problems:
        block['status'] = {'source_chunk_issues': problems}
    return block


def inspect_funding_quote(block, cutoff, maximum_ages):
    """Inspect OTC rate records without equity size or positive-price rules.

    The numerical side order is inspected; rate units, economic conventions and
    historical rule revisions remain external evidence requirements.
    """
    end = timestamp_ns(cutoff)
    fields = [h['name'] for h in block.get('headers', [])]
    if len(fields) != len(set(fields)):
        raise ValueError('Duplicate funding fields')
    issues = []
    if block.get('status'):
        issues.append('source_coverage_or_provider_status')
    required = {'DATE_TIME', 'EVENT_TYPE', 'BID', 'ASK', 'SOURCE_DATETIME', 'QUALIFIERS'}
    missing = sorted(required - set(fields))
    if missing:
        issues.append('missing_fields:' + ','.join(missing))
    timed = []
    for cells in block.get('data', []):
        if len(cells) != len(fields):
            raise ValueError('Funding row width differs')
        row = dict(zip(fields, cells))
        if row.get('EVENT_TYPE') != 'quote':
            issues.append('non_quote_event')
            continue
        try:
            stamp = timestamp_ns(row['DATE_TIME'])
        except (TypeError, ValueError, KeyError):
            issues.append('invalid_event_timestamp')
            continue
        if stamp <= end:
            timed.append((stamp, row))
        else:
            issues.append('event_after_cutoff')
    latest = max((t for t, _ in timed), default=None)
    candidates = [r for t, r in timed if t == latest]
    if not candidates:
        issues.append('no_historical_quote')
    sources = []
    for row in candidates:
        valid = all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                    and math.isfinite(row[k]) for k in ('BID', 'ASK'))
        if not valid:
            issues.append('missing_or_invalid_rate')
        elif row['BID'] > row['ASK']:
            issues.append('reversed_numeric_sides')
        if row.get('QUALIFIERS') not in (None, ''):
            issues.append('unreviewed_qualifier')
        try:
            source = timestamp_ns(row['SOURCE_DATETIME'])
            sources.append(source)
            if source > end:
                issues.append('source_after_cutoff')
            if source > latest:
                issues.append('source_after_event_requires_review')
        except (TypeError, ValueError, KeyError):
            issues.append('missing_or_invalid_source_time')
    states = {json.dumps(r, sort_keys=True) for r in candidates}
    if len(states) > 1:
        issues.append('unresolved_latest_state')
    event_age = None if latest is None else (end - latest) / 10**9
    source_age = None if len(sources) != len(candidates) or not sources else (end - min(sources)) / 10**9
    return dict(ric=block.get('universe', {}).get('ric'), cutoff=cutoff,
                latest_candidates=candidates, event_age_seconds=event_age,
                source_age_seconds=source_age, issues=sorted(set(issues)),
                mechanical_age_checks={str(age): not issues and all(
                    v is not None and 0 <= v <= age for v in (event_age, source_age)) for age in maximum_ages},
                rate_units_verified=False, swap_conventions_verified=False,
                discount_curve_constructed=False)
