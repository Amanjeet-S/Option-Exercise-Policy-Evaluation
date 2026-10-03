"""Mechanical as-of selection of provider-derived factors, without date inference."""
import json
import math

from .option_quotes import timestamp_ns


def inspect_front_factor(block, cutoff, maximum_ages, *, count_cap):
    """Keep the latest raw state, including ties and invalid values.

    Age is measured against the returned event timestamp only. Neither source
    freshness nor the cash date represented by the factor is established here.
    Later rows in a retained historical window cannot enter a past cutoff.
    """
    end = timestamp_ns(cutoff)
    fields = [h['name'] for h in block.get('headers', [])]
    if len(fields) != len(set(fields)):
        raise ValueError('Duplicate front-factor fields')
    if isinstance(count_cap, bool) or not isinstance(count_cap, int) or count_cap <= 0:
        raise ValueError('Require a positive event count cap')
    if not maximum_ages or any(isinstance(x, bool) or not isinstance(x, (float, int)) or
            not math.isfinite(x) or x < 0 for x in maximum_ages):
        raise ValueError('Invalid maximum event ages')
    issues, timed = [], []
    if block.get('status'):
        issues.append('provider_status_requires_review')
    if len(block.get('data', [])) >= count_cap:
        issues.append('possibly_saturated_history')
    missing = {'DATE_TIME', 'EVENT_TYPE', 'DISC_FACT', 'ZERO_YLD1', 'QUALIFIERS'} - set(fields)
    if missing:
        issues.append('missing_fields:' + ','.join(sorted(missing)))
    for cells in block.get('data', []):
        if len(cells) != len(fields):
            raise ValueError('Front-factor row width differs')
        row = dict(zip(fields, cells))
        try:
            stamp = timestamp_ns(row['DATE_TIME'])
        except (KeyError, TypeError, ValueError):
            issues.append('invalid_event_timestamp')
            continue
        if stamp <= end:
            timed.append((stamp, row))
    latest = max((t for t, _ in timed), default=None)
    candidates = [r for t, r in timed if t == latest]
    if not candidates:
        issues.append('no_historical_factor')
    if len({json.dumps(r, sort_keys=True) for r in candidates}) > 1:
        issues.append('unresolved_latest_state')
    for row in candidates:
        if row.get('EVENT_TYPE') != 'quote':
            issues.append('unexpected_event_type')
        factor = row.get('DISC_FACT')
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
            issues.append('invalid_discount_factor')
        value = row.get('ZERO_YLD1')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append('invalid_zero_yield')
    age = None if latest is None else (end - latest) / 10**9
    return dict(ric=block.get('universe', {}).get('ric'), cutoff=cutoff,
        latest_candidates=candidates, event_age_seconds=age, issues=sorted(set(issues)),
        mechanical_age_checks={str(a): not issues and age is not None and 0 <= age <= a for a in maximum_ages},
        factor=None if issues else candidates[0]['DISC_FACT'], source_freshness_verified=False,
        historical_cash_dates_verified=False, qualifier_semantics_verified=False,
        empirical_calibration_accepted=False)
