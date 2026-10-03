"""Prospective sampling clocks and complete-window checks.

This module contains no market observations or service calls. Its clock rules
do not certify historical receipt times, exchange entitlements or execution.
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from .event_calendar import EventCalendar


def instant(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError('An explicit time zone is required')
    return result.astimezone(timezone.utc)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def sample_plan(config, calendar_config):
    calendar = EventCalendar.from_config(calendar_config)
    zone = ZoneInfo(config['timezone'])
    if config['timezone'] != 'America/New_York':
        raise ValueError('Only the reviewed New York calendar is supported')
    days = [date.fromisoformat(d) for d in config['observation_dates']]
    if not days or days != sorted(set(days)):
        raise ValueError('Dates must be nonempty, unique and ordered')
    expiry = date.fromisoformat(config['expiry'])
    calendar.session(expiry)
    cutoff_time = time.fromisoformat(config['cutoff_local'])
    if cutoff_time.tzinfo is not None:
        raise ValueError('The cutoff must use the separately specified local zone')
    lag = positive_integer(config['release_lag_minutes'], 'release lag')
    quote_lookback = positive_integer(config['quote_lookback_seconds'], 'quote lookback')
    funding_lookback = positive_integer(config['funding_lookback_seconds'], 'funding lookback')
    positive_integer(config['event_count_cap'], 'response cap')
    positive_integer(config['funding_chunk_seconds'], 'funding chunk')
    if config['primary_quote_age_seconds'] > quote_lookback:
        raise ValueError('The quote screen exceeds the acquired lookback')
    positive_integer(config['primary_quote_age_seconds'], 'quote age screen')
    positive_integer(config['primary_funding_age_seconds'], 'funding age screen')
    if config['primary_funding_age_seconds'] > funding_lookback:
        raise ValueError('The funding screen exceeds the acquired lookback')
    ex_dates = sorted(date.fromisoformat(d) for d in config['scheduled_ex_dates'])
    if any(days[0] <= d <= expiry for d in ex_dates):
        raise ValueError('A scheduled cash event intersects the proposed no-cash horizon')
    result = []
    for day in days:
        session = calendar.session(day)
        cutoff = datetime.combine(day, cutoff_time, zone).astimezone(timezone.utc)
        if not session.opens_at < cutoff < session.closes_at or day >= expiry:
            raise ValueError('Cutoff must precede both session close and expiry')
        quote_start = cutoff - timedelta(seconds=quote_lookback)
        funding_start = cutoff - timedelta(seconds=funding_lookback)
        if min(quote_start, funding_start) < session.opens_at:
            raise ValueError('The acquisition window begins before the reviewed session')
        result.append(dict(date=day.isoformat(), cutoff_utc=cutoff.isoformat(),
            quote_start_utc=quote_start.isoformat(), funding_start_utc=funding_start.isoformat(),
            opens_at_utc=session.opens_at.astimezone(timezone.utc).isoformat(),
            release_at_utc=(session.closes_at+timedelta(minutes=lag)).astimezone(timezone.utc).isoformat(),
            share_settlement_date=calendar.settlement_after(day).day.isoformat()))
    return result


def require_prospective(plan, frozen_at):
    frozen = instant(frozen_at)
    if not plan or frozen >= instant(plan[0]['opens_at_utc']):
        raise ValueError('Freeze the protocol before the first observation session opens')


def available_dates(plan, now):
    current = instant(now)
    return [p['date'] for p in plan if instant(p['release_at_utc']) <= current]


def require_day(plan, day, now):
    entries = [p for p in plan if p['date'] == day]
    if len(entries) != 1:
        raise ValueError('The requested date is outside the frozen sample')
    if day not in available_dates(plan, now):
        raise ValueError('The prescribed completed-session release time has not arrived')
    return entries[0]


def require_complete_sample(plan, acquisitions, now):
    """Gate evaluation on complete date coverage, including failed acquisitions.

    A failed request remains a documented missing observation. Passing this
    check permits an outcome audit; it does not admit every quote to pricing.
    """
    expected = [p['date'] for p in plan]
    if available_dates(plan, now) != expected:
        raise ValueError('Wait for the complete prescribed observation window')
    if sorted(acquisitions) != expected:
        raise ValueError('Retain one final acquisition record for every prescribed date')
    for day, record in acquisitions.items():
        if record['date'] != day or record['status'] not in ('retained_review_pending', 'partial_evidence_retained', 'session_failed_evidence_retained'):
            raise ValueError('Missing, unfinished or mislabelled date evidence')


def request_plan(config, slot, option_rics):
    if not option_rics or len(set(option_rics)) != len(option_rics) or config['underlying'] in option_rics:
        raise ValueError('Distinct option identities and a separate underlying are required')
    common = dict(eventTypes=['quote'], adjustments=['unadjusted'], count=config['event_count_cap'])
    result = [dict(name='options', method='historical_pricing.events', arguments=dict(common,
        universe=sorted(option_rics)+[config['underlying']], fields=config['quote_fields'],
        start=slot['quote_start_utc'], end=slot['cutoff_utc']))]
    start, end = instant(slot['funding_start_utc']), instant(slot['cutoff_utc'])
    index = 0
    while start < end:
        stop = min(end, start+timedelta(seconds=config['funding_chunk_seconds']))
        for label in ('term', 'front'):
            result.append(dict(name=f'{label}_{index:02d}',method='historical_pricing.events',
                arguments=dict(common, universe=config[label+'_rics'],fields=config[label+'_fields'],
                               start=start.isoformat(),end=stop.isoformat())))
        start, index = stop, index+1
    return result
