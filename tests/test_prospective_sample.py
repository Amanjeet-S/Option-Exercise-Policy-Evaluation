"""Invented protocol fixtures test clock and coverage safeguards, not markets."""
from copy import deepcopy
import pytest
from qf_research.market_data.prospective_sample import (
    sample_plan, require_prospective, available_dates, require_day,
    require_complete_sample, request_plan,
)


@pytest.fixture
def setup():
    calendar=dict(timezone='America/New_York',core_open='09:30:00',core_close='16:00:00',
        coverage_start='2026-09-01',coverage_end='2026-12-31',dtc_schedule_through='2026-12-31',
        trading_holidays={},settlement_holidays={'2026-10-12':'fixture banking closure'},
        early_closes={'2026-11-27':'13:00:00'})
    config=dict(timezone='America/New_York',observation_dates=['2026-10-09','2026-10-13'],
        expiry='2026-10-16',cutoff_local='15:45:00',release_lag_minutes=60,
        quote_lookback_seconds=60,funding_lookback_seconds=600,funding_chunk_seconds=300,
        event_count_cap=1000,primary_quote_age_seconds=1,primary_funding_age_seconds=300,
        scheduled_ex_dates=['2026-09-18','2026-12-18'],underlying='FIXTURE',
        quote_fields=['BID','ASK'],term_rics=['TERM'],front_rics=['FRONT'],
        term_fields=['BID','ASK'],front_fields=['DISC_FACT'])
    return config,calendar


def test_local_clock_and_banking_holiday_have_distinct_roles(setup):
    plan=sample_plan(*setup)
    assert plan[0]['cutoff_utc']=='2026-10-09T19:45:00+00:00'
    assert plan[0]['release_at_utc']=='2026-10-09T21:00:00+00:00'
    assert plan[0]['share_settlement_date']=='2026-10-13'


def test_late_registration_and_early_collection_rejected(setup):
    plan=sample_plan(*setup)
    require_prospective(plan,'2026-10-09T13:29:59+00:00')
    with pytest.raises(ValueError,match='Freeze'):require_prospective(plan,plan[0]['opens_at_utc'])
    assert available_dates(plan,'2026-10-09T20:59:59+00:00')==[]
    with pytest.raises(ValueError,match='release'):require_day(plan,'2026-10-09','2026-10-09T20:59:59+00:00')
    assert require_day(plan,'2026-10-09','2026-10-09T21:00:00+00:00')==plan[0]
    with pytest.raises(ValueError,match='outside'):require_day(plan,'2026-10-10','2026-10-14T21:00:00+00:00')


@pytest.mark.parametrize('change', ['duplicate','unsorted','holiday','cash','late_cutoff','boolean','uncovered'])
def test_invalid_designs_fail_before_requests(setup,change):
    config,calendar=deepcopy(setup)
    if change=='duplicate':config['observation_dates']*=2
    elif change=='unsorted':config['observation_dates'].reverse()
    elif change=='holiday':calendar['trading_holidays']['2026-10-09']='fixture closure'
    elif change=='cash':config['scheduled_ex_dates'].append('2026-10-16')
    elif change=='late_cutoff':config['cutoff_local']='16:01:00'
    elif change=='boolean':config['release_lag_minutes']=True
    else:config['expiry']='2027-01-15'
    with pytest.raises(ValueError):sample_plan(config,calendar)


def test_daylight_saving_and_early_close(setup):
    config,calendar=deepcopy(setup)
    config.update(observation_dates=['2026-11-27'],expiry='2026-12-04',cutoff_local='12:45:00')
    plan=sample_plan(config,calendar)
    assert plan[0]['cutoff_utc']=='2026-11-27T17:45:00+00:00'
    assert plan[0]['release_at_utc']=='2026-11-27T19:00:00+00:00'


def test_complete_window_includes_failed_dates_and_rejects_omissions(setup):
    plan=sample_plan(*setup)
    records={r['date']:dict(date=r['date'],status='partial_evidence_retained') for r in plan}
    require_complete_sample(plan,records,'2026-10-13T21:00:00+00:00')
    with pytest.raises(ValueError,match='complete'):require_complete_sample(plan,records,'2026-10-12T21:00:00+00:00')
    records.pop(plan[-1]['date'])
    with pytest.raises(ValueError,match='every'):require_complete_sample(plan,records,'2026-10-13T21:00:00+00:00')


def test_request_intervals_preserve_boundaries_and_do_not_request_outcomes(setup):
    cfg,_=setup;slot=sample_plan(*setup)[0]
    requests=request_plan(cfg,slot,['PUT','CALL'])
    assert len(requests)==5
    assert requests[0]['arguments']['universe']==['CALL','PUT','FIXTURE']
    term=[r['arguments'] for r in requests if r['name'].startswith('term')]
    assert term[0]['end']==term[1]['start']
    assert term[-1]['end']==slot['cutoff_utc']
    assert all(r['arguments']['adjustments']==['unadjusted'] for r in requests)
