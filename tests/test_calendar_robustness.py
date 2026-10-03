"""Declared arithmetic and calendar fixtures, not observed option quotes."""
from datetime import date, datetime
import json
from pathlib import Path
import pytest

from qf_research.market_data.calendar_robustness import mesh_changes, compare_calendars
from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK, act365f
from qf_research.market_data.exercise_calendar_sensitivity import core_session_schedules
from qf_research.optimal_stopping.funded_cash import FundedExercise


def mesh(value,space=0.,time=0.,interaction=0.,domain=0.):
    return dict(base=value,space_only=value+space,time_only=value+time,
                fine=value+space+time+interaction,wide=value+space+time+interaction+domain)


def test_space_and_time_cancellation_does_not_pass_independent_screens():
    p=mesh(10.,space=.02,time=-.02)
    assert p['fine']==p['base']
    d=compare_calendars({'terminal_only':mesh(9.),'core_close':p},
                        ['terminal_only','core_close'],10.,9.,11.,.01)
    assert not d['calendars'][1]['space_screen']
    assert not d['calendars'][1]['time_screen']


def test_interaction_is_not_misattributed_to_domain_or_exercise():
    c=mesh_changes(mesh(10.,space=.01,time=.02,interaction=.003,domain=.004))
    assert c['interaction']==pytest.approx(.003)
    assert c['domain']==pytest.approx(.004)
    assert c['space_at_fine_time']==pytest.approx(.013)
    assert c['time_at_fine_space']==pytest.approx(.023)


def test_common_anchor_and_exercise_changes_telescope_separately():
    prices={'core_close':mesh(10.01),'core_5_minutes':mesh(10.04)}
    d=compare_calendars(prices,list(prices),10.,9.,11.,.02)
    assert d['anchor_change']==pytest.approx(.01)
    r=d['calendars'][-1]
    assert r['exercise_change_from_close']==pytest.approx(.03)
    assert r['total_change_from_retained']==pytest.approx(.04)
    assert abs(r['accounting_residual'])<1e-14


def test_quote_compatibility_requires_numerical_margin():
    prices={'core_close':mesh(10.5),'core_5_minutes':mesh(10.995)}
    d=compare_calendars(prices,list(prices),10.5,10.,11.,.01)
    assert d['calendars'][0]['quote_with_margin']
    assert not d['calendars'][1]['quote_with_margin']


def test_negative_calendar_increment_is_retained_and_screened():
    prices={'core_close':mesh(10.),'core_5_minutes':mesh(9.9)}
    d=compare_calendars(prices,list(prices),10.,9.,11.,.01)
    assert d['increments'][0]['minimum_increment']==pytest.approx(-.1)
    assert not d['increments'][0]['ordering_with_allowance']
    assert all(x==0. for x in d['increments'][0]['changes'].values())


@pytest.mark.parametrize('bad',[{'base':1.},mesh(-1.),mesh(float('nan')),mesh(float('inf'))])
def test_incomplete_or_invalid_mesh_prices_rejected(bad):
    with pytest.raises(ValueError):mesh_changes(bad)


@pytest.mark.parametrize('order',[['core_close','core_close'],['terminal_only'],['core_close','absent']])
def test_calendar_coverage_must_match_declared_order(order):
    with pytest.raises(ValueError):compare_calendars({'core_close':mesh(1.)},order,1.,.5,1.5,.01)


def test_five_minute_rights_preserve_dates_nesting_and_post_event_phase():
    root=Path(__file__).resolve().parents[1]
    cal=EventCalendar.from_config(json.loads((root/'configs/us_event_calendar_2026-09-16.json').read_text()))
    obs=datetime(2026,9,17,15,40,6,tzinfo=NEW_YORK);expiry=date(2026,9,21)
    # Flat numerical funding fixture on attributed source calendar dates.
    baseline=tuple(FundedExercise(act365f(obs,t),.99,'after') for t in cal.research_decisions(obs,expiry))
    schedules=core_session_schedules(cal,obs,expiry,baseline,(60,30,15,5))
    values=list(schedules.values())
    assert all(set(a)<set(b) for a,b in zip(values,values[1:]))
    ex=act365f(obs,cal.model_ex_event(date(2026,9,18)))
    assert next(r for r in values[-1] if r.time==ex).phase=='after'
    assert all(r.settlement_factor==.99 for rights in values for r in rights)
    assert min(r.time for r in values[-1])>0
    assert values[-1][-1]==baseline[-1]
