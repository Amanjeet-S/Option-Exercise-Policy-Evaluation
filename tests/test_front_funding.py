"""As-of selection guards use declared records, never substitute market observations."""
from copy import deepcopy

import pytest
from qf_research.market_data.front_funding import inspect_front_factor


def block():
    return {'universe': {'ric': 'FIXTURE'}, 'headers': [{'name': s} for s in
        ['DATE_TIME', 'EVENT_TYPE', 'DISC_FACT', 'ZERO_YLD1', 'QUALIFIERS']],
        'data': [['2026-09-10T19:40:00.000000001Z', 'quote', .95, 3., 'fixture'],
                 ['2026-09-10T19:39:59.999999999Z', 'quote', .99, 2., 'fixture']]}


def inspect(b):
    return inspect_front_factor(b, '2026-09-10T19:40:00Z', [0, 60], count_cap=10)


def test_nanosecond_cutoff_excludes_future_and_keeps_age_exact():
    r = inspect(block())
    assert r['factor'] == .99
    assert r['event_age_seconds'] == 1e-9
    assert r['mechanical_age_checks'] == {'0': False, '60': True}
    assert not r['historical_cash_dates_verified'] and not r['empirical_calibration_accepted']


@pytest.mark.parametrize('factor', [None, 0., -1., float('inf'), True])
def test_invalid_latest_does_not_fall_back(factor):
    b = block()
    b['data'][1][2] = factor
    b['data'].append(['2026-09-10T19:39:00Z', 'quote', .98, 2., 'fixture'])
    r = inspect(b)
    assert r['factor'] is None and not r['mechanical_age_checks']['60']
    assert 'invalid_discount_factor' in r['issues']


def test_conflicting_ties_are_held_but_identical_duplicates_agree():
    b = block()
    b['data'].append(deepcopy(b['data'][1]))
    assert inspect(b)['factor'] == .99
    b['data'][-1][2] = .98
    assert 'unresolved_latest_state' in inspect(b)['issues']


def test_source_status_and_saturation_do_not_pass():
    b = block()
    b['status'] = {'error': 'fixture'}
    assert inspect(b)['factor'] is None
    b.pop('status')
    r = inspect_front_factor(b, '2026-09-10T19:40:00Z', [60], count_cap=2)
    assert 'possibly_saturated_history' in r['issues']


def test_empty_or_all_future_history_has_no_factor():
    b = block()
    b['data'] = b['data'][:1]
    assert 'no_historical_factor' in inspect(b)['issues']
    b['data'] = []
    assert inspect(b)['factor'] is None


def test_increasing_discount_factor_and_negative_yield_are_allowed():
    b = block()
    b['data'][1][2:4] = [1.01, -2.]
    assert inspect(b)['factor'] == 1.01
