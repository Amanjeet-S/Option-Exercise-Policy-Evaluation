"""Financial invariants for funding-path attribution on mathematical fixtures."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
import json
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.funding_sensitivity import lower_corner_components
from qf_research.market_data.settlement_funding import SettlementFunding, build_premium_normalised_problem
from qf_research.market_data.settlement_comparison import (
    spot_relative_curve, funding_paths, check_normalised_equivalence, matched_changes, restore_problem,
)
from qf_research.market_data.valuation_inputs import DiscountCurve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def curve():
    ds = tuple(date(2026, 9, 9) + timedelta(days=n) for n in (0, 8, 40, 90))
    return DiscountCurve(ds[0], ds, (1., 1.001, .98, .96))


def test_spot_restriction_preserves_every_daily_ratio(curve):
    spot = date(2026, 9, 14)
    relative = spot_relative_curve(curve, spot)
    for i in range((curve.dates[-1]-spot).days+1):
        day = spot + timedelta(days=i)
        assert relative.discount(day) == pytest.approx(curve.between(spot, day), abs=3e-16)
    assert relative.anchor == spot
    with pytest.raises(ValueError, match='extrapolation'):
        relative.discount(spot-timedelta(days=1))


@pytest.mark.parametrize('spot', [date(2026, 9, 8), date(2027, 1, 1)])
def test_uncovered_spot_is_rejected(curve, spot):
    with pytest.raises(ValueError, match='domain'):
        spot_relative_curve(curve, spot)


def test_front_replacement_preserves_term_ratios(curve):
    h, s, u, v = map(date.fromisoformat, ['2026-09-11', '2026-09-14', '2026-10-19', '2026-10-30'])
    nodes = [dict(accrual_end='2026-11-30', lower=.95, midpoint=.96, upper=.97)]
    paths = funding_paths(curve, h, s, .995, nodes)
    before, after = paths['retained_lagged'], paths['front_replacement']
    assert before.factor(v)/before.factor(u) == pytest.approx(after.factor(v)/after.factor(u), abs=3e-16)
    assert after.factor(v)/before.factor(v) == pytest.approx(.995/before.settlement_to_spot)
    assert before.factor(h) == after.factor(h) == 1.
    assert all(paths['hybrid_'+k].settlement_to_spot == .995 for k in ('lower', 'midpoint', 'upper'))
    assert paths['hybrid_lower'].factor(v) < paths['hybrid_midpoint'].factor(v) < paths['hybrid_upper'].factor(v)


def test_reversed_node_paths_fail(curve):
    with pytest.raises(ValueError, match='Reversed'):
        funding_paths(curve, date(2026,9,11), date(2026,9,14), .999,
                      [dict(accrual_end='2026-11-30', lower=.98, midpoint=.96, upper=.97)])


def problems(curve):
    cal = EventCalendar.from_config(json.loads((ROOT / 'configs/us_event_calendar_2026-09-16.json').read_text()))
    h, s = date(2026,9,11), date(2026,9,14)
    args = (datetime(2026,9,10,15,40,tzinfo=NEW_YORK), date(2026,10,16), 100., 100., .2, 'call',
            (DatedDividend(date(2026,9,18),date(2026,10,30), 2., 'Mathematical fixture'),))
    f = SettlementFunding(h, spot_relative_curve(curve, s), curve.between(h,s))
    return (build_dated_problem(cal, curve, *args, premium_payment_date=h),
            build_premium_normalised_problem(cal, f, *args, premium_payment_date=h))


def test_equivalence_checks_and_development_quote_roundtrip(curve):
    full, normalised = problems(curve)
    assert check_normalised_equivalence(full, normalised) < 1e-13
    raw = dict(bid=1., ask=2., model=asdict(normalised.model), option=asdict(normalised.option),
               decisions=[asdict(d) for d in normalised.decisions])
    result = restore_problem(json.loads(json.dumps(raw)))
    assert result.model == normalised.model and result.decisions == normalised.decisions
    assert not hasattr(result, 'split')
    raw['ask'] = raw['bid']
    with pytest.raises(ValueError, match='quote width'):
        restore_problem(raw)


@pytest.mark.parametrize('change', ['time', 'cash', 'factor'])
def test_equivalence_does_not_hide_clock_or_cash_changes(curve, change):
    full, p = problems(curve)
    if change == 'time':
        p = replace(p, decisions=(replace(p.decisions[0], time=p.decisions[0].time/2),) + p.decisions[1:])
    elif change == 'cash':
        p = replace(p, model=replace(p.model, dividends=(replace(p.model.dividends[0], discounted_cash=4.),)))
    else:
        p = replace(p, decisions=(replace(p.decisions[0], settlement_factor=.8),) + p.decisions[1:])
    with pytest.raises(ValueError, match='Normalis'):
        check_normalised_equivalence(full,p)


def rows():
    result = []
    for name, put in [('retained_lagged', 4.), ('front_replacement', 4.1), ('hybrid_midpoint', 3.9)]:
        d = lower_corner_components(call_premium=3., ownership_premium=100., terminal_strike_premium=100.,
            expected_cash_premium=0., put_premium=put, cap_allowance=0., numerical_allowance=.01)
        result.append(dict(cutoff='2026-09-10T19:40:00Z', strike=100., dividend_model='fixture', variant=name,
            underlying_bid=99., underlying_ask=101., call_bid=2., call_ask=4., put_bid=3., put_ask=5.,
            decomposition=d, lower_minus_put_ask=d['conditional_lower_bound']-5.,
            numerical_screens={'fixture_check':True}, rectangle={'volatility_lower':.1,'volatility_upper':.2}))
    return result


def test_front_and_term_changes_telescope_without_rounding():
    r = matched_changes(rows())
    assert r[0]['gap_change'] + r[1]['gap_change'] == pytest.approx(r[2]['gap_change'], abs=1e-15)
    assert all(abs(x['accounting']['identity_residual']) < 1e-14 for x in r)


def test_same_strike_at_another_cutoff_is_never_matched():
    rs = rows()
    rs[1]['cutoff'] = '2026-09-10T19:40:01Z'
    output = matched_changes(rs)
    assert sum(r['status']=='computed' for r in output) == 1
    assert next(r for r in output if r['status']=='computed')['comparison'] == 'total'


def test_duplicate_or_changed_quote_comparisons_fail():
    rs = rows()
    with pytest.raises(ValueError, match='Duplicate'):
        matched_changes(rs + [deepcopy(rs[0])])
    rs[1]['put_ask'] = 6.
    with pytest.raises(ValueError, match='observed quote'):
        matched_changes(rs)


def test_failed_numerical_check_remains_visible_in_accounting():
    rs = rows()
    rs[1]['numerical_screens']['fixture_check'] = False
    result = matched_changes(rs)
    assert result[0]['status']=='computed' and not result[0]['both_numerical_screens_pass']
    assert result[2]['both_numerical_screens_pass']
