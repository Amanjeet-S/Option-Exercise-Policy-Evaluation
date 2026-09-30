"""Scaling identities and date restrictions using declared mathematical fixtures."""
from datetime import date, datetime
import json
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar, NEW_YORK
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.settlement_funding import SettlementFunding, build_premium_normalised_problem
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.funded_cash import european_reference, finite_difference_price

ROOT = Path(__file__).resolve().parents[1]


def case(kind='call', scale=.93, *, observation_day=date(2026, 9, 10), cash=3.):
    calendar = EventCalendar.from_config(json.loads((ROOT /
        'projects/01_optimal_stopping/configs/us_event_calendar_2026-09-16.json').read_text()))
    observation = datetime.combine(observation_day, datetime.min.time(), NEW_YORK).replace(hour=15, minute=45)
    h = calendar.settlement_after(observation_day).day
    s = calendar.settlement_after(observation_day, lag=2).day
    end = date(2026, 11, 30)
    # Deliberately large front scaling tests a mathematical invariance, not a rate estimate.
    bridge = 1.002
    spot_curve = DiscountCurve(s, (s, end), (1., .97))
    curve = DiscountCurve(observation_day, (observation_day, h, s, end),
                          (1., scale, scale * bridge, scale * bridge * .97))
    funding = SettlementFunding(h, spot_curve, bridge)
    dividends = (DatedDividend(date(2026, 9, 18), date(2026, 10, 30), cash,
                               'Declared mathematical fixture, not observed cash'),)
    args = (observation, date(2026, 10, 16), 100., 101., .23, kind, dividends)
    direct = build_dated_problem(calendar, curve, *args, premium_payment_date=h)
    normalised = build_premium_normalised_problem(calendar, funding, *args, premium_payment_date=h)
    return calendar, funding, args, direct, normalised


@pytest.mark.parametrize('kind', ['call', 'put'])
@pytest.mark.parametrize('scale', [.93, 1.07])
@pytest.mark.parametrize('cash', [0., 3.])
def test_full_curve_and_premium_units_give_identical_prices(kind, scale, cash):
    _, _, _, direct, normalised = case(kind, scale, cash=cash)
    assert direct.deliveries == normalised.deliveries
    assert direct.option == normalised.option
    assert normalised.model.initial_state == 100.
    for a, b in zip(direct.decisions, normalised.decisions):
        assert a.time == b.time
        assert a.settlement_factor / scale == pytest.approx(b.settlement_factor)
    for a, b in zip(direct.model.dividends, normalised.model.dividends):
        assert a.ex_time == b.ex_time
        assert a.discounted_cash / scale == pytest.approx(b.discounted_cash)
    # Scaling the entire spatial domain preserves the same numerical problem.
    full = finite_difference_price(direct.model, direct.option, 220, 160, 320. * scale, direct.decisions)
    reduced = finite_difference_price(normalised.model, normalised.option, 220, 160, 320., normalised.decisions)
    assert full.premium_quote == pytest.approx(reduced.premium_quote, abs=2e-11)
    assert full.present_value / scale == pytest.approx(reduced.present_value, abs=2e-11)
    e_full = european_reference(direct.model, direct.option, direct.decisions[-1])
    e_reduced = european_reference(normalised.model, normalised.option, normalised.decisions[-1])
    assert e_full.premium_quote == pytest.approx(e_reduced.premium_quote, abs=2e-11)
    assert reduced.present_value == reduced.premium_quote
    assert 'premium units' in normalised.cash_discount_clock
    assert not normalised.empirical_calibration_accepted and not normalised.premium_date_verified


@pytest.mark.parametrize('day', [date(2026, 9, 10), date(2026, 9, 11), date(2026, 9, 14),
                               date(2026, 9, 15), date(2026, 9, 17), date(2026, 9, 18)])
def test_development_dates_have_no_unidentified_cash_date(day):
    _, funding, _, _, normalised = case(observation_day=day)
    assert normalised.decisions[0].settlement_factor == 1.
    assert all(d == funding.settlement_date or d >= funding.spot_curve.anchor
               for _, d, _ in normalised.deliveries)
    if day == date(2026, 9, 18):
        assert not normalised.model.dividends


def test_unknown_between_settlement_and_spot_or_after_curve_is_rejected():
    _, funding, _, _, _ = case()
    with pytest.raises(ValueError, match='identified'):
        funding.factor(date(2026, 9, 12))
    with pytest.raises(ValueError, match='extrapolation'):
        funding.factor(date(2026, 12, 1))
    with pytest.raises(ValueError, match='identified'):
        funding.factor(date(2026, 9, 10))


def test_different_premium_date_is_rejected():
    calendar, funding, args, _, _ = case()
    with pytest.raises(ValueError, match='coincident'):
        build_premium_normalised_problem(calendar, funding, *args,
                                        premium_payment_date=funding.spot_curve.anchor)


@pytest.mark.parametrize('bridge', [0., -1., float('nan'), float('inf'), True])
def test_invalid_bridge_is_rejected(bridge):
    _, funding, _, _, _ = case()
    with pytest.raises(ValueError, match='finite and positive'):
        SettlementFunding(funding.settlement_date, funding.spot_curve, bridge)


def test_coincident_spot_and_settlement():
    _, funding, _, _, _ = case()
    same = SettlementFunding(funding.spot_curve.anchor, funding.spot_curve, 1.)
    assert same.factor(same.settlement_date) == 1.
    with pytest.raises(ValueError, match='factor one'):
        SettlementFunding(funding.spot_curve.anchor, funding.spot_curve, .99)


@pytest.mark.parametrize('amount,occurs,error', [(None, True, 'amount'), (2., None, 'occurrence')])
def test_normalisation_does_not_resolve_unknown_dividends(amount, occurs, error):
    calendar, funding, args, _, _ = case()
    dividend = DatedDividend(date(2026, 9, 18), date(2026, 10, 30), amount, 'Unresolved', occurs)
    with pytest.raises(ValueError, match=error):
        build_premium_normalised_problem(calendar, funding, *args[:-1], (dividend,),
                                        premium_payment_date=funding.settlement_date)


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_fixed_volatility_bridge_comparative_statics(kind):
    calendar, funding, args, _, _ = case(kind)
    values = []
    for bridge in [.98, 1., 1.02]:
        changed = SettlementFunding(funding.settlement_date, funding.spot_curve, bridge)
        p = build_premium_normalised_problem(calendar, changed, *args,
                                            premium_payment_date=funding.settlement_date)
        values.append(finite_difference_price(p.model, p.option, 240, 160, 320., p.decisions).premium_quote)
    assert (values[0] >= values[1] >= values[2] if kind == 'call'
            else values[0] <= values[1] <= values[2])
