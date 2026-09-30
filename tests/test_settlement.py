"""Analytical accounting fixtures, not estimated market prices or rates."""

from datetime import date, timedelta
import math

import pytest

from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.settlement import (
    matched_exercise_ledger, quote_to_present_value, present_value_to_quote,
)

START = date(2026, 9, 9)
END = START + timedelta(days=365)
DECISION = date(2026, 10, 9)
SETTLE = date(2026, 10, 13)


def curve(rate):
    return DiscountCurve(START, (START, END), (1., math.exp(-rate)))


@pytest.mark.parametrize("rate", [-.04, 0., .05])
@pytest.mark.parametrize("kind,quote,sign", [("call", 105., 1), ("put", 95., -1)])
def test_physical_legs_equal_discounted_net_cash(rate, kind, quote, sign):
    ledger = matched_exercise_ledger(kind, quote, 100., DECISION, SETTLE, SETTLE)
    expected = math.exp(-rate * (SETTLE-START).days/365) * sign * (quote-100.)
    assert sum(leg.shares for leg in ledger.legs) == 0
    assert ledger.present_value(curve(rate), START) == pytest.approx(expected)
    assert ledger.exercise_reward(curve(rate), START) == pytest.approx(expected)


def test_at_the_money_has_no_funding_windfall():
    c = curve(.1)
    for kind in ("call", "put"):
        ledger = matched_exercise_ledger(kind, 100., 100., DECISION, SETTLE, SETTLE)
        assert ledger.exercise_reward(c, DECISION) == 0.
    # This tempting but inconsistent expression discounts only the strike.
    assert 100. - c.between(DECISION, SETTLE) * 100. > 0.


def test_out_of_money_signed_ledger_and_abandonment():
    ledger = matched_exercise_ledger("call", 90., 100., DECISION, SETTLE, SETTLE)
    assert ledger.present_value(curve(.04), START) < 0.
    assert ledger.exercise_reward(curve(.04), START) == 0.


def test_multiplier_is_explicit():
    one = matched_exercise_ledger("call", 110., 100., DECISION, SETTLE, SETTLE)
    hundred = matched_exercise_ledger("call", 110., 100., DECISION, SETTLE, SETTLE, units=100)
    assert hundred.present_value(curve(.04), START) == pytest.approx(100 * one.present_value(curve(.04), START))


def test_mismatched_settlements_not_silently_netted():
    with pytest.raises(ValueError, match="same date"):
        matched_exercise_ledger("call", 110., 100., DECISION, SETTLE, SETTLE+timedelta(days=1))


@pytest.mark.parametrize("rate", [-.04, 0., .05])
def test_premium_quote_and_prepaid_share_accounting(rate):
    c = curve(rate)
    premium_pv = quote_to_present_value(3., c, START, SETTLE)
    assert present_value_to_quote(premium_pv, c, START, SETTLE) == pytest.approx(3.)
    # At the decision, prepaid underlying and discounted strike use the same date.
    prepaid = quote_to_present_value(105., c, DECISION, SETTLE)
    reward = prepaid - quote_to_present_value(100., c, DECISION, SETTLE)
    ledger = matched_exercise_ledger("call", 105., 100., DECISION, SETTLE, SETTLE)
    assert c.between(START, DECISION) * reward == pytest.approx(ledger.present_value(c, START))


@pytest.mark.parametrize("quote", [float("nan"), float("inf"), -1., True])
def test_bad_prices_rejected(quote):
    with pytest.raises(ValueError):
        matched_exercise_ledger("call", quote, 100., DECISION, SETTLE, SETTLE)


@pytest.mark.parametrize("units", [0, -1, True, 1.5])
def test_bad_units_rejected(units):
    with pytest.raises(ValueError, match="Units"):
        matched_exercise_ledger("call", 105., 100., DECISION, SETTLE, SETTLE, units=units)


def test_no_discount_extrapolation_or_backdated_cash():
    with pytest.raises(ValueError, match="extrapolation"):
        quote_to_present_value(3., curve(.04), START, END+timedelta(days=1))
    with pytest.raises(ValueError, match="Ordered"):
        matched_exercise_ledger("call", 105., 100., DECISION, START, START)
    with pytest.raises(ValueError, match="Ordered"):
        quote_to_present_value(3., curve(.04), SETTLE, START)
