"""Deterministic validation fixtures, not observed or fitted market inputs."""

from datetime import date, timedelta
import math

import numpy as np
import pytest

from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.escrowed_dividends import (
    CashDividend, Decision, DiscountFactors, EscrowedModel,
    decision_schedule, european_price, finite_difference_price, tree_price,
)


def model(rate=0.03, volatility=0.2, dividends=None):
    curve = DiscountFactors((0.0, 0.75, 2.0), (1.0, math.exp(-rate * 0.75), math.exp(-rate * 2)))
    if dividends is None:
        dividends = (CashDividend(0.5, 0.625, 4.0), CashDividend(1.25, 1.375, 3.0))
    return EscrowedModel(100.0, volatility, curve, dividends)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-0.03, 0.0, 0.05])
def test_no_dividends_reduces_to_black_scholes(kind, rate):
    m = model(rate=rate, dividends=())
    option = Option(103.0, 1.0, kind)
    assert european_price(m, option) == pytest.approx(black_scholes(Market(100, rate, 0.2), option), abs=3e-13)


def test_dated_curve_adapter_preserves_rebased_factors():
    anchor = date(2026, 1, 1)
    dates = tuple(anchor + timedelta(days=d) for d in (0, 91, 183, 365, 730))
    dated = DiscountCurve(anchor, dates, (1.0, 0.994, 0.984, 0.961, 0.91))
    valuation = anchor + timedelta(days=31)
    converted = DiscountFactors.from_dated_curve(dated, valuation)
    for days in (0, 15, 60, 152, 250, 699):
        target = valuation + timedelta(days=days)
        assert converted.discount(days / 365) == pytest.approx(dated.between(valuation, target), abs=3e-15)


def test_delayed_payment_jump_and_entitlement_value():
    m = model()
    before, after = Decision(0.5, "before"), Decision(0.5)
    states = np.array([0.0, m.risky_spot, 200.0])
    jump = 4 * m.curve.discount(0.625) / m.curve.discount(0.5)
    np.testing.assert_allclose(m.stock(states, before) - m.stock(states, after), jump, atol=5e-14)
    assert jump < 4  # Paying at the ex-date would value a different entitlement.
    assert m.reserve(before) - m.reserve(after) == pytest.approx(4 * m.curve.discount(0.625))
    assert m.stock(m.risky_spot, Decision(0)) == pytest.approx(100)


@pytest.mark.parametrize("maturity", [0.25, 0.5, 1.0, 1.5])
def test_parity_counts_ex_dates_but_discounts_payment_dates(maturity):
    m = model()
    call, put = Option(100, maturity, "call"), Option(100, maturity, "put")
    detached = sum(d.amount * m.curve.discount(d.pay_time) for d in m.dividends if d.ex_time <= maturity)
    assert european_price(m, call) - european_price(m, put) == pytest.approx(
        m.spot - detached - 100 * m.curve.discount(maturity), abs=5e-14)


def test_common_schedule_retains_post_maturity_reserve():
    m = model()
    assert m.reserve(Decision(1.0)) == pytest.approx(3 * m.curve.discount(1.375))
    truncated = model(dividends=m.dividends[:1])
    assert truncated.risky_spot != m.risky_spot
    assert european_price(truncated, Option(100, 1, "call")) > european_price(m, Option(100, 1, "call"))


def test_nonpositive_effective_strike_and_zero_volatility():
    m = model(dividends=(CashDividend(1.5, 1.75, 20),))
    call, put = Option(1, 1, "call"), Option(1, 1, "put")
    assert european_price(m, call) == pytest.approx(100 - m.curve.discount(1))
    assert european_price(m, put) == 0
    deterministic = model(volatility=0)
    option = Option(100, 1, "call")
    assert european_price(deterministic, option) == pytest.approx(
        float(deterministic.reward(option, Decision(1), deterministic.risky_spot)))


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-0.03, 0.03])
def test_independent_solvers_against_exact_european(kind, rate):
    m = model(rate=rate)
    option = Option(100, 1, kind)
    exact = european_price(m, option)
    tree = tree_price(m, option, 1600).value
    pde = finite_difference_price(m, option, 1600, 800, 400).value
    assert abs(tree - exact) < 0.002
    assert abs(pde - exact) < 0.0006


@pytest.mark.parametrize("kind", ["call", "put"])
def test_fixed_calendar_cross_method_agreement_and_nested_rights(kind):
    m, option = model(), Option(100, 1, kind)
    sparse = (Decision(0.5, "before"),)
    rich = (Decision(0.25), Decision(0.5, "before"), Decision(0.5), Decision(0.75))
    tree = [tree_price(m, option, 1600, d).value for d in (None, sparse, rich)]
    # Use the same temporal propagator: extra damping alone is not an exercise right.
    pde = [finite_difference_price(m, option, 1600, 800, 400, d,
                                  damping_times=(0.25, 0.5, 0.75)).value for d in (None, sparse, rich)]
    assert tree[0] <= tree[1] <= tree[2]
    assert pde[0] <= pde[1] <= pde[2]
    assert abs(tree[-1] - pde[-1]) < 0.003


def test_pre_dividend_exercise_can_change_deterministic_call_value():
    m = model(rate=0, volatility=0, dividends=(CashDividend(0.5, 0.625, 10),))
    option = Option(95, 1, "call")
    assert tree_price(m, option, 8, [Decision(0.5)]).value == 0
    assert tree_price(m, option, 8, [Decision(0.5, "before")]).value == 5
    assert tree_price(m, option, 8, [Decision(0)]).value == 5


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expiry_day_entitlement_phase_is_explicit(kind):
    m = model(rate=0, dividends=(CashDividend(1, 1.25, 10),))
    option = Option(100, 1, kind)
    terminal = tree_price(m, option, 1600).value
    # At a coincident time, the two non-negative obstacles are applied after
    # conditioning, with the backward order after then before.
    both = [Decision(1, "before"), Decision(1)]
    revised = tree_price(m, option, 1600, both).value
    pde = finite_difference_price(m, option, 1600, 800, 400, both).value
    assert abs(revised - pde) < 0.003
    if kind == "call":
        assert revised > terminal
        expected = black_scholes(Market(90, 0, 0.2), Option(90, 1, "call"))
        assert abs(revised - expected) < 0.002
    else:
        assert revised == pytest.approx(terminal, abs=1e-12)


def test_positive_rate_non_dividend_call_has_no_early_exercise_gain():
    m, option = model(dividends=()), Option(100, 1, "call")
    decisions = [Decision(i / 16) for i in range(17)]
    assert tree_price(m, option, 800, decisions).value == pytest.approx(tree_price(m, option, 800).value, abs=1e-11)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_longer_maturity_contains_short_contract_rights_under_one_model(kind):
    m = model()
    short, long = Option(100, 0.5, kind), Option(100, 1, kind)
    calendar = [Decision(0.25), Decision(0.5, "before"), Decision(0.5)]
    # Same time increment and risky process; do not truncate the dividend reserve.
    assert tree_price(m, long, 1600, calendar).value >= tree_price(m, short, 800, calendar).value - 1e-12


@pytest.mark.parametrize("kind", ["call", "put"])
def test_low_volatility_short_maturity_off_grid_strike(kind):
    m, option = model(rate=-0.02, volatility=0.05), Option(100.3, 0.25, kind)
    exact = european_price(m, option)
    assert abs(tree_price(m, option, 1600).value - exact) < 0.0004
    assert abs(finite_difference_price(m, option, 3200, 800, 200).value - exact) < 0.0003


@pytest.mark.parametrize("invalid", [
    lambda: DiscountFactors((0., 1.), (1., 0.)),
    lambda: DiscountFactors((0., 0.), (1., 1.)),
    lambda: DiscountFactors((0., 1.), (0.99, 0.98)),
    lambda: CashDividend(0.5, 0.4, 2),
    lambda: CashDividend(0.5, 0.6, float("nan")),
    lambda: CashDividend(0.5, 0.6, -1),
    lambda: Decision(0.5, "unknown"),
    lambda: model(dividends=(CashDividend(0.5, 0.6, 200),)),
    lambda: model(dividends=(CashDividend(0.5, 3, 2),)),
    lambda: model(dividends=(CashDividend(0.5, 0.6, 1), CashDividend(0.5, 0.7, 1))),
    lambda: model().curve.discount(2.1),
])
def test_invalid_or_uncovered_inputs_rejected(invalid):
    with pytest.raises(ValueError):
        invalid()


@pytest.mark.parametrize("decisions", [
    [Decision(0.3, "before")], [Decision(1.1)], [Decision(0.5), Decision(0.5)],
])
def test_invalid_decision_rights_rejected(decisions):
    with pytest.raises(ValueError):
        decision_schedule(model(), Option(100, 1), decisions)


@pytest.mark.parametrize("solver", ["tree", "pde"])
def test_no_silent_rounding_of_exercise_dates(solver):
    m, option, decisions = model(), Option(100, 1), [Decision(1 / 3)]
    with pytest.raises(ValueError, match="align"):
        if solver == "tree":
            tree_price(m, option, 100, decisions)
        else:
            finite_difference_price(m, option, 100, 100, 400, decisions)
