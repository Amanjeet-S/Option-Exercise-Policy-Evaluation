"""Specified mathematical fixtures for the capped-payment cash-jump model."""

import math
import numpy as np
import pytest

from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.escrowed_dividends import CashDividend, Decision, DiscountFactors
from qf_research.optimal_stopping.cash_dividends import (
    CashJumpModel, deterministic_price, finite_difference_price, integral_price,
    lognormal_transition, one_event_european,
)


def model(rate=.03, sigma=.2, amount=4, ex=.5, pay=.625):
    return CashJumpModel(100, sigma, DiscountFactors((0., 2.), (1., math.exp(-rate*2))),
                         (CashDividend(ex, pay, amount),))


def test_jump_preserves_discounted_share_plus_receivable_and_zero_atom():
    m = model()
    event = m.dividends[0]
    cap = 4*math.exp(-.03*.625)
    states = np.array([0., 1., cap, 100.])
    after, paid = m.jump(states, event)
    np.testing.assert_allclose(after+paid, states, rtol=0, atol=1e-14)
    np.testing.assert_allclose(after[:3], 0)
    assert after[-1] == pytest.approx(100-cap)
    assert paid[1] == 1  # Paying four as well as flooring would create wealth.
    assert paid[-1]/m.curve.discount(event.pay_time) == pytest.approx(4)


def test_transition_integrates_affine_functions_and_preserves_zero():
    grid = np.linspace(0, 400, 401)
    expected = 2 + .7*grid
    for sd in (.01, .2, 1.2):
        np.testing.assert_allclose(lognormal_transition(grid, expected, sd, .7), expected, atol=2e-12, rtol=0)
    assert lognormal_transition(grid, expected, .2, .7)[0] == 2


@pytest.mark.parametrize("kind", ["call", "put"])
def test_transition_hinge_matches_exact_lognormal_integral(kind):
    grid = np.linspace(0, 400, 401)
    values = Option(100, 1, kind).payoff(grid)
    computed = lognormal_transition(grid, values, .2, 1 if kind == "call" else 0)
    for index in (20, 99, 100, 150, 400):
        expected = black_scholes(Market(grid[index], 0, .2), Option(100, 1, kind))
        assert computed[index] == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-.03, .03])
def test_no_dividend_reduction(kind, rate):
    original = model(rate=rate)
    m = CashJumpModel(100, .2, original.curve)
    option = Option(100, 1, kind)
    exact = black_scholes(Market(100, rate, .2), option)
    assert abs(finite_difference_price(m, option, 1600, 800, 400).value-exact) < .0005
    assert abs(integral_price(m, option, 1600, 400).value-exact) < .0003


@pytest.mark.parametrize("kind", ["call", "put"])
def test_one_event_independent_oracle_and_two_solvers(kind):
    m, option = model(), Option(100, 1, kind)
    reference = one_event_european(m, option)
    assert reference.quadrature_error_estimate < 1e-8
    assert reference.omitted_tail_bound < 1e-25
    assert abs(finite_difference_price(m, option, 1600, 800, 400).value-reference.value) < .0003
    assert abs(integral_price(m, option, 1600, 400).value-reference.value) < .0005


@pytest.mark.parametrize("kind", ["call", "put"])
def test_material_liquidation_probability_and_expected_payment_parity(kind):
    m, option = model(rate=0, sigma=.6, amount=90), Option(100, 1, kind)
    result = one_event_european(m, option)
    assert .45 < result.cap_probability < .5
    assert result.expected_discounted_payment < 80
    call = one_event_european(m, Option(100, 1, "call"))
    put = one_event_european(m, Option(100, 1, "put"))
    assert call.value-put.value == pytest.approx(-result.expected_discounted_payment, abs=1e-9)
    assert abs(integral_price(m, option, 2400, 1200).value-result.value) < .001
    assert abs(finite_difference_price(m, option, 2400, 1600, 1200).value-result.value) < .002


@pytest.mark.parametrize("kind", ["call", "put"])
def test_expiry_day_event_oracle_and_pre_exercise_phase(kind):
    m, option = model(rate=0, ex=1, pay=1.25, amount=10), Option(100, 1, kind)
    after = one_event_european(m, option).value
    assert abs(integral_price(m, option, 1600, 400).value-after) < 1e-10  # Grid-aligned terminal kinks.
    both = [Decision(1, "before")]
    result = integral_price(m, option, 1600, 400, both).value
    if kind == "call":
        expected = black_scholes(Market(100, 0, .2), option)
        assert result == pytest.approx(expected, abs=1e-10)
        assert result > after
    else:
        assert result == pytest.approx(after, abs=1e-10)


def test_zero_volatility_exact_liquidation_and_exercise():
    m = model(rate=0, sigma=0, amount=120)
    call, put = Option(95, 1, "call"), Option(95, 1, "put")
    rights = [Decision(.5, "before")]
    assert deterministic_price(m, call) == 0
    assert deterministic_price(m, call, rights) == 5
    assert deterministic_price(m, put) == 95
    assert finite_difference_price(m, call, 100, 100, 400, rights).value == 5
    assert integral_price(m, put, 100, 400).value == 95


@pytest.mark.parametrize("kind", ["call", "put"])
def test_post_expiry_dividends_cannot_change_this_models_dynamics(kind):
    m, option = model(), Option(100, 1, kind)
    extended = CashJumpModel(m.spot, m.volatility, m.curve, m.dividends+(CashDividend(1.5, 1.75, 80),))
    assert integral_price(m, option, 400, 400) == integral_price(extended, option, 400, 400)
    assert one_event_european(m, option) == one_event_european(extended, option)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_multiple_events_and_nested_rights(kind):
    base = model()
    m = CashJumpModel(100, .2, base.curve, (CashDividend(.25, .375, 3), CashDividend(.75, .875, 4)))
    option = Option(100, 1, kind)
    sparse = [Decision(.25, "before"), Decision(.75, "before")]
    richer = sparse + [Decision(.25), Decision(.5), Decision(.75)]
    pde = [finite_difference_price(m, option, 1600, 800, 400, d, damping_times=(.25, .5, .75)).value
           for d in (None, sparse, richer)]
    integral = [integral_price(m, option, 1600, 400, d, integration_times=(.25, .5, .75)).value
                for d in (None, sparse, richer)]
    assert np.min(np.diff(pde)) > -1e-10
    assert np.min(np.diff(integral)) > -1e-10
    assert abs(pde[-1]-integral[-1]) < .002


def test_nonuniform_event_calendar_not_rounded():
    m, option = model(ex=1/3, pay=.4), Option(100, 1)
    with pytest.raises(ValueError, match="align"):
        finite_difference_price(m, option, 400, 100, 400)
    assert integral_price(m, option, 800, 400).event_count == 1
    assert abs(integral_price(m, option, 1600, 400).value-one_event_european(m, option).value) < .0005


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delaying_payment_reduces_detached_value_at_positive_rates(kind):
    early, late, option = model(pay=.5), model(pay=1.5), Option(100, 1, kind)
    early_value, late_value = one_event_european(early, option).value, one_event_european(late, option).value
    assert late_value > early_value if kind == "call" else late_value < early_value


@pytest.mark.parametrize("invalid", [
    lambda: CashJumpModel(-1, .2, model().curve),
    lambda: CashJumpModel(100, -.2, model().curve),
    lambda: model().jump(-1, model().dividends[0]),
    lambda: model(pay=3),
    lambda: model(amount=float("nan")),
    lambda: model(pay=.4),
    lambda: integral_price(model(), Option(100, 1), 100, 90),
    lambda: integral_price(model(), Option(100, 1), 100, 400, [Decision(.25, "before")]),
    lambda: integral_price(model(), Option(100, 1), 100, 400, [Decision(1.5)]),
    lambda: integral_price(model(), Option(100, 1), 100, 400, [Decision(.5), Decision(.5)]),
    lambda: lognormal_transition([0, 1, 1], [0, 1, 2], .2, 1),
    lambda: lognormal_transition([0, 1, 2], [0, 1, 2], -.2, 1),
])
def test_invalid_or_ambiguous_inputs_are_rejected(invalid):
    with pytest.raises(ValueError):
        invalid()
