"""Specified analytical fixtures, independent continuation checks and limits."""

from dataclasses import replace
import math

import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.escrowed_dividends import CashDividend, Decision, DiscountFactors
from qf_research.optimal_stopping.cash_dividends import CashJumpModel, integral_price as immediate_integral
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, deterministic_value,
    european_reference, integral_price, finite_difference_price,
)


def fixture(rate=.04, sigma=.2, amount=4., ex=.371, pay=.49, lag=.013):
    discount = lambda t: math.exp(-rate*t)
    model = FundedCashModel(100., sigma, discount(lag), discount(lag),
                            (DividendCap(ex, amount*discount(pay)),))
    rights = tuple(FundedExercise(t, discount(t+lag)) for t in (.217, .369, .743, 1.))
    return model, rights


def test_state_quote_jump_and_ledger_identities():
    model, rights = fixture()
    assert model.quote(model.initial_state, model.initial_share_factor) == pytest.approx(100.)
    event = model.dividends[0]
    grid = np.array([0., 1., event.discounted_cash, 100.])
    after, receivable = model.jump(grid, event)
    np.testing.assert_allclose(after+receivable, grid)
    assert np.all(after[:3] == 0.)
    for right in rights:
        quote = model.quote(grid, right.settlement_factor)
        for kind, sign in (("call", 1), ("put", -1)):
            reward = model.reward(Option(100., 1., kind), right, grid)
            np.testing.assert_allclose(reward, right.settlement_factor*np.maximum(sign*(quote-100.), 0.), atol=1e-13)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-.03, .04])
def test_equal_constant_lags_cancel_in_no_dividend_quoted_price(kind, rate):
    model, rights = fixture(rate=rate)
    model = replace(model, dividends=())
    option = Option(100., 1., kind)
    expected = black_scholes(Market(100., rate, .2), option)
    reference = european_reference(model, option, rights[-1])
    assert reference.premium_quote == pytest.approx(expected, abs=1e-12)
    assert abs(integral_price(model, option, 1600, 400., (rights[-1],)).premium_quote-expected) < .0006
    assert abs(finite_difference_price(model, option, 1600, 800, 400., (rights[-1],)).premium_quote-expected) < .0006


@pytest.mark.parametrize("kind", ["call", "put"])
def test_irregular_events_pde_and_integral_against_conditioning(kind):
    model, rights = fixture()
    option = Option(100., 1., kind)
    oracle = european_reference(model, option, rights[-1])
    assert oracle.quadrature_error_estimate < 1e-8 and oracle.omitted_tail_bound < 1e-25
    for result in (integral_price(model, option, 1600, 400., (rights[-1],)),
                   finite_difference_price(model, option, 1600, 800, 400., (rights[-1],))):
        assert abs(result.present_value-oracle.present_value) < .0008
        assert result.premium_quote == result.present_value/model.premium_factor
    a = integral_price(model, option, 1600, 400., rights)
    b = finite_difference_price(model, option, 1600, 800, 400., rights)
    assert abs(a.present_value-b.present_value) < .0015


def test_expected_payment_parity_under_material_cap_probability():
    model, rights = fixture(rate=0., sigma=.6, amount=90., ex=.5)
    refs = [european_reference(model, Option(100., 1., kind), rights[-1]) for kind in ("call", "put")]
    assert .45 < refs[0].cap_probability < .5
    assert refs[0].expected_discounted_payment < 90.
    assert refs[0].present_value-refs[1].present_value == pytest.approx(
        model.initial_state-refs[0].expected_discounted_payment-100.*rights[-1].settlement_factor, abs=1e-9)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_zero_lag_reduces_to_original_cash_model(kind):
    model, rights = fixture(lag=0.)
    original = CashJumpModel(100., .2, DiscountFactors((0., 2.), (1., math.exp(-.08))),
                             (CashDividend(.371, .49, 4.),))
    option = Option(100., 1., kind)
    a = immediate_integral(original, option, 600, 400., tuple(Decision(d.time) for d in rights)).value
    b = integral_price(model, option, 600, 400., rights).present_value
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_nested_rights_common_propagation_and_negative_rates(kind):
    model, rights = fixture(rate=-.05)
    option = Option(100., 1., kind)
    common = tuple(d.time for d in rights)
    for solver in (lambda ds: integral_price(model, option, 800, 400., ds, propagation_times=common),
                   lambda ds: finite_difference_price(model, option, 800, 600, 400., ds, propagation_times=common)):
        values = [solver(ds).present_value for ds in ((rights[-1],), rights)]
        assert values[1] >= values[0]-1e-10


def test_expiry_event_order_and_zero_volatility():
    model = FundedCashModel(100., 0., .99, .98, (DividendCap(1., 10.),))
    after = FundedExercise(1., .97)
    before = FundedExercise(1., .97, "before")
    option = Option(100., 1., "call")
    assert deterministic_value(model, option, (after,)) == 0.
    assert deterministic_value(model, option, (before, after)) == 2.
    assert integral_price(model, option, 100, 400., (before, after)).premium_quote == pytest.approx(2/.98)
    assert finite_difference_price(model, option, 100, 10, 400., (before, after)).present_value == 2.


def test_multiple_event_zero_volatility_enumeration():
    model = FundedCashModel(100., 0., .99, 1.01, (DividendCap(.33, 4.), DividendCap(.71, 100.)))
    rights = (FundedExercise(.2, .97), FundedExercise(.5, .96), FundedExercise(1., .94))
    for kind, expected in (("call", 2.), ("put", 94.)):
        option = Option(100., 1., kind)
        assert deterministic_value(model, option, rights) == pytest.approx(expected)
        assert finite_difference_price(model, option, 100, 20, 400., rights).present_value == pytest.approx(expected)


def test_premium_clock_changes_quote_only():
    model, rights = fixture()
    alternative = replace(model, premium_factor=1.02)
    option = Option(100., 1., "call")
    a = integral_price(model, option, 400, 400., rights)
    b = integral_price(alternative, option, 400, 400., rights)
    assert a.present_value == b.present_value
    assert a.premium_quote/b.premium_quote == pytest.approx(1.02/model.premium_factor)


def test_post_expiry_event_has_no_effect():
    model, rights = fixture()
    other = replace(model, dividends=model.dividends+(DividendCap(1.5, 80.),))
    assert integral_price(model, Option(100., 1.), 400, 400., rights) == integral_price(other, Option(100., 1.), 400, 400., rights)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_funding_unit_scaling_preserves_quote_and_scales_pv(kind):
    model, rights = fixture()
    scale = .973
    other = replace(model, initial_share_factor=model.initial_share_factor*scale,
                    premium_factor=model.premium_factor*scale,
                    dividends=tuple(DividendCap(e.ex_time, e.discounted_cash*scale) for e in model.dividends))
    scaled_rights = tuple(replace(d, settlement_factor=d.settlement_factor*scale) for d in rights)
    option = Option(100., 1., kind)
    for solver in (lambda m, ds, upper: integral_price(m, option, 500, upper, ds),
                   lambda m, ds, upper: finite_difference_price(m, option, 500, 400, upper, ds)):
        a, b = solver(model, rights, 400.), solver(other, scaled_rights, 400.*scale)
        assert b.present_value == pytest.approx(a.present_value*scale, abs=2e-12)
        assert b.premium_quote == pytest.approx(a.premium_quote, abs=2e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_numerical_expiry_jump_and_pre_event_right(kind):
    model = FundedCashModel(100., .2, .99, .98, (DividendCap(1., 10.),))
    terminal = FundedExercise(1., .97)
    before = replace(terminal, phase="before")
    option = Option(100., 1., kind)
    oracle = european_reference(model, option, terminal)
    if kind == "call":
        with_before = european_reference(replace(model, dividends=()), option, terminal).present_value
        assert with_before > oracle.present_value
    else:
        with_before = oracle.present_value
    for solver in (lambda ds: integral_price(model, option, 1600, 400., ds),
                   lambda ds: finite_difference_price(model, option, 1600, 800, 400., ds)):
        assert abs(solver((terminal,)).present_value-oracle.present_value) < .0005
        assert abs(solver((before, terminal)).present_value-with_before) < .0005


@pytest.mark.parametrize("action", [
    lambda m, ds: replace(m, initial_share_factor=0.),
    lambda m, ds: replace(m, premium_factor=float("nan")),
    lambda m, ds: replace(m, volatility=-.2),
    lambda m, ds: DividendCap(0., 4.),
    lambda m, ds: FundedExercise(.5, -1.),
    lambda m, ds: integral_price(m, Option(100., 1.), 400, 400., ds[:-1]),
    lambda m, ds: integral_price(m, Option(100., 1.), 400, 400., ds+(replace(ds[-1], settlement_factor=.8),)),
    lambda m, ds: integral_price(m, Option(100., 1.), 400, 400., ds+(FundedExercise(.5, .98, "before"),)),
    lambda m, ds: integral_price(m, Option(100., 1.), 400, 50., ds),
    lambda m, ds: finite_difference_price(m, Option(100., 1.), 400, 0, 400., ds),
    lambda m, ds: finite_difference_price(m, Option(100., 1.), 400, 100, 400., ds, propagation_times=(1.1,)),
])
def test_invalid_terms_rejected(action):
    with pytest.raises(ValueError):
        action(*fixture())
