"""Analytical validation inputs only, not observations or fitted market data."""

from dataclasses import replace
from fractions import Fraction
import math

import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, deterministic_value,
    european_reference, integral_price, finite_difference_price,
)
from qf_research.optimal_stopping.dividend_sensitivity import dividend_value_change_bound


def fixture(*, volatility=0., caps=(3.,), times=(.4,), premium=.97):
    baseline = FundedCashModel(100., volatility, 1., premium,
                              tuple(DividendCap(t, c) for t, c in zip(times, caps)))
    return baseline, (FundedExercise(1., 1.),)


@pytest.mark.parametrize('kind,strike,expected', [('call', 50., -5.), ('put', 150., 5.)])
def test_sharp_deterministic_bound_and_separate_premium_units(kind, strike, expected):
    baseline, rights = fixture()
    alternative = replace(baseline, dividends=(DividendCap(.4, 8.),))
    option = Option(strike, 1., kind)
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    change = deterministic_value(alternative, option, rights)-deterministic_value(baseline, option, rights)
    assert change == expected
    assert bound.present_value_lower_change <= change <= bound.present_value_upper_change
    assert expected in (bound.present_value_lower_change, bound.present_value_upper_change)
    assert bound.premium_quote_lower_change <= change/.97 <= bound.premium_quote_upper_change
    assert bound.discounted_cash_increase_bound == 5.
    assert bound.changed_events_in_horizon == 1


@pytest.mark.parametrize('kind,expected_interval', [('call', (-5., 3.)), ('put', (-3., 5.))])
def test_mixed_changes_keep_directional_budgets_and_reverse_consistency(kind, expected_interval):
    baseline, rights = fixture(caps=(2., 8.), times=(.3, .7))
    alternative = replace(baseline, dividends=(DividendCap(.3, 7.), DividendCap(.7, 5.)))
    option = Option(100., 1., kind)
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    reverse = dividend_value_change_bound(alternative, baseline, option, rights)
    assert (bound.present_value_lower_change, bound.present_value_upper_change) == expected_interval
    assert reverse.present_value_lower_change == -bound.present_value_upper_change
    assert reverse.present_value_upper_change == -bound.present_value_lower_change
    assert bound.discounted_cash_increase_bound == 5.
    assert bound.discounted_cash_decrease_bound == 3.
    assert bound.changed_events_in_horizon == 2


@pytest.mark.parametrize('kind', ['call', 'put'])
@pytest.mark.parametrize('cash', [8., 130.])
def test_independent_one_event_quadrature_with_diffusion_and_cap_risk(kind, cash):
    baseline, rights = fixture(volatility=.6, caps=(90.,))
    alternative = replace(baseline, dividends=(DividendCap(.4, cash),))
    option = Option(100., 1., kind)
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    first = european_reference(baseline, option, rights[-1])
    second = european_reference(alternative, option, rights[-1])
    assert first.cap_probability > .2
    change = second.present_value-first.present_value
    # Independent oracle has numerical quadrature error; it is not an exact certificate.
    tolerance = first.quadrature_error_estimate+second.quadrature_error_estimate+1e-8
    assert bound.present_value_lower_change-tolerance <= change <= bound.present_value_upper_change+tolerance


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_multiple_event_stopping_with_both_continuation_methods(kind):
    baseline, _ = fixture(volatility=.25, caps=(2., 8.), times=(.3, .7))
    alternative = replace(baseline, dividends=(DividendCap(.3, 7.), DividendCap(.7, 5.)))
    rights = tuple(FundedExercise(t, factor) for t, factor in ((.29, 1.02), (.69, .98), (1., 1.01)))
    option = Option(100., 1., kind)
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    for price in (
        lambda m: integral_price(m, option, 1200, 400., rights),
        lambda m: finite_difference_price(m, option, 1200, 600, 400., rights),
    ):
        change = price(alternative).present_value-price(baseline).present_value
        assert bound.present_value_lower_change-.002 <= change <= bound.present_value_upper_change+.002


def test_pre_event_exercise_can_remove_call_sensitivity():
    baseline, _ = fixture(caps=(3.,), times=(1.,))
    alternative = replace(baseline, dividends=(DividendCap(1., 8.),))
    rights = (FundedExercise(1., 1., 'before'), FundedExercise(1., 1.))
    option = Option(50., 1., 'call')
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    assert deterministic_value(baseline, option, rights) == deterministic_value(alternative, option, rights) == 50.
    assert bound.present_value_lower_change == -5.
    assert bound.present_value_upper_change == 0.


def test_absorbing_cap_can_make_bound_loose():
    baseline, rights = fixture(caps=(120.,))
    alternative = replace(baseline, dividends=(DividendCap(.4, 160.),))
    option = Option(100., 1., 'put')
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    assert deterministic_value(baseline, option, rights) == deterministic_value(alternative, option, rights) == 100.
    assert bound.present_value_upper_change == 40.


def test_identical_inputs_and_post_expiry_cash_have_zero_effect():
    baseline, rights = fixture(caps=(3., 8.), times=(.4, 1.1))
    alternative = replace(baseline, dividends=(DividendCap(.4, 3.), DividendCap(1.1, 80.)))
    option = Option(100., 1.)
    for model in (baseline, alternative):
        bound = dividend_value_change_bound(baseline, model, option, rights)
        assert bound.present_value_lower_change == bound.present_value_upper_change == 0.
        assert bound.changed_events_in_horizon == 0
    assert bound.changed_events_after_expiry == 1


def test_zero_cash_placeholder_and_expiry_event_are_included():
    baseline, rights = fixture(caps=(0.,), times=(1.,))
    alternative = replace(baseline, dividends=(DividendCap(1., 2.),))
    option = Option(50., 1., 'call')
    bound = dividend_value_change_bound(baseline, alternative, option, rights)
    assert bound.changed_events_in_horizon == 1
    assert bound.present_value_lower_change == -2.
    assert deterministic_value(alternative, option, rights)-deterministic_value(baseline, option, rights) == -2.


@pytest.mark.parametrize('field,value', [('quoted_spot', 101.), ('volatility', .1), ('initial_share_factor', .98), ('premium_factor', .98)])
def test_rejects_non_dividend_model_changes(field, value):
    baseline, rights = fixture()
    with pytest.raises(ValueError, match='unchanged '+field):
        dividend_value_change_bound(baseline, replace(baseline, **{field:value}), Option(100., 1.), rights)


@pytest.mark.parametrize('events', [(), (DividendCap(.5, 3.),), (DividendCap(.4, 3.), DividendCap(.7, 1.))])
def test_rejects_added_removed_or_moved_dates(events):
    baseline, rights = fixture()
    with pytest.raises(ValueError, match='matching event dates'):
        dividend_value_change_bound(baseline, replace(baseline, dividends=events), Option(100., 1.), rights)


@pytest.mark.parametrize('rights', [(), (FundedExercise(.9, 1.),), (FundedExercise(.3, 1., 'before'), FundedExercise(1., 1.))])
def test_validates_actual_exercise_calendar(rights):
    baseline, _ = fixture()
    with pytest.raises(ValueError):
        dividend_value_change_bound(baseline, baseline, Option(100., 1.), rights)


def test_outward_rounding_uses_exact_binary_inputs_for_cash_and_premium():
    baseline, rights = fixture(caps=(.1, .2), times=(.3, .7), premium=.3)
    alternative = replace(baseline, dividends=(DividendCap(.3, .4), DividendCap(.7, .9)))
    bound = dividend_value_change_bound(baseline, alternative, Option(100., 1.), rights)
    exact = sum((Fraction.from_float(b.discounted_cash)-Fraction.from_float(a.discounted_cash)
                 for a, b in zip(baseline.dividends, alternative.dividends)), Fraction(0))
    for value, rational in ((bound.present_value_upper_change, exact),
                            (bound.premium_quote_upper_change, exact/Fraction.from_float(.3))):
        assert Fraction.from_float(value) >= rational
        assert Fraction.from_float(math.nextafter(value, -math.inf)) < rational


def test_rejects_unrepresentable_finite_quote_bound():
    baseline, rights = fixture(premium=1e-308)
    alternative = replace(baseline, dividends=(DividendCap(.4, 10.),))
    with pytest.raises(ValueError, match='finite float range'):
        dividend_value_change_bound(baseline, alternative, Option(100., 1.), rights)
