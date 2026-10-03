"""Deterministic mathematical fixtures; no synthetic market observations."""

from copy import deepcopy
from datetime import date, timedelta
import math

import pytest

from qf_research.market_data.valuation_inputs import (
    DiscountCurve, discounted_distribution, inspect_sofr_curve, released_report_forecast,
)


ANCHOR = date(2025, 1, 1)


def curve(rate=0.04):
    return DiscountCurve(ANCHOR, tuple(ANCHOR + timedelta(days=d) for d in (0, 30, 90, 365)),
                         tuple(math.exp(-rate * d / 365) for d in (0, 30, 90, 365)))


def test_log_linear_curve_recovers_constant_rate_at_non_nodes():
    c = curve()
    for day in (1, 12, 37, 89, 364):
        end = ANCHOR + timedelta(days=day)
        assert c.discount(end) == pytest.approx(math.exp(-0.04 * day / 365), rel=1e-14)
        assert c.continuous_zero(ANCHOR, end) == pytest.approx(0.04, abs=2e-14)
    for day, factor in zip(c.dates, c.factors):
        assert c.discount(day) == factor


def test_roll_down_uses_discount_ratios_and_is_multiplicatively_consistent():
    c = DiscountCurve(ANCHOR, (ANCHOR, date(2025, 2, 1), date(2025, 5, 1)), (1.0, 0.997, 0.98))
    a, b, end = date(2025, 1, 15), date(2025, 2, 15), date(2025, 4, 15)
    assert c.between(a, a) == 1
    assert c.between(a, end) == pytest.approx(c.between(a, b) * c.between(b, end), rel=1e-14)
    assert c.between(a, end) != c.discount(end)


def test_negative_rates_are_not_misclassified_as_bad_data():
    c = curve(-0.01)
    assert c.discount(ANCHOR + timedelta(days=45)) > 1
    assert c.continuous_zero(ANCHOR, ANCHOR + timedelta(days=45)) == pytest.approx(-0.01)


@pytest.mark.parametrize("factor", [0, -1, math.inf, math.nan, True])
def test_invalid_discount_factors_fail(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        DiscountCurve(ANCHOR, (ANCHOR, ANCHOR + timedelta(days=1)), (1.0, factor))


def test_bad_curve_dates_and_anchor_are_rejected():
    for dates, factors in [((ANCHOR, ANCHOR), (1.0, 0.99)),
                           ((ANCHOR + timedelta(days=1), ANCHOR + timedelta(days=2)), (1.0, 0.99)),
                           ((ANCHOR, ANCHOR + timedelta(days=1)), (0.999, 0.99))]:
        with pytest.raises(ValueError):
            DiscountCurve(ANCHOR, dates, factors)


def test_no_extrapolation_or_negative_horizons():
    c = curve()
    for day in (-1, 366):
        with pytest.raises(ValueError, match="extrapolation"):
            c.discount(ANCHOR + timedelta(days=day))
    with pytest.raises(ValueError, match="precedes"):
        c.between(ANCHOR + timedelta(days=1), ANCHOR)
    with pytest.raises(ValueError, match="positive horizon"):
        c.continuous_zero(ANCHOR, ANCHOR)


def test_dividend_claim_obeys_tower_discounting_and_payment_lag():
    c = curve()
    ex, pay = date(2025, 3, 21), date(2025, 4, 30)
    result = discounted_distribution(c, ANCHOR, ex, pay, 1.5)
    assert result["present_value"] == pytest.approx(c.between(ANCHOR, ex) * result["ex_date_claim_value"])
    assert result["ex_date_claim_value"] < 1.5
    assert result["present_value"] < 1.5 * c.between(ANCHOR, ex)
    assert discounted_distribution(c, ANCHOR, ex, ex, 1.5)["ex_date_claim_value"] == 1.5


def test_dividend_cashflow_rejects_invalid_timing_and_amount():
    c = curve()
    ex, pay = date(2025, 3, 21), date(2025, 4, 30)
    for inputs in [(ANCHOR, ANCHOR, pay, 1), (ANCHOR, pay, ex, 1), (ANCHOR, ex, pay, -1)]:
        with pytest.raises(ValueError):
            discounted_distribution(c, *inputs)


def report(**changes):
    result = {"model": "fixture_income_run_rate", "period_start": "2024-07-01", "period_end": "2024-12-31",
              "period_months": 6, "net_investment_income_per_unit": "4.00"}
    result.update(changes)
    return result


def test_forecast_requires_publication_not_just_period_end():
    published = "2025-02-01T00:00:00-05:00"
    with pytest.raises(ValueError, match="not available"):
        released_report_forecast(report(), published, "2025-01-31T15:45:00-05:00")
    result = released_report_forecast(report(), published, published)
    assert result["forecast_cash_per_regular_distribution"] == "2.00"
    assert not result["declared_dividend"] and not result["risk_neutral_expectation_established"]


def test_annual_and_semiannual_forecasts_use_reporting_period_units():
    published = "2025-02-01T00:00:00-05:00"
    annual = report(period_start="2024-01-01", period_months=12, net_investment_income_per_unit="8.00")
    assert released_report_forecast(annual, published, published)["forecast_cash_per_regular_distribution"] == "2.00"


@pytest.mark.parametrize("changes", [{"period_months": True}, {"period_months": 12}, {"period_start": "2024-07-02"},
                                     {"period_end": "2025-02-01"}, {"net_investment_income_per_unit": "NaN"}])
def test_inconsistent_income_periods_and_amounts_fail(changes):
    with pytest.raises(ValueError):
        released_report_forecast(report(**changes), "2025-02-01T00:00:00-05:00", "2025-02-02T00:00:00-05:00")


def test_date_only_availability_is_not_assumed_known_at_midnight():
    with pytest.raises(ValueError, match="zoned"):
        released_report_forecast(report(), "2025-02-01", "2025-02-02T00:00:00-05:00")


def curve_response():
    return {"data": [{"curveDefinition": {"id": "fixture_curve", "currency": "USD", "indexName": "SOFR",
                      "mainConstituentAssetClass": "Swap", "discountingTenor": "OIS"},
                      "curveParameters": {"valuationDate": "2025-01-01", "priceSide": "Mid",
                          "marketDataLookBack": {"value": 0, "unit": "CalendarDay"},
                          "useDelayedDataIfDenied": False, "ignoreInvalidInstrument": False},
                      "curves": {"OIS": {"isDiscountCurve": True, "curvePoints": [
                          {"startDate": "2025-01-01", "endDate": "2025-01-01", "discountFactor": 1.0},
                          {"startDate": "2025-01-01", "endDate": "2025-02-01", "discountFactor": 0.995}]}},
                      "constituents": {"interestRateInstruments": {"USD": {"overnightIndexSwaps": [
                          {"instrumentDefinition": {"instrumentCode": "fixture_masked****"}}]}}}}]}


def test_masked_constituents_do_not_become_an_independent_bootstrap():
    payload = curve_response()
    before = deepcopy(payload)
    c, audit = inspect_sofr_curve(payload, "2025-01-01", "fixture_curve")
    assert c.discount(date(2025, 2, 1)) == 0.995
    assert audit["masked_constituents"] == 1 and not audit["independent_bootstrap_ready"]
    assert not audit["historical_publication_vintage_verified"]
    assert payload == before


def test_visible_stale_constituents_and_provider_date_mismatch_fail():
    p = curve_response()
    item = p["data"][0]
    item["constituents"]["interestRateInstruments"]["USD"]["overnightIndexSwaps"][0]["fields"] = {"marketDataDate": "2024-12-31"}
    with pytest.raises(ValueError, match="requested curve date"):
        inspect_sofr_curve(p, "2025-01-01", "fixture_curve")
    with pytest.raises(ValueError, match="curve date"):
        inspect_sofr_curve(curve_response(), "2025-01-02", "fixture_curve")


def test_nonzero_lookback_and_delayed_fallback_are_not_accepted():
    for changes in ({"marketDataLookBack": {"value": 10, "unit": "CalendarDay"}}, {"useDelayedDataIfDenied": True}, {"ignoreInvalidInstrument": True}):
        p = curve_response()
        p["data"][0]["curveParameters"].update(changes)
        with pytest.raises(ValueError):
            inspect_sofr_curve(p, "2025-01-01", "fixture_curve")
