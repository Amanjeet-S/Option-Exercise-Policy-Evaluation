"""Date-based discounting and explicitly labelled income forecast benchmarks."""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import math
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DiscountCurve:
    """Positive node factors with log-linear interpolation and no extrapolation.

    Times are calendar-day offsets from anchor. The curve is deterministic when
    used for roll-down; ratios do not claim to forecast tomorrow's market curve.
    Negative rates are allowed, so discount factors need not be decreasing.
    """

    anchor: date
    dates: tuple[date, ...]
    factors: tuple[float, ...]

    def __post_init__(self):
        if len(self.dates) < 2 or len(self.dates) != len(self.factors):
            raise ValueError("Need at least two equally sized date and factor arrays")
        if self.dates[0] != self.anchor or any(b <= a for a, b in zip(self.dates, self.dates[1:])):
            raise ValueError("Curve dates must start at anchor and be strictly increasing")
        if any(isinstance(p, bool) or not math.isfinite(p) or p <= 0 for p in self.factors):
            raise ValueError("Discount factors must be finite and positive")
        if self.factors[0] != 1.0:
            raise ValueError("The anchor discount factor must equal one")

    def discount(self, target: date) -> float:
        if not self.anchor <= target <= self.dates[-1]:
            raise ValueError("Curve extrapolation is prohibited")
        i = bisect_left(self.dates, target)
        if self.dates[i] == target:
            return self.factors[i]
        left, right = self.dates[i - 1], self.dates[i]
        weight = (target - left).days / (right - left).days
        return math.exp((1 - weight) * math.log(self.factors[i - 1]) + weight * math.log(self.factors[i]))

    def between(self, start: date, end: date) -> float:
        if end < start:
            raise ValueError("Cash-flow date precedes valuation")
        return self.discount(end) / self.discount(start)

    def continuous_zero(self, start: date, end: date) -> float:
        if end <= start:
            raise ValueError("A zero rate needs a positive horizon")
        return -math.log(self.between(start, end)) * 365 / (end - start).days


def inspect_sofr_curve(payload, expected_date, expected_id):
    """Inspect an IPA response without equating masked quotes with observed data."""
    if len(payload.get("data", [])) != 1:
        raise ValueError("Expected one curve response")
    item = payload["data"][0]
    if item.get("error") or item.get("errorCode"):
        raise ValueError("Provider reported a curve error")
    definition, parameters = item["curveDefinition"], item["curveParameters"]
    required = {"id": expected_id, "currency": "USD", "indexName": "SOFR",
                "mainConstituentAssetClass": "Swap", "discountingTenor": "OIS"}
    if any(definition.get(k) != v for k, v in required.items()):
        raise ValueError("Unexpected curve definition")
    if parameters.get("valuationDate") != expected_date or parameters.get("priceSide") != "Mid":
        raise ValueError("Unexpected curve date or price side")
    if parameters.get("marketDataLookBack") != {"value": 0, "unit": "CalendarDay"}:
        raise ValueError("Non-zero or unreported provider lookback")
    if parameters.get("useDelayedDataIfDenied") is not False or parameters.get("ignoreInvalidInstrument") is not False:
        raise ValueError("Unverified delayed-data or invalid-instrument policy")
    block = item["curves"]["OIS"]
    if block.get("isDiscountCurve") is not True:
        raise ValueError("OIS block is not identified as a discount curve")
    nodes = block["curvePoints"]
    if any(p["startDate"] != expected_date for p in nodes):
        raise ValueError("Inconsistent discount-factor anchor")
    curve = DiscountCurve(date.fromisoformat(expected_date),
                          tuple(date.fromisoformat(p["endDate"]) for p in nodes),
                          tuple(p["discountFactor"] for p in nodes))
    instruments = item.get("constituents", {}).get("interestRateInstruments", {}).get("USD", {})
    diagnostics = []
    for family, rows in instruments.items():
        for row in rows:
            definition = row["instrumentDefinition"]
            fields = row.get("fields", {})
            bid, ask = fields.get("bid", {}).get("value"), fields.get("ask", {}).get("value")
            quote_date = fields.get("marketDataDate")
            if quote_date is not None and quote_date != expected_date:
                raise ValueError("A visible constituent is not from the requested curve date")
            quoted = (bid is not None and ask is not None and quote_date == expected_date
                      and all(not isinstance(x, bool) and math.isfinite(x) for x in (bid, ask)) and bid <= ask)
            diagnostics.append({"family": family, "instrument_code": definition["instrumentCode"],
                                "market_data_date": quote_date, "quote_visible": quoted,
                                "masked_identifier": "*" in definition["instrumentCode"]})
    return curve, {"nodes": len(nodes), "provider_parameters": parameters,
                   "constituents": diagnostics,
                   "visible_constituents": sum(d["quote_visible"] for d in diagnostics),
                   "masked_constituents": sum(d["masked_identifier"] for d in diagnostics),
                   "independent_bootstrap_ready": bool(diagnostics) and all(d["quote_visible"] and not d["masked_identifier"] for d in diagnostics),
                   "historical_publication_vintage_verified": False}


def released_report_forecast(report, available_at, cutoff):
    """An income run-rate projection, not a declaration or risk-neutral estimate.

    Explicit source checks are performed by the offline runner. This function
    enforces timing and arithmetic; a historical period end is not publication.
    """
    release, valuation = datetime.fromisoformat(available_at), datetime.fromisoformat(cutoff)
    if any(d.tzinfo is None or d.utcoffset() is None for d in (release, valuation)):
        raise ValueError("Explicitly zoned publication and valuation times are required")
    if release > valuation:
        raise ValueError("Income report was not available at the cutoff")
    end, start = date.fromisoformat(report["period_end"]), date.fromisoformat(report["period_start"])
    months = report["period_months"]
    if isinstance(months, bool) or months not in (6, 12) or start >= end or end >= release.date():
        raise ValueError("Unsupported or inconsistent income reporting period")
    # The specified intervals must cover complete calendar months.
    after_end = end + timedelta(days=1)
    if start.day != 1 or after_end.day != 1 or (after_end.year - start.year) * 12 + after_end.month - start.month != months:
        raise ValueError("Income reporting months disagree with its dates")
    income = Decimal(report["net_investment_income_per_unit"])
    if not income.is_finite() or income < 0:
        raise ValueError("Income must be finite and non-negative for this benchmark")
    return {"model": report["model"], "forecast_cash_per_regular_distribution": str(income * 3 / months),
            "currency": "USD", "available_at": available_at, "source_period_end": end.isoformat(),
            "status": "income_run_rate_forecast", "declared_dividend": False,
            "risk_neutral_expectation_established": False}


def discounted_distribution(curve, valuation, ex_date, pay_date, amount):
    """Present value and ex-date claim value of an assumed deterministic payment.

    Under this model D * P(t,p) = P(t,e) * [D * P(e,p)]. The dividend is not
    discarded when p is later than option expiry; entitlement changes at e.
    """
    if not valuation < ex_date <= pay_date:
        raise ValueError("Require future ex-date and payment no earlier than ex-date")
    if isinstance(amount, bool) or not math.isfinite(amount) or amount < 0:
        raise ValueError("Cash amount must be finite and non-negative")
    return {"amount": amount, "ex_date": ex_date.isoformat(), "pay_date": pay_date.isoformat(),
            "payment_discount_factor": curve.between(valuation, pay_date),
            "ex_date_claim_value": amount * curve.between(ex_date, pay_date),
            "present_value": amount * curve.between(valuation, pay_date)}
