"""Dated exercise cash flows, separate from the immediate-settlement solvers.

The matched-offset benchmark assumes simultaneous exercise and an offsetting
share trade with identical delivery dates and entitlement. It excludes spread,
borrow costs, funding asymmetry, fails, and broker processing uncertainty.
"""

from dataclasses import dataclass
from datetime import date
import math

from qf_research.market_data.valuation_inputs import DiscountCurve


def _nonnegative(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("Value must be finite and non-negative")


def _dates(earlier, later):
    if type(earlier) is not date or type(later) is not date or later < earlier:
        raise ValueError("Ordered calendar dates are required")


@dataclass(frozen=True)
class SettlementLeg:
    label: str
    day: date
    cash: float
    shares: int


@dataclass(frozen=True)
class ExerciseLedger:
    decision_date: date
    legs: tuple[SettlementLeg, SettlementLeg]

    def __post_init__(self):
        if len(self.legs) != 2 or self.legs[0].day != self.legs[1].day:
            raise ValueError("The offset and exercise must settle on the same date")
        if sum(leg.shares for leg in self.legs) != 0:
            raise ValueError("The matched exercise ledger must have zero net shares")
        for leg in self.legs:
            _dates(self.decision_date, leg.day)
            if not math.isfinite(leg.cash):
                raise ValueError("Non-finite cash flow")

    def present_value(self, curve: DiscountCurve, valuation_date: date) -> float:
        _dates(valuation_date, self.decision_date)
        return math.fsum(curve.between(valuation_date, leg.day) * leg.cash for leg in self.legs)

    def exercise_reward(self, curve: DiscountCurve, valuation_date: date) -> float:
        """Positive part of an observed ledger, not a stochastic option price."""
        return max(self.present_value(curve, valuation_date), 0.)


def matched_exercise_ledger(kind: str, quote: float, strike: float,
                            decision_date: date, exercise_settlement: date,
                            share_settlement: date, *, units: int = 1) -> ExerciseLedger:
    """Physical exercise plus an opposite underlying trade at the same quote time.

    `units` is a number of shares, default one. A contract multiplier must be
    independently verified before supplying it here.
    """
    _nonnegative(quote)
    _nonnegative(strike)
    if strike == 0 or kind not in ("call", "put"):
        raise ValueError("A positive strike and call/put kind are required")
    if isinstance(units, bool) or not isinstance(units, int) or units < 1:
        raise ValueError("Units must be a positive integer")
    sign = 1 if kind == "call" else -1
    return ExerciseLedger(decision_date, (
        SettlementLeg("exercise", exercise_settlement, -sign * strike * units, sign * units),
        SettlementLeg("offsetting_share_trade", share_settlement, sign * quote * units, -sign * units),
    ))


def quote_to_present_value(quote: float, curve: DiscountCurve,
                           valuation_date: date, payment_date: date) -> float:
    """Convert a quoted future cash amount to date-resolution present value."""
    _nonnegative(quote)
    _dates(valuation_date, payment_date)
    return quote * curve.between(valuation_date, payment_date)


def present_value_to_quote(value: float, curve: DiscountCurve,
                           valuation_date: date, payment_date: date) -> float:
    _nonnegative(value)
    _dates(valuation_date, payment_date)
    return value / curve.between(valuation_date, payment_date)
