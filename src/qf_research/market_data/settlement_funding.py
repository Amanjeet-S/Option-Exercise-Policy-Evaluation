"""Premium-unit valuation from settlement-to-spot and spot-relative factors.

No observation-date discount curve is invented. The adapter covers coincident
share and premium settlement only. Its value unit is explicit in the result.
"""
from dataclasses import dataclass
from datetime import date, datetime
import math

from .event_calendar import EventCalendar, NEW_YORK, act365f
from .funded_inputs import DatedDividend, DatedFundedProblem
from .valuation_inputs import DiscountCurve
from ..optimal_stopping.contracts import Option
from ..optimal_stopping.funded_cash import DividendCap, FundedCashModel, FundedExercise


@dataclass(frozen=True)
class SettlementFunding:
    """P(t,u)/P(t,h), with h settlement and s the spot-curve anchor.

    The explicit bridge is P(t,s)/P(t,h). Cash dates strictly between h and s
    remain unidentified; they are not interpolated from an assumed short rate.
    """
    settlement_date: date
    spot_curve: DiscountCurve
    settlement_to_spot: float

    def __post_init__(self):
        if type(self.settlement_date) is not date or not isinstance(self.spot_curve, DiscountCurve):
            raise ValueError('Require a settlement date and a spot-relative curve')
        if self.settlement_date > self.spot_curve.anchor:
            raise ValueError('Settlement must be no later than swap spot')
        value = self.settlement_to_spot
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Settlement-to-spot factor must be finite and positive')
        if self.settlement_date == self.spot_curve.anchor and value != 1.:
            raise ValueError('Coincident settlement and swap spot require factor one')

    def factor(self, payment_date: date) -> float:
        if type(payment_date) is not date:
            raise ValueError('Cash flows require dates without time components')
        if payment_date == self.settlement_date:
            return 1.
        if payment_date < self.spot_curve.anchor:
            raise ValueError('Cash date outside the identified settlement and spot domain')
        result = self.settlement_to_spot * self.spot_curve.discount(payment_date)
        if not math.isfinite(result) or result <= 0:
            raise ValueError('Normalised factor overflow or underflow')
        return result


def build_premium_normalised_problem(calendar: EventCalendar, funding: SettlementFunding,
        observation: datetime, expiry: date, quoted_spot: float, strike: float,
        volatility: float, kind: str, dividends: tuple[DatedDividend, ...], *,
        premium_payment_date: date, allow_provisional=False) -> DatedFundedProblem:
    """Construct the existing stopping model directly in premium-settlement units.

    All factors are divided by P(t,h), so the returned numerical present_value
    equals premium_quote. It is not an observation-date present value. Broker
    premium timing and the empirical provenance of supplied factors are not
    certified by this mathematical adapter.
    """
    timestamps = calendar.research_decisions(observation, expiry)
    valuation = observation.astimezone(NEW_YORK).date()
    initial = calendar.settlement_after(valuation, allow_provisional=allow_provisional)
    if type(premium_payment_date) is not date or not (
            premium_payment_date == initial.day == funding.settlement_date):
        raise ValueError('Require coincident share, premium and funding settlement dates')
    if not isinstance(dividends, tuple) or not all(isinstance(d, DatedDividend) for d in dividends):
        raise ValueError('Use an immutable dated dividend schedule')
    if len({d.ex_date for d in dividends}) != len(dividends):
        raise ValueError('Aggregate multiple dividends on the same date explicitly')
    deliveries, rights = [], []
    for timestamp in timestamps:
        settled = calendar.settlement_after(timestamp.date(), allow_provisional=allow_provisional)
        rights.append(FundedExercise(act365f(observation, timestamp), funding.factor(settled.day)))
        deliveries.append((timestamp, settled.day, settled.evidence))
    relevant, caps = [], []
    for dividend in sorted(dividends, key=lambda d: d.ex_date):
        if dividend.ex_date < valuation or dividend.ex_date > expiry:
            continue
        ex = calendar.model_ex_event(dividend.ex_date)
        if ex <= observation.astimezone(NEW_YORK):
            continue
        relevant.append(dividend)
        if dividend.occurs is None:
            raise ValueError('Unresolved occurrence of an in-horizon dividend')
        if not dividend.occurs:
            continue
        if dividend.amount is None:
            raise ValueError('Unresolved amount for an in-horizon dividend')
        caps.append(DividendCap(act365f(observation, ex), dividend.amount * funding.factor(dividend.pay_date)))
    model = FundedCashModel(quoted_spot, volatility, 1., 1., tuple(caps))
    option = Option(strike, act365f(observation, timestamps[-1]), kind)
    if option.maturity <= 0:
        raise ValueError('Observation must precede the terminal research decision')
    return DatedFundedProblem(model, option, tuple(rights), initial.day, premium_payment_date,
        tuple(deliveries), tuple(relevant), initial.evidence.startswith('provisional') or
        any(status.startswith('provisional') for _, _, status in deliveries),
        cash_discount_clock='Factors relative to premium settlement, not observation-date discount factors; '
                            'present_value is expressed in premium units; no intraday cash accrual')
