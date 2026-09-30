"""Map a dated research convention to funded numerical inputs without quotes IO.

Amounts, volatility and a premium payment date must be supplied explicitly.
Unknown in-horizon dividends and unapproved settlement projections are rejected.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math

from .event_calendar import EventCalendar, NEW_YORK, act365f
from .valuation_inputs import DiscountCurve
from ..optimal_stopping.contracts import Option
from ..optimal_stopping.funded_cash import DividendCap, FundedCashModel, FundedExercise


@dataclass(frozen=True)
class DatedDividend:
    ex_date: date
    pay_date: date
    amount: float | None
    amount_basis: str
    occurs: bool | None = True

    def __post_init__(self):
        if type(self.ex_date) is not date or type(self.pay_date) is not date or self.pay_date < self.ex_date:
            raise ValueError("Dividend payment cannot precede its ex-date")
        if not isinstance(self.amount_basis, str) or not self.amount_basis.strip():
            raise ValueError("An explicit dividend amount basis is required")
        if self.occurs is not None and type(self.occurs) is not bool:
            raise ValueError("Occurrence must be true, false or unresolved")
        if self.amount is not None and (isinstance(self.amount, bool) or not math.isfinite(self.amount) or self.amount < 0):
            raise ValueError("Dividend amount must be finite and non-negative, or unresolved")


@dataclass(frozen=True)
class DatedFundedProblem:
    model: FundedCashModel
    option: Option
    decisions: tuple[FundedExercise, ...]
    share_payment_date: date
    premium_payment_date: date
    deliveries: tuple[tuple[datetime, date, str], ...]
    dividend_inputs: tuple[DatedDividend, ...]
    provisional_delivery: bool
    cash_discount_clock: str = "Frozen date-curve ratios from observation's New York date; no intraday cash accrual"
    premium_date_verified: bool = False
    empirical_calibration_accepted: bool = False


def build_dated_problem(calendar: EventCalendar, curve: DiscountCurve,
                        observation: datetime, expiry: date, quoted_spot: float,
                        strike: float, volatility: float, kind: str,
                        dividends: tuple[DatedDividend, ...], *,
                        premium_payment_date: date, allow_provisional=False) -> DatedFundedProblem:
    """T+1 matched share/exercise settlement, with a separately supplied premium date.

    Exercise is at the underlying core close. Dividends jump at core opening by
    convention. Funding uses date-only curve ratios; diffusion uses UTC seconds.
    This adapter certifies neither broker timing nor the input amount basis.
    """
    timestamps = calendar.research_decisions(observation, expiry)
    valuation = observation.astimezone(NEW_YORK).date()
    if type(premium_payment_date) is not date or premium_payment_date < valuation:
        raise ValueError("Premium payment date must be explicit and no earlier than valuation")
    if not isinstance(dividends, tuple) or not all(isinstance(d, DatedDividend) for d in dividends):
        raise ValueError("Use an immutable dated dividend schedule")
    if len({d.ex_date for d in dividends}) != len(dividends):
        raise ValueError("Aggregate multiple dividends on the same date explicitly")
    initial = calendar.settlement_after(valuation, allow_provisional=allow_provisional)
    deliveries, rights = [], []
    for timestamp in timestamps:
        settled = calendar.settlement_after(timestamp.date(), allow_provisional=allow_provisional)
        rights.append(FundedExercise(act365f(observation, timestamp), curve.between(valuation, settled.day)))
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
            raise ValueError("Unresolved occurrence of an in-horizon dividend")
        if not dividend.occurs:
            continue
        if dividend.amount is None:
            raise ValueError("Unresolved amount for an in-horizon dividend")
        cap = dividend.amount * curve.between(valuation, dividend.pay_date)
        caps.append(DividendCap(act365f(observation, ex), cap))
    model = FundedCashModel(quoted_spot, volatility, curve.between(valuation, initial.day),
                            curve.between(valuation, premium_payment_date), tuple(caps))
    option = Option(strike, act365f(observation, timestamps[-1]), kind)
    if option.maturity <= 0:
        raise ValueError("Observation must precede the terminal research decision")
    return DatedFundedProblem(model, option, tuple(rights), initial.day, premium_payment_date,
                              tuple(deliveries), tuple(relevant),
                              initial.evidence.startswith("provisional") or
                              any(status.startswith("provisional") for _, _, status in deliveries))
