"""Cash-input bounds for the existing funded, capped-dividend stopping model.

These bound exact model value changes with the state, volatility, funding and
exercise rights fixed. They do not bound a numerical solver's error or provide
forecast coverage. See dividend_sensitivity.md.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

from .contracts import Option
from .funded_cash import FundedCashModel, FundedExercise, _schedule


@dataclass(frozen=True)
class DividendValueChangeBound:
    """Inclusive bounds on alternative minus baseline, per underlying unit."""

    present_value_lower_change: float
    present_value_upper_change: float
    premium_quote_lower_change: float
    premium_quote_upper_change: float
    discounted_cash_increase_bound: float
    discounted_cash_decrease_bound: float
    changed_events_in_horizon: int
    changed_events_after_expiry: int


def _upward_float(value: Fraction) -> float:
    """Smallest binary float at least a non-negative exact rational value."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError("Sensitivity bound exceeds finite float range") from exc
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise ValueError("Sensitivity bound exceeds finite float range")
    return rounded


def dividend_value_change_bound(
    baseline: FundedCashModel,
    alternative: FundedCashModel,
    option: Option,
    decisions: tuple[FundedExercise, ...],
) -> DividendValueChangeBound:
    """Bound exact value changes when only deterministic dividend caps change.

    Inputs use the existing model's discounted cash units. Event dates must
    match, including zero-cash events; event addition/deletion and changes to
    state, volatility or funding are outside this interface's scope. Both
    models share the supplied finite exercise calendar and settlement factors.

    If A sums positive cap changes and B sums negative changes in absolute
    value, call PV changes lie in [-A, B] and put changes lie in [-B, A].
    Premium-quote bounds divide by the unchanged premium factor. Fraction
    arithmetic and outward conversion avoid understating these budgets for
    the supplied binary float inputs; input measurement error remains separate.
    """
    if not isinstance(baseline, FundedCashModel) or not isinstance(alternative, FundedCashModel):
        raise TypeError("Two FundedCashModel instances are required")
    if not isinstance(option, Option):
        raise TypeError("An Option instance is required")
    for field in ("quoted_spot", "volatility", "initial_share_factor", "premium_factor"):
        if getattr(baseline, field) != getattr(alternative, field):
            raise ValueError(f"Dividend sensitivity requires unchanged {field}")
    if tuple(e.ex_time for e in baseline.dividends) != tuple(e.ex_time for e in alternative.dividends):
        raise ValueError("Dividend sensitivity requires matching event dates")
    _schedule(baseline, option, decisions)
    _schedule(alternative, option, decisions)

    increase = decrease = Fraction(0)
    changed = ignored = 0
    for first, second in zip(baseline.dividends, alternative.dividends):
        delta = Fraction.from_float(second.discounted_cash) - Fraction.from_float(first.discounted_cash)
        if delta == 0:
            continue
        if first.ex_time > option.maturity:
            ignored += 1
            continue
        changed += 1
        increase += max(delta, 0)
        decrease += max(-delta, 0)

    down, up = (increase, decrease) if option.kind == "call" else (decrease, increase)
    factor = Fraction.from_float(baseline.premium_factor)
    return DividendValueChangeBound(
        present_value_lower_change=-_upward_float(down) if down else 0.,
        present_value_upper_change=_upward_float(up),
        premium_quote_lower_change=-_upward_float(down/factor) if down else 0.,
        premium_quote_upper_change=_upward_float(up/factor),
        discounted_cash_increase_bound=_upward_float(increase),
        discounted_cash_decrease_bound=_upward_float(decrease),
        changed_events_in_horizon=changed,
        changed_events_after_expiry=ignored,
    )
