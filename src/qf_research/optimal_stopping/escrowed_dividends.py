"""Common-schedule escrowed-dividend benchmark, not a cash-jump GBM model.

The volatility belongs to the risky component X, with S = X + dividend reserve.
Its discounted state Y = D(0,t) X has drift zero. Solvers operate on discounted
rewards, retain payment lags, and distinguish pre- and post-entitlement exercise.
See discrete_dividend_model.md for assumptions, derivations and attribution.
"""

from dataclasses import dataclass
from datetime import date
import math

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr

from .contracts import Option, positive_integer


@dataclass(frozen=True)
class DiscountFactors:
    """Discount factors on an ACT/365F year axis; log-linear, no extrapolation."""

    times: tuple[float, ...]
    factors: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.times, tuple) or not isinstance(self.factors, tuple):
            raise ValueError("Use immutable tuples for the curve")
        if len(self.times) < 2 or len(self.times) != len(self.factors):
            raise ValueError("Need at least two paired curve nodes")
        if not all(math.isfinite(t) for t in self.times) or self.times[0] != 0 or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("Times must start at zero and increase strictly")
        if self.factors[0] != 1 or any(isinstance(p, bool) or not math.isfinite(p) or p <= 0 for p in self.factors):
            raise ValueError("Discount factors must be positive, finite and anchored at one")

    def discount(self, time: float) -> float:
        if not math.isfinite(time) or not 0 <= time <= self.times[-1]:
            raise ValueError("Curve extrapolation is prohibited")
        return float(np.exp(np.interp(time, self.times, np.log(self.factors))))

    @classmethod
    def from_dated_curve(cls, curve, valuation: date):
        """Freeze and rebase a retained date curve; no claim to intraday rates."""
        curve.discount(valuation)  # Check coverage before creating the time axis.
        dates = [d for d in curve.dates if d > valuation]
        return cls((0.0,) + tuple((d - valuation).days / 365 for d in dates),
                   (1.0,) + tuple(curve.between(valuation, d) for d in dates))


@dataclass(frozen=True)
class CashDividend:
    ex_time: float
    pay_time: float
    amount: float

    def __post_init__(self):
        if any(isinstance(x, bool) or not math.isfinite(x) for x in (self.ex_time, self.pay_time, self.amount)):
            raise ValueError("Dividend inputs must be finite numbers")
        if not 0 < self.ex_time <= self.pay_time or self.amount < 0:
            raise ValueError("Require positive ex-time, payment no earlier, and non-negative cash")


@dataclass(frozen=True)
class Decision:
    time: float
    phase: str = "after"

    def __post_init__(self):
        if isinstance(self.time, bool) or not math.isfinite(self.time) or self.time < 0:
            raise ValueError("Decision time must be finite and non-negative")
        if self.phase not in {"before", "after"}:
            raise ValueError("Decision phase must be before or after entitlement")


@dataclass(frozen=True)
class EscrowedModel:
    spot: float
    risky_volatility: float
    curve: DiscountFactors
    dividends: tuple[CashDividend, ...] = ()

    def __post_init__(self):
        if any(isinstance(x, bool) or not math.isfinite(x) for x in (self.spot, self.risky_volatility)) or self.spot <= 0 or self.risky_volatility < 0:
            raise ValueError("Require positive spot and non-negative finite risky volatility")
        if not isinstance(self.dividends, tuple):
            raise ValueError("Use an immutable dividend schedule")
        if any(b.ex_time <= a.ex_time for a, b in zip(self.dividends, self.dividends[1:])):
            raise ValueError("Use a strictly ordered schedule with one cash payment per ex-time")
        for dividend in self.dividends:
            self.curve.discount(dividend.pay_time)
        if self.risky_spot <= 0:
            raise ValueError("Dividend reserve exhausts spot; this model is inadmissible")

    @property
    def risky_spot(self):
        return self.spot - self.reserve(Decision(0.0))

    def reserve(self, decision: Decision) -> float:
        """Time-zero value of entitlements still attached to the share."""
        self.curve.discount(decision.time)
        if decision.phase == "before" and not any(d.ex_time == decision.time for d in self.dividends):
            raise ValueError("Before-entitlement phase must refer to an exact dividend event")
        return sum(d.amount * self.curve.discount(d.pay_time) for d in self.dividends
                   if d.ex_time > decision.time or (decision.phase == "before" and d.ex_time == decision.time))

    def stock(self, discounted_risky_state, decision: Decision):
        return (np.asarray(discounted_risky_state) + self.reserve(decision)) / self.curve.discount(decision.time)

    def reward(self, option: Option, decision: Decision, discounted_risky_state):
        """Discounted intrinsic value; immediate frictionless exercise settlement."""
        difference = np.asarray(discounted_risky_state) + self.reserve(decision) - option.strike * self.curve.discount(decision.time)
        return np.maximum(difference if option.kind == "call" else -difference, 0.0)


def decision_schedule(model, option, decisions=None):
    """Always include post-entitlement maturity. Other decisions are explicit.

    A pre-entitlement phase represents a mathematical left-limit decision. It is
    not an assertion about an exchange's last cum-dividend exercise deadline.
    """
    if option.maturity <= 0:
        raise ValueError("Require positive maturity")
    model.curve.discount(option.maturity)
    schedule = list(() if decisions is None else decisions)
    if len(set(schedule)) != len(schedule):
        raise ValueError("Duplicate exercise decisions")
    terminal = Decision(option.maturity, "after")
    if terminal not in schedule:
        schedule.append(terminal)
    for decision in schedule:
        if decision.time > option.maturity:
            raise ValueError("Exercise decision after maturity")
        model.reserve(decision)
    return tuple(sorted(schedule, key=lambda d: (d.time, d.phase == "after")))


def european_price(model: EscrowedModel, option: Option) -> float:
    """Exact European expectation under this model, including post-maturity cash."""
    terminal = decision_schedule(model, option)[0]
    effective_strike = option.strike * model.curve.discount(option.maturity) - model.reserve(terminal)
    y0 = model.risky_spot
    if effective_strike <= 0:
        return float(y0 - effective_strike if option.kind == "call" else 0.0)
    if model.risky_volatility == 0:
        return float(model.reward(option, terminal, y0))
    sd = model.risky_volatility * math.sqrt(option.maturity)
    d1 = math.log(y0 / effective_strike) / sd + sd / 2
    d2 = d1 - sd
    sign = 1 if option.kind == "call" else -1
    return float(sign * (y0 * ndtr(sign * d1) - effective_strike * ndtr(sign * d2)))


def _indexed_schedule(model, option, decisions, steps):
    steps = positive_integer(steps, "steps")
    schedule = decision_schedule(model, option, decisions)
    result = {}
    for decision in schedule:
        location = decision.time / option.maturity * steps
        index = round(location)
        if not math.isclose(location, index, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Every exercise decision must align with the numerical time grid")
        result.setdefault(index, []).append(decision)
    return schedule, result


@dataclass(frozen=True)
class NumericalResult:
    value: float
    time_steps: int
    space_steps: int | None
    decision_count: int


def tree_price(model: EscrowedModel, option: Option, steps: int, decisions=None) -> NumericalResult:
    """Recombining CRR tree for the driftless discounted risky component Y."""
    schedule, indexed = _indexed_schedule(model, option, decisions, steps)
    if model.risky_volatility == 0:
        exact = max(float(model.reward(option, d, model.risky_spot)) for d in schedule)
        return NumericalResult(exact, steps, None, len(schedule))
    jump = model.risky_volatility * math.sqrt(option.maturity / steps)
    probability = 1 / (1 + math.exp(jump))  # E[Y_next | Y] = Y.
    with np.errstate(over="raise", invalid="raise"):
        y = model.risky_spot * np.exp((2 * np.arange(steps + 1) - steps) * jump)
        values = np.zeros(steps + 1)
        for decision in reversed(indexed[steps]):
            values = np.maximum(values, model.reward(option, decision, y))
        for index in range(steps - 1, -1, -1):
            values = (1 - probability) * values[:-1] + probability * values[1:]
            if index in indexed:
                y = model.risky_spot * np.exp((2 * np.arange(index + 1) - index) * jump)
                for decision in reversed(indexed[index]):
                    values = np.maximum(values, model.reward(option, decision, y))
    if not np.isfinite(values[0]):
        raise FloatingPointError("Non-finite tree result")
    return NumericalResult(float(values[0]), steps, None, len(schedule))


def finite_difference_price(model: EscrowedModel, option: Option, space_steps: int,
                            time_steps: int, y_max: float, decisions=None,
                            smoothing: bool = True,
                            damping_times: tuple[float, ...] = ()) -> NumericalResult:
    """Solve v_t + sigma^2 y^2 v_yy / 2 = 0 between discounted obstacles.

    Central spatial differences and Crank-Nicolson time steps; two backward-Euler
    half steps follow each obstacle when smoothing=True. Upper boundary uses the
    asymptotic linear payoff, so domain expansion remains a required check.
    Extra damping times allow nested-rights comparisons to share a propagator.
    They add no exercise rights; actual obstacles are always damped if enabled.
    """
    space_steps = positive_integer(space_steps, "space_steps", 3)
    schedule, indexed = _indexed_schedule(model, option, decisions, time_steps)
    _, extra_damping = _indexed_schedule(
        model, option, [Decision(t) for t in damping_times], time_steps)
    damping_indices = set(indexed) | set(extra_damping)
    if model.risky_volatility <= 0:
        raise ValueError("Use the exact deterministic/tree branch at zero volatility")
    intercepts = {d: model.reserve(d) - option.strike * model.curve.discount(d.time) for d in schedule}
    if not math.isfinite(y_max) or y_max <= max(model.risky_spot, max(abs(v) for v in intercepts.values())):
        raise ValueError("Upper domain must exceed initial risky state and payoff kinks")
    grid = np.linspace(0.0, y_max, space_steps + 1)
    diffusion = 0.5 * model.risky_volatility**2 * np.arange(1, space_steps, dtype=float)**2
    diagonal = -2 * diffusion
    dt = option.maturity / time_steps

    def boundaries(index_position):
        future = [intercepts[d] for index, ds in indexed.items() if index > index_position for d in ds]
        if option.kind == "call":
            best = max(future)
            return max(0.0, best), max(0.0, y_max + best)
        best = max(-v for v in future)
        return max(0.0, best), max(0.0, best - y_max)

    def step(values, new_index, h, theta):
        rhs = values[1:-1] + (1 - theta) * h * (
            diffusion * values[:-2] + diagonal * values[1:-1] + diffusion * values[2:])
        left, right = boundaries(new_index)
        rhs[0] += theta * h * diffusion[0] * left
        rhs[-1] += theta * h * diffusion[-1] * right
        band = np.zeros((3, space_steps - 1))
        band[0, 1:] = -theta * h * diffusion[:-1]
        band[1] = 1 - theta * h * diagonal
        band[2, :-1] = -theta * h * diffusion[1:]
        updated = np.empty_like(values)
        updated[0], updated[-1] = left, right
        updated[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        return updated

    values = np.zeros(space_steps + 1)
    for decision in reversed(indexed[time_steps]):
        values = np.maximum(values, model.reward(option, decision, grid))
    for index in range(time_steps - 1, -1, -1):
        if smoothing and index + 1 in damping_indices:
            values = step(values, index + 0.5, dt / 2, 1.0)
            values = step(values, index, dt / 2, 1.0)
        else:
            values = step(values, index, dt, 0.5)
        for decision in reversed(indexed.get(index, ())):
            values = np.maximum(values, model.reward(option, decision, grid))
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite PDE result")
    return NumericalResult(float(np.interp(model.risky_spot, grid, values)), time_steps, space_steps, len(schedule))
