"""Finite-calendar stopping with matched physical settlement and cash dividends.

Y is the discounted economic ownership value. A trade quote Q with cash factor
f is Q = Y/f, conditional on identical economic entitlement. Between dividend
events dY = sigma Y dW. At an event, min(Y, cap) becomes a receivable. Exercise
has discounted reward (sign * (Y - K*f_exercise))^+. See funded_cash_model.md.

This is a frictionless benchmark, not certification of listed-contract terms.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_banded

from .analytical import black_scholes
from .contracts import Market, Option, positive_integer
from .escrowed_dividends import CashDividend, Decision, DiscountFactors
from .cash_dividends import CashJumpModel, lognormal_transition, one_event_european


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


@dataclass(frozen=True)
class FundedExercise:
    time: float
    settlement_factor: float
    phase: str = "after"

    def __post_init__(self):
        Decision(self.time, self.phase)
        _finite(self.settlement_factor, "Settlement factor", positive=True)


@dataclass(frozen=True)
class DividendCap:
    ex_time: float
    discounted_cash: float

    def __post_init__(self):
        _finite(self.ex_time, "Ex-time", positive=True)
        _finite(self.discounted_cash, "Dividend cap")


@dataclass(frozen=True)
class FundedCashModel:
    quoted_spot: float
    volatility: float
    initial_share_factor: float
    premium_factor: float
    dividends: tuple[DividendCap, ...] = ()

    def __post_init__(self):
        _finite(self.quoted_spot, "Quoted spot", positive=True)
        _finite(self.volatility, "Volatility")
        _finite(self.initial_share_factor, "Initial share factor", positive=True)
        _finite(self.premium_factor, "Premium factor", positive=True)
        if not isinstance(self.dividends, tuple) or not all(isinstance(e, DividendCap) for e in self.dividends):
            raise ValueError("Use an immutable schedule of DividendCap objects")
        if any(b.ex_time <= a.ex_time for a, b in zip(self.dividends, self.dividends[1:])):
            raise ValueError("Dividend ex-times must increase strictly")
        _finite(self.initial_state, "Initial prepaid state", positive=True)

    @property
    def initial_state(self):
        return self.quoted_spot * self.initial_share_factor

    def quote(self, state, settlement_factor):
        _finite(settlement_factor, "Settlement factor", positive=True)
        values = np.asarray(state, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("States must be finite and non-negative")
        return values / settlement_factor

    def reward(self, option, right, state):
        difference = np.asarray(state) - option.strike * right.settlement_factor
        return np.maximum(difference if option.kind == "call" else -difference, 0.)

    def jump(self, state, event):
        values = np.asarray(state, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("States must be finite and non-negative")
        receivable = np.minimum(values, event.discounted_cash)
        return values - receivable, receivable


def _schedule(model, option, decisions):
    if option.maturity <= 0 or not isinstance(decisions, tuple) or not decisions:
        raise ValueError("Positive maturity and an explicit immutable exercise schedule are required")
    if not all(isinstance(d, FundedExercise) for d in decisions):
        raise ValueError("Every exercise right requires its settlement factor")
    keys = [(d.time, d.phase) for d in decisions]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate exercise time/phase, possibly conflicting factors")
    if (option.maturity, "after") not in keys:
        raise ValueError("An explicit terminal post-event settlement factor is required")
    events = {e.ex_time: e for e in model.dividends if e.ex_time <= option.maturity}
    for right in decisions:
        if right.time > option.maturity:
            raise ValueError("Exercise after maturity")
        if right.phase == "before" and right.time not in events:
            raise ValueError("Before phase must match a dividend event exactly")
    return tuple(sorted(decisions, key=lambda d: (d.time, d.phase == "after"))), events


def _apply(model, option, grid, values, t, rights, events):
    for d in rights:
        if d.time == t and d.phase == "after":
            values = np.maximum(values, model.reward(option, d, grid))
    if t in events:
        after, _ = model.jump(grid, events[t])
        values = np.interp(after, grid, values)
    for d in rights:
        if d.time == t and d.phase == "before":
            values = np.maximum(values, model.reward(option, d, grid))
    return values


def _grid(model, option, rights, events, space_steps, y_max):
    space_steps = positive_integer(space_steps, "space_steps", 3)
    _finite(y_max, "Domain maximum", positive=True)
    threshold = max(option.strike*d.settlement_factor for d in rights)
    threshold += sum(e.discounted_cash for e in events.values())
    if y_max <= max(model.initial_state, threshold):
        raise ValueError("Domain must exceed the initial state and exercise thresholds")
    return np.linspace(0., y_max, space_steps+1)


def _anchors(option, rights, events, extra_times):
    for t in extra_times:
        _finite(t, "Additional propagation time")
        if t > option.maturity:
            raise ValueError("Propagation time after maturity")
    return sorted({0., option.maturity} | {d.time for d in rights} | set(events) | set(extra_times))


@dataclass(frozen=True)
class FundedResult:
    present_value: float
    premium_quote: float
    space_steps: int
    time_intervals: int
    decision_count: int
    event_count: int


def _result(value, model, grid, intervals, rights, events):
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite funded price")
    return FundedResult(value, value/model.premium_factor, len(grid)-1,
                        intervals, len(rights), len(events))


def deterministic_value(model, option, decisions):
    if model.volatility != 0:
        raise ValueError("Deterministic valuation requires zero volatility")
    rights, events = _schedule(model, option, decisions)
    state, best = model.initial_state, 0.
    for t in _anchors(option, rights, events, ()):
        for d in rights:
            if d.time == t and d.phase == "before":
                best = max(best, float(model.reward(option, d, state)))
        if t in events:
            state = float(model.jump(state, events[t])[0])
        for d in rights:
            if d.time == t and d.phase == "after":
                best = max(best, float(model.reward(option, d, state)))
    return best


def integral_price(model, option, space_steps, y_max, decisions, *, propagation_times=()):
    """Exact lognormal integration of a spatial interpolant on exact event times."""
    rights, events = _schedule(model, option, decisions)
    grid = _grid(model, option, rights, events, space_steps, y_max)
    times = _anchors(option, rights, events, propagation_times)
    if model.volatility == 0:
        return _result(deterministic_value(model, option, rights), model, grid, len(times)-1, rights, events)
    values = _apply(model, option, grid, np.zeros_like(grid), option.maturity, rights, events)
    for a, b in reversed(list(zip(times, times[1:]))):
        values = lognormal_transition(grid, values, model.volatility*math.sqrt(b-a),
                                      1. if option.kind == "call" else 0.)
        values = _apply(model, option, grid, values, a, rights, events)
    return _result(float(np.interp(model.initial_state, grid, values)), model, grid, len(times)-1, rights, events)


def finite_difference_price(model, option, space_steps, time_steps, y_max, decisions,
                            *, propagation_times=()):
    """CN/Rannacher on intervals split at every exact event/exercise timestamp.

    `time_steps` bounds maximum step width by maturity/time_steps. Each mandatory
    interval is independently subdivided, so the actual count may be larger.
    Additional propagation times grant no rights, but keep grids and damping
    fixed across comparisons of nested exercise sets.
    """
    time_steps = positive_integer(time_steps, "time_steps")
    rights, events = _schedule(model, option, decisions)
    grid = _grid(model, option, rights, events, space_steps, y_max)
    anchors = _anchors(option, rights, events, propagation_times)
    maximum_step = option.maturity/time_steps
    times = [0.]
    for a, b in zip(anchors, anchors[1:]):
        count = max(1, math.ceil((b-a)/maximum_step))
        times.extend(np.linspace(a, b, count+1)[1:])
    if model.volatility == 0:
        return _result(deterministic_value(model, option, rights), model, grid, len(times)-1, rights, events)
    diffusion = .5*model.volatility**2*np.arange(1, len(grid)-1, dtype=float)**2

    def boundary(t):
        future = [d for d in rights if d.time > t]
        if option.kind == "put":
            return max(option.strike*d.settlement_factor for d in future), 0.
        high = []
        for d in future:
            detached = sum(e.discounted_cash for e in events.values()
                           if e.ex_time > t and (e.ex_time < d.time or
                                               (e.ex_time == d.time and d.phase == "after")))
            high.append(y_max - detached - option.strike*d.settlement_factor)
        return 0., max(0., max(high))

    def step(values, earlier, width, theta):
        rhs = values[1:-1] + (1-theta)*width*diffusion*(values[:-2]-2*values[1:-1]+values[2:])
        left, right = boundary(earlier)
        rhs[0] += theta*width*diffusion[0]*left
        rhs[-1] += theta*width*diffusion[-1]*right
        band = np.zeros((3, len(grid)-2))
        band[0, 1:] = -theta*width*diffusion[:-1]
        band[1] = 1 + 2*theta*width*diffusion
        band[2, :-1] = -theta*width*diffusion[1:]
        return np.r_[left, solve_banded((1, 1), band, rhs, check_finite=False), right]

    mandatory = set(anchors)
    values = _apply(model, option, grid, np.zeros_like(grid), option.maturity, rights, events)
    for a, b in reversed(list(zip(times, times[1:]))):
        width = b-a
        if b in mandatory:
            values = step(values, (a+b)/2, width/2, 1.)
            values = step(values, a, width/2, 1.)
        else:
            values = step(values, a, width, .5)
        if a in mandatory:
            values = _apply(model, option, grid, values, a, rights, events)
    return _result(float(np.interp(model.initial_state, grid, values)), model, grid, len(times)-1, rights, events)


@dataclass(frozen=True)
class EuropeanReference:
    present_value: float
    premium_quote: float
    expected_discounted_payment: float
    cap_probability: float
    quadrature_error_estimate: float
    omitted_tail_bound: float


def european_reference(model, option, terminal):
    """Zero-event formula or one-event conditioning, without a spatial grid.

    Reduce to a zero-rate cash-jump problem in discounted units. Reuses the
    previously checked conditional quadrature, not either funded continuation
    solver. More than one event is deliberately unsupported by this oracle.
    """
    _, events = _schedule(model, option, (terminal,))
    transformed = Option(option.strike*terminal.settlement_factor, option.maturity, option.kind)
    if not events:
        value = black_scholes(Market(model.initial_state, 0., model.volatility), transformed)
        return EuropeanReference(value, value/model.premium_factor, 0., 0., 0., 0.)
    if len(events) != 1:
        raise ValueError("European reference supports at most one event")
    event = next(iter(events.values()))
    unit_curve = DiscountFactors((0., option.maturity), (1., 1.))
    reduced = CashJumpModel(model.initial_state, model.volatility, unit_curve,
                           (CashDividend(event.ex_time, event.ex_time, event.discounted_cash),))
    result = one_event_european(reduced, transformed)
    return EuropeanReference(result.value, result.value/model.premium_factor,
                             result.expected_discounted_payment, result.cap_probability,
                             result.quadrature_error_estimate, result.omitted_tail_bound)
