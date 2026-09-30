"""Piecewise-lognormal share dynamics with capped dividend entitlements.

Y = D(0,t) S diffuses with zero drift between events and jumps to (Y-c)^+
at an ex-event, where c = declared cash times its payment-date discount factor.
The cap changes the payment when assets are insufficient; it is not fixed cash
plus a free stock-price floor. See the project's cash_dividend_model.md.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_banded
from scipy.special import ndtr

from .contracts import Option, positive_integer
from .escrowed_dividends import CashDividend, Decision, DiscountFactors


@dataclass(frozen=True)
class CashJumpModel:
    spot: float
    volatility: float
    curve: DiscountFactors
    dividends: tuple[CashDividend, ...] = ()

    def __post_init__(self):
        if any(isinstance(x, bool) or not math.isfinite(x) for x in (self.spot, self.volatility)):
            raise ValueError("Spot and volatility must be finite numbers")
        if self.spot <= 0 or self.volatility < 0:
            raise ValueError("Require positive spot and non-negative volatility")
        if not isinstance(self.dividends, tuple):
            raise ValueError("Use an immutable dividend schedule")
        if any(b.ex_time <= a.ex_time for a, b in zip(self.dividends, self.dividends[1:])):
            raise ValueError("Use strictly ordered, unique ex-times")
        for event in self.dividends:
            self.curve.discount(event.pay_time)

    def event_cap(self, event: CashDividend) -> float:
        """Time-zero currency value of the maximum detached receivable."""
        return event.amount * self.curve.discount(event.pay_time)

    def jump(self, state, event: CashDividend):
        state = np.asarray(state, dtype=float)
        if np.any(~np.isfinite(state)) or np.any(state < 0):
            raise ValueError("Discounted share states must be finite and non-negative")
        paid = np.minimum(state, self.event_cap(event))
        return state - paid, paid

    def reward(self, option: Option, time: float, state):
        difference = np.asarray(state) - option.strike * self.curve.discount(time)
        return np.maximum(difference if option.kind == "call" else -difference, 0.0)


def _schedule(model, option, decisions):
    if option.maturity <= 0:
        raise ValueError("Require positive maturity")
    model.curve.discount(option.maturity)
    rights = list(() if decisions is None else decisions)
    if len(set(rights)) != len(rights):
        raise ValueError("Duplicate decisions")
    terminal = Decision(option.maturity)
    if terminal not in rights:
        rights.append(terminal)
    events = {d.ex_time: d for d in model.dividends if d.ex_time <= option.maturity}
    for right in rights:
        if right.time > option.maturity:
            raise ValueError("Decision after maturity")
        if right.phase == "before" and right.time not in events:
            raise ValueError("Before phase must match an exact ex-event")
    return tuple(sorted(rights, key=lambda d: (d.time, d.phase == "after"))), events


def _apply_event(model, option, grid, values, time, rights, events):
    # Reverse chronology: post-event exercise, jump mapping, pre-event exercise.
    if Decision(time) in rights:
        values = np.maximum(values, model.reward(option, time, grid))
    if time in events:
        after, _ = model.jump(grid, events[time])
        values = np.interp(after, grid, values)
    if Decision(time, "before") in rights:
        values = np.maximum(values, model.reward(option, time, grid))
    return values


def deterministic_price(model, option, decisions=None):
    """Exact stopping along a zero-volatility path, including the cash cap."""
    if model.volatility != 0:
        raise ValueError("Deterministic valuation requires zero volatility")
    rights, events = _schedule(model, option, decisions)
    state, best = model.spot, 0.0
    for time in sorted(set(events) | {d.time for d in rights}):
        if Decision(time, "before") in rights:
            best = max(best, float(model.reward(option, time, state)))
        if time in events:
            state = float(model.jump(state, events[time])[0])
        if Decision(time) in rights:
            best = max(best, float(model.reward(option, time, state)))
    return best


@dataclass(frozen=True)
class CashJumpResult:
    value: float
    space_steps: int
    time_steps: int
    decision_count: int
    event_count: int


def _domain(model, option, rights, events, space_steps, y_max):
    space_steps = positive_integer(space_steps, "space_steps", 3)
    threshold = max(option.strike * model.curve.discount(d.time) for d in rights)
    threshold += sum(model.event_cap(e) for e in events.values())
    if not math.isfinite(y_max) or y_max <= max(model.spot, threshold):
        raise ValueError("Domain must exceed spot and the high-state exercise thresholds")
    return np.linspace(0.0, y_max, space_steps + 1)


def finite_difference_price(model, option, space_steps, time_steps, y_max,
                            decisions=None, damping_times=()):
    """Discounted-state PDE with cash jumps, obstacles and Rannacher damping.

    All ex-events up to maturity must be grid aligned, even without exercise.
    Events and actual exercise rights always trigger damping. Extra damping
    times allow comparisons with an otherwise common temporal propagator.
    """
    time_steps = positive_integer(time_steps, "time_steps")
    rights, events = _schedule(model, option, decisions)
    grid = _domain(model, option, rights, events, space_steps, y_max)

    def index(time):
        if not math.isfinite(time) or not 0 <= time <= option.maturity:
            raise ValueError("Invalid time")
        location = time / option.maturity * time_steps
        result = round(location)
        if not math.isclose(location, result, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Every decision and ex-event must align with the time grid")
        return result

    event_indices = {index(t): (t, e) for t, e in events.items()}
    right_indices = {d: index(d.time) for d in rights}
    active_times = {index(t): t for t in set(events) | {d.time for d in rights}}
    damping = set(active_times) | {index(t) for t in damping_times}
    if model.volatility == 0:
        return CashJumpResult(deterministic_price(model, option, rights), space_steps,
                              time_steps, len(rights), len(events))
    diffusion = 0.5 * model.volatility**2 * np.arange(1, space_steps, dtype=float)**2
    dt = option.maturity / time_steps

    def boundaries(position):
        future = [d for d in rights if right_indices[d] > position]
        if option.kind == "put":
            return max(option.strike * model.curve.discount(d.time) for d in future), 0.0
        upper = []
        for d in future:
            detached = sum(model.event_cap(e) for i, (t, e) in event_indices.items()
                           if i > position and (t < d.time or (t == d.time and d.phase == "after")))
            upper.append(y_max - detached - option.strike * model.curve.discount(d.time))
        return 0.0, max(0.0, max(upper))

    def step(values, position, h, theta):
        rhs = values[1:-1] + (1-theta)*h*diffusion*(values[:-2]-2*values[1:-1]+values[2:])
        left, right = boundaries(position)
        rhs[0] += theta*h*diffusion[0]*left
        rhs[-1] += theta*h*diffusion[-1]*right
        band = np.zeros((3, space_steps-1))
        band[0, 1:] = -theta*h*diffusion[:-1]
        band[1] = 1 + 2*theta*h*diffusion
        band[2, :-1] = -theta*h*diffusion[1:]
        return np.r_[left, solve_banded((1, 1), band, rhs, check_finite=False), right]

    values = _apply_event(model, option, grid, np.zeros_like(grid), option.maturity, rights, events)
    for i in range(time_steps-1, -1, -1):
        if i+1 in damping:
            values = step(values, i+0.5, dt/2, 1.0)
            values = step(values, i, dt/2, 1.0)
        else:
            values = step(values, i, dt, 0.5)
        if i in active_times:
            values = _apply_event(model, option, grid, values, active_times[i], rights, events)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite PDE result")
    return CashJumpResult(float(np.interp(model.spot, grid, values)), space_steps,
                          time_steps, len(rights), len(events))


def lognormal_transition(grid, values, volatility_time, tail_slope, block_size=128):
    """Integrate a piecewise-linear value function under a mean-preserving GBM.

    The transition is exact for the interpolant, not the underlying value
    function. A linear upper tail is an explicit domain approximation.
    Uses lognormal probabilities and first partial moments, not PDE steps.
    """
    block_size = positive_integer(block_size, "block_size")
    grid, values = np.asarray(grid, dtype=float), np.asarray(values, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or grid.shape != values.shape or grid[0] != 0
            or np.any(np.diff(grid) <= 0) or not np.all(np.isfinite(grid))
            or not np.all(np.isfinite(values)) or not math.isfinite(tail_slope)):
        raise ValueError("Require finite values on an increasing non-negative grid")
    if not math.isfinite(volatility_time) or volatility_time < 0:
        raise ValueError("Require finite non-negative volatility times square-root time")
    if volatility_time == 0:
        return values.copy()
    slopes = np.diff(values) / np.diff(grid)
    intercepts = values[:-1] - slopes * grid[:-1]
    result = np.empty_like(values)
    result[0] = values[0]  # Zero is absorbing, with a genuine atom after liquidation.
    sd = volatility_time
    for start in range(1, len(grid), block_size):
        end = min(start + block_size, len(grid))
        states = grid[start:end, None]
        z = (np.log(grid[None, 1:]) - np.log(states)) / sd
        probabilities = ndtr(z + sd/2)
        moments = states * ndtr(z - sd/2)
        masses = np.diff(probabilities, axis=1, prepend=0.0)
        partial_moments = np.diff(moments, axis=1, prepend=0.0)
        tail_mass = ndtr(-z[:, -1] - sd/2)
        tail_moment = states[:, 0] * ndtr(-z[:, -1] + sd/2)
        result[start:end] = (masses @ intercepts + partial_moments @ slopes
                             + (values[-1] - tail_slope*grid[-1])*tail_mass
                             + tail_slope*tail_moment)
    return result


def integral_price(model, option, space_steps, y_max, decisions=None, integration_times=()):
    """Backward lognormal integration on an event/exercise calendar.

    No diffusion time discretisation is required. Spatial interpolation,
    event mapping and the linear tail still require convergence checks. Extra
    integration times hold projection dates fixed when comparing exercise sets;
    they do not grant additional exercise rights.
    """
    rights, events = _schedule(model, option, decisions)
    grid = _domain(model, option, rights, events, space_steps, y_max)
    if any(not math.isfinite(t) or not 0 <= t <= option.maturity for t in integration_times):
        raise ValueError("Integration times must lie within the valuation horizon")
    times = sorted({0.0, option.maturity} | set(events) | {d.time for d in rights} | set(integration_times))
    if model.volatility == 0:
        return CashJumpResult(deterministic_price(model, option, rights), space_steps,
                              len(times)-1, len(rights), len(events))
    values = _apply_event(model, option, grid, np.zeros_like(grid), option.maturity, rights, events)
    for previous, current in reversed(list(zip(times, times[1:]))):
        values = lognormal_transition(grid, values, model.volatility*math.sqrt(current-previous),
                                      1.0 if option.kind == "call" else 0.0)
        values = _apply_event(model, option, grid, values, previous, rights, events)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite integral result")
    return CashJumpResult(float(np.interp(model.spot, grid, values)), space_steps,
                          len(times)-1, len(rights), len(events))


def _lognormal_option(state, strike, sd, kind):
    if state == 0 or sd == 0:
        return max(state-strike if kind == "call" else strike-state, 0.0)
    if strike == 0:
        return state if kind == "call" else 0.0
    d1 = (math.log(state)-math.log(strike))/sd + sd/2
    d2 = d1-sd
    sign = 1 if kind == "call" else -1
    return float(sign*(state*ndtr(sign*d1)-strike*ndtr(sign*d2)))


@dataclass(frozen=True)
class QuadratureResult:
    value: float
    quadrature_error_estimate: float
    omitted_tail_bound: float
    cap_probability: float
    expected_discounted_payment: float


def one_event_european(model, option):
    """Independent one-event oracle via conditioning and adaptive quadrature.

    A finite normal domain has a separate analytic tail bound. The integration
    library's error estimate is diagnostic, not a rigorous numerical certificate.
    Events after maturity have no effect and are excluded.
    """
    _, events = _schedule(model, option, None)
    if len(events) != 1:
        raise ValueError("Oracle requires exactly one ex-event by maturity")
    event = next(iter(events.values()))
    cap = model.event_cap(event)
    strike = option.strike * model.curve.discount(option.maturity)
    sd = model.volatility * math.sqrt(event.ex_time)
    if sd == 0:
        paid = min(model.spot, cap)
        value = float(model.reward(option, option.maturity, model.spot-paid))
        return QuadratureResult(value, 0.0, 0.0, float(model.spot <= cap), paid)
    cap_probability = float(ndtr((math.log(cap/model.spot)+sd**2/2)/sd)) if cap > 0 else 0.0
    expected_payment = cap - _lognormal_option(model.spot, cap, sd, "put")
    if event.ex_time == option.maturity:
        if option.kind == "call":
            value = _lognormal_option(model.spot, strike+cap, sd, "call")
        else:
            value = (_lognormal_option(model.spot, strike+cap, sd, "put")
                     - _lognormal_option(model.spot, cap, sd, "put"))
        return QuadratureResult(value, 0.0, 0.0, cap_probability, expected_payment)
    remaining_sd = model.volatility * math.sqrt(option.maturity-event.ex_time)
    limit = max(12.0, sd+12.0)
    cut = (math.log(cap/model.spot)+sd**2/2)/sd if cap > 0 else -limit
    nodes = sorted({-limit, min(limit, max(-limit, cut)), limit})

    def integrand(z):
        before = model.spot * math.exp(-sd**2/2 + sd*z)
        after = max(before-cap, 0.0)
        return _lognormal_option(after, strike, remaining_sd, option.kind) * math.exp(-z*z/2) / math.sqrt(2*math.pi)

    value, error = 0.0, 0.0
    for left, right in zip(nodes, nodes[1:]):
        part, estimate = quad(integrand, left, right, epsabs=1e-10, epsrel=1e-11, limit=200)
        value += part
        error += estimate
    tail = model.spot*(ndtr(sd-limit)+ndtr(-sd-limit)) + strike*2*ndtr(-limit)
    return QuadratureResult(value, error, float(tail), cap_probability, expected_payment)
