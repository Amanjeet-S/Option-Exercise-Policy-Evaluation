"""Bermudan backward PDE solver with a tridiagonal spatial operator.

Crank-Nicolson and Rannacher smoothing are established methods; see SOURCES.md.
The obstacle is applied only at the contract's exercise dates, not every PDE step.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .contracts import Market, Option, exercise_indices, exercise_times, positive_integer


@dataclass(frozen=True)
class FiniteDifferenceResult:
    value: float
    spots: np.ndarray
    values: np.ndarray
    exercise_boundary: dict[float, float]
    upwind_nodes: int


def finite_difference_price(market: Market, option: Option, space_steps: int,
                            time_steps: int, s_max: float,
                            dates: tuple[float, ...] | list[float] | np.ndarray | None = None,
                            theta: float = 0.5,
                            smoothing: bool = True) -> FiniteDifferenceResult:
    """Uniform spot grid with linear interpolation at the valuation spot.

    Two implicit half steps replace the first step after expiry and each exercise
    projection when smoothing=True. Central drift is used unless it creates a
    negative off-diagonal generator entry, in which case that node uses upwind
    drift. The number of adjusted nodes is returned. Boundary and grid errors
    remain, so convergence must be checked. Volatility must be positive.
    """
    space_steps = positive_integer(space_steps, "space_steps", 3)
    time_steps = positive_integer(time_steps, "time_steps")
    if market.volatility <= 0:
        raise ValueError("The PDE solver requires positive volatility; use the deterministic tree otherwise")
    if not np.isfinite(s_max) or s_max <= max(market.spot, option.strike):
        raise ValueError("s_max must be finite and exceed both spot and strike")
    if not np.isfinite(theta) or not 0.5 <= theta <= 1:
        raise ValueError("theta must lie in [0.5, 1]")
    times = exercise_times(option, dates)
    allowed = set(exercise_indices(option, times, time_steps).tolist())
    grid = np.linspace(0.0, s_max, space_steps + 1)
    dt = option.maturity / time_steps
    index = np.arange(1, space_steps, dtype=float)
    diffusion = 0.5 * market.volatility**2 * index**2
    drift = (market.rate - market.dividend) * index
    lower = diffusion - 0.5 * drift
    upper = diffusion + 0.5 * drift
    adjust = (lower < 0) | (upper < 0)
    lower[adjust] = diffusion[adjust] + np.maximum(-drift[adjust], 0)
    upper[adjust] = diffusion[adjust] + np.maximum(drift[adjust], 0)
    diagonal = -lower - upper - market.rate

    def boundary(calendar_time: float, include_current: bool = False) -> tuple[float, float]:
        # S=0 is absorbing in GBM. At the upper truncation, ignore the remote
        # chance of returning across the strike and optimise the linear payoff.
        remaining = times - calendar_time
        eligible = remaining >= -1e-12 if include_current else remaining > 1e-12
        remaining = np.maximum(remaining[eligible], 0)
        if len(remaining) == 0:
            return float(option.payoff(0)), float(option.payoff(s_max))
        if option.kind == "put":
            return float(np.max(option.strike * np.exp(-market.rate * remaining))), 0.0
        candidates = s_max * np.exp(-market.dividend * remaining) - option.strike * np.exp(-market.rate * remaining)
        return 0.0, float(max(0.0, np.max(candidates)))

    def step(values: np.ndarray, old_time: float, new_time: float, weight: float) -> np.ndarray:
        h = old_time - new_time
        rhs = values[1:-1] + (1 - weight) * h * (
            lower * values[:-2] + diagonal * values[1:-1] + upper * values[2:])
        left, right = boundary(new_time)
        rhs[0] += weight * h * lower[0] * left
        rhs[-1] += weight * h * upper[-1] * right
        band = np.zeros((3, space_steps - 1))
        band[0, 1:] = -weight * h * upper[:-1]
        band[1] = 1 - weight * h * diagonal
        band[2, :-1] = -weight * h * lower[1:]
        updated = np.empty_like(values)
        updated[0], updated[-1] = left, right
        updated[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        return updated

    values = option.payoff(grid)
    exercise_boundary = {}
    for j in range(time_steps - 1, -1, -1):
        old_time, new_time = (j + 1) * dt, j * dt
        if smoothing and j + 1 in allowed:
            midpoint = (old_time + new_time) / 2
            values = step(values, old_time, midpoint, 1.0)
            values = step(values, midpoint, new_time, 1.0)
        else:
            values = step(values, old_time, new_time, theta)
        if j in allowed:
            immediate = option.payoff(grid)
            exercising = (immediate[1:-1] > 0) & (immediate[1:-1] > values[1:-1])
            if np.any(exercising):
                nodes = grid[1:-1][exercising]
                edge = nodes.max() if option.kind == "put" else nodes.min()
                exercise_boundary[float(new_time)] = float(edge)
            values = np.maximum(values, immediate)
            values[0], values[-1] = boundary(new_time, include_current=True)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("PDE solve produced non-finite values")
    return FiniteDifferenceResult(float(np.interp(market.spot, grid, values)), grid, values,
                                  dict(sorted(exercise_boundary.items())), int(adjust.sum()))
