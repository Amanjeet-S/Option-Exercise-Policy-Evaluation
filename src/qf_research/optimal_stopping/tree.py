"""Cox-Ross-Rubinstein backward induction with exact exercise-grid alignment."""

from dataclasses import dataclass

import numpy as np

from .contracts import Market, Option, exercise_indices, exercise_times, positive_integer


@dataclass(frozen=True)
class TreeResult:
    value: float
    steps: int
    exercise_boundary: dict[float, float]


def binomial_price(market: Market, option: Option, steps: int,
                   dates: tuple[float, ...] | list[float] | np.ndarray | None = None) -> TreeResult:
    """Value European or Bermudan options; dates=None means exercise only at expiry.

    Finite-grid prices are numerical references, not rigorous bounds on a diffusion
    price. American-style tests can pass every positive tree date explicitly.
    """
    steps = positive_integer(steps, "steps")
    times = exercise_times(option, dates)
    allowed = set(exercise_indices(option, times, steps).tolist())
    dt = option.maturity / steps
    discount = np.exp(-market.rate * dt)
    if market.volatility == 0:
        deterministic = market.spot * np.exp((market.rate - market.dividend) * times)
        value = np.max(np.exp(-market.rate * times) * option.payoff(deterministic))
        return TreeResult(float(value), steps, {})
    log_u = market.volatility * np.sqrt(dt)
    up, down = np.exp(log_u), np.exp(-log_u)
    probability = (np.exp((market.rate - market.dividend) * dt) - down) / (up - down)
    if not 0 <= probability <= 1:
        raise ValueError("CRR no-arbitrage probability is outside [0, 1]; increase steps")
    terminal = market.spot * np.exp((2 * np.arange(steps + 1) - steps) * log_u)
    values = option.payoff(terminal)
    boundary = {}
    for index in range(steps - 1, -1, -1):
        values = discount * ((1 - probability) * values[:-1] + probability * values[1:])
        if index in allowed:
            nodes = market.spot * np.exp((2 * np.arange(index + 1) - index) * log_u)
            immediate = option.payoff(nodes)
            exercise = (immediate > 0) & (immediate > values)
            if np.any(exercise):
                bound = np.max(nodes[exercise]) if option.kind == "put" else np.min(nodes[exercise])
                boundary[float(index * dt)] = float(bound)
            values = np.maximum(values, immediate)
    return TreeResult(float(values[0]), steps, dict(sorted(boundary.items())))

