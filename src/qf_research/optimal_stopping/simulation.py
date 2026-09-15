"""Exact GBM sampling at specified dates and sampling-only uncertainty estimates."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from .contracts import Market, Option, positive_integer


@dataclass(frozen=True)
class Estimate:
    value: float
    standard_error: float
    lower: float
    upper: float
    observations: int
    confidence: float


def estimate_samples(samples: np.ndarray, confidence: float = 0.95) -> Estimate:
    """Approximate Student-t interval for independent discounted payoff samples.

    The interval is not exact for non-Gaussian payoffs and does not cover model,
    exercise-policy training or deterministic discretisation error.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("At least two finite one-dimensional samples are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    value = float(samples.mean())
    se = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    radius = float(student_t.ppf((1 + confidence) / 2, len(samples) - 1) * se)
    return Estimate(value, se, value - radius, value + radius, len(samples), confidence)


def simulate_gbm(market: Market, times: np.ndarray, n_paths: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Risk-neutral paths of shape (n_paths, len(times)); first time must be zero.

    Uses exact GBM transitions. Refining the date grid adds exercise opportunities,
    not an Euler approximation to the underlying process.
    """
    positive_integer(n_paths, "n_paths")
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("times must start at zero and be finite and strictly increasing")
    dt = np.diff(times)
    z = rng.standard_normal((n_paths, len(dt)))
    increments = ((market.rate - market.dividend - market.volatility**2 / 2) * dt
                  + market.volatility * np.sqrt(dt) * z)
    paths = np.empty((n_paths, len(times)))
    paths[:, 0] = market.spot
    paths[:, 1:] = market.spot * np.exp(np.cumsum(increments, axis=1))
    if not np.all(np.isfinite(paths)) or np.any(paths <= 0):
        raise FloatingPointError("GBM paths overflowed or underflowed; reduce input extremes")
    return paths


def european_monte_carlo(market: Market, option: Option, n_paths: int,
                         rng: np.random.Generator, antithetic: bool = False) -> Estimate:
    """Price a European option. An antithetic pair is one independent observation.

    With antithetic=True, n_paths must be even; uncertainty uses n_paths / 2 pair
    averages rather than incorrectly treating the correlated paths as independent.
    """
    positive_integer(n_paths, "n_paths", 2)
    if antithetic and (n_paths < 4 or n_paths % 2):
        raise ValueError("Antithetic simulation requires an even n_paths of at least four")
    size = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(size)
    t = option.maturity
    base = (market.rate - market.dividend - market.volatility**2 / 2) * t
    shock = market.volatility * np.sqrt(t) * z
    payoff = option.payoff(market.spot * np.exp(base + shock))
    if antithetic:
        payoff = (payoff + option.payoff(market.spot * np.exp(base - shock))) / 2
    return estimate_samples(np.exp(-market.rate * t) * payoff)

