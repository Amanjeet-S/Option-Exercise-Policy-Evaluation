"""Black-Scholes-Merton benchmarks. Standard formulae, implemented for this project.

See projects/01_optimal_stopping/SOURCES.md and theory.md for attribution.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .contracts import Market, Option


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes(market: Market, option: Option) -> float:
    """European value, including zero volatility and expiry limiting cases."""
    s, k, t = market.spot, option.strike, option.maturity
    if t == 0:
        return float(option.payoff(s))
    r, q, sigma = market.rate, market.dividend, market.volatility
    if sigma == 0:
        return float(np.exp(-r * t) * option.payoff(s * np.exp((r - q) * t)))
    sd = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + sigma**2 / 2) * t) / sd
    d2 = d1 - sd
    sign = 1 if option.kind == "call" else -1
    return float(sign * (s * np.exp(-q * t) * ndtr(sign * d1)
                         - k * np.exp(-r * t) * ndtr(sign * d2)))


def black_scholes_greeks(market: Market, option: Option) -> Greeks:
    """Vega/rho are per unit change; theta is per calendar year, dV/dt_calendar.

    Singular boundary cases are rejected rather than assigned arbitrary Greeks.
    """
    s, k, t = market.spot, option.strike, option.maturity
    r, q, sigma = market.rate, market.dividend, market.volatility
    if t <= 0 or sigma <= 0:
        raise ValueError("Greeks require positive maturity and volatility")
    root_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + sigma**2 / 2) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    density = np.exp(-d1**2 / 2) / np.sqrt(2 * np.pi)
    sign = 1 if option.kind == "call" else -1
    ds, dk = np.exp(-q * t), np.exp(-r * t)
    delta = sign * ds * ndtr(sign * d1)
    gamma = ds * density / (s * sigma * root_t)
    vega = s * ds * density * root_t
    theta = (-s * ds * density * sigma / (2 * root_t)
             - sign * r * k * dk * ndtr(sign * d2)
             + sign * q * s * ds * ndtr(sign * d1))
    rho = sign * k * t * dk * ndtr(sign * d2)
    return Greeks(*(float(x) for x in (delta, gamma, vega, theta, rho)))

