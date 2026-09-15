"""Contract and risk-neutral market inputs, with explicit exercise conventions."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def positive_integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class Market:
    """Constant coefficients under Q; rates are continuously compounded per year."""

    spot: float
    rate: float
    volatility: float
    dividend: float = 0.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(x) for x in (self.spot, self.rate, self.volatility, self.dividend)):
            raise ValueError("Market inputs must be finite")
        if self.spot <= 0 or self.volatility < 0:
            raise ValueError("spot must be positive and volatility non-negative")


@dataclass(frozen=True)
class Option:
    strike: float
    maturity: float
    kind: str = "put"

    def __post_init__(self) -> None:
        if not np.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be finite and positive")
        if not np.isfinite(self.maturity) or self.maturity < 0:
            raise ValueError("maturity must be finite and non-negative")
        if self.kind not in {"call", "put"}:
            raise ValueError("kind must be 'call' or 'put'")

    def payoff(self, spots: np.ndarray | float) -> np.ndarray:
        sign = 1.0 if self.kind == "call" else -1.0
        return np.maximum(sign * (np.asarray(spots) - self.strike), 0.0)


def exercise_times(option: Option, dates: tuple[float, ...] | list[float] | np.ndarray | None) -> np.ndarray:
    """Return sorted valid exercise dates. None means European. Time zero is excluded.

    Maturity is always included. The same calendar must be supplied to every method
    in a Bermudan comparison. An American contract is approximated by refinement.
    """
    if option.maturity <= 0:
        raise ValueError("Exercise methods require positive maturity")
    if dates is None:
        return np.array([option.maturity], dtype=float)
    values = np.asarray(dates, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Exercise dates must be a finite one-dimensional sequence")
    if np.any(values <= 0) or np.any(values > option.maturity):
        raise ValueError("Exercise dates must lie in (0, maturity]")
    if len(np.unique(values)) != len(values):
        raise ValueError("Exercise dates must not contain duplicates")
    return np.unique(np.append(values, option.maturity))


def exercise_indices(option: Option, dates: np.ndarray, steps: int) -> np.ndarray:
    """Require exact alignment instead of silently rounding a contract's dates."""
    positive_integer(steps, "steps")
    locations = dates / option.maturity * steps
    indices = np.rint(locations).astype(int)
    if not np.allclose(locations, indices, atol=1e-9, rtol=0.0):
        raise ValueError("Every exercise date must align with the numerical time grid")
    return indices

