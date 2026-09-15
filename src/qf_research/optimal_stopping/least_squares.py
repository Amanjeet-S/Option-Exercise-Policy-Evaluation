"""Longstaff-Schwartz exercise policies with independent forward evaluation.

The regression method is due to Longstaff and Schwartz (2001). This implementation
uses normalised monomials and SVD least squares, with explicit rank diagnostics.
See projects/01_optimal_stopping/SOURCES.md for methodological attribution.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyvander, polyval

from .analytical import black_scholes
from .contracts import Market, Option, exercise_times, positive_integer
from .simulation import Estimate, estimate_samples, simulate_gbm


@dataclass(frozen=True)
class Regression:
    centre: float
    scale: float
    coefficients: tuple[float, ...]
    observations: int
    rank: int
    condition_number: float

    def predict(self, spots: np.ndarray) -> np.ndarray:
        # Continuation values for non-negative payoffs cannot be negative.
        return np.maximum(polyval((spots - self.centre) / self.scale, self.coefficients), 0)


@dataclass(frozen=True)
class ExercisePolicy:
    market: Market
    option: Option
    times: tuple[float, ...]
    regressions: tuple[Regression | None, ...]
    requested_degree: int
    training_paths: int


@dataclass(frozen=True)
class PolicyEvaluation:
    raw: Estimate
    european_control: Estimate
    discounted_payoffs: np.ndarray
    stopping_indices: np.ndarray


def _regress(spots: np.ndarray, targets: np.ndarray, degree: int) -> Regression | None:
    if len(spots) < 2:
        return None  # Explicitly continue if the exercise region has no useful sample.
    centre = float(spots.mean())
    scale = float(spots.std())
    if scale < 1e-12 * max(abs(centre), 1.0):
        scale = 1.0
    x = (spots - centre) / scale
    degree = min(degree, len(spots) - 2)
    while degree >= 0:
        design = polyvander(x, degree)
        coefficients, _, rank, singular = np.linalg.lstsq(design, targets, rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
        if rank == degree + 1 and condition <= 1e10:
            return Regression(centre, scale, tuple(float(c) for c in coefficients),
                              len(spots), int(rank), condition)
        degree -= 1
    return None


def train_policy(market: Market, option: Option,
                 dates: tuple[float, ...] | list[float] | np.ndarray | None,
                 n_paths: int, degree: int, rng: np.random.Generator) -> ExercisePolicy:
    """Train backwards on one sample. No training-sample price is reported.

    At each date the response is the discounted realised cash flow selected by the
    already learnt later policy. Fits use in-the-money paths. Only current spot is
    a predictor; realised future states are never evaluation-time predictors.
    """
    positive_integer(n_paths, "n_paths", 2)
    degree = positive_integer(degree, "degree", 0)
    times = np.insert(exercise_times(option, dates), 0, 0.0)
    paths = simulate_gbm(market, times, n_paths, rng)
    cash = option.payoff(paths[:, -1])
    cash_times = np.full(n_paths, times[-1])
    regressions: list[Regression | None] = [None] * len(times)
    for j in range(len(times) - 2, 0, -1):
        immediate = option.payoff(paths[:, j])
        itm = immediate > 0
        targets = cash[itm] * np.exp(-market.rate * (cash_times[itm] - times[j]))
        fit = _regress(paths[itm, j], targets, degree)
        regressions[j] = fit
        if fit is not None:
            rows = np.flatnonzero(itm)
            chosen = rows[immediate[itm] > fit.predict(paths[itm, j])]
            cash[chosen] = immediate[chosen]
            cash_times[chosen] = times[j]
    return ExercisePolicy(market, option, tuple(float(t) for t in times),
                          tuple(regressions), degree, n_paths)


def evaluate_policy(policy: ExercisePolicy, paths: np.ndarray) -> PolicyEvaluation:
    """Evaluate a frozen policy forwards on independent, risk-neutral GBM paths.

    The caller must supply an independent sample with columns matching policy.times.
    Independence and the generating law cannot be established from an array alone.
    A fixed coefficient of one on the European control keeps the estimate unbiased
    conditional on the trained policy, without fitting to the evaluation sample.
    """
    paths = np.asarray(paths, dtype=float)
    if (paths.ndim != 2 or paths.shape[0] < 2 or paths.shape[1] != len(policy.times)
            or not np.all(np.isfinite(paths)) or np.any(paths <= 0)
            or not np.allclose(paths[:, 0], policy.market.spot, rtol=1e-12, atol=0)):
        raise ValueError("paths must be finite positive paths with the policy's initial spot and date columns")
    n_paths = len(paths)
    stopping = np.full(n_paths, len(policy.times) - 1, dtype=int)
    alive = np.ones(n_paths, dtype=bool)
    for j in range(1, len(policy.times) - 1):
        fit = policy.regressions[j]
        if fit is None:
            continue
        immediate = policy.option.payoff(paths[:, j])
        candidates = np.flatnonzero(alive & (immediate > 0))
        exercise = candidates[immediate[candidates] > fit.predict(paths[candidates, j])]
        stopping[exercise] = j
        alive[exercise] = False
    times = np.asarray(policy.times)
    cash = policy.option.payoff(paths[np.arange(n_paths), stopping]) * np.exp(-policy.market.rate * times[stopping])
    terminal = policy.option.payoff(paths[:, -1]) * np.exp(-policy.market.rate * times[-1])
    controlled = cash - terminal + black_scholes(policy.market, policy.option)
    return PolicyEvaluation(estimate_samples(cash), estimate_samples(controlled), cash, stopping)
