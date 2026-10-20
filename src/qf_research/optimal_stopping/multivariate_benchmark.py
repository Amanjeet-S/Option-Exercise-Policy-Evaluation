"""Public multiasset max-call model and independent Longstaff-Schwartz baseline.

The financial setting is Andersen and Broadie (2004), Management Science 50,
1222-1234, equation (19) and Table 2, doi:10.1287/mnsc.1040.0258. The sorted
full-state polynomial basis here differs from their 13-function basis. This
module implements an established baseline, not their primal-dual algorithm.
"""

from dataclasses import dataclass
import hashlib
from itertools import combinations_with_replacement
import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr

from .contracts import positive_integer


@dataclass(frozen=True)
class MaxCallModel:
    spots: tuple[float, ...]
    volatilities: tuple[float, ...]
    dividends: tuple[float, ...]
    correlation: tuple[tuple[float, ...], ...]
    rate: float
    strike: float
    exercise_times: tuple[float, ...]

    def __post_init__(self):
        dimension = len(self.spots)
        if dimension < 1 or len(self.volatilities) != dimension or len(self.dividends) != dimension:
            raise ValueError('Require matching nonempty asset inputs')
        values = np.asarray([self.spots, self.volatilities, self.dividends], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values[0] <= 0) or np.any(values[1] < 0):
            raise ValueError('Require positive spots, nonnegative volatilities and finite inputs')
        if not math.isfinite(self.rate) or not math.isfinite(self.strike) or self.strike <= 0:
            raise ValueError('Require a finite rate and positive strike')
        times = np.asarray(self.exercise_times, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
                or times[0] != 0. or np.any(np.diff(times) <= 0)):
            raise ValueError('Exercise times must increase strictly from zero')
        correlation = np.asarray(self.correlation, dtype=float)
        if (correlation.shape != (dimension, dimension) or not np.all(np.isfinite(correlation))
                or not np.allclose(correlation, correlation.T, atol=1e-12, rtol=0.)
                or not np.allclose(np.diag(correlation), 1., atol=1e-12, rtol=0.)
                or np.linalg.eigvalsh(correlation).min() < -1e-12):
            raise ValueError('Require a finite symmetric positive-semidefinite correlation matrix')
        for name in ('spots', 'volatilities', 'dividends', 'exercise_times'):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        object.__setattr__(self, 'correlation', tuple(tuple(float(x) for x in row) for row in correlation))

    @property
    def dimension(self):
        return len(self.spots)

    @property
    def maturity(self):
        return self.exercise_times[-1]

    def discounted_payoff(self, states, time_index):
        states = np.asarray(states, dtype=float)
        if (states.ndim != 2 or states.shape[1] != self.dimension
                or not np.all(np.isfinite(states)) or np.any(states <= 0)):
            raise ValueError('Require finite positive states for every asset')
        return np.exp(-self.rate*self.exercise_times[time_index])*np.maximum(
            states.max(axis=1)-self.strike, 0.)


def published_max_call(dimension=5, spot=100., *, correlation=0., volatility=.20):
    """Andersen-Broadie Table 2 model, allowing explicit sensitivity variants."""
    dimension = positive_integer(dimension, 'dimension')
    matrix = np.full((dimension, dimension), correlation, dtype=float)
    np.fill_diagonal(matrix, 1.)
    return MaxCallModel((float(spot),)*dimension, (float(volatility),)*dimension,
        (.10,)*dimension, tuple(tuple(x for x in row) for row in matrix), .05, 100.,
        tuple(i*3./9 for i in range(10)))


def correlation_factor(correlation):
    """An eigenvalue square root supports singular positive-semidefinite cases."""
    correlation = np.asarray(correlation, dtype=float)
    if (correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]
            or not len(correlation) or not np.all(np.isfinite(correlation))
            or not np.allclose(correlation, correlation.T, atol=1e-12, rtol=0.)
            or not np.allclose(np.diag(correlation), 1., atol=1e-12, rtol=0.)):
        raise ValueError('Require a finite symmetric correlation matrix with unit diagonal')
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    if eigenvalues.min() < -1e-12:
        raise ValueError('Correlation must be positive semidefinite')
    return eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))


@dataclass(frozen=True)
class MaxCallPaths:
    model: MaxCallModel
    states: np.ndarray
    stream_id: str

    def __post_init__(self):
        states = np.asarray(self.states)
        if (states.ndim != 3 or states.shape[1:] != (len(self.model.exercise_times), self.model.dimension)
                or len(states) < 1 or not np.all(np.isfinite(states)) or np.any(states <= 0)
                or not np.array_equal(states[:, 0], np.broadcast_to(self.model.spots, states[:, 0].shape))
                or not isinstance(self.stream_id, str) or not self.stream_id):
            raise ValueError('Invalid complete multiasset paths or stream identity')
        states = np.array(states, dtype=float, copy=True)
        states.setflags(write=False)
        object.__setattr__(self, 'states', states)


def simulate_max_call(model, count, rng):
    """Exact GBM transitions on every exercise right, with no time discretisation."""
    count = positive_integer(count, 'path count')
    if not isinstance(rng, np.random.Generator):
        raise TypeError('An explicit independent NumPy Generator is required')
    stream = hashlib.sha256(json.dumps(rng.bit_generator.state, sort_keys=True,
        default=lambda x: x.tolist()).encode()).hexdigest()
    factor = correlation_factor(model.correlation)
    volatility = np.asarray(model.volatilities)
    drift = model.rate-np.asarray(model.dividends)-.5*volatility*volatility
    states = np.empty((count, len(model.exercise_times), model.dimension))
    states[:, 0] = model.spots
    for index, duration in enumerate(np.diff(model.exercise_times), 1):
        noise = rng.standard_normal((count, model.dimension)) @ factor.T
        states[:, index] = states[:, index-1]*np.exp(
            drift*duration+noise*(volatility*math.sqrt(duration)))
    return MaxCallPaths(model, states, stream)


def monomial_indices(dimension, degree):
    dimension = positive_integer(dimension, 'dimension')
    degree = positive_integer(degree, 'degree', 0)
    if degree > 3:
        raise ValueError('This baseline supports polynomial degrees zero to three')
    return ((), *(powers for order in range(1, degree+1)
                  for powers in combinations_with_replacement(range(dimension), order)))


def sorted_polynomial_basis(states, strike, degree):
    """All monomials through the degree in sorted full-state coordinates.

    Sorting makes the basis invariant to asset permutation. Every asset enters;
    it is not a maximum-only approximation. The exchangeable published model
    permits this symmetry. General heterogeneous models remain feasible but
    may lose useful asset identity through this basis choice.
    """
    states = np.asarray(states, dtype=float)
    if (states.ndim != 2 or states.shape[1] < 1 or not np.all(np.isfinite(states))
            or np.any(states <= 0) or not math.isfinite(strike) or strike <= 0):
        raise ValueError('Require a positive strike and finite positive asset states')
    coordinates = np.sort(states, axis=1)[:, ::-1]/strike-1.
    indices = monomial_indices(states.shape[1], degree)
    result = np.ones((len(states), len(indices)))
    for column, powers in enumerate(indices[1:], 1):
        result[:, column] = np.prod(coordinates[:, powers], axis=1)
    return result


@dataclass(frozen=True)
class MaxCallRule:
    coefficients: tuple[float, ...]
    training_itm_paths: int
    rank: int
    condition_number: float | None


@dataclass(frozen=True)
class MaxCallPolicy:
    model: MaxCallModel
    degree: int
    rules: tuple[MaxCallRule, ...]
    training_stream_id: str
    training_paths: int

    def __post_init__(self):
        expected = len(monomial_indices(self.model.dimension, self.degree))
        if (len(self.rules) != len(self.model.exercise_times) or self.training_paths < 1
                or not self.training_stream_id):
            raise ValueError('Require a complete frozen policy and training identity')
        if self.rules[-1].coefficients:
            raise ValueError('The final right must exercise every surviving path')
        for rule in self.rules:
            if rule.coefficients and (len(rule.coefficients) != expected
                    or not np.all(np.isfinite(rule.coefficients))):
                raise ValueError('Require finite coefficients for the declared full-state basis')


def fit_max_call_policy(training, degree=2, *, minimum_itm_paths=64):
    """Longstaff-Schwartz continuation regression using training outcomes only.

    Future cash targets are discounted to time zero. Insufficient samples or a
    deficient nonconstant design continue. Time zero has a constant state and
    uses its training mean continuation instead of an unidentified regression.
    Strict reward-greater-than-continuation selects exercise; ties continue.
    """
    minimum_itm_paths = positive_integer(minimum_itm_paths, 'minimum ITM paths', 2)
    model, states = training.model, training.states
    basis_count = len(monomial_indices(model.dimension, degree))
    rules = [MaxCallRule((), 0, 0, None) for _ in model.exercise_times]
    cash = model.discounted_payoff(states[:, -1], len(rules)-1)
    for index in range(len(rules)-2, -1, -1):
        immediate = model.discounted_payoff(states[:, index], index)
        itm = immediate > 0.
        count = int(itm.sum())
        if index == 0 and count:
            coefficients = (float(cash.mean()),)+(0.,)*(basis_count-1)
            rules[index] = MaxCallRule(coefficients, count, 1, None)
            if immediate[0] > coefficients[0]:
                cash[:] = immediate[0]
        elif count >= max(minimum_itm_paths, basis_count+1):
            design = sorted_polynomial_basis(states[itm, index], model.strike, degree)
            coefficients, _, rank, singular = np.linalg.lstsq(design, cash[itm], rcond=None)
            condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
            if rank == basis_count:
                frozen = tuple(float(x) for x in coefficients)
                rules[index] = MaxCallRule(frozen, count, int(rank), condition)
                candidates = np.flatnonzero(itm)
                exercise = immediate[itm] > np.maximum(design @ coefficients, 0.)
                cash[candidates[exercise]] = immediate[candidates[exercise]]
            else:
                rules[index] = MaxCallRule((), count, int(rank), condition)
        else:
            rules[index] = MaxCallRule((), count, 0, None)
    return MaxCallPolicy(model, degree, tuple(rules), training.stream_id, len(states))


@dataclass(frozen=True)
class MaxCallEvaluation:
    discounted_rewards: np.ndarray
    terminal_rewards: np.ndarray
    matched_gains: np.ndarray
    stopping_indices: np.ndarray


def evaluate_max_call(policy, paths):
    """Forward exercise uses only the frozen rule and current full asset state."""
    if paths.model != policy.model:
        raise ValueError('Policy and evaluation model must agree')
    if paths.stream_id == policy.training_stream_id:
        raise ValueError('Evaluation must not reuse the training stream')
    model, states = paths.model, paths.states
    stopping = np.full(len(states), len(model.exercise_times)-1, dtype=int)
    rewards = model.discounted_payoff(states[:, -1], len(model.exercise_times)-1)
    terminal = rewards.copy()
    alive = np.ones(len(states), dtype=bool)
    for index, rule in enumerate(policy.rules[:-1]):
        if not rule.coefficients or not alive.any():
            continue
        locations = np.flatnonzero(alive)
        current = states[locations, index]
        immediate = model.discounted_payoff(current, index)
        continuation = np.maximum(sorted_polynomial_basis(current, model.strike, policy.degree)
                                  @ rule.coefficients, 0.)
        exercise = immediate > continuation
        chosen = locations[exercise]
        rewards[chosen] = immediate[exercise]
        stopping[chosen] = index
        alive[chosen] = False
    gains = rewards-terminal
    for value in (rewards, terminal, gains, stopping):
        value.setflags(write=False)
    return MaxCallEvaluation(rewards, terminal, gains, stopping)


def independent_european_max_call(model, *, epsabs=1e-9, epsrel=1e-10):
    """One-dimensional integral of the independent maximum's survival function.

    E[(max S_T-K)^+] = integral_K^infinity (1-product_i F_i(x)) dx.
    SciPy quadrature error is a diagnostic, not a certified price enclosure.
    Correlated marginals do not permit this product CDF and are rejected.
    """
    if not np.array_equal(np.asarray(model.correlation), np.eye(model.dimension)):
        raise ValueError('Product-CDF integration requires independent assets')
    maturity = model.maturity
    volatility = np.asarray(model.volatilities)
    means = np.log(model.spots)+(model.rate-np.asarray(model.dividends)-.5*volatility**2)*maturity
    deviations = volatility*math.sqrt(maturity)
    stochastic = deviations > 0
    deterministic = np.exp(means[~stochastic])
    if not stochastic.any():
        return dict(value=float(math.exp(-model.rate*maturity)*max(deterministic.max()-model.strike, 0.)),
                    quadrature_error_estimate=0., certified_error_enclosure=False)

    def survival(x):
        if len(deterministic) and x < deterministic.max():
            return 1.
        z = (math.log(x)-means[stochastic])/deviations[stochastic]
        return float(-np.expm1(log_ndtr(z).sum()))

    breaks = [model.strike, *(float(x) for x in sorted(set(deterministic)) if x > model.strike), math.inf]
    integrals = [quad(survival, left, right, epsabs=epsabs, epsrel=epsrel, limit=250)
                 for left, right in zip(breaks[:-1], breaks[1:])]
    factor = math.exp(-model.rate*maturity)
    return dict(value=float(factor*sum(value for value, _ in integrals)),
                quadrature_error_estimate=float(factor*sum(error for _, error in integrals)),
                certified_error_enclosure=False)
