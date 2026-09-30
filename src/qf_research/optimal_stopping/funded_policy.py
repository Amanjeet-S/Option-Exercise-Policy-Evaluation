"""Longstaff-Schwartz policies for the funded model, with independent evaluation.

All targets and rewards are already in valuation-date currency. Paths use exact
lognormal transitions and explicit capped dividend jumps. A separately fitted
total-gain control has known mean without a numerical European-price input.
See funded_policy_study.md for statistical scope and methodological attribution.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
from numpy.polynomial.polynomial import polyvander, polyval

from .contracts import Option, positive_integer
from .funded_cash import FundedCashModel, FundedExercise, _schedule
from .simulation import Estimate, estimate_samples


def _digest(value):
    def encode_numpy(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported provenance value: {type(item).__name__}")
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     default=encode_numpy).encode()).hexdigest()


@dataclass(frozen=True)
class FundedPaths:
    model: FundedCashModel
    maturity: float
    decisions: tuple[FundedExercise, ...]
    states: np.ndarray
    detached_payments: np.ndarray
    stream_id: str

    @property
    def total_gain(self):
        return self.states[:, -1] + self.detached_payments.sum(axis=1)


def _validate_paths(paths, option):
    rights, events = _schedule(paths.model, option, paths.decisions)
    if paths.maturity != option.maturity or rights != paths.decisions:
        raise ValueError("Paths and sorted exercise convention must match the option")
    states, payments = paths.states, paths.detached_payments
    if (not isinstance(states, np.ndarray) or states.ndim != 2 or states.shape[0] < 2
            or states.shape[1] != len(rights) or not np.all(np.isfinite(states)) or np.any(states < 0)):
        raise ValueError("Require finite non-negative state paths with matching decision columns")
    if (not isinstance(payments, np.ndarray) or payments.shape != (len(states), len(events))
            or not np.all(np.isfinite(payments)) or np.any(payments < 0)):
        raise ValueError("Invalid detached-payment paths")
    for j, event in enumerate(sorted(events.values(), key=lambda e: e.ex_time)):
        if np.any(payments[:, j] > event.discounted_cash):
            raise ValueError("Detached payment exceeds its stated cap")
        before = next((k for k, d in enumerate(rights) if d.time == event.ex_time and d.phase == "before"), None)
        after = next((k for k, d in enumerate(rights) if d.time == event.ex_time and d.phase == "after"), None)
        if before is not None:
            expected_after, expected_payment = paths.model.jump(states[:, before], event)
            if not np.array_equal(payments[:, j], expected_payment):
                raise ValueError("Detached payment disagrees with observed pre-event state")
            if after is not None and not np.array_equal(states[:, after], expected_after):
                raise ValueError("Same-time states disagree with the dividend jump")
    if rights[0].time == 0 and not np.all(states[:, 0] == paths.model.initial_state):
        raise ValueError("Time-zero state differs from the model")
    if np.any((states[:, :-1] == 0) & (states[:, 1:] > 0)):
        raise ValueError("A liquidated state must remain zero")
    if not isinstance(paths.stream_id, str) or not paths.stream_id:
        raise ValueError("A random-stream provenance identifier is required")


def simulate_paths(model, option, decisions, n_paths, rng):
    """Independent paths; exact transitions at the union of events and decisions.

    Before/after columns at the same timestamp share one state transition and
    differ only by the jump. The stream identifier detects exact stream reuse,
    not arbitrary dependence or all possible overlapping generator sequences.
    """
    n_paths = positive_integer(n_paths, "n_paths", 2)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("An explicit NumPy Generator is required")
    rights, events = _schedule(model, option, decisions)
    stream_id = _digest(rng.bit_generator.state)
    states = np.empty((n_paths, len(rights)))
    payments = np.empty((n_paths, len(events)))
    event_index = {t: j for j, t in enumerate(sorted(events))}
    current = np.full(n_paths, model.initial_state)
    previous = 0.
    for t in sorted({d.time for d in rights} | set(events)):
        dt = t-previous
        if dt > 0 and model.volatility > 0:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                updated = current * np.exp(-.5*model.volatility**2*dt +
                                            model.volatility*math.sqrt(dt)*rng.standard_normal(n_paths))
            if np.any((current > 0) & (updated == 0)):
                raise FloatingPointError("Diffusion underflow must not create a liquidation atom")
            current = updated
        for j, right in enumerate(rights):
            if right.time == t and right.phase == "before":
                states[:, j] = current
        if t in events:
            current, paid = model.jump(current, events[t])
            payments[:, event_index[t]] = paid
        for j, right in enumerate(rights):
            if right.time == t and right.phase == "after":
                states[:, j] = current
        previous = t
    states.setflags(write=False)
    payments.setflags(write=False)
    result = FundedPaths(model, option.maturity, rights, states, payments, stream_id)
    _validate_paths(result, option)
    return result


def prefix_paths(paths, n_paths):
    """Nested training prefixes retain their original stream provenance."""
    n_paths = positive_integer(n_paths, "n_paths", 2)
    if n_paths > len(paths.states):
        raise ValueError("Requested prefix exceeds the retained sample")
    return FundedPaths(paths.model, paths.maturity, paths.decisions, paths.states[:n_paths],
                       paths.detached_payments[:n_paths], paths.stream_id)


@dataclass(frozen=True)
class ContinuationRule:
    centre: float
    scale: float
    coefficients: tuple[float, ...]
    observations: int
    rank: int
    condition_number: float | None
    training_min: float | None
    training_max: float | None
    status: str

    def raw_prediction(self, states):
        if not self.coefficients:
            raise ValueError("No regression is available for this decision")
        result = polyval((np.asarray(states)-self.centre)/self.scale, self.coefficients)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Non-finite continuation extrapolation")
        return result

    def predict(self, states):
        return np.maximum(self.raw_prediction(states), 0.)


def _fit(states, targets, degree, minimum_paths, condition_limit):
    n = len(states)
    if n < minimum_paths:
        return ContinuationRule(0., 1., (), n, 0, None, None, None, "continue_insufficient_itm")
    centre, scale = float(states.mean()), float(states.std())
    if scale < 1e-12*max(abs(centre), 1.):
        return ContinuationRule(centre, 1., (float(targets.mean()),), n, 1, 1.,
                                float(states.min()), float(states.max()), "constant_state")
    x = (states-centre)/scale
    effective = min(degree, n-2)
    while effective >= 0:
        design = polyvander(x, effective)
        coefficients, _, rank, singular = np.linalg.lstsq(design, targets, rcond=None)
        condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
        if rank == effective+1 and condition <= condition_limit:
            return ContinuationRule(centre, scale, tuple(float(c) for c in coefficients), n,
                                    int(rank), condition, float(states.min()), float(states.max()),
                                    "full_degree" if effective == degree else "reduced_degree")
        effective -= 1
    raise ArithmeticError("Even the intercept regression failed conditioning")


@dataclass(frozen=True)
class FundedPolicy:
    model: FundedCashModel
    option: Option
    decisions: tuple[FundedExercise, ...]
    rules: tuple[ContinuationRule, ...]
    requested_degree: int
    training_paths: int
    minimum_itm_paths: int
    condition_limit: float
    training_stream_id: str

    @property
    def fingerprint(self):
        return _digest(asdict(self))


def fit_policy(option, paths, degree, *, minimum_itm_paths=32, condition_limit=1e8):
    """Backward realised-cash regression. No in-sample price is returned.

    Only the current discounted ownership state is a predictor. Future realised
    rewards are responses for training, never evaluation-time features. Targets
    are already in time-zero currency and must not be discounted a second time.
    """
    _validate_paths(paths, option)
    degree = positive_integer(degree, "degree", 0)
    minimum_itm_paths = positive_integer(minimum_itm_paths, "minimum_itm_paths", 2)
    if isinstance(condition_limit, bool) or not math.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("Condition limit must be finite and at least one")
    model, rights, states = paths.model, paths.decisions, paths.states
    cash = model.reward(option, rights[-1], states[:, -1]).copy()
    rules = [ContinuationRule(0., 1., (), 0, 0, None, None, None, "terminal") for _ in rights]
    for j in range(len(rights)-2, -1, -1):
        immediate = model.reward(option, rights[j], states[:, j])
        itm = immediate > 0
        rule = _fit(states[itm, j], cash[itm], degree, minimum_itm_paths, condition_limit)
        rules[j] = rule
        if rule.coefficients:
            candidates = np.flatnonzero(itm)
            chosen = candidates[immediate[itm] > rule.predict(states[itm, j])]
            cash[chosen] = immediate[chosen]
    return FundedPolicy(model, option, rights, tuple(rules), degree, len(states),
                         minimum_itm_paths, float(condition_limit), paths.stream_id)


def train_policy(model, option, decisions, n_paths, degree, rng, **kwargs):
    return fit_policy(option, simulate_paths(model, option, decisions, n_paths, rng), degree, **kwargs)


@dataclass(frozen=True)
class PolicyRealisation:
    discounted_payoffs: np.ndarray
    stopping_indices: np.ndarray
    terminal_gain_control: np.ndarray
    diagnostics: tuple[dict, ...]


def realise_policy(policy, paths):
    """Apply a frozen stopping rule forwards, separately from statistical fitting."""
    _validate_paths(paths, policy.option)
    if paths.model != policy.model or paths.decisions != policy.decisions:
        raise ValueError("Evaluation model and rights differ from the trained policy")
    if paths.stream_id == policy.training_stream_id:
        raise ValueError("Training stream cannot be reused for independent policy evaluation")
    if len(policy.rules) != len(policy.decisions) or policy.rules[-1].status != "terminal":
        raise ValueError("Malformed policy rules or terminal decision")
    n, count = paths.states.shape
    stopping = np.full(n, count-1, dtype=int)
    alive = np.ones(n, dtype=bool)
    diagnostics = []
    for j in range(count-1):
        right, rule = policy.decisions[j], policy.rules[j]
        immediate = policy.model.reward(policy.option, right, paths.states[:, j])
        candidates = np.flatnonzero(alive & (immediate > 0))
        outside = negative = 0
        chosen = np.empty(0, dtype=int)
        if rule.coefficients and len(candidates):
            state = paths.states[candidates, j]
            raw = rule.raw_prediction(state)
            outside = int(np.count_nonzero((state < rule.training_min) | (state > rule.training_max)))
            negative = int(np.count_nonzero(raw < 0))
            chosen = candidates[immediate[candidates] > np.maximum(raw, 0.)]
            stopping[chosen] = j
            alive[chosen] = False
        diagnostics.append(dict(decision_index=j, time=right.time, phase=right.phase,
                                candidates=len(candidates), exercises=len(chosen),
                                outside_training_support=outside, negative_predictions=negative,
                                no_fit_continues=len(candidates) if not rule.coefficients else 0))
    cash = np.empty(n)
    for j, right in enumerate(policy.decisions):
        mask = stopping == j
        cash[mask] = policy.model.reward(policy.option, right, paths.states[mask, j])
    control = paths.total_gain-policy.model.initial_state
    for array in (cash, stopping, control):
        array.setflags(write=False)
    return PolicyRealisation(cash, stopping, control, tuple(diagnostics))


@dataclass(frozen=True)
class GainControl:
    coefficient: float
    policy_fingerprint: str
    pilot_stream_id: str
    pilot_paths: int


def calibrate_control(policy, pilot_paths):
    """Fit covariance/variance on a separate pilot, never on evaluation paths."""
    realised = realise_policy(policy, pilot_paths)
    x, y = realised.terminal_gain_control, realised.discounted_payoffs
    centred_x = x-x.mean()
    denominator = float(centred_x @ centred_x)
    coefficient = float(centred_x @ (y-y.mean())/denominator) if denominator > 0 else 0.
    if not math.isfinite(coefficient):
        raise FloatingPointError("Non-finite gain-control coefficient")
    return GainControl(coefficient, policy.fingerprint, pilot_paths.stream_id, len(x))


@dataclass(frozen=True)
class FundedPolicyEvaluation:
    raw_pv: Estimate
    controlled_pv: Estimate
    raw_quote: Estimate
    controlled_quote: Estimate
    control_mean: Estimate
    discounted_payoffs: np.ndarray
    controlled_payoffs: np.ndarray
    stopping_indices: np.ndarray
    diagnostics: tuple[dict, ...]


def evaluate_policy(policy, paths, *, control=None):
    if control is not None:
        if control.policy_fingerprint != policy.fingerprint:
            raise ValueError("Control coefficient belongs to another frozen policy")
        if paths.stream_id == control.pilot_stream_id:
            raise ValueError("Control pilot stream cannot be reused for evaluation")
    realised = realise_policy(policy, paths)
    beta = 0. if control is None else control.coefficient
    adjusted = realised.discounted_payoffs-beta*realised.terminal_gain_control
    adjusted.setflags(write=False)
    factor = policy.model.premium_factor
    return FundedPolicyEvaluation(estimate_samples(realised.discounted_payoffs), estimate_samples(adjusted),
                                   estimate_samples(realised.discounted_payoffs/factor), estimate_samples(adjusted/factor),
                                   estimate_samples(realised.terminal_gain_control), realised.discounted_payoffs,
                                   adjusted, realised.stopping_indices, realised.diagnostics)
