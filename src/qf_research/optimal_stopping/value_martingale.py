"""Analytically centred value-function martingales for the funded model.

The dual construction follows Haugh and Kogan (2004), Section 3.2. Spatial
projection and deterministic compression affect tightness, not centring:
each chosen interpolant is integrated analytically under the model transition.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.special import ndtr

from .contracts import Option, positive_integer
from .funded_cash import FundedCashModel, FundedExercise, _schedule
from .funded_policy import FundedPaths, _digest, _validate_paths, realise_policy
from .funded_dual import DualEvaluation
from .simulation import estimate_samples


@dataclass(frozen=True)
class LinearValue:
    knots: tuple[float, ...]
    values: tuple[float, ...]
    tail_slope: float = 0.

    def __post_init__(self):
        x, y = np.asarray(self.knots), np.asarray(self.values)
        if (not isinstance(self.knots, tuple) or not isinstance(self.values, tuple)
                or x.ndim != 1 or len(x) < 2 or x.shape != y.shape or x[0] != 0.
                or np.any(np.diff(x) <= 0) or not np.all(np.isfinite(x))
                or not np.all(np.isfinite(y)) or not math.isfinite(self.tail_slope)):
            raise ValueError('Finite values on an immutable increasing grid starting at zero required')

    def __call__(self, states):
        states = np.asarray(states, dtype=float)
        if np.any(~np.isfinite(states)) or np.any(states < 0):
            raise ValueError('Finite non-negative states required')
        return np.interp(states, self.knots, self.values) + self.tail_slope*np.maximum(states-self.knots[-1], 0.)

    def expectation(self, states, sd, *, block_size=512):
        """Exact lognormal expectation of this interpolant, up to floating point.

        A linear tail plus put hinges spans any continuous linear interpolant.
        No conditional-mean interpolation, tail truncation or inner sampling
        is used. A zero state remains an exact atom.
        """
        states = np.asarray(states, dtype=float)
        self(states)  # Validate without silently flooring negative states.
        block_size = positive_integer(block_size, 'block_size')
        if not math.isfinite(sd) or sd < 0:
            raise ValueError('Finite non-negative volatility-time required')
        if sd == 0:
            return self(states)
        x, y = np.asarray(self.knots), np.asarray(self.values)
        slopes = np.diff(y)/np.diff(x)
        weights = np.diff(np.r_[slopes, self.tail_slope])
        active = weights != 0.
        strikes, weights = x[1:][active], weights[active]
        flat = states.reshape(-1)
        result = y[-1]+self.tail_slope*(flat-x[-1])
        locations = np.flatnonzero(flat > 0)
        for start in range(0, len(locations), block_size):
            where = locations[start:start+block_size]
            current = flat[where, None]
            z = (np.log(strikes)[None, :]-np.log(current))/sd
            puts = strikes*ndtr(z+sd/2)-current*ndtr(z-sd/2)
            result[where] += puts @ weights
        result[flat == 0] = y[0]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('Non-finite conditional expectation')
        return result.reshape(states.shape)


def compress(value, tolerance):
    """Endpoint-preserving linear simplification with a uniform error bound.

    Each accepted chord differs by at most tolerance at every original knot,
    hence everywhere on the intervening linear segments. The upper tail is
    unchanged. This bound concerns this compression only, not Snell error.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite non-negative compression tolerance required')
    x, y = np.asarray(value.knots), np.asarray(value.values)
    keep = {0, len(x)-1}
    pending = [(0, len(x)-1)]
    while pending:
        a, b = pending.pop()
        if b-a <= 1:
            continue
        errors = np.abs(y[a+1:b]-(y[a]+(y[b]-y[a])*(x[a+1:b]-x[a])/(x[b]-x[a])))
        index = a+1+int(np.argmax(errors))
        if errors[index-a-1] > tolerance:
            keep.add(index)
            pending.extend(((a, index), (index, b)))
    chosen = sorted(keep)
    return LinearValue(tuple(x[chosen]), tuple(y[chosen]), value.tail_slope)


def jump_value(value, cap):
    """Exact composition f((y-cap)^+), including the liquidation plateau."""
    if not math.isfinite(cap) or cap < 0:
        raise ValueError('Finite non-negative cash cap required')
    if cap == 0:
        return value
    return LinearValue((0., *(x+cap for x in value.knots)),
                       (value.values[0], *value.values), value.tail_slope)


@dataclass(frozen=True)
class ValueNode:
    time: float
    before: LinearValue | None
    after: LinearValue


@dataclass(frozen=True)
class ValueApproximation:
    model: FundedCashModel
    option: Option
    decisions: tuple[FundedExercise, ...]
    nodes: tuple[ValueNode, ...]
    core_steps: int
    wing_steps: int
    compression_tolerance: float
    initial_value: float

    @property
    def fingerprint(self):
        return _digest(asdict(self))


def build_values(model, option, decisions, *, core_steps=400, wing_steps=40, tolerance=.001):
    """Backward integration on a non-uniform state grid; no simulation fitting.

    The central grid spans half to one-and-a-half times the largest settlement
    strike; wings extend to zero and four times that strike. Every payoff kink
    is inserted. The stated constant put tail is part of the approximation.
    """
    if option.kind != 'put':
        raise ValueError('This value construction currently supports puts only')
    core_steps = positive_integer(core_steps, 'core_steps', 4)
    wing_steps = positive_integer(wing_steps, 'wing_steps', 2)
    rights, events = _schedule(model, option, decisions)
    thresholds = [option.strike*d.settlement_factor for d in rights]
    scale = max(thresholds)
    if scale <= 0:
        raise ValueError('Positive settlement strike required')
    grid = np.unique(np.r_[np.linspace(0., .5*scale, wing_steps+1),
        np.linspace(.5*scale, 1.5*scale, core_steps+1),
        np.linspace(1.5*scale, 4*scale, wing_steps+1), thresholds, model.initial_state])
    times = sorted({0., option.maturity, *events, *(d.time for d in rights)})
    at = {(d.time, d.phase): d for d in rights}
    nodes = []
    next_value = None
    later = option.maturity
    for time in reversed(times):
        values = np.zeros_like(grid) if next_value is None else next_value.expectation(grid, model.volatility*math.sqrt(later-time))
        if (time, 'after') in at:
            values = np.maximum(values, model.reward(option, at[time, 'after'], grid))
        after = compress(LinearValue(tuple(grid), tuple(values)), tolerance)
        before = jump_value(after, events[time].discounted_cash) if time in events else None
        if (time, 'before') in at:
            # A pre-event right is a genuine obstacle, unlike a mere hedge node.
            pre_grid = np.unique(np.r_[grid, before.knots])
            before = compress(LinearValue(tuple(pre_grid), tuple(np.maximum(
                before(pre_grid), model.reward(option, at[time, 'before'], pre_grid)))), tolerance)
        nodes.append(ValueNode(time, before, after))
        next_value = before if before is not None else after
        later = time
    nodes.reverse()
    return ValueApproximation(model, option, rights, tuple(nodes), core_steps, wing_steps,
                              tolerance, float(nodes[0].after(model.initial_state)))


def observation_rights(approximation):
    """Add cash-event states for integration, without granting exercise rights."""
    rights = {(d.time, d.phase): d for d in approximation.decisions}
    for node in approximation.nodes:
        if node.time == 0:
            continue
        rights.setdefault((node.time, 'after'), FundedExercise(node.time, 1.))
        if node.before is not None:
            rights.setdefault((node.time, 'before'), FundedExercise(node.time, 1., 'before'))
    return tuple(sorted(rights.values(), key=lambda d: (d.time, d.phase == 'after')))


@dataclass(frozen=True)
class ValueMartingale:
    """Martingale at genuine exercise rights, with propagation-only columns omitted."""

    paths: FundedPaths
    values: np.ndarray
    approximation_fingerprint: str


def value_martingale(approximation, observations):
    """Centre at diffusion endpoints, then retain genuine exercise rights only.

    Cash jumps have zero increments. Propagation-only observations contribute
    to the construction but grant no exercise rights in the returned sample.
    """
    _validate_paths(observations, approximation.option)
    if observations.model != approximation.model or observations.decisions != observation_rights(approximation):
        raise ValueError('Observation paths must match the value approximation')
    index = {(d.time, d.phase): j for j, d in enumerate(observations.decisions)}
    count = len(observations.states)
    current = np.zeros(count)
    previous_state = np.full(count, approximation.model.initial_state)
    previous_time = 0.
    at_time = {0.: current.copy()}
    for node in approximation.nodes:
        if node.time == 0:
            continue
        phase = 'before' if node.before is not None else 'after'
        value = node.before if node.before is not None else node.after
        state = observations.states[:, index[node.time, phase]]
        sd = approximation.model.volatility*math.sqrt(node.time-previous_time)
        current = current+value(state)-value.expectation(previous_state, sd)
        at_time[node.time] = current.copy()
        # The cash event is deterministic conditional on its pre-event state.
        previous_state = observations.states[:, index[node.time, 'after']]
        previous_time = node.time
    columns = [index[d.time, d.phase] for d in approximation.decisions]
    states = observations.states[:, columns].copy()
    states.setflags(write=False)
    paths = FundedPaths(observations.model, observations.maturity, approximation.decisions,
                        states, observations.detached_payments, observations.stream_id)
    values = np.column_stack([at_time[d.time] for d in approximation.decisions])
    values.setflags(write=False)
    return ValueMartingale(paths, values, approximation.fingerprint)


def evaluate_value_dual(policy, sample):
    """Evaluate a frozen exercise rule with an already constructed martingale."""
    realised = realise_policy(policy, sample.paths)
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError('Finite martingale values at every original right required')
    rewards = np.column_stack([policy.model.reward(policy.option, right, sample.paths.states[:, j])
                               for j, right in enumerate(policy.decisions)])
    upper = (rewards-martingale).max(axis=1)
    stopped = martingale[np.arange(len(rewards)), realised.stopping_indices]
    lower = realised.discounted_payoffs-stopped
    gap = upper-lower
    if np.any(gap < 0.):
        raise ArithmeticError('Pathwise dual inequality failed')
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for a in (lower, upper, gap):
        a.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
        estimate_samples(martingale[:, -1]), lo.lower, hi.upper, lower, upper, gap)
