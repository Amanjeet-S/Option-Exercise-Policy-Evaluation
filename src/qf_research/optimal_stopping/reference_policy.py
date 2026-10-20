"""Forward stopping rules induced by deterministic value approximations.

The continuation is the exact model expectation of the next retained linear
value function. The rule is evaluated on separate paths. Its reward is not the
initial value of the backward grid calculation, and neither quantity is an
error enclosure for the optimal value.
"""

from dataclasses import dataclass
import math

import numpy as np

from .funded_cash import _schedule
from .funded_policy import PolicyRealisation, _digest, _validate_paths
from .funded_dual import DualEvaluation
from .simulation import estimate_samples
from .value_martingale import ValueApproximation, ValueMartingale


@dataclass(frozen=True)
class ReferencePolicy:
    approximation: ValueApproximation

    def __post_init__(self):
        if not isinstance(self.approximation, ValueApproximation):
            raise TypeError("A deterministic ValueApproximation is required")
        rights, events = _schedule(self.model, self.option, self.decisions)
        times = tuple(node.time for node in self.approximation.nodes)
        expected = tuple(sorted({0., self.option.maturity, *events,
                                 *(right.time for right in rights)}))
        if rights != self.decisions or times != expected:
            raise ValueError("Approximation nodes must cover exactly its model calendar")
        if any((node.before is not None) != (node.time in events)
               for node in self.approximation.nodes):
            raise ValueError("Cash-event phases disagree with the approximation")

    @property
    def model(self):
        return self.approximation.model

    @property
    def option(self):
        return self.approximation.option

    @property
    def decisions(self):
        return self.approximation.decisions

    @property
    def fingerprint(self):
        return _digest({"construction": "analytical_next_node_continuation_v1",
                        "approximation": self.approximation.fingerprint})


def continuation_values(policy, index, states):
    """Continuation before the current exercise obstacle is applied.

    A before-event right continues through the deterministic cash jump and any
    genuine after-event right at that time. An after-event right continues to
    the next propagation node, which may itself be a cash event without any
    exercise right. Only the declared rights can stop a path.
    """
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("An integer exercise index is required")
    if not 0 <= index < len(policy.decisions)-1:
        raise ValueError("Continuation is defined only before the terminal right")
    states = np.asarray(states, dtype=float)
    if np.any(~np.isfinite(states)) or np.any(states < 0):
        raise ValueError("Finite non-negative states are required")
    right = policy.decisions[index]
    nodes = policy.approximation.nodes
    location = next(j for j, node in enumerate(nodes) if node.time == right.time)
    if right.phase == "before":
        event = next(event for event in policy.model.dividends
                     if event.ex_time == right.time)
        after, _ = policy.model.jump(states, event)
        return nodes[location].after(after)
    following = nodes[location+1]
    value = following.before if following.before is not None else following.after
    sd = policy.model.volatility*math.sqrt(following.time-right.time)
    return value.expectation(states, sd)


def realise_reference_policy(policy, paths):
    """Apply the frozen grid rule forwards and retain its actual rewards.

    Strict exercise inequalities match the fitted-policy convention. Ties
    continue, and the final genuine right is compulsory. No numerical value is
    substituted for a realised payoff.
    """
    _validate_paths(paths, policy.option)
    if paths.model != policy.model or paths.decisions != policy.decisions:
        raise ValueError("Evaluation paths must match the reference policy's original rights")
    n, count = paths.states.shape
    stopping = np.full(n, count-1, dtype=int)
    alive = np.ones(n, dtype=bool)
    diagnostics = []
    for index, right in enumerate(policy.decisions[:-1]):
        immediate = policy.model.reward(policy.option, right, paths.states[:, index])
        candidates = np.flatnonzero(alive & (immediate > 0.))
        chosen = np.empty(0, dtype=int)
        if len(candidates):
            continuation = continuation_values(policy, index, paths.states[candidates, index])
            chosen = candidates[immediate[candidates] > np.maximum(continuation, 0.)]
            stopping[chosen] = index
            alive[chosen] = False
        diagnostics.append(dict(decision_index=index, time=right.time, phase=right.phase,
                                candidates=len(candidates), exercises=len(chosen)))
    cash = np.empty(n)
    for index, right in enumerate(policy.decisions):
        selected = stopping == index
        cash[selected] = policy.model.reward(policy.option, right, paths.states[selected, index])
    control = paths.total_gain-policy.model.initial_state
    for array in (cash, stopping, control):
        array.setflags(write=False)
    return PolicyRealisation(cash, stopping, control, tuple(diagnostics))


def evaluate_reference_policy(policy, sample):
    """Evaluate the stopping rule and a separately supplied centred martingale.

    The martingale can use a different approximation from the stopping rule.
    A tighter assessment martingale therefore does not silently change the
    policy being assessed. Sampling intervals are approximate and pointwise.
    """
    if not isinstance(sample, ValueMartingale):
        raise TypeError("A ValueMartingale evaluation sample is required")
    realised = realise_reference_policy(policy, sample.paths)
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError("Finite martingale values at every original right are required")
    rewards = np.column_stack([
        policy.model.reward(policy.option, right, sample.paths.states[:, index])
        for index, right in enumerate(policy.decisions)])
    upper = (rewards-martingale).max(axis=1)
    stopped = martingale[np.arange(len(rewards)), realised.stopping_indices]
    lower = realised.discounted_payoffs-stopped
    gap = upper-lower
    if np.any(gap < 0.):
        raise ArithmeticError("Pathwise dual inequality failed")
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for array in (lower, upper, gap):
        array.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
                          estimate_samples(martingale[:, -1]), lo.lower, hi.upper,
                          lower, upper, gap)
