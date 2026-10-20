"""Defer the compulsory terminal decision of an otherwise frozen put policy.

Conditional integration isolates the expected benefit of waiting. A common
martingale on the original clock yields comparable loss bounds before and
after the change. The derivation and its restrictions are in paper/main.tex.
"""

from dataclasses import dataclass
import numpy as np

from .expiry_extension import check_extension, tail_put, restrict_paths
from .funded_policy import realise_policy
from .funded_dual import DualEvaluation
from .simulation import estimate_samples


@dataclass(frozen=True)
class TerminalRepair:
    original: DualEvaluation
    repaired: DualEvaluation
    direct_repaired: DualEvaluation
    integrated_gain: np.ndarray
    realised_gain: np.ndarray
    survives: np.ndarray
    stopping_indices: np.ndarray
    maximum_roundoff_correction: float


def _dual(lower, upper, stopped, terminal):
    gap = upper-lower
    if np.any(gap < 0):
        raise ArithmeticError('Negative pathwise dual gap')
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for values in (lower, upper, gap):
        values.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
        estimate_samples(terminal), lo.lower, hi.upper, lower, upper, gap)


def evaluate_terminal_repair(policy, sample, extension):
    """Compare the original policy with waiting at its compulsory old endpoint.

    The earlier policy is realised using only its original exercise columns.
    The sample contains genuine expanded exercise rights; propagation-only
    observations have already been removed by the martingale constructor.
    The primary gain conditions out the added diffusion. The separately
    retained direct payoff uses the simulated state at the new endpoint.
    """
    if policy.option.kind != 'put' or sample.paths.model != policy.model:
        raise ValueError('An unchanged put model is required')
    check_extension(policy.model, policy.option, extension)
    terminal = policy.decisions[-1]
    last = sample.paths.decisions[-1]
    if (terminal.phase != 'after' or last.phase != 'after'
            or last.time != policy.option.maturity+extension
            or sample.paths.maturity != last.time
            or last.settlement_factor != terminal.settlement_factor):
        raise ValueError('A matching extended terminal right is required')
    original_rights = tuple(right for right in sample.paths.decisions
                            if right.time <= policy.option.maturity)
    if original_rights != policy.decisions:
        raise ValueError('Unchanged original exercise rights are required through the old endpoint')
    if any(right.settlement_factor != terminal.settlement_factor
           for right in sample.paths.decisions if right.time > policy.option.maturity):
        raise ValueError('Added exercise rights must retain the terminal settlement factor')
    paths = restrict_paths(sample.paths, policy.option, policy.decisions)
    realised = realise_policy(policy, paths)
    lookup = {right: j for j, right in enumerate(sample.paths.decisions)}
    columns = [lookup[right] for right in policy.decisions]
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError('Finite martingale values on the expanded exercise clock required')
    original_martingale = martingale[:, columns]
    rewards = np.column_stack([policy.model.reward(policy.option, right, paths.states[:, j])
                               for j, right in enumerate(policy.decisions)])
    tail = tail_put(paths.states[:, -1], policy.option.strike*terminal.settlement_factor,
                    policy.model.volatility, extension)
    uplift = tail-rewards[:, -1]
    tolerance = 64*np.finfo(float).eps*max(1., policy.option.strike*terminal.settlement_factor)
    correction = float(max(0., -uplift.min()))
    if correction > tolerance:
        raise ArithmeticError('Terminal uplift violates convexity beyond roundoff')
    # Only negative roundoff is removed; the maximum adjustment is reported.
    uplift = np.maximum(uplift, 0.)
    survives = realised.stopping_indices == len(policy.decisions)-1
    gain = np.where(survives, uplift, 0.)
    stopped = original_martingale[np.arange(len(survives)), realised.stopping_indices]
    old_lower = realised.discounted_payoffs-stopped
    rewards[:, -1] += uplift
    adjusted = rewards-original_martingale
    repaired_lower = adjusted[np.arange(len(survives)), realised.stopping_indices]
    upper = adjusted.max(axis=1)
    original = _dual(old_lower, upper.copy(), stopped, original_martingale[:, -1])
    repaired = _dual(repaired_lower, upper.copy(), stopped, original_martingale[:, -1])

    later_reward = policy.model.reward(policy.option, last, sample.paths.states[:, -1])
    direct_payoff = np.where(survives, later_reward, realised.discounted_payoffs)
    direct_stop = np.where(survives, martingale[:, -1], stopped)
    expanded_rewards = np.column_stack([policy.model.reward(policy.option, right, sample.paths.states[:, j])
                                        for j, right in enumerate(sample.paths.decisions)])
    direct = _dual(direct_payoff-direct_stop, (expanded_rewards-martingale).max(axis=1),
                   direct_stop, martingale[:, -1])
    realised_gain = direct_payoff-realised.discounted_payoffs
    for values in (gain, realised_gain, survives):
        values.setflags(write=False)
    return TerminalRepair(original, repaired, direct, gain, realised_gain, survives,
                           realised.stopping_indices, correction)
