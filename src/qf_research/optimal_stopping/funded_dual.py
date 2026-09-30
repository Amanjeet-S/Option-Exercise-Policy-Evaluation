"""Finite-calendar martingale dual bounds from predictable, bounded gain hedges.

The expectation inequality follows the published optimal-stopping duality
literature. This bounded slope construction is a declared approximation, not
the exact Doob martingale or a replication of a paper's numerical algorithm.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.polynomial import polyder, polyval

from .funded_policy import _validate_paths, realise_policy
from .simulation import Estimate, estimate_samples


def gain_at_right(paths, index):
    """Discounted state plus cash detached by this decision's exact phase."""
    right = paths.decisions[index]
    gain = paths.states[:, index].copy()
    for j, event in enumerate(e for e in paths.model.dividends if e.ex_time <= paths.maturity):
        if event.ex_time < right.time or (event.ex_time == right.time and right.phase == "after"):
            gain += paths.detached_payments[:, j]
    return gain


def hedge_weights(policy, index, states, fallback):
    """Bounded derivative of max(immediate, fitted continuation, zero).

    Current states and frozen fits only. Clipping ensures integrability and is
    a hedge choice, not an assertion that a fitted polynomial is a valid value
    function. A missing fit uses the fixed pilot coefficient.
    """
    sign = 1. if policy.option.kind == "call" else -1.
    lower, upper = (0., 1.) if sign > 0 else (-1., 0.)
    rule = policy.rules[index]
    if not rule.coefficients:
        return np.full(np.asarray(states).shape, np.clip(fallback, lower, upper))
    raw = rule.raw_prediction(states)
    derivative = polyval((states-rule.centre)/rule.scale, polyder(rule.coefficients))/rule.scale
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Non-finite continuation derivative")
    weights = np.where(raw > 0., derivative, 0.)
    immediate = policy.model.reward(policy.option, policy.decisions[index], states)
    exercise = (immediate > 0.) & (immediate > np.maximum(raw, 0.))
    weights = np.where(exercise, sign, weights)
    return np.clip(weights, lower, upper)


def martingale_values(policy, paths, initial_delta, hedge="policy_slope"):
    """M_j = sum predictable h_(i-1) * (G_i-G_(i-1)), starting at zero.

    No conditional mean is approximated or subtracted from the same paths.
    The exact model gain is the martingale integrator. This routine retains
    values at the stated exercise rights, including same-time event phases.
    """
    _validate_paths(paths, policy.option)
    if paths.model != policy.model or paths.decisions != policy.decisions:
        raise ValueError("Martingale paths must match the frozen policy")
    if paths.stream_id == policy.training_stream_id:
        raise ValueError("Training stream cannot evaluate a frozen martingale")
    if not math.isfinite(initial_delta) or isinstance(initial_delta, bool):
        raise ValueError("Initial hedge must be finite")
    if hedge not in ("zero", "constant", "policy_slope"):
        raise ValueError("Unknown predictable hedge")
    limits = (0., 1.) if policy.option.kind == "call" else (-1., 0.)
    initial = float(np.clip(initial_delta, *limits))
    weight = 0. if hedge == "zero" else initial
    previous_gain = paths.model.initial_state
    current = np.zeros(len(paths.states))
    values = np.empty_like(paths.states)
    for j in range(len(policy.decisions)):
        gain = gain_at_right(paths, j)
        current = current+weight*(gain-previous_gain)
        values[:, j] = current
        # Choose the NEXT interval's hedge only after the current gain update.
        if hedge == "policy_slope" and j < len(policy.decisions)-1:
            weight = hedge_weights(policy, j, paths.states[:, j], initial)
        previous_gain = gain
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class DualEvaluation:
    policy_value: Estimate
    upper_value: Estimate
    gap_bound: Estimate
    stopped_martingale: Estimate
    terminal_martingale: Estimate
    pricing_lower: float
    pricing_upper: float
    adjusted_policy_payoffs: np.ndarray
    upper_payoffs: np.ndarray
    gap_payoffs: np.ndarray


def evaluate_dual(policy, paths, control, *, hedge="policy_slope"):
    """Expectation bounds with approximate, conditional Monte Carlo intervals.

    U=max_j(H_j-M_j); L=H_tau-M_tau. Pathwise U-L>=0, and
    E[L]=V_policy <= V_optimal <= E[U]. E[U-L] bounds policy loss.
    The price interval combines two marginal 97.5% one-sided limits by a
    union bound (approximate 95% coverage); it does not require independence.
    """
    if control.policy_fingerprint != policy.fingerprint:
        raise ValueError("Control must belong to this frozen policy")
    if control.pilot_stream_id == paths.stream_id:
        raise ValueError("Control pilot stream cannot be used for dual evaluation")
    realised = realise_policy(policy, paths)
    martingale = martingale_values(policy, paths, control.coefficient, hedge)
    adjusted_rewards = np.column_stack([
        policy.model.reward(policy.option, right, paths.states[:, j])-martingale[:, j]
        for j, right in enumerate(policy.decisions)])
    upper = adjusted_rewards.max(axis=1)
    stopped_m = martingale[np.arange(len(paths.states)), realised.stopping_indices]
    lower = adjusted_rewards[np.arange(len(paths.states)), realised.stopping_indices]
    gap = upper-lower
    if np.any(gap < 0):
        raise ArithmeticError("Dual pathwise inequality failed")
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for a in (lower, upper, gap):
        a.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped_m),
                           estimate_samples(martingale[:, -1]), lo.lower, hi.upper,
                           lower, upper, gap)
