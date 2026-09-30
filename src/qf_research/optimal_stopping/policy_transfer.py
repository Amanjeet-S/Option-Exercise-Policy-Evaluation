"""Frozen quote-space exercise rules evaluated under alternative model inputs."""

import math

import numpy as np
from scipy.stats import t as student_t

from .funded_policy import PolicyRealisation, _validate_paths
from .funded_dual import DualEvaluation
from .simulation import estimate_samples


def realise_transferred_policy(source, paths):
    """Keep the original decision predicate, but pay the scenario reward.

    The scenario state y at right j implies quote q=y/f_scenario,j. Feed the
    original rule q*f_source,j, preserving its decisions for any given quote
    history. Replacing only the model would instead change its exercise rule.
    """
    _validate_paths(paths, source.option)
    if [(d.time, d.phase) for d in paths.decisions] != [(d.time, d.phase) for d in source.decisions]:
        raise ValueError('Policy transfer requires identical decision clocks and phases')
    if paths.stream_id == source.training_stream_id:
        raise ValueError('Training stream cannot evaluate a transferred policy')
    if len(source.rules) != len(source.decisions) or source.rules[-1].status != 'terminal':
        raise ValueError('Malformed original stopping rule')
    count, dates = paths.states.shape
    stopping = np.full(count, dates-1, dtype=int)
    alive = np.ones(count, dtype=bool)
    diagnostics = []
    for j, (old, new, rule) in enumerate(zip(source.decisions[:-1], paths.decisions[:-1], source.rules[:-1])):
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            original_state = paths.states[:, j]*(old.settlement_factor/new.settlement_factor)
        immediate = source.model.reward(source.option, old, original_state)
        candidates = np.flatnonzero(alive & (immediate > 0))
        chosen = np.empty(0, dtype=int)
        outside = negative = 0
        if rule.coefficients and len(candidates):
            state = original_state[candidates]
            continuation = rule.raw_prediction(state)
            outside = int(np.count_nonzero((state < rule.training_min) | (state > rule.training_max)))
            negative = int(np.count_nonzero(continuation < 0))
            chosen = candidates[immediate[candidates] > np.maximum(continuation, 0.)]
            stopping[chosen] = j
            alive[chosen] = False
        diagnostics.append(dict(decision_index=j, candidates=len(candidates), exercises=len(chosen),
            outside_original_training_support=outside, negative_predictions=negative,
            no_fit_continues=len(candidates) if not rule.coefficients else 0))
    rewards = np.empty(count)
    for j, right in enumerate(paths.decisions):
        selected = stopping == j
        rewards[selected] = paths.model.reward(source.option, right, paths.states[selected, j])
    gain = paths.total_gain-paths.model.initial_state
    for value in (rewards, stopping, gain):
        value.setflags(write=False)
    return PolicyRealisation(rewards, stopping, gain, tuple(diagnostics))


def evaluate_transferred_dual(source, sample):
    """Scenario-specific expectation bounds for an unchanged quoted-price rule."""
    realised = realise_transferred_policy(source, sample.paths)
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError('Finite martingale values at every scenario right required')
    rewards = np.column_stack([sample.paths.model.reward(source.option, right, sample.paths.states[:, j])
                               for j, right in enumerate(sample.paths.decisions)])
    upper = (rewards-martingale).max(axis=1)
    stopped = martingale[np.arange(len(rewards)), realised.stopping_indices]
    lower = realised.discounted_payoffs-stopped
    gap = upper-lower
    if np.any(gap < 0):
        raise ArithmeticError('Pathwise dual inequality failed')
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for value in (lower, upper, gap):
        value.setflags(write=False)
    result = DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
        estimate_samples(martingale[:, -1]), lo.lower, hi.upper, lower, upper, gap)
    return result, realised


def family_upper_limit(estimate, family_size, error_probability=.05):
    """Approximate one-sided Bonferroni limit; dependence needs no correction.

    Coverage concerns one finite, prespecified scenario family for one policy.
    It is not simultaneous across policies, models or an untested continuum.
    """
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        raise ValueError('A positive integer family size is required')
    if not math.isfinite(error_probability) or not 0 < error_probability < 1:
        raise ValueError('Family error probability must lie in (0, 1)')
    if estimate.observations < 2 or not math.isfinite(estimate.value) or not math.isfinite(estimate.standard_error) or estimate.standard_error < 0:
        raise ValueError('A finite sampling estimate with at least two observations is required')
    alpha = error_probability/family_size
    return float(estimate.value+student_t.ppf(1-alpha, estimate.observations-1)*estimate.standard_error)
