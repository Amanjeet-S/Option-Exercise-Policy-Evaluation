"""Unchanged quote-space policies embedded in a longer scenario calendar.

Financial transfer and calendar embedding are distinct operations. The former
preserves the observed-price stopping predicate; the latter adds opportunities
to the upper problem while the original rule still stops by its old expiry.
"""

from dataclasses import replace

import numpy as np

from .expiry_extension import restrict_paths
from .funded_policy import _validate_paths
from .policy_transfer import realise_transferred_policy
from .funded_dual import DualEvaluation
from .simulation import estimate_samples


def evaluate_joint_policy(source, sample, scenario_rights):
    """Scenario rewards at the frozen rule's original, quote-space stopping time."""
    if not scenario_rights or scenario_rights[-1].time != source.option.maturity:
        raise ValueError('Original scenario expiry is required')
    if [(d.time, d.phase) for d in scenario_rights] != [(d.time, d.phase) for d in source.decisions]:
        raise ValueError('Financial transfer must preserve the original decision clock')
    expanded_option = replace(source.option, maturity=sample.paths.maturity)
    _validate_paths(sample.paths, expanded_option)
    original_paths = restrict_paths(sample.paths, source.option, scenario_rights)
    realised = realise_transferred_policy(source, original_paths)
    lookup = {d: i for i, d in enumerate(sample.paths.decisions)}
    embedded = np.array([lookup[d] for d in scenario_rights])[realised.stopping_indices]
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError('Finite martingale values required at every extended right')
    rewards = np.column_stack([sample.paths.model.reward(expanded_option, d, sample.paths.states[:, j])
                               for j, d in enumerate(sample.paths.decisions)])
    upper = (rewards-martingale).max(axis=1)
    stopped = martingale[np.arange(len(embedded)), embedded]
    lower = realised.discounted_payoffs-stopped
    gap = upper-lower
    if np.any(gap < 0):
        raise ArithmeticError('Negative pathwise dual gap')
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for array in (upper, lower, gap):
        array.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
        estimate_samples(martingale[:, -1]), lo.lower, hi.upper, lower, upper, gap), realised
