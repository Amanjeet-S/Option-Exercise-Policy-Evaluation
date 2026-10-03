"""Paired comparisons for a crossed training/evaluation experiment.

The identities apply established martingale duality. Inference is conditional
on the financial model; Student intervals are approximate. No numerical price
is treated as the exact optimum. See reports/training_evaluation.md.
"""

from dataclasses import asdict
import math

import numpy as np

from .simulation import estimate_samples


def paired_difference(right, left, *, confidence=.95):
    """Retain covariance as well as the estimate from pathwise differences."""
    right, left = np.asarray(right, dtype=float), np.asarray(left, dtype=float)
    if (right.ndim != 1 or right.shape != left.shape or len(right) < 2
            or not np.all(np.isfinite(right)) or not np.all(np.isfinite(left))):
        raise ValueError('Require matching finite one-dimensional paired samples')
    difference = right-left
    covariance = float(np.cov(right, left, ddof=1)[0, 1])
    return dict(estimate=asdict(estimate_samples(difference, confidence)),
                right_mean=float(right.mean()), left_mean=float(left.mean()),
                right_variance=float(right.var(ddof=1)),
                left_variance=float(left.var(ddof=1)), covariance=covariance)


def policy_gain(low_training, high_training, factor, *, confidence=.95):
    """A common upper makes gap reduction an estimate of policy reward gain."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Require a positive premium discount factor')
    if not np.array_equal(low_training.upper_payoffs, high_training.upper_payoffs):
        raise ValueError('Policy-gain comparison requires exactly the same upper payoff')
    gain = high_training.adjusted_policy_payoffs-low_training.adjusted_policy_payoffs
    reduction = low_training.gap_payoffs-high_training.gap_payoffs
    if not np.allclose(gain, reduction, rtol=1e-11, atol=1e-11):
        raise ArithmeticError('Common-upper cancellation failed')
    return paired_difference(high_training.adjusted_policy_payoffs/factor,
                             low_training.adjusted_policy_payoffs/factor,
                             confidence=confidence)


def replicate_summary(estimates, *, confidence=.95):
    """Inference over independent repetitions of training and evaluation.

    The moment estimate for training variance may be negative. It is retained
    without truncation and is not a variance bound or a test of zero variance.
    Dependence across fitted market cases is not removed by this calculation.
    """
    if len(estimates) < 2:
        raise ValueError('At least two independent repetitions required')
    for e in estimates:
        if (not math.isfinite(e['value']) or not math.isfinite(e['standard_error'])
                or e['standard_error'] < 0 or type(e['observations']) is not int
                or e['observations'] < 2):
            raise ValueError('Invalid repetition estimate')
    means = np.array([e['value'] for e in estimates])
    variances = np.array([e['standard_error']**2 for e in estimates])
    observed = float(means.var(ddof=1))
    evaluation = float(variances.mean())
    return dict(estimate=asdict(estimate_samples(means, confidence)),
                repetition_estimate_variance=observed,
                mean_conditional_evaluation_variance=evaluation,
                training_variance_moment_estimate=observed-evaluation,
                evaluation_variance_of_grand_mean=evaluation/len(means),
                approximate_intervals=True)
