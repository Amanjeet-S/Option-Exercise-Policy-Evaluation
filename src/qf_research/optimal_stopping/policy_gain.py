"""Paired value of a frozen stopping rule relative to terminal exercise.

The terminal payoff is evaluated on the same path. Its expectation is never
replaced by an estimated or numerically integrated European price. A separate
pilot fits a zero-mean total-gain control to the payoff difference itself.
"""

from dataclasses import dataclass
import math

import numpy as np

from .funded_policy import realise_policy
from .simulation import Estimate, estimate_samples


@dataclass(frozen=True)
class ExerciseGainControl:
    coefficient: float
    policy_fingerprint: str
    pilot_stream_id: str
    pilot_paths: int


def _payoffs(policy, paths):
    realised = realise_policy(policy, paths)
    terminal = policy.model.reward(policy.option, policy.decisions[-1], paths.states[:, -1])
    return realised, realised.discounted_payoffs-terminal


def calibrate_gain_control(policy, pilot_paths):
    """Regress exercise gain on total gain using an independent pilot."""
    realised, difference = _payoffs(policy, pilot_paths)
    x = realised.terminal_gain_control
    centred = x-x.mean()
    denominator = float(centred @ centred)
    beta = float(centred @ (difference-difference.mean())/denominator) if denominator > 0 else 0.
    if not math.isfinite(beta):
        raise FloatingPointError("Non-finite exercise-gain control coefficient")
    return ExerciseGainControl(beta, policy.fingerprint, pilot_paths.stream_id, len(x))


@dataclass(frozen=True)
class ExerciseGainEvaluation:
    raw_pv: Estimate
    controlled_pv: Estimate
    raw_quote: Estimate
    controlled_quote: Estimate
    control_mean: Estimate
    raw_gains: np.ndarray
    controlled_gains: np.ndarray
    stopping_indices: np.ndarray
    diagnostics: tuple[dict, ...]


def evaluate_gain(policy, paths, *, control=None):
    """Estimate E[H_tau-H_T], conditional on the frozen policy and pilot.

    Approximate pointwise intervals concern sampling error alone. A negative
    estimate or a policy worse than terminal exercise is retained unchanged.
    """
    if control is not None:
        if not isinstance(control, ExerciseGainControl):
            raise TypeError("Use a control fitted to exercise gains")
        if control.policy_fingerprint != policy.fingerprint:
            raise ValueError("Gain control belongs to another frozen policy")
        if control.pilot_stream_id == paths.stream_id:
            raise ValueError("Gain-control pilot cannot be reused for evaluation")
        if not math.isfinite(control.coefficient):
            raise ValueError("Gain-control coefficient must be finite")
    realised, raw = _payoffs(policy, paths)
    beta = control.coefficient if control is not None else 0.
    adjusted = raw-beta*realised.terminal_gain_control
    raw.setflags(write=False)
    adjusted.setflags(write=False)
    factor = policy.model.premium_factor
    return ExerciseGainEvaluation(
        estimate_samples(raw), estimate_samples(adjusted),
        estimate_samples(raw/factor), estimate_samples(adjusted/factor),
        estimate_samples(realised.terminal_gain_control), raw, adjusted,
        realised.stopping_indices, realised.diagnostics,
    )
