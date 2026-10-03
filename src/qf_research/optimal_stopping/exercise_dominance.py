"""Sufficient exercise-dominance conditions for the funded cash-dividend model."""

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import math

from .funded_cash import _schedule


@dataclass(frozen=True)
class CoarseningCheck:
    sufficient_condition_satisfied: bool
    fine_decision_count: int
    coarse_decision_count: int
    added_decisions: int
    factor_mismatches: int
    intervals_crossing_cash_events: int


def coarsening_dominance(model, option, coarse, fine):
    """Check that waiting to the next coarse right dominates every added right.

    Both sides use post-event rewards. Over each added-to-coarse interval the
    state is a martingale without cash jumps and the strike funding factor is
    identical. Conditional Jensen then dominates exercise for calls and puts.
    Failure is inconclusive; conditions are sufficient, not necessary.
    """
    coarse, events = _schedule(model, option, coarse)
    fine, _ = _schedule(model, option, fine)
    if not set(coarse).issubset(set(fine)):
        raise ValueError('The coarse exercise set must be a subset with identical factors')
    if any(r.phase != 'after' for r in coarse+fine):
        raise ValueError('This coarsening condition requires post-event rights')
    times = [r.time for r in coarse]
    mismatches = crossings = 0
    for right in fine:
        later = coarse[bisect_left(times,right.time)]
        mismatches += right.settlement_factor != later.settlement_factor
        crossings += any(right.time < t <= later.time for t in events)
    return CoarseningCheck(mismatches == crossings == 0, len(fine), len(coarse),
                           len(fine)-len(coarse), mismatches, crossings)


def _downward(value):
    """Lower float representation of an exact rational comparison margin."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Comparison margin exceeds finite floating-point range')
    return math.nextafter(result, -math.inf) if Fraction(result) > value else result


@dataclass(frozen=True)
class CallDominanceCheck:
    sufficient_condition_satisfied: bool
    decision_count: int
    failed_conditions: int
    minimum_margin_lower: float
    margins_lower: tuple[float, ...]


def terminal_call_dominance(model, option, decisions):
    """Test K(f_t-f_T) >= remaining discounted cash caps at every right.

    If satisfied, the terminal conditional payoff dominates each exercise
    reward, so the exact finite-calendar value equals its European value.
    Failure is inconclusive. Fractions compare the supplied binary inputs
    exactly; this does not verify the underlying financial assumptions.
    """
    if option.kind != 'call':
        raise ValueError('This sufficient condition is for calls only')
    rights, events = _schedule(model, option, decisions)
    terminal = next(d for d in rights if d.time == option.maturity and d.phase == 'after')
    margins = []
    for right in rights:
        remaining = sum((Fraction(event.discounted_cash) for event in events.values()
                         if event.ex_time > right.time or
                         (event.ex_time == right.time and right.phase == 'before')), Fraction(0))
        margin = Fraction(option.strike)*(Fraction(right.settlement_factor)-Fraction(terminal.settlement_factor))-remaining
        margins.append(margin)
    failed = sum(margin < 0 for margin in margins)
    return CallDominanceCheck(failed == 0, len(rights), failed, _downward(min(margins)),
                              tuple(_downward(margin) for margin in margins))
