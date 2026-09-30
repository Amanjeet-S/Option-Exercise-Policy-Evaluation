"""Exact component sensitivity for fixed, feasible computational allocations.

The compact coordinate is the evaluation share at a reference path count.
Deletion diagnostics do not supply a confidence region for that coordinate.
"""

from fractions import Fraction
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f'{name} must be a finite real number')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return Fraction(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return value


def moments(means, standard_errors, paths):
    """Reconstruct signed components from retained paired cluster summaries.

    Arithmetic is exact for the supplied summaries, including squaring each
    recorded standard error. It does not enclose their earlier rounding error.
    """
    _integer(paths, 'paths', 2)
    means = [_number(x, 'mean') for x in means]
    errors = [_number(x, 'standard error') for x in standard_errors]
    if len(means) < 2 or len(means) != len(errors) or any(x < 0 for x in errors):
        raise ValueError('Require at least two paired means and nonnegative errors')
    count = len(means)
    average = sum(means, Fraction()) / count
    observed = sum(((x-average)**2 for x in means), Fraction()) / (count-1)
    evaluation = sum((x*x for x in errors), Fraction()) / count
    return dict(observed_variance=observed, evaluation_mean_variance=evaluation,
                A=observed-evaluation, B=paths*evaluation)


def share(a, b, reference_paths=4096):
    """Return the normalised evaluation contribution, including axis cases."""
    _integer(reference_paths, 'reference_paths')
    a, b = _number(a, 'A'), _number(b, 'B')
    if a < 0 or b < 0 or a+b == 0:
        raise ValueError('Require nonnegative components, not both zero')
    return b/(reference_paths*a+b)


def _candidates(candidates, reference_paths):
    _integer(reference_paths, 'reference_paths')
    rows = []
    for row in candidates:
        n = _integer(row['evaluation_paths'], 'evaluation_paths')
        r = _integer(row['repetitions'], 'repetitions')
        rows.append(dict(evaluation_paths=n, repetitions=r))
    if not rows or len({row['evaluation_paths'] for row in rows}) != len(rows):
        raise ValueError('Require a nonempty set of distinct evaluation sizes')
    return sorted(rows, key=lambda row: (row['evaluation_paths'], row['repetitions']))


def _weight(w):
    w = _number(w, 'share')
    if not 0 <= w <= 1:
        raise ValueError('Share must lie in [0, 1]')
    return w


def _line(row, reference_paths):
    n, r = row['evaluation_paths'], row['repetitions']
    intercept = Fraction(1, r)
    return intercept, Fraction(reference_paths, r*n)-intercept


def _value(row, w, reference_paths):
    intercept, slope = _line(row, reference_paths)
    return intercept+slope*w


def _target(rows, target_N):
    _integer(target_N, 'target_N')
    for row in rows:
        if row['evaluation_paths'] == target_N:
            return row
    raise ValueError('Target evaluation size is absent')


def select(candidates, w, reference_paths=4096):
    """Select the least normalised variance with deterministic exact ties."""
    rows = _candidates(candidates, reference_paths)
    w = _weight(w)
    values = [_value(row, w, reference_paths) for row in rows]
    best = min(values)
    minimisers = [row for row, value in zip(rows, values) if value == best]
    return dict(**minimisers[0], criterion=best, minimisers=minimisers)


def region(candidates, target_N, reference_paths=4096):
    """Intersect all affine inequalities for a target's weak optimality set.

    A weak tie at a boundary is retained even when the deterministic selection
    belongs to another size. Empty weak optimality sets return None.
    """
    rows = _candidates(candidates, reference_paths)
    target = _target(rows, target_N)
    intercept, slope = _line(target, reference_paths)
    lower, upper = Fraction(0), Fraction(1)
    for row in rows:
        other_intercept, other_slope = _line(row, reference_paths)
        delta_intercept = intercept-other_intercept
        delta_slope = slope-other_slope
        if delta_slope == 0:
            if delta_intercept > 0:
                return None
        elif delta_slope > 0:
            upper = min(upper, -delta_intercept/delta_slope)
        else:
            lower = max(lower, -delta_intercept/delta_slope)
    if lower > upper:
        return None
    return dict(lower=lower, upper=upper,
                lower_selected=select(rows, lower, reference_paths)['evaluation_paths'] == target_N,
                upper_selected=select(rows, upper, reference_paths)['evaluation_paths'] == target_N)


def envelope(candidates, reference_paths=4096):
    """Return all nonempty closed weak optimality regions and endpoint owners."""
    rows = _candidates(candidates, reference_paths)
    result = []
    for row in rows:
        interval = region(rows, row['evaluation_paths'], reference_paths)
        if interval is not None:
            result.append(dict(**row, region=interval))
    return result


def regret(candidates, target_N, lower, upper, reference_paths=4096):
    """Worst multiplicative variance ratio on a declared share interval.

    Each pairwise ratio of positive affine functions is monotone or constant.
    The supremum of their finite maximum therefore occurs at an endpoint.
    """
    rows = _candidates(candidates, reference_paths)
    target = _target(rows, target_N)
    lower, upper = _weight(lower), _weight(upper)
    if lower > upper:
        raise ValueError('Interval endpoints are reversed')
    ratios = [_value(target, w, reference_paths)/select(rows, w, reference_paths)['criterion']
              for w in (lower, upper)]
    maximum = max(ratios)
    endpoints = sorted({w for w, value in zip((lower, upper), ratios) if value == maximum})
    return dict(maximum=maximum, lower_ratio=ratios[0], upper_ratio=ratios[1],
                worst_endpoints=endpoints)
