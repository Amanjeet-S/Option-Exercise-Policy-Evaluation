"""Backward-residual envelopes for an existing approximate-value martingale.

The identities are finite telescoping sums, applied to the approximate-value
construction of Haugh and Kogan (2004). Monotone interval enclosures cover the
whole non-negative state domain in exact arithmetic. The implementation uses
ordinary floating point and does not certify rounding or normal-CDF errors.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from .value_martingale import LinearValue, jump_value


@dataclass(frozen=True)
class RangeEnclosure:
    lower: float
    upper: float
    sampled_minimum: float
    sampled_maximum: float
    tolerance: float
    converged: bool
    evaluations: int
    cells: int
    tail_lower: float
    tail_upper: float


def _bounded_decreasing(value):
    if not isinstance(value, LinearValue) or value.tail_slope != 0.:
        raise ValueError('A constant-tail LinearValue is required')
    if np.any(np.diff(value.values) > 0):
        raise ValueError('The monotone enclosure requires non-increasing values')


def linear_difference_range(left, right):
    """Extrema of two constant-tail linear functions occur at their knot union."""
    if left.tail_slope != 0. or right.tail_slope != 0.:
        raise ValueError('Constant tails are required')
    grid = np.union1d(left.knots, right.knots)
    difference = left(grid) - right(grid)
    return float(difference.min()), float(difference.max())


def _transition_curvature(right, sd, lower, upper):
    """Upper bound for |(P right)''| on each positive-width state interval.

    The put-hinge gamma peaks at strike*exp(-3*sd**2/2). Absolute
    hinge weights remain valid when a cash cap destroys convexity.
    """
    knots, values = np.asarray(right.knots), np.asarray(right.values)
    weights = np.abs(np.diff(np.r_[np.diff(values)/np.diff(knots), 0.]))
    active = weights > 0.
    log_strikes, weights = np.log(knots[1:][active]), weights[active]
    result = np.zeros(len(lower))
    if not len(weights):
        return result
    for start in range(0, len(lower), 512):
        stop = start + 512
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            log_a, log_b = np.log(lower[start:stop]), np.log(upper[start:stop])
            peak = np.clip(log_strikes[None, :] - 1.5*sd**2,
                           log_a[:, None], log_b[:, None])
            z = (log_strikes[None, :] - peak)/sd - sd/2
            gamma = np.exp(-z**2/2 - peak - math.log(sd) - .5*math.log(2*math.pi))
            result[start:stop] = gamma @ weights
    return result


def diffusion_residual_range(left, right, sd, *, tolerance=.01,
                             max_rounds=24, max_points=200000, curvature=True):
    """Enclose left(y)-E[right(y exp(sd Z-sd²/2))] for every y>=0.

    On [a,b], monotonicity gives [left(b)-P right(a),
    left(a)-P right(b)]. Only cells which could improve an observed extremum
    by more than tolerance are bisected. The unbounded constant-tail region
    is retained explicitly between its endpoint value and its limit. Because
    left is constant there, monotonicity gives its full range directly.
    On cells where left is affine, a bound on the second derivative of the
    conditional expectation supplies an additional chord-error enclosure.
    A resource limit returns a valid mathematical enclosure with converged
    false, rather than presenting sampled extrema as global extrema.
    """
    _bounded_decreasing(left)
    _bounded_decreasing(right)
    if not math.isfinite(sd) or sd < 0:
        raise ValueError('Finite non-negative volatility-time is required')
    if isinstance(tolerance, bool) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('A finite positive tolerance is required')
    if type(max_rounds) is not int or max_rounds < 0:
        raise ValueError('max_rounds must be a non-negative integer')
    if type(max_points) is not int or max_points < 2:
        raise ValueError('max_points must be an integer of at least two')
    if sd == 0:
        lo, hi = linear_difference_range(left, right)
        tail = left.values[-1] - right.values[-1]
        return RangeEnclosure(lo, hi, lo, hi, tolerance, True,
                              len(np.union1d(left.knots, right.knots)),
                              len(np.union1d(left.knots, right.knots))-1, tail, tail)
    grid = np.union1d(left.knots, right.knots)
    values = left(grid)
    means = right.expectation(grid, sd)
    evaluations = len(grid)
    # The limit at infinity is itself a justified limiting extremum candidate.
    limit = left.values[-1] - right.values[-1]
    for iteration in range(max_rounds + 1):
        observed = values - means
        sample_lo = min(float(observed.min()), limit)
        sample_hi = max(float(observed.max()), limit)
        cell_lo = values[1:] - means[:-1]
        cell_hi = values[:-1] - means[1:]
        if curvature:
            gamma = _transition_curvature(right, sd, grid[:-1], grid[1:])
            with np.errstate(over='ignore', invalid='ignore'):
                allowance = gamma * np.diff(grid)**2 / 8
            allowance = np.where(np.isfinite(allowance), allowance, np.inf)
            cell_lo = np.maximum(cell_lo, np.minimum(observed[:-1], observed[1:]) - allowance)
            cell_hi = np.minimum(cell_hi, np.maximum(observed[:-1], observed[1:]) + allowance)
        tail_lo, tail_hi = left.values[-1] - means[-1], limit
        lo = min(float(cell_lo.min()), tail_lo, sample_lo)
        hi = max(float(cell_hi.max()), tail_hi, sample_hi)
        converged = sample_lo - lo <= tolerance and hi - sample_hi <= tolerance
        if converged or iteration == max_rounds or len(grid) >= max_points:
            return RangeEnclosure(lo, hi, sample_lo, sample_hi, tolerance,
                converged, evaluations, len(grid)-1, float(tail_lo), float(tail_hi))
        active = (cell_lo < sample_lo - tolerance) | (cell_hi > sample_hi + tolerance)
        additions = grid[:-1][active] + np.diff(grid)[active] / 2
        additions = np.setdiff1d(additions, grid)
        if not len(additions) or len(grid) + len(additions) > max_points:
            return RangeEnclosure(lo, hi, sample_lo, sample_hi, tolerance,
                False, evaluations, len(grid)-1, float(tail_lo), float(tail_hi))
        merged = np.r_[grid, additions]
        order = np.argsort(merged)
        values = np.r_[values, left(additions)][order]
        means = np.r_[means, right.expectation(additions, sd)][order]
        grid = merged[order]
        evaluations += len(additions)
    raise AssertionError('Unreachable refinement state')


def stopping_gain_envelope(approximation, *, tolerance=.01, max_rounds=24):
    """Apply residual cancellation between all pairs of ordered exercise rights.

    For vertices k<l, adjusted stopped rewards differ by
    d_l(Y_l)-d_k(Y_k)-sum_{j=k}^{l-1} r_j(Y_j), where d=H-v.
    Both possible signs are bounded. Deterministic cash edges are included
    even though their martingale increments vanish.
    """
    if approximation.option.kind != 'put':
        raise ValueError('The reward envelope currently requires a put')
    nodes = approximation.nodes
    if not nodes or nodes[0].time != 0. or nodes[0].before is not None:
        raise ValueError('Require a post-event initial node at time zero')
    vertices = {(0., 'after'): (0, nodes[0].after)}
    edges = []
    events = {e.ex_time: e for e in approximation.model.dividends}
    previous = nodes[0]
    for node in nodes[1:]:
        if node.time <= previous.time:
            raise ValueError('Node times must strictly increase')
        destination = node.before if node.before is not None else node.after
        result = diffusion_residual_range(previous.after, destination,
            approximation.model.volatility * math.sqrt(node.time - previous.time),
            tolerance=tolerance, max_rounds=max_rounds)
        edges.append(dict(kind='diffusion', from_time=previous.time, to_time=node.time,
                          **asdict(result)))
        if node.before is not None:
            if node.time not in events:
                raise ValueError('A before-event node requires a cash event')
            vertices[node.time, 'before'] = (len(edges), node.before)
            composed = jump_value(node.after, events[node.time].discounted_cash)
            lo, hi = linear_difference_range(node.before, composed)
            edges.append(dict(kind='cash', from_time=node.time, to_time=node.time,
                              lower=lo, upper=hi, converged=True))
        vertices[node.time, 'after'] = (len(edges), node.after)
        previous = node
    rights = []
    for right in approximation.decisions:
        index, value = vertices[right.time, right.phase]
        threshold = approximation.option.strike * right.settlement_factor
        payoff = LinearValue((0., threshold), (threshold, 0.))
        lo, hi = linear_difference_range(payoff, value)
        rights.append(dict(vertex=index, time=right.time, phase=right.phase,
                           defect_lower=lo, defect_upper=hi))
    if any(b['vertex'] <= a['vertex'] for a, b in zip(rights, rights[1:])):
        raise ValueError('Exercise rights must be strictly ordered')
    bounds = [0.]
    pairs = []
    for i, earlier in enumerate(rights):
        for j in range(i+1, len(rights)):
            later = rights[j]
            between = edges[earlier['vertex']:later['vertex']]
            residual_lo = math.fsum(e['lower'] for e in between)
            residual_hi = math.fsum(e['upper'] for e in between)
            lower = later['defect_lower'] - earlier['defect_upper'] - residual_hi
            upper = later['defect_upper'] - earlier['defect_lower'] - residual_lo
            bound = max(abs(lower), abs(upper))
            bounds.append(bound)
            pairs.append(dict(earlier=i, later=j, lower=lower, upper=upper))
    factor = approximation.model.premium_factor
    return dict(approximation_fingerprint=approximation.fingerprint,
                gain_absolute_bound=max(bounds)/factor, edges=edges, rights=rights,
                stopping_pairs=pairs, tolerance=tolerance,
                all_refinements_converged=all(e['converged'] for e in edges),
                floating_point_bound_certified=False)
