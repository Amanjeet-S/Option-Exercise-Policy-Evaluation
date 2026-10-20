"""Signed centring-error bounds for a frozen stopping rule.

The starting point is the established martingale dual (Rogers, 2002;
Haugh and Kogan, 2004) and predictable compensation. I retain cancellation
of increments shared by two stopping dates before bounding the difference.
This is an independently written implementation of those arguments, not a
claim of a new general duality theorem.

Inputs are exact binary64 endpoint records. Outward arithmetic encloses the
algebra on those records; it does not certify how an upstream integrator,
simulator or exercise rule supplied them. In particular, the supplied drift
intervals must enclose the actual conditional centring errors, and the chosen
indices must come from an admissible policy. Numerical corrections alone
do not establish confidence coverage.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompensatedGapBounds:
    lower: np.ndarray
    upper: np.ndarray
    uncorrected_gap: np.ndarray
    absolute_drift_allowance: np.ndarray


def _add(first, second, direction):
    result = first + second
    result = np.where((first == 0) & (second == 0), 0.,
                      np.nextafter(result, direction))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError('The interval operation overflowed.')
    return result


def compensated_gap_bounds(rewards, penalty, drift_lower, drift_upper, stopping_indices):
    """Enclose the exactly compensated dual gap for each complete path.

    Column j of the drift bounds encloses the error in the increment from
    the preceding information set to right j, including j=0 from valuation.
    If A is the supplied penalty and B its true accumulated centring error,
    M=A-B is the exact centred martingale. For a selected right tau,
    G(M)=max_j{H_j-A_j-H_tau+A_tau+B_j-B_tau}. Only increments strictly
    between j and tau enter B_j-B_tau. The maximum includes j=tau, whose
    contribution is exactly zero. Shared prefixes are never counted twice.

    This operation is linear in the number of rights and stored paths.
    All inputs must be finite; shape, interval order and indices are checked.
    """
    arrays = [np.asarray(x, dtype=float) for x in
              (rewards, penalty, drift_lower, drift_upper)]
    h, a, lo, hi = arrays
    if h.ndim != 2 or not h.shape[0] or not h.shape[1]:
        raise ValueError('A non-empty path-by-right reward matrix is required.')
    if any(x.shape != h.shape or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError('Finite matrices of the same shape are required.')
    if np.any(lo > hi):
        raise ValueError('Every lower endpoint must be at most its upper endpoint.')
    indices = np.asarray(stopping_indices)
    if (indices.shape != (h.shape[0],) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= h.shape[1])):
        raise ValueError('One valid integer stopping index per path is required.')
    indices = indices.astype(np.intp)
    n, rights = h.shape
    rows = np.arange(n)
    z_lower = _add(h, -a, -np.inf)
    z_upper = _add(h, -a, np.inf)
    lower = np.zeros(n)
    upper = np.zeros(n)
    for sign in (-1, 1):
        accumulated_lower = np.zeros(n)
        accumulated_upper = np.zeros(n)
        for distance in range(1, rights):
            other = indices + sign * distance
            active = (other >= 0) & (other < rights)
            if not np.any(active):
                continue
            selected = rows[active]
            target = other[active]
            increment = target if sign == 1 else target + 1
            delta_lower = lo[selected, increment] if sign == 1 else -hi[selected, increment]
            delta_upper = hi[selected, increment] if sign == 1 else -lo[selected, increment]
            accumulated_lower[active] = _add(accumulated_lower[active], delta_lower, -np.inf)
            accumulated_upper[active] = _add(accumulated_upper[active], delta_upper, np.inf)
            base_lower = _add(z_lower[selected, target], -z_upper[selected, indices[active]], -np.inf)
            base_upper = _add(z_upper[selected, target], -z_lower[selected, indices[active]], np.inf)
            candidate_lower = _add(base_lower, accumulated_lower[active], -np.inf)
            candidate_upper = _add(base_upper, accumulated_upper[active], np.inf)
            lower[active] = np.maximum(lower[active], candidate_lower)
            upper[active] = np.maximum(upper[active], candidate_upper)
    uncorrected = np.max(h-a, axis=1) - (h-a)[rows, indices]
    allowance = np.zeros(n)
    # Increment zero is shared by every right and cancels from every gap.
    for j in range(1, rights):
        allowance = _add(allowance, np.maximum(np.abs(lo[:, j]), np.abs(hi[:, j])), np.inf)
    for result in (lower, upper, uncorrected, allowance):
        result.setflags(write=False)
    return CompensatedGapBounds(lower, upper, uncorrected, allowance)
