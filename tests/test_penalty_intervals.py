"""Exact finite comparisons of signed conditional-error bounds."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from qf_research.optimal_stopping.penalty_intervals import compensated_gap_bounds


def exact_gap(h, a, drift, stopping):
    total = Fraction(0)
    cumulative = []
    for value in drift:
        total += Fraction(float(value))
        cumulative.append(total)
    adjusted = [Fraction(float(x))-Fraction(float(y))+b
                for x, y, b in zip(h, a, cumulative)]
    return max(adjusted)-adjusted[stopping]


def test_all_interval_corners_against_exact_rational_compensation():
    h = np.array([[2., 1., 4., 0.], [0., 5., 2., 4.]])
    a = np.array([[.5, 2., -1., 3.], [1., -1., 2., .25]])
    lo = np.array([[-2., -.75, .1, -.2], [2., -.1, -.5, .2]])
    hi = np.array([[3., -.25, .3, .5], [3., .2, .1, .6]])
    for stopping in range(4):
        bound = compensated_gap_bounds(h, a, lo, hi, np.full(2, stopping))
        for row in range(2):
            for corner in product(*zip(lo[row], hi[row])):
                actual = exact_gap(h[row], a[row], corner, stopping)
                assert Fraction(float(bound.lower[row])) <= actual <= Fraction(float(bound.upper[row]))


def test_shared_initial_error_cancels_even_if_its_interval_is_large():
    h = np.array([[1., 3., 2.]])
    a = np.array([[.2, .5, .1]])
    zero = np.zeros_like(h)
    narrow = compensated_gap_bounds(h, a, zero, zero, np.array([1]))
    lo, hi = zero.copy(), zero.copy()
    lo[0, 0], hi[0, 0] = -1e100, 1e100
    wide = compensated_gap_bounds(h, a, lo, hi, np.array([1]))
    np.testing.assert_array_equal(wide.lower, narrow.lower)
    np.testing.assert_array_equal(wide.upper, narrow.upper)
    assert wide.absolute_drift_allowance[0] == 0


def test_signed_correction_preserves_strictly_smaller_allowance():
    h = np.array([[0., 10., 0.]])
    a = np.zeros_like(h)
    drift = np.array([[0., 2., -2.]])
    result = compensated_gap_bounds(h, a, drift, drift, np.array([1]))
    assert result.upper[0] < 1e-12
    assert result.absolute_drift_allowance[0] >= 4


def test_single_right_has_exactly_zero_gap():
    result = compensated_gap_bounds([[5.]], [[2.]], [[-3.]], [[7.]], np.array([0]))
    assert result.lower[0] == result.upper[0] == result.uncorrected_gap[0] == 0


def test_mixed_stopping_rights_and_absorbing_zero_path():
    h = np.array([[0., 0., 0.], [2., 3., 4.]])
    a = np.zeros_like(h)
    drift = np.array([[0., 0., 0.], [.1, -.2, .3]])
    result = compensated_gap_bounds(h, a, drift, drift, np.array([0, 2]))
    for row, stopping in enumerate((0, 2)):
        actual = exact_gap(h[row], a[row], drift[row], stopping)
        assert Fraction(float(result.lower[row])) <= actual <= Fraction(float(result.upper[row]))
    assert not result.upper.flags.writeable


@pytest.mark.parametrize('dtype', [np.uint8, np.uint64])
def test_unsigned_stopping_indices_accept_valid_first_middle_and_last_rights(dtype):
    h = np.array([[1., 3., 2.], [2., 1., 4.], [0., 5., 1.]])
    a = np.array([[.2, .5, .1], [1., -.25, .5], [.1, 1., -.5]])
    lo = np.array([[0., -.2, .1], [0., -.5, -.3], [0., .2, -.1]])
    hi = np.array([[0., .1, .3], [0., .2, .4], [0., .5, .2]])
    signed = compensated_gap_bounds(h, a, lo, hi, np.array([0, 1, 2], dtype=np.int64))
    unsigned = compensated_gap_bounds(h, a, lo, hi, np.array([0, 1, 2], dtype=dtype))
    np.testing.assert_array_equal(unsigned.lower, signed.lower)
    np.testing.assert_array_equal(unsigned.upper, signed.upper)


@pytest.mark.parametrize('indices', [np.array([-1]), np.array([2]), np.array([.5]), np.array([True])])
def test_invalid_stopping_indices_rejected(indices):
    with pytest.raises(ValueError, match='stopping index'):
        compensated_gap_bounds([[1., 2.]], [[0., 0.]], [[0., 0.]], [[0., 0.]], indices)


def test_invalid_intervals_shapes_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError, match='endpoint'):
        compensated_gap_bounds([[1.]], [[0.]], [[2.]], [[1.]], np.array([0]))
    for h in ([[float('nan')]], [], [[1., 2.]]):
        with pytest.raises(ValueError):
            compensated_gap_bounds(h, [[0.]], [[0.]], [[1.]], np.array([0]))


def test_overflow_rejected_instead_of_returning_a_certificate():
    with np.errstate(over='ignore'):
        with pytest.raises(FloatingPointError, match='overflow'):
            compensated_gap_bounds([[1e308, 1e308]], [[-1e308, 0.]], [[0., 0.]], [[0., 0.]], np.array([0]))
