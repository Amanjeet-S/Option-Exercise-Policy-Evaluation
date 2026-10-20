"""Clipping, independent cap selection and analytic-tail checks."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from validate_gap_coverage import GapProcess, centred_gap
from validate_clipped_gap_coverage import (
    analytic_tail_allowance, baseline_replay_status, bounded_upper_limits, envelope_coefficients, select_caps,
)


@pytest.mark.parametrize("martingale", ["zero", "terminal_call"])
@pytest.mark.parametrize("cap", [0., 4., 24., 96.])
def test_analytic_envelope_tail_against_independent_normal_quadrature(martingale, cap):
    process = GapProcess()
    scale, shift = envelope_coefficients(process, martingale)
    sd = process.volatility*math.sqrt(process.first_right)
    drift = (process.rate-process.dividend_yield-.5*process.volatility**2)*process.first_right
    threshold = (math.log((shift+cap)/(scale*process.spot))-drift)/sd
    integrand = lambda z: max(scale*process.spot*math.exp(drift+sd*z)-shift-cap, 0.) \
        *math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    expected = quad(integrand, max(-12., threshold), 12., epsabs=1e-11)[0]
    assert analytic_tail_allowance(process, martingale, cap) == pytest.approx(expected, abs=2e-10)


def test_shifted_jensen_envelope_bounds_centred_gap_and_improves_tail_allowance():
    process = GapProcess()
    states = np.geomspace(.01, 1e6, 500)
    scale, shift = envelope_coefficients(process, "terminal_call")
    assert np.all(centred_gap(states, process) <= np.maximum(scale*states-shift, 0.)+1e-10)
    for cap in (1., 8., 16., 32.):
        assert analytic_tail_allowance(process, "terminal_call", cap) <= \
            analytic_tail_allowance(process, "terminal_call", cap, shifted=False)


def test_published_empirical_bernstein_variance_convention_and_constants():
    clipped = np.array([[0., 1., 2., 4.]])
    mean, upper = bounded_upper_limits(clipped, 4., .25)
    pairwise = sum((a-b)**2 for i, a in enumerate(clipped[0]) for b in clipped[0][i+1:])/(4*3)
    assert pairwise == pytest.approx(np.var(clipped[0], ddof=1))
    expected = mean[0]+math.sqrt(2*pairwise*math.log(80)/4)+7*4*math.log(80)/(3*3)+.25
    assert upper["empirical_bernstein"][0] == pytest.approx(expected)
    assert upper["hoeffding"][0] == pytest.approx(mean[0]+4*math.sqrt(math.log(40)/8)+.25)


def test_constant_and_zero_cap_samples_retain_nonnegative_deterministic_allowance():
    samples = np.full((3, 8), 2.)
    mean, upper = bounded_upper_limits(samples, 2., .5)
    assert np.all(upper["empirical_bernstein"] > mean+.5)
    zero_mean, zero_upper = bounded_upper_limits(np.zeros((3, 8)), 0., .5)
    np.testing.assert_array_equal(zero_mean, 0.)
    np.testing.assert_array_equal(zero_upper["hoeffding"], .5)
    np.testing.assert_array_equal(zero_upper["empirical_bernstein"], .5)


@pytest.mark.parametrize("samples,cap,tail,alpha", [
    (np.zeros((2, 1)), 1., 0., .025), (np.array([[0., -1.]]), 1., 0., .025),
    (np.array([[0., 2.]]), 1., 0., .025), (np.array([[0., np.nan]]), 1., 0., .025),
    (np.zeros((2, 2)), -1., 0., .025), (np.zeros((2, 2)), 1., -.1, .025),
    (np.zeros((2, 2)), 1., 0., 0.), (np.zeros((2, 2)), 1., 0., 1.),
])
def test_invalid_bounded_samples_and_probability_limits_rejected(samples, cap, tail, alpha):
    with pytest.raises(ValueError):
        bounded_upper_limits(samples, cap, tail, alpha=alpha)


def test_cap_selection_uses_only_independent_pilot_and_fixed_grid():
    process = GapProcess()
    pilot = {"terminal_call": np.array([0., 0., 1., 2., 8., 30.])}
    records = select_caps(pilot, [1., 4., 16., 32.], [8, 512], process)
    assert len(records) == 4
    assert all(record["cap"] in (1., 4., 16., 32.) for record in records)
    # No evaluation sample is an argument to selection, and no observed maximum sets the tail.
    assert all(record["tail_allowance"] == analytic_tail_allowance(
        process, "terminal_call", record["cap"]) for record in records)


@pytest.mark.parametrize("alpha", [0., 1., -0.1, float("nan"), float("inf")])
def test_cap_selection_rejects_invalid_probability_before_freezing_forecasts(alpha):
    with pytest.raises(ValueError, match="probability"):
        select_caps({"terminal_call": np.array([0., 1.])}, [1., 2.], [2], GapProcess(), alpha=alpha)


def test_zero_dividend_yield_has_zero_centred_tail_envelope():
    process = GapProcess(dividend_yield=0.)
    assert analytic_tail_allowance(process, "terminal_call", 0.) == 0.
    assert analytic_tail_allowance(process, "terminal_call", 8.) == 0.


def test_replay_discloses_stable_formula_roundoff_without_relaxing_counts_or_means():
    original = dict(expected_gap_covered=5000, policy_loss_covered=5000,
                    average_sample_mean=.5, average_upper_limit=2.)
    rounded = dict(original, average_upper_limit=math.nextafter(2., math.inf))
    status = baseline_replay_status(rounded, original)
    assert not status["original_baseline_replayed_exactly"]
    assert status["original_baseline_replayed_to_roundoff"]
    assert status["original_average_upper_difference"] > 0.
    for changed in (dict(rounded, expected_gap_covered=4999),
                    dict(rounded, average_sample_mean=math.nextafter(.5, math.inf)),
                    dict(rounded, average_upper_limit=2.+1e-10)):
        with pytest.raises(ValueError):
            baseline_replay_status(changed, original)
