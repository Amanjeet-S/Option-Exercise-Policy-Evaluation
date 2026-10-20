"""Independent conditioning, martingale, moment-bound and coverage checks."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from validate_gap_coverage import (
    GapProcess, analytic_second_moment_bounds, centred_gap, conditional_terminal_reward,
    coverage_record, first_reward, gap_samples, numerical_oracle, undiscounted_call,
    upper_limits, wilson_interval, zero_gap_conditional_mean,
)
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option


@pytest.mark.parametrize("state", [65., 100., 110., 150., 300.])
def test_conditional_terminal_call_matches_independent_black_scholes(state):
    process = GapProcess()
    horizon = process.maturity-process.first_right
    price = black_scholes(Market(state, process.rate, process.volatility, process.dividend_yield),
                          Option(process.strike, horizon, "call"))
    assert float(conditional_terminal_reward(state, process)) == pytest.approx(
        math.exp(-process.rate*process.first_right)*price, abs=1e-12)


@pytest.mark.parametrize("state", [90., 105., 120., 180.])
def test_capped_call_identity_against_independent_conditional_integration(state):
    process = GapProcess()
    horizon = process.maturity-process.first_right
    sd = process.volatility*math.sqrt(horizon)
    drift = (process.rate-process.dividend_yield-.5*process.volatility**2)*horizon
    first = float(first_reward(state, process))
    cap = process.strike+first*math.exp(process.rate*process.maturity)
    breaks = sorted({-12., 12., *[(math.log(strike/state)-drift)/sd
                     for strike in (process.strike, cap)]})
    def integrand(z):
        terminal_state = state*math.exp(drift+sd*z)
        reward = math.exp(-process.rate*process.maturity)*max(terminal_state-process.strike, 0.)
        return max(first-reward, 0.)*math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    expected = sum(quad(integrand, a, b, epsabs=1e-11)[0] for a, b in zip(breaks, breaks[1:]))
    assert float(zero_gap_conditional_mean(state, process)) == pytest.approx(expected, abs=2e-10)


def test_terminal_call_martingale_gap_equals_explicit_pathwise_dual_loss():
    process = GapProcess()
    first = np.array([80., 105., 125., 180.])
    terminal = np.array([150., 80., 170., 120.])
    h1 = first_reward(first, process)
    h2 = math.exp(-process.rate*process.maturity)*np.maximum(terminal-process.strike, 0.)
    expected_h2 = math.exp(-process.rate*process.maturity)*float(undiscounted_call(
        process.spot, process.strike, process.maturity, process))
    m1 = conditional_terminal_reward(first, process)-expected_h2
    m2 = h2-expected_h2
    explicit_gap = np.maximum(h1-m1, h2-m2)-(h2-m2)
    np.testing.assert_allclose(gap_samples(first, terminal, process)["terminal_call"], explicit_gap, atol=1e-12)
    np.testing.assert_array_equal(gap_samples(first, terminal, process)["zero"], np.maximum(h1-h2, 0.))


def test_oracle_distinguishes_policy_loss_from_zero_martingale_slack():
    oracle = numerical_oracle(GapProcess())
    assert oracle["expected_gaps"]["zero"]["value"] > oracle["true_two_right_policy_loss"] > 0.
    assert oracle["terminal_call_martingale_dual_slack"] == 0.
    assert oracle["exercise_boundary_at_first_right"] > 100.
    assert not oracle["numerical_oracle_is_rigorous_enclosure"]
    for record in oracle["expected_gaps"].values():
        assert record["quadrature_error_estimate"] < 1e-8
        assert record["omitted_tail_expectation_bound"] < 1e-25


def test_centred_gap_has_the_declared_analytic_envelope():
    process = GapProcess()
    states = np.geomspace(.01, 1e6, 500)
    a = math.exp(-process.rate*process.first_right)*(1-math.exp(-process.dividend_yield
                                               *(process.maturity-process.first_right)))
    assert np.all(centred_gap(states, process) <= a*states+1e-10)
    bounds = analytic_second_moment_bounds(process)
    expected = a*a*process.spot**2*math.exp((2*(process.rate-process.dividend_yield)
                                          +process.volatility**2)*process.first_right)
    assert bounds["terminal_call"] == pytest.approx(expected)
    assert 0 < bounds["terminal_call"] < bounds["zero"]


def test_positive_small_yield_cannot_cancel_the_cantelli_moment_bound_to_zero():
    process = GapProcess(rate=0., dividend_yield=1e-20)
    horizon = process.maturity-process.first_right
    scale = math.exp(-process.rate*process.first_right)*(-math.expm1(-process.dividend_yield*horizon))
    second = process.spot**2*math.exp((2*(process.rate-process.dividend_yield)
                                     +process.volatility**2)*process.first_right)
    bound = analytic_second_moment_bounds(process)["terminal_call"]
    assert bound > 0.
    assert bound == pytest.approx(scale**2*second, rel=1e-14, abs=0.)


@pytest.mark.parametrize("rate", [0., .05])
def test_zero_dividend_call_has_exactly_zero_early_exercise_loss(rate):
    process = GapProcess(rate=rate, dividend_yield=0.)
    states = np.geomspace(.01, 1e12, 400)
    np.testing.assert_array_equal(centred_gap(states, process), np.zeros_like(states))
    assert numerical_oracle(process)["true_two_right_policy_loss"] == 0.


def test_student_and_cantelli_limits_have_distinct_finite_sample_behaviour():
    samples = np.array([[0., 0., 0., 0.], [0., 1., 0., 3.]])
    means, upper = upper_limits(samples, 10.)
    assert upper["student"][0] == 0.
    assert upper["cantelli"][0] > 0.
    assert upper["student"][1] == pytest.approx(
        means[1]+student_t.ppf(.975, 3)*np.std(samples[1], ddof=1)/2)
    np.testing.assert_allclose(upper["cantelli"]-means, math.sqrt(10.*39/4))
    record = coverage_record(upper["student"], means, expected_gap=1., policy_loss=.25)
    assert record["expected_gap_covered"] == record["policy_loss_covered"] == 1
    assert record["expected_gap_coverage"] == .5
    interval = wilson_interval(50, 100)
    assert interval[0] < .5 < interval[1]
    assert interval[0]+interval[1] == pytest.approx(1.)


def test_modified_calendar_process_is_validated():
    with pytest.raises(ValueError):
        GapProcess(first_right=3.)
    with pytest.raises(ValueError):
        GapProcess(dividend_yield=-.1)
