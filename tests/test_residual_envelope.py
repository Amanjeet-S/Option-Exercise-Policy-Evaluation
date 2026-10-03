"""Analytical and pathwise checks on stipulated mathematical examples."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import DividendCap, FundedCashModel, FundedExercise
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.residual_envelope import (
    diffusion_residual_range, linear_difference_range, stopping_gain_envelope,
    _transition_curvature,
)
from qf_research.optimal_stopping.value_martingale import (
    LinearValue, build_values, observation_rights, value_martingale,
)


def test_zero_volatility_matches_exact_knot_extrema():
    left = LinearValue((0., 1., 3.), (4., 2., 0.))
    right = LinearValue((0., 2., 4.), (3., 1., 0.))
    result = diffusion_residual_range(left, right, 0.)
    assert result.lower == -.5
    assert result.upper == 1.
    assert result.converged


@pytest.mark.parametrize('sd', [.01, .3, 1.5])
def test_single_put_residual_contains_analytical_global_extrema(sd):
    # By convexity, the residual is non-positive; its minimum is at the kink.
    strike = 2.
    put = LinearValue((0., strike), (strike, 0.))
    minimum = -strike * math.erf(sd/(2*math.sqrt(2)))
    result = diffusion_residual_range(put, put, sd, tolerance=1e-5)
    assert result.lower <= minimum + 1e-13
    assert result.upper >= 0.
    assert minimum - result.lower <= 1.0001e-5
    assert result.upper <= 1.0001e-5
    assert result.converged


def test_unbounded_tail_and_resource_exhaustion_are_retained():
    left = LinearValue((0., .01), (2., 2.))
    right = LinearValue((0., 1.), (1., 0.))
    result = diffusion_residual_range(left, right, 2., max_rounds=0)
    assert result.lower <= 1. and result.upper >= 2.
    assert result.tail_upper == 2.
    assert result.tail_lower < 2.
    put = LinearValue((0., 1.), (1., 0.))
    unfinished = diffusion_residual_range(put, put, 1., tolerance=1e-15, max_rounds=0)
    assert not unfinished.converged


def test_signed_hinge_curvature_bounds_independent_second_differences():
    # Cash-like plateaux destroy convexity while preserving monotonicity.
    function = LinearValue((0., 1., 2., 4., 6.), (5., 5., 2., 1., 1.))
    lower, upper = np.array([0., .5, 2., 5.]), np.array([.5, 2., 5., 10.])
    bounds = _transition_curvature(function, .4, lower, upper)
    for a, b, bound in zip(lower, upper, bounds):
        for x in np.linspace(max(a, .01), b, 13):
            h = x*1e-3
            second = (function.expectation(x+h, .4) - 2*function.expectation(x, .4)
                      + function.expectation(x-h, .4))/h**2
            assert abs(second) <= bound + 2e-5


def test_signed_hinges_against_independent_segment_quadrature():
    function = LinearValue((0., 1., 2., 4., 6.), (5., 5., 2., 1., 1.))
    left = LinearValue((0., 2., 5.), (4., 2., 0.))
    result = diffusion_residual_range(left, function, .5, tolerance=1e-4)
    for y in [.1, .8, 1.5, 2.5, 8., 100.]:
        breaks = [-np.inf, *[(math.log(k/y)+.125)/.5 for k in function.knots[1:]], np.inf]
        def integrand(z):
            if abs(z) > 38:
                return 0.
            return float(function(y*math.exp(.5*z-.125)))*math.exp(-z*z/2)/math.sqrt(2*math.pi)
        expected = sum(quad(integrand, a, b, epsabs=1e-11)[0]
                       for a, b in zip(breaks, breaks[1:]))
        residual = float(left(y)) - expected
        assert result.lower - 1e-10 <= residual <= result.upper + 1e-10


@pytest.mark.parametrize('sd', [0., .3])
def test_expanded_cash_identity_and_all_exercise_pair_bounds(sd):
    model = FundedCashModel(100., sd, 1., .97,
                           (DividendCap(.4, 30.), DividendCap(.8, 40.)))
    option = Option(100., 1., 'put')
    rights = (FundedExercise(0., 1.), FundedExercise(.2, .99),
              FundedExercise(.4, .98, 'before'), FundedExercise(.4, .98),
              FundedExercise(.7, .97), FundedExercise(1., .96))
    approximation = build_values(model, option, rights, core_steps=40, wing_steps=8, tolerance=.01)
    envelope = stopping_gain_envelope(approximation, tolerance=.001)
    observations = simulate_paths(model, option, observation_rights(approximation),
                                  64, np.random.default_rng(6209))
    sample = value_martingale(approximation, observations)
    index = {(d.time, d.phase): j for j, d in enumerate(observations.decisions)}
    accumulated = np.zeros(64)
    state = np.full(64, model.initial_state)
    previous = approximation.nodes[0]
    scores = {(0., 'after'): float(previous.after(model.initial_state))
              + model.reward(option, rights[0], state) - previous.after(state)}
    at = {(d.time, d.phase): d for d in rights}
    for node in approximation.nodes[1:]:
        value = node.before if node.before is not None else node.after
        accumulated += previous.after(state) - value.expectation(state, sd*math.sqrt(node.time-previous.time))
        phase = 'before' if node.before is not None else 'after'
        state = observations.states[:, index[node.time, phase]]
        if (node.time, phase) in at:
            scores[node.time, phase] = (approximation.initial_value
                + model.reward(option, at[node.time, phase], state) - value(state) - accumulated)
        if node.before is not None:
            after = observations.states[:, index[node.time, 'after']]
            accumulated += node.before(state) - node.after(after)
            state = after
            if (node.time, 'after') in at:
                scores[node.time, 'after'] = (approximation.initial_value
                    + model.reward(option, at[node.time, 'after'], state) - node.after(state) - accumulated)
        previous = node
    direct = np.column_stack([model.reward(option, d, sample.paths.states[:, j]) - sample.values[:, j]
                              for j, d in enumerate(rights)])
    telescoped = np.column_stack([scores[d.time, d.phase] for d in rights])
    np.testing.assert_allclose(telescoped, direct, atol=1e-11)
    for pair in envelope['stopping_pairs']:
        difference = direct[:, pair['later']] - direct[:, pair['earlier']]
        assert difference.min() >= pair['lower']-1e-10
        assert difference.max() <= pair['upper']+1e-10
    assert np.ptp(direct, axis=1).max()/model.premium_factor <= envelope['gain_absolute_bound']+1e-10


def test_single_exercise_has_zero_paired_gain_envelope():
    model = FundedCashModel(100., .2, 1., 1.)
    option = Option(100., 1., 'put')
    approximation = build_values(model, option, (FundedExercise(1., 1.),), core_steps=20)
    assert stopping_gain_envelope(approximation)['gain_absolute_bound'] == 0.


def test_invalid_monotonicity_and_unbounded_tail_are_rejected():
    valid = LinearValue((0., 1.), (1., 0.))
    for invalid in [LinearValue((0., 1., 2.), (1., 0., 1.)),
                    LinearValue((0., 1.), (1., 0.), -.1)]:
        with pytest.raises(ValueError):
            diffusion_residual_range(valid, invalid, .2)
    with pytest.raises(ValueError):
        linear_difference_range(valid, LinearValue((0., 1.), (1., 0.), .1))
