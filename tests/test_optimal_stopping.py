"""Financial identities, independent numerical checks and information-set tests.

Fixtures are constructed for this repository, not transcribed from paper tables.
"""

from dataclasses import replace

import numpy as np
import pytest

from qf_research.optimal_stopping.analytical import black_scholes, black_scholes_greeks
from qf_research.optimal_stopping.contracts import Market, Option, exercise_times
from qf_research.optimal_stopping.finite_difference import finite_difference_price
from qf_research.optimal_stopping.least_squares import ExercisePolicy, Regression, evaluate_policy, train_policy
from qf_research.optimal_stopping.simulation import european_monte_carlo, simulate_gbm
from qf_research.optimal_stopping.tree import binomial_price


@pytest.mark.parametrize("spot", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("rate,dividend", [(0.05, 0.0), (-0.02, 0.03)])
def test_european_parity_and_bounds(spot, rate, dividend):
    m = Market(spot, rate, 0.3, dividend)
    call, put = Option(100, 1.7, "call"), Option(100, 1.7)
    c, p = black_scholes(m, call), black_scholes(m, put)
    forward_difference = spot * np.exp(-dividend * 1.7) - 100 * np.exp(-rate * 1.7)
    assert c - p == pytest.approx(forward_difference, abs=1e-12)
    assert max(forward_difference, 0) <= c <= spot * np.exp(-dividend * 1.7)
    assert max(-forward_difference, 0) <= p <= 100 * np.exp(-rate * 1.7)


@pytest.mark.parametrize("kind,price", [("call", 10.450583572185565), ("put", 5.573526022256971)])
def test_standard_analytical_benchmark(kind, price):
    assert black_scholes(Market(100, 0.05, 0.2), Option(100, 1, kind)) == pytest.approx(price, abs=1e-12)


@pytest.mark.parametrize("kind", ["put", "call"])
def test_greeks_against_price_perturbations(kind):
    m, o = Market(107, 0.03, 0.27, 0.012), Option(103, 1.4, kind)
    g = black_scholes_greeks(m, o)
    def derivative(field, h):
        return (black_scholes(replace(m, **{field: getattr(m, field) + h}), o)
                - black_scholes(replace(m, **{field: getattr(m, field) - h}), o)) / (2 * h)
    assert g.delta == pytest.approx(derivative("spot", 1e-3), abs=1e-8)
    assert g.vega == pytest.approx(derivative("volatility", 1e-5), rel=1e-8)
    assert g.rho == pytest.approx(derivative("rate", 1e-5), rel=1e-8)
    h = 0.01
    gamma = (black_scholes(replace(m, spot=m.spot + h), o) - 2 * black_scholes(m, o)
             + black_scholes(replace(m, spot=m.spot - h), o)) / h**2
    assert g.gamma == pytest.approx(gamma, rel=1e-6)
    theta = (black_scholes(m, replace(o, maturity=o.maturity - 1e-5))
             - black_scholes(m, replace(o, maturity=o.maturity + 1e-5))) / 2e-5
    assert g.theta == pytest.approx(theta, rel=1e-8)


@pytest.mark.parametrize("kind", ["put", "call"])
@pytest.mark.parametrize("rate", [-0.03, 0.06])
def test_deterministic_and_expiry_limits(kind, rate):
    m, o = Market(90, rate, 0, 0.01), Option(100, 1, kind)
    expected = np.exp(-rate) * o.payoff(90 * np.exp(rate - 0.01))
    assert black_scholes(m, o) == pytest.approx(expected)
    assert black_scholes(m, replace(o, maturity=0)) == o.payoff(90)
    dates = np.array([0.25, 0.5, 0.75, 1])
    best = max(np.exp(-rate * dates) * o.payoff(90 * np.exp((rate - 0.01) * dates)))
    assert binomial_price(m, o, 40, dates).value == pytest.approx(best)


def test_exact_simulation_discounted_moments():
    m = Market(97, 0.04, 0.25, 0.015)
    times = np.array([0, 0.1, 0.5, 1.3])
    paths = simulate_gbm(m, times, 100_000, np.random.default_rng(412))
    adjusted = paths * np.exp(-(m.rate - m.dividend) * times)
    for column in range(1, len(times)):
        se = adjusted[:, column].std(ddof=1) / np.sqrt(len(paths))
        assert abs(adjusted[:, column].mean() - m.spot) < 6 * se
    log_returns = np.log(paths[:, 1:] / paths[:, :-1])
    assert np.max(np.abs(np.corrcoef(log_returns.T) - np.eye(3))) < 0.015
    assert np.all(paths[:, 0] == m.spot)


def test_antithetic_uncertainty_uses_independent_pairs():
    m, o = Market(100, 0, 0.2), Option(100, 1, "call")
    n, seed = 2000, 812
    z = np.random.default_rng(seed).standard_normal(n // 2)
    paired = 0.5 * (o.payoff(100 * np.exp(-0.02 + 0.2 * z))
                    + o.payoff(100 * np.exp(-0.02 - 0.2 * z)))
    result = european_monte_carlo(m, o, n, np.random.default_rng(seed), antithetic=True)
    assert result.observations == n // 2
    assert result.standard_error == pytest.approx(paired.std(ddof=1) / np.sqrt(n // 2))
    assert result.value == pytest.approx(paired.mean())


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_analytical_agreement(antithetic):
    m, o = Market(100, 0.05, 0.2), Option(100, 1)
    result = european_monte_carlo(m, o, 100_000, np.random.default_rng(199), antithetic)
    assert abs(result.value - black_scholes(m, o)) < 6 * result.standard_error


@pytest.mark.parametrize("kind", ["put", "call"])
@pytest.mark.parametrize("rate,dividend", [(0.05, 0), (-0.01, 0.04)])
def test_tree_and_pde_against_analytical_solution(kind, rate, dividend):
    m, o = Market(100, rate, 0.2, dividend), Option(100, 1, kind)
    exact = black_scholes(m, o)
    assert abs(binomial_price(m, o, 2000).value - exact) < 0.003
    pde = finite_difference_price(m, o, 800, 960, 400)
    assert abs(pde.value - exact) < 0.002


def test_no_early_exercise_value_for_non_dividend_call():
    m, o = Market(100, 0.05, 0.2), Option(100, 1, "call")
    dates = np.arange(1, 13) / 12
    european = binomial_price(m, o, 960)
    bermudan = binomial_price(m, o, 960, dates)
    assert bermudan.value == pytest.approx(european.value, abs=1e-11)
    assert not bermudan.exercise_boundary


def test_nested_exercise_calendars_and_independent_solvers():
    m, o = Market(100, 0.05, 0.2), Option(100, 1)
    quarterly, monthly = np.arange(1, 5) / 4, np.arange(1, 13) / 12
    euro = binomial_price(m, o, 1920).value
    quarter = binomial_price(m, o, 1920, quarterly).value
    month = binomial_price(m, o, 1920, monthly).value
    assert euro < quarter < month
    pde = finite_difference_price(m, o, 800, 1920, 400, monthly)
    assert abs(month - pde.value) < 0.005
    assert len(pde.exercise_boundary) == 11


def test_pde_spatial_refinement_and_domain_sensitivity():
    m, o = Market(100, 0.05, 0.2), Option(100, 1)
    exact = black_scholes(m, o)
    coarse = finite_difference_price(m, o, 100, 960, 400)
    fine = finite_difference_price(m, o, 400, 960, 400)
    wider = finite_difference_price(m, o, 500, 960, 500)
    assert abs(fine.value - exact) < abs(coarse.value - exact) / 8
    assert abs(fine.value - wider.value) < 1e-5
    assert np.min(fine.values) >= -1e-10


def test_trained_policy_evaluates_against_reference():
    m, o = Market(100, 0.05, 0.2), Option(100, 1)
    dates = np.arange(1, 13) / 12
    train_seed, test_seed = np.random.SeedSequence(1881).spawn(2)
    policy = train_policy(m, o, dates, 40_000, 3, np.random.default_rng(train_seed))
    paths = simulate_gbm(m, np.array(policy.times), 80_000, np.random.default_rng(test_seed))
    result = evaluate_policy(policy, paths)
    ref = binomial_price(m, o, 3840, dates).value
    assert result.european_control.value > black_scholes(m, o)
    assert abs(result.european_control.value - ref) < 0.04 + 6 * result.european_control.standard_error
    assert np.all(result.stopping_indices > 0)
    for fit in policy.regressions[1:-1]:
        assert fit is not None and fit.rank == len(fit.coefficients)


def test_policy_decision_does_not_use_future_path():
    m, o = Market(100, 0.03, 0.2), Option(100, 1)
    fit = Regression(100, 1, (4.0,), 10, 1, 1)
    policy = ExercisePolicy(m, o, (0, 0.5, 1), (None, fit, None), 0, 10)
    # Identical observed states, opposite future outcomes. Both must exercise now.
    paths = np.array([[100, 90, 200], [100, 90, 20]], dtype=float)
    evaluated = evaluate_policy(policy, paths)
    assert np.array_equal(evaluated.stopping_indices, [1, 1])
    assert np.allclose(evaluated.discounted_payoffs, 10 * np.exp(-0.03 * 0.5))


def test_european_control_is_exact_when_there_is_no_early_exercise():
    m, o = Market(100, 0.03, 0.2), Option(100, 1)
    policy = train_policy(m, o, None, 10, 2, np.random.default_rng(1))
    paths = simulate_gbm(m, np.array(policy.times), 1000, np.random.default_rng(2))
    result = evaluate_policy(policy, paths)
    assert result.european_control.value == pytest.approx(black_scholes(m, o))
    assert result.european_control.standard_error < 1e-14
    assert result.raw.standard_error > 0


def test_rank_deficiency_reduces_degree():
    m, o = Market(80, 0.05, 0), Option(100, 1)
    policy = train_policy(m, o, [0.5, 1], 10, 5, np.random.default_rng(5))
    assert len(policy.regressions[1].coefficients) == 1
    paths = simulate_gbm(m, np.array(policy.times), 10, np.random.default_rng(6))
    result = evaluate_policy(policy, paths)
    reference = binomial_price(m, o, 10, [0.5, 1]).value
    assert result.raw.value == pytest.approx(reference)


@pytest.mark.parametrize("dates", [[0, 1], [0.5, 0.5], [1.1], [np.nan]])
def test_invalid_exercise_dates(dates):
    with pytest.raises(ValueError):
        exercise_times(Option(100, 1), dates)


@pytest.mark.parametrize("solver", ["tree", "pde"])
def test_misaligned_dates_are_rejected(solver):
    m, o = Market(100, 0.05, 0.2), Option(100, 1)
    with pytest.raises(ValueError, match="align"):
        if solver == "tree":
            binomial_price(m, o, 10, [1 / 3, 1])
        else:
            finite_difference_price(m, o, 100, 10, 400, [1 / 3, 1])


def test_invalid_crr_probability_is_rejected():
    with pytest.raises(ValueError, match="no-arbitrage"):
        binomial_price(Market(100, 0.5, 0.01), Option(100, 1), 1)


@pytest.mark.parametrize("kwargs", [{"spot": 0}, {"volatility": -1}, {"rate": np.inf}])
def test_invalid_market_inputs(kwargs):
    params = dict(spot=100, rate=0.05, volatility=0.2)
    params.update(kwargs)
    with pytest.raises(ValueError):
        Market(**params)
