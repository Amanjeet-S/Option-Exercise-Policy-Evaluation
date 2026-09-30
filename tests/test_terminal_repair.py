"""Feasibility, conditional integration and paired terminal-policy comparisons."""

from dataclasses import replace
import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.optimal_stopping.funded_policy import fit_policy, simulate_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale, evaluate_value_dual
from qf_research.optimal_stopping.expiry_extension import extended_calendar
from qf_research.optimal_stopping.terminal_repair import evaluate_terminal_repair


def setup(extension=.1, volatility=.2, terminal_only=False):
    model = FundedCashModel(100., volatility, 1., 1.)
    option = Option(100., 1., 'put')
    rights = (FundedExercise(1., 1.),) if terminal_only else (FundedExercise(.5, 1.01), FundedExercise(1., 1.))
    policy = fit_policy(option, simulate_paths(model, option, rights, 1000, np.random.default_rng(11)), 2)
    contract, calendar = extended_calendar(model, option, rights, (extension,)) if extension else (option, rights)
    values = build_values(model, contract, calendar, core_steps=100)
    paths = simulate_paths(model, contract, observation_rights(values), 4000, np.random.default_rng(12))
    return policy, value_martingale(values, paths)


def test_zero_extension_recovers_original_dual_exactly():
    policy, sample = setup(0.)
    repaired = evaluate_terminal_repair(policy, sample, 0.)
    reference = evaluate_value_dual(policy, sample)
    for record in (repaired.original, repaired.repaired, repaired.direct_repaired):
        np.testing.assert_array_equal(record.gap_payoffs, reference.gap_payoffs)
    np.testing.assert_array_equal(repaired.integrated_gain, 0.)


def test_policy_gain_is_nonnegative_and_only_terminal_survivors_change():
    policy, sample = setup()
    result = evaluate_terminal_repair(policy, sample, .1)
    assert 0 < result.survives.sum() < len(result.survives)
    assert result.integrated_gain.min() >= 0
    np.testing.assert_array_equal(result.integrated_gain[~result.survives], 0.)
    np.testing.assert_array_equal(result.realised_gain[~result.survives], 0.)
    np.testing.assert_allclose(result.original.gap_payoffs-result.repaired.gap_payoffs,
                               result.integrated_gain, rtol=0, atol=3e-14)
    assert np.any(result.realised_gain < 0)  # Expected improvement is not pathwise profit.


def test_integrated_gain_does_not_use_future_draws():
    policy, sample = setup()
    result = evaluate_terminal_repair(policy, sample, .1)
    states = sample.paths.states.copy(); states[:, -1] *= 1.2
    changed = evaluate_terminal_repair(policy, replace(sample, paths=replace(sample.paths, states=states)), .1)
    np.testing.assert_array_equal(result.stopping_indices, changed.stopping_indices)
    np.testing.assert_array_equal(result.integrated_gain, changed.integrated_gain)
    np.testing.assert_array_equal(result.repaired.gap_payoffs, changed.repaired.gap_payoffs)
    assert not np.array_equal(result.realised_gain, changed.realised_gain)


def test_zero_volatility_has_no_policy_gain():
    policy, sample = setup(volatility=0.)
    result = evaluate_terminal_repair(policy, sample, .1)
    np.testing.assert_array_equal(result.integrated_gain, 0.)
    np.testing.assert_array_equal(result.realised_gain, 0.)


def test_terminal_only_value_matches_independent_quadrature():
    policy, sample = setup(terminal_only=True)
    # Integrate the terminal conditional payoff over the original GBM law.
    from scipy.integrate import quad
    from scipy.stats import norm
    from qf_research.optimal_stopping.expiry_extension import tail_put
    f = lambda z: float(tail_put(100*np.exp(-.02+.2*z), 100., .2, .1))*norm.pdf(z)
    expected, error = quad(f, -10., 10., epsabs=1e-10)
    analytic = black_scholes(Market(100., 0., .2), Option(100., 1.1, 'put'))
    assert expected == pytest.approx(analytic, abs=2e-10)
    result = evaluate_terminal_repair(policy, sample, .1)
    assert result.survives.all()


@pytest.mark.parametrize('extension', [-.1, float('nan'), True])
def test_invalid_extension_rejected(extension):
    policy, sample = setup()
    with pytest.raises(ValueError):
        evaluate_terminal_repair(policy, sample, extension)


def test_changed_settlement_and_wrong_endpoint_rejected():
    policy, sample = setup()
    with pytest.raises(ValueError, match='matching extended'):
        evaluate_terminal_repair(policy, sample, .2)
    rights = sample.paths.decisions[:-1]+(replace(sample.paths.decisions[-1], settlement_factor=.9),)
    with pytest.raises(ValueError, match='matching extended'):
        evaluate_terminal_repair(policy, replace(sample, paths=replace(sample.paths, decisions=rights)), .1)


def test_cash_in_added_interval_and_changed_model_rejected():
    policy, sample = setup()
    changed_model = replace(policy.model, dividends=(DividendCap(1.05, 1.),))
    with pytest.raises(ValueError, match='unchanged put'):
        evaluate_terminal_repair(policy, replace(sample, paths=replace(sample.paths, model=changed_model)), .1)
    with pytest.raises(ValueError, match='no cash transfer'):
        evaluate_terminal_repair(replace(policy, model=changed_model),
            replace(sample, paths=replace(sample.paths, model=changed_model)), .1)


def test_training_paths_cannot_be_reused_for_policy_evaluation():
    policy, sample = setup()
    with pytest.raises(ValueError, match='Training'):
        evaluate_terminal_repair(policy, replace(sample, paths=replace(sample.paths, stream_id=policy.training_stream_id)), .1)


def test_direct_and_integrated_gain_difference_is_centred_with_lower_variance():
    policy, sample = setup()
    result = evaluate_terminal_repair(policy, sample, .1)
    noise = result.realised_gain-result.integrated_gain
    assert abs(noise.mean()) < 5*noise.std(ddof=1)/np.sqrt(len(noise))
    assert result.integrated_gain.var() < result.realised_gain.var()
