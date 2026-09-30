"""Declared numerical fixtures, not empirical market observations."""

from dataclasses import replace
import math

import numpy as np
import pytest

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, deterministic_value, european_reference,
)
from qf_research.optimal_stopping.funded_policy import (
    ContinuationRule, FundedPaths, _fit, simulate_paths, prefix_paths,
    fit_policy, train_policy, realise_policy, calibrate_control, evaluate_policy,
)
from qf_research.optimal_stopping.simulation import estimate_samples


def fixture(sigma=.2):
    model = FundedCashModel(100., sigma, .99, .98, (DividendCap(.37, 4.),))
    rights = tuple(FundedExercise(t, math.exp(-.04*(t+.01))) for t in (.2, .36, .7, 1.))
    return model, Option(100., 1., "put"), rights


def paths(seed, n=1000, terms=None):
    return simulate_paths(*(terms or fixture()), n, np.random.default_rng(seed))


def test_exact_nonuniform_diffusion_matches_independently_constructed_increments():
    model, option, rights = fixture()
    sample = paths(192, 40)
    rng = np.random.default_rng(192)
    current, previous, expected = np.full(40, 99.), 0., []
    payments = None
    for t in (.2, .36, .37, .7, 1.):
        dt = t-previous
        current *= np.exp(-.5*.2**2*dt+.2*np.sqrt(dt)*rng.standard_normal(40))
        if t == .37:
            payments = np.minimum(current, 4.)
            current -= payments
        else:
            expected.append(current.copy())
        previous = t
    np.testing.assert_array_equal(sample.states, np.array(expected).T)
    np.testing.assert_array_equal(sample.detached_payments[:, 0], payments)
    assert not sample.states.flags.writeable


def test_same_time_phases_cap_and_absorption():
    model = FundedCashModel(100., 0., .99, .98, (DividendCap(.5, 120.), DividendCap(.8, 4.)))
    rights = (FundedExercise(0., 1.), FundedExercise(.5, .98, "before"),
              FundedExercise(.5, .98), FundedExercise(1., .96))
    sample = paths(11, 10, (model, Option(100., 1.), rights))
    np.testing.assert_array_equal(sample.states, np.tile([99., 99., 0., 0.], (10, 1)))
    np.testing.assert_array_equal(sample.detached_payments, np.tile([99., 0.], (10, 1)))
    np.testing.assert_array_equal(sample.total_gain, np.full(10, 99.))
    broken = sample.states.copy()
    broken[:, 2] = 1.
    with pytest.raises(ValueError, match="Same-time"):
        fit_policy(Option(100., 1.), replace(sample, states=broken), 2)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-.04, .04])
def test_european_mc_against_analytical_reference(kind, rate):
    model = FundedCashModel(100., .2, math.exp(-rate*.01), .98)
    option, rights = Option(100., 1., kind), (FundedExercise(1., math.exp(-rate*1.01)),)
    terms = model, option, rights
    policy = fit_policy(option, paths(20, terms=terms), 2)
    control = calibrate_control(policy, paths(21, 10000, terms))
    result = evaluate_policy(policy, paths(22, 100000, terms), control=control)
    expected = european_reference(model, option, rights[-1]).present_value
    assert abs(result.controlled_pv.value-expected) < 5*result.controlled_pv.standard_error
    assert result.controlled_pv.standard_error < result.raw_pv.standard_error
    assert result.controlled_quote.value == pytest.approx(result.controlled_pv.value/.98)


def test_multi_event_total_gain_is_a_martingale_with_material_liquidation():
    model, option, rights = fixture(.6)
    model = replace(model, dividends=(DividendCap(.37, 65.), DividendCap(.8, 30.)))
    sample = paths(312, 150000, (model, option, rights))
    estimate = estimate_samples(sample.total_gain)
    assert abs(estimate.value-model.initial_state) < 5*estimate.standard_error
    assert .3 < np.mean(sample.states[:, -1] == 0) < .8
    assert np.all(sample.states[sample.states[:, -2] == 0, -1] == 0)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("rate", [-.04, .04])
def test_deterministic_policy_no_second_discount(kind, rate):
    model, _, rights = fixture(0.)
    model = replace(model, quoted_spot=110.)
    option = Option(100., 1., kind)
    rights = tuple(replace(d, settlement_factor=math.exp(-rate*(d.time+.01))) for d in rights)
    terms = model, option, rights
    policy = fit_policy(option, paths(30, 100, terms), 4)
    control = calibrate_control(policy, paths(31, 100, terms))
    result = evaluate_policy(policy, paths(32, 100, terms), control=control)
    assert result.raw_pv.value == pytest.approx(deterministic_value(model, option, rights), abs=1e-12)
    assert control.coefficient == 0.
    assert result.controlled_pv.standard_error < 1e-14


def test_frozen_policy_does_not_use_future_states_or_payments_to_stop():
    model, option, rights = fixture()
    policy = fit_policy(option, paths(1), 2)
    force_exercise = ContinuationRule(0., 1., (0.,), 100, 1, 1., 0., 200., "fixture")
    policy = replace(policy, rules=(force_exercise,)+policy.rules[1:])
    sample = paths(2)
    states = sample.states.copy()
    states[:, 1:] = 80.
    alternative = replace(sample, states=states, detached_payments=np.zeros_like(sample.detached_payments))
    a, b = realise_policy(policy, sample), realise_policy(policy, alternative)
    early = a.stopping_indices == 0
    assert early.sum() > 100
    np.testing.assert_array_equal(b.stopping_indices[early], a.stopping_indices[early])
    np.testing.assert_array_equal(b.discounted_payoffs[early], a.discounted_payoffs[early])


def test_separate_training_pilot_and_evaluation_streams_enforced():
    training, pilot, evaluation = paths(1), paths(2), paths(3)
    policy = fit_policy(Option(100., 1., "put"), training, 2)
    with pytest.raises(ValueError, match="Training stream"):
        evaluate_policy(policy, training)
    with pytest.raises(ValueError, match="Training stream"):
        calibrate_control(policy, training)
    control = calibrate_control(policy, pilot)
    with pytest.raises(ValueError, match="pilot stream"):
        evaluate_policy(policy, pilot, control=control)
    with pytest.raises(ValueError, match="another frozen policy"):
        evaluate_policy(replace(policy, requested_degree=3), evaluation, control=control)
    result = evaluate_policy(policy, evaluation, control=control)
    np.testing.assert_allclose(result.controlled_payoffs,
                               result.discounted_payoffs-control.coefficient*(evaluation.total_gain-99.))


def test_identical_policies_have_zero_paired_difference():
    policy = fit_policy(Option(100., 1., "put"), paths(1), 2)
    a = evaluate_policy(policy, paths(2)).discounted_payoffs
    b = evaluate_policy(policy, paths(2)).discounted_payoffs
    difference = estimate_samples(a-b)
    assert difference.value == difference.standard_error == 0.
    assert estimate_samples(a).standard_error > 0.


def test_prefix_preserves_stream_and_nests_training_sample():
    sample = paths(43, 2000)
    prefix = prefix_paths(sample, 500)
    assert prefix.stream_id == sample.stream_id
    np.testing.assert_array_equal(prefix.states, sample.states[:500])
    with pytest.raises(ValueError, match="exceeds"):
        prefix_paths(sample, 3000)


def test_insufficient_itm_continues_and_reports_it():
    terms = fixture(0.)
    policy = fit_policy(terms[1], paths(4, 20, terms), 4, minimum_itm_paths=32)
    assert all(r.status == "continue_insufficient_itm" for r in policy.rules[:-1])
    outcome = realise_policy(policy, paths(5, 20, terms))
    assert np.all(outcome.stopping_indices == len(terms[2])-1)
    assert sum(r["no_fit_continues"] for r in outcome.diagnostics) > 0


def test_regression_reduces_ill_conditioned_basis_and_handles_constant_states():
    x = np.linspace(80., 120., 100)
    rule = _fit(x, 3*x+1, 8, 32, 1.01)
    assert rule.status == "reduced_degree" and rule.rank <= 2
    np.testing.assert_allclose(rule.predict(x), 3*x+1, atol=1e-10)
    constant = _fit(np.full(100, 90.), np.arange(100.), 4, 32, 1e8)
    assert constant.status == "constant_state"
    assert constant.predict(90.) == 49.5


def test_negative_continuation_clipped_and_extrapolation_counted():
    model, option, rights = fixture()
    policy = fit_policy(option, paths(1), 2)
    bad = ContinuationRule(99., 1., (-2.,), 100, 1, 1., 99., 99., "fixture")
    policy = replace(policy, rules=(bad,)+policy.rules[1:])
    result = realise_policy(policy, paths(2))
    first = result.diagnostics[0]
    assert first["negative_predictions"] == first["exercises"] > 0
    assert first["outside_training_support"] == first["candidates"]


@pytest.mark.parametrize("generator", [np.random.PCG64, np.random.MT19937, np.random.Philox])
def test_generator_provenance_is_reproducible(generator):
    a = simulate_paths(*fixture(), 100, np.random.Generator(generator(7)))
    b = simulate_paths(*fixture(), 100, np.random.Generator(generator(7)))
    assert a.stream_id == b.stream_id
    np.testing.assert_array_equal(a.states, b.states)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, maturity=.5),
    lambda p: replace(p, states=p.states[:, :-1]),
    lambda p: replace(p, states=-p.states),
    lambda p: replace(p, detached_payments=p.detached_payments+100.),
    lambda p: replace(p, stream_id=""),
])
def test_invalid_path_contract_rejected(change):
    with pytest.raises(ValueError):
        fit_policy(fixture()[1], change(paths(6)), 2)


def test_evaluation_model_mismatch_rejected():
    model, option, rights = fixture()
    policy = train_policy(model, option, rights, 1000, 2, np.random.default_rng(5))
    with pytest.raises(ValueError, match="model and rights"):
        evaluate_policy(policy, paths(6, terms=(replace(model, premium_factor=1.), option, rights)))
