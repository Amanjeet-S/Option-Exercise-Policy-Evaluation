"""Declared mathematical and timing fixtures, not market observations."""

from dataclasses import asdict
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ttest_ind

from qf_research.optimal_stopping.cost_budget import timing_model, allocate_paths, independent_difference
from qf_research.optimal_stopping.simulation import estimate_samples
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy, calibrate_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from study_cost_budget import measure, METHODS


def test_timing_model_uses_medians_and_charges_setup_once():
    model = timing_model([10000, 30000], [[.5, .51, 8.], [.9, .91, 12.]])
    assert model['setup_seconds'] == pytest.approx(.31)
    assert model['seconds_per_path'] == pytest.approx(.00002)
    allocated = allocate_paths(model, 1., maximum=100000)
    assert allocated['paths'] == 34000
    assert allocated['predicted_seconds'] <= 1.
    assert model['setup_seconds']+model['seconds_per_path']*(allocated['paths']+1000) > 1.


def test_negative_intercept_is_reported_and_conservatively_clipped():
    model = timing_model([10000, 30000], [[.1]*3, [.9]*3])
    assert model['unconstrained_setup_seconds'] < 0
    assert model['setup_seconds'] == 0


@pytest.mark.parametrize('timings', [[[1.]*3, [1.]*3], [[2.]*3, [1.]*3],
                                      [[1.]*2, [2.]*3], [[1., np.nan, 1.], [2.]*3]])
def test_unidentified_or_incomplete_timing_pilot_rejected(timings):
    with pytest.raises(ValueError):
        timing_model([10000, 30000], timings)


def test_allocation_caps_and_infeasibility_are_explicit():
    model = dict(setup_seconds=.1, seconds_per_path=.00001)
    capped = allocate_paths(model, 100., maximum=20500)
    assert capped['paths'] == 20000 and capped['capped']
    assert allocate_paths(model, .11)['status'] == 'infeasible'
    assert allocate_paths(model, .05)['paths'] is None


@pytest.mark.parametrize('sizes', [[2., 4], [True, 100], [100, 100], [100, 2]])
def test_invalid_pilot_sizes_rejected(sizes):
    with pytest.raises(ValueError):
        timing_model(sizes, [[1.]*3, [2.]*3])


def test_independent_difference_matches_scipy_welch_with_unequal_sizes():
    right, left = np.array([1., 4., 9., 16., 25.]), np.array([.1, .5, .8, 1., 1.2, 3., 12.])
    result = independent_difference(asdict(estimate_samples(right)), asdict(estimate_samples(left)))
    reference = ttest_ind(right, left, equal_var=False)
    interval = reference.confidence_interval(.95)
    assert result['lower'] == pytest.approx(interval.low)
    assert result['upper'] == pytest.approx(interval.high)
    assert result['degrees_of_freedom'] == pytest.approx(reference.df)
    assert result['standard_error']**2 == pytest.approx(right.var(ddof=1)/len(right)+left.var(ddof=1)/len(left))


def test_zero_variance_comparison_and_unit_scaling():
    x = asdict(estimate_samples(np.array([2., 2., 2.])))
    y = asdict(estimate_samples(np.array([1., 1.])))
    assert independent_difference(x, y)['lower'] == 1.
    z = asdict(estimate_samples(np.array([1., 3., 8.])))
    r = independent_difference(x, z)
    scaled = [{**e, 'value': e['value']*7, 'standard_error': e['standard_error']*7} for e in (x, z)]
    s = independent_difference(*scaled)
    assert s['upper'] == pytest.approx(r['upper']*7)
    assert s['degrees_of_freedom'] == pytest.approx(r['degrees_of_freedom'])


@pytest.mark.parametrize('method', METHODS)
def test_complete_workload_repeats_numerically_and_preserves_policy(method):
    model = FundedCashModel(100., .25, 1., .98, (DividendCap(.4, 3.),))
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.2, .99), FundedExercise(.4, .98, 'before'), FundedExercise(1., .96))
    training = simulate_paths(model, option, rights, 1000, np.random.default_rng(712))
    policy = fit_policy(option, training, 2)
    pilot = simulate_paths(model, option, rights, 1000, np.random.default_rng(713))
    control = calibrate_control(policy, pilot)
    entry = dict(policy=asdict(policy), control=asdict(control), fingerprint=policy.fingerprint)
    task = dict(case=0, prior_rows=[dict(repetition=0, policies=[entry],
        streams=[dict(stream_id=p.stream_id) for p in (training, pilot)])],
        configuration=dict(approximations={n: dict(core_steps=40, wing_steps=8, tolerance=.01)
                                           for n in ('coarse', 'fine')}))
    rows = [measure(task, method, 500, np.random.SeedSequence(714)) for _ in range(2)]
    assert rows[0]['policies'] == rows[1]['policies']
    assert rows[0]['policies'][0]['fingerprint'] == policy.fingerprint
    assert rows[0]['policies'][0]['minimum_gap_quote'] >= 0
    assert rows[0]['paths'] == 500
    assert sum(rows[0]['phases'].values()) <= rows[0]['elapsed_seconds']
    for p in rows[0]['policies']:
        e = p['estimates_quote']
        assert e['gap_bound']['value'] == pytest.approx(e['upper_value']['value']-e['policy_value']['value'])
