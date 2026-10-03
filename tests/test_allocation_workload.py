"""Declared financial fixtures for the independent paired-workload kernel."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from qf_research.optimal_stopping import allocation_workload as workload
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise,
)
from qf_research.optimal_stopping.funded_policy import fit_policy, simulate_paths
from qf_research.optimal_stopping.training_evaluation import policy_gain
from qf_research.optimal_stopping.value_martingale import (
    evaluate_value_dual, observation_rights, value_martingale,
)


def fixture():
    model = FundedCashModel(100., .25, 1., .98, (DividendCap(.4, 3.),))
    option = Option(100., 1., 'put')
    rights = (FundedExercise(.2, .99), FundedExercise(.4, .98, 'before'),
              FundedExercise(.4, .98), FundedExercise(.7, .97),
              FundedExercise(1., .96))
    paths = simulate_paths(model, option, rights, 100, np.random.default_rng(930))
    base = fit_policy(option, paths, 2)
    seeds = tuple(np.random.SeedSequence(931).spawn(2))
    return base, seeds


def small_pair():
    base, seeds = fixture()
    policies, record = workload.train_pair(base, (200, 500), degree=2, seeds=seeds)
    approximation, setup = workload.build_approximation(
        base, core_steps=30, wing_steps=6, tolerance=.02)
    return base, seeds, policies, record, approximation, setup


def without_timings(value):
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if not key.endswith('_seconds') and key != 'phases'}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def test_training_matches_independent_original_fits_and_records_json():
    base, seeds, policies, record, _, setup = small_pair()
    for count, seed, policy, retained in zip((200, 500), seeds, policies, record['policies']):
        paths = simulate_paths(base.model, base.option, base.decisions, count,
                               np.random.Generator(np.random.PCG64(seed)))
        expected = fit_policy(base.option, paths, 2)
        assert policy == expected
        assert retained == json.loads(json.dumps(asdict(expected)))
    assert len({row['stream_id'] for row in record['streams']}) == 2
    assert record['training_paths'] == [200, 500]
    for row in (record, setup):
        assert row['cpu_seconds'] > 0 and row['elapsed_seconds'] > 0
        assert json.loads(json.dumps(row, allow_nan=False)) == row
    assert sum(record['phases'][key] for key in ('simulation_cpu_seconds', 'fitting_cpu_seconds')) <= record['cpu_seconds']


def test_shared_sample_gain_matches_original_primitives(monkeypatch):
    _, _, policies, _, approximation, _ = small_pair()
    seed = np.random.SeedSequence(932)
    observations = simulate_paths(policies[0].model, policies[0].option,
        observation_rights(approximation), 400, np.random.Generator(np.random.PCG64(seed)))
    sample = value_martingale(approximation, observations)
    results = tuple(evaluate_value_dual(policy, sample) for policy in policies)
    expected = policy_gain(*results, policies[0].model.premium_factor)

    seen = []
    original = workload.evaluate_value_dual
    def observed(policy, shared):
        seen.append(shared)
        return original(policy, shared)
    monkeypatch.setattr(workload, 'evaluate_value_dual', observed)
    record = workload.evaluate_pair(policies, approximation, 400, seed)
    assert len(seen) == 2 and seen[0] is seen[1]
    assert record['gain'] == expected
    assert record['stream']['stream_id'] == observations.stream_id
    assert record['approximation_fingerprint'] == approximation.fingerprint
    assert record['nonzero_gain_count'] == np.count_nonzero(
        results[1].adjusted_policy_payoffs-results[0].adjusted_policy_payoffs)
    assert all(value >= 0 for value in record['minimum_gap_quote'])
    assert record['cpu_seconds'] > 0 and record['elapsed_seconds'] > 0
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_non_timing_records_reproduce_exactly():
    base, seeds, policies, training, approximation, setup = small_pair()
    other, other_training = workload.train_pair(base, (200, 500), degree=2, seeds=seeds)
    other_approximation, other_setup = workload.build_approximation(
        base, core_steps=30, wing_steps=6, tolerance=.02)
    first = workload.evaluate_pair(policies, approximation, 128, np.random.SeedSequence(933))
    second = workload.evaluate_pair(other, other_approximation, 128, np.random.SeedSequence(933))
    for left, right in ((training, other_training), (setup, other_setup), (first, second)):
        assert without_timings(left) == without_timings(right)


def test_stream_reuse_and_invalid_seed_types_are_rejected():
    base, seeds, policies, _, approximation, _ = small_pair()
    with pytest.raises(ValueError, match='Independent training'):
        workload.train_pair(base, (200, 500), seeds=(seeds[0], seeds[0]))
    with pytest.raises(ValueError, match='base policy training'):
        workload.train_pair(base, (200, 500), seeds=(np.random.SeedSequence(930), seeds[1]))
    with pytest.raises(ValueError, match='SeedSequence'):
        workload.train_pair(base, (200, 500), seeds=(1, 2))
    for seed in seeds:
        with pytest.raises(ValueError, match='either training stream'):
            workload.evaluate_pair(policies, approximation, 128, seed)
    with pytest.raises(ValueError, match='Independent training'):
        workload.evaluate_pair((policies[0], policies[0]), approximation, 128,
                               np.random.SeedSequence(934))


@pytest.mark.parametrize('change', ['model', 'option', 'calendar'])
def test_problem_mismatch_is_rejected(change):
    _, _, policies, _, approximation, _ = small_pair()
    if change == 'model':
        altered = replace(policies[1], model=replace(policies[1].model, volatility=.3))
    elif change == 'option':
        altered = replace(policies[1], option=replace(policies[1].option, strike=101.))
    else:
        altered = replace(policies[1], decisions=policies[1].decisions[1:])
    with pytest.raises(ValueError, match='share model, option and calendar'):
        workload.evaluate_pair((policies[0], altered), approximation, 128,
                               np.random.SeedSequence(935))
    with pytest.raises(ValueError, match='share model, option and calendar'):
        workload.evaluate_pair(policies, replace(approximation, option=Option(101., 1., 'put')),
                               128, np.random.SeedSequence(936))


def test_terminal_only_pair_retains_zero_gain_diagnostic():
    model = FundedCashModel(100., .2, 1., .98)
    option = Option(100., 1., 'put')
    rights = (FundedExercise(1., .97),)
    paths = simulate_paths(model, option, rights, 20, np.random.default_rng(940))
    base = fit_policy(option, paths, 2)
    policies, _ = workload.train_pair(base, (30, 50), degree=2,
                                     seeds=tuple(np.random.SeedSequence(941).spawn(2)))
    approximation, _ = workload.build_approximation(base, core_steps=20, wing_steps=4)
    record = workload.evaluate_pair(policies, approximation, 32, np.random.SeedSequence(942))
    assert record['nonzero_gain_count'] == 0
    assert record['gain']['estimate']['value'] == 0.
    assert record['gain']['estimate']['standard_error'] == 0.
