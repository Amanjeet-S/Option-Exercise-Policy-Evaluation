"""Independent integer enumeration and calculus checks for pilot allocation."""

from fractions import Fraction
import json
import math

import pytest
from scipy.optimize import minimize_scalar

from qf_research.optimal_stopping.training_allocation import choose_allocation, continuous_optimum


@pytest.mark.parametrize('A,B,costs', [
    (3, 17, {1: 3, 2: 5, 4: 8, 8: 15}),
    (1, 32, {1: 8, 2: 5, 4: 7, 8: 6}),
    (5, 1, {1: 1, 2: 2, 4: 8, 8: 100}),
])
def test_measured_cost_selection_matches_exhaustive_exact_integer_oracle(A, B, costs):
    result = choose_allocation(A, B, 7, costs, reference_repetitions=4,
                               reference_paths=4, min_repetitions=2, max_repetitions=20)
    budget = 7+4*costs[4]
    choices = [(Fraction(A, R)+Fraction(B, R*N), N, R)
               for N in costs for R in range(2, 21) if 7+R*costs[N] <= budget]
    variance, N, R = min(choices)
    assert (result['selected']['evaluation_paths'], result['selected']['repetitions']) == (N, R)
    assert result['selected']['predicted_variance'] == pytest.approx(float(variance))
    assert result['budget']['cpu_seconds'] == budget
    assert result['reference']['repetitions'] == 4
    assert [row['evaluation_paths'] for row in result['candidates']] == sorted(costs)
    for row in result['candidates']:
        affordable = [r for r in range(21) if 7+r*costs[row['evaluation_paths']] <= budget]
        assert row['repetitions'] == max(affordable)
        assert row['feasible'] == (row['repetitions'] >= 2)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_exact_tie_uses_ascending_paths_independent_of_dictionary_order():
    # At N=1,R=4 and N=2,R=3 both objective values equal 1/2.
    result = choose_allocation(1, 1, 0, {2: 4, 1: 3},
                               reference_repetitions=3, reference_paths=2,
                               min_repetitions=2, max_repetitions=10)
    assert result['selected']['evaluation_paths'] == 1
    assert result['selected']['predicted_variance'] == .5


def test_nonmonotone_measured_costs_are_preserved_without_interpolation():
    costs = {128: 4., 512: .5, 4096: 2.}
    result = choose_allocation(1, 100, .25, costs)
    assert result['selected']['evaluation_paths'] == 512
    assert result['selected']['repetitions'] == 256
    assert {row['evaluation_paths']: row['pair_cpu_seconds'] for row in result['candidates']} == costs


def test_cap_unused_budget_and_infeasible_candidates_are_explicit():
    result = choose_allocation(1, 10, 3, {128: .001, 1024: 4., 4096: 1.})
    small, expensive, _ = result['candidates']
    assert small['repetitions'] == 1024 and small['capped']
    assert small['unused_cpu_seconds'] == pytest.approx(64-1.024)
    assert expensive['repetitions'] == 16 and not expensive['feasible']
    assert expensive['predicted_variance'] is None
    assert not expensive['capped']
    assert result['selected']['feasible']


def test_equal_to_cap_does_not_claim_that_cap_prevented_an_affordable_repetition():
    result = choose_allocation(1, 1, 0, {4096: 1.}, max_repetitions=64)
    assert result['selected']['repetitions'] == 64
    assert not result['selected']['capped']


def test_boundary_rounding_uses_small_float_allowance_but_rejects_real_overspending():
    # 0.1+0.2 lies above 0.3 in binary floating-point; the boundary pair fits.
    near = .1+.2
    result = choose_allocation(1, 1, .7, {1: near, 2: .3},
                               reference_repetitions=3, reference_paths=2,
                               min_repetitions=1, max_repetitions=10)
    first = result['candidates'][0]
    assert first['repetitions'] == 3
    assert first['total_cpu_seconds']-result['budget']['cpu_seconds'] <= result['budget']['tolerance_cpu_seconds']
    far = .3+64*math.ulp(.9)/3
    result = choose_allocation(1, 1, .7, {1: far, 2: .3},
                               reference_repetitions=3, reference_paths=2,
                               min_repetitions=1, max_repetitions=10)
    assert result['candidates'][0]['repetitions'] == 2


def test_large_setup_does_not_create_free_variable_budget_through_rounding():
    result = choose_allocation(1, 1, 1e20, {128: .01, 4096: .01})
    assert all(row['repetitions'] == 64 for row in result['candidates'])


def test_tiny_measured_cost_is_capped_without_overflowing_cost_quotient():
    result = choose_allocation(1, 1, 0, {1: 5e-324, 4096: 1.})
    assert result['candidates'][0]['repetitions'] == 1024
    assert result['candidates'][0]['capped']


@pytest.mark.parametrize('A,B', [(0., 1.), (-2., 1.), (1., 0.), (1., -3.), (-2., -3.)])
def test_nonpositive_variance_components_retain_raw_values_and_reference_fallback(A, B):
    result = choose_allocation(A, B, .5, {128: .1, 4096: 1.})
    assert result['status'] == 'unresolved_variance'
    assert result['raw_variance_components'] == dict(training=A, evaluation=B)
    assert result['selected'] == result['reference']
    assert result['selected']['repetitions'] == 64
    assert result['selected']['evaluation_paths'] == 4096
    assert all(row['predicted_variance'] is None for row in result['candidates'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('replacement', [
    {'A': math.nan}, {'A': math.inf}, {'A': True}, {'A': '1'},
    {'B': -math.inf}, {'setup_cpu': -1}, {'setup_cpu': math.nan},
    {'pair_costs': {}}, {'pair_costs': {128: 1}}, {'pair_costs': {4096: 0}},
    {'pair_costs': {4096: -1}}, {'pair_costs': {4096: math.inf}},
    {'pair_costs': {4096: math.nan}}, {'pair_costs': {4096: True}},
    {'pair_costs': {4096: '1'}}, {'pair_costs': {4096.: 1}},
    {'pair_costs': {True: 1, 4096: 1}}, {'pair_costs': {0: 1, 4096: 1}},
    {'pair_costs': [(4096, 1)]}, {'reference_repetitions': 63},
    {'reference_repetitions': 1025}, {'reference_repetitions': 64.},
    {'reference_paths': 0}, {'min_repetitions': 0}, {'max_repetitions': False},
    {'max_repetitions': 63}, {'pair_costs': {4096: 1e308}},
])
def test_invalid_allocation_inputs_raise_value_error(replacement):
    arguments = dict(A=1., B=1., setup_cpu=0., pair_costs={4096: 1.})
    arguments.update(replacement)
    with pytest.raises(ValueError):
        choose_allocation(**arguments)


@pytest.mark.parametrize('A,B,training,evaluation,budget,setup', [
    (2., 18., 8., .5, 200., 5.),
    (.01, 4., .1, .0002, 500., 20.),
    (8., .1, .02, 3., 2., 0.),
])
def test_continuous_diagnostic_matches_independent_numerical_optimisation(
        A, B, training, evaluation, budget, setup):
    result = continuous_optimum(A, B, training, evaluation, budget, setup)

    def objective(log_paths):
        N = math.exp(log_paths)
        R = (budget-setup)/(training+evaluation*N)
        return A/R+B/(R*N)

    numeric = minimize_scalar(objective, bounds=(-20., 20.), method='bounded',
                              options={'xatol': 1e-11})
    assert numeric.success
    assert result['evaluation_paths'] == pytest.approx(math.exp(numeric.x), rel=2e-6)
    assert result['minimum_variance'] == pytest.approx(numeric.fun, rel=1e-10)
    assert setup+result['repetitions']*(training+evaluation*result['evaluation_paths']) == pytest.approx(budget)
    closed_form = (math.sqrt(A*training)+math.sqrt(B*evaluation))**2/(budget-setup)
    assert result['minimum_variance'] == pytest.approx(closed_form)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('replacement', [
    {'A': 0}, {'A': -1}, {'A': math.nan}, {'B': 0}, {'B': math.inf},
    {'training_cpu': 0}, {'training_cpu': -1}, {'evaluation_cpu_per_path': 0},
    {'evaluation_cpu_per_path': math.nan}, {'budget': 0}, {'budget': 1},
    {'setup': -1}, {'setup': math.inf}, {'A': True},
])
def test_invalid_continuous_inputs_raise_value_error(replacement):
    arguments = dict(A=1., B=1., training_cpu=1., evaluation_cpu_per_path=1., budget=10., setup=1.)
    arguments.update(replacement)
    with pytest.raises(ValueError):
        continuous_optimum(**arguments)
