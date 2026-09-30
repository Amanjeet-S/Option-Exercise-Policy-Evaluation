"""Exact analytical fixtures for finite-grid cost sensitivity.

The inputs are stipulated arithmetic examples, not financial observations,
measured computational costs or additional simulation results.
"""

from fractions import Fraction

import pytest

from qf_research.optimal_stopping.allocation_sensitivity import choose, evaluate, frontier


def costs(setup=0, pair=1):
    return dict(setup=setup, pair=pair)


def brute_force(A, B, budget, scenarios, overhead=0, minimum=64, maximum=1024):
    """Enumerate every admissible pair without using the allocation kernel."""
    A, B, budget, overhead = map(Fraction, (A, B, budget, overhead))
    feasible = []
    for n in sorted(next(iter(scenarios.values()))):
        for r in range(minimum, maximum + 1):
            charges = {name: overhead + Fraction(row[n]['setup']) + r * Fraction(row[n]['pair'])
                       for name, row in scenarios.items()}
            if all(value <= budget for value in charges.values()):
                feasible.append(dict(evaluation_paths=n, repetitions=r,
                                     variance=(A + B / n) / r,
                                     scenario_costs=charges, required_budget=max(charges.values())))
    return min(feasible, key=lambda row: (row['variance'], row['evaluation_paths'], row['repetitions'])) if feasible else None


def winner(row):
    if row is None:
        return None
    return row['evaluation_paths'], row['repetitions'], row['variance']


def joint_thresholds(scenarios, overhead, minimum, maximum):
    return sorted({Fraction(overhead) + max(Fraction(table[n]['setup']) + r * Fraction(table[n]['pair'])
                                           for table in scenarios.values())
                   for n in next(iter(scenarios.values())) for r in range(minimum, maximum + 1)})


def varied_scenarios():
    return {
        'first': {1: costs(2, 1), 3: costs(0, Fraction(7, 4)), 8: costs(1, 3)},
        'second': {1: costs(0, Fraction(3, 2)), 3: costs(3, 1), 8: costs(Fraction(1, 2), Fraction(5, 2))},
    }


def test_choice_and_frontier_match_full_enumeration_at_and_between_all_thresholds():
    scenarios = varied_scenarios()
    A, B, H, minimum, maximum = Fraction(3, 5), Fraction(7, 2), Fraction(2, 3), 2, 9
    thresholds = joint_thresholds(scenarios, H, minimum, maximum)
    budgets = [thresholds[0] / 2, *thresholds,
               *((a + b) / 2 for a, b in zip(thresholds, thresholds[1:])), thresholds[-1] + 1]
    for budget in budgets:
        actual = choose(A, B, budget, scenarios, H, minimum, maximum)
        expected = brute_force(A, B, budget, scenarios, H, minimum, maximum)
        assert winner(actual['selected']) == winner(expected)
        assert actual['budget'] == budget
        if expected is not None:
            assert actual['selected']['scenario_costs'] == expected['scenario_costs']
    expected_frontier = []
    previous = None
    for budget in thresholds:
        selected = brute_force(A, B, budget, scenarios, H, minimum, maximum)
        if winner(selected) != winner(previous):
            expected_frontier.append((budget, winner(selected)))
            previous = selected
    actual_frontier = frontier(A, B, scenarios, H, minimum, maximum)
    assert [(row['required_budget'], winner(row)) for row in actual_frontier] == expected_frontier


def test_matched_scenario_costs_and_change_in_binding_scenario():
    scenarios = {'large_setup': {2: costs(100, 1)}, 'large_pair': {2: costs(0, 2)}}
    result = choose(1, 1, 164, scenarios)
    assert result['selected']['repetitions'] == 64
    assert result['selected']['required_budget'] == 164
    assert result['selected']['scenario_costs'] == {'large_setup': 164, 'large_pair': 128}
    assert 100 + 64 * 2 > result['budget']
    at_crossing = evaluate(1, 1, 2, 100, 200, scenarios)
    assert at_crossing['scenario_costs'] == {'large_setup': 200, 'large_pair': 200}
    later = choose(1, 1, 256, scenarios)
    assert later['selected']['repetitions'] == 128
    assert later['selected']['scenario_costs'] == {'large_setup': 228, 'large_pair': 256}


def test_minimum_repetition_boundary_is_inclusive_on_both_sides():
    scenarios = {'one': {4: costs(Fraction(1, 3), Fraction(2, 7))}}
    H, epsilon = Fraction(2, 5), Fraction(1, 2**80)
    boundary = H + Fraction(1, 3) + 64 * Fraction(2, 7)
    assert choose(1, 1, boundary - epsilon, scenarios, H)['selected'] is None
    for budget in (boundary, boundary + epsilon):
        assert choose(1, 1, budget, scenarios, H)['selected']['repetitions'] == 64


def test_interior_integer_boundary_is_inclusive_on_both_sides():
    scenarios = {'one': {4: costs(5, Fraction(3, 7))}}
    boundary, epsilon = 5 + 81 * Fraction(3, 7), Fraction(1, 2**80)
    assert choose(1, 1, boundary - epsilon, scenarios)['selected']['repetitions'] == 80
    assert choose(1, 1, boundary, scenarios)['selected']['repetitions'] == 81
    assert choose(1, 1, boundary + epsilon, scenarios)['selected']['repetitions'] == 81


def test_default_cap_and_absence_of_further_frontier_changes():
    scenarios = {'one': {1: costs()}}
    assert choose(1, 1, 1024, scenarios)['selected']['repetitions'] == 1024
    assert choose(1, 1, 10000, scenarios)['selected']['repetitions'] == 1024
    assert choose(1, 1, 10000, scenarios)['selected']['capped'] is True
    rows = frontier(1, 1, scenarios)
    assert len(rows) == 1024 - 64 + 1
    assert rows[-1]['required_budget'] == 1024


def test_floor_respects_exact_stored_binary64_cost():
    scenarios = {'one': {1: costs(0, .1)}}
    # Ten exact copies of stored binary64 0.1 exceed one, although 1./.1 rounds to ten.
    assert 10 * Fraction(.1) > 1
    result = choose(1, 1, 1, scenarios, minimum=1, maximum=20)
    assert result['selected']['repetitions'] == 9
    assert result['selected']['required_budget'] == 9 * Fraction(.1)


def test_infeasible_candidates_are_retained_without_invented_allocations():
    scenarios = {'one': {1: costs(100, 2), 2: costs(0, 3)}}
    result = choose(1, 1, 0, scenarios)
    assert result['status'] == 'no_feasible_allocation'
    assert result['selected'] is None
    assert all(row['repetitions'] == 0 and row['variance'] is None for row in result['candidates'])
    assert result['candidates'][0]['unconstrained_integer_limits']['one'] == -50
    below_minimum = choose(1, 1, 150, scenarios)
    assert below_minimum['selected'] is None
    assert all(not row['feasible'] for row in below_minimum['candidates'])


def test_only_feasible_candidate_can_be_selected():
    scenarios = {'one': {1: costs(0, 1), 100: costs(1000, 1)}}
    result = choose(1, 1000, 80, scenarios)
    assert result['selected']['evaluation_paths'] == 1
    assert result['selected']['repetitions'] == 80


def test_overhead_is_an_exact_horizontal_translation():
    scenarios = varied_scenarios()
    H = Fraction(29, 7)
    base_rows = frontier(1, 2, scenarios, minimum=2, maximum=9)
    shifted_rows = frontier(1, 2, scenarios, H, minimum=2, maximum=9)
    assert [winner(row) for row in base_rows] == [winner(row) for row in shifted_rows]
    assert [row['required_budget'] + H for row in base_rows] == [row['required_budget'] for row in shifted_rows]
    for budget in (0, 5, 10, 30):
        base = choose(1, 2, budget, scenarios, minimum=2, maximum=9)
        shifted = choose(1, 2, budget + H, scenarios, H, minimum=2, maximum=9)
        assert winner(base['selected']) == winner(shifted['selected'])


def test_equal_variance_tie_uses_smaller_evaluation_count():
    scenarios = {'one': {1: costs(0, 1), 2: costs(Fraction(1, 2), 1)}}
    before = choose(1, 1, Fraction(7, 2), scenarios, minimum=1, maximum=4)
    tied = choose(1, 1, 4, scenarios, minimum=1, maximum=4)
    assert winner(before['selected']) == (2, 3, Fraction(1, 2))
    assert winner(tied['selected']) == (1, 4, Fraction(1, 2))
    row = next(row for row in frontier(1, 1, scenarios, minimum=1, maximum=4)
               if row['required_budget'] == 4)
    assert row['strict_variance_improvement'] is False


def test_equal_cost_breakpoints_are_considered_together():
    scenarios = {'one': {1: costs(0, 1), 2: costs(1, 1)}}
    rows = frontier(1, 1, scenarios, minimum=1, maximum=4)
    matching = [row for row in rows if row['required_budget'] == 4]
    assert len(matching) == 1
    assert winner(matching[0]) == (1, 4, Fraction(1, 2))


def test_risk_is_nonincreasing_with_budget_and_best_grid_point_wins_at_all_caps():
    scenarios = varied_scenarios()
    values = [choose(1, 5, budget, scenarios, minimum=2, maximum=9)['selected']
              for budget in range(40)]
    variances = [row['variance'] for row in values if row is not None]
    assert all(b <= a for a, b in zip(variances, variances[1:]))
    final = choose(1, 5, 1000, scenarios, minimum=2, maximum=9)['selected']
    assert (final['evaluation_paths'], final['repetitions']) == (8, 9)


def test_joint_risk_cannot_improve_on_any_scenario_specific_optimum():
    scenarios = varied_scenarios()
    for budget in (7, 12, 20, 50):
        robust = choose(1, 3, budget, scenarios, minimum=2, maximum=9)['selected']
        for name, table in scenarios.items():
            individual = choose(1, 3, budget, {name: table}, minimum=2, maximum=9)['selected']
            if robust is not None:
                assert individual is not None
                assert robust['variance'] >= individual['variance']


def test_duplicate_and_dominated_scenarios_do_not_change_allocation():
    scenarios = varied_scenarios()
    augmented = dict(scenarios)
    augmented['duplicate'] = scenarios['first']
    augmented['cheaper'] = {n: costs(0, Fraction(1, 10)) for n in scenarios['first']}
    for budget in (0, 7, 12, 20, 50):
        assert winner(choose(1, 3, budget, scenarios, minimum=2, maximum=9)['selected']) == winner(
            choose(1, 3, budget, augmented, minimum=2, maximum=9)['selected'])


def test_enlarging_evaluation_grid_cannot_worsen_the_optimum():
    scenarios = varied_scenarios()
    reduced = {name: {n: row for n, row in table.items() if n != 3} for name, table in scenarios.items()}
    for budget in (7, 12, 20, 50):
        small = choose(1, 3, budget, reduced, minimum=2, maximum=9)['selected']
        large = choose(1, 3, budget, scenarios, minimum=2, maximum=9)['selected']
        if small is not None:
            assert large is not None and large['variance'] <= small['variance']


def test_cost_and_variance_rescaling_preserve_allocations():
    scenarios, factor, H = varied_scenarios(), Fraction(11, 13), Fraction(2, 7)
    scaled = {name: {n: costs(factor * Fraction(row['setup']), factor * Fraction(row['pair']))
                     for n, row in table.items()} for name, table in scenarios.items()}
    for budget in (0, 7, 12, 20, 50):
        original = choose(1, 3, budget, scenarios, H, minimum=2, maximum=9)['selected']
        changed_cost = choose(1, 3, factor * budget, scaled, factor * H, minimum=2, maximum=9)['selected']
        changed_variance = choose(factor, 3 * factor, budget, scenarios, H, minimum=2, maximum=9)['selected']
        assert winner(original) == winner(changed_cost)
        if original is None:
            assert changed_variance is None
        else:
            assert (original['evaluation_paths'], original['repetitions']) == (
                changed_variance['evaluation_paths'], changed_variance['repetitions'])
            assert changed_variance['variance'] == factor * original['variance']


def test_single_allowed_repetition_count():
    scenarios = {'one': {1: costs(), 2: costs(0, 2)}}
    assert choose(1, 1, 63, scenarios, minimum=64, maximum=64)['selected'] is None
    assert winner(choose(1, 1, 64, scenarios, minimum=64, maximum=64)['selected']) == (1, 64, Fraction(1, 32))
    assert winner(choose(1, 1, 128, scenarios, minimum=64, maximum=64)['selected']) == (2, 64, Fraction(3, 128))


def test_evaluate_preserves_frozen_counts_and_original_budget():
    scenarios = {'first': {2: costs(10, 2)}, 'second': {2: costs(0, 1)}}
    row = evaluate(1, 2, 2, 80, 100, scenarios, overhead=5)
    assert (row['evaluation_paths'], row['repetitions'], row['variance']) == (2, 80, Fraction(1, 40))
    assert row['scenario_costs'] == {'first': 175, 'second': 85}
    assert row['scenario_feasible'] == {'first': False, 'second': True}
    assert row['scenario_budget_excess'] == {'first': 75, 'second': 0}
    assert row['scenario_cost_variance'] == {'first': Fraction(35, 8), 'second': Fraction(17, 8)}
    assert row['jointly_feasible'] is False
    assert choose(1, 2, 100, scenarios, overhead=5)['selected'] is None


def test_evaluate_budget_equality_is_feasible_without_changing_counts():
    scenarios = {'one': {1: costs(3, 2)}}
    row = evaluate(1, 1, 1, 70, 143, scenarios)
    assert row['repetitions'] == 70 and row['jointly_feasible'] is True
    assert row['scenario_budget_excess'] == {'one': 0}


@pytest.mark.parametrize('A,B', [(0, 1), (1, 0), (-1, 1), (1, -1), (True, 1),
                                    (float('nan'), 1), (1, float('inf')), ('1', 1)])
def test_nonpositive_or_invalid_variance_components_are_rejected(A, B):
    scenarios = {'one': {1: costs()}}
    for operation in (lambda: choose(A, B, 100, scenarios),
                      lambda: frontier(A, B, scenarios),
                      lambda: evaluate(A, B, 1, 64, 100, scenarios)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize('scenarios', [
    {}, [], {'': {1: costs()}}, {'one': {}}, {'one': {0: costs()}},
    {'one': {True: costs()}}, {'one': {1.5: costs()}},
    {'one': {1: costs(-1, 1)}}, {'one': {1: costs(0, 0)}}, {'one': {1: costs(0, -1)}},
    {'one': {1: costs(float('nan'), 1)}}, {'one': {1: costs(0, float('inf'))}},
    {'one': {1: costs(False, 1)}}, {'one': {1: costs(0, '1')}},
    {'one': {1: {'setup': 0}}}, {'one': {1: {'setup': 0, 'pair': 1, 'extra': 2}}},
    {'one': {1: costs()}, 'two': {2: costs()}},
])
def test_degenerate_cost_scenarios_are_rejected(scenarios):
    for operation in (lambda: choose(1, 1, 100, scenarios),
                      lambda: frontier(1, 1, scenarios),
                      lambda: evaluate(1, 1, 1, 64, 100, scenarios)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize('minimum,maximum', [(0, 2), (3, 2), (True, 2), (1, False), (1.5, 4), (1, 4.)])
def test_invalid_repetition_limits_are_rejected(minimum, maximum):
    scenarios = {'one': {1: costs()}}
    for operation in (lambda: choose(1, 1, 100, scenarios, minimum=minimum, maximum=maximum),
                      lambda: frontier(1, 1, scenarios, minimum=minimum, maximum=maximum)):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize('budget,overhead', [(-1, 0), (1, -1), (float('inf'), 0),
                                            (1, float('nan')), (True, 0), (1, False)])
def test_invalid_budget_and_overhead_are_rejected(budget, overhead):
    scenarios = {'one': {1: costs()}}
    with pytest.raises(ValueError):
        choose(1, 1, budget, scenarios, overhead)
    with pytest.raises(ValueError):
        evaluate(1, 1, 1, 64, budget, scenarios, overhead)
    if overhead != 0 or isinstance(overhead, bool):
        with pytest.raises(ValueError):
            frontier(1, 1, scenarios, overhead)


@pytest.mark.parametrize('n,r', [(2, 64), (0, 64), (True, 64), (1, 0), (1, -1), (1, True), (1, 64.)])
def test_invalid_frozen_allocations_are_rejected(n, r):
    with pytest.raises(ValueError):
        evaluate(1, 1, n, r, 100, {'one': {1: costs()}})
