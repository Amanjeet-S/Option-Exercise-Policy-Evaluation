"""Exact mathematical fixtures for prospective allocation and moment inference.

These finite populations and variance curves are stipulated algebraic examples.
They are not financial observations or estimates of empirical coverage.
"""

from fractions import Fraction as F
from itertools import combinations, product

import pytest

from qf_research.optimal_stopping.prospective_allocation import (
    compare_summaries,
    minimise_relative_variance,
    sample_summary,
)


def candidates(*pairs):
    return [dict(evaluation_paths=n, repetitions=r) for n, r in pairs]


def components(*intervals):
    return [dict(lower=lo, upper=hi) for lo, hi in intervals]


def raw_moment_oracle(values):
    """Central moments from raw power sums, independently of the kernel."""
    values = list(map(F, values))
    m = len(values)
    first, second, third, fourth = (
        sum(x**power for x in values) for power in range(1, 5)
    )
    mean = first/m
    central_two = second - first*first/m
    central_three = third - 3*mean*second + 2*m*mean**3
    central_four = fourth - 4*mean*third + 6*mean**2*second - 3*m*mean**4
    return dict(
        mean=mean,
        row_variance=central_two/(m-1),
        variance_of_mean=central_two/(m*(m-1)),
        variance_of_variance=(central_four-central_two**2/m)/(m**3*(m-1)),
        covariance_mean_variance=central_three/(m*m*(m-1)),
    )


def pairwise_variance_of_mean(values, fixed_count):
    """The pairwise-difference identity uses a separately specified divisor."""
    values = list(map(F, values))
    count = len(values)
    total = sum((a-b)**2 for a, b in combinations(values, 2))
    return total/(count*(count-1)*fixed_count)


def allocation_variance(row, share):
    return (1-F(share)+4096*F(share)/row['evaluation_paths'])/row['repetitions']


def exhaustive_curve_oracle(rows, intervals):
    """Inspect endpoints, pairwise crossings and every intervening midpoint."""
    points = set()
    for interval in intervals:
        lo, hi = F(interval['lower']), F(interval['upper'])
        boundaries = {lo, hi}
        for left, right in combinations(rows, 2):
            at_zero = allocation_variance(left, 0)-allocation_variance(right, 0)
            at_one = allocation_variance(left, 1)-allocation_variance(right, 1)
            if at_zero != at_one:
                cross = -at_zero/(at_one-at_zero)
                if lo <= cross <= hi:
                    boundaries.add(cross)
        ordered = sorted(boundaries)
        points.update(boundaries)
        points.update((a+b)/2 for a, b in zip(ordered, ordered[1:]))
    scores = {}
    for row in rows:
        scores[row['evaluation_paths']] = max(
            allocation_variance(row, w)/min(allocation_variance(other, w) for other in rows)
            for w in points
        )
    return scores


@pytest.mark.parametrize('values', [
    [-1, 0, 1],
    [F(-5, 3), F(2, 7), F(11, 5), F(19, 4), F(-9, 2)],
    [3, 3, 3, 3],
    [0, 1, 2, 8, 13, 21, 34],
])
def test_moments_match_independent_raw_power_sums(values):
    result = sample_summary(values)
    assert result['pairs'] == len(values)
    for key, value in raw_moment_oracle(values).items():
        assert result[key] == value
        assert isinstance(result[key], F)
    assert result['variance_of_mean'] == pairwise_variance_of_mean(values, len(values))


@pytest.mark.parametrize('count', [3, 4, 5])
def test_variance_of_mean_is_unbiased_by_complete_finite_population_enumeration(count):
    # Under uniform independent draws from {-2,4}, the population variance is 9.
    # Exhaustive enumeration, rather than random sampling, verifies expectation.
    records = [sample_summary(values) for values in product([-2, 4], repeat=count)]
    assert sum(row['row_variance'] for row in records)/len(records) == 9
    assert sum(row['variance_of_mean'] for row in records)/len(records) == F(9, count)
    assert sum(row['mean'] for row in records)/len(records) == 1


def test_variance_influence_is_an_asymptotic_estimate_not_an_exact_finite_variance():
    records = [sample_summary(values) for values in product([0, 1], repeat=3)]
    true_variance_of_mean = F(1, 12)
    true_estimator_variance = sum(
        (row['variance_of_mean']-true_variance_of_mean)**2 for row in records
    )/len(records)
    expected_influence_estimate = sum(row['variance_of_variance'] for row in records)/len(records)
    assert true_estimator_variance == F(1, 432)
    assert expected_influence_estimate == F(1, 972)
    assert expected_influence_estimate != true_estimator_variance


@pytest.mark.parametrize('values', [[-1, 0, 1], [0, 1, 9, 10], [1, 1, 1], [1, 2, 4, 8, 16]])
def test_leave_one_variances_use_original_count_and_record_first_extremum(values):
    result = sample_summary(values)['leave_one_variance_at_fixed_count']
    leave_one = [
        pairwise_variance_of_mean(values[:i]+values[i+1:], len(values))
        for i in range(len(values))
    ]
    assert result == dict(lower=min(leave_one), upper=max(leave_one),
                          lower_row=leave_one.index(min(leave_one)), upper_row=leave_one.index(max(leave_one)))
    if max(leave_one) > 0:
        varying_count = max(
            pairwise_variance_of_mean(values[:i]+values[i+1:], len(values)-1)
            for i in range(len(values))
        )
        assert result['upper'] == F(len(values)-1, len(values))*varying_count


def test_mean_and_variance_influences_have_expected_joint_covariance():
    result = sample_summary([-1, -1, 2])
    assert result['mean'] == 0
    assert result['row_variance'] == 3
    assert result['variance_of_mean'] == 1
    assert result['variance_of_variance'] == F(1, 9)
    assert result['covariance_mean_variance'] == F(1, 3)
    assert result['largest_squared_mean_influence_share'] == F(2, 3)
    assert result['largest_squared_variance_influence_share'] == F(2, 3)
    assert result['mean_variance_degenerate'] is False
    assert result['variance_influence_degenerate'] is False


@pytest.mark.parametrize('values', [[1, 1, 1], [-1, -1, 1, 1], [-1, -1, 2], [-3, 0, 1, 8, 11]])
def test_covariance_is_positive_semidefinite_including_singular_cases(values):
    result = sample_summary(values)
    a, b, c = (result[k] for k in ('variance_of_mean', 'covariance_mean_variance', 'variance_of_variance'))
    assert min(a, c) >= 0 and a*c >= b*b
    for x, y in [(1, 0), (0, 1), (1, -3), (F(-11, 7), F(19, 3))]:
        assert x*x*a + 2*x*y*b + y*y*c >= 0


def test_constant_and_equal_distance_samples_have_distinct_degeneracies():
    constant = sample_summary([7, 7, 7])
    assert constant['mean_variance_degenerate'] is True
    assert constant['variance_influence_degenerate'] is True
    assert constant['largest_squared_mean_influence_share'] is None
    assert constant['largest_squared_variance_influence_share'] is None
    symmetric = sample_summary([-1, -1, 1, 1])
    assert symmetric['mean_variance_degenerate'] is False
    assert symmetric['variance_influence_degenerate'] is True
    assert symmetric['largest_squared_mean_influence_share'] == F(1, 4)
    assert symmetric['largest_squared_variance_influence_share'] is None


@pytest.mark.parametrize('scale', [F(2, 3), F(-11, 7), F(10**30)])
def test_location_scale_and_permutation_invariance(scale):
    values = [F(-3), F(1, 4), F(6), F(9)]
    base = sample_summary(values)
    shifted = sample_summary([scale*x+10**100 for x in values])
    assert shifted['mean'] == scale*base['mean'] + 10**100
    for key in ('row_variance', 'variance_of_mean'):
        assert shifted[key] == scale**2*base[key]
    assert shifted['variance_of_variance'] == scale**4*base['variance_of_variance']
    assert shifted['covariance_mean_variance'] == scale**3*base['covariance_mean_variance']
    for key in ('largest_squared_mean_influence_share', 'largest_squared_variance_influence_share',
                'mean_variance_degenerate', 'variance_influence_degenerate'):
        assert shifted[key] == base[key]
    for key in ('lower', 'upper'):
        assert shifted['leave_one_variance_at_fixed_count'][key] == scale**2*base['leave_one_variance_at_fixed_count'][key]
    reversed_result = sample_summary(values[::-1])
    for key in set(base)-{'leave_one_variance_at_fixed_count'}:
        assert reversed_result[key] == base[key]


def test_binary64_values_are_treated_as_their_exact_recorded_rationals():
    values = [.1, .3, .7, 1.1]
    assert sample_summary(values) == sample_summary(list(map(F, values)))


@pytest.mark.parametrize('values', [[], [0], [0, 1], [False, 1, 2], [0, '1', 2],
                                   [0, float('nan'), 2], [0, float('inf'), 2],
                                   [0, -float('inf'), 2], [0, complex(1, 0), 2]])
def test_invalid_samples_are_rejected(values):
    with pytest.raises(ValueError):
        sample_summary(values)


def test_independent_arm_contrasts_add_sampling_variances_and_retain_signs():
    target, reference = sample_summary([-3, 1, 2, 8, 9]), sample_summary([-2, 0, 1, 4])
    result = compare_summaries(target, reference)
    assert result['mean_contrast']['difference'] == target['mean']-reference['mean']
    assert result['mean_contrast']['estimated_sampling_variance'] == target['variance_of_mean']+reference['variance_of_mean']
    assert result['variance_contrast']['difference'] == target['variance_of_mean']-reference['variance_of_mean']
    assert result['variance_contrast']['estimated_sampling_variance'] == target['variance_of_variance']+reference['variance_of_variance']
    assert result['estimated_variance_ratio'] == target['variance_of_mean']/reference['variance_of_mean']
    assert result['ratio_has_confidence_interval'] is False
    assert result['critical_value'] == F(13, 4)
    for key in ('mean_contrast', 'variance_contrast'):
        row = result[key]
        assert row['radius_squared'] == F(169, 16)*row['estimated_sampling_variance']
        assert row['asymptotic_only'] is True
        assert row['normal_interval_status'] == 'includes_zero'


def test_swapping_arms_reverses_contrasts_and_inverts_positive_descriptive_ratio():
    first, second = sample_summary([-2, 0, 8]), sample_summary([-10, -9, -9, -8])
    forward, reverse = compare_summaries(first, second), compare_summaries(second, first)
    for key in ('mean_contrast', 'variance_contrast'):
        assert forward[key]['difference'] == -reverse[key]['difference']
        assert forward[key]['estimated_sampling_variance'] == reverse[key]['estimated_sampling_variance']
        assert forward[key]['radius_squared'] == reverse[key]['radius_squared']
    assert forward['estimated_variance_ratio']*reverse['estimated_variance_ratio'] == 1
    assert forward['mean_contrast']['normal_interval_status'] == 'positive'
    assert reverse['mean_contrast']['normal_interval_status'] == 'negative'


def test_exact_equality_and_arbitrarily_close_mean_interval_boundaries():
    target = sample_summary([-1, -1, 2])
    epsilon = F(1, 2**200)
    for sign in (-1, 1):
        for offset, expected in [(-epsilon, 'includes_zero'), (F(0), 'includes_zero'),
                                 (epsilon, 'positive' if sign > 0 else 'negative')]:
            location = -sign*(F(13, 4)+offset)
            result = compare_summaries(target, sample_summary([location]*3))
            assert result['mean_contrast']['normal_interval_status'] == expected


def test_exact_equality_and_arbitrarily_close_variance_interval_boundaries():
    target, reference = sample_summary([-1, -1, 2]), sample_summary([0, 0, 0])
    epsilon = F(1, 2**200)
    for cutoff, status in [(3-epsilon, 'positive'), (F(3), 'includes_zero'), (3+epsilon, 'includes_zero')]:
        result = compare_summaries(target, reference, cutoff)
        assert result['variance_contrast']['normal_interval_status'] == status
    assert compare_summaries(reference, target, 3-epsilon)['variance_contrast']['normal_interval_status'] == 'negative'


def test_zero_estimated_sampling_variance_is_flagged_not_treated_as_significance():
    result = compare_summaries(sample_summary([4, 4, 4]), sample_summary([0, 0, 0]))
    assert result['mean_contrast']['difference'] == 4
    assert result['mean_contrast']['normal_interval_status'] == 'degenerate'
    assert result['variance_contrast']['normal_interval_status'] == 'degenerate'
    assert result['estimated_variance_ratio'] is None
    symmetric = compare_summaries(sample_summary([-2, -2, 2, 2]), sample_summary([-1, -1, 1, 1]))
    assert symmetric['variance_contrast']['difference'] > 0
    assert symmetric['variance_contrast']['normal_interval_status'] == 'degenerate'
    assert symmetric['estimated_variance_ratio'] == 4
    assert compare_summaries(sample_summary([0, 0, 0]), sample_summary([-1, 0, 1]))['estimated_variance_ratio'] == 0


@pytest.mark.parametrize('cutoff', [0, -1, True, '3.25', float('nan'), float('inf')])
def test_invalid_normal_cutoffs_are_rejected(cutoff):
    summary = sample_summary([0, 1, 2])
    with pytest.raises(ValueError):
        compare_summaries(summary, summary, cutoff)


@pytest.mark.parametrize('side,key,value', [
    ('target', 'mean', float('nan')), ('reference', 'mean', False),
    ('target', 'variance_of_mean', -1), ('reference', 'variance_of_mean', float('inf')),
    ('target', 'variance_of_variance', -1), ('reference', 'variance_of_variance', '1'),
])
def test_invalid_comparison_summaries_are_rejected(side, key, value):
    samples = dict(target=sample_summary([-1, 0, 1]), reference=sample_summary([1, 2, 4]))
    samples[side][key] = value
    with pytest.raises(ValueError):
        compare_summaries(**samples)


def test_minimax_allocation_matches_hand_calculated_three_line_problem():
    rows = candidates((1024, 4), (2048, 3), (4096, 2))
    result = minimise_relative_variance(rows, components((0, 1)))
    assert result['scores'] == [
        dict(evaluation_paths=1024, repetitions=4, maximum_relative_variance=F(2)),
        dict(evaluation_paths=2048, repetitions=3, maximum_relative_variance=F(4, 3)),
        dict(evaluation_paths=4096, repetitions=2, maximum_relative_variance=F(2)),
    ]
    assert result['selected'] == result['scores'][1]
    assert result['minimisers'] == [2048]


def test_exact_minimax_ties_choose_smallest_path_count_independently_of_input_order():
    rows = candidates((4096, 2), (1024, 4))
    result = minimise_relative_variance(rows, components((0, 1)))
    assert result['minimisers'] == [1024, 4096]
    assert result['selected']['evaluation_paths'] == 1024
    assert result['selected']['maximum_relative_variance'] == 2
    assert minimise_relative_variance(rows[::-1], components((0, 1))) == result
    at_crossing = minimise_relative_variance(rows, components((F(1, 3), F(1, 3))))
    assert at_crossing['minimisers'] == [1024, 4096]
    assert all(row['maximum_relative_variance'] == 1 for row in at_crossing['scores'])


@pytest.mark.parametrize('pairs,intervals', [
    ([(128, 666), (256, 420), (4096, 93)], [(0, 1)]),
    ([(1, 31), (3, 22), (7, 13), (11, 9), (50, 8), (100, 2)], [(F(1, 29), F(7, 13))]),
    ([(1024, 4), (2048, 3), (4096, 2)], [(0, 0), (1, 1)]),
    ([(1024, 4), (2048, 3), (4096, 2)], [(0, F(1, 5)), (F(7, 9), 1)]),
    ([(128, 9), (256, 9), (4096, 9)], [(F(1, 7), F(1, 7))]),
    ([(128, 9)], [(0, 1)]),
])
def test_minimax_scores_match_complete_independent_curve_partition(pairs, intervals):
    rows, interval_rows = candidates(*pairs), components(*intervals)
    expected = exhaustive_curve_oracle(rows, interval_rows)
    result = minimise_relative_variance(rows, interval_rows)
    assert {row['evaluation_paths']: row['maximum_relative_variance'] for row in result['scores']} == expected
    minima = sorted(n for n, value in expected.items() if value == min(expected.values()))
    assert result['minimisers'] == minima
    assert result['selected']['evaluation_paths'] == minima[0]
    assert all(isinstance(row['maximum_relative_variance'], F) for row in result['scores'])


def test_disconnected_sets_overlaps_and_repeated_components_obey_endpoint_geometry():
    rows = candidates((128, 67), (256, 59), (4096, 41))
    disconnected = components((0, F(1, 7)), (F(3, 4), 1))
    base = minimise_relative_variance(rows, disconnected)
    assert minimise_relative_variance(rows, list(reversed(disconnected))) == base
    assert minimise_relative_variance(rows, disconnected+disconnected) == base
    assert minimise_relative_variance(rows, components((0, 1))) == base
    assert minimise_relative_variance(rows, components((0, F(3, 4)), (F(1, 7), 1))) == base


@pytest.mark.parametrize('rows', [[], candidates((0, 4)), candidates((128, 0)),
                                  candidates((True, 4)), candidates((128, 2.0)),
                                  candidates((128, 4), (128, 5))])
def test_invalid_candidate_sets_are_rejected(rows):
    with pytest.raises(ValueError):
        minimise_relative_variance(rows, components((0, 1)))


@pytest.mark.parametrize('intervals', [[], components((1, 0)), components((-1, 0)),
                                      components((0, 2)), components((False, 1)),
                                      components((0, float('nan'))), components((0, float('inf'))),
                                      components(('0', 1))])
def test_invalid_component_sets_are_rejected(intervals):
    with pytest.raises(ValueError):
        minimise_relative_variance(candidates((128, 4)), intervals)
