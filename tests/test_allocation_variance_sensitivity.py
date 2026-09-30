"""Exact arithmetic fixtures for variance-component allocation sensitivity.

The fixtures stipulate mathematical examples. They are neither financial data
nor estimated computational or financial performance.
"""

from fractions import Fraction
from itertools import combinations

import pytest

from qf_research.optimal_stopping.allocation_variance_sensitivity import (
    envelope,
    moments,
    regret,
    region,
    select,
    share,
)


def candidates(*pairs):
    return [dict(evaluation_paths=n, repetitions=r) for n, r in pairs]


def identity(row):
    return row['evaluation_paths'], row['repetitions']


def variance(row, w, n0):
    """Evaluate a unit-total variance model independently of the kernel."""
    between_training = 1 - Fraction(w)
    per_path = Fraction(n0) * Fraction(w)
    return (between_training + per_path / row['evaluation_paths']) / row['repetitions']


def oracle(rows, w, n0):
    values = [(variance(row, w, n0), *identity(row)) for row in rows]
    best = min(values)
    minimisers = sorted((n, r) for value, n, r in values if value == best[0])
    return (best[1], best[2]), best[0], minimisers


def partition(rows, n0):
    """Enumerate pairwise intersections and every intervening open interval."""
    boundaries = {Fraction(0), Fraction(1)}
    for left, right in combinations(rows, 2):
        d0 = variance(left, 0, n0) - variance(right, 0, n0)
        d1 = variance(left, 1, n0) - variance(right, 1, n0)
        if d0 != d1:
            crossing = -d0 / (d1 - d0)
            if 0 <= crossing <= 1:
                boundaries.add(crossing)
    ordered = sorted(boundaries)
    probes = sorted(boundaries | {(a + b) / 2 for a, b in zip(ordered, ordered[1:])})
    return ordered, probes


def test_moments_match_pairwise_sample_variance_and_paired_sampling_errors():
    means = [Fraction(1, 3), Fraction(7, 3), Fraction(-2, 3), Fraction(4, 3)]
    errors = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    count = len(means)
    observed = sum((x - y) ** 2 for x, y in combinations(means, 2)) / (count * (count - 1))
    evaluation = sum(x ** 2 for x in errors) / count
    result = moments(means, errors, 256)
    assert result['observed_variance'] == observed
    assert result['evaluation_mean_variance'] == evaluation
    assert result['A'] == observed - evaluation
    assert result['B'] == 256 * evaluation
    assert all(isinstance(result[key], Fraction) for key in (
        'observed_variance', 'evaluation_mean_variance', 'A', 'B'))
    # Squaring the mean error would underestimate the evaluation contribution.
    assert result['evaluation_mean_variance'] != (sum(errors) / count) ** 2


def test_negative_between_training_estimate_is_retained():
    result = moments([3, 3, 3], [1, 2, 3], 128)
    assert result['observed_variance'] == 0
    assert result['evaluation_mean_variance'] == Fraction(14, 3)
    assert result['A'] == Fraction(-14, 3)
    assert result['B'] == Fraction(1792, 3)


def test_degenerate_errors_and_minimum_number_of_pairs():
    result = moments([0, 2], [0, 0], 2)
    assert result['observed_variance'] == 2
    assert result['evaluation_mean_variance'] == 0
    assert result['A'] == 2
    assert result['B'] == 0
    zero = moments([1, 1], [0, 0], 2)
    assert all(zero[key] == 0 for key in ('observed_variance', 'evaluation_mean_variance', 'A', 'B'))


def test_moments_preserve_exact_binary64_inputs_and_large_common_location():
    means, errors = [.1, .2, .3], [.03, .04, .08]
    exact = moments([Fraction(x) for x in means], [Fraction(x) for x in errors], 4096)
    assert moments(means, errors, 4096) == exact
    # Exact subtraction avoids loss from centring around a large rational location.
    shifted = moments([Fraction(x) + 10**100 for x in means], errors, 4096)
    assert shifted == exact


@pytest.mark.parametrize('factor', [Fraction(2, 3), Fraction(11), Fraction(-5, 7)])
def test_moments_have_quadratic_scale_and_translation_invariance(factor):
    means, errors = [Fraction(1), Fraction(4), Fraction(9)], [Fraction(1, 4), Fraction(1, 2), Fraction(1)]
    base = moments(means, errors, 4096)
    changed = moments([factor * x + 100 for x in means], [abs(factor) * x for x in errors], 4096)
    for key in ('observed_variance', 'evaluation_mean_variance', 'A', 'B'):
        assert changed[key] == factor**2 * base[key]


def test_paired_reordering_preserves_moments_but_omission_changes_them():
    means, errors = [1, 2, 6, 11], [Fraction(1, 2), 1, 2, 4]
    base = moments(means, errors, 128)
    order = [2, 0, 3, 1]
    assert moments([means[i] for i in order], [errors[i] for i in order], 128) == base
    omitted = moments(means[:-1], errors[:-1], 128)
    assert omitted['A'] != base['A']
    assert omitted['B'] != base['B']


def test_share_is_scale_invariant_and_handles_boundary_components():
    assert share(2, 24, reference_paths=4) == Fraction(3, 4)
    assert share(Fraction(2, 7), Fraction(24, 7), reference_paths=4) == Fraction(3, 4)
    assert share(1, 0) == 0
    assert share(0, 1) == 1
    assert isinstance(share(2, 24, reference_paths=4), Fraction)
    assert share(.1, .2) == Fraction(.2) / (4096 * Fraction(.1) + Fraction(.2))


def test_normalised_criterion_preserves_original_variance_ordering():
    rows = candidates((128, 67), (256, 59), (4096, 41))
    a, b = Fraction(3, 7), Fraction(91, 5)
    w = share(a, b)
    scale = a + b / 4096
    for row in rows:
        assert variance(row, w, 4096) * scale == (a + b / row['evaluation_paths']) / row['repetitions']
    expected = min(rows, key=lambda row: ((a + b / row['evaluation_paths']) / row['repetitions'], *identity(row)))
    assert identity(select(rows, w)) == identity(expected)


def test_exact_crossing_and_tie_rule_are_order_independent():
    rows = candidates((4, 2), (1, 4))
    x = Fraction(1, 3)
    at = select(rows, x, reference_paths=4)
    assert identity(at) == (1, 4)
    assert at['criterion'] == Fraction(1, 2)
    assert [identity(row) for row in at['minimisers']] == [(1, 4), (4, 2)]
    assert select(list(reversed(rows)), x, reference_paths=4) == at
    epsilon = Fraction(1, 2**100)
    assert identity(select(rows, x - epsilon, reference_paths=4)) == (1, 4)
    assert identity(select(rows, x + epsilon, reference_paths=4)) == (4, 2)
    assert region(rows, 1, reference_paths=4) == dict(lower=Fraction(0), upper=x, lower_selected=True, upper_selected=True)
    assert region(rows, 4, reference_paths=4) == dict(lower=x, upper=Fraction(1), lower_selected=False, upper_selected=True)


def test_coincident_intercepts_include_weak_singleton_at_zero():
    rows = candidates((2, 2), (4, 2))
    assert region(rows, 2, reference_paths=4) == dict(lower=Fraction(0), upper=Fraction(0), lower_selected=True, upper_selected=True)
    assert region(rows, 4, reference_paths=4) == dict(lower=Fraction(0), upper=Fraction(1), lower_selected=False, upper_selected=True)
    assert identity(select(rows, 0, reference_paths=4)) == (2, 2)
    assert identity(select(rows, Fraction(1, 2**100), reference_paths=4)) == (4, 2)


def test_equal_path_repetition_product_includes_weak_singleton_at_one():
    rows = candidates((2, 4), (4, 2))
    assert region(rows, 2, reference_paths=4) == dict(lower=Fraction(0), upper=Fraction(1), lower_selected=True, upper_selected=True)
    assert region(rows, 4, reference_paths=4) == dict(lower=Fraction(1), upper=Fraction(1), lower_selected=False, upper_selected=False)
    assert [identity(row) for row in select(rows, 1, reference_paths=4)['minimisers']] == [(2, 4), (4, 2)]


def test_interior_singletons_are_weakly_optimal_even_when_never_selected():
    rows = candidates((2, 7), (3, 5), (4, 4), (6, 3), (12, 2))
    for n in (3, 4, 6):
        assert region(rows, n, reference_paths=12) == dict(lower=Fraction(1, 2), upper=Fraction(1, 2), lower_selected=False, upper_selected=False)
    at = select(rows, Fraction(1, 2), reference_paths=12)
    assert identity(at) == (2, 7)
    assert len(at['minimisers']) == 5
    assert len(envelope(rows, reference_paths=12)) == 5


def test_strictly_dominated_candidate_has_no_region_and_is_absent_from_envelope():
    rows = candidates((1, 4), (2, 1), (4, 2))
    assert region(rows, 2, reference_paths=4) is None
    assert {identity(row) for row in envelope(rows, reference_paths=4)} == {(1, 4), (4, 2)}


@pytest.mark.parametrize('pairs,n0', [
    ([(1, 4), (2, 1), (4, 2)], 4),
    ([(2, 7), (3, 5), (4, 4), (6, 3), (12, 2)], 12),
    ([(128, 666), (256, 420), (4096, 93)], 4096),
    ([(1, 1)], 3),
    ([(1, 7), (2, 7), (3, 7), (19, 7)], 12),
    ([(1, 31), (3, 22), (7, 13), (11, 9), (50, 8), (100, 2)], 64),
])
def test_full_envelope_and_selections_match_independent_partition(pairs, n0):
    rows = candidates(*pairs)
    boundaries, probes = partition(rows, n0)
    reported = {row['evaluation_paths']: row for row in envelope(rows, reference_paths=n0)}
    for w in probes:
        expected, value, minimisers = oracle(rows, w, n0)
        actual = select(rows, w, reference_paths=n0)
        assert identity(actual) == expected
        assert actual['criterion'] == value
        assert [identity(row) for row in actual['minimisers']] == minimisers
    for row in rows:
        n = row['evaluation_paths']
        weak_points = [w for w in boundaries if identity(row) in oracle(rows, w, n0)[2]]
        actual = region(rows, n, reference_paths=n0)
        if not weak_points:
            assert actual is None
            assert n not in reported
            continue
        lower, upper = min(weak_points), max(weak_points)
        expected = dict(lower=lower, upper=upper,
                        lower_selected=identity(row) == oracle(rows, lower, n0)[0],
                        upper_selected=identity(row) == oracle(rows, upper, n0)[0])
        assert actual == expected
        assert reported[n]['region'] == expected
        assert identity(reported[n]) == identity(row)
        for w in probes:
            assert (lower <= w <= upper) == (identity(row) in oracle(rows, w, n0)[2])


def test_regret_uses_larger_endpoint_and_includes_interior_switches():
    rows = candidates((1, 4), (2, 1), (4, 2))
    result = regret(rows, 1, 0, 1, reference_paths=4)
    assert result == dict(maximum=Fraction(2), lower_ratio=Fraction(1), upper_ratio=Fraction(2), worst_endpoints=[Fraction(1)])
    assert regret(rows, 4, 0, 1, reference_paths=4) == dict(maximum=Fraction(2), lower_ratio=Fraction(2), upper_ratio=Fraction(1), worst_endpoints=[Fraction(0)])
    dominated = regret(rows, 2, 0, 1, reference_paths=4)
    assert dominated == dict(maximum=Fraction(4), lower_ratio=Fraction(4), upper_ratio=Fraction(4), worst_endpoints=[Fraction(0), Fraction(1)])


def test_regret_on_singleton_interval_reports_one_endpoint():
    rows = candidates((1, 4), (4, 2))
    result = regret(rows, 1, Fraction(1, 3), Fraction(1, 3), reference_paths=4)
    assert result == dict(maximum=Fraction(1), lower_ratio=Fraction(1), upper_ratio=Fraction(1), worst_endpoints=[Fraction(1, 3)])


@pytest.mark.parametrize('lower,upper', [(0, 1), (Fraction(1, 7), Fraction(5, 6)), (Fraction(1, 2), Fraction(1, 2))])
def test_regret_endpoint_formula_bounds_every_partition_cell(lower, upper):
    rows = candidates((1, 31), (3, 22), (7, 13), (11, 9), (50, 8), (100, 2))
    _, probes = partition(rows, 64)
    probes = sorted({Fraction(lower), Fraction(upper)} | {w for w in probes if lower <= w <= upper})
    for row in rows:
        actual = regret(rows, row['evaluation_paths'], lower, upper, reference_paths=64)
        ratios = {w: variance(row, w, 64) / oracle(rows, w, 64)[1] for w in probes}
        assert actual['maximum'] == max(ratios.values())
        assert actual['lower_ratio'] == ratios[lower]
        assert actual['upper_ratio'] == ratios[upper]
        assert actual['worst_endpoints'] == [w for w in sorted({Fraction(lower), Fraction(upper)}) if ratios[w] == actual['maximum']]
        assert actual['maximum'] >= 1


@pytest.mark.parametrize('means,errors,paths', [
    ([], [], 2), ([1], [0], 2), ([1, 2], [0], 2),
    ([1, 2], [0, 0, 0], 2), ([1, 2], [0, -1], 2),
    ([1, float('nan')], [0, 0], 2), ([1, float('inf')], [0, 0], 2),
    ([1, 2], [0, float('inf')], 2), ([1, 2], [0, float('nan')], 2),
    ([True, 2], [0, 0], 2), ([1, 2], [False, 0], 2),
    ([1, 2], [0, 0], 1), ([1, 2], [0, 0], 0),
    ([1, 2], [0, 0], 2.0), ([1, 2], [0, 0], True),
])
def test_invalid_moment_inputs_are_rejected(means, errors, paths):
    with pytest.raises((ValueError, TypeError)):
        moments(means, errors, paths)


@pytest.mark.parametrize('a,b', [(0, 0), (-1, 1), (1, -1), (float('nan'), 1), (1, float('inf')), (True, 1)])
def test_invalid_variance_components_are_rejected(a, b):
    with pytest.raises((ValueError, TypeError)):
        share(a, b)


@pytest.mark.parametrize('rows', [
    [], candidates((0, 1)), candidates((-1, 1)), candidates((1, 0)),
    candidates((1, -1)), candidates((1.0, 1)), candidates((1, 2.0)),
    candidates((True, 1)), candidates((1, False)),
    candidates((1, 2), (1, 3)), candidates((1, 2), (1, 2)),
])
def test_invalid_candidate_sets_are_rejected(rows):
    for operation in (
        lambda: select(rows, Fraction(1, 2)),
        lambda: region(rows, 1),
        lambda: regret(rows, 1, 0, 1),
        lambda: envelope(rows),
    ):
        with pytest.raises((ValueError, TypeError)):
            operation()


@pytest.mark.parametrize('w', [-1, Fraction(1001, 1000), float('inf'), float('nan'), True])
def test_invalid_share_argument_is_rejected(w):
    with pytest.raises((ValueError, TypeError)):
        select(candidates((1, 1)), w)


@pytest.mark.parametrize('n0', [0, -1, 4.0, True])
def test_invalid_reference_path_counts_are_rejected(n0):
    rows = candidates((1, 2), (2, 1))
    for operation in (
        lambda: share(1, 1, reference_paths=n0),
        lambda: select(rows, Fraction(1, 2), reference_paths=n0),
        lambda: region(rows, 1, reference_paths=n0),
        lambda: regret(rows, 1, 0, 1, reference_paths=n0),
        lambda: envelope(rows, reference_paths=n0),
    ):
        with pytest.raises((ValueError, TypeError)):
            operation()


@pytest.mark.parametrize('lower,upper', [(-1, 1), (0, 2), (1, 0), (float('nan'), 1), (0, float('inf')), (False, 1)])
def test_invalid_regret_intervals_are_rejected(lower, upper):
    with pytest.raises((ValueError, TypeError)):
        regret(candidates((1, 2), (2, 1)), 1, lower, upper)


@pytest.mark.parametrize('target', [3, 0, 1.0, True])
def test_unknown_or_invalid_target_is_rejected(target):
    rows = candidates((1, 2), (2, 1))
    with pytest.raises((ValueError, TypeError)):
        region(rows, target)
    with pytest.raises((ValueError, TypeError)):
        regret(rows, target, 0, 1)
