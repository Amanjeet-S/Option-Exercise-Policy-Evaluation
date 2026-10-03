"""Mathematical fixtures for joint moments and exact ratio-set inversion.

These stipulated examples test algebra and boundary ownership. They are not
financial observations or simulated estimates of financial performance.
"""

from fractions import Fraction as F
from itertools import combinations

import pytest

from qf_research.optimal_stopping.component_ratio import (
    allocation_choices,
    endpoint_compare,
    fieller_coefficients,
    intersects,
    moment_covariance,
    quadratic_set,
)


def sample_covariance(vectors):
    """Covariance of a sample mean, evaluated as a raw matrix product."""
    m, p = len(vectors), len(vectors[0])
    centre = [sum(row[j] for row in vectors) / m for j in range(p)]
    return [[sum((row[j] - centre[j]) * (row[k] - centre[k])
                 for row in vectors) / (m * (m - 1))
             for k in range(p)] for j in range(p)]


def raw_moment_oracle(means, errors, paths, reference_paths):
    x, u = list(map(F, means)), [F(e)**2 for e in errors]
    m = len(x)
    mu = sum(x) / m
    matrix = sample_covariance(list(zip(x, [z*z for z in x], u)))
    k = F(paths, reference_paths)
    jacobian = [[-2 * mu, F(1), k - 1], [F(0), F(0), k]]
    covariance = [[sum(jacobian[j][s] * matrix[s][t] * jacobian[l][t]
                       for s in range(3) for t in range(3))
                   for l in range(2)] for j in range(2)]
    observed = sum((a-b)**2 for a, b in combinations(x, 2)) / (m*(m-1))
    average_u = sum(u) / m
    return observed, average_u, covariance


@pytest.mark.parametrize('paths,reference', [(2, 4096), (128, 4096), (4096, 4096), (8192, 4096), (7, 13)])
def test_joint_moments_match_raw_moment_jacobian(paths, reference):
    means = [F(-5, 3), F(2, 7), F(11, 5), F(19, 4), F(-9, 2)]
    errors = [F(1, 8), F(4, 7), F(2), F(3, 5), F(7, 4)]
    result = moment_covariance(means, errors, paths, reference)
    observed, ubar, covariance = raw_moment_oracle(means, errors, paths, reference)
    assert result['observed_variance'] == observed
    assert result['evaluation_mean_variance'] == ubar
    assert result['A'] == observed - ubar
    assert result['B'] == paths * ubar
    assert result['denominator'] == result['A'] + result['B'] / reference
    assert result['numerator'] == result['B'] / reference
    assert result['covariance'] == dict(dd=covariance[0][0], de=covariance[0][1], ee=covariance[1][1])
    assert covariance[0][1] == covariance[1][0]
    assert covariance[0][0] * covariance[1][1] > covariance[0][1]**2
    assert result['covariance_rank'] == 2


def test_covariance_uses_centring_and_covariance_of_mean_not_individual():
    result = moment_covariance([-1, 0, 1], [1, 2, 3], 4096)
    # Centred squared deviations are (1/3,-2/3,1/3), and centred
    # squared errors are (-11/3,-2/3,13/3). The divisor is m(m-1)=6.
    assert result['covariance'] == dict(dd=F(1, 9), de=F(1, 9), ee=F(49, 9))
    assert result['denominator'] == 1
    assert result['numerator'] == F(14, 3)
    assert result['A'] == F(-11, 3)
    assert result['point_share'] == F(14, 3)
    assert result['point_in_parameter_domain'] is False
    influence_d = [F(1, 3), F(-2, 3), F(1, 3)]
    influence_e = [F(-11, 3), F(-2, 3), F(13, 3)]
    contrast = [e - F(14, 3)*d for d, e in zip(influence_d, influence_e)]
    concentration = lambda v: max(t*t for t in v)/sum(t*t for t in v)
    assert result['largest_squared_influence_share'] == dict(
        denominator=F(2, 3), numerator=F(169, 294), ratio_contrast=concentration(contrast))


@pytest.mark.parametrize('means,errors,rank', [
    ([1, 1, 1], [0, 0, 0], 0),
    ([1, 1, 1], [1, 1, 1], 0),
    ([-1, 0, 1], [1, 1, 1], 1),
    ([1, 1, 1], [1, 2, 3], 1),
    ([-1, 0, 1], [1, 2, 3], 2),
])
def test_covariance_rank_and_positive_semidefiniteness(means, errors, rank):
    result = moment_covariance(means, errors, 128)
    dd, de, ee = (result['covariance'][key] for key in ('dd', 'de', 'ee'))
    assert dd >= 0 and ee >= 0 and dd*ee >= de*de
    assert result['covariance_rank'] == rank
    for x, y in [(1, 0), (0, 1), (1, -3), (-8, 5), (F(1, 7), F(-11, 3))]:
        assert dd*x*x + 2*de*x*y + ee*y*y >= 0


def test_zero_concentrations_and_nonpositive_denominator_remain_undefined():
    zero = moment_covariance([3, 3, 3], [0, 0, 0], 128)
    assert zero['point_share'] is None
    assert zero['point_in_parameter_domain'] is False
    assert all(value is None for value in zero['largest_squared_influence_share'].values())
    negative = moment_covariance([3, 3, 3], [1, 2, 3], 128)
    assert negative['A'] < 0 and negative['denominator'] < 0
    assert negative['point_share'] is None
    assert negative['largest_squared_influence_share']['ratio_contrast'] is None
    boundary = moment_covariance([-1, 0, 1], [0, 0, 0], 128)
    assert boundary['point_share'] == 0 and boundary['point_in_parameter_domain']
    assert boundary['largest_squared_influence_share']['numerator'] is None
    assert boundary['largest_squared_influence_share']['ratio_contrast'] is None


@pytest.mark.parametrize('scale', [F(2, 3), F(-11, 7), F(10**20)])
def test_scale_location_and_pair_order_invariance(scale):
    means, errors = [F(-3), F(1), F(6), F(9)], [F(1, 2), F(3, 4), F(2), F(3)]
    base = moment_covariance(means, errors, 256)
    scaled = moment_covariance([scale*x+10**100 for x in means], [abs(scale)*e for e in errors], 256)
    for key in ('observed_variance', 'evaluation_mean_variance', 'A', 'B', 'denominator', 'numerator'):
        assert scaled[key] == scale**2 * base[key]
    assert scaled['covariance'] == {key: scale**4*value for key, value in base['covariance'].items()}
    for key in ('point_share', 'point_in_parameter_domain', 'covariance_rank', 'largest_squared_influence_share'):
        assert scaled[key] == base[key]
    assert moment_covariance(means[::-1], errors[::-1], 256) == base


def test_binary64_inputs_are_exactly_their_stored_rationals():
    means, errors = [.1, .3, .7, 1.1], [.01, .04, .05, .08]
    assert moment_covariance(means, errors, 128) == moment_covariance(list(map(F, means)), list(map(F, errors)), 128)


def test_joint_covariance_retains_pairing_information_absent_from_marginal_moments():
    means, errors = [-3, 0, 1, 7], [1, 2, 3, 4]
    matched = moment_covariance(means, errors, 4096)
    rearranged = moment_covariance(means, errors[::-1], 4096)
    for key in ('A', 'B', 'denominator', 'numerator'):
        assert matched[key] == rearranged[key]
    assert matched['covariance']['dd'] == rearranged['covariance']['dd']
    assert matched['covariance']['ee'] == rearranged['covariance']['ee']
    assert matched['covariance']['de'] != rearranged['covariance']['de']
    assert matched['largest_squared_influence_share']['ratio_contrast'] != rearranged['largest_squared_influence_share']['ratio_contrast']


def test_influence_covariance_is_asymptotic_not_exact_delete_one_covariance():
    means, errors = list(map(F, [-2, 0, 1, 5, 7])), list(map(F, [1, 2, 1, 3, 4]))
    m, k = len(means), F(128, 4096)
    result = moment_covariance(means, errors, 128)
    observed, ubar, _ = raw_moment_oracle(means, errors, 128, 4096)
    mean = sum(means)/m
    centred_squares = [(x-mean)**2 - F(m-1, m)*observed for x in means]
    centred_u = [e*e-ubar for e in errors]
    # Exact delete-one pseudovalues have the finite-multiplier m/(m-2)
    # on the sample-variance influence; this tends to one asymptotically.
    pseudo = []
    for i in range(m):
        leave = moment_covariance(means[:i]+means[i+1:], errors[:i]+errors[i+1:], 128)
        pseudo.append([m*result[key]-(m-1)*leave[key] for key in ('denominator', 'numerator')])
    for i, row in enumerate(pseudo):
        assert row[0]-result['denominator'] == F(m, m-2)*centred_squares[i]+(k-1)*centred_u[i]
        assert row[1]-result['numerator'] == k*centred_u[i]
    jackknife = sample_covariance(pseudo)
    assert jackknife[0][0] != result['covariance']['dd']
    assert jackknife[1][1] == result['covariance']['ee']
    repeated = moment_covariance(means*20, errors*20, 128)
    # Replication stipulates a sequence of empirical distributions. It does
    # not supply additional independent observations of a financial model.
    relative_finite_multiplier = F(100, 98)-1
    assert relative_finite_multiplier < F(5, 3)-1
    assert repeated['covariance_rank'] == result['covariance_rank']


@pytest.mark.parametrize('means,errors,paths,reference', [
    ([0, 1], [1, 1], 128, 4096),
    ([0, 1, 2], [1, 1], 128, 4096),
    ([0, 1, 2], [1, -1, 1], 128, 4096),
    ([False, 1, 2], [1, 1, 1], 128, 4096),
    ([0, 1, float('inf')], [1, 1, 1], 128, 4096),
    ([0, 1, 2], [1, float('nan'), 1], 128, 4096),
    ([0, 1, 2], [1, 1, 1], True, 4096),
    ([0, 1, 2], [1, 1, 1], 1, 4096),
    ([0, 1, 2], [1, 1, 1], 128, 1),
    ([0, 1, 2], [1, 1, 1], 128.0, 4096),
])
def test_invalid_joint_moment_inputs_are_rejected(means, errors, paths, reference):
    with pytest.raises(ValueError):
        moment_covariance(means, errors, paths, reference)


def test_fieller_polynomial_matches_direct_studentised_contrast():
    d, e, z = F(7, 3), F(2, 5), F(13, 4)
    covariance = dict(dd=F(3, 8), de=F(-1, 7), ee=F(4, 9))
    q = fieller_coefficients(d, e, covariance, z)
    for w in [F(-3), F(0), F(1, 7), F(1), F(13, 2)]:
        direct = (e-w*d)**2-z*z*(covariance['ee']-2*w*covariance['de']+w*w*covariance['dd'])
        assert q['a']*w*w+q['b']*w+q['c'] == direct
    scale = F(-11, 3)
    scaled = fieller_coefficients(scale*d, scale*e, {k: scale**2*v for k, v in covariance.items()}, z)
    assert scaled == {k: scale**2*v for k, v in q.items()}


def test_zero_covariance_inversion_retains_signed_denominator_and_exact_ratio():
    covariance = dict(dd=0, de=0, ee=0)
    q = fieller_coefficients(-4, -1, covariance)
    assert rational_parts(quadratic_set(**q)) == [(F(1, 4), F(1, 4))]
    assert quadratic_set(**fieller_coefficients(0, 1, covariance))['physical_kind'] == 'empty'
    assert quadratic_set(**fieller_coefficients(0, 0, covariance))['physical_kind'] == 'full_domain'
    assert quadratic_set(**fieller_coefficients(1, 2, covariance))['physical_kind'] == 'empty'


@pytest.mark.parametrize('covariance,z', [
    (dict(dd=-1, de=0, ee=1), 2),
    (dict(dd=1, de=0, ee=-1), 2),
    (dict(dd=1, de=2, ee=1), 2),
    (dict(dd=1, de=0, ee=1), 0),
    (dict(dd=1, de=0, ee=1), -1),
    (dict(dd=1, de=float('nan'), ee=1), 2),
])
def test_invalid_fieller_covariance_or_cutoff_is_rejected(covariance, z):
    with pytest.raises(ValueError):
        fieller_coefficients(1, 1, covariance, z)


def rational_parts(result):
    assert all(endpoint['kind'] == 'rational' for part in result['components'] for endpoint in part.values())
    return [(part['lower']['value'], part['upper']['value']) for part in result['components']]


@pytest.mark.parametrize('a,b,c,unrestricted,kind,parts', [
    (0, 0, 1, 'empty', 'empty', []),
    (0, 0, 0, 'all_real', 'full_domain', [(0, 1)]),
    (0, 0, -1, 'all_real', 'full_domain', [(0, 1)]),
    (0, 2, -1, 'half_line', 'interval', [(0, F(1, 2))]),
    (0, -2, 1, 'half_line', 'interval', [(F(1, 2), 1)]),
    (0, 1, 1, 'half_line', 'empty', []),
    (0, -1, 2, 'half_line', 'empty', []),
    (0, 1, -2, 'half_line', 'full_domain', [(0, 1)]),
    (0, -1, -1, 'half_line', 'full_domain', [(0, 1)]),
    (0, 1, 0, 'half_line', 'singleton', [(0, 0)]),
    (0, -1, 1, 'half_line', 'singleton', [(1, 1)]),
    (1, 0, 1, 'empty', 'empty', []),
    (-1, 0, -1, 'all_real', 'full_domain', [(0, 1)]),
    (4, -4, 1, 'singleton', 'singleton', [(F(1, 2), F(1, 2))]),
    (1, 0, 0, 'singleton', 'singleton', [(0, 0)]),
    (1, -2, 1, 'singleton', 'singleton', [(1, 1)]),
    (1, -4, 4, 'singleton', 'empty', []),
    (-4, 4, -1, 'all_real', 'full_domain', [(0, 1)]),
    (16, -16, 3, 'bounded_interval', 'interval', [(F(1, 4), F(3, 4))]),
    (-16, 16, -3, 'two_rays', 'two_intervals', [(0, F(1, 4)), (F(3, 4), 1)]),
    (1, -1, 0, 'bounded_interval', 'full_domain', [(0, 1)]),
    (-1, 1, 0, 'two_rays', 'two_intervals', [(0, 0), (1, 1)]),
    (1, 3, 2, 'bounded_interval', 'empty', []),
    (-1, -3, -2, 'two_rays', 'full_domain', [(0, 1)]),
    (1, 0, -1, 'bounded_interval', 'full_domain', [(0, 1)]),
    (-1, 0, 1, 'two_rays', 'singleton', [(1, 1)]),
])
def test_complete_exact_quadratic_topology(a, b, c, unrestricted, kind, parts):
    result = quadratic_set(a, b, c)
    assert result['unrestricted_kind'] == unrestricted
    assert result['physical_kind'] == kind
    assert result['discriminant'] == b*b-4*a*c
    assert rational_parts(result) == parts


def test_tiny_nonzero_discriminant_and_degree_are_not_rounded_to_zero():
    epsilon = F(1, 2**1000)
    positive = quadratic_set(1, -1, F(1, 4)-epsilon**2)
    assert positive['physical_kind'] == 'interval'
    assert rational_parts(positive) == [(F(1, 2)-epsilon, F(1, 2)+epsilon)]
    assert quadratic_set(1, -1, F(1, 4)+epsilon**2)['physical_kind'] == 'empty'
    assert rational_parts(quadratic_set(epsilon, -epsilon, 0)) == [(0, 1)]
    assert rational_parts(quadratic_set(0, epsilon, -epsilon/3)) == [(0, F(1, 3))]


def test_irrational_roots_remain_exact_algebraic_endpoints():
    # Roots are 1/2 +/- sqrt(1/8), both inside [0,1].
    result = quadratic_set(1, -1, F(1, 8))
    assert result['physical_kind'] == 'interval'
    lower, upper = (result['components'][0][name] for name in ('lower', 'upper'))
    assert lower == dict(kind='root', centre=F(1, 2), radius_squared=F(1, 8), sign=-1)
    assert upper == dict(kind='root', centre=F(1, 2), radius_squared=F(1, 8), sign=1)
    assert endpoint_compare(lower, F(1, 8)) == 1
    assert endpoint_compare(lower, F(1, 4)) == -1
    assert endpoint_compare(upper, F(3, 4)) == 1
    assert endpoint_compare(upper, F(7, 8)) == -1
    disconnected = quadratic_set(-1, 1, F(-1, 8))
    assert disconnected['physical_kind'] == 'two_intervals'
    assert disconnected['components'][0]['upper'] == lower
    assert disconnected['components'][1]['lower'] == upper


@pytest.mark.parametrize('sign', [-1, 1])
def test_algebraic_endpoint_comparison_is_exact_near_rational_boundary(sign):
    epsilon = F(1, 2**2000)
    for squared, expected in [(F(9, 16)-epsilon, -1), (F(9, 16), 0), (F(9, 16)+epsilon, 1)]:
        endpoint = dict(kind='root', centre=F(1, 4), radius_squared=squared, sign=sign)
        assert endpoint_compare(endpoint, F(1, 4)+sign*F(3, 4)) == sign*expected
    root = dict(kind='root', centre=F(1, 4), radius_squared=F(2), sign=sign)
    assert endpoint_compare(root, F(1, 4)) == sign
    assert endpoint_compare(root, F(1, 4)-sign) == sign
    rational = dict(kind='rational', value=F(3, 7))
    assert endpoint_compare(rational, F(3, 7)) == 0
    assert endpoint_compare(rational, F(3, 7)+epsilon) == -1
    assert endpoint_compare(rational, F(3, 7)-epsilon) == 1


@pytest.mark.parametrize('a,b,c,lower,upper,include_lower,include_upper,expected', [
    (1, 0, 0, 0, 1, False, True, False),
    (1, 0, 0, 0, 1, True, False, True),
    (1, -2, 1, 0, 1, True, False, False),
    (1, -2, 1, 0, 1, False, True, True),
    (-1, 1, 0, 0, 1, False, False, False),
    (-1, 1, 0, 0, 1, True, False, True),
    (-1, 1, 0, 0, 1, False, True, True),
    (-1, 0, 0, 0, 1, False, False, True),
    (0, 0, 0, 0, 1, False, False, True),
    (0, 0, 1, 0, 1, True, True, False),
    (1, F(-2, 7), F(1, 49), 0, 1, False, False, True),
    (1, F(-2, 7), F(1, 49)+F(1, 2**1000), 0, 1, True, True, False),
    (1, F(-2, 7), F(1, 49)-F(1, 2**1000), 0, 1, False, False, True),
    (0, 1, -1, 0, 1, False, False, True),
    (0, 1, 0, 0, 1, False, False, False),
    (0, -1, 1, 0, 1, False, False, False),
    (1, -1, 0, F(1, 2), F(1, 2), True, True, True),
    (1, -1, 0, F(1, 2), F(1, 2), False, True, False),
    (1, -1, 0, F(1, 2), F(1, 2), True, False, False),
])
def test_quadratic_intersection_respects_open_boundaries_and_vertices(
        a, b, c, lower, upper, include_lower, include_upper, expected):
    assert intersects(a, b, c, lower, upper, include_lower=include_lower, include_upper=include_upper) is expected


def rows(*pairs):
    return [dict(evaluation_paths=n, repetitions=r) for n, r in pairs]


def test_disconnected_ratio_set_does_not_fill_an_unaccepted_gap():
    candidates = rows((1024, 4), (2048, 3), (4096, 2))
    # The middle allocation owns (1/5,1/2]. Accepted shares instead lie
    # in [0,1/10] union [4/5,1], so a convex-hull shortcut is incorrect.
    result = allocation_choices(-1, F(9, 10), F(-2, 25), candidates)
    assert result == dict(weakly_compatible_sizes=[1024, 4096], selected_sizes=[1024, 4096], uniformly_selected_size=None)


def test_irrational_disconnected_set_is_intersected_without_rounding_its_endpoints():
    candidates = rows((1024, 4), (2048, 3), (4096, 2))
    result = allocation_choices(-1, 1, F(-1, 8), candidates)
    assert result['weakly_compatible_sizes'] == result['selected_sizes'] == [1024, 4096]
    bounded = allocation_choices(1, -1, F(1, 8), candidates)
    assert bounded['weakly_compatible_sizes'] == bounded['selected_sizes'] == [1024, 2048, 4096]


def test_singleton_tie_distinguishes_weak_compatibility_from_selection():
    candidates = rows((1024, 4), (2048, 3), (4096, 2))
    result = allocation_choices(1, F(-2, 5), F(1, 25), candidates)
    assert result == dict(weakly_compatible_sizes=[1024, 2048], selected_sizes=[1024], uniformly_selected_size=1024)
    triple = rows((512, 9), (1024, 5), (2048, 3), (4096, 2))
    result = allocation_choices(4, -4, 1, triple)
    assert result['weakly_compatible_sizes'] == [512, 1024, 2048, 4096]
    assert result['selected_sizes'] == [512]
    assert result['uniformly_selected_size'] == 512


def allocation_oracle(candidates, polynomial, rational_roots):
    """Partition all crossings and polynomial roots, then test each cell."""
    a, b, c = polynomial
    value = lambda row, w: ((1-w)+F(4096, row['evaluation_paths'])*w)/row['repetitions']
    critical = {F(0), F(1)} | {F(x) for x in rational_roots if 0 <= x <= 1}
    for left, right in combinations(candidates, 2):
        at_zero = value(left, 0)-value(right, 0)
        at_one = value(left, 1)-value(right, 1)
        if at_zero != at_one:
            crossing = at_zero/(at_zero-at_one)
            if 0 <= crossing <= 1:
                critical.add(crossing)
    ordered = sorted(critical)
    probes = critical | {(x+y)/2 for x, y in zip(ordered, ordered[1:])}
    weak, selected = set(), set()
    for w in probes:
        if a*w*w+b*w+c > 0:
            continue
        values = [(value(row, w), row['evaluation_paths']) for row in candidates]
        minimum = min(x for x, _ in values)
        ties = {n for x, n in values if x == minimum}
        weak.update(ties)
        selected.add(min(ties))
    return dict(weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
                uniformly_selected_size=next(iter(selected)) if len(selected) == 1 else None)


@pytest.mark.parametrize('pairs', [
    [(1024, 4), (2048, 3), (4096, 2)],
    [(512, 9), (1024, 5), (2048, 3), (4096, 2)],
    [(128, 666), (256, 420), (4096, 93)],
    [(1024, 2), (2048, 2), (4096, 2)],
    [(1024, 8), (2048, 4), (4096, 2)],
    [(1, 7), (3, 5), (17, 3), (4096, 1)],
    [(128, 3)],
])
@pytest.mark.parametrize('polynomial,roots', [
    ((0, 0, -1), []),
    ((0, 0, 1), []),
    ((0, 0, 0), []),
    ((0, 5, -1), [F(1, 5)]),
    ((0, -5, 1), [F(1, 5)]),
    ((1, 0, 0), [0]),
    ((1, -2, 1), [1]),
    ((4, -4, 1), [F(1, 2)]),
    ((1, F(-2, 5), F(1, 25)), [F(1, 5)]),
    ((-1, 1, 0), [0, 1]),
    ((16, -16, 3), [F(1, 4), F(3, 4)]),
    ((-16, 16, -3), [F(1, 4), F(3, 4)]),
    ((-1, F(9, 10), F(-2, 25)), [F(1, 10), F(4, 5)]),
])
def test_allocation_sets_match_independent_crossing_and_interval_oracle(pairs, polynomial, roots):
    candidates = rows(*pairs)
    expected = allocation_oracle(candidates, polynomial, roots)
    assert allocation_choices(*polynomial, candidates) == expected
    assert allocation_choices(*polynomial, candidates[::-1]) == expected


@pytest.mark.parametrize('lower,upper,ownership', [(1, 0, True), (0, 1, 1), (0, 1, None)])
def test_invalid_interval_or_ownership_is_rejected(lower, upper, ownership):
    with pytest.raises(ValueError):
        intersects(1, 0, 0, lower, upper, include_lower=ownership)
