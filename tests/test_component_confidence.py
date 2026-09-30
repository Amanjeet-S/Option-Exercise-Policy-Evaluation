"""Exact mathematical fixtures for bounded variance-component regions.

These stipulated examples check geometry and decisions. They are not market
observations and do not establish statistical coverage of supplied radii.
"""

from fractions import Fraction
from itertools import combinations

import pytest

from qf_research.optimal_stopping.component_confidence import (
    decision_summary,
    region,
)


def pairs(result):
    return [(row['v'], row['u']) for row in result['vertices']]


def hull(points):
    """Remove repeated and collinear points from a clipped convex polygon."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def clip(polygon, signed_distance):
    """Intersect polygon edges with one half-plane, without line enumeration."""
    if not polygon:
        return []
    output = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        dp, dq = signed_distance(p), signed_distance(q)
        if dp <= 0:
            output.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp-dq)
            output.append(tuple(x+t*(y-x) for x, y in zip(p, q)))
    return hull(output)


def clipped_region(s, u, rv, ru, cap, *, n=4096, delta=Fraction(1, 1040), ev=0, eu=0):
    """Independent rectangle clipping oracle for the bounded moment region."""
    s, u, rv, ru, cap, ev, eu = map(Fraction, (s, u, rv, ru, cap, ev, eu))
    vl, vu = delta*max(0, s-ev), s+ev+rv
    ul, uu = delta*max(0, u-eu), u+eu+ru
    polygon = hull([(vl, ul), (vl, uu), (vu, ul), (vu, uu)])
    polygon = clip(polygon, lambda p: p[1]-p[0])
    polygon = clip(polygon, lambda p: p[0]+(n-1)*p[1]-cap)
    return sorted(polygon)


def candidates(*items):
    return [dict(evaluation_paths=n, repetitions=r) for n, r in items]


def criterion(row, share):
    return (1-share+Fraction(4096, row['evaluation_paths'])*share) / row['repetitions']


def decision_oracle(rows, lower, upper, retained):
    """Partition at all pairwise crossings and inspect each open cell."""
    boundaries = {Fraction(lower), Fraction(upper)}
    for a, b in combinations(rows, 2):
        d0 = criterion(a, Fraction(0))-criterion(b, Fraction(0))
        d1 = criterion(a, Fraction(1))-criterion(b, Fraction(1))
        if d0 != d1:
            w = -d0/(d1-d0)
            if lower <= w <= upper:
                boundaries.add(w)
    ordered = sorted(boundaries)
    probes = boundaries | {(a+b)/2 for a, b in zip(ordered, ordered[1:])}
    weak, selected, ratios = set(), set(), {}
    target = next(row for row in rows if row['evaluation_paths'] == retained)
    for w in probes:
        values = {row['evaluation_paths']: criterion(row, w) for row in rows}
        best = min(values.values())
        minimisers = {n for n, value in values.items() if value == best}
        weak.update(minimisers)
        selected.add(min(minimisers))
        ratios[w] = criterion(target, w)/best
    maximum = max(ratios.values())
    return dict(
        weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
        uniformly_selected_size=next(iter(selected)) if len(selected) == 1 else None,
        retained_choice_regret=dict(
            maximum=maximum, lower_ratio=ratios[lower], upper_ratio=ratios[upper],
            worst_endpoints=[w for w in sorted({Fraction(lower), Fraction(upper)})
                             if ratios[w] == maximum],
        ),
    )


def test_hand_derived_pentagon_and_component_mapping():
    actual = region(2, 1, 2, 2, 5, pilot_paths=2, tail_probability=Fraction(1, 2))
    assert actual['intervals'] == {'v': [Fraction(1), Fraction(4)], 'u': [Fraction(1, 2), Fraction(3)]}
    assert pairs(actual) == [(1, Fraction(1, 2)), (1, 1), (Fraction(5, 2), Fraction(5, 2)),
                             (4, Fraction(1, 2)), (4, 1)]
    assert actual['share_bounds'] == {'lower': Fraction(1, 8), 'upper': Fraction(1)}
    assert actual['status'] == 'nondegenerate'
    assert actual['zero_variance_possible'] is False
    for row in actual['vertices']:
        assert all(type(row[key]) is Fraction for key in ('v', 'u', 'A', 'B'))
        assert row['A'] == row['v']-row['u'] >= 0
        assert row['B'] == 2*row['u'] >= 0
        assert row['A']+row['B']/2 == row['v']
        assert row['A']+row['B'] <= 5


def test_component_support_cuts_rectangle_at_nontrivial_pilot_size():
    actual = region(4, 1, 4, 2, 8, pilot_paths=4, tail_probability=Fraction(1, 2))
    assert pairs(actual) == [(2, Fraction(1, 2)), (2, 2), (Fraction(13, 2), Fraction(1, 2))]
    assert actual['share_bounds'] == {'lower': Fraction(1, 13), 'upper': Fraction(1)}
    assert all(row['A']+row['B'] <= 8 for row in actual['vertices'])


@pytest.mark.parametrize('s,u,rv,ru,cap,n,delta,ev,eu', [
    (2, 1, 2, 2, 5, 2, Fraction(1, 2), 0, 0),
    (4, 1, 4, 2, 8, 4, Fraction(1, 2), 0, 0),
    (1, 2, 4, 3, 9, 7, Fraction(1, 3), 0, 0),
    (3, 2, 1, 2, 200, 4096, Fraction(1, 1040), 0, 0),
    (0, 0, 0, 0, 2, 2, Fraction(1, 2), 0, 0),
    (0, 0, 3, 2, 7, 3, Fraction(1, 2), 0, 0),
    (2, 0, 0, 0, 3, 2, Fraction(1, 2), 0, 0),
    (1, 2, 0, 0, 2, 2, Fraction(1, 2), 0, 0),
    (1, 4, 0, 0, 99, 2, Fraction(1, 2), 0, 0),
    (4, 2, 0, 0, 2, 2, Fraction(1, 2), 0, 0),
    (1, 4, 0, 0, 9, 3, Fraction(1, 2), 2, 5),
    (.1, .02, .07, .01, .2, 4, Fraction(1, 1040), .001, .003),
])
def test_exact_vertices_match_independent_polygon_clipping(s, u, rv, ru, cap, n, delta, ev, eu):
    actual = region(s, u, rv, ru, cap, pilot_paths=n, tail_probability=delta,
                    variance_error=ev, evaluation_error=eu)
    expected = clipped_region(s, u, rv, ru, cap, n=n, delta=delta, ev=ev, eu=eu)
    assert pairs(actual) == expected
    ratios = [q/v for v, q in expected if v > 0]
    assert actual['share_bounds'] == ({'lower': min(ratios), 'upper': max(ratios)} if ratios else None)
    assert actual['zero_variance_possible'] == ((0, 0) in expected)
    assert actual['status'] == ('empty' if not expected else 'nondegenerate' if ratios else 'degenerate_zero')


def test_empty_region_is_distinguished_from_only_zero_variance():
    empty = region(1, 4, 0, 0, 99, pilot_paths=2, tail_probability=Fraction(1, 2))
    assert empty['status'] == 'empty'
    assert empty['vertices'] == []
    assert empty['share_bounds'] is None
    assert empty['zero_variance_possible'] is False
    zero = region(0, 0, 0, 0, 99)
    assert zero['status'] == 'degenerate_zero'
    assert pairs(zero) == [(0, 0)]
    assert zero['share_bounds'] is None
    assert zero['zero_variance_possible'] is True


def test_positive_line_and_point_regions_are_retained():
    line = region(2, 0, 0, 0, 3, pilot_paths=2, tail_probability=Fraction(1, 2))
    assert pairs(line) == [(1, 0), (2, 0)]
    assert line['share_bounds'] == {'lower': Fraction(0), 'upper': Fraction(0)}
    point = region(1, 2, 0, 0, 2, pilot_paths=2, tail_probability=Fraction(1, 2))
    assert pairs(point) == [(1, 1)]
    assert point['share_bounds'] == {'lower': Fraction(1), 'upper': Fraction(1)}
    assert line['status'] == point['status'] == 'nondegenerate'


def test_origin_with_positive_variance_does_not_assign_a_share_to_origin():
    actual = region(0, 0, 3, 2, 7, pilot_paths=3)
    assert actual['zero_variance_possible'] is True
    assert actual['status'] == 'nondegenerate'
    assert actual['share_bounds'] == {'lower': Fraction(0), 'upper': Fraction(1)}
    line = region(0, 0, 3, 0, 7, pilot_paths=3)
    assert line['zero_variance_possible'] is True
    assert pairs(line) == [(0, 0), (3, 0)]
    assert line['share_bounds'] == {'lower': Fraction(0), 'upper': Fraction(0)}


def test_every_intermediate_share_has_an_explicit_convex_mixture_witness():
    actual = region(4, 1, 4, 2, 8, pilot_paths=4, tail_probability=Fraction(1, 2))
    rows = actual['vertices']
    p = min(rows, key=lambda row: row['u']/row['v'])
    q = max(rows, key=lambda row: row['u']/row['v'])
    lo, hi = p['u']/p['v'], q['u']/q['v']
    for step in range(11):
        w = lo+(hi-lo)*Fraction(step, 10)
        # Solve u(t)=w*v(t), so this constructs rather than samples the image.
        t = (w*p['v']-p['u']) / ((q['u']-p['u'])-w*(q['v']-p['v']))
        v, u = (1-t)*p['v']+t*q['v'], (1-t)*p['u']+t*q['u']
        assert 0 <= t <= 1
        assert u/v == w
        assert actual['intervals']['v'][0] <= v <= actual['intervals']['v'][1]
        assert actual['intervals']['u'][0] <= u <= actual['intervals']['u'][1]
        assert 0 <= u <= v
        assert v+3*u <= 8


@pytest.mark.parametrize('field', ['variance_radius', 'evaluation_radius', 'variance_error', 'evaluation_error'])
def test_widening_allowances_contains_original_region_and_share_image(field):
    arguments = dict(observed_variance=4, evaluation_mean_variance=1, variance_radius=4,
                     evaluation_radius=2, bound_squared=8, pilot_paths=4,
                     tail_probability=Fraction(1, 2), variance_error=0, evaluation_error=0)
    original = region(**arguments)
    arguments[field] += Fraction(3, 5)
    wider = region(**arguments)
    for v, u in pairs(original):
        assert wider['intervals']['v'][0] <= v <= wider['intervals']['v'][1]
        assert wider['intervals']['u'][0] <= u <= wider['intervals']['u'][1]
    assert wider['share_bounds']['lower'] <= original['share_bounds']['lower']
    assert wider['share_bounds']['upper'] >= original['share_bounds']['upper']


def test_error_allowances_clip_lower_limits_at_zero_and_add_to_upper_limits():
    actual = region(1, 4, 3, 2, 100, pilot_paths=2, tail_probability=Fraction(1, 7),
                    variance_error=2, evaluation_error=5)
    assert actual['intervals'] == {'v': [0, 6], 'u': [0, 11]}
    assert actual['zero_variance_possible'] is True
    assert actual['share_bounds'] == {'lower': Fraction(0), 'upper': Fraction(1)}


@pytest.mark.parametrize('factor', [Fraction(1, 7), Fraction(11), Fraction(10**50)])
def test_common_positive_scaling_preserves_shares_and_scales_components(factor):
    values = [4, 1, 4, 2, 8]
    original = region(*values, pilot_paths=4, tail_probability=Fraction(1, 2),
                      variance_error=Fraction(1, 5), evaluation_error=Fraction(1, 7))
    scaled = region(*(factor*x for x in values), pilot_paths=4, tail_probability=Fraction(1, 2),
                    variance_error=factor/5, evaluation_error=factor/7)
    assert scaled['share_bounds'] == original['share_bounds']
    assert scaled['status'] == original['status']
    assert scaled['vertices'] == [{key: factor*value for key, value in row.items()}
                                  for row in original['vertices']]


def test_binary64_inputs_are_interpreted_as_exact_stored_numbers():
    values = [.1, .02, .07, .01, .2]
    floating = region(*values, pilot_paths=4, variance_error=.001, evaluation_error=.003)
    rational = region(*map(Fraction, values), pilot_paths=4,
                      variance_error=Fraction(.001), evaluation_error=Fraction(.003))
    assert floating == rational


def test_weak_crossing_compatibility_is_distinct_from_deterministic_selection():
    rows = candidates((4096, 2), (1024, 4))
    result = decision_summary(rows, Fraction(1, 3), Fraction(1, 3), 4096)
    assert result['weakly_compatible_sizes'] == [1024, 4096]
    assert result['selected_sizes'] == [1024]
    assert result['uniformly_selected_size'] == 1024
    assert result['retained_choice_regret']['maximum'] == 1
    epsilon = Fraction(1, 2**100)
    assert decision_summary(rows, Fraction(1, 3)+epsilon, 1, 4096)['uniformly_selected_size'] == 4096
    assert decision_summary(rows, 0, Fraction(1, 3), 1024)['uniformly_selected_size'] == 1024
    assert decision_summary(rows, Fraction(1, 3), 1, 4096)['selected_sizes'] == [1024, 4096]


def test_weak_singletons_at_share_endpoints_are_not_spurious_selected_sizes():
    rows = candidates((2048, 4), (4096, 2))
    full = decision_summary(rows, 0, 1, 2048)
    assert full['weakly_compatible_sizes'] == [2048, 4096]
    assert full['selected_sizes'] == [2048]
    assert full['uniformly_selected_size'] == 2048
    at_zero = decision_summary(candidates((2048, 2), (4096, 2)), 0, 0, 4096)
    assert at_zero['weakly_compatible_sizes'] == [2048, 4096]
    assert at_zero['selected_sizes'] == [2048]


@pytest.mark.parametrize('rows,lower,upper,retained', [
    (candidates((1024, 4), (2048, 1), (4096, 2)), 0, 1, 2048),
    (candidates((1024, 4), (4096, 2)), Fraction(1, 3), Fraction(1, 3), 4096),
    (candidates((128, 666), (256, 420), (4096, 93)), 0, 1, 128),
    (candidates((128, 666), (256, 420), (4096, 93)), Fraction(1, 100), Fraction(1, 10), 256),
    (candidates((128, 666), (256, 420), (4096, 93)), Fraction(1, 9), Fraction(1, 9), 4096),
    (candidates((1, 31), (3, 22), (7, 13), (11, 9), (50, 8), (100, 2)), 0, 1, 11),
    (candidates((1024, 5), (2048, 3), (4096, 2)), Fraction(1, 3), Fraction(1, 3), 2048),
    (candidates((128, 64)), 0, 1, 128),
])
def test_decisions_and_regret_match_independent_partition(rows, lower, upper, retained):
    expected = decision_oracle(rows, lower, upper, retained)
    assert decision_summary(rows, lower, upper, retained) == expected
    assert decision_summary(list(reversed(rows)), lower, upper, retained) == expected


@pytest.mark.parametrize('field,bad', [
    *[(key, -1) for key in ('observed_variance', 'evaluation_mean_variance', 'variance_radius',
                            'evaluation_radius', 'bound_squared', 'variance_error', 'evaluation_error')],
    *[(key, float('nan')) for key in ('observed_variance', 'evaluation_mean_variance', 'variance_radius',
                                      'evaluation_radius', 'bound_squared', 'variance_error', 'evaluation_error')],
    *[(key, float('inf')) for key in ('observed_variance', 'evaluation_mean_variance', 'variance_radius',
                                      'evaluation_radius', 'bound_squared', 'variance_error', 'evaluation_error')],
    ('bound_squared', 0), ('observed_variance', True), ('variance_error', False),
    ('evaluation_mean_variance', '1'), ('pilot_paths', 1), ('pilot_paths', 0),
    ('pilot_paths', 2.0), ('pilot_paths', True), ('tail_probability', 0),
    ('tail_probability', 1), ('tail_probability', -1), ('tail_probability', Fraction(3, 2)),
    ('tail_probability', float('nan')), ('tail_probability', float('inf')), ('tail_probability', True),
])
def test_invalid_region_inputs_are_rejected(field, bad):
    kwargs = dict(observed_variance=2, evaluation_mean_variance=1, variance_radius=2,
                  evaluation_radius=2, bound_squared=5, pilot_paths=2,
                  tail_probability=Fraction(1, 2), variance_error=0, evaluation_error=0)
    kwargs[field] = bad
    with pytest.raises((ValueError, TypeError)):
        region(**kwargs)


@pytest.mark.parametrize('rows', [
    [], candidates((0, 1)), candidates((-1, 1)), candidates((1, 0)), candidates((1, -1)),
    candidates((1.0, 1)), candidates((1, 2.0)), candidates((True, 1)), candidates((1, False)),
    candidates((1, 2), (1, 3)), candidates((1, 2), (1, 2)),
])
def test_invalid_decision_candidates_are_rejected(rows):
    with pytest.raises((ValueError, TypeError)):
        decision_summary(rows, 0, 1, 1)


@pytest.mark.parametrize('lower,upper', [
    (-1, 1), (0, 2), (1, 0), (float('nan'), 1), (0, float('inf')), (False, 1), (0, True),
])
def test_invalid_share_intervals_are_rejected(lower, upper):
    with pytest.raises((ValueError, TypeError)):
        decision_summary(candidates((1, 2), (2, 1)), lower, upper, 1)


@pytest.mark.parametrize('target', [3, 0, -1, 1.0, True])
def test_absent_or_invalid_retained_size_is_rejected(target):
    with pytest.raises((ValueError, TypeError)):
        decision_summary(candidates((1, 2), (2, 1)), 0, 1, target)
