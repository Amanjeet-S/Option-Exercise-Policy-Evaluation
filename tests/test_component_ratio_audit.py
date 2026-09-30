"""Negative controls for the independent raw-moment and sign-cell audit."""

from fractions import Fraction as F
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ratio_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_component_ratio.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_raw_polynomial_moments_retain_signed_component():
    result = audit.raw_moments([F(1), F(2), F(4)], [F(1), F(2), F(3)], 4096)
    assert result['observed_variance'] == F(7, 3)
    assert result['evaluation_mean_variance'] == F(14, 3)
    assert result['A'] == F(-7, 3)
    assert result['point_share'] == 2
    assert result['point_in_parameter_domain'] is False
    assert result['covariance'] == dict(dd=F(49,81), de=F(49,54), ee=F(49,9))
    assert result['covariance_rank'] == 2


def test_raw_moments_translation_and_scale_invariance():
    x, se = list(map(F, [1, 2, 4])), list(map(F, [1, 2, 3]))
    source = audit.raw_moments(x, se, 256)
    translated = audit.raw_moments([w+10**20 for w in x], se, 256)
    assert translated == source
    scaled = audit.raw_moments([3*w for w in x], [3*s for s in se], 256)
    for key in ('observed_variance','evaluation_mean_variance','A','B','denominator','numerator'):
        assert scaled[key] == 9*source[key]
    assert scaled['covariance'] == {k:81*v for k,v in source['covariance'].items()}
    assert scaled['largest_squared_influence_share'] == source['largest_squared_influence_share']


@pytest.mark.parametrize('coefficient,expected', [
    ((0,0,0),'full_domain'), ((0,0,1),'empty'),
    ((1,-1,F(1,4)),'singleton'), ((-1,1,F(-3,16)),'two_intervals'),
    ((1,-1,F(-1,4)),'full_domain'), ((0,1,F(-1,2)),'interval'),
    ((1,-3,2),'singleton'), ((1,-5,6),'empty')])
def test_topology_all_degrees(coefficient, expected):
    result = audit.topology(*map(F, coefficient))
    assert result['physical_kind'] == expected


@pytest.mark.parametrize('coefficient,lower,upper,expected', [
    ((0,0,0),0,1,True), ((0,0,1),0,1,False),
    ((1,0,0),0,1,False), ((-1,0,0),0,1,True),
    ((1,-1,F(1,4)),0,1,True),
    ((1,-1,F(1,4)),0,F(1,2),False),
    ((1,-1,F(1,4)),F(1,2),1,False),
    ((-1,1,0),0,1,False), ((0,1,-1),0,1,True)])
def test_open_cell_equality_and_boundary_ownership(coefficient,lower,upper,expected):
    assert audit.open_cell_has_point(*map(F, coefficient), F(lower), F(upper)) is expected


def test_exact_tie_selection_is_not_weak_compatibility():
    grid = [dict(evaluation_paths=128,repetitions=128), dict(evaluation_paths=4096,repetitions=32)]
    point = F(3,31)
    result, _, _ = audit.comparison(F(1), -2*point, point*point, grid)
    assert result == dict(weakly_compatible_sizes=[128,4096],selected_sizes=[128],uniformly_selected_size=128)


def test_empty_region_maps_to_no_allocation():
    result, _, _ = audit.comparison(F(0), F(0), F(1), [dict(evaluation_paths=128,repetitions=1)])
    assert result == dict(weakly_compatible_sizes=[],selected_sizes=[],uniformly_selected_size=None)


def test_rational_serialisation_is_canonical():
    assert audit.q('1/2') == F(1,2)
    for bad in ('2/4', '0.5', 1, float('nan')):
        with pytest.raises(ValueError):
            audit.q(bad)
