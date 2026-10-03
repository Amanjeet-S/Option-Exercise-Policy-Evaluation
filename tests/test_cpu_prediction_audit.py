"""Exact comparison and temporal-order failures in the separate timing audit."""

from fractions import Fraction
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_cpu_prediction import F, precedes, require_equal


def test_binary64_audit_retains_addition_error_instead_of_rounding_it_away():
    actual = str(F(.1)+F(.2))
    require_equal(actual, Fraction(.1)+Fraction(.2), 'two raw timers')
    with pytest.raises(ValueError, match='Reconstruction differs'):
        require_equal(str(F(.1+.2)), Fraction(.1)+Fraction(.2), 'rounded sum')


def test_nested_exact_audit_compares_all_timer_components():
    require_equal(dict(cpu='7/3', elapsed='5/2'), dict(cpu=Fraction(7, 3), elapsed=Fraction(5, 2)), 'charge')
    with pytest.raises(ValueError, match='charge'):
        require_equal(dict(cpu='7/3', elapsed='2'), dict(cpu=Fraction(7, 3), elapsed=Fraction(5, 2)), 'charge')


@pytest.mark.parametrize('invalid', [-1, '-1/10', float('nan'), float('inf')])
def test_nonfinite_and_negative_recorded_durations_are_rejected(invalid):
    with pytest.raises((ValueError, OverflowError)):
        F(invalid)


def test_forecast_must_precede_validation_in_absolute_time():
    precedes('2026-09-28T13:00:00+01:00', '2026-09-28T12:00:01+00:00')
    with pytest.raises(ValueError, match='postdates'):
        precedes('2026-09-28T13:00:02+01:00', '2026-09-28T12:00:01+00:00')


def test_equal_recorded_instants_are_accepted():
    precedes('2026-09-28T12:00:00+00:00', '2026-09-28T12:00:00+00:00')
