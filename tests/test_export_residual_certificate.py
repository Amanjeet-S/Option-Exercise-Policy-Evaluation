"""Reject malformed private certification inputs without research simulation."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_residual_certificate import (
    APPROXIMATION_FIELDS, CONVENTION, case_coverage, hexadecimal_leaves,
    merge_hashes, validate_document, validate_value,
)


def stipulated_document():
    """An exact zero-volatility put fixture, not an empirical observation."""
    value = dict(knots=[0., 1.], values=[1., 0.], tail_slope=0.)
    spec = dict(model=dict(quoted_spot=1., volatility=0., initial_share_factor=1.,
                          premium_factor=1., dividends=[]),
        option=dict(strike=1., maturity=1., kind='put'),
        decisions=[dict(time=1., settlement_factor=1., phase='after')],
        nodes=[dict(time=0., before=None, after=value),
               dict(time=1., before=None, after=value)],
        core_steps=800, wing_steps=40, compression_tolerance=.001, initial_value=0.)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    row = dict(**spec, approximation_fingerprint=fingerprint,
        matching_arm_fingerprints=dict(reference=fingerprint, selected=fingerprint),
        residual_fine_fingerprint=fingerprint, exact_binary64_hex=hexadecimal_leaves(spec))
    return dict(schema_version=1, exact_input_convention=CONVENTION,
        approximation_fields=list(APPROXIMATION_FIELDS),
        models=[dict(case=case, **copy.deepcopy(row)) for case in range(13)])


def test_full_document_survives_json_roundtrip():
    validate_document(json.loads(json.dumps(stipulated_document())))


@pytest.mark.parametrize('cases', [[], list(range(12)), [0]*13, list(reversed(range(13))),
                                  [False, *range(1, 13)]])
def test_incomplete_duplicate_or_reordered_cases_rejected(cases):
    with pytest.raises(ValueError, match='thirteen cases'):
        case_coverage([dict(case=case) for case in cases], 'test')


def test_changed_value_rejected_by_fingerprint():
    document = stipulated_document()
    document['models'][0]['nodes'][0]['after']['values'][0] = 1.1
    with pytest.raises(ValueError, match='fingerprint'):
        validate_document(document)


def test_changed_hex_or_retained_identity_rejected():
    document = stipulated_document()
    document['models'][0]['exact_binary64_hex']['/model/volatility'] = '0x1.0p-1'
    with pytest.raises(ValueError, match='hexadecimal'):
        validate_document(document)
    document = stipulated_document()
    document['models'][0]['matching_arm_fingerprints']['selected'] = '0'*64
    with pytest.raises(ValueError, match='identity'):
        validate_document(document)


@pytest.mark.parametrize('value', [
    dict(knots=[], values=[], tail_slope=0.),
    dict(knots=[0., 0.], values=[1., 0.], tail_slope=0.),
    dict(knots=[1., 2.], values=[1., 0.], tail_slope=0.),
    dict(knots=[0., 1.], values=[float('nan'), 0.], tail_slope=0.),
    dict(knots=[0., 1.], values=[1., 0.], tail_slope=.1),
])
def test_invalid_interpolant_rejected(value):
    with pytest.raises(ValueError):
        validate_value(value)


def test_nonmonotone_values_are_permitted():
    validate_value(dict(knots=[0., 1., 2., 3.], values=[2., 1., 3., 0.], tail_slope=0.))


def test_conflicting_source_history_rejected():
    assert merge_hashes({'a': '1'*64}, {'a': '1'*64}) == {'a': '1'*64}
    with pytest.raises(ValueError, match='Conflicting'):
        merge_hashes({'a': '1'*64}, {'a': '2'*64})


def test_nonfinite_and_signed_zero_binary64_identity():
    assert hexadecimal_leaves({'x': -0.}) == {'/x': '-0x0.0p+0'}
    with pytest.raises(ValueError, match='Non-finite'):
        hexadecimal_leaves({'x': float('inf')})
