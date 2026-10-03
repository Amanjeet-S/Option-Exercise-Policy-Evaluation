"""Independent polynomial reconstruction and minimax comparison checks."""

from fractions import Fraction as Q
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_prospective_allocation as audit
from qf_research.optimal_stopping.prospective_allocation import (
    sample_summary, compare_summaries, minimise_relative_variance)


@pytest.mark.parametrize('values', [
    [Q(0),Q(1),Q(3)],
    [Q(0),Q(0),Q(0),Q(4)],
    [Q(-2),Q(-1),Q(1),Q(2)],
    [Q(17,11)]*7,
    [Q(10000)+Q(i,10**12) for i in range(9)],
    [Q(i*i-5*i,7) for i in range(19)],
])
def test_raw_polynomial_reconstruction(values):
    assert audit.raw_moments(values) == sample_summary(values)


def test_leave_one_retains_original_grand_mean_count():
    values = list(map(Q,[0,0,0,4]))
    result = audit.raw_moments(values)
    assert result['leave_one_variance_at_fixed_count'] == dict(
        lower=Q(0),upper=Q(4,3),lower_row=3,upper_row=0)
    assert result['variance_of_mean'] == 1
    assert result['variance_of_variance'] == Q(1,4)


@pytest.mark.parametrize('difference,variance,status', [
    (Q(13,4),Q(1),'includes_zero'),
    (Q(13,4)+Q(1,10**20),Q(1),'positive'),
    (-Q(13,4)-Q(1,10**20),Q(1),'negative'),
    (Q(8),Q(0),'degenerate'),
    (Q(0),Q(0),'degenerate'),
])
def test_signed_classification_keeps_boundary_and_degeneracy(difference,variance,status):
    assert audit.signed_interval(difference,variance)['normal_interval_status'] == status


def test_signed_independent_comparisons():
    left = audit.raw_moments(list(map(Q,[0,1,2,5,12])))
    right = audit.raw_moments(list(map(Q,[2,2,2,3,3,7])))
    assert audit.comparison(left,right) == compare_summaries(left,right)
    reversed_record = audit.comparison(right,left)
    assert reversed_record['variance_contrast']['difference'] == -audit.comparison(left,right)['variance_contrast']['difference']
    assert reversed_record['variance_contrast']['estimated_sampling_variance'] == audit.comparison(left,right)['variance_contrast']['estimated_sampling_variance']


@pytest.mark.parametrize('intervals', [
    [(Q(0),Q(1))],[(Q(1,100),Q(1,50))],
    [(Q(0),Q(0)),(Q(1),Q(1))],
    [(Q(1,4),Q(1,4))],[(Q(0),Q(1,5)),(Q(7,10),Q(4,5))]
])
def test_pairwise_minimax_matches_envelope_route(intervals):
    candidates = [dict(evaluation_paths=n,repetitions=r) for n,r in
                  [(128,520),(256,450),(4096,70)]]
    assert audit.minimax(candidates,intervals) == minimise_relative_variance(
        candidates,[dict(lower=lo,upper=hi) for lo,hi in intervals])


def test_allocation_ties_use_smaller_path_count():
    candidates = [dict(evaluation_paths=256,repetitions=100),
                  dict(evaluation_paths=128,repetitions=100)]
    result = audit.minimax(candidates,[(Q(0),Q(0))])
    assert result['minimisers'] == [128,256]
    assert result['selected']['evaluation_paths'] == 128


@pytest.mark.parametrize('value',['1/2','0','-9/11'])
def test_canonical_rational(value):
    assert str(audit.q(value)) == value


@pytest.mark.parametrize('value',['0/1','2/4','1.0',True,1])
def test_noncanonical_rational_rejected(value):
    with pytest.raises(ValueError):
        audit.q(value)


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    path = tmp_path/'record.json'
    for value in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            audit.read(path)


def test_evidence_rejects_stale_hash_and_unbound_files(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    folder = tmp_path/'run'
    folder.mkdir()
    source = tmp_path/'source.py'
    source.write_text('x = 1\n')
    audit.save(folder/'result.json',dict(complete=True))
    audit.save(folder/'manifest.json',dict(
        source_sha256={'source.py':audit.sha(source)},
        artifact_sha256={'result.json':audit.sha(folder/'result.json')}))
    evidence = audit.Evidence()
    evidence.manifest(folder)
    assert evidence.bound(folder/'result.json') == dict(complete=True)
    source.write_text('x = 2\n')
    with pytest.raises(ValueError,match='changed during audit'):
        evidence.unchanged()
    with pytest.raises(ValueError,match='checksum differs'):
        audit.Evidence().manifest(folder)
    source.write_text('x = 1\n')
    (folder/'unbound.json').write_text('{}')
    with pytest.raises(ValueError,match='Unbound or missing'):
        audit.Evidence().manifest(folder)


def test_evidence_rejects_path_escape(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    folder = tmp_path/'run'
    folder.mkdir()
    (tmp_path/'outside.json').write_text('{}')
    with pytest.raises(ValueError,match='escapes declared root'):
        audit.Evidence().mapping({'../outside.json':audit.sha(tmp_path/'outside.json')},folder)


def test_stream_roles_and_paths_are_reconstructed():
    first = audit.expected_stream(2718,(31,5,128,0,0),5000)
    second = audit.expected_stream(2718,(31,5,128,0,1),5000)
    assert first != second
    assert first['stream_id'] != second['stream_id']
    assert first['spawn_key'] == [31,5,128,0,0]
    assert first['paths'] == 5000
    assert len(first['stream_id']) == 64
