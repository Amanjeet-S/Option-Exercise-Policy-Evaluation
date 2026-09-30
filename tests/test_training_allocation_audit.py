"""Independent audit checks on a small declared mathematical pilot."""

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_training_allocation as audit
import study_training_allocation as study
from test_training_allocation_study import registration


def test_complete_independent_audit_and_temporary_tamper_probe(tmp_path, monkeypatch):
    registered, cfg = registration(tmp_path, monkeypatch)
    pilot = tmp_path/'pilot'
    study.run_pilot(registered, pilot)
    original_hashes = {str(path.relative_to(pilot)): study.sha(path)
                       for path in pilot.rglob('*.json')}
    output = tmp_path/'audit'
    result = audit.audit(pilot, output)
    assert len(result['models']) == 13
    assert [(row['case'], row['repetition']) for row in result['repeated_pairs']] == [
        (0, 0), (0, cfg['pilot_repetitions']-1),
        (12, 0), (12, cfg['pilot_repetitions']-1)]
    assert result['alteration_check']['rejected']
    assert not result['alteration_check']['allocation_output_created']
    assert not list(output.glob('alteration_check_*'))
    assert {str(path.relative_to(pilot)): study.sha(path)
            for path in pilot.rglob('*.json')} == original_hashes
    manifest = study.read(output/'manifest.json')
    study.verify_mapping(manifest['artifact_sha256'], output)
    assert len(manifest['artifact_sha256']) == 5
    assert not result['main_outcomes_generated']

    # The independent calculation must reject a changed selector result even
    # when it is supplied directly, outside the hash-verification wrapper.
    row = study.read(pilot/'blocks/case_000.json')
    retained = study.read(pilot/'summary.json')['models'][0]
    altered = copy.deepcopy(retained)
    altered['allocation']['selected']['repetitions'] += 1
    with pytest.raises(ValueError, match='Independent reconstruction'):
        audit.reconstruct_model(row, cfg, altered)
    altered = copy.deepcopy(retained)
    altered['pilot_inference']['training_variance_moment_estimate'] += 1.
    with pytest.raises(ValueError, match='Independent reconstruction'):
        audit.reconstruct_model(row, cfg, altered)

    # Rejection must precede creation of either an audit or allocation output.
    target = pilot/'blocks/case_000.json'
    target.write_text(target.read_text()+' ')
    for action, destination in ((audit.audit, tmp_path/'rejected_audit'),
                                (study.allocate, tmp_path/'rejected_allocation')):
        with pytest.raises(ValueError, match='checksum'):
            action(pilot, destination)
        assert not destination.exists()


def test_numerical_record_normalises_containers_and_removes_only_timings():
    original = dict(policies=({'coefficients': (1., 2.)},), cpu_seconds=1.,
                    elapsed_seconds=2., phases=dict(fitting_cpu_seconds=.2),
                    evaluation=dict(cpu_seconds=.5, paths=128,
                                    stream=dict(spawn_key=(10, 1, 0))))
    assert audit.numerical_record(original) == dict(
        policies=[dict(coefficients=[1., 2.])],
        evaluation=dict(paths=128, stream=dict(spawn_key=[10, 1, 0])))
