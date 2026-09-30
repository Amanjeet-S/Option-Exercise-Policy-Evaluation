"""Main-run audit on fresh, small mathematical inputs and deliberate alterations."""

from contextlib import contextmanager
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_training_allocation_main as audit
import study_training_allocation as study
from test_training_allocation_study import registration


@contextmanager
def changed_artifact(run, name, change):
    """Change fixture evidence and its hash to exercise algebraic checks."""
    target = run/name
    original, manifest_bytes = target.read_bytes(), (run/'manifest.json').read_bytes()
    try:
        record = study.read(target)
        change(record)
        study.save(target, record)
        manifest = study.read(run/'manifest.json')
        manifest['artifact_sha256'][name] = study.sha(target)
        study.save(run/'manifest.json', manifest)
        yield
    finally:
        target.write_bytes(original)
        (run/'manifest.json').write_bytes(manifest_bytes)


def test_complete_main_audit_and_independent_alteration_rejection(tmp_path, monkeypatch):
    registered, cfg = registration(tmp_path, monkeypatch)
    pilot, plan, run = tmp_path/'pilot', tmp_path/'main_registration', tmp_path/'main'
    study.run_pilot(registered, pilot)
    study.allocate(pilot, plan)
    study.run_main(plan, run)
    original_hashes = {str(path.relative_to(run)): study.sha(path)
                       for path in run.rglob('*.json')}
    registration_hashes = {str(path.relative_to(plan)): study.sha(path)
                           for path in plan.rglob('*') if path.is_file()}
    result = audit.audit(run, tmp_path/'audit')
    count = sum(row['arms'][arm]['allocation']['repetitions']
                for row in study.read(run/'summary.json')['models'] for arm in audit.ARMS)
    assert result['checks']['blocks'] == count
    assert result['checks']['independent_training_streams'] == 2*count
    assert result['checks']['independent_evaluation_streams'] == count
    assert len(result['repeated_pairs']) == 8
    assert {(row['case'], row['arm']) for row in result['repeated_pairs']} == {
        (case, arm) for case in (0, 12) for arm in audit.ARMS}
    assert result['alteration_check']['rejected']
    assert not result['alteration_check']['audit_output_created']
    assert not list((tmp_path/'audit').glob('alteration_check_*'))
    assert {str(path.relative_to(run)): study.sha(path)
            for path in run.rglob('*.json')} == original_hashes
    assert {str(path.relative_to(plan)): study.sha(path)
            for path in plan.rglob('*') if path.is_file()} == registration_hashes
    manifest = study.read(tmp_path/'audit/manifest.json')
    study.verify_mapping(manifest['artifact_sha256'], tmp_path/'audit')
    assert len(manifest['artifact_sha256']) == 9
    assert len(manifest['additional_audit_source_sha256']) == 2
    assert manifest['registration_unchanged']
    assert result['main_manifest_sha256'] == study.sha(run/'manifest.json')
    assert not result['verification_plan']['part_of_original_experiment_registration']
    assert not result['verification_plan']['main_outcomes_inspected_when_selection_fixed']

    target = 'blocks/case_000_reference_rep_0000.json'
    def changed_seed(row):
        row['evaluation']['stream']['spawn_key'][-1] += 1
    with changed_artifact(run, target, changed_seed):
        with pytest.raises(ValueError, match='seed provenance'):
            audit.verify(run)
    with changed_artifact(run, target, lambda row: row['evaluation'].update(evaluation_paths=17)):
        with pytest.raises(ValueError, match='evaluation path count'):
            audit.verify(run)
    with changed_artifact(run, target, lambda row: row['training']['policies'][0].update(requested_degree=7)):
        with pytest.raises(ValueError, match='training specification'):
            audit.verify(run)

    # Extra blocks are outside the frozen counts even without a manifest entry.
    extra = run/'blocks/unexpected.json'
    extra.write_text('{}')
    with pytest.raises(ValueError, match='block-file coverage'):
        audit.verify(run)
    extra.unlink()

    model = study.read(run/'model_000.json')['arms']['reference']
    frozen = study.read(plan/'allocations.json')['models'][0]['allocation']
    blocks = [study.read(run/'blocks'/f'case_000_reference_rep_{r:04d}.json')
              for r in range(model['allocation']['repetitions'])]
    for location, key in (('inference', 'lower'), ('cost', 'standalone_cpu_seconds'),
                          ('influence', 'maximum_absolute_mean_change')):
        altered = copy.deepcopy(model)
        if location == 'inference':
            altered['inference']['estimate'][key] += 1.
        elif location == 'influence':
            altered['influence'][key] += 1.
        else:
            altered[key] += 1.
        with pytest.raises(ValueError, match='Independent reconstruction'):
            audit.reconstruct_arm(blocks, altered, cfg, frozen['reference'], frozen['budget']['cpu_seconds'])

    # Raw corruption must fail before a new output directory is created.
    path = run/target
    path.write_text(path.read_text()+' ')
    rejected = tmp_path/'rejected_audit'
    with pytest.raises(ValueError, match='checksum'):
        audit.audit(run, rejected)
    assert not rejected.exists()


def test_stream_checker_rejects_prior_reuse_on_a_small_fixture(tmp_path, monkeypatch):
    registered, cfg = registration(tmp_path, monkeypatch)
    base = study.load_frozen(study.read(registered/'tasks.json')[0]['source_policy'])[0]
    # A tiny independently constructed block exercises role and stream checks.
    policies, training = study.fit(base, cfg, 20, 0, 0, 0)
    approximation, setup = study.build_approximation(base, **cfg['approximation'])
    evaluation = study.evaluate_pair(policies, approximation, 64, study.seed(cfg, 21, 0, 0, 0))
    block = dict(case=0, arm='reference', repetition=0, training=training, evaluation=evaluation)
    reused = {evaluation['stream']['stream_id']}
    with pytest.raises(ValueError, match='stream reused'):
        audit.check_block(block, cfg, base, 0, 'reference', 0, 64,
                          setup['approximation_fingerprint'], reused)
