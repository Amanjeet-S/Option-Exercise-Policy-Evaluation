"""Stipulated mathematical and file-system fixtures for exact replay comparison.

No historical archive, market observation or provider account is used.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'scripts/compare_licensed_replay.py'
SPEC = importlib.util.spec_from_file_location('public_replay_comparator_tests', TOOL)
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def seal(folder):
    manifest = {'complete': True, 'completed_at': 'stipulated',
                'source_sha256': {'src/model.py': 'a' * 64},
                'input_sha256': {'data/private/fixture.json': 'b' * 64}}
    manifest['artifact_sha256'] = {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.rglob('*.json') if p.name != 'manifest.json'}
    save(folder / 'manifest.json', manifest)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(COMPARATOR, 'ROOT', tmp_path)
    root = tmp_path / 'data/private/fixture_archive'
    root.mkdir(parents=True)
    for label in ('original', 'fresh'):
        folder = root / label
        save(folder / 'rows.json', [
            {'case': 0, 'policies': [{'value': 1.25, 'seed': 123,
                'policy_fingerprint': 'frozen_policy',
                'approximation_fingerprint': 'frozen_approximation',
                'cpu_seconds': 0.25}]},
            {'case': 1, 'policies': [{'value': 2.5, 'seed': 456,
                'policy_fingerprint': 'second_policy',
                'approximation_fingerprint': 'frozen_approximation',
                'cpu_seconds': 0.5}]},
        ])
        save(folder / 'launch.json', {'started_at': 'stipulated',
                                     'registration_sha256': 'c' * 64})
        save(folder / 'configuration.json', {'paths': 8, 'cpu_seconds': 2.0})
        save(folder / 'tasks.json', [{'case': 0}, {'case': 1}])
        seal(folder)
    yield root

def mutate(case, filename, change, *, reseal=True):
    path = case / 'fresh' / filename
    data = json.loads(path.read_text())
    change(data)
    save(path, data)
    if reseal:
        seal(case / 'fresh')


def run(case, study='value_martingale'):
    return COMPARATOR.run(study, case/'original', case/'fresh', case/'comparison')


def rejected(record):
    return not record['complete'] or not record['numerical_agreement']


def test_identical_stipulated_records_pass(case):
    record = run(case)
    assert record['complete'] and record['numerical_agreement']
    assert record['original_coverage']['model_rows'] == 2


def test_self_comparison_cannot_support_new_execution(case):
    record = COMPARATOR.run('value_martingale', case/'original', case/'original', case/'comparison')
    assert record['numerical_agreement']
    assert record['same_run_directories']
    assert not record['distinct_run_manifests']
    assert record['execution_provenance_independently_required']


def test_new_timer_values_are_allowed(case):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(cpu_seconds=0.75))
    assert run(case)['numerical_agreement']


@pytest.mark.parametrize('field,new', [('value', 1.2501), ('seed', 124),
    ('policy_fingerprint', 'changed'), ('approximation_fingerprint', 'changed')])
def test_changed_financial_or_random_identity_fails(case, field, new):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update({field: new}))
    assert rejected(run(case))


@pytest.mark.parametrize('study', ['value_martingale', 'training_allocation'])
def test_one_ulp_financial_difference_fails_exact_comparison(case, study):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(value=math.nextafter(1.25, 2.0)))
    record = run(case, study)
    assert rejected(record)
    if study == 'training_allocation':
        assert record['counts']['roundoff_candidate_leaves'] == 1


@pytest.mark.parametrize('field', ['value', 'cpu_seconds'])
def test_missing_retained_or_excluded_field_fails(case, field):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].pop(field))
    assert rejected(run(case))


def test_boolean_does_not_replace_numeric_seed(case):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(seed=True))
    assert rejected(run(case))


@pytest.mark.parametrize('field', ['paths', 'cpu_seconds'])
def test_registered_design_fields_are_not_timing_exclusions(case, field):
    mutate(case, 'configuration.json', lambda x: x.update({field: 3.0}))
    assert rejected(run(case))


@pytest.mark.parametrize('mapping', ['source_sha256', 'input_sha256'])
def test_changed_manifest_provenance_fails(case, mapping):
    def change(data):
        key = next(iter(data[mapping]))
        data[mapping][key] = 'd' * 64
    mutate(case, 'manifest.json', change, reseal=False)
    assert rejected(run(case))


@pytest.mark.parametrize('change', ['delete', 'duplicate', 'reorder'])
def test_missing_extra_or_reordered_rows_fail(case, change):
    def update(rows):
        if change == 'delete': rows.pop()
        elif change == 'duplicate': rows.append(dict(rows[0]))
        else: rows.reverse()
    mutate(case, 'rows.json', update)
    assert rejected(run(case))


@pytest.mark.parametrize('manifested', [False, True])
def test_extra_json_artifact_fails(case, manifested):
    save(case/'fresh/blocks/extra.json', {'stipulated_extra_row': 1})
    if manifested: seal(case/'fresh')
    assert rejected(run(case))


def test_missing_declared_artifact_fails(case):
    (case/'fresh/tasks.json').unlink()
    assert rejected(run(case))


def test_unsealed_financial_change_fails_hash_check(case):
    mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(value=9.0), reseal=False)
    assert rejected(run(case))


@pytest.mark.parametrize('marker', ['incomplete.json', 'failure.json'])
def test_failure_markers_prevent_success(case, marker):
    save(case/'fresh'/marker, {'stipulated_failure': True})
    assert rejected(run(case))


def test_explicit_incomplete_manifest_fails(case):
    mutate(case, 'manifest.json', lambda x: x.update(complete=False), reseal=False)
    assert rejected(run(case))


def test_symlink_manifest_is_rejected(case):
    path=case/'fresh/manifest.json'; target=case/'held_manifest.json'
    path.rename(target); path.symlink_to(target)
    assert rejected(run(case))


def test_symlink_input_ancestor_is_rejected(case):
    linked=case/'link'; linked.symlink_to(case/'fresh', target_is_directory=True)
    with pytest.raises(ValueError):
        COMPARATOR.run('value_martingale', case/'original', linked, case/'comparison')


def test_similar_private_name_is_not_an_authorised_output(case):
    target=case.parent.parent/'private-public/comparison'
    with pytest.raises(ValueError):
        COMPARATOR.run('value_martingale', case/'original', case/'fresh', target)
    assert not target.exists()


def test_ancestor_output_symlink_is_rejected(case):
    linked=case/'output_alias'; linked.symlink_to(case/'fresh', target_is_directory=True)
    with pytest.raises(ValueError):
        COMPARATOR.run('value_martingale', case/'original', case/'fresh', linked/'comparison')


def test_change_after_initial_artifact_check_is_detected(case, monkeypatch):
    original=COMPARATOR.checked_manifest
    seen=[]
    def after_check(folder):
        result=original(folder)
        seen.append(folder)
        if folder==case/'fresh' and len(seen)==2:
            mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(cpu_seconds=0.75), reseal=False)
        return result
    monkeypatch.setattr(COMPARATOR, 'checked_manifest', after_check)
    assert rejected(run(case))


def test_cpu_digest_exclusion_does_not_claim_unsaved_values(case):
    for label in ('original', 'fresh'):
        folder=case/label
        save(folder/'pass_0/validation/row_0000.json', {
            'phase': 'validation', 'policy_fingerprints': ['stipulated_policy'],
            'paths_sha256': 'a'*64, 'stream': {'id':'stipulated_stream'},
            'complete_financial_record_sha256': ('b' if label=='original' else 'c')*64})
        seal(folder)
    record=run(case, 'cpu_prediction')
    assert record['numerical_agreement']
    assert record['original_coverage']['financial_estimate_values_retained']==0
    assert any('omits financial estimates' in text for text in record['scope_limitations'])


@pytest.mark.parametrize('identical', [True, False])
def test_unmanifested_checkpoint_must_duplicate_final_rows_exactly(case, identical):
    rows=case/'fresh/rows.json'; partial=case/'fresh/partial_rows.json'
    partial.write_bytes(rows.read_bytes())
    if not identical:
        data=json.loads(partial.read_text()); data.pop(); save(partial,data)
    record=run(case)
    assert record['numerical_agreement'] is identical
    if identical:
        assert record['unmanifested_duplicate_checkpoint']['fresh']


def test_resealed_manifest_changed_after_initial_pass_fails(case, monkeypatch):
    original=COMPARATOR.checked_manifest
    seen=[]
    def after_check(folder):
        result=original(folder)
        seen.append(folder)
        if folder==case/'fresh' and len(seen)==2:
            mutate(case, 'rows.json', lambda x: x[0]['policies'][0].update(cpu_seconds=0.75))
        return result
    monkeypatch.setattr(COMPARATOR, 'checked_manifest', after_check)
    assert rejected(run(case))


@pytest.mark.parametrize('field', ['policy_fingerprints','paths_sha256','stream'])
def test_cpu_retained_identity_change_fails(case, field):
    name='pass_0/validation/row_0000.json'
    for label in ('original','fresh'):
        save(case/label/name, {'phase':'validation',
            'policy_fingerprints':['stipulated_policy'],
            'paths_sha256':'a'*64,'stream':{'id':'stipulated_stream'},
            'complete_financial_record_sha256':('b' if label=='original' else 'c')*64})
        seal(case/label)
    updates={'policy_fingerprints':['changed_policy'],'paths_sha256':'d'*64,
             'stream':{'id':'changed_stream'}}
    mutate(case,name,lambda x:x.update({field:updates[field]}))
    assert rejected(run(case,'cpu_prediction'))
