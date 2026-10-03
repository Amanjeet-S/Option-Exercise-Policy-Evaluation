"""Private reproduction checks on declared mathematical fixtures, not market data."""

import copy
import math
from pathlib import Path
import shutil
from statistics import median
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_training_allocation_main as audit
import build_training_allocation_paper_assets as assets
import reproduce_training_allocation as replay
import report_training_allocation as report
import study_training_allocation as study
from test_training_allocation_study import registration


def completed_fixture(tmp_path, monkeypatch):
    """Execute the existing small mathematical experiment and bind its evidence."""
    original_root = replay.ROOT
    registered, cfg = registration(tmp_path, monkeypatch)
    original_verify = replay.verify_mapping

    def guarded(path):
        path = Path(path).resolve()
        if not path.is_relative_to((tmp_path/'data/private').resolve()):
            raise ValueError('Keep fixture evidence private')
        return path

    for module in (assets, report, replay):
        monkeypatch.setattr(module, 'ROOT', tmp_path)
        # The retained fixture's registration is deliberately outside data/private.
        monkeypatch.setattr(module, 'private', guarded if module is replay else lambda p: Path(p).resolve())
        monkeypatch.setattr(module, 'verify_mapping', lambda mapping, base=None:
                            original_verify(mapping, tmp_path if base is None else base))
    source_names = (
        'scripts/study_training_allocation.py', 'scripts/audit_training_allocation.py',
        'scripts/audit_training_allocation_main.py', 'scripts/audit_training_evaluation.py',
        'scripts/validate_funded_dual.py', 'scripts/run_conditional_calibration.py',
        'scripts/review_training_inference.py', 'scripts/report_training_allocation.py',
        'scripts/build_training_allocation_paper_assets.py',
        'scripts/reproduce_training_allocation.py',
        'src/qf_research/optimal_stopping/allocation_workload.py',
        'src/qf_research/optimal_stopping/funded_policy.py',
        'tests/test_training_allocation_main_audit.py',
        'tests/test_training_allocation_reproduction.py')
    for name in source_names:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root/name, target)
    for module in (audit, assets, report, replay):
        monkeypatch.setattr(module, '__file__', str(tmp_path/'scripts'/Path(module.__file__).name))
    root = tmp_path/'data/private/declared_study'
    root.mkdir(parents=True)
    pilot, plan = root/'pilot_v1', root/'main_registration_v1'
    main, checked = root/'main_run_v1', root/'main_audit_v1'
    study.run_pilot(registered, pilot)
    study.allocate(pilot, plan)
    study.run_main(plan, main)
    audit.audit(main, checked)
    report.report(main, checked, root/'main_report_v1')
    components = []
    # Construct a retained diagnostic from observed fixture workload records.
    for model in study.read(main/'summary.json')['models']:
        block = study.read(pilot/'blocks'/f'case_{model["case"]:03d}.json')
        for arm, record in model['arms'].items():
            allocation = record['allocation']
            count, paths = allocation['repetitions'], allocation['evaluation_paths']
            candidates = [(block['statistical'][row['repetition']]['training']['cpu_seconds']
                           + row['evaluation']['cpu_seconds'], row['repetition'], row['evaluation'])
                          for row in block['timing'] if row['evaluation']['evaluation_paths'] == paths]
            _, index, evaluation = sorted(candidates)[len(candidates)//2]
            observed = [study.read(main/'blocks'/f'case_{model["case"]:03d}_{arm}_rep_{r:04d}.json')
                        for r in range(count)]
            components.append(dict(case=model['case'], arm=arm, median_pair_repetition=index,
                predicted_cpu_seconds=dict(setup=median(row['cpu_seconds'] for row in block['setup']),
                    training=count*block['statistical'][index]['training']['cpu_seconds'],
                    evaluation=count*evaluation['cpu_seconds']),
                actual_cpu_seconds=dict(setup=record['setup']['cpu_seconds'],
                    training=math.fsum(row['training']['cpu_seconds'] for row in observed),
                    evaluation=math.fsum(row['evaluation']['cpu_seconds'] for row in observed))))
    totals = {arm: {kind: {part: math.fsum(row[kind][part] for row in components if row['arm'] == arm)
                          for part in ('setup', 'training', 'evaluation')}
                    for kind in ('predicted_cpu_seconds', 'actual_cpu_seconds')}
              for arm in audit.ARMS}
    timing = root/'timing_review_v1'
    timing.mkdir()
    inputs = {str((folder/'manifest.json').relative_to(tmp_path)): study.sha(folder/'manifest.json')
              for folder in (pilot, main)}
    diagnostic = dict(components=components, totals=totals, input_sha256=inputs,
                      source_sha256={}, publication_clearance=False)
    study.save(timing/'diagnostic.json', diagnostic)
    study.save(timing/'manifest.json', dict(complete=True, publication_clearance=False,
        input_sha256=inputs, source_sha256={},
        artifact_sha256={'diagnostic.json': study.sha(timing/'diagnostic.json')}))
    return root


def test_complete_reproduction_and_rejection_before_output(tmp_path, monkeypatch):
    root = completed_fixture(tmp_path, monkeypatch)
    hashes = {str(path.relative_to(root)): study.sha(path)
              for path in root.rglob('*') if path.is_file()}
    prepared = replay.preflight(root)
    output = tmp_path/'data/private/notebook_reproduction'
    result = replay.reproduce(root, output)
    assert result['values'] == study.read(root/'main_report_v1/values.json')
    assert result['verification']['checked_main_pairs'] == prepared['checks']['blocks']
    assert result['verification']['repeated_pairs'] == prepared['retained_repeated_pairs']
    assert all(row['exact_numerical_match'] for row in result['verification']['repeated_pairs'])
    assert not result['verification']['full_numerical_rerun']
    assert not result['verification']['timings_expected_to_match']
    assert result['verification']['alteration_check']['rejected']
    assert {str(path.relative_to(root)): study.sha(path)
            for path in root.rglob('*') if path.is_file()} == hashes
    manifest = study.read(output/'manifest.json')
    replay.verify_mapping(manifest['artifact_sha256'], output)
    replay.verify_mapping(manifest['input_sha256'])
    replay.verify_mapping(manifest['source_sha256'])
    assert len(list((output/'pair_audit').glob('repeated_*.json'))) == len(prepared['retained_repeated_pairs'])

    with pytest.raises(FileExistsError, match='fresh'):
        replay.reproduce(root, output)
    for bad in (tmp_path/'public', root/'main_run_v1/new', root/'timing_review_v1/new'):
        with pytest.raises(ValueError, match='private|separate'):
            replay.reproduce(root, bad)
        assert not bad.exists()
    missing = tmp_path/'data/private/missing'
    rejected = tmp_path/'data/private/missing_rejected'
    with pytest.raises(ValueError, match='missing'):
        replay.reproduce(missing, rejected)
    assert not rejected.exists()

    # Altering private evidence must fail before creating even an empty output.
    target = root/'main_report_v1/values.json'
    original = target.read_bytes()
    target.write_bytes(original+b' ')
    rejected = tmp_path/'data/private/altered_rejected'
    with pytest.raises(ValueError, match='checksum'):
        replay.reproduce(root, rejected)
    assert not rejected.exists()
    target.write_bytes(original)

    # A self-consistent diagnostic file hash cannot conceal false cost accounting.
    diagnostic_path = root/'timing_review_v1/diagnostic.json'
    diagnostic = study.read(diagnostic_path)
    diagnostic['components'][0]['actual_cpu_seconds']['training'] += 1.
    study.save(diagnostic_path, diagnostic)
    timing_manifest = study.read(root/'timing_review_v1/manifest.json')
    timing_manifest['artifact_sha256']['diagnostic.json'] = study.sha(diagnostic_path)
    study.save(root/'timing_review_v1/manifest.json', timing_manifest)
    rejected = tmp_path/'data/private/accounting_rejected'
    with pytest.raises(ValueError, match='Independent reconstruction'):
        replay.reproduce(root, rejected)
    assert not rejected.exists()


def test_exact_pair_comparison_excludes_only_recorded_workload_times():
    retained = dict(case=2, arm='selected', repetition=3,
        training=dict(cpu_seconds=1., elapsed_seconds=2., phases=dict(fitting_cpu_seconds=.4),
                      policy_fingerprints=['first', 'second']),
        evaluation=dict(cpu_seconds=.5, elapsed_seconds=.8, paths_sha256='path-evidence',
                        gain=dict(value=.125, standard_error=.01)))
    repeated = copy.deepcopy(retained)
    repeated['training'].update(cpu_seconds=7., elapsed_seconds=8., phases=dict(fitting_cpu_seconds=3.))
    repeated['evaluation'].update(cpu_seconds=4., elapsed_seconds=5.)
    assert audit.numerical_record(retained) == audit.numerical_record(repeated)
    repeated['evaluation']['gain']['value'] += 1e-15
    assert audit.numerical_record(retained) != audit.numerical_record(repeated)
    repeated = copy.deepcopy(retained)
    repeated['training']['policy_fingerprints'][0] = 'different-fit'
    assert audit.numerical_record(retained) != audit.numerical_record(repeated)
