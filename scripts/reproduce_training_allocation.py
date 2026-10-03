"""Reconstruct private allocation findings and repeat prescribed complete pairs.

No empirical observations are embedded here. The supplied study must retain its
completed main run, report, audit and timing review. Numerical reproduction uses
the recorded random streams; process and elapsed times are not equality targets.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from statistics import median

import audit_training_allocation_main as main_audit
import build_training_allocation_paper_assets as assets
import report_training_allocation as report
from audit_training_allocation import _compare
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping

FOLDERS = dict(main='main_run_v1', report='main_report_v1',
               audit='main_audit_v1', timing='timing_review_v1')
COMPONENTS = ('setup', 'training', 'evaluation')


def _timing_components(main, pilot, configuration):
    """Reconstruct additive costs using the observation supplying the median."""
    components = []
    repetitions = configuration['timing_repetitions']
    if repetitions < 1 or repetitions % 2 != 1:
        raise ValueError('The retained timing decomposition requires an odd timing sample')
    for model in read(main/'summary.json')['models']:
        case = model['case']
        block = read(pilot/'blocks'/f'case_{case:03d}.json')
        timings = {(row['repetition'], row['evaluation']['evaluation_paths']): row['evaluation']
                   for row in block['timing']}
        for arm in main_audit.ARMS:
            record = model['arms'][arm]
            allocation = record['allocation']
            count, paths = allocation['repetitions'], allocation['evaluation_paths']
            costs = sorted((block['statistical'][r]['training']['cpu_seconds']
                            + timings[r, paths]['cpu_seconds'], r)
                           for r in range(repetitions))
            pair_cost, index = costs[repetitions//2]
            _compare(pair_cost, allocation['pair_cpu_seconds'], 'median pair cost')
            predicted = dict(setup=median(row['cpu_seconds'] for row in block['setup']),
                training=count*block['statistical'][index]['training']['cpu_seconds'],
                evaluation=count*timings[index, paths]['cpu_seconds'])
            records = [read(main/'blocks'/f'case_{case:03d}_{arm}_rep_{r:04d}.json')
                       for r in range(count)]
            actual = dict(setup=record['setup']['cpu_seconds'],
                training=math.fsum(row['training']['cpu_seconds'] for row in records),
                evaluation=math.fsum(row['evaluation']['cpu_seconds'] for row in records))
            _compare(sum(predicted.values()), allocation['total_cpu_seconds'], 'predicted cost')
            _compare(sum(actual.values()), record['standalone_cpu_seconds'], 'actual cost')
            components.append(dict(case=case, arm=arm, median_pair_repetition=index,
                predicted_cpu_seconds=predicted, actual_cpu_seconds=actual))
    totals = {arm: {category: {component: math.fsum(row[category][component]
                            for row in components if row['arm'] == arm)
                             for component in COMPONENTS}
                    for category in ('predicted_cpu_seconds', 'actual_cpu_seconds')}
              for arm in main_audit.ARMS}
    return components, totals


def preflight(study_root):
    """Validate retained evidence and reconstruct all main inference without writing."""
    study_root = private(Path(study_root))
    folders = {name: study_root/folder for name, folder in FOLDERS.items()}
    if not study_root.is_dir() or any(not path.is_dir() for path in folders.values()):
        raise ValueError('The private allocation study is missing completed evidence folders')
    values, diagnostic, cfg, _, _, inputs, sources, evidence = assets.preflight(
        folders['report'], folders['audit'], folders['timing'])
    if private(ROOT/evidence['main_run']) != folders['main']:
        raise ValueError('The report identifies a different main run from the supplied study')
    _, _, reconstructed, report_inputs, report_sources = report.preflight(
        folders['main'], folders['audit'])
    if values != reconstructed:
        raise ValueError('The complete report differs from reconstructed main inference')
    components, totals = _timing_components(
        folders['main'], private(ROOT/evidence['pilot']), cfg)
    order = lambda row: (row['case'], row['arm'])
    _compare(sorted(diagnostic['components'], key=order), sorted(components, key=order),
             'timing components')
    _compare(diagnostic['totals'], totals, 'timing totals')
    for original, extra in ((inputs, report_inputs), (sources, report_sources)):
        for name, digest in extra.items():
            if name in original and original[name] != digest:
                raise ValueError('Reproduction provenance contains inconsistent hashes')
            original[name] = digest
    for path in (Path(__file__).resolve(), ROOT/'tests/test_training_allocation_reproduction.py'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    verify_mapping(inputs)
    verify_mapping(sources)
    verification = read(folders['audit']/'verification.json')
    return dict(study_root=str(study_root.relative_to(ROOT)), values=values,
        diagnostic=diagnostic, configuration=cfg, checks=verification['checks'],
        retained_repeated_pairs=verification['repeated_pairs'],
        retained_verification_plan=verification['verification_plan'],
        input_sha256=inputs, source_sha256=sources, evidence=evidence,
        reconstructed_full_main_inference=True, reconstructed_report_exactly=True,
        reconstructed_timing_components=True, publication_clearance=False)


def reproduce(study_root, output):
    """Repeat the established complete-pair selection in a fresh private directory."""
    study_root, output = private(Path(study_root)), private(Path(output))
    if output.exists():
        raise FileExistsError('Use a fresh private reproduction directory')
    for name in FOLDERS.values():
        retained = (study_root/name).resolve()
        if output.is_relative_to(retained) or retained.is_relative_to(output):
            raise ValueError('Keep reproduction output separate from retained evidence')
    prepared = preflight(study_root)
    # The pilot and registration may be stored outside the supplied study folder.
    for name in prepared['input_sha256']:
        retained = (ROOT/name).resolve()
        if retained.is_relative_to(output) or output.is_relative_to(retained.parent):
            raise ValueError('Keep reproduction output separate from retained evidence')
    if any(os.environ.get(key) != '1' for key in main_audit.study.THREADS):
        raise ValueError('Use the registered numerical-library thread environment')
    output.mkdir(parents=True)
    replay = main_audit.audit(private(ROOT/prepared['evidence']['main_run']), output/'pair_audit')
    _compare(replay['checks'], prepared['checks'], 'replayed main inference')
    _compare(replay['repeated_pairs'], prepared['retained_repeated_pairs'], 'retained pair selection')
    _compare(replay['verification_plan'], prepared['retained_verification_plan'], 'retained verification plan')
    verify_mapping(prepared['input_sha256'])
    verify_mapping(prepared['source_sha256'])
    save(output/'values.json', prepared['values'])
    save(output/'timing_diagnostic.json', prepared['diagnostic'])
    result = dict(study_root=prepared['study_root'], evidence=prepared['evidence'],
        reconstructed_full_main_inference=True, reconstructed_report_exactly=True,
        reconstructed_timing_components=True, checked_main_pairs=prepared['checks']['blocks'],
        repeated_pairs=replay['repeated_pairs'], selection=replay['verification_plan']['selection'],
        selection_from_retained_audit=True, full_numerical_rerun=False,
        timings_expected_to_match=False, costs_reconstructed_from_retained_measurements=True,
        alteration_check=replay['alteration_check'], approximate_intervals=True,
        measured_mean_squared_error=False, validated_interval_coverage=False,
        validated_equal_budget=False, independent_market_validation=False,
        publication_clearance=False)
    save(output/'verification.json', result)
    artefacts = {str(path.relative_to(output)): sha(path)
                for path in sorted(output.rglob('*')) if path.is_file()}
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        complete=True, input_sha256=prepared['input_sha256'],
        source_sha256=prepared['source_sha256'], artifact_sha256=artefacts,
        full_numerical_rerun=False, publication_clearance=False))
    return dict(values=prepared['values'], diagnostic=prepared['diagnostic'],
                verification=result, output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = reproduce(arguments.study, arguments.output)
    print('Reconstructed retained findings and reproduced '
          f'{len(result["verification"]["repeated_pairs"])} complete pairs; output remains private.')
