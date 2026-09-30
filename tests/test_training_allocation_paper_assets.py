"""Paper-asset consistency and evidence rejection on synthetic retained records."""

import base64
import copy
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import build_training_allocation_paper_assets as assets


def fixture(tmp_path, monkeypatch):
    source = Path(assets.__file__).resolve()
    local_source = tmp_path/'scripts'/source.name
    local_source.parent.mkdir()
    local_source.write_bytes(source.read_bytes())
    monkeypatch.setattr(assets, '__file__', str(local_source))
    monkeypatch.setattr(assets, 'ROOT', tmp_path)
    monkeypatch.setattr(assets, 'private', lambda path: Path(path).resolve())
    original_verify = assets.verify_mapping
    monkeypatch.setattr(assets, 'verify_mapping', lambda mapping, base=None:
                        original_verify(mapping, tmp_path if base is None else base))
    folders = {name: tmp_path/name for name in ('pilot', 'registration', 'main', 'audit', 'report', 'timing')}
    for folder in folders.values():
        folder.mkdir()
    def manifest(name, **extra):
        folder = folders[name]
        record = dict(input_sha256={}, source_sha256={},
                      artifact_sha256={p.name: assets.sha(p) for p in folder.iterdir() if p.is_file()},
                      complete=True, publication_clearance=False)
        record.update(extra)
        assets.save(folder/'manifest.json', record)
    cfg = dict(training_paths=[5000, 20000], degree=4, budget_ratio_limits=[.9, 1.1],
               pilot_repetitions=32, pilot_paths=4096, reference_repetitions=64, reference_paths=4096)
    manifest('pilot', process_cpu_seconds=10., elapsed_seconds=15.)
    assets.save(folders['registration']/'configuration.json', cfg)
    manifest('registration', stage='main_allocation', pilot='pilot')
    models, components = [], []
    for case in range(13):
        arms = {}
        for name in assets.ARMS:
            reference = name == 'reference'
            count, paths = (64, 4096) if reference else (256+case, 256)
            cpu, elapsed, se = (3., 4., .1) if reference else (2., 3., .05)
            arms[name] = dict(
                allocation=dict(repetitions=count, evaluation_paths=paths,
                                predicted_variance=.01 if reference else .005, total_cpu_seconds=6.),
                inference=dict(estimate=dict(value=1., standard_error=se, lower=1.-2*se,
                                              upper=1.+2*se, confidence=.95)),
                standalone_cpu_seconds=cpu, standalone_elapsed_seconds=elapsed,
                actual_to_budget_ratio=cpu/6., material_budget_mismatch=True, zero_gain_sample_pairs=0)
            components.append(dict(case=case, arm=name,
                predicted_cpu_seconds=dict(setup=1., training=2., evaluation=3.),
                actual_cpu_seconds=(dict(setup=1., training=1., evaluation=1.) if reference
                                    else dict(setup=.5, training=.75, evaluation=.75))))
        models.append(dict(case=case, arms=arms))
    summary = dict(models=models)
    assets.save(folders['main']/'summary.json', summary)
    manifest('main', process_cpu_seconds=30., elapsed_seconds=40., registration='registration',
             registration_sha256=assets.sha(folders['registration']/'manifest.json'))
    main_hash = assets.sha(folders['main']/'manifest.json')
    assets.save(folders['audit']/'verification.json', dict(main_run='main', main_manifest_sha256=main_hash))
    manifest('audit', input_sha256={'main/manifest.json': main_hash}, registration_unchanged=True,
             registered_source_sha256={}, additional_audit_source_sha256={}, audit_source_sha256={})
    values = assets.report_values(summary, cfg)
    values['main_stage_execution'] = dict(process_cpu_seconds=30., elapsed_seconds=40.)
    values['evidence'] = dict(main_run='main', main_manifest_sha256=main_hash, audit='audit',
                              audit_manifest_sha256=assets.sha(folders['audit']/'manifest.json'))
    assets.save(folders['report']/'values.json', values)
    (folders['report']/'allocation_comparison.png').write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXs8AAAAASUVORK5CYII='))
    manifest('report', main_run='main', audit='audit')
    timing_inputs = {'main/manifest.json': main_hash,
                     'pilot/manifest.json': assets.sha(folders['pilot']/'manifest.json')}
    totals = {arm: {category: {component: sum(row[category][component] for row in components if row['arm'] == arm)
                               for component in assets.COMPONENTS}
                    for category in ('predicted_cpu_seconds', 'actual_cpu_seconds')} for arm in assets.ARMS}
    assets.save(folders['timing']/'diagnostic.json', dict(components=components, totals=totals,
        input_sha256=timing_inputs, source_sha256={}, publication_clearance=False))
    manifest('timing', input_sha256=timing_inputs)
    return folders


def test_consistent_assets_and_rejection_before_output_creation(tmp_path, monkeypatch):
    folders = fixture(tmp_path, monkeypatch)
    output = tmp_path/'assets'
    result = assets.build(folders['report'], folders['audit'], folders['timing'], output)
    assert result['macros']['AllocationModels'] == 13
    assert result['macros']['AllocationSelectedPairs'] == sum(256+case for case in range(13))
    assert result['macros']['AllocationTotalPairs'] == 64*13+sum(256+case for case in range(13))
    assert result['macros']['AllocationBudgetFailures'] == 26
    assert result['macros']['AllocationPositiveReferenceIntervals'] == 13
    assert all(re.fullmatch(r'Allocation[A-Za-z]+', name) for name in result['macros'])
    assert len(result['comparison_rows']) == 13 and len(result['cost_rows']) == 3
    assert len(result['gain_rows']) == 13
    definitions = (output/'allocation_values.tex').read_text()
    assert definitions.count(r'\newcommand') == len(result['macros'])
    assert '\\newcommand{\\AllocationTotalPairs}{'+str(result['macros']['AllocationTotalPairs'])+'}' in definitions
    assert assets.sha(output/'figures/allocation_comparison.png') == assets.sha(folders['report']/'allocation_comparison.png')
    manifest = assets.read(output/'manifest.json')
    assets.verify_mapping(manifest['artifact_sha256'], output)
    assert len(manifest['artifact_sha256']) == 7
    assert not manifest['numerical_reruns']

    # A self-consistent file hash cannot conceal an inconsistent decomposition.
    values, diagnostic, cfg, pilot, main, *_ = assets.preflight(folders['report'], folders['audit'], folders['timing'])
    altered = copy.deepcopy(diagnostic)
    altered['components'][0]['actual_cpu_seconds']['training'] += 1.
    with pytest.raises(ValueError, match='evidence differs'):
        assets.derive(values, altered, cfg, pilot, main)

    figure = folders['report']/'allocation_comparison.png'
    figure.write_bytes(figure.read_bytes()+b'altered')
    rejected = tmp_path/'rejected'
    with pytest.raises(ValueError, match='checksum'):
        assets.build(folders['report'], folders['audit'], folders['timing'], rejected)
    assert not rejected.exists()
