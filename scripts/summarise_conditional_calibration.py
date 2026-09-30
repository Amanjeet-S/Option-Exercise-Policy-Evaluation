"""Verify a retained conditional fit and report every date and option side privately."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping


def verified_results(sample, fit):
    sample, fit = private(sample), private(fit)
    for folder in (sample, fit):
        manifest = read(folder/'manifest.json')
        for field in ('input_sha256', 'source_sha256'):
            verify_mapping(manifest[field])
        verify_mapping(manifest['artifact_sha256'], folder)
    # The earlier numerical experiment also pins the solver's transitive local
    # dependencies, including cash-jump, escrowed and analytical primitives.
    foundation = read(PROJECT/'results/funded_cash/manifest.json')
    verify_mapping(foundation['input_sha256'])
    verify_mapping(foundation['source_sha256'])
    manifest = read(fit/'manifest.json')
    expected = str((sample/'manifest.json').relative_to(ROOT))
    if manifest['input_sha256'].get(expected) != sha(sample/'manifest.json'):
        raise ValueError('Fit does not identify this sample')
    cfg = read(sample/'config.json')
    selected = read(sample/'selected.json')
    rows = read(fit/'price_diagnostics.json')
    fits = read(fit/'fits.json')
    required = {(r['observation_id'], m) for r in selected for m in cfg['dividend_models']}
    actual = [(r['observation_id'], r['dividend_model']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError('Missing or duplicate model/observation diagnostic')
    by_id = {r['observation_id']: r for r in selected}
    for row in rows:
        source = by_id[row['observation_id']]
        if any(row[k] != source[k] for k in ('split', 'bid', 'ask', 'kind')):
            raise ValueError('Diagnostic quote or split differs from the frozen sample')
        if row['date'] != source['observation_date']:
            raise ValueError('Diagnostic date differs from the frozen sample')
    for name in cfg['dividend_models']:
        stages = [f for f in fits if f['dividend_model'] == name]
        if [f['grid'] for f in stages] != cfg['fit_grids']:
            raise ValueError('Missing or reordered training fits')
        sigma = stages[-1]['best']['volatility']
        if any(r['volatility'] != sigma for r in rows if r['dividend_model'] == name):
            raise ValueError('Evaluation does not use frozen training volatility')
    return cfg, selected, fits, rows


def grouped_diagnostics(rows):
    groups = sorted({(r['dividend_model'], r['split'], r['date'], r['kind']) for r in rows})
    result = []
    for model, split, day, kind in groups:
        group = [r for r in rows if (r['dividend_model'], r['split'], r['date'], r['kind']) == (model, split, day, kind)]
        errors = np.array([r['diagnostics']['midpoint_residual'] for r in group])
        result.append({'dividend_model': model, 'split': split, 'date': day, 'kind': kind,
            'rows': len(group), 'mean_error': float(errors.mean()),
            'mae': float(np.abs(errors).mean()), 'rmse': float(np.sqrt(np.square(errors).mean())),
            'inside_quote_intervals': sum(r['diagnostics']['inside_quote_interval'] for r in group),
            'maximum_outside_full_spreads': max(r['diagnostics']['outside_quote_distance_in_full_spreads'] for r in group),
            'refinement_failures': sum(not r['refinement_screen_pass'] for r in group),
            'fit_grid_accuracy_failures': sum(not r['refit_grid_accuracy_screen_pass'] for r in group)})
    return result


def summarise(sample, fit, output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh diagnostic directory')
    cfg, selected, fits, rows = verified_results(sample, fit)
    grouped = grouped_diagnostics(rows)
    counts = {'observations': len(selected), 'split_counts': dict(Counter(r['split'] for r in selected)),
        'model_variants': len(cfg['dividend_models']), 'training_fits': len(fits),
        'scenario_rows': len(rows), 'date_side_groups': len(grouped),
        'local_search_failures': sum(not f['all_local_searches_converged'] for f in fits),
        'boundary_fits': sum(f['boundary_selected'] for f in fits),
        'refinement_failures': sum(not r['refinement_screen_pass'] for r in rows),
        'fit_grid_accuracy_failures': sum(not r['refit_grid_accuracy_screen_pass'] for r in rows),
        'full_empirical_acceptance': False}
    report = ['# Conditional calibration by date and option side', '',
        'Private descriptive diagnostics. Repeated contracts and shared dates are not independent observations.', '',
        '| Dividend model | Split | Date | Side | Rows | Mean error | MAE | RMSE | Inside intervals | Maximum outside distance / full spread |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in grouped:
        report.append(f'| {g["dividend_model"]} | {g["split"]} | {g["date"]} | {g["kind"]} | {g["rows"]} | {g["mean_error"]:.8f} | {g["mae"]:.8f} | {g["rmse"]:.8f} | {g["inside_quote_intervals"]} | {g["maximum_outside_full_spreads"]:.8f} |')
    output.mkdir(parents=True)
    save(output/'date_side_diagnostics.json', grouped)
    save(output/'audit_counts.json', counts)
    (output/'report.md').write_text('\n'.join(report)+'\n')
    save(output/'manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': {str((p/'manifest.json').relative_to(ROOT)): sha(p/'manifest.json') for p in (sample, fit)},
        'source_sha256': {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())},
        'artifact_sha256': {p.name: sha(p) for p in sorted(output.iterdir())},
        'foundation_manifest_sha256': sha(PROJECT/'results/funded_cash/manifest.json')})
    print(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarise(args.sample.resolve(), args.fit.resolve(), args.output.resolve())
