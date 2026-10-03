"""Construct the prescribed local cash/funding alternatives before policy tests."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import time

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from study_value_martingale import verified_results as prior_results
from construct_joint_funding_witnesses import evaluate as fit_candidate
from cover_joint_cash_funding import model_at
from qf_research.market_data.settlement_comparison import restore_problem

CONFIG = PROJECT/'configs/policy_input_sensitivity_2026-09-20.json'


def scenario_name(fraction, side):
    return f'cash_{fraction:g}_funding_{side}'


def candidate_task(original, fraction, side, repeat_python):
    """Adapt one selected source curve to the existing candidate-fit interface."""
    task = deepcopy(original)
    low, high = original['component']
    task.update(cash=low+(high-low)*fraction, fraction=fraction,
                candidate_id=scenario_name(fraction, side), repeat_python=repeat_python)
    # The earlier constructor reads its chosen path through this named slot.
    # Preserve the actual source-side identity in the outer scenario record.
    task['variants'] = {'midpoint': deepcopy(original['variants'][side])}
    return task


def planned_inputs():
    cfg = read(CONFIG)
    if cfg['family_size'] != len(cfg['cash_fractions'])*len(cfg['funding_paths']):
        raise ValueError('Family size differs from the prescribed cross-product')
    folder = ROOT/cfg['prior_run']
    previous, _ = prior_results(folder)
    verification = read(ROOT/cfg['prior_verification'])
    if verification['experiment_manifest_sha256'] != sha(folder/'manifest.json'):
        raise ValueError('Verified preceding experiment changed')
    verify_mapping(verification['source_sha256'])
    manifest = read(folder/'manifest.json')
    inputs = dict(manifest['input_sha256'])
    sources = {**manifest['source_sha256'], **verification['source_sha256']}
    inputs.update({str((folder/k).relative_to(ROOT)): v for k, v in manifest['artifact_sha256'].items()})
    old_policy_run = ROOT/read(folder/'configuration.json')['policy_run']
    policy_tasks = read(old_policy_run/'tasks.json')
    by_case = {t['case']: t for t in policy_tasks if t['repetition'] == 0}
    prior_tasks = {t['case']: t for t in read(folder/'tasks.json')}
    tasks = []
    for row in previous:
        original = by_case[row['case']]['original']['original_task']
        retained = by_case[row['case']]['original']['retained']
        if (original['cutoff'], original['strike']) != (row['cutoff'], row['strike']):
            raise ValueError('Original calibration and policy order differ')
        for fraction in cfg['cash_fractions']:
            for side in cfg['funding_paths']:
                tasks.append(dict(case=row['case'], fraction=fraction, funding_path=side,
                    scenario=scenario_name(fraction, side), configuration=cfg,
                    fit_task=candidate_task(original, fraction, side, row['case'] in (previous[0]['case'], previous[-1]['case'])),
                    baseline_retained=retained, policy_task=prior_tasks[row['case']]))
    for p in (CONFIG, folder/'manifest.json', ROOT/cfg['prior_verification'], old_policy_run/'tasks.json'):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).resolve())
    return tasks, inputs, sources


def evaluate(task):
    result = fit_candidate(task['fit_task'])
    row = dict(case=task['case'], fraction=task['fraction'], funding_path=task['funding_path'],
        scenario=task['scenario'], cutoff=result['cutoff'], strike=result['strike'], fit=result)
    if result['status'] == 'conditional_numerical_witness':
        p = restore_problem(task['fit_task']['variants']['midpoint']['raw_call'])
        model = model_at(task['fit_task'], 'midpoint', result['nominal_cash'], result['state_at_cutoff'], result['volatility'])
        row.update(model=asdict(model), option=asdict(replace(p.option, kind='put')), decisions=[asdict(d) for d in p.decisions])
    cfg = task['configuration']
    baseline = (task['fraction'], task['funding_path']) == (cfg['baseline']['cash_fraction'], cfg['baseline']['funding_path'])
    if baseline:
        old = task['baseline_retained']
        if result['status'] != 'conditional_numerical_witness' or any(result[k] != old[k] for k in
                ('nominal_cash', 'state_at_cutoff', 'volatility', 'call_value', 'put_prices')):
            raise ValueError('Baseline candidate does not reproduce exactly')
    row['baseline_reproduced'] = baseline
    return row


def summarise(rows):
    keys = [(r['case'], r['scenario']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('Require unique case/scenario records')
    scenarios = []
    for name in dict.fromkeys(r['scenario'] for r in rows):
        selected = [r for r in rows if r['scenario'] == name]
        scenarios.append(dict(scenario=name, fraction=selected[0]['fraction'], funding_path=selected[0]['funding_path'],
            observations=len(selected), compatible=sum(r['fit']['status'] == 'conditional_numerical_witness' for r in selected),
            all_observations_compatible=all(r['fit']['status'] == 'conditional_numerical_witness' for r in selected),
            statuses=[r['fit']['status'] for r in selected]))
    return dict(models=len({r['case'] for r in rows}), attempted=len(rows), scenarios=scenarios,
        compatible=sum(r['fit']['status'] == 'conditional_numerical_witness' for r in rows),
        failures_prove_infeasibility=False, empirical_calibration_accepted=False, publication_clearance=False)


def verified_results(folder):
    folder = private(folder)
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    rows, tasks = read(folder/'rows.json'), read(folder/'tasks.json')
    if [(r['case'], r['scenario']) for r in rows] != [(t['case'], t['scenario']) for t in tasks]:
        raise ValueError('Incomplete calibration coverage')
    if summarise(rows) != read(folder/'summary.json'):
        raise ValueError('Calibration aggregation differs')
    return rows, read(folder/'summary.json')


def run(output, workers):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private output directory')
    if not 1 <= workers <= 3:
        raise ValueError('Use one to three workers')
    tasks, inputs, sources = planned_inputs()
    output.mkdir(parents=True)
    save(output/'configuration.json', read(CONFIG))
    save(output/'tasks.json', tasks)
    save(output/'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(), input_sha256=inputs, source_sha256=sources, workers=workers))
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row)
            save(output/'partial_rows.json', rows)
            if len(rows) % 9 == 0:
                print(f'Completed {len(rows)}/{len(tasks)} prescribed calibrations', flush=True)
    verify_mapping(inputs)
    verify_mapping(sources)
    save(output/'rows.json', rows)
    save(output/'summary.json', summarise(rows))
    names = ('configuration.json', 'tasks.json', 'launch.json', 'rows.json', 'summary.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs, source_sha256=sources, artifact_sha256={n: sha(output/n) for n in names}, publication_clearance=False))
    verified_results(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    run(args.output, args.workers)
