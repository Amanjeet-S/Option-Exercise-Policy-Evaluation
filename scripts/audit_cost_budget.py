"""Reconstruct budget allocations and audit complete, fixed-policy workloads."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import t

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from qf_research.optimal_stopping.funded_policy import _digest
from study_cost_budget import METHODS, measure


def close(a, b):
    if not math.isclose(a, b, rel_tol=3e-12, abs_tol=2e-13):
        raise ValueError(f'Numerical identity differs: {a} versus {b}')


def audit(folder, output, *, rerun=True):
    folder, output = private(folder), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private audit directory')
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    registration = ROOT/manifest['registration']
    if sha(registration/'manifest.json') != manifest['registration_sha256']:
        raise ValueError('Registration changed')
    registered = read(registration/'manifest.json')
    verify_mapping(registered['artifact_sha256'], registration)
    cfg, tasks = read(registration/'configuration.json'), read(registration/'tasks.json')
    pilot, allocations, rows = [read(folder/f'{n}.json') for n in ('pilot', 'allocations', 'rows')]
    lock = read(folder/'allocation_lock.json')
    if (lock['pilot_sha256'] != sha(folder/'pilot.json')
            or lock['allocations_sha256'] != sha(folder/'allocations.json')
            or lock['source_sha256'] != registered['source_sha256']):
        raise ValueError('Allocation lock changed')
    if not (registered['recorded_at'] <= read(folder/'launch.json')['started_at']
            <= lock['recorded_at'] <= manifest['completed_at']):
        raise ValueError('Stage timestamps out of order')
    expected_pilot = {(task['case'], method, n, rep) for task in tasks for method in METHODS
                      for n in cfg['pilot_paths'] for rep in range(cfg['pilot_repetitions'])}
    actual_pilot = {(r['case'], r['method'], r['paths'], r['stream']['spawn_key'][2]) for r in pilot}
    if actual_pilot != expected_pilot or len(pilot) != len(expected_pilot):
        raise ValueError('Timing pilot coverage differs')
    expected = {(task['case'], b, rep, method) for task in tasks for b in cfg['target_seconds']
                for rep in range(cfg['evaluation_repetitions']) for method in METHODS}
    actual = {(r['case'], r['budget_seconds'], r['evaluation_repetition'], r['method']) for r in rows}
    if expected != actual or len(rows) != len(expected):
        raise ValueError('Evaluation coverage differs')
    # Verify balance from the retained execution order, independently of the runner's rotation formula.
    for sample, field, levels, repetitions in (
            (pilot, 'paths', cfg['pilot_paths'], cfg['pilot_repetitions']),
            (rows, 'budget_seconds', cfg['target_seconds'], cfg['evaluation_repetitions'])):
        for task in tasks:
            for level in levels:
                blocks = [[r for r in sample if r['case'] == task['case'] and r[field] == level
                           and (r['stream']['spawn_key'][2] if field == 'paths' else r['evaluation_repetition']) == rep]
                          for rep in range(repetitions)]
                for method in METHODS:
                    positions = [next(i for i, row in enumerate(block) if row['method'] == method) for block in blocks]
                    if sorted(positions) != [0, 1, 2]:
                        raise ValueError('Within-block method order is not balanced')
    allocation_keys = [(r['case'], r['budget_seconds'], r['method']) for r in allocations]
    expected_allocations = {(task['case'], b, m) for task in tasks for b in cfg['target_seconds'] for m in METHODS}
    if set(allocation_keys) != expected_allocations or len(allocation_keys) != len(expected_allocations):
        raise ValueError('Allocation coverage differs')
    table = dict(zip(allocation_keys, allocations))
    for (case, budget, method), allocation in table.items():
        times = [[r['elapsed_seconds'] for r in pilot if r['case'] == case
                  and r['method'] == method and r['paths'] == n] for n in cfg['pilot_paths']]
        lo, hi = map(median, times)
        nlo, nhi = cfg['pilot_paths']
        slope = (hi-lo)/(nhi-nlo)
        intercept = max(0., lo-slope*nlo)
        q = cfg['path_limits']['quantum']
        affordable = math.floor((budget-intercept)/(q*slope))*q
        capped = cfg['path_limits']['maximum']//q*q
        paths = min(affordable, capped)
        if paths < cfg['path_limits']['minimum'] or paths != allocation['paths']:
            raise ValueError('Allocation path count differs')
        if allocation['capped'] != (affordable > capped) or allocation['status'] != 'allocated':
            raise ValueError('Allocation status differs')
        close(allocation['predicted_seconds'], intercept+slope*paths)
        close(allocation['timing_model']['setup_seconds'], intercept)
        close(allocation['timing_model']['seconds_per_path'], slope)
    warmup = read(folder/'warmup.json')
    if len(warmup) != 3 or {r['method'] for r in warmup} != set(METHODS):
        raise ValueError('Warm-up coverage differs')
    streams = []
    for phase, sample in enumerate((warmup, pilot, rows)):
        for row in sample:
            mi = METHODS.index(row['method'])
            if phase == 0:
                key = [0, tasks[0]['case'], mi]
            elif phase == 1:
                key = [1, row['case'], row['stream']['spawn_key'][2], cfg['pilot_paths'].index(row['paths']), mi]
            else:
                key = [2, row['case'], row['evaluation_repetition'], cfg['target_seconds'].index(row['budget_seconds']), mi]
            state = np.random.PCG64(np.random.SeedSequence(cfg['root_seed'], spawn_key=key)).state
            if (row['stream']['spawn_key'] != key or row['stream']['entropy'] != cfg['root_seed']
                    or row['stream']['stream_id'] != _digest(state)):
                raise ValueError('Incorrect random-stream role')
            streams.append(row['stream']['stream_id'])
            if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('elapsed_seconds', 'cpu_seconds')):
                raise ValueError('Invalid timing')
            if sum(row['phases'].values()) > row['elapsed_seconds']:
                raise ValueError('Component times exceed elapsed time')
    old_streams = {s['stream_id'] for task in tasks for r in task['prior_rows'] for s in r['streams']}
    if len(streams) != len(set(streams)) or old_streams.intersection(streams):
        raise ValueError('Stream reuse detected')
    policy_count, interval_count = 0, 0
    task_index = {task['case']: task for task in tasks}
    for row in rows:
        planned = table[row['case'], row['budget_seconds'], row['method']]
        if row['paths'] != planned['paths'] or row['predicted_seconds'] != planned['predicted_seconds']:
            raise ValueError('Evaluation departed from locked size')
        frozen = [(p['fingerprint'], r['repetition'], p['training_paths'], p['degree'])
                  for r in task_index[row['case']]['prior_rows'] for p in r['policies']]
        present = [(p['fingerprint'], p['training_repetition'], p['training_paths'], p['degree']) for p in row['policies']]
        if frozen != present:
            raise ValueError('Frozen policies changed')
        for p in row['policies']:
            policy_count += 1
            e = p['estimates_quote']
            close(e['gap_bound']['value'], e['upper_value']['value']-e['policy_value']['value'])
            if p['minimum_gap_quote'] < 0 or e['gap_bound']['value'] < 0:
                raise ValueError('Pathwise dual inequality failed')
            for estimate in e.values():
                if estimate['observations'] != row['paths'] or estimate['confidence'] != .95:
                    raise ValueError('Wrong interval sample or confidence')
                radius = t.ppf(.975, row['paths']-1)*estimate['standard_error']
                close(estimate['lower'], estimate['value']-radius)
                close(estimate['upper'], estimate['value']+radius)
                interval_count += 1
    summary = read(folder/'summary.json')
    def extent(values):
        return dict(minimum=float(min(values)), median=float(median(values)), maximum=float(max(values)))
    reconstructed = []
    for budget in cfg['target_seconds']:
        methods = {}
        for method in METHODS:
            selected = [r for r in rows if r['budget_seconds'] == budget and r['method'] == method]
            estimates = [p['estimates_quote']['gap_bound'] for r in selected for p in r['policies']]
            methods[method] = dict(workloads=len(selected), policy_evaluations=len(estimates),
                paths=extent([r['paths'] for r in selected]), seconds=extent([r['elapsed_seconds'] for r in selected]),
                time_ratio=extent([r['elapsed_seconds']/budget for r in selected]),
                gap_quote=extent([e['value'] for e in estimates]), upper_quote=extent([e['upper'] for e in estimates]),
                sampling_radius_quote=extent([e['upper']-e['value'] for e in estimates]),
                overshoots=sum(r['elapsed_seconds'] > budget for r in selected))
        reconstructed.append(dict(budget_seconds=budget, methods=methods))
    if reconstructed != summary['groups']:
        raise ValueError('Descriptive summary differs')
    indexed = {(r['case'], r['budget_seconds'], r['evaluation_repetition'], r['method']):
               {p['fingerprint']: p for p in r['policies']} for r in rows}
    expected_comparisons = {(key[:3], name, fingerprint)
        for key, policies in indexed.items() if key[-1] == 'baseline'
        for name in ('coarse_minus_baseline', 'fine_minus_baseline', 'fine_minus_coarse')
        for fingerprint in policies}
    comparison_keys = set()
    for comparison in summary['comparisons']:
        base = (comparison['case'], comparison['budget_seconds'], comparison['evaluation_repetition'])
        name, fingerprint = comparison['comparison'], comparison['fingerprint']
        if (base, name, fingerprint) in comparison_keys:
            raise ValueError('Duplicated comparison')
        comparison_keys.add((base, name, fingerprint))
        right, left = name.split('_minus_')
        r, l = [indexed[(*base, m)][fingerprint]['estimates_quote']['gap_bound'] for m in (right, left)]
        variance = r['standard_error']**2+l['standard_error']**2
        result = comparison['gap_difference_quote']
        close(result['value'], r['value']-l['value'])
        close(result['standard_error']**2, variance)
        if variance:
            df = variance**2/sum(e['standard_error']**4/(e['observations']-1) for e in (r, l))
            close(result['degrees_of_freedom'], df)
            radius = t.ppf(.975, df)*math.sqrt(variance)
        else:
            radius = 0.
        close(result['lower'], result['value']-radius)
        close(result['upper'], result['value']+radius)
        close(comparison['upper_endpoint_difference_quote'], r['upper']-l['upper'])
    if comparison_keys != expected_comparisons or len(comparison_keys) != policy_count:
        raise ValueError('Missing comparisons')
    reruns = []
    if rerun:
        for method in METHODS:
            retained = next(r for r in rows if r['case'] == tasks[0]['case']
                            and r['evaluation_repetition'] == 0 and r['budget_seconds'] == cfg['target_seconds'][0]
                            and r['method'] == method)
            repeated = measure(tasks[0], method, retained['paths'],
                np.random.SeedSequence(cfg['root_seed'], spawn_key=retained['stream']['spawn_key']))
            for key in ('case', 'method', 'paths', 'stream', 'approximation_fingerprint', 'policies'):
                if repeated[key] != retained[key]:
                    raise ValueError(f'Numerical rerun differs: {method}, {key}')
            reruns.append(dict(case=retained['case'], method=method, paths=retained['paths'],
                               policies=len(retained['policies']), exact_numerical_match=True))
    output.mkdir(parents=True)
    save(output/'verification.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(folder/'manifest.json'), auditor_sha256=sha(Path(__file__)),
        allocations=len(allocations), timing_pilot_workloads=len(pilot), evaluation_workloads=len(rows),
        policy_evaluations=policy_count, marginal_intervals=interval_count, comparisons=len(comparison_keys),
        unique_streams=len(streams), numerical_reruns=reruns, full_benchmark_repeated=False,
        summary_reconstructed=True, allocation_reconstructed=True, publication_clearance=False))
    print(f'Verified {len(rows)} workloads and {policy_count} policy evaluations; {len(reruns)} numerical reruns')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.experiment, args.output)
