"""Compare analytically centred value martingales on all frozen policies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from study_fitted_policy_gain import verified_results as policy_results
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.funded_dual import evaluate_dual
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale, evaluate_value_dual,
)
from qf_research.optimal_stopping.simulation import estimate_samples

CONFIG = PROJECT/'configs/value_martingale_2026-09-20.json'


def planned_inputs():
    cfg = read(CONFIG)
    if (cfg['bit_generator'] != 'PCG64' or cfg['confidence'] != .95
            or cfg['baseline_hedge'] != 'policy_slope'
            or cfg['empirical_calibration_accepted'] or cfg['publication_clearance']):
        raise ValueError('Unsupported protocol or scope')
    folder = ROOT/cfg['policy_run']
    rows, _ = policy_results(folder)
    check = read(ROOT/cfg['prior_verification'])
    if check['experiment_manifest_sha256'] != sha(folder/'manifest.json'):
        raise ValueError('Frozen policy experiment changed')
    verify_mapping(check['source_sha256'])
    manifest = read(folder/'manifest.json')
    inputs = dict(manifest['input_sha256'])
    sources = {**manifest['source_sha256'], **check['source_sha256']}
    inputs.update({str((folder/k).relative_to(ROOT)): v for k, v in manifest['artifact_sha256'].items()})
    for p in (CONFIG, folder/'manifest.json', ROOT/cfg['prior_verification'], ROOT/cfg['source_review']):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    for k, v in read(ROOT/cfg['source_review'])['input_sha256'].items():
        inputs[k] = v
    for name in ('scripts/study_value_martingale.py', 'src/qf_research/optimal_stopping/value_martingale.py'):
        sources[name] = sha(ROOT/name)
    tasks = []
    for case in sorted({r['case'] for r in rows}):
        old = [r for r in rows if r['case'] == case]
        tasks.append(dict(case=case, prior_rows=old, configuration=cfg))
    return tasks, inputs, sources


def serialise_dual(result, factor):
    return dict(**{k: asdict(getattr(result, k)) for k in
        ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale')},
        loss_upper_97_5_quote=result.gap_bound.upper/factor,
        pricing_lower_pv=result.pricing_lower, pricing_upper_pv=result.pricing_upper,
        sample_min_gap_pv=float(result.gap_payoffs.min()),
        sample_zero_gap_fraction=float(np.mean(result.gap_payoffs == 0.)))


def evaluate(task, *, paths_override=None):
    cfg = task['configuration']
    entries = [(row['repetition'], entry) for row in task['prior_rows'] for entry in row['policies']]
    policies = [load_frozen(entry) for _, entry in entries]
    first = policies[0][0]
    if any((p.model, p.option, p.decisions) != (first.model, first.option, first.decisions) for p, _ in policies):
        raise ValueError('Policies assigned to different local models')
    factor = first.model.premium_factor
    approximations, built, costs = {}, {}, {}
    for name, settings in cfg['approximations'].items():
        start = time.perf_counter()
        a = build_values(first.model, first.option, first.decisions, **settings)
        costs[name] = dict(build_seconds=time.perf_counter()-start)
        built[name] = a
        approximations[name] = dict(fingerprint=a.fingerprint, value=asdict(a),
            maximum_knots=max(len(n.before.knots if n.before is not None else n.after.knots) for n in a.nodes),
            total_transition_knots=sum(len(n.before.knots if n.before is not None else n.after.knots) for n in a.nodes if n.time > 0))
    rights = observation_rights(next(iter(built.values())))
    if any(observation_rights(a) != rights for a in built.values()):
        raise ValueError('Observation calendars differ across approximations')
    size = cfg['evaluation_paths'] if paths_override is None else paths_override
    seed = np.random.SeedSequence(cfg['root_seed'], spawn_key=(task['case'],))
    start = time.perf_counter()
    observations = simulate_paths(first.model, first.option, rights, size, np.random.Generator(np.random.PCG64(seed)))
    simulation_seconds = time.perf_counter()-start
    prior_streams = {s['stream_id'] for row in task['prior_rows'] for s in row['streams']}
    if observations.stream_id in prior_streams:
        raise ValueError('New study must not reuse any previous random-stream role')
    samples = {}
    for name, approximation in built.items():
        start = time.perf_counter()
        samples[name] = value_martingale(approximation, observations)
        costs[name]['martingale_seconds'] = time.perf_counter()-start
    paths = next(iter(samples.values())).paths
    rows = []
    for (repetition, entry), (policy, control) in zip(entries, policies):
        results, seconds = {}, {}
        start = time.perf_counter()
        results['baseline'] = evaluate_dual(policy, paths, control, hedge=cfg['baseline_hedge'])
        seconds['baseline'] = time.perf_counter()-start
        for name, sample in samples.items():
            start = time.perf_counter()
            results[name] = evaluate_value_dual(policy, sample)
            seconds[name] = time.perf_counter()-start
        comparisons = {}
        for name in cfg['comparisons']:
            right, left = name.split('_minus_')
            comparisons[name] = asdict(estimate_samples((results[right].gap_payoffs-results[left].gap_payoffs)/factor))
        rows.append(dict(repetition=repetition, training_paths=policy.training_paths, degree=policy.requested_degree,
            fingerprint=policy.fingerprint, results={n: serialise_dual(r, factor) for n, r in results.items()},
            paired_gap_comparisons=comparisons, evaluation_seconds=seconds))
    old = task['prior_rows'][0]
    return dict(case=task['case'], cutoff=old['cutoff'], strike=old['strike'], premium_factor=factor,
        observed_spread=old['observed_spread'], pde_quote=old['pde_quote'], pde_allowance=old['pde_allowance'],
        stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key), stream_id=observations.stream_id, paths=size),
        approximations=approximations, shared_costs=costs, simulation_seconds=simulation_seconds, policies=rows)


def numerical_record(row):
    """Timing is observational metadata and is not expected to repeat exactly."""
    return {k: ([{a: b for a, b in p.items() if a != 'evaluation_seconds'} for p in v] if k == 'policies' else v)
            for k, v in row.items() if k not in ('shared_costs', 'simulation_seconds')}


def summarise(rows):
    if not rows or len({r['case'] for r in rows}) != len(rows):
        raise ValueError('Require unique nonempty model records')
    if len({r['stream']['stream_id'] for r in rows}) != len(rows):
        raise ValueError('Model streams overlap')
    variants = []
    for count, degree in sorted({(p['training_paths'], p['degree']) for p in rows[0]['policies']}):
        entries = [(r, p) for r in rows for p in r['policies'] if (p['training_paths'], p['degree']) == (count, degree)]
        methods = {}
        for method in ('baseline', 'coarse', 'fine'):
            gaps = [p['results'][method]['gap_bound']['value']/r['premium_factor'] for r, p in entries]
            bounds = [p['results'][method]['loss_upper_97_5_quote'] for _, p in entries]
            methods[method] = dict(mean_gap_quote=float(np.mean(gaps)), gap_range=[min(gaps), max(gaps)],
                upper_bound_range=[min(bounds), max(bounds)],
                bounds_below_spread=sum(p['results'][method]['loss_upper_97_5_quote'] < r['observed_spread'] for r, p in entries))
        variants.append(dict(training_paths=count, degree=degree, policies=len(entries), methods=methods))
    return dict(models=len(rows), frozen_policies=sum(len(r['policies']) for r in rows), variants=variants,
        refit=False, intervals_are_pointwise=True, empirical_calibration_accepted=False, publication_clearance=False)


def verified_results(folder):
    folder = private(folder)
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    rows = read(folder/'rows.json')
    tasks = read(folder/'tasks.json')
    if [r['case'] for r in rows] != [t['case'] for t in tasks] or summarise(rows) != read(folder/'summary.json'):
        raise ValueError('Experiment coverage or summary differs')
    for task, row in zip(tasks, rows):
        if [p['fingerprint'] for p in row['policies']] != [p['fingerprint'] for old in task['prior_rows'] for p in old['policies']]:
            raise ValueError('Frozen policy coverage differs')
    return rows, read(folder/'summary.json')


def run(output, workers):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private output directory')
    if not 1 <= workers <= 2:
        raise ValueError('Use one or two memory-bounded workers')
    tasks, inputs, sources = planned_inputs()
    output.mkdir(parents=True)
    save(output/'tasks.json', tasks)
    save(output/'configuration.json', read(CONFIG))
    save(output/'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(), workers=workers,
        input_sha256=inputs, source_sha256=sources, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row)
            save(output/'partial_rows.json', rows)
            print(f'Completed {len(rows)}/{len(tasks)} frozen-model martingale comparisons', flush=True)
    verify_mapping(inputs)
    verify_mapping(sources)
    save(output/'rows.json', rows)
    save(output/'summary.json', summarise(rows))
    names = ('tasks.json', 'configuration.json', 'launch.json', 'rows.json', 'summary.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start, input_sha256=inputs, source_sha256=sources,
        artifact_sha256={name: sha(output/name) for name in names}, publication_clearance=False))
    verified_results(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)
