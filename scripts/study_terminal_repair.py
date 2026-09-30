"""Evaluate a terminal-only policy change under verified expiry scenarios."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from study_expiry_extension import verified_results as prior_results, scenario_calendars, YEAR
from study_policy_input_sensitivity import restore_scenario
from study_value_martingale import serialise_dual
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.expiry_extension import restrict_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.terminal_repair import evaluate_terminal_repair
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples

CONFIG = PROJECT/'configs/terminal_repair_2026-09-20.json'


def planned_inputs():
    cfg = read(CONFIG)
    if (cfg['extensions_seconds'] != [0, 900, 5400] or cfg['family_size'] != 3
            or cfg['bit_generator'] != 'PCG64' or cfg['empirical_calibration_accepted'] or cfg['publication_clearance']):
        raise ValueError('Unsupported terminal-repair protocol')
    prior = ROOT/cfg['prior_run']; rows, _ = prior_results(prior)
    check = read(ROOT/cfg['prior_verification'])
    if check['experiment_manifest_sha256'] != sha(prior/'manifest.json'):
        raise ValueError('Preceding verified experiment changed')
    verify_mapping(check['source_sha256'])
    manifest = read(prior/'manifest.json')
    sources = {**manifest['source_sha256'], **check['source_sha256']}
    inputs = dict(manifest['input_sha256'])
    inputs.update({str((prior/name).relative_to(ROOT)): digest for name, digest in manifest['artifact_sha256'].items()})
    for path in (CONFIG, prior/'manifest.json', ROOT/cfg['prior_verification']):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for name in ('scripts/study_terminal_repair.py', 'src/qf_research/optimal_stopping/terminal_repair.py'):
        sources[name] = sha(ROOT/name)
    tasks = []
    for old, row in zip(read(prior/'tasks.json'), rows):
        references = [dict(extension_seconds=s['extension_seconds'], pde_value=s['prices']['fine'],
            pde_increment=s['increment_prices']['fine'], european_increment=s['european']['increment_quote'],
            approximation_fingerprint=s['approximation_fingerprint'],
            conditional_joint_compatibility=s['conditional_joint_compatibility']) for s in row['scenarios']]
        tasks.append(dict(case=old['case'], baseline=old['baseline'], policy_rows=old['policy_rows'],
            previous_streams=sorted(set(old['previous_streams']) | {row['stream']['stream_id']}),
            references=references, configuration=cfg))
    return tasks, inputs, sources


def evaluate(task):
    cfg = task['configuration']; model, option, rights = restore_scenario(task['baseline'])
    factor = model.premium_factor; fit = task['baseline']['fit']
    entries = [(r['repetition'], p) for r in task['policy_rows'] for p in r['policies']]
    policies = [load_frozen(p)[0] for _, p in entries]
    if any((p.model, p.option, p.decisions) != (model, option, rights) for p in policies):
        raise ValueError('Frozen policy model changed')
    calendars = scenario_calendars(model, option, rights, cfg)
    start = time.perf_counter()
    approximations = {seconds: build_values(model, contract, calendar, **cfg['value_approximation'])
                      for seconds, contract, calendar in calendars}
    build_seconds = time.perf_counter()-start
    for ref in task['references']:
        if approximations[ref['extension_seconds']].fingerprint != ref['approximation_fingerprint']:
            raise ValueError('Preceding deterministic value approximation changed')
    seed = np.random.SeedSequence(cfg['root_seed'], spawn_key=(task['case'],))
    start = time.perf_counter()
    observations = simulate_paths(model, calendars[-1][1], observation_rights(approximations[5400]),
        cfg['evaluation_paths'], np.random.Generator(np.random.PCG64(seed)))
    simulation_seconds = time.perf_counter()-start
    if observations.stream_id in task['previous_streams']:
        raise ValueError('Previous random stream reused')
    rows = []; old_stops = []
    for (seconds, contract, _), ref in zip(calendars, task['references']):
        approximation = approximations[seconds]
        restricted = restrict_paths(observations, contract, observation_rights(approximation))
        start = time.perf_counter(); sample = value_martingale(approximation, restricted)
        martingale_seconds = time.perf_counter()-start
        start = time.perf_counter(); records = []
        for index, ((rep, source), policy) in enumerate(zip(entries, policies)):
            result = evaluate_terminal_repair(policy, sample, seconds/YEAR)
            if seconds == 0:
                old_stops.append(result.stopping_indices.copy())
            elif not np.array_equal(old_stops[index], result.stopping_indices):
                raise ValueError('An earlier stopping instruction changed')
            gain = result.integrated_gain/factor; realised = result.realised_gain/factor
            error = float(np.max(np.abs((result.original.gap_payoffs-result.repaired.gap_payoffs)/factor-gain)))
            if error > 1e-11:
                raise ArithmeticError('Paired repair accounting failed')
            duals = {name: serialise_dual(getattr(result, name), factor)
                     for name in ('original', 'repaired', 'direct_repaired')}
            limits = {name: family_upper_limit(getattr(result, name).gap_bound,
                cfg['family_size'], cfg['family_error_probability'])/factor for name in ('original', 'repaired')}
            variance = float(gain.var(ddof=1))
            records.append(dict(repetition=rep, training_paths=policy.training_paths, degree=policy.requested_degree,
                fingerprint=source['fingerprint'], duals=duals, family_upper_quote=limits,
                integrated_gain=asdict(estimate_samples(gain)), realised_gain=asdict(estimate_samples(realised)),
                conditional_noise=asdict(estimate_samples(realised-gain)),
                direct_minus_collapsed_gap=asdict(estimate_samples((result.direct_repaired.gap_payoffs-result.repaired.gap_payoffs)/factor)),
                terminal_survivors=int(result.survives.sum()), stopping_counts=np.bincount(result.stopping_indices, minlength=len(rights)).tolist(),
                realised_negative_gain_count=int((realised < 0).sum()), minimum_integrated_gain=float(gain.min()),
                direct_to_integrated_variance_ratio=float(realised.var(ddof=1))/variance if variance else None,
                maximum_accounting_error=error, maximum_roundoff_correction=result.maximum_roundoff_correction,
                gain_to_pde_increment=float(gain.mean())/ref['pde_increment'] if seconds else None))
        rows.append(dict(**ref, policies=records, martingale_seconds=martingale_seconds,
            policy_evaluation_seconds=time.perf_counter()-start))
    families = []
    for i, (rep, source) in enumerate(entries):
        maximum = {name: max(s['policies'][i]['family_upper_quote'][name] for s in rows) for name in ('original', 'repaired')}
        families.append(dict(repetition=rep, training_paths=source['training_paths'], degree=source['degree'],
            maximum_upper_quote=maximum, below_spread={name: v < fit['put_ask']-fit['put_bid'] for name, v in maximum.items()}))
    return dict(case=task['case'], premium_factor=factor, spread=fit['put_ask']-fit['put_bid'],
        stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key), stream_id=observations.stream_id, paths=cfg['evaluation_paths']),
        build_seconds=build_seconds, simulation_seconds=simulation_seconds, scenarios=rows, families=families,
        refit=False, retraining=False)


def numerical_record(row):
    return {**{k: v for k, v in row.items() if k not in ('build_seconds', 'simulation_seconds')},
        'scenarios': [{k: v for k, v in s.items() if k not in ('martingale_seconds', 'policy_evaluation_seconds')} for s in row['scenarios']]}


def summarise(rows):
    if not rows or len({r['case'] for r in rows}) != len(rows) or len({r['stream']['stream_id'] for r in rows}) != len(rows):
        raise ValueError('Unique case and stream identities required')
    scenarios = []
    for seconds in (0, 900, 5400):
        group = [(r, next(s for s in r['scenarios'] if s['extension_seconds'] == seconds)) for r in rows]
        variants = []
        for n, d in sorted({(p['training_paths'], p['degree']) for _, s in group for p in s['policies']}):
            chosen = [(r, p) for r, s in group for p in s['policies'] if (p['training_paths'], p['degree']) == (n, d)]
            variants.append(dict(training_paths=n, degree=d, policies=len(chosen),
                mean_integrated_gain=float(np.mean([p['integrated_gain']['value'] for _, p in chosen])),
                mean_gap_quote={name: float(np.mean([p['duals'][name]['gap_bound']['value']/r['premium_factor'] for r, p in chosen])) for name in ('original', 'repaired', 'direct_repaired')},
                pointwise_below_spread={name: sum(p['duals'][name]['loss_upper_97_5_quote'] < r['spread'] for r, p in chosen) for name in ('original', 'repaired')},
                adjusted_below_spread={name: sum(p['family_upper_quote'][name] < r['spread'] for r, p in chosen) for name in ('original', 'repaired')}))
        scenarios.append(dict(extension_seconds=seconds, variants=variants))
    families = [f for r in rows for f in r['families']]
    return dict(models=len(rows), policy_scenarios=sum(len(s['policies']) for r in rows for s in r['scenarios']),
        scenarios=scenarios, families=[dict(training_paths=n, degree=d, policies=sum((f['training_paths'], f['degree']) == (n, d) for f in families),
            below_spread={name: sum(f['below_spread'][name] for f in families if (f['training_paths'], f['degree']) == (n, d)) for name in ('original', 'repaired')})
            for n, d in sorted({(f['training_paths'], f['degree']) for f in families})],
        empirical_calibration_accepted=False, publication_clearance=False)


def verified_results(folder):
    folder = private(folder); manifest = read(folder/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    rows = read(folder/'rows.json'); tasks = read(folder/'tasks.json')
    if [r['case'] for r in rows] != [t['case'] for t in tasks] or summarise(rows) != read(folder/'summary.json'):
        raise ValueError('Experiment coverage or summary differs')
    return rows, read(folder/'summary.json')


def run(output, workers):
    output = private(output)
    if output.exists(): raise FileExistsError('Use a fresh private output directory')
    if not 1 <= workers <= 2: raise ValueError('Use one or two workers')
    tasks, inputs, sources = planned_inputs(); output.mkdir(parents=True)
    save(output/'configuration.json', read(CONFIG)); save(output/'tasks.json', tasks)
    save(output/'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(), input_sha256=inputs, source_sha256=sources))
    start = time.monotonic(); rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row); save(output/'partial_rows.json', rows)
            print(f'Completed {len(rows)}/{len(tasks)} terminal-repair cases', flush=True)
    verify_mapping(inputs); verify_mapping(sources)
    save(output/'rows.json', rows); save(output/'summary.json', summarise(rows))
    names = ('configuration.json', 'tasks.json', 'launch.json', 'rows.json', 'summary.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs, source_sha256=sources, artifact_sha256={name: sha(output/name) for name in names}, publication_clearance=False))
    verified_results(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=2)
    args = p.parse_args(); run(args.output, args.workers)
