"""Independent policy gains and martingale bounds under frozen fitted models."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, PROJECT, private, read, save, sha, verify_mapping
from diagnose_fitted_calendars import task_problem, verified_results as calendar_results
from qf_research.optimal_stopping.exercise_redundancy import same_settlement_redundancy
from qf_research.optimal_stopping.funded_policy import (
    simulate_paths, prefix_paths, fit_policy, calibrate_control,
)
from qf_research.optimal_stopping.policy_gain import calibrate_gain_control, evaluate_gain
from qf_research.optimal_stopping.funded_dual import evaluate_dual
from qf_research.optimal_stopping.simulation import estimate_samples

CONFIG = PROJECT/'configs/fitted_policy_gain_2026-09-20.json'


def planned_inputs():
    cfg = read(CONFIG)
    if (cfg['bit_generator'] != 'PCG64' or cfg['confidence'] != .95
            or cfg['stream_roles'] != ['training', 'pilot', 'evaluation', 'dual']
            or cfg['option_kind'] != 'put' or cfg['exercise_calendar'] != 'core_close'
            or cfg['empirical_calibration_accepted'] or cfg['publication_clearance']):
        raise ValueError('Unsupported protocol or scope')
    folder = ROOT/cfg['calendar_run']
    references, _ = calendar_results(folder)
    prior = read(ROOT/cfg['prior_verification'])
    if prior['experiment_manifest_sha256'] != sha(folder/'manifest.json'):
        raise ValueError('Prior experiment changed')
    verify_mapping(prior['source_sha256'])
    originals = read(folder/'tasks.json')
    if len(originals) != len(references):
        raise ValueError('Task/reference coverage differs')
    tasks = []
    for case, (original, reference) in enumerate(zip(originals, references)):
        if any(original['retained'][k] != reference[k] for k in ('candidate_id', 'cutoff', 'strike')):
            raise ValueError('Frozen cases are out of order')
        model, p, schedules = task_problem(original)
        if not same_settlement_redundancy(model, p.decisions, schedules['core_5_minutes'])['sufficient_condition_satisfied']:
            raise ValueError('Retained daily exercise reduction is not justified')
        if not reference['conditional_joint_compatibility']['core_close']:
            raise ValueError('Retained conditional example failed its original screen')
        for repetition in range(cfg['repetitions']):
            tasks.append(dict(case=case, repetition=repetition, original=original,
                              reference=reference, configuration=cfg))
    manifest = read(folder/'manifest.json')
    inputs = dict(manifest['input_sha256'])
    sources = {**manifest['source_sha256'], **prior['source_sha256']}
    inputs.update({str((folder/k).relative_to(ROOT)): v for k, v in manifest['artifact_sha256'].items()})
    for path in (CONFIG, folder/'manifest.json', ROOT/cfg['prior_verification']):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for name in ('scripts/study_fitted_policy_gain.py',
                 'src/qf_research/optimal_stopping/policy_gain.py',
                 'src/qf_research/optimal_stopping/funded_policy.py',
                 'src/qf_research/optimal_stopping/funded_dual.py',
                 'src/qf_research/optimal_stopping/simulation.py'):
        sources[name] = sha(ROOT/name)
    return tasks, inputs, sources


def evaluate(task):
    cfg, reference = task['configuration'], task['reference']
    model, problem, _ = task_problem(task['original'])
    option = replace(problem.option, kind='put')
    rights = problem.decisions
    streams = []

    def sample(role, size):
        key = (task['case'], task['repetition'], cfg['stream_roles'].index(role))
        seed = np.random.SeedSequence(cfg['root_seed'], spawn_key=key)
        paths = simulate_paths(model, option, rights, size, np.random.Generator(np.random.PCG64(seed)))
        streams.append(dict(role=role, spawn_key=list(key), entropy=seed.entropy,
                            stream_id=paths.stream_id, paths=size))
        return paths

    training = sample('training', max(cfg['training_paths']))
    pilot = sample('pilot', cfg['pilot_paths'])
    policies = []
    for count, degree in product(cfg['training_paths'], cfg['degrees']):
        policy = fit_policy(option, prefix_paths(training, count), degree,
                            minimum_itm_paths=cfg['minimum_itm_paths'], condition_limit=cfg['condition_limit'])
        policies.append((policy, calibrate_gain_control(policy, pilot), calibrate_control(policy, pilot)))
    # Freeze every policy and both controls before drawing either evaluation sample.
    del training, pilot
    paths = sample('evaluation', cfg['evaluation_paths'])
    estimates, gain_samples, raw_samples = [], [], []
    european = reference['put_reference']['premium_quote']
    pde = reference['prices']['core_close']['fine']
    premium = pde-european
    for policy, gain_control, dual_control in policies:
        result = evaluate_gain(policy, paths, control=gain_control)
        controlled = result.controlled_quote
        gain_samples.append(result.controlled_gains/model.premium_factor)
        raw_samples.append(result.raw_gains/model.premium_factor)
        estimates.append(dict(training_paths=policy.training_paths, degree=policy.requested_degree,
            fingerprint=policy.fingerprint, policy=asdict(policy), gain_control=asdict(gain_control),
            control=asdict(dual_control),
            gain={k: asdict(getattr(result, k)) for k in ('raw_pv', 'controlled_pv', 'raw_quote', 'controlled_quote', 'control_mean')},
            early_exercise_fraction=float(np.mean(result.stopping_indices < len(rights)-1)),
            stopping_counts=np.bincount(result.stopping_indices, minlength=len(rights)).tolist(),
            decision_diagnostics=list(result.diagnostics),
            pde_shortfall=premium-controlled.value,
            pde_shortfall_sampling_interval=[premium-controlled.upper, premium-controlled.lower],
            captured_fraction=controlled.value/premium if premium > 0 else None))
    covariance = np.cov(np.asarray(gain_samples), ddof=1)
    raw_covariance = np.cov(np.asarray(raw_samples), ddof=1)
    contrasts = []
    for left, right in combinations(range(len(policies)), 2):
        contrasts.append(dict(left=left, right=right, direction='right_minus_left',
            controlled=asdict(estimate_samples(gain_samples[right]-gain_samples[left])),
            raw=asdict(estimate_samples(raw_samples[right]-raw_samples[left]))))
    del paths, gain_samples, raw_samples
    dual_paths = sample('dual', cfg['dual_paths'])
    for entry, (policy, _, control) in zip(estimates, policies):
        entry['duals'] = {}
        for hedge in cfg['hedges']:
            dual = evaluate_dual(policy, dual_paths, control, hedge=hedge)
            entry['duals'][hedge] = dict(
                **{k: asdict(getattr(dual, k)) for k in
                   ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale')},
                pricing_lower_pv=dual.pricing_lower, pricing_upper_pv=dual.pricing_upper,
                loss_upper_97_5_quote=dual.gap_bound.upper/model.premium_factor,
                sample_min_gap_pv=float(dual.gap_payoffs.min()),
                sample_zero_gap_fraction=float(np.mean(dual.gap_payoffs == 0.)))
    if len({s['stream_id'] for s in streams}) != len(streams):
        raise ValueError('Random-stream roles overlap')
    return dict(case=task['case'], repetition=task['repetition'], cutoff=reference['cutoff'],
        strike=reference['strike'], state=model.initial_state, volatility=model.volatility,
        premium_factor=model.premium_factor, put_bid=reference['put_bid'], put_ask=reference['put_ask'],
        observed_spread=reference['put_ask']-reference['put_bid'],
        pde_quote=pde, european_quote=european, model_exercise_premium=premium,
        pde_allowance=reference['numerical_allowance'],
        european_reference_allowance=reference['call_tolerance'],
        streams=streams, policies=estimates, contrasts=contrasts,
        controlled_gain_covariance_quote=covariance.tolist(), raw_gain_covariance_quote=raw_covariance.tolist())


def summarise(rows):
    if not rows or len({(r['case'], r['repetition']) for r in rows}) != len(rows):
        raise ValueError('Require unique case/repetition records')
    streams = [s['stream_id'] for r in rows for s in r['streams']]
    if len(set(streams)) != len(streams):
        raise ValueError('Streams overlap across tasks')
    variants = []
    keys = [(p['training_paths'], p['degree']) for p in rows[0]['policies']]
    for key in keys:
        entries = [(r, p) for r in rows for p in r['policies'] if (p['training_paths'], p['degree']) == key]
        values = np.array([p['gain']['controlled_quote']['value'] for _, p in entries])
        shortfalls = np.array([p['pde_shortfall'] for _, p in entries])
        variants.append(dict(training_paths=key[0], degree=key[1], policies=len(entries),
            gain_range=[float(values.min()), float(values.max())],
            gain_positive_pointwise_intervals=sum(p['gain']['controlled_quote']['lower'] > 0 for _, p in entries),
            shortfall_range=[float(shortfalls.min()), float(shortfalls.max())],
            captured_fraction_range=[min(p['captured_fraction'] for _, p in entries), max(p['captured_fraction'] for _, p in entries)],
            mean_shortfall=float(shortfalls.mean()),
            duals={h: dict(mean_gap_quote=float(np.mean([p['duals'][h]['gap_bound']['value']/r['premium_factor'] for r, p in entries])),
                loss_upper_range=[min(p['duals'][h]['loss_upper_97_5_quote'] for _, p in entries),
                                  max(p['duals'][h]['loss_upper_97_5_quote'] for _, p in entries)],
                bounds_below_observed_spread=sum(p['duals'][h]['loss_upper_97_5_quote'] < r['observed_spread'] for r, p in entries))
                for h in rows[0]['policies'][0]['duals']}))
    by_case = []
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        for index, (count, degree) in enumerate(keys):
            gains = [r['policies'][index]['gain']['controlled_quote']['value'] for r in selected]
            by_case.append(dict(case=case, training_paths=count, degree=degree, repetitions=len(gains),
                mean_gain=float(np.mean(gains)), sd_across_repetitions=float(np.std(gains, ddof=1)),
                mean_shortfall=selected[0]['model_exercise_premium']-float(np.mean(gains))))
    return dict(cases=len({r['case'] for r in rows}), repetitions=len(rows), policies=sum(len(r['policies']) for r in rows),
        variants=variants, by_case=by_case, empirical_calibration_accepted=False, publication_clearance=False,
        intervals_are_pointwise=True, numerical_allowance_is_certified=False)


def verified_results(folder):
    folder = private(folder)
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256', 'source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'], folder)
    rows = read(folder/'rows.json')
    tasks = read(folder/'tasks.json')
    if [(r['case'], r['repetition']) for r in rows] != [(t['case'], t['repetition']) for t in tasks]:
        raise ValueError('Incomplete or reordered experiment')
    if summarise(rows) != read(folder/'summary.json'):
        raise ValueError('Summary differs')
    return rows, read(folder/'summary.json')


def run(output, workers):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh private directory')
    if not 1 <= workers <= 2:
        raise ValueError('Use one or two memory-bounded workers')
    tasks, inputs, sources = planned_inputs()
    output.mkdir(parents=True)
    save(output/'tasks.json', tasks)
    save(output/'configuration.json', read(CONFIG))
    save(output/'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=sources, workers=workers,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row)
            save(output/'partial_rows.json', rows)
            print(f'Completed {len(rows)}/{len(tasks)} policy experiments', flush=True)
    verify_mapping(inputs)
    verify_mapping(sources)
    save(output/'rows.json', rows)
    save(output/'summary.json', summarise(rows))
    names = ('tasks.json', 'configuration.json', 'launch.json', 'rows.json', 'summary.json')
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start, input_sha256=inputs, source_sha256=sources,
        artifact_sha256={name: sha(output/name) for name in names},
        empirical_calibration_accepted=False, publication_clearance=False))
    verified_results(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.workers)
