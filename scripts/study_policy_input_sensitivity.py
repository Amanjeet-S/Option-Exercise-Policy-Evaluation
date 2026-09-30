"""Revalue unchanged quote-space policies over screened cash/funding scenarios."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from calibrate_policy_sensitivity import verified_results as calibration_results, scenario_name
from validate_funded_dual import load_frozen
from study_value_martingale import serialise_dual
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.policy_transfer import evaluate_transferred_dual, family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples


def restore_scenario(row):
    raw = dict(row['model'])
    raw['dividends'] = tuple(DividendCap(**d) for d in raw['dividends'])
    return FundedCashModel(**raw), Option(**row['option']), tuple(FundedExercise(**d) for d in row['decisions'])


def planned_inputs(calibration):
    calibration = private(calibration)
    rows, _ = calibration_results(calibration)
    cfg = read(calibration/'configuration.json')
    if cfg['bit_generator'] != 'PCG64' or cfg['pointwise_confidence'] != .95:
        raise ValueError('Unsupported random generator or pointwise interval')
    old_tasks = read(calibration/'tasks.json')
    old_duals = {r['case']: r for r in read(ROOT/cfg['prior_run']/'rows.json')}
    tasks = []
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        original = next(t for t in old_tasks if t['case'] == case)['policy_task']
        old_streams = {s['stream_id'] for r in original['prior_rows'] for s in r['streams']}
        old_streams.add(old_duals[case]['stream']['stream_id'])
        tasks.append(dict(case=case, scenarios=selected, policy_rows=original['prior_rows'],
                          previous_streams=sorted(old_streams), configuration=cfg))
    manifest = read(calibration/'manifest.json')
    inputs, sources = dict(manifest['input_sha256']), dict(manifest['source_sha256'])
    inputs.update({str((calibration/n).relative_to(ROOT)): v for n,v in manifest['artifact_sha256'].items()})
    inputs[str((calibration/'manifest.json').relative_to(ROOT))] = sha(calibration/'manifest.json')
    for name in ('scripts/study_policy_input_sensitivity.py', 'src/qf_research/optimal_stopping/policy_transfer.py'):
        sources[name] = sha(ROOT/name)
    return tasks, inputs, sources


def evaluate(task):
    cfg = task['configuration']
    entries = [(r['repetition'], p) for r in task['policy_rows'] for p in r['policies']]
    policies = [load_frozen(p)[0] for _,p in entries]
    base_name = scenario_name(cfg['baseline']['cash_fraction'], cfg['baseline']['funding_path'])
    if task['scenarios'][0]['scenario'] != base_name:
        raise ValueError('Baseline must precede the paired comparisons')
    scenario_rows = []
    baseline_gaps, baseline_stops, observation_clock = [], [], None
    seed = np.random.SeedSequence(cfg['root_seed'], spawn_key=(task['case'],))
    for record in task['scenarios']:
        common = dict(scenario=record['scenario'], fraction=record['fraction'], funding_path=record['funding_path'],
            cutoff=record['cutoff'], strike=record['strike'], calibration_status=record['fit']['status'])
        if record['fit']['status'] != 'conditional_numerical_witness':
            scenario_rows.append(dict(common, status='no_screened_construction', policies=[]))
            continue
        model, option, rights = restore_scenario(record)
        baseline = record['scenario'] == base_name
        if option != policies[0].option or model.volatility <= 0:
            raise ValueError('Changed contract or unsupported zero-volatility coupling')
        if baseline and any(p.model != model or p.decisions != rights for p in policies):
            raise ValueError('Baseline model or rights changed')
        start = time.perf_counter()
        approximation = build_values(model, option, rights, **cfg['value_approximation'])
        build_seconds = time.perf_counter()-start
        observations = observation_rights(approximation)
        clock = [(d.time, d.phase) for d in observations]
        if baseline:
            observation_clock = clock
        elif observation_clock != clock:
            raise ValueError('Scenario changed the common-shock observation clock')
        start = time.perf_counter()
        draws = simulate_paths(model, option, observations, cfg['evaluation_paths'], np.random.Generator(np.random.PCG64(seed)))
        if draws.stream_id in task['previous_streams']:
            raise ValueError('New coupled sample overlaps a previous stream role')
        sample = value_martingale(approximation, draws)
        simulation_and_martingale_seconds = time.perf_counter()-start
        results = []
        for index, ((repetition, original), policy) in enumerate(zip(entries, policies)):
            start = time.perf_counter()
            result, realised = evaluate_transferred_dual(policy, sample)
            gap = result.gap_payoffs/model.premium_factor
            if baseline:
                baseline_gaps.append(gap.copy())
                baseline_stops.append(realised.stopping_indices.copy())
            difference = estimate_samples(gap-baseline_gaps[index])
            family_limit = family_upper_limit(result.gap_bound, cfg['family_size'], cfg['family_error_probability'])/model.premium_factor
            results.append(dict(repetition=repetition, training_paths=policy.training_paths, degree=policy.requested_degree,
                source_fingerprint=original['fingerprint'], result=serialise_dual(result, model.premium_factor),
                family_adjusted_upper_quote=family_limit, paired_gap_change=asdict(difference),
                stopping_counts=np.bincount(realised.stopping_indices, minlength=len(rights)).tolist(),
                stopping_disagreement_fraction=float(np.mean(realised.stopping_indices != baseline_stops[index])),
                earlier_fraction=float(np.mean(realised.stopping_indices < baseline_stops[index])),
                later_fraction=float(np.mean(realised.stopping_indices > baseline_stops[index])),
                diagnostics=list(realised.diagnostics), evaluation_seconds=time.perf_counter()-start))
        scenario_rows.append(dict(common, status='evaluated', nominal_cash=record['fit']['nominal_cash'],
            volatility=model.volatility, premium_factor=model.premium_factor,
            observed_spread=record['fit']['put_ask']-record['fit']['put_bid'],
            call_margin=record['fit']['call_margin'], put_margin=record['fit']['put_margin'],
            approximation=asdict(approximation), approximation_fingerprint=approximation.fingerprint,
            stream_id=draws.stream_id, build_seconds=build_seconds,
            simulation_and_martingale_seconds=simulation_and_martingale_seconds, policies=results))
    evaluated = [r for r in scenario_rows if r['status'] == 'evaluated']
    if not evaluated or evaluated[0]['scenario'] != base_name:
        raise ValueError('A screened baseline is required')
    if len({r['stream_id'] for r in evaluated}) != 1:
        raise ValueError('Common-shock scenario coupling differs')
    complete = len(evaluated) == cfg['family_size']
    families = []
    for index, (repetition, original) in enumerate(entries):
        bounds = [(r['policies'][index]['family_adjusted_upper_quote'], r['scenario']) for r in evaluated]
        maximum, name = max(bounds)
        families.append(dict(repetition=repetition, training_paths=original['training_paths'], degree=original['degree'],
            source_fingerprint=original['fingerprint'], evaluated_scenarios=len(evaluated), full_family=complete,
            maximum_family_adjusted_upper_quote=maximum, maximising_scenario=name,
            observed_spread=evaluated[0]['observed_spread'],
            full_family_bound_below_spread=complete and maximum < evaluated[0]['observed_spread']))
    return dict(case=task['case'], stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key),
        stream_id=evaluated[0]['stream_id'], paths=cfg['evaluation_paths']), scenarios=scenario_rows, families=families)


def numerical_record(row):
    result = dict(row)
    result['scenarios'] = []
    for scenario in row['scenarios']:
        clean = {k:v for k,v in scenario.items() if k not in ('build_seconds','simulation_and_martingale_seconds')}
        clean['policies'] = [{k:v for k,v in p.items() if k != 'evaluation_seconds'} for p in scenario['policies']]
        result['scenarios'].append(clean)
    return result


def summarise(rows):
    if not rows or len({r['case'] for r in rows}) != len(rows):
        raise ValueError('Require unique model cases')
    if len({r['stream']['stream_id'] for r in rows}) != len(rows):
        raise ValueError('Distinct models must use distinct streams')
    scenarios = []
    for name in dict.fromkeys(s['scenario'] for r in rows for s in r['scenarios']):
        selected = [(r,s) for r in rows for s in r['scenarios'] if s['scenario'] == name]
        accepted = [(r,s) for r,s in selected if s['status'] == 'evaluated']
        variants = []
        for count, degree in sorted({(p['training_paths'],p['degree']) for _,s in accepted for p in s['policies']}):
            entries = [p for _,s in accepted for p in s['policies'] if (p['training_paths'],p['degree']) == (count,degree)]
            variants.append(dict(training_paths=count, degree=degree, policies=len(entries),
                mean_gap_quote=float(np.mean([p['result']['gap_bound']['value']/s['premium_factor'] for _,s in accepted for p in s['policies'] if (p['training_paths'],p['degree']) == (count,degree)])),
                mean_paired_gap_change=float(np.mean([p['paired_gap_change']['value'] for p in entries])),
                pointwise_bounds_below_spread=sum(p['result']['loss_upper_97_5_quote'] < s['observed_spread'] for _,s in accepted for p in s['policies'] if (p['training_paths'],p['degree']) == (count,degree))))
        scenarios.append(dict(scenario=name, evaluated_models=len(accepted), total_models=len(selected), variants=variants))
    families = [f for r in rows for f in r['families']]
    family_summary = []
    for count, degree in sorted({(f['training_paths'],f['degree']) for f in families}):
        selected = [f for f in families if (f['training_paths'],f['degree']) == (count,degree)]
        family_summary.append(dict(training_paths=count, degree=degree, policies=len(selected),
            complete_families=sum(f['full_family'] for f in selected),
            bounds_below_spread=sum(f['full_family_bound_below_spread'] for f in selected),
            maximum_bound_range=[min(f['maximum_family_adjusted_upper_quote'] for f in selected),max(f['maximum_family_adjusted_upper_quote'] for f in selected)]))
    return dict(models=len(rows), scenario_models=sum(s['status'] == 'evaluated' for r in rows for s in r['scenarios']),
        policy_evaluations=sum(len(s['policies']) for r in rows for s in r['scenarios']),
        scenarios=scenarios, families=family_summary, policy_retraining=False,
        empirical_calibration_accepted=False, publication_clearance=False)


def verified_results(folder):
    folder = private(folder)
    manifest = read(folder/'manifest.json')
    for key in ('input_sha256','source_sha256'):
        verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'],folder)
    rows,tasks = read(folder/'rows.json'),read(folder/'tasks.json')
    if [r['case'] for r in rows] != [t['case'] for t in tasks] or summarise(rows) != read(folder/'summary.json'):
        raise ValueError('Sensitivity coverage or summary differs')
    return rows,read(folder/'summary.json')


def run(calibration, output, workers):
    output = private(output)
    if output.exists():raise FileExistsError('Use a fresh private output directory')
    if not 1 <= workers <= 2:raise ValueError('Use one or two memory-bounded workers')
    tasks,inputs,sources = planned_inputs(calibration)
    output.mkdir(parents=True)
    save(output/'configuration.json',tasks[0]['configuration']);save(output/'tasks.json',tasks)
    save(output/'launch.json',dict(started_at=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,source_sha256=sources,workers=workers))
    start = time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate,tasks,chunksize=1):
            rows.append(row);save(output/'partial_rows.json',rows)
            print(f'Completed {len(rows)}/{len(tasks)} model sensitivity families',flush=True)
    verify_mapping(inputs);verify_mapping(sources)
    save(output/'rows.json',rows);save(output/'summary.json',summarise(rows))
    names=('configuration.json','tasks.json','launch.json','rows.json','summary.json')
    save(output/'manifest.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        input_sha256=inputs,source_sha256=sources,artifact_sha256={n:sha(output/n) for n in names},publication_clearance=False))
    verified_results(output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();run(args.calibration,args.output,args.workers)
