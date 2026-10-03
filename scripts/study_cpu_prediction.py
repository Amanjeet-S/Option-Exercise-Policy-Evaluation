"""Register and measure fixed complete-arm CPU forecasts on fresh local workloads."""

import argparse
from fractions import Fraction
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

from study_prospective_allocation import (
    ROOT, encode, merge, private, read, save, sha, stream_ids, utc, verify_manifest,
)
from study_cpu_budget_transfer import environment, telemetry
from study_training_allocation import THREADS
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import build_approximation, train_pair, evaluate_pair
from qf_research.optimal_stopping.funded_policy import _digest

BASE = ROOT/'data/private/cpu_prediction/2026-09-28'
PRIOR = ROOT/'data/private/prospective_allocation/2026-09-28'
EXPECTED = dict(cases=[5, 6, 10], evaluation_grid=[128, 256, 4096], passes=2,
    calibration_pairs=16, warmup_pairs=4, training_paths=[5000, 20000],
    degree=4, minimum_itm_paths=32, condition_limit=1e8,
    approximation=dict(core_steps=800, wing_steps=40, tolerance=.001),
    root_seed=202609281816256409675, schedule_seed=202609285610918,
    expected_validation_pairs=7956, expected_calibration_pairs=576,
    cpu_limit=1800, elapsed_limit=5400, tolerance=['9/10', '11/10'],
    allowance='11/10', public_registration=False, publication_clearance=False,
    financial_gain_inference=False, prediction_coverage_claim=False,
    counts_depend_on_new_timings=False, parallel_workers=1)
PHASES = {'warmup': 0, 'pass_calibration': 1, 'local_calibration': 2, 'validation': 3}
IDENTITY = ('pass', 'position', 'case', 'evaluation_paths', 'repetitions')


def schedule():
    cells = [(case, n) for case in EXPECTED['cases'] for n in EXPECTED['evaluation_grid']]
    order = np.random.default_rng(EXPECTED['schedule_seed']).permutation(len(cells)).tolist()
    return [dict(pass_id=p, position=i, case=cells[k][0], evaluation_paths=cells[k][1])
            for p in range(2) for i, k in enumerate(order if p == 0 else list(reversed(order)))]


def seed_sequence(phase, pass_id, case, n, repetition, role):
    return np.random.SeedSequence(EXPECTED['root_seed'],
        spawn_key=(PHASES[phase], pass_id, case, n, repetition, role))


def fraction(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('A timer must be finite')
    result = Fraction(value)
    if result < 0:
        raise ValueError('A timer must be nonnegative')
    return result


def costs(row, clock='cpu'):
    key = 'cpu_seconds' if clock == 'cpu' else 'elapsed_seconds'
    return fraction(row['training'][key]) + fraction(row['evaluation'][key])


def check_rows(item, rows, phase, count):
    if len(rows) != count or [r['repetition'] for r in rows] != list(range(count)):
        raise ValueError('A complete ordered block is required')
    for row in rows:
        if row['phase'] != phase or any(row[k] != item[k] for k in ('pass_id', 'case', 'evaluation_paths')):
            raise ValueError('Block identity differs')
        for clock in ('cpu', 'elapsed'):
            if costs(row, clock) <= 0:
                raise ValueError('A complete-pair charge must be positive')


def forecast(item, rows, setup, phase):
    if phase not in ('pass_calibration', 'local_calibration'):
        raise ValueError('A calibration phase is required')
    check_rows(item, rows, phase, EXPECTED['calibration_pairs'])
    predictions, charge, first, last = {}, {}, {}, {}
    for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds')):
        values = [costs(r, clock) for r in rows]
        predictions[clock] = fraction(setup[key]) + item['repetitions']*sum(values)/len(values)
        charge[clock] = fraction(setup[key]) + sum(values)
        first[clock] = sum(values[:8])/8
        last[clock] = sum(values[-8:])/8
    return dict(**item, phase=phase, recorded_at_utc=utc(), predictions=predictions,
        allowance_predictions={k: Fraction(EXPECTED['allowance'])*v for k, v in predictions.items()},
        calibration_charge=charge, first_half_pair_mean=first, last_half_pair_mean=last,
        calibration_pairs=16, includes_calibration_cost_in_validation_prediction=False,
        prediction_coverage_claim=False)


def summarise(item, rows, setup, predictions, calibration_charge):
    check_rows(item, rows, 'validation', item['repetitions'])
    actual = {clock: fraction(setup[key]) + sum(costs(r, clock) for r in rows)
              for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds'))}
    comparisons = {}
    budget = Fraction(item['budget'])
    for name, prediction in predictions.items():
        if any(prediction[k] != item[k] for k in ('pass_id', 'case', 'evaluation_paths', 'repetitions')):
            raise ValueError('Forecast identity differs from validation')
        comparisons[name] = {}
        for clock, pred in prediction['predictions'].items():
            pred = fraction(pred)
            if pred <= 0:
                raise ValueError('A prediction must be positive')
            ratio = actual[clock]/pred
            allowance = Fraction(EXPECTED['allowance'])*pred
            comparisons[name][clock] = dict(actual=actual[clock], predicted=pred,
                ratio=ratio, absolute_relative_error=abs(ratio-1),
                within_tolerance=Fraction(9, 10) <= ratio <= Fraction(11, 10),
                allowance=allowance, within_allowance=actual[clock] <= allowance)
            if clock == 'cpu':
                comparisons[name][clock].update(forecast_affordable=pred <= budget,
                    allowance_affordable=allowance <= budget,
                    actual_affordable=actual[clock] <= budget,
                    falsely_forecast_affordable=pred <= budget < actual[clock],
                    falsely_allowance_affordable=allowance <= budget < actual[clock])
    q = max(1, len(rows)//4)
    return dict(**item, actual=actual, comparisons=comparisons,
        calibration_charge=calibration_charge,
        charged_numerical_cost={k: actual[k]+Fraction(calibration_charge[k]) for k in actual},
        charged_cpu_affordable=actual['cpu']+Fraction(calibration_charge['cpu']) <= budget,
        first_quarter_pair_cpu=sum(costs(r) for r in rows[:q])/q,
        last_quarter_pair_cpu=sum(costs(r) for r in rows[-q:])/q,
        diagnostic_quarter_count=q, complete=True)


def register(output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh registration directory')
    evidence = verify_manifest(PRIOR/'completion_v1.json')
    merge(evidence, verify_manifest(PRIOR/'registration_v1/manifest.json'))
    allocation = read(PRIOR/'registration_v1/allocations.json')
    tasks = read(PRIOR/'registration_v1/tasks.json')
    arms = []
    for item in schedule():
        model = next(m for m in allocation['models'] if m['case'] == item['case'])
        candidate = next(c for c in model['candidates'] if c['evaluation_paths'] == item['evaluation_paths'])
        arms.append(dict(**item, repetitions=candidate['repetitions'], budget=model['budget'],
            historical_prediction=candidate['planning_charge'],
            approximation_fingerprint=model['approximation_fingerprint']))
    if sum(a['repetitions'] for a in arms) != EXPECTED['expected_validation_pairs']:
        raise ValueError('Registered complete-arm count differs')
    forbidden = set()
    for name in evidence:
        if name.endswith('.json'):
            forbidden.update(stream_ids(read(ROOT/name)))
    for path in (Path(__file__).resolve(), ROOT/'tests/test_cpu_prediction.py',
                 BASE/'design_v1/protocol.md'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    merge({}, evidence)
    output.mkdir(parents=True)
    save(output/'configuration.json', EXPECTED)
    save(output/'arms.json', arms)
    save(output/'tasks.json', tasks)
    save(output/'forbidden_streams.json', sorted(forbidden))
    (output/'protocol.md').write_text((BASE/'design_v1/protocol.md').read_text())
    save(output/'manifest.json', dict(complete=True, recorded_at_utc=utc(),
        stage='complete_arm_cpu_prediction_registration', evidence_sha256=evidence,
        artifact_sha256={p.name: sha(p) for p in output.iterdir()},
        public_registration=False, new_outcomes_inspected=False))


def pair(base, approximation, phase, item, rep, seen):
    seeds = tuple(seed_sequence(phase, item['pass_id'], item['case'], item['evaluation_paths'], rep, role)
                  for role in range(3))
    identifiers = [_digest(np.random.PCG64(s).state) for s in seeds]
    if len(set(identifiers)) != 3 or seen.intersection(identifiers):
        raise ValueError('A random stream was reused')
    seen.update(identifiers)
    wall, cpu = time.perf_counter(), time.process_time()
    policies, training = train_pair(base, tuple(EXPECTED['training_paths']),
        degree=EXPECTED['degree'], minimum_itm_paths=EXPECTED['minimum_itm_paths'],
        condition_limit=EXPECTED['condition_limit'], seeds=seeds[:2])
    evaluation = evaluate_pair(policies, approximation, item['evaluation_paths'], seeds[2])
    outer_cpu, outer_elapsed = time.process_time()-cpu, time.perf_counter()-wall
    if [s['stream_id'] for s in training['streams']]+[evaluation['stream']['stream_id']] != identifiers:
        raise ValueError('Returned random stream differs')
    # Financial outputs are executed but are not analysed as additional financial evidence.
    kept_training = {k: training[k] for k in ('training_paths', 'policy_fingerprints', 'streams',
        'phases', 'elapsed_seconds', 'cpu_seconds')}
    kept_evaluation = {k: evaluation[k] for k in ('training_paths', 'policy_fingerprints',
        'evaluation_paths', 'stream', 'paths_sha256', 'approximation_fingerprint',
        'elapsed_seconds', 'cpu_seconds')}
    return dict(pass_id=item['pass_id'], case=item['case'], evaluation_paths=item['evaluation_paths'],
        phase=phase, repetition=rep, training=kept_training, evaluation=kept_evaluation,
        outer_cpu_seconds=outer_cpu, outer_elapsed_seconds=outer_elapsed,
        complete_financial_record_sha256=_digest(dict(training=training, evaluation=evaluation)))


def run(registration, output):
    registration, output = private(registration), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh execution directory')
    evidence = verify_manifest(registration/'manifest.json')
    if read(registration/'configuration.json') != EXPECTED:
        raise ValueError('Frozen configuration differs')
    if any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Limit all four numerical thread variables to one')
    env = environment()
    if any(p.get('num_threads') != 1 for p in env.get('reported_native_pools') or []):
        raise ValueError('A reported numerical pool exceeds one thread')
    arms = read(registration/'arms.json')
    bases = {t['case']: load_frozen(t['source_policy'])[0] for t in read(registration/'tasks.json')}
    seen = set(read(registration/'forbidden_streams.json'))
    output.mkdir(parents=True)
    save(output/'launch.json', dict(started_at_utc=utc(), environment=env, telemetry=telemetry(),
        registration=str(registration.relative_to(ROOT)), registration_sha256=sha(registration/'manifest.json'),
        process_id=os.getpid()))
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    totals = dict(cpu=Fraction(0), elapsed=Fraction(0))
    results = []

    def guard():
        if time.process_time()-cpu_start > EXPECTED['cpu_limit'] or time.perf_counter()-wall_start > EXPECTED['elapsed_limit']:
            raise RuntimeError('Declared safety cap reached; retain an incomplete run')

    def block(item, phase, count, folder):
        guard()
        folder.mkdir()
        started = dict(recorded_at_utc=utc(), cpu_since_launch=time.process_time()-cpu_start,
                       elapsed_since_launch=time.perf_counter()-wall_start, telemetry=telemetry())
        save(folder/'started.json', started)
        approx, setup = build_approximation(bases[item['case']], **EXPECTED['approximation'])
        if setup['approximation_fingerprint'] != item['approximation_fingerprint']:
            raise ValueError('Approximation changed from the retained workload')
        save(folder/'setup.json', setup)
        rows = []
        for rep in range(count):
            guard()
            row = pair(bases[item['case']], approx, phase, item, rep, seen)
            rows.append(row)
            save(folder/f'row_{rep:04d}.json', row)
        charges = {clock: fraction(setup[key])+sum(costs(r, clock) for r in rows)
                   for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds'))}
        for clock in charges:
            totals[clock] += charges[clock]
        save(folder/'finished.json', dict(recorded_at_utc=utc(), cpu_since_launch=time.process_time()-cpu_start,
            elapsed_since_launch=time.perf_counter()-wall_start, numerical_charge=charges, telemetry=telemetry()))
        return rows, setup, charges

    try:
        _, _, warm_charge = block(arms[0], 'warmup', EXPECTED['warmup_pairs'], output/'warmup')
        for pass_id in range(EXPECTED['passes']):
            pass_arms = [a for a in arms if a['pass_id'] == pass_id]
            pass_folder = output/f'pass_{pass_id}'
            pass_folder.mkdir()
            early = {}
            for item in pass_arms:
                key = f"case_{item['case']:02d}_n_{item['evaluation_paths']:04d}"
                folder = pass_folder/f'calibration_{key}'
                rows, setup, charge = block(item, 'pass_calibration', 16, folder)
                pred = forecast(item, rows, setup, 'pass_calibration')
                save(folder/'forecast.json', pred)
                early[key] = (pred, sha(folder/'forecast.json'))
            # Every pass-start prediction is fixed before any validation workload in this pass.
            save(pass_folder/'forecasts_frozen.json', dict(recorded_at_utc=utc(),
                forecast_sha256={k: v[1] for k, v in early.items()}))
            for item in pass_arms:
                key = f"case_{item['case']:02d}_n_{item['evaluation_paths']:04d}"
                local = pass_folder/f'local_{key}'
                rows, setup, charge = block(item, 'local_calibration', 16, local)
                immediate = forecast(item, rows, setup, 'local_calibration')
                save(local/'forecast.json', immediate)
                historical = dict(**item, predictions=dict(cpu=item['historical_prediction']))
                preds = dict(historical=historical, pass_start=early[key][0], immediate=immediate)
                prediction_path = pass_folder/f'predictions_{key}.json'
                save(prediction_path, dict(recorded_at_utc=utc(), predictions=preds))
                frozen_hash = sha(prediction_path)
                rows, setup, _ = block(item, 'validation', item['repetitions'], pass_folder/f'validation_{key}')
                if sha(prediction_path) != frozen_hash:
                    raise ValueError('A forecast changed during validation')
                combined_cal = {k: Fraction(immediate['calibration_charge'][k])+
                    Fraction(early[key][0]['calibration_charge'][k]) for k in ('cpu', 'elapsed')}
                result = summarise(item, rows, setup, preds, combined_cal)
                result.update(prediction_sha256_before_validation=frozen_hash,
                    validation_started=read(pass_folder/f'validation_{key}'/'started.json'),
                    validation_finished=read(pass_folder/f'validation_{key}'/'finished.json'))
                save(pass_folder/f'result_{key}.json', result)
                results.append(result)
                print(f"Pass {pass_id+1}, case {item['case']}, N={item['evaluation_paths']}: "
                      f"CPU ratios historical={float(result['comparisons']['historical']['cpu']['ratio']):.3f}, "
                      f"pass-start={float(result['comparisons']['pass_start']['cpu']['ratio']):.3f}, "
                      f"immediate={float(result['comparisons']['immediate']['cpu']['ratio']):.3f}", flush=True)
        summary = aggregate(results)
        summary.update(numerical_totals=totals, warmup_charge=warm_charge,
            process_cpu_seconds=time.process_time()-cpu_start,
            process_elapsed_seconds=time.perf_counter()-wall_start,
            numerical_random_streams=len(seen)-len(read(registration/'forbidden_streams.json')),
            ended_at_utc=utc(), complete=True)
        save(output/'summary.json', summary)
        save(output/'manifest.json', dict(complete=True, stage='complete_arm_cpu_prediction_run',
            recorded_at_utc=utc(), evidence_sha256=evidence,
            artifact_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
            publication_clearance=False, new_market_observations=False))
    except Exception as exc:
        save(output/'incomplete.json', dict(complete=False, error_type=type(exc).__name__, message=str(exc),
            recorded_at_utc=utc(), process_cpu_seconds=time.process_time()-cpu_start,
            process_elapsed_seconds=time.perf_counter()-wall_start))
        raise


def aggregate(results):
    if len(results) != 18 or len({(r['pass_id'], r['case'], r['evaluation_paths']) for r in results}) != 18:
        raise ValueError('All eighteen complete arms are required')
    answer = dict(validation_arms=18, validation_pairs=sum(r['repetitions'] for r in results),
        calibration_pairs=EXPECTED['expected_calibration_pairs'], warmup_pairs=4,
        counts_changed=False, numerical_validation_cost={k: sum(Fraction(r['actual'][k]) for r in results)
        for k in ('cpu', 'elapsed')}, calibration_cost={k: sum(Fraction(r['calibration_charge'][k]) for r in results)
        for k in ('cpu', 'elapsed')}, charged_arms_affordable=sum(r['charged_cpu_affordable'] for r in results),
        prediction_coverage_claim=False, causal_effect_identified=False,
        results_are_local_to_retained_execution=True, comparisons={})
    for name in ('historical', 'pass_start', 'immediate'):
        entries = [r['comparisons'][name]['cpu'] for r in results]
        answer['comparisons'][name] = dict(within_tolerance=sum(e['within_tolerance'] for e in entries),
            within_allowance=sum(e['within_allowance'] for e in entries),
            forecast_affordable=sum(e['forecast_affordable'] for e in entries),
            allowance_affordable=sum(e['allowance_affordable'] for e in entries),
            falsely_forecast_affordable=sum(e['falsely_forecast_affordable'] for e in entries),
            falsely_allowance_affordable=sum(e['falsely_allowance_affordable'] for e in entries),
            actual_affordable=sum(e['actual_affordable'] for e in entries),
            mean_absolute_relative_error=sum(Fraction(e['absolute_relative_error']) for e in entries)/18,
            ratio_min=min(Fraction(e['ratio']) for e in entries), ratio_max=max(Fraction(e['ratio']) for e in entries))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'run'])
    parser.add_argument('--registration', type=Path, default=BASE/'registration_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'register':
        register(args.output)
    else:
        run(args.registration, args.output)


if __name__ == '__main__':
    main()
