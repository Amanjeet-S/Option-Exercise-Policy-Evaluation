"""Register and execute fixed-count local CPU-cost pilots and validation blocks."""

import argparse
from fractions import Fraction
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_cpu_budget_transfer import checked, environment, telemetry, utc
from study_training_allocation import THREADS
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import (
    build_approximation, evaluate_pair, train_pair,
)

BASE = ROOT / 'data/private/cpu_budget_transfer/2026-09-27'
EXPECTED = dict(
    cases=list(range(13)), evaluation_grid=[128, 256, 4096],
    pilot_pairs=8, validation_pairs=64, warmup_pairs=4,
    pilot_setup_repetitions=1, validation_setup_repetitions=1,
    training_paths=[5000, 20000], degree=4, minimum_itm_paths=32,
    condition_limit=1e8,
    approximation=dict(core_steps=800, wing_steps=40, tolerance=.001),
    root_seed=20260927211308641282564096,
    budget_ratio_limits=[.9, 1.1], cpu_limit=1800., elapsed_limit=5400.,
    primary_cost='fresh setup plus sum of unchanged numerical training and evaluation CPU timers',
    primary_predictor='pilot setup plus fixed validation count times pilot mean complete-pair CPU',
    statistical_gain_inference=False, public_registration=False,
)
PHASES = {'warmup': 0, 'pilot': 1, 'validation': 2}


def schedule():
    """Spread the three path counts over a fixed case permutation."""
    return [dict(position=i, case=(5 * i) % 13,
                 evaluation_paths=EXPECTED['evaluation_grid'][i % 3]) for i in range(39)]


def stream(phase, position, repetition, case, paths, role):
    return np.random.SeedSequence(EXPECTED['root_seed'],
        spawn_key=(PHASES[phase], position, repetition, case, paths, role))


def reserve_streams(seeds, seen):
    from qf_research.optimal_stopping.funded_policy import _digest
    identifiers = [_digest(np.random.PCG64(seed).state) for seed in seeds]
    if len(identifiers) != 3 or len(set(identifiers)) != 3 or seen.intersection(identifiers):
        raise ValueError('A numerical random stream was reused')
    seen.update(identifiers)
    return identifiers


def pair(base, approximation, phase, position, rep, case, n, seen):
    seeds = tuple(stream(phase, position, rep, case, n, role) for role in range(3))
    identifiers = reserve_streams(seeds, seen)
    before = telemetry()
    wall, cpu = time.perf_counter(), time.process_time()
    policies, training = train_pair(base, tuple(EXPECTED['training_paths']),
        degree=EXPECTED['degree'], minimum_itm_paths=EXPECTED['minimum_itm_paths'],
        condition_limit=EXPECTED['condition_limit'], seeds=seeds[:2])
    evaluation = evaluate_pair(policies, approximation, n, seeds[2])
    outer_cpu, outer_wall = time.process_time() - cpu, time.perf_counter() - wall
    recorded = [s['stream_id'] for s in training['streams']] + [evaluation['stream']['stream_id']]
    if recorded != identifiers:
        raise ValueError('Recorded numerical streams differ from the reserved streams')
    cost = training['cpu_seconds'] + evaluation['cpu_seconds']
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError('Complete-pair CPU cost must be finite and positive')
    return dict(case=case, evaluation_paths=n, phase=phase, position=position,
        repetition=rep, training=training, evaluation=evaluation,
        pair_cpu_seconds=cost,
        pair_elapsed_seconds=training['elapsed_seconds'] + evaluation['elapsed_seconds'],
        outer_cpu_seconds=outer_cpu, outer_elapsed_seconds=outer_wall,
        outside_numerical_timers_cpu_seconds=outer_cpu - cost,
        telemetry_before=before, telemetry_after=telemetry())


def verify_completion(path):
    record = read(path)
    if record.get('complete') is not True:
        raise ValueError('Preceding stage must be complete')
    verify_mapping(record['evidence_sha256'])
    return record


def register(previous, design, output, completion):
    previous, design, output, completion = map(private, (previous, design, output, completion))
    if output.exists():
        raise FileExistsError('Use a fresh registration directory')
    cfg = read(design / 'configuration.json')
    if cfg != EXPECTED:
        raise ValueError('Configuration differs from the specified local timing study')
    old = checked(previous)
    if old['stage'] != 'timing_transfer_registration':
        raise ValueError('Original timing registration required')
    if read(previous / 'configuration.json')['root_seed'] == cfg['root_seed']:
        raise ValueError('Use a fresh random-stream namespace')
    tasks = read(previous / 'tasks.json')
    if [t['case'] for t in tasks] != cfg['cases']:
        raise ValueError('All thirteen original cases are required')
    preceding = verify_completion(completion)
    inputs = {**old['input_sha256'], **preceding['evidence_sha256']}
    sources = dict(old['source_sha256'])
    for path in (previous / 'manifest.json', previous / 'tasks.json', completion,
                 design / 'configuration.json', design / 'protocol.md'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in (Path(__file__).resolve(), ROOT / 'tests/test_local_cpu_transfer.py',
                 ROOT / 'scripts/audit_local_cpu_transfer.py'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    verify_mapping(inputs)
    verify_mapping(sources)
    output.mkdir(parents=True)
    save(output / 'configuration.json', cfg)
    save(output / 'tasks.json', tasks)
    save(output / 'schedule.json', schedule())
    (output / 'protocol.md').write_text((design / 'protocol.md').read_text())
    save(output / 'manifest.json', dict(recorded_at_utc=utc(), stage='local_timing_registration',
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={p.name: sha(p) for p in output.iterdir()},
        public_registration=False, new_market_observations=False))


def phase_costs(row):
    t = row['training']
    simulation = Fraction(t['phases']['simulation_cpu_seconds'])
    fitting = Fraction(t['phases']['fitting_cpu_seconds'])
    training, evaluation = Fraction(t['cpu_seconds']), Fraction(row['evaluation']['cpu_seconds'])
    return dict(training_simulation=simulation, training_fitting=fitting,
                other_training=training-simulation-fitting, evaluation=evaluation,
                timer_sum_rounding=Fraction(row['pair_cpu_seconds'])-training-evaluation)


def forecast(item, rows, setup):
    if len(rows) != EXPECTED['pilot_pairs'] or [r['repetition'] for r in rows] != list(range(EXPECTED['pilot_pairs'])):
        raise ValueError('A complete ordered pilot is required')
    if any(r['phase'] != 'pilot' or any(r[k] != item[k] for k in ('case', 'position', 'evaluation_paths')) for r in rows):
        raise ValueError('Pilot identity differs from the scheduled cell')
    scale = Fraction(EXPECTED['validation_pairs'], EXPECTED['pilot_pairs'])
    phases = {k: scale * sum((phase_costs(r)[k] for r in rows), Fraction(0))
              for k in phase_costs(rows[0])}
    phases['setup'] = Fraction(setup['cpu_seconds'])
    total = sum(phases.values(), Fraction(0))
    if total <= 0:
        raise ValueError('Prediction must be positive')
    return dict(**item, recorded_at_utc=utc(), validation_pairs=EXPECTED['validation_pairs'],
        predicted_block_cpu=float(total), exact_predicted_block_cpu=str(total),
        phase_predictions_cpu={k: float(v) for k, v in phases.items()},
        exact_phase_predictions_cpu={k: str(v) for k, v in phases.items()},
        pilot_mean_pair_cpu=statistics.mean(r['pair_cpu_seconds'] for r in rows),
        pilot_sd_pair_cpu=statistics.stdev(r['pair_cpu_seconds'] for r in rows),
        pilot_cv_pair_cpu=statistics.stdev(r['pair_cpu_seconds'] for r in rows) / statistics.mean(r['pair_cpu_seconds'] for r in rows),
        pilot_numerical_cpu=float(Fraction(setup['cpu_seconds']) + sum((Fraction(r['pair_cpu_seconds']) for r in rows), Fraction(0))),
        includes_pilot_charge_in_validation_prediction=False)


def summarise_cell(item, rows, setup, prediction):
    if prediction['validation_pairs'] != EXPECTED['validation_pairs'] or any(prediction[k] != item[k] for k in item):
        raise ValueError('Forecast identity differs')
    if len(rows) != EXPECTED['validation_pairs'] or [r['repetition'] for r in rows] != list(range(EXPECTED['validation_pairs'])):
        raise ValueError('A complete ordered validation block is required')
    if any(r['phase'] != 'validation' or any(r[k] != item[k] for k in item) for r in rows):
        raise ValueError('Validation identity differs')
    phases = {k: sum((phase_costs(r)[k] for r in rows), Fraction(0)) for k in phase_costs(rows[0])}
    phases['setup'] = Fraction(setup['cpu_seconds'])
    actual = sum(phases.values(), Fraction(0))
    predicted = Fraction(prediction['exact_predicted_block_cpu'])
    ratio = actual / predicted
    deltas = {k: v - Fraction(prediction['exact_phase_predictions_cpu'][k]) for k, v in phases.items()}
    if sum(deltas.values(), Fraction(0)) != actual - predicted:
        raise ValueError('Phase decomposition does not close')
    return dict(**item, actual_block_cpu=float(actual), predicted_block_cpu=float(predicted),
        exact_actual_block_cpu=str(actual), exact_predicted_block_cpu=str(predicted),
        actual_to_predicted_ratio=float(ratio), exact_ratio=str(ratio),
        within_registered_tolerance=Fraction(9, 10) <= ratio <= Fraction(11, 10),
        phase_actual_cpu={k: float(v) for k, v in phases.items()},
        exact_phase_actual_cpu={k: str(v) for k, v in phases.items()},
        phase_changes_cpu={k: float(v) for k, v in deltas.items()},
        pilot_numerical_cpu=prediction['pilot_numerical_cpu'],
        first_eight_mean_pair_cpu=statistics.mean(r['pair_cpu_seconds'] for r in rows[:8]),
        last_eight_mean_pair_cpu=statistics.mean(r['pair_cpu_seconds'] for r in rows[-8:]),
        first_half_mean_pair_cpu=statistics.mean(r['pair_cpu_seconds'] for r in rows[:32]),
        second_half_mean_pair_cpu=statistics.mean(r['pair_cpu_seconds'] for r in rows[32:]))


def run(registration, output):
    registration, output = map(private, (registration, output))
    if output.exists():
        raise FileExistsError('Use a fresh execution directory')
    manifest = checked(registration)
    registration_hash = sha(registration / 'manifest.json')
    cfg = read(registration / 'configuration.json')
    if manifest['stage'] != 'local_timing_registration' or cfg != EXPECTED or read(registration / 'schedule.json') != schedule():
        raise ValueError('Frozen design differs')
    if any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Set the four recorded thread variables to one')
    tasks = read(registration / 'tasks.json')
    bases = {t['case']: load_frozen(t['source_policy'])[0] for t in tasks}
    seen = {s for t in tasks for s in t['prior_streams']}
    output.mkdir(parents=True)
    save(output / 'launch.json', dict(started_at_utc=utc(), process_id=os.getpid(),
        registration=str(registration.relative_to(ROOT)), registration_sha256=registration_hash,
        environment=environment(), telemetry=telemetry()))
    cpu_start, wall_start = time.process_time(), time.monotonic()

    def guard():
        if time.process_time()-cpu_start > cfg['cpu_limit'] or time.monotonic()-wall_start > cfg['elapsed_limit']:
            raise RuntimeError('Declared resource cap reached; run remains incomplete')

    try:
        approx, setup = build_approximation(bases[0], **cfg['approximation'])
        warm = []
        for rep in range(cfg['warmup_pairs']):
            guard()
            warm.append(pair(bases[0], approx, 'warmup', 0, rep, 0, 4096, seen))
        save(output / 'warmup.json', dict(setup=setup, pairs=warm))
        cells, excluded_io_cpu, excluded_io_wall = [], 0., 0.
        for item in schedule():
            guard()
            pos, case, n = item['position'], item['case'], item['evaluation_paths']
            folder = output / f'cell_{pos:02d}'
            folder.mkdir()
            block_cpu, block_wall = time.process_time(), time.monotonic()
            approx, pilot_setup = build_approximation(bases[case], **cfg['approximation'])
            pilots = []
            for rep in range(cfg['pilot_pairs']):
                guard()
                row = pair(bases[case], approx, 'pilot', pos, rep, case, n, seen)
                pilots.append(row)
                save(folder / f'pilot_{rep:03d}.json', row)
            prediction = forecast(item, pilots, pilot_setup)
            save(folder / 'pilot_setup.json', pilot_setup)
            save(folder / 'forecast.json', prediction)
            forecast_hash = sha(folder / 'forecast.json')
            validation_started = utc()
            guard()
            approx, validation_setup = build_approximation(bases[case], **cfg['approximation'])
            if pilot_setup['approximation_fingerprint'] != validation_setup['approximation_fingerprint']:
                raise ValueError('Fresh value construction changed')
            save(folder / 'validation_setup.json', validation_setup)
            rows = []
            for rep in range(cfg['validation_pairs']):
                guard()
                row = pair(bases[case], approx, 'validation', pos, rep, case, n, seen)
                rows.append(row)
                # Pair files preserve partial evidence after interruption. These writes
                # are outside every numerical timer and are reported as overhead.
                mark_cpu, mark_wall = time.process_time(), time.perf_counter()
                save(folder / f'validation_{rep:03d}.json', row)
                excluded_io_cpu += time.process_time()-mark_cpu
                excluded_io_wall += time.perf_counter()-mark_wall
            result = summarise_cell(item, rows, validation_setup, prediction)
            if sha(folder / 'forecast.json') != forecast_hash:
                raise ValueError('The forecast changed during validation')
            result.update(validation_started_at_utc=validation_started,
                forecast_sha256_before_validation=forecast_hash,
                whole_cell_cpu_seconds=time.process_time()-block_cpu,
                whole_cell_elapsed_seconds=time.monotonic()-block_wall)
            save(folder / 'result.json', result)
            cells.append(result)
            print(f"cell {pos+1}/39: {case}:{n}, ratio {result['actual_to_predicted_ratio']:.6f}, within tolerance {result['within_registered_tolerance']}", flush=True)
        guard()
        if sha(registration / 'manifest.json') != registration_hash:
            raise ValueError('Registration changed during execution')
        checked(registration)
        ratios = [c['actual_to_predicted_ratio'] for c in cells]
        save(output / 'summary.json', dict(complete=True, cells=cells, timing_cells=39,
            pilot_pairs=39*cfg['pilot_pairs'], validation_pairs=39*cfg['validation_pairs'],
            within_registered_tolerance=sum(c['within_registered_tolerance'] for c in cells),
            minimum_ratio=min(ratios), maximum_ratio=max(ratios), median_ratio=statistics.median(ratios),
            pilot_numerical_cpu=sum(c['pilot_numerical_cpu'] for c in cells),
            validation_numerical_cpu=sum(c['actual_block_cpu'] for c in cells),
            validation_record_writing_cpu=excluded_io_cpu, validation_record_writing_elapsed=excluded_io_wall,
            all_registered_cells_retained=True, statistical_gain_inference=False,
            causal_comparison_to_previous_study=False, budget_ratio_limits=cfg['budget_ratio_limits']))
        save(output / 'manifest.json', dict(complete=True, completed_at_utc=utc(),
            input_sha256={**manifest['input_sha256'], str((registration / 'manifest.json').relative_to(ROOT)): registration_hash},
            source_sha256=manifest['source_sha256'],
            artifact_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))},
            process_cpu_seconds=time.process_time()-cpu_start,
            elapsed_seconds=time.monotonic()-wall_start, statistical_gain_inference=False))
    except BaseException as exc:
        save(output / 'incomplete.json', dict(complete=False, recorded_at_utc=utc(),
            exception_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    reg = commands.add_parser('register')
    reg.add_argument('--previous', type=Path, default=BASE/'registration_v1')
    reg.add_argument('--design', type=Path, default=BASE/'local_timing_design_v1')
    reg.add_argument('--completion', type=Path, default=BASE/'gain_replay_completion_v1.json')
    reg.add_argument('--output', type=Path, required=True)
    runner = commands.add_parser('run')
    runner.add_argument('--registration', type=Path, required=True)
    runner.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        register(args.previous, args.design, args.output, args.completion)
    else:
        run(args.registration, args.output)
