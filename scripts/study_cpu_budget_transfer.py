"""Register and measure prospective transfer of complete-pair CPU predictions.

This timing study does not extend the preceding statistical experiment. Numerical
workloads are imported unchanged; all new observations remain in private storage.
"""

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_training_allocation import stage_registration, THREADS
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import (
    build_approximation, evaluate_pair, train_pair,
)


DESIGN = ROOT / 'data/private/cpu_budget_transfer/2026-09-27/design_v1'
EXPECTED = dict(
    cases=list(range(13)), evaluation_grid=[128, 256, 4096], repetitions=8,
    warmup_pairs=4, setup_calibration_repetitions=8,
    training_paths=[5000, 20000], degree=4, minimum_itm_paths=32,
    condition_limit=1e8,
    approximation=dict(core_steps=800, wing_steps=40, tolerance=.001),
    root_seed=2026092708131282564096, budget_ratio_limits=[.9, 1.1],
    cpu_limit=1800., elapsed_limit=5400.,
    primary_cost='sum of unchanged numerical training and evaluation CPU timers',
    primary_predictor='arithmetic mean complete-pair cost',
    statistical_gain_inference=False,
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def schedule(stage):
    """Balanced rotations fixed without observing timings or financial outputs."""
    cells = [(case, n) for case in EXPECTED['cases'] for n in EXPECTED['evaluation_grid']]
    rows = []
    for rep in range(EXPECTED['repetitions']):
        # Thirteen-cell steps place each path count in each position over rounds.
        offset = (13 * rep + 7 * stage) % len(cells)
        order = cells[offset:] + cells[:offset]
        if (rep + stage) % 2:
            order = list(reversed(order))
        rows.extend(dict(repetition=rep, case=case, evaluation_paths=n) for case, n in order)
    return rows


def checked(folder):
    folder = private(folder)
    manifest = read(folder / 'manifest.json')
    verify_mapping(manifest['input_sha256'])
    verify_mapping(manifest['source_sha256'])
    verify_mapping(manifest['artifact_sha256'], folder)
    return manifest


def register(previous, design, output):
    previous, design, output = map(private, (previous, design, output))
    if output.exists():
        raise FileExistsError('Use a fresh registration directory')
    cfg = read(design / 'configuration.json')
    if cfg != EXPECTED:
        raise ValueError('Configuration differs from the specified timing study')
    old, old_cfg, tasks = stage_registration(previous)
    if old_cfg['root_seed'] == cfg['root_seed']:
        raise ValueError('Use a new random-stream namespace')
    sources = dict(old['source_sha256'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    inputs = dict(old['input_sha256'])
    for path in (previous / 'manifest.json', previous / 'tasks.json',
                 design / 'configuration.json', design / 'protocol.md'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    verify_mapping(sources)
    verify_mapping(inputs)
    output.mkdir(parents=True)
    save(output / 'configuration.json', cfg)
    save(output / 'tasks.json', tasks)
    save(output / 'schedules.json', dict(calibration=schedule(0), validation=schedule(1)))
    (output / 'protocol.md').write_text((design / 'protocol.md').read_text())
    save(output / 'manifest.json', dict(
        recorded_at_utc=utc(), stage='timing_transfer_registration',
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={p.name: sha(p) for p in output.iterdir()},
        public_registration=False, new_market_observations=False,
    ))


def telemetry():
    record = dict(recorded_at_utc=utc(), load_average=list(os.getloadavg()))
    try:
        import psutil
        process = psutil.Process()
        times = process.cpu_times()
        record.update(process_threads=process.num_threads(), rss_bytes=process.memory_info().rss,
                      process_user_cpu=times.user, process_system_cpu=times.system,
                      psutil_version=psutil.__version__)
    except ImportError:
        record['process_telemetry'] = 'psutil unavailable'
    return record


def environment():
    record = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                  platform=platform.platform(), machine=platform.machine(),
                  numpy_build=np.show_config(mode='dicts'),
                  thread_environment={k: os.environ.get(k) for k in THREADS},
                  python_workers=1, native_thread_count_verified=False)
    try:
        from threadpoolctl import threadpool_info
        record['reported_native_pools'] = threadpool_info()
    except ImportError:
        record['reported_native_pools'] = None
    return record


def stream(stage, phase, repetition, case, paths, role):
    return np.random.SeedSequence(EXPECTED['root_seed'],
        spawn_key=(stage, phase, repetition, case, paths, role))


def pair(base, approximation, stage, phase, rep, case, n, seen):
    before = telemetry()
    wall, cpu = time.perf_counter(), time.process_time()
    policies, training = train_pair(base, tuple(EXPECTED['training_paths']),
        degree=EXPECTED['degree'], minimum_itm_paths=EXPECTED['minimum_itm_paths'],
        condition_limit=EXPECTED['condition_limit'],
        seeds=tuple(stream(stage, phase, rep, case, n, role) for role in (0, 1)))
    evaluation = evaluate_pair(policies, approximation, n, stream(stage, phase, rep, case, n, 2))
    outer_cpu, outer_wall = time.process_time() - cpu, time.perf_counter() - wall
    identifiers = [x['stream_id'] for x in training['streams']] + [evaluation['stream']['stream_id']]
    if len(set(identifiers)) != 3 or seen.intersection(identifiers):
        raise ValueError('A numerical random stream was reused')
    seen.update(identifiers)
    cost = training['cpu_seconds'] + evaluation['cpu_seconds']
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError('Complete-pair CPU cost must be finite and positive')
    return dict(case=case, evaluation_paths=n, repetition=rep, training=training,
        evaluation=evaluation, pair_cpu_seconds=cost,
        pair_elapsed_seconds=training['elapsed_seconds'] + evaluation['elapsed_seconds'],
        outer_cpu_seconds=outer_cpu, outer_elapsed_seconds=outer_wall,
        outside_numerical_timers_cpu_seconds=outer_cpu - cost,
        telemetry_before=before, telemetry_after=telemetry())


def run(registration, output, stage, calibration=None):
    registration, output = map(private, (registration, output))
    if output.exists():
        raise FileExistsError('Use a fresh execution directory')
    manifest = checked(registration)
    registration_sha256 = sha(registration / 'manifest.json')
    if manifest['stage'] != 'timing_transfer_registration':
        raise ValueError('A timing-transfer registration is required')
    cfg = read(registration / 'configuration.json')
    if cfg != EXPECTED or read(registration / 'schedules.json') != dict(calibration=schedule(0), validation=schedule(1)):
        raise ValueError('The frozen configuration or ordering changed')
    if any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Set all four recorded numerical-library thread variables to one')
    extra_inputs = {}
    calibration_summary = None
    if stage == 'validation':
        if calibration is None:
            raise ValueError('Validation requires a completed independent calibration run')
        calibration = private(calibration)
        previous = checked(calibration)
        launch = read(calibration / 'launch.json')
        if not previous.get('complete') or launch['stage'] != 'calibration':
            raise ValueError('Calibration must be complete before validation')
        if launch['registration_sha256'] != sha(registration / 'manifest.json'):
            raise ValueError('Calibration uses another registration')
        if launch['process_id'] == os.getpid():
            raise ValueError('Launch validation in a separate process')
        calibration_summary = read(calibration / 'summary.json')
        for name in ('manifest.json', 'summary.json'):
            extra_inputs[str((calibration / name).relative_to(ROOT))] = sha(calibration / name)
    elif calibration is not None:
        raise ValueError('Calibration must not depend on preceding timing outcomes')
    tasks = read(registration / 'tasks.json')
    if [x['case'] for x in tasks] != cfg['cases']:
        raise ValueError('Registered case coverage changed')
    bases = {x['case']: load_frozen(x['source_policy'])[0] for x in tasks}
    seen = {s for x in tasks for s in x['prior_streams']}
    stage_id = ('calibration', 'validation').index(stage)
    output.mkdir(parents=True)
    (output / 'pairs').mkdir()
    save(output / 'launch.json', dict(started_at_utc=utc(), stage=stage, process_id=os.getpid(),
        registration=str(registration.relative_to(ROOT)),
        registration_sha256=registration_sha256, environment=environment(),
        telemetry=telemetry()))
    cpu_start, wall_start = time.process_time(), time.monotonic()

    def guard():
        if time.process_time() - cpu_start > cfg['cpu_limit'] or time.monotonic() - wall_start > cfg['elapsed_limit']:
            raise RuntimeError('Declared resource cap reached; run remains incomplete')

    try:
        approx, warm_setup = build_approximation(bases[0], **cfg['approximation'])
        warm = []
        for rep in range(cfg['warmup_pairs']):
            guard()
            warm.append(pair(bases[0], approx, stage_id, 0, rep, 0, 4096, seen))
        save(output / 'warmup.json', dict(setup=warm_setup, pairs=warm))
        setups, approximations, rows = {}, {}, []
        if stage == 'calibration':
            for case in cfg['cases']:
                setups[str(case)] = []
                for rep in range(cfg['setup_calibration_repetitions']):
                    guard()
                    approximations[case], cost = build_approximation(bases[case], **cfg['approximation'])
                    setups[str(case)].append(cost)
            save(output / 'setups.json', setups)
        for position, item in enumerate(schedule(stage_id)):
            guard()
            case, n, rep = item['case'], item['evaluation_paths'], item['repetition']
            key = f'{case}:{n}'
            if stage == 'validation' and key not in setups:
                approximations[key], setup = build_approximation(bases[case], **cfg['approximation'])
                setups[key] = setup
                save(output / 'setups.json', setups)
            approximation = approximations[case if stage == 'calibration' else key]
            row = pair(bases[case], approximation, stage_id, 1, rep, case, n, seen)
            row['schedule_position'] = position
            save(output / 'pairs' / f'pair_{position:04d}.json', row)
            rows.append(row)
            if (position + 1) % 39 == 0:
                print(f'{stage}: {position + 1}/{len(schedule(stage_id))} complete pairs', flush=True)
        summary = summarise(rows, setups, calibration_summary)
        guard()
        if sha(registration / 'manifest.json') != registration_sha256:
            raise ValueError('The registration changed during execution')
        verify_mapping(manifest['input_sha256'])
        verify_mapping(manifest['source_sha256'])
        verify_mapping(manifest['artifact_sha256'], registration)
        checked(registration)
        verify_mapping(extra_inputs)
        save(output / 'summary.json', summary)
        save(output / 'manifest.json', dict(completed_at_utc=utc(), complete=True,
            stage=stage, input_sha256={**manifest['input_sha256'], **extra_inputs,
                str((registration / 'manifest.json').relative_to(ROOT)): registration_sha256},
            source_sha256=manifest['source_sha256'],
            artifact_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))},
            process_cpu_seconds=time.process_time() - cpu_start,
            elapsed_seconds=time.monotonic() - wall_start, statistical_gain_inference=False))
    except Exception as exc:
        save(output / 'incomplete.json', dict(recorded_at_utc=utc(), complete=False,
            exception_type=type(exc).__name__, reason=str(exc)))
        raise


def summarise(rows, setups, calibration=None):
    if len(rows) != 13 * 3 * EXPECTED['repetitions']:
        raise ValueError('A complete fixed-count stage is required')
    prior = {(x['case'], x['evaluation_paths']): x for x in calibration['cells']} if calibration else {}
    cells = []
    for case in EXPECTED['cases']:
        for n in EXPECTED['evaluation_grid']:
            group = [r for r in rows if r['case'] == case and r['evaluation_paths'] == n]
            if sorted(r['repetition'] for r in group) != list(range(EXPECTED['repetitions'])):
                raise ValueError('Cell repetition coverage differs')
            costs = [r['pair_cpu_seconds'] for r in group]
            row = dict(case=case, evaluation_paths=n, repetitions=len(group),
                mean_pair_cpu=statistics.mean(costs), median_pair_cpu=statistics.median(costs),
                minimum_pair_cpu=min(costs), maximum_pair_cpu=max(costs),
                mean_training_cpu=statistics.mean(r['training']['cpu_seconds'] for r in group),
                mean_evaluation_cpu=statistics.mean(r['evaluation']['cpu_seconds'] for r in group),
                mean_outer_cpu=statistics.mean(r['outer_cpu_seconds'] for r in group),
                first_half_mean_pair_cpu=statistics.mean(costs[:4]),
                second_half_mean_pair_cpu=statistics.mean(costs[4:]))
            if calibration is None:
                row['mean_setup_cpu'] = statistics.mean(x['cpu_seconds'] for x in setups[str(case)])
                row['median_setup_cpu'] = statistics.median(x['cpu_seconds'] for x in setups[str(case)])
            else:
                prediction = prior[case, n]
                actual = setups[f'{case}:{n}']['cpu_seconds'] + sum(costs)
                predicted = prediction['mean_setup_cpu'] + len(group) * prediction['mean_pair_cpu']
                median_prediction = prediction['median_setup_cpu'] + len(group) * prediction['median_pair_cpu']
                ratio = actual / predicted
                row.update(actual_block_cpu=actual, predicted_block_cpu=predicted,
                    actual_to_predicted_ratio=ratio, median_predicted_block_cpu=median_prediction,
                    actual_to_median_prediction_ratio=actual / median_prediction,
                    within_registered_tolerance=EXPECTED['budget_ratio_limits'][0] <= ratio <= EXPECTED['budget_ratio_limits'][1])
            cells.append(row)
    return dict(cells=cells, pairs=len(rows), timing_cells=len(cells),
        within_registered_tolerance=sum(c.get('within_registered_tolerance', False) for c in cells) if calibration else None,
        budget_ratio_limits=EXPECTED['budget_ratio_limits'],
        statistical_gain_inference=False, causal_hardware_explanation=False,
        all_registered_cells_retained=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    reg = commands.add_parser('register')
    reg.add_argument('--previous', type=Path, required=True)
    reg.add_argument('--design', type=Path, default=DESIGN)
    reg.add_argument('--output', type=Path, required=True)
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--stage', choices=('calibration', 'validation'), required=True)
    run_parser.add_argument('--registration', type=Path, required=True)
    run_parser.add_argument('--calibration', type=Path)
    run_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'register':
        register(args.previous, args.design, args.output)
    else:
        run(args.registration, args.output, args.stage, args.calibration)


if __name__ == '__main__':
    main()
