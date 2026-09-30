"""Supplement the local timing audit with file and workload record integrity.

No policies are fitted and no paths are simulated. Policy fingerprints use
the existing sorted-key JSON convention, implemented here with hashlib rather
than importing the numerical workload. PCG64 states are reconstructed solely
to verify retained stream identities.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'cpu_budget_transfer/2026-09-27'
DEFAULT_MODELS = BASE / 'certified_residual_inputs_v1/models.json'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
APPROXIMATION_FIELDS = ('model', 'option', 'decisions', 'nodes', 'core_steps',
                        'wing_steps', 'compression_tolerance', 'initial_value')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Evidence must remain in data/private')
    return path


def finite(value, name, *, positive=False):
    require(type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), 'Invalid timer: ' + name)
    return Fraction(value)


def contained(inside, outside, name):
    # This allowance concerns subtraction/aggregation rounding, not clock accuracy.
    allowance = Fraction(1, 10**9)
    require(inside <= outside + allowance, 'Timer containment differs: ' + name)


def expected_artifacts():
    names = {'launch.json', 'summary.json', 'warmup.json'}
    for position in range(39):
        prefix = f'cell_{position:02d}/'
        names.update(prefix + f'pilot_{rep:03d}.json' for rep in range(8))
        names.update(prefix + f'validation_{rep:03d}.json' for rep in range(64))
        names.update(prefix + name for name in ('pilot_setup.json', 'validation_setup.json',
                                               'forecast.json', 'result.json'))
    return names


def inspect_timers(row):
    training, evaluation = row['training'], row['evaluation']
    t_cpu, e_cpu = (finite(part['cpu_seconds'], name, positive=True)
                    for part, name in ((training, 'training CPU'), (evaluation, 'evaluation CPU')))
    t_wall, e_wall = (finite(part['elapsed_seconds'], name, positive=True)
                      for part, name in ((training, 'training elapsed'), (evaluation, 'evaluation elapsed')))
    pair_cpu = finite(row['pair_cpu_seconds'], 'pair CPU', positive=True)
    pair_wall = finite(row['pair_elapsed_seconds'], 'pair elapsed', positive=True)
    outer_cpu = finite(row['outer_cpu_seconds'], 'outer CPU', positive=True)
    outer_wall = finite(row['outer_elapsed_seconds'], 'outer elapsed', positive=True)
    require(row['pair_cpu_seconds'] == training['cpu_seconds'] + evaluation['cpu_seconds'],
            'The retained CPU timer sum differs')
    require(row['pair_elapsed_seconds'] == training['elapsed_seconds'] + evaluation['elapsed_seconds'],
            'The retained elapsed timer sum differs')
    require(row['outside_numerical_timers_cpu_seconds'] == row['outer_cpu_seconds'] - row['pair_cpu_seconds'],
            'The outer CPU remainder differs')
    contained(t_cpu + e_cpu, outer_cpu, 'complete-pair CPU')
    contained(t_wall + e_wall, outer_wall, 'complete-pair elapsed')
    phases = training['phases']
    simulation_cpu = finite(phases['simulation_cpu_seconds'], 'training simulation CPU')
    fitting_cpu = finite(phases['fitting_cpu_seconds'], 'training fitting CPU')
    simulation_wall = finite(phases['simulation_seconds'], 'training simulation elapsed')
    fitting_wall = finite(phases['fitting_seconds'], 'training fitting elapsed')
    contained(simulation_cpu + fitting_cpu, t_cpu, 'training phase CPU')
    contained(simulation_wall + fitting_wall, t_wall, 'training phase elapsed')
    return dict(pair_cpu=pair_cpu, pair_elapsed=pair_wall, outer_cpu=outer_cpu, outer_elapsed=outer_wall)


def audit(registration, run, output, models=DEFAULT_MODELS):
    registration, run, output, models = map(private, (registration, run, output, models))
    require(not output.exists() and not output.is_relative_to(run)
            and not output.is_relative_to(registration), 'Use a fresh, separate private audit directory')
    evidence = {}

    def retain(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence file')
        value = sha(path)
        require(expected is None or expected == value, 'Evidence checksum differs: ' + str(path.relative_to(ROOT)))
        key = str(path.relative_to(ROOT))
        require(key not in evidence or evidence[key] == value, 'Conflicting evidence checksum')
        evidence[key] = value
        return path

    def manifest(folder):
        record = read(retain(folder / 'manifest.json'))
        for key, base in (('input_sha256', ROOT), ('source_sha256', ROOT), ('artifact_sha256', folder)):
            require(isinstance(record[key], dict) and record[key], 'A non-empty checksum map is required')
            for name, expected in record[key].items():
                path = (base / name).resolve()
                require(path.is_relative_to(base.resolve()), 'An evidence path escapes its declared root')
                retain(path, expected)
        return record

    registered, completed, exported = manifest(registration), manifest(run), manifest(models.parent)
    require(registered.get('stage') == 'local_timing_registration', 'The local timing registration is required')
    require(completed.get('complete') is True and not (run / 'incomplete.json').exists(),
            'The timing execution is incomplete')
    require(exported.get('complete') is True and exported.get('original_evidence_unchanged') is True
            and models.name in exported['artifact_sha256'], 'A completed authenticated model export is required')
    require(completed['source_sha256'] == registered['source_sha256'], 'The run source closure differs from registration')
    registered_input = dict(registered['input_sha256'])
    registered_input[str((registration / 'manifest.json').relative_to(ROOT))] = sha(registration / 'manifest.json')
    require(completed['input_sha256'] == registered_input, 'The run input closure differs from registration')
    expected = expected_artifacts()
    actual = {str(path.relative_to(run)) for path in run.rglob('*.json')}
    require(set(completed['artifact_sha256']) == expected and actual == expected | {'manifest.json'},
            'The run manifest or JSON file coverage is incomplete or contains extra records')
    require({p.name for p in run.iterdir() if p.is_dir()} == {f'cell_{i:02d}' for i in range(39)},
            'Unexpected or missing cell directories')
    registration_artifacts = {'configuration.json', 'tasks.json', 'schedule.json', 'protocol.md'}
    require(set(registered['artifact_sha256']) == registration_artifacts,
            'Registration artifact coverage differs')
    cfg, tasks = read(registration / 'configuration.json'), read(registration / 'tasks.json')
    require(cfg['cases'] == list(range(13)) and cfg['evaluation_grid'] == [128, 256, 4096]
            and cfg['pilot_pairs'] == 8 and cfg['validation_pairs'] == 64 and cfg['warmup_pairs'] == 4
            and cfg['pilot_setup_repetitions'] == cfg['validation_setup_repetitions'] == 1
            and cfg['training_paths'] == [5000, 20000] and cfg['degree'] == 4,
            'Registered workload counts or settings differ')
    require([task['case'] for task in tasks] == list(range(13)), 'Registered financial case coverage differs')
    schedule = [dict(position=i, case=5*i % 13, evaluation_paths=[128, 256, 4096][i % 3]) for i in range(39)]
    require(read(registration / 'schedule.json') == schedule, 'Registered schedule differs')
    payload = read(models)
    require(payload.get('schema_version') == 1 and payload.get('approximation_fields') == list(APPROXIMATION_FIELDS)
            and [row['case'] for row in payload['models']] == list(range(13)), 'Certified model export differs')
    expected_approximations, financial_problems = {}, {}
    for task, row in zip(tasks, payload['models']):
        base = task['source_policy']['policy']
        require(canonical(base) == task['source_policy']['fingerprint'], 'Registered base-policy fingerprint differs')
        fingerprint = canonical({key: row[key] for key in APPROXIMATION_FIELDS})
        require(row['approximation_fingerprint'] == fingerprint == row['residual_fine_fingerprint']
                and row['matching_arm_fingerprints'] == dict(reference=fingerprint, selected=fingerprint),
                'Certified approximation identity differs')
        require(all(row[key] == base[key] for key in ('model', 'option', 'decisions')),
                'Certified approximation and registered financial case differ')
        expected_approximations[task['case']] = fingerprint
        financial_problems[task['case']] = {key: base[key] for key in ('model', 'option', 'decisions')}
    launch, summary = read(run / 'launch.json'), read(run / 'summary.json')
    require(launch['registration'] == str(registration.relative_to(ROOT))
            and launch['registration_sha256'] == sha(registration / 'manifest.json'), 'Launch registration differs')
    require(launch['environment']['thread_environment'] == {key: '1' for key in THREADS}
            and launch['environment']['python_workers'] == 1, 'Recorded worker or thread requests differ')
    seen = {identifier for task in tasks for identifier in task['prior_streams']}
    prior_seen = len(seen)
    pair_count, policy_count = 0, 0

    def inspect_setup(setup, case):
        require(setup['settings'] == cfg['approximation']
                and setup['approximation_fingerprint'] == expected_approximations[case],
                'Warmup, pilot or validation setup differs from the retained case')
        return dict(cpu=finite(setup['cpu_seconds'], 'setup CPU', positive=True),
                    elapsed=finite(setup['elapsed_seconds'], 'setup elapsed', positive=True))

    def inspect(row, item, phase, repetition):
        nonlocal pair_count, policy_count
        require(all(row[key] == value for key, value in item.items()) and row['phase'] == phase
                and type(row['repetition']) is int and row['repetition'] == repetition,
                'Pair identity differs from its scheduled record')
        training, evaluation = row['training'], row['evaluation']
        require(training['training_paths'] == evaluation['training_paths'] == cfg['training_paths']
                and evaluation['evaluation_paths'] == item['evaluation_paths'], 'Numerical workload counts differ')
        require(len(training['policies']) == len(training['policy_fingerprints']) == len(training['streams']) == 2
                and len(evaluation['policy_fingerprints']) == 2, 'Exactly two fitted policies and training streams are required')
        stream_records = training['streams'] + [evaluation['stream']]
        for role, record in enumerate(stream_records):
            key = ({'warmup': 0, 'pilot': 1, 'validation': 2}[phase], item['position'], repetition,
                   item['case'], item['evaluation_paths'], role)
            count = (*cfg['training_paths'], item['evaluation_paths'])[role]
            require(record['entropy'] == cfg['root_seed'] and record['spawn_key'] == list(key)
                    and record['pool_size'] == 4 and record['bit_generator'] == 'PCG64'
                    and record['paths'] == count, 'Generator metadata differs from the declared role')
            state = np.random.PCG64(np.random.SeedSequence(cfg['root_seed'], spawn_key=key, pool_size=4)).state
            identity = canonical(state)
            require(record['stream_id'] == identity and identity not in seen, 'Generator identity differs or is reused')
            seen.add(identity)
        problem = financial_problems[item['case']]
        for role, policy in enumerate(training['policies']):
            fingerprint = canonical(policy)
            require(fingerprint == training['policy_fingerprints'][role] == evaluation['policy_fingerprints'][role],
                    'Retained policy contents do not match both recorded fingerprints')
            require(all(policy[key] == value for key, value in problem.items()),
                    'A retained policy belongs to a different financial case or exercise calendar')
            require(policy['requested_degree'] == cfg['degree']
                    and policy['training_paths'] == cfg['training_paths'][role]
                    and policy['minimum_itm_paths'] == cfg['minimum_itm_paths']
                    and policy['condition_limit'] == cfg['condition_limit']
                    and policy['training_stream_id'] == training['streams'][role]['stream_id'],
                    'Retained policy fitting specification or training stream differs')
            require(len(policy['rules']) == len(problem['decisions'])
                    and policy['rules'][-1]['status'] == 'terminal', 'Policy rule coverage differs from the calendar')
            require(all(isinstance(rule['coefficients'], list)
                        and len(rule['coefficients']) <= cfg['degree'] + 1 for rule in policy['rules']),
                    'A continuation polynomial exceeds the declared degree')
            policy_count += 1
        require(evaluation['approximation_fingerprint'] == expected_approximations[item['case']],
                'Evaluation uses a different financial approximation')
        require(isinstance(evaluation['paths_sha256'], str) and len(evaluation['paths_sha256']) == 64
                and all(c in '0123456789abcdef' for c in evaluation['paths_sha256']),
                'Invalid retained simulation-path digest')
        require(evaluation['gain']['estimate']['observations'] == item['evaluation_paths']
                and evaluation['gain']['estimate']['confidence'] == .95, 'Evaluation summary sample definition differs')
        pair_count += 1
        return inspect_timers(row)

    warm = read(run / 'warmup.json')
    require(len(warm['pairs']) == 4, 'Warmup pair coverage differs')
    warm_setup = inspect_setup(warm['setup'], 0)
    warm_timers = [inspect(row, dict(position=0, case=0, evaluation_paths=4096), 'warmup', rep)
                   for rep, row in enumerate(warm['pairs'])]
    warm_cpu = warm_setup['cpu'] + sum((row['outer_cpu'] for row in warm_timers), Fraction(0))
    warm_wall = warm_setup['elapsed'] + sum((row['outer_elapsed'] for row in warm_timers), Fraction(0))
    numerical_cpu = warm_setup['cpu'] + sum((row['pair_cpu'] for row in warm_timers), Fraction(0))
    numerical_wall = warm_setup['elapsed'] + sum((row['pair_elapsed'] for row in warm_timers), Fraction(0))
    cell_cpu, cell_wall = Fraction(0), Fraction(0)
    cells = []
    for item in schedule:
        folder = run / f"cell_{item['position']:02d}"
        pilot_setup = inspect_setup(read(folder / 'pilot_setup.json'), item['case'])
        validation_setup = inspect_setup(read(folder / 'validation_setup.json'), item['case'])
        timers = [inspect(read(folder / f'{phase}_{rep:03d}.json'), item, phase, rep)
                  for phase, repetitions in (('pilot', 8), ('validation', 64)) for rep in range(repetitions)]
        result = read(folder / 'result.json')
        require(all(result[key] == value for key, value in item.items()), 'Cell result identity differs')
        whole_cpu = finite(result['whole_cell_cpu_seconds'], 'whole-cell CPU', positive=True)
        whole_wall = finite(result['whole_cell_elapsed_seconds'], 'whole-cell elapsed', positive=True)
        required_cpu = pilot_setup['cpu'] + validation_setup['cpu'] + sum((row['outer_cpu'] for row in timers), Fraction(0))
        required_wall = pilot_setup['elapsed'] + validation_setup['elapsed'] + sum((row['outer_elapsed'] for row in timers), Fraction(0))
        contained(required_cpu, whole_cpu, 'whole-cell CPU')
        contained(required_wall, whole_wall, 'whole-cell elapsed')
        cell_cpu += whole_cpu
        cell_wall += whole_wall
        numerical_cpu += pilot_setup['cpu'] + validation_setup['cpu'] + sum((row['pair_cpu'] for row in timers), Fraction(0))
        numerical_wall += pilot_setup['elapsed'] + validation_setup['elapsed'] + sum((row['pair_elapsed'] for row in timers), Fraction(0))
        cells.append(dict(**item, policies_verified=144, streams_verified=216,
            unassigned_whole_cell_cpu_seconds=float(whole_cpu - required_cpu),
            unassigned_whole_cell_elapsed_seconds=float(whole_wall - required_wall)))
    require(pair_count == 2812 and policy_count == 5624 and len(seen) - prior_seen == 8436,
            'Complete workload, policy or stream coverage differs')
    process_cpu = finite(completed['process_cpu_seconds'], 'recorded process CPU', positive=True)
    process_wall = finite(completed['elapsed_seconds'], 'recorded process elapsed', positive=True)
    contained(warm_cpu + cell_cpu, process_cpu, 'process CPU')
    contained(warm_wall + cell_wall, process_wall, 'process elapsed')
    writing_cpu = finite(summary['validation_record_writing_cpu'], 'aggregate validation writing CPU')
    writing_wall = finite(summary['validation_record_writing_elapsed'], 'aggregate validation writing elapsed')
    contained(numerical_cpu + writing_cpu, process_cpu, 'numerical plus separately timed writing CPU')
    contained(numerical_wall + writing_wall, process_wall, 'numerical plus separately timed writing elapsed')
    retain(Path(__file__))
    for name, value in evidence.items():
        require(sha(ROOT / name) == value, 'Evidence changed during the record audit')
    report = dict(complete=True, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Supplemental file, fitted-policy, stream and timer record integrity; no numerical workloads replayed.',
        canonical_fingerprint_convention='SHA-256 of json.dumps(record, sort_keys=True, allow_nan=False), independently implemented using the original stored JSON records.',
        exact_run_artifact_coverage=True, consumed_run_json_files_bound=True,
        registered_cases=13, cells=39, pair_records=pair_count, policy_fingerprints_verified=policy_count,
        distinct_role_streams_verified=len(seen)-prior_seen, warmup_setup_and_policy_identity_verified=True,
        setup_approximation_fingerprints_verified=79, all_timer_identities_and_containments_pass=True,
        timer_containment_allowance_seconds='1/1000000000', clock_accuracy_certified=False,
        individually_recorded_write_costs_available=False,
        write_cost_scope='The retained aggregate validation-writing costs satisfy process containment. Individual write durations were not stored and are not independently reconstructed.',
        policies_refitted=0, paths_resimulated=0, financial_estimates_recomputed=False,
        cells_details=cells, total_numerical_cpu_seconds=float(numerical_cpu),
        total_numerical_elapsed_seconds=float(numerical_wall), original_evidence_unchanged=True,
        python=platform.python_version(), numpy=np.__version__, publication_clearance=False,
        evidence_sha256=evidence)
    output.mkdir(parents=True, exist_ok=False)
    pending = output / 'checks.pending.json'
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, value in evidence.items():
        require(sha(ROOT / name) == value, 'Evidence changed before audit completion')
    pending.rename(output / 'checks.json')
    return {key: report[key] for key in ('complete', 'cells', 'pair_records', 'policy_fingerprints_verified',
                                       'distinct_role_streams_verified', 'all_timer_identities_and_containments_pass')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    print(json.dumps(audit(args.registration, args.run, args.output, args.models), sort_keys=True))
