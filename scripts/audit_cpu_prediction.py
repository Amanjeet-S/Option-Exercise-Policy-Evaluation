"""Reconstruct local CPU forecasts and expenditure from retained timing records."""

import argparse
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from study_prospective_allocation import ROOT, encode, private, read, save, sha, utc, verify_manifest
from study_cpu_prediction import BASE, EXPECTED, PHASES, schedule
from qf_research.optimal_stopping.funded_policy import _digest


def F(value):
    result = Fraction(value)
    if result < 0:
        raise ValueError('A retained duration is negative')
    return result


def require_equal(actual, expected, context):
    if actual != encode(expected):
        raise ValueError('Reconstruction differs: '+context)


def precedes(first, second):
    if datetime.fromisoformat(first) > datetime.fromisoformat(second):
        raise ValueError('A prediction or calibration postdates validation')


def audit(registration, run, output):
    registration, run, output = map(private, (registration, run, output))
    if output.exists():
        raise FileExistsError('Use a fresh audit directory')
    evidence = verify_manifest(run/'manifest.json')
    if read(registration/'configuration.json') != EXPECTED:
        raise ValueError('Configuration differs from the retained design')
    launch = read(run/'launch.json')
    if launch['registration_sha256'] != sha(registration/'manifest.json'):
        raise ValueError('Launch used another registration')
    if any(v != '1' for v in launch['environment']['thread_environment'].values()):
        raise ValueError('Numerical thread settings differ')
    if any(p['num_threads'] != 1 for p in launch['environment'].get('reported_native_pools') or []):
        raise ValueError('Reported native pool used multiple threads')
    arms = read(registration/'arms.json')
    if [{k: a[k] for k in ('pass_id', 'position', 'case', 'evaluation_paths')} for a in arms] != schedule():
        raise ValueError('Registered ordering differs')
    forbidden = set(read(registration/'forbidden_streams.json'))
    observed = set()
    row_count = 0
    numerical_totals = dict(cpu=Fraction(0), elapsed=Fraction(0))
    validation_totals = dict(cpu=Fraction(0), elapsed=Fraction(0))
    calibration_totals = dict(cpu=Fraction(0), elapsed=Fraction(0))
    all_comparisons = {name: [] for name in ('historical', 'pass_start', 'immediate')}
    charged_affordable = 0
    preceding_finish = dict(cpu_since_launch=0., elapsed_since_launch=0.,
                            recorded_at_utc=launch['started_at_utc'])

    def block(item, folder, phase, count):
        nonlocal row_count, preceding_finish
        start, finish = read(folder/'started.json'), read(folder/'finished.json')
        precedes(preceding_finish['recorded_at_utc'], start['recorded_at_utc'])
        precedes(start['recorded_at_utc'], finish['recorded_at_utc'])
        if any(F(start[key]) < F(preceding_finish[key]) for key in ('cpu_since_launch', 'elapsed_since_launch')):
            raise ValueError('Consecutive numerical blocks overlap')
        if start['cpu_since_launch'] > finish['cpu_since_launch'] or start['elapsed_since_launch'] > finish['elapsed_since_launch']:
            raise ValueError('A monotonic timer decreased')
        setup = read(folder/'setup.json')
        if setup['approximation_fingerprint'] != item['approximation_fingerprint']:
            raise ValueError('Approximation differs from the registered task')
        files = sorted(folder.glob('row_*.json'))
        if [p.name for p in files] != [f'row_{i:04d}.json' for i in range(count)]:
            raise ValueError('Incomplete or excess timing rows')
        cost = {clock: F(setup[key]) for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds'))}
        values = {clock: [] for clock in cost}
        for rep, path in enumerate(files):
            row = read(path)
            if row['phase'] != phase or row['repetition'] != rep or any(row[k] != item[k] for k in ('pass_id', 'case', 'evaluation_paths')):
                raise ValueError('Timing-row identity differs')
            streams = row['training']['streams']+[row['evaluation']['stream']]
            if row['training']['training_paths'] != [5000, 20000] or row['evaluation']['evaluation_paths'] != item['evaluation_paths']:
                raise ValueError('Numerical workload size changed')
            if row['training']['policy_fingerprints'] != row['evaluation']['policy_fingerprints']:
                raise ValueError('Evaluation used different trained policies')
            if row['evaluation']['approximation_fingerprint'] != item['approximation_fingerprint']:
                raise ValueError('Evaluation approximation differs')
            for role, stream in enumerate(streams):
                spawn = [PHASES[phase], item['pass_id'], item['case'], item['evaluation_paths'], rep, role]
                seed = np.random.SeedSequence(EXPECTED['root_seed'], spawn_key=tuple(spawn))
                identity = _digest(np.random.PCG64(seed).state)
                if stream['entropy'] != EXPECTED['root_seed'] or stream['spawn_key'] != spawn or stream['stream_id'] != identity:
                    raise ValueError('Random stream is not the registered stream')
                if identity in forbidden or identity in observed:
                    raise ValueError('A random stream was reused')
                observed.add(identity)
            for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds')):
                value = F(row['training'][key])+F(row['evaluation'][key])
                if value <= 0 or F(row[f'outer_{key}'])+Fraction(1, 10**8) < value:
                    raise ValueError('Numerical duration is not inside the pair timer')
                values[clock].append(value)
                cost[clock] += value
            row_count += 1
        require_equal(finish['numerical_charge'], cost, str(folder)+' charge')
        for clock in cost:
            key = clock+'_since_launch'
            if F(finish[key])-F(start[key])+Fraction(1, 10**8) < cost[clock]:
                raise ValueError('Numerical charge exceeds its complete block timer')
            numerical_totals[clock] += cost[clock]
        preceding_finish = finish
        return setup, values, cost, start, finish

    def check_prediction(item, folder, phase):
        setup, values, charge, start, finish = block(item, folder, phase, 16)
        record = read(folder/'forecast.json')
        precedes(finish['recorded_at_utc'], record['recorded_at_utc'])
        expected = {clock: F(setup[key])+item['repetitions']*sum(values[clock])/16
            for clock, key in (('cpu', 'cpu_seconds'), ('elapsed', 'elapsed_seconds'))}
        require_equal(record['predictions'], expected, 'local mean prediction')
        require_equal(record['calibration_charge'], charge, 'calibration charge')
        require_equal(record['allowance_predictions'], {k: Fraction(11, 10)*v for k, v in expected.items()}, 'allowance')
        for clock in values:
            require_equal(record['first_half_pair_mean'][clock], sum(values[clock][:8])/8, 'first half')
            require_equal(record['last_half_pair_mean'][clock], sum(values[clock][-8:])/8, 'last half')
            calibration_totals[clock] += charge[clock]
        return record, expected, charge

    _, _, warmup_cost, _, _ = block(arms[0], run/'warmup', 'warmup', 4)
    for pass_id in range(2):
        pass_arms = [a for a in arms if a['pass_id'] == pass_id]
        folder = run/f'pass_{pass_id}'
        frozen = read(folder/'forecasts_frozen.json')
        early = {}
        for item in pass_arms:
            key = f"case_{item['case']:02d}_n_{item['evaluation_paths']:04d}"
            pred_folder = folder/f'calibration_{key}'
            early[key] = check_prediction(item, pred_folder, 'pass_calibration')
            if frozen['forecast_sha256'][key] != sha(pred_folder/'forecast.json'):
                raise ValueError('Pass-start forecast changed')
            precedes(early[key][0]['recorded_at_utc'], frozen['recorded_at_utc'])
        for item in pass_arms:
            key = f"case_{item['case']:02d}_n_{item['evaluation_paths']:04d}"
            local_record, local_pred, local_charge = check_prediction(item, folder/f'local_{key}', 'local_calibration')
            pred_file = folder/f'predictions_{key}.json'
            predictions = read(pred_file)
            result = read(folder/f'result_{key}.json')
            require_equal(predictions['predictions']['pass_start'], early[key][0], 'frozen early prediction')
            require_equal(predictions['predictions']['immediate'], local_record, 'frozen immediate prediction')
            if result['prediction_sha256_before_validation'] != sha(pred_file):
                raise ValueError('Forecast hash changed after validation')
            setup, values, actual, start, finish = block(item, folder/f'validation_{key}', 'validation', item['repetitions'])
            for record in (frozen, local_record, predictions):
                precedes(record['recorded_at_utc'], start['recorded_at_utc'])
            require_equal(result['actual'], actual, 'validation actual')
            require_equal(result['validation_started'], start, 'validation start')
            require_equal(result['validation_finished'], finish, 'validation finish')
            combined = {k: local_charge[k]+early[key][2][k] for k in actual}
            require_equal(result['calibration_charge'], combined, 'combined calibration')
            require_equal(result['charged_numerical_cost'], {k: actual[k]+combined[k] for k in actual}, 'operational charge')
            budget = Fraction(item['budget'])
            charged_affordable += actual['cpu']+combined['cpu'] <= budget
            require_equal(result['charged_cpu_affordable'], actual['cpu']+combined['cpu'] <= budget, 'charged affordability')
            q = max(1, item['repetitions']//4)
            require_equal(result['first_quarter_pair_cpu'], sum(values['cpu'][:q])/q, 'first quarter')
            require_equal(result['last_quarter_pair_cpu'], sum(values['cpu'][-q:])/q, 'last quarter')
            for k in actual:
                validation_totals[k] += actual[k]
            forecasts = dict(historical=dict(cpu=Fraction(item['historical_prediction'])),
                             pass_start=early[key][1], immediate=local_pred)
            for name, clocks in forecasts.items():
                for clock, pred in clocks.items():
                    ratio, allowance = actual[clock]/pred, Fraction(11, 10)*pred
                    expect = dict(actual=actual[clock], predicted=pred, ratio=ratio,
                        absolute_relative_error=abs(ratio-1), within_tolerance=Fraction(9, 10) <= ratio <= Fraction(11, 10),
                        allowance=allowance, within_allowance=actual[clock] <= allowance)
                    if clock == 'cpu':
                        expect.update(forecast_affordable=pred <= budget, allowance_affordable=allowance <= budget,
                            actual_affordable=actual[clock] <= budget,
                            falsely_forecast_affordable=pred <= budget < actual[clock],
                            falsely_allowance_affordable=allowance <= budget < actual[clock])
                        all_comparisons[name].append(expect)
                    require_equal(result['comparisons'][name][clock], expect, name+' comparison')
    summary = read(run/'summary.json')
    require_equal(summary['numerical_totals'], numerical_totals, 'all numerical timers')
    require_equal(summary['numerical_validation_cost'], validation_totals, 'validation timers')
    require_equal(summary['calibration_cost'], calibration_totals, 'calibration timers')
    require_equal(summary['warmup_charge'], warmup_cost, 'warm-up timers')
    process_overhead = {}
    for clock, key in (('cpu', 'process_cpu_seconds'), ('elapsed', 'process_elapsed_seconds')):
        total = F(summary[key])
        if total < numerical_totals[clock] or total < F(preceding_finish[clock+'_since_launch']):
            raise ValueError('Process total fails to contain the retained numerical blocks')
        process_overhead[clock] = total-numerical_totals[clock]
    for key, expected in dict(validation_arms=18, validation_pairs=7956, calibration_pairs=576,
                             warmup_pairs=4, counts_changed=False, prediction_coverage_claim=False,
                             causal_effect_identified=False).items():
        require_equal(summary[key], expected, key)
    for name, entries in all_comparisons.items():
        expected = {k: sum(e[k] for e in entries) for k in ('within_tolerance', 'within_allowance', 'forecast_affordable',
            'allowance_affordable', 'falsely_forecast_affordable', 'falsely_allowance_affordable', 'actual_affordable')}
        expected.update(mean_absolute_relative_error=sum(e['absolute_relative_error'] for e in entries)/18,
            ratio_min=min(e['ratio'] for e in entries), ratio_max=max(e['ratio'] for e in entries))
        require_equal(summary['comparisons'][name], expected, name+' aggregate')
    if summary['charged_arms_affordable'] != charged_affordable or row_count != 8536 or len(observed) != 25608:
        raise ValueError('Final counts differ')
    if summary['numerical_random_streams'] != len(observed) or not summary['complete']:
        raise ValueError('Run is incomplete')
    output.mkdir(parents=True)
    save(output/'checks.json', dict(complete=True, checked_at_utc=utc(), timing_rows=row_count,
        validation_pairs=7956, calibration_pairs=576, warmup_pairs=4, fresh_streams=len(observed),
        forbidden_streams=len(forbidden), exact_timer_reconstruction=True,
        forecasts_precede_validation=True, independent_forecast_and_summary_reconstruction=True,
        source_and_artifact_hashes_verified=True, financial_replay=False,
        consecutive_block_order_verified=True, block_and_process_timer_containment=True,
        process_minus_numerical_cost=process_overhead,
        prediction_coverage_claim=False, clock_accuracy_certified=False))
    sources = [Path(__file__).resolve(), ROOT/'tests/test_cpu_prediction_audit.py']
    for path in sources:
        evidence[str(path.relative_to(ROOT))] = sha(path)
    save(output/'manifest.json', dict(complete=True, recorded_at_utc=utc(), evidence_sha256=evidence,
        artifact_sha256={'checks.json': sha(output/'checks.json')}, publication_clearance=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=BASE/'registration_v1')
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.registration, args.run, args.output)
    print('All complete-arm forecasts, timing rows and expenditure summaries reconstructed.')


if __name__ == '__main__':
    main()
