"""Independently reconstruct the complete local timing study from retained timers."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def audit(registration, run, output):
    registration, run, output = (p.resolve() for p in (registration, run, output))
    require(all(p.is_relative_to(ROOT/'data/private') for p in (registration, run, output)), 'Private paths required')
    require(not output.exists(), 'Fresh audit output required')
    evidence = {}

    def retain(path, digest=None):
        require(path.resolve().is_relative_to(ROOT), 'Unsafe evidence path')
        actual = sha(path)
        require(digest is None or actual == digest, 'Changed evidence: '+str(path))
        name = str(path.relative_to(ROOT))
        require(name not in evidence or evidence[name] == actual, 'Conflicting evidence')
        evidence[name] = actual

    for folder in (registration, run):
        manifest = read(folder/'manifest.json')
        retain(folder/'manifest.json')
        for group, base in (('input_sha256', ROOT), ('source_sha256', ROOT), ('artifact_sha256', folder)):
            for name, digest in manifest[group].items():
                require((base/name).resolve().is_relative_to(base.resolve()), 'Unsafe manifest entry')
                retain(base/name, digest)
    manifest = read(run/'manifest.json')
    require(manifest.get('complete') is True and not (run/'incomplete.json').exists(), 'Incomplete timing evidence')
    cfg = read(registration/'configuration.json')
    require(cfg['cases'] == list(range(13)) and cfg['evaluation_grid'] == [128,256,4096]
            and cfg['pilot_pairs'] == 8 and cfg['validation_pairs'] == 64 and cfg['warmup_pairs'] == 4,
            'Registered counts differ')
    schedule = [dict(position=i, case=5*i%13, evaluation_paths=[128,256,4096][i%3]) for i in range(39)]
    require(read(registration/'schedule.json') == schedule, 'Schedule differs')
    launch, summary = read(run/'launch.json'), read(run/'summary.json')
    require(launch['registration_sha256'] == sha(registration/'manifest.json'), 'Launch binding differs')
    require(datetime.fromisoformat(read(registration/'manifest.json')['recorded_at_utc']) < datetime.fromisoformat(launch['started_at_utc']), 'Registration was not recorded before launch')
    require(all(v == '1' for v in launch['environment']['thread_environment'].values()), 'Thread request differs')
    tasks = read(registration/'tasks.json')
    seen = {s for t in tasks for s in t['prior_streams']}
    initial_seen = len(seen)
    keys = set()
    count = 0
    last_time = datetime.fromisoformat(launch['started_at_utc'])

    def inspect(row, item, phase, rep):
        nonlocal count, last_time
        require(all(row[k] == v for k,v in item.items()) and row['phase'] == phase and row['repetition'] == rep, 'Pair identity differs')
        before, after = (datetime.fromisoformat(row['telemetry_'+suffix]['recorded_at_utc']) for suffix in ('before','after'))
        require(last_time <= before <= after, 'Pair chronology differs')
        last_time = after
        t, e = row['training'], row['evaluation']
        require(t['training_paths'] == [5000,20000] and e['training_paths'] == [5000,20000]
                and e['evaluation_paths'] == item['evaluation_paths'], 'Numerical workload counts differ')
        require(t['policy_fingerprints'] == e['policy_fingerprints'], 'Evaluation does not use trained policies')
        stream_records = t['streams'] + [e['stream']]
        for role, record in enumerate(stream_records):
            key = ({'warmup':0,'pilot':1,'validation':2}[phase], item['position'], rep, item['case'], item['evaluation_paths'], role)
            require(tuple(record['spawn_key']) == key and record['entropy'] == cfg['root_seed']
                    and record['pool_size'] == 4 and record['bit_generator'] == 'PCG64', 'Generator specification differs')
            require(record['paths'] == ([5000,20000,item['evaluation_paths']][role]), 'Stream path count differs')
            state = np.random.PCG64(np.random.SeedSequence(cfg['root_seed'], spawn_key=key)).state
            digest = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
            require(digest == record['stream_id'] and digest not in seen and key not in keys, 'Stream identity mismatch or reuse')
            seen.add(digest)
            keys.add(key)
        require(row['pair_cpu_seconds'] == t['cpu_seconds']+e['cpu_seconds'] and row['pair_cpu_seconds']>0, 'Pair timer sum differs')
        count += 1
        return row

    def components(row):
        # Decimal arithmetic independently reproduces the runner's rational sums.
        t = row['training']
        sim, fit = (Decimal.from_float(t['phases'][k]) for k in ('simulation_cpu_seconds','fitting_cpu_seconds'))
        tr, ev, pair = (Decimal.from_float(v) for v in (t['cpu_seconds'], row['evaluation']['cpu_seconds'], row['pair_cpu_seconds']))
        return dict(training_simulation=sim, training_fitting=fit, other_training=tr-sim-fit,
                    evaluation=ev, timer_sum_rounding=pair-tr-ev)

    warm = read(run/'warmup.json')
    require(len(warm['pairs']) == 4, 'Warmup count differs')
    for rep,row in enumerate(warm['pairs']):
        inspect(row,dict(position=0,case=0,evaluation_paths=4096),'warmup',rep)
    warm_cpu = sum((Fraction(r['pair_cpu_seconds']) for r in warm['pairs']), Fraction(warm['setup']['cpu_seconds']))
    cells = []
    with localcontext() as ctx:
        ctx.prec = 200
        for item in schedule:
            folder = run/f"cell_{item['position']:02d}"
            pilots = [inspect(read(folder/f'pilot_{rep:03d}.json'),item,'pilot',rep) for rep in range(8)]
            forecast, result = read(folder/'forecast.json'), read(folder/'result.json')
            require(all(forecast[k] == v and result[k] == v for k,v in item.items()), 'Cell identity differs')
            forecast_time = datetime.fromisoformat(forecast['recorded_at_utc'])
            start_time = datetime.fromisoformat(result['validation_started_at_utc'])
            require(last_time <= forecast_time <= start_time, 'Forecast does not precede validation')
            require(result['forecast_sha256_before_validation'] == sha(folder/'forecast.json'), 'Forecast changed')
            validation = [inspect(read(folder/f'validation_{rep:03d}.json'),item,'validation',rep) for rep in range(64)]
            require(start_time <= datetime.fromisoformat(validation[0]['telemetry_before']['recorded_at_utc']), 'Validation pair predates forecast')
            ps, vs = read(folder/'pilot_setup.json'), read(folder/'validation_setup.json')
            require(ps['settings'] == cfg['approximation'] == vs['settings'] and ps['approximation_fingerprint'] == vs['approximation_fingerprint'], 'Setup changed')
            require(all(r['evaluation']['approximation_fingerprint'] == ps['approximation_fingerprint'] for r in pilots+validation), 'Pair approximation differs')
            predictions = {key:8*sum(components(r)[key] for r in pilots) for key in components(pilots[0])}
            actuals = {key:sum(components(r)[key] for r in validation) for key in predictions}
            predictions['setup'], actuals['setup'] = Decimal.from_float(ps['cpu_seconds']), Decimal.from_float(vs['cpu_seconds'])
            predicted, actual = sum(predictions.values()), sum(actuals.values())
            require(Fraction(predicted) == Fraction(forecast['exact_predicted_block_cpu']) == Fraction(result['exact_predicted_block_cpu']), 'Forecast arithmetic differs')
            require(Fraction(actual) == Fraction(result['exact_actual_block_cpu']), 'Actual arithmetic differs')
            ratio = Fraction(actual)/Fraction(predicted)
            within = Fraction(9,10)<=ratio<=Fraction(11,10)
            require(str(ratio) == result['exact_ratio'] and float(ratio) == result['actual_to_predicted_ratio'] and within == result['within_registered_tolerance'], 'Ratio or tolerance differs')
            for key in predictions:
                require(Fraction(predictions[key]) == Fraction(forecast['exact_phase_predictions_cpu'][key]) and float(predictions[key]) == forecast['phase_predictions_cpu'][key], 'Phase forecast differs')
                require(Fraction(actuals[key]) == Fraction(result['exact_phase_actual_cpu'][key]) and float(actuals[key]) == result['phase_actual_cpu'][key], 'Actual phase differs')
                require(float(actuals[key]-predictions[key]) == result['phase_changes_cpu'][key], 'Phase discrepancy differs')
            require(result == summary['cells'][item['position']], 'Summary cell differs')
            costs = [r['pair_cpu_seconds'] for r in pilots]
            require(forecast['pilot_mean_pair_cpu'] == statistics.mean(costs) and forecast['pilot_sd_pair_cpu'] == statistics.stdev(costs)
                    and forecast['pilot_cv_pair_cpu'] == statistics.stdev(costs)/statistics.mean(costs), 'Pilot dispersion differs')
            overhead = Fraction(ps['cpu_seconds'])+sum((Fraction(x) for x in costs),Fraction(0))
            require(float(overhead) == forecast['pilot_numerical_cpu'] == result['pilot_numerical_cpu'], 'Pilot charge differs')
            for name, subset in [('first_eight',validation[:8]),('last_eight',validation[-8:]),('first_half',validation[:32]),('second_half',validation[32:])]:
                require(result[name+'_mean_pair_cpu'] == statistics.mean(r['pair_cpu_seconds'] for r in subset), 'Drift diagnostic differs')
            cells.append(dict(**item, predicted_cpu=float(predicted), actual_cpu=float(actual),
                ratio=float(ratio), within_tolerance=within, pilot_cpu=float(overhead),
                pilot_cv=forecast['pilot_cv_pair_cpu'],
                last_to_first_eight_ratio=result['last_eight_mean_pair_cpu']/result['first_eight_mean_pair_cpu'],
                phase_change={k:float(actuals[k]-predictions[k]) for k in predictions}))
    require(count == 2812 and len(seen)-initial_seen == 8436 and len(keys) == 8436, 'Pair or stream coverage differs')
    require(len(list(run.glob('cell_*'))) == 39, 'Unexpected cell directory')
    require(summary['pilot_pairs']==312 and summary['validation_pairs']==2496 and summary['timing_cells']==39, 'Summary counts differ')
    require(summary['within_registered_tolerance'] == sum(c['within_tolerance'] for c in cells), 'Pass count differs')
    require(summary['minimum_ratio']==min(c['ratio'] for c in cells) and summary['maximum_ratio']==max(c['ratio'] for c in cells)
            and summary['median_ratio']==statistics.median(c['ratio'] for c in cells), 'Ratio summary differs')
    require(summary['pilot_numerical_cpu']==sum(c['pilot_cpu'] for c in cells)
            and summary['validation_numerical_cpu']==sum(c['actual_cpu'] for c in cells), 'Aggregate charge differs')
    retain(Path(__file__).resolve())
    for name,digest in evidence.items():
        require(sha(ROOT/name)==digest, 'Evidence changed during audit')
    output.mkdir(parents=True)
    (output/'cells.json').write_text(json.dumps(cells,indent=2)+'\n')
    record=dict(complete=True, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        method='Separate implementation using 200-digit Decimal timer aggregation and exact rational comparisons',
        cases=13,cells=39,pilot_pairs=312,validation_pairs=2496,warmup_pairs=4,
        unique_streams=8436,all_forecasts_precede_validation=True,
        all_phase_identities_pass=True,all_registered_cells_retained=True,
        within_tolerance=sum(c['within_tolerance'] for c in cells),
        warmup_numerical_cpu=float(warm_cpu),original_evidence_unchanged=True,
        statistical_gain_inference=False,causal_timing_comparison=False,
        publication_clearance=False,evidence_sha256=evidence,
        artifact_sha256={'cells.json':sha(output/'cells.json')})
    (output/'checks.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('evidence_sha256','artifact_sha256')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.registration,args.run,args.output)
