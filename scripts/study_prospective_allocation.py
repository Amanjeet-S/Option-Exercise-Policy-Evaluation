"""Freeze and execute a targeted prospective comparison of CPU allocations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from study_cpu_budget_transfer import environment, telemetry
from study_training_allocation import THREADS
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.allocation_workload import build_approximation, train_pair, evaluate_pair
from qf_research.optimal_stopping.allocation_variance_sensitivity import select
from qf_research.optimal_stopping.funded_policy import _digest
from qf_research.optimal_stopping.prospective_allocation import (
    minimise_relative_variance, sample_summary, compare_summaries,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/private/prospective_allocation/2026-09-28'
PREVIOUS = ROOT/'data/private/component_ratio/2026-09-28'
COST = ROOT/'data/private/allocation_cost_sensitivity/2026-09-27'
TRAINING = ROOT/'data/private/training_allocation/2026-09-27'
EXPECTED = dict(cases=[5, 6, 10], evaluation_grid=[128, 256, 4096],
    training_paths=[5000, 20000], degree=4, minimum_itm_paths=32, condition_limit=1e8,
    approximation=dict(core_steps=800, wing_steps=40, tolerance=.001),
    root_seed=2026092856101282564096, warmup_pairs=4,
    cpu_limit=1800, elapsed_limit=5400, critical_value='13/4', contrast_family_size=12,
    accounting='marginal', scope='joint', reference_paths=4096,
    budget_ratio_limits=['9/10', '11/10'], expected_primary_pairs=3978,
    prior_outcomes_used_for_design=True, new_outcomes_inspected=False,
    public_registration=False, publication_clearance=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(encode(value), indent=2, allow_nan=False)+'\n')


def private(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT/'data/private'):
        raise ValueError('Require private numerical storage')
    return path


def merge(evidence, mapping, anchor=ROOT):
    for name, digest in mapping.items():
        path = (anchor/name).resolve()
        if not path.is_relative_to(anchor.resolve()) or sha(path) != digest:
            raise ValueError('Changed or unsafe evidence: '+name)
        relative = str(path.relative_to(ROOT))
        if relative in evidence and evidence[relative] != digest:
            raise ValueError('Conflicting evidence: '+relative)
        evidence[relative] = digest


def verify_manifest(path):
    result = read(path)
    if result.get('complete') is not True:
        raise ValueError('Completed evidence required')
    checked = {}
    merge(checked, result['evidence_sha256'])
    if 'artifact_sha256' in result:
        merge(checked, result['artifact_sha256'], path.parent)
    checked[str(path.relative_to(ROOT))] = sha(path)
    return checked


def stream_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('stream_id', 'training_stream_id') and isinstance(item, str):
                found.add(item)
            elif key in ('prior_streams', 'forbidden_streams', 'stream_ids') and isinstance(item, list):
                found.update(x for x in item if isinstance(x, str))
            else:
                found.update(stream_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(stream_ids(item))
    return found


def schedule():
    return [dict(position=i, case=EXPECTED['cases'][i % 3],
                 evaluation_paths=EXPECTED['evaluation_grid'][(i//3+i % 3) % 3])
            for i in range(9)]


def register(output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh registration directory')
    evidence = verify_manifest(PREVIOUS/'completion_v1.json')
    merge(evidence, verify_manifest(PREVIOUS/'figures_v1/manifest.json'))
    models = []
    old_cost = read(COST/'inputs_v1/inputs.json')
    cost_results = read(COST/'run_v1/summary.json')
    enclosures = read(PREVIOUS/'run_v1/enclosures.json')
    if Fraction(enclosures['normal_family_tail']['upper']) >= Fraction(1, 20):
        raise ValueError('The retained normal calibration is unresolved')
    for path in (COST/'inputs_v1/inputs.json', COST/'run_v1/summary.json',
                 ROOT/'data/private/allocation_variance_sensitivity/2026-09-28/inputs_v1/inputs.json',
                 TRAINING/'main_registration_v1/tasks.json', PREVIOUS/'run_v1/enclosures.json'):
        if str(path.relative_to(ROOT)) not in evidence:
            raise ValueError('Missing verified input: '+str(path.relative_to(ROOT)))
    for case in EXPECTED['cases']:
        old = next(r for r in old_cost['models'] if r['case'] == case)
        result = read(PREVIOUS/f'run_v1/case_{case:02d}_selected.json')
        enc = next(r for r in enclosures['sets'] if (r['case'], r['source']) == (case, 'selected'))
        parts = [dict(lower=max(Fraction(0), Fraction(r['lower']['lower'])),
                      upper=min(Fraction(1), Fraction(r['upper']['upper']))) for r in enc['components']]
        old_candidates = next(r for r in cost_results['models'] if r['case'] == case)['accounting']['marginal']['choices']['joint']['candidates']
        budget = Fraction(old['budget'])
        candidates = []
        for n in EXPECTED['evaluation_grid']:
            candidate = next(r for r in old_candidates if r['evaluation_paths'] == n)
            r = candidate['repetitions']
            predictions = {name: Fraction(costs[str(n)]['setup'])+r*Fraction(costs[str(n)]['pair'])
                           for name, costs in old['scenarios'].items()}
            if not candidate['feasible'] or max(predictions.values()) > budget:
                raise ValueError('A registered candidate is not jointly affordable')
            if predictions != {name: Fraction(v) for name, v in candidate['scenario_costs'].items()}:
                raise ValueError('Cost reconstruction differs')
            candidates.append(dict(evaluation_paths=n, repetitions=r, predicted_scenario_costs=predictions,
                                   planning_charge=max(predictions.values())))
        basic = [{k: r[k] for k in ('evaluation_paths', 'repetitions')} for r in candidates]
        old_regime = next(r for r in result['regimes'] if (r['scope'], r['accounting']) == ('joint', 'marginal'))
        if basic != old_regime['candidates']:
            raise ValueError('Candidate family differs from the ratio analysis')
        point = Fraction(result['moments']['point_share'])
        models.append(dict(case=case, budget=budget, components=parts,
            source='selected', point_share=point, candidates=candidates,
            minimax=minimise_relative_variance(basic, parts), nominal=select(basic, point),
            uniform_size=old_regime['uniformly_selected_size'],
            approximation_fingerprint=next(m for m in read(PREVIOUS/'inputs_v1/inputs.json')['models'] if m['case'] == case)['approximation_fingerprint']))
    if sum(c['repetitions'] for m in models for c in m['candidates']) != EXPECTED['expected_primary_pairs']:
        raise ValueError('Registered row count differs')
    tasks = [t for t in read(TRAINING/'main_registration_v1/tasks.json') if t['case'] in EXPECTED['cases']]
    forbidden = set()
    for name in evidence:
        if name.endswith('.json'):
            forbidden.update(stream_ids(read(ROOT/name)))
    for path in (Path(__file__).resolve(), ROOT/'src/qf_research/optimal_stopping/prospective_allocation.py',
                 ROOT/'tests/test_prospective_allocation.py', BASE/'design_v1/protocol.md',
                 BASE/'theory_v1/prospective_comparison.md', ROOT/'scripts/audit_prospective_allocation.py',
                 ROOT/'tests/test_prospective_allocation_audit.py'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    merge({}, evidence)
    output.mkdir(parents=True)
    save(output/'configuration.json', EXPECTED)
    save(output/'allocations.json', dict(models=models, primary_pairs=3978,
         planned_numerical_charge=sum(c['planning_charge'] for m in models for c in m['candidates']),
         forbidden_stream_ids=sorted(forbidden), new_outcomes_inspected=False))
    save(output/'tasks.json', tasks); save(output/'schedule.json', schedule())
    (output/'protocol.md').write_text((BASE/'design_v1/protocol.md').read_text())
    save(output/'manifest.json', dict(complete=True, recorded_at_utc=utc(),
         stage='prospective_allocation_registration', evidence_sha256=evidence,
         artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir())},
         public_registration=False, publication_clearance=False))


def seeds(phase, case, n, repetition):
    return tuple(np.random.SeedSequence(EXPECTED['root_seed'], spawn_key=(phase, case, n, repetition, role))
                 for role in range(3))


def pair(base, approximation, phase, case, n, repetition, seen):
    streams = seeds(phase, case, n, repetition)
    identifiers = [_digest(np.random.PCG64(s).state) for s in streams]
    if len(set(identifiers)) != 3 or seen.intersection(identifiers):
        raise ValueError('Numerical random stream reused')
    seen.update(identifiers)
    wall, cpu = time.perf_counter(), time.process_time()
    policies, training = train_pair(base, tuple(EXPECTED['training_paths']), degree=EXPECTED['degree'],
        minimum_itm_paths=EXPECTED['minimum_itm_paths'], condition_limit=EXPECTED['condition_limit'], seeds=streams[:2])
    evaluation = evaluate_pair(policies, approximation, n, streams[2])
    if [s['stream_id'] for s in training['streams']]+[evaluation['stream']['stream_id']] != identifiers:
        raise ValueError('Returned streams differ from registered streams')
    return dict(case=case, evaluation_paths=n, repetition=repetition, phase=phase,
        training=training, evaluation=evaluation,
        outer_cpu_seconds=time.process_time()-cpu, outer_elapsed_seconds=time.perf_counter()-wall)


def arm_summary(case, candidate, budget, setup, rows):
    count, n = candidate['repetitions'], candidate['evaluation_paths']
    if len(rows) != count or [r['repetition'] for r in rows] != list(range(count)):
        raise ValueError('Incomplete arm cannot supply the registered estimator')
    if any((r['case'], r['evaluation_paths'], r['phase']) != (case, n, 1) for r in rows):
        raise ValueError('Arm identity differs')
    means = [r['evaluation']['gain']['estimate']['value'] for r in rows]
    cpu_rows = [Fraction(r['training']['cpu_seconds'])+Fraction(r['evaluation']['cpu_seconds']) for r in rows]
    elapsed_rows = [Fraction(r['training']['elapsed_seconds'])+Fraction(r['evaluation']['elapsed_seconds']) for r in rows]
    charge = Fraction(setup['cpu_seconds'])+sum(cpu_rows)
    prediction = Fraction(candidate['planning_charge'])
    q = max(1, count//4)
    ratio = charge/prediction
    return dict(case=case, evaluation_paths=n, repetitions=count,
        statistics=sample_summary(means), setup=setup,
        exact_numerical_cpu=charge, exact_numerical_elapsed=Fraction(setup['elapsed_seconds'])+sum(elapsed_rows),
        budget=budget, planning_charge=prediction, actual_to_budget=charge/budget,
        over_budget=charge > budget, actual_to_forecast=ratio,
        within_forecast_tolerance=Fraction(9, 10) <= ratio <= Fraction(11, 10),
        cpu_quartiles=dict(first_mean=sum(cpu_rows[:q])/q, last_mean=sum(cpu_rows[-q:])/q, rows_per_quartile=q),
        zero_gain_pairs=sum(r['evaluation']['nonzero_gain_count'] == 0 for r in rows),
        complete=True)


def run(registration, output):
    registration, output = private(registration), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh execution directory')
    evidence = verify_manifest(registration/'manifest.json')
    if read(registration/'configuration.json') != EXPECTED or read(registration/'schedule.json') != schedule():
        raise ValueError('Frozen design differs')
    if any(os.environ.get(k) != '1' for k in THREADS):
        raise ValueError('Set the four numerical thread variables to one')
    allocations = read(registration/'allocations.json')
    bycase = {m['case']: m for m in allocations['models']}
    bases = {t['case']: load_frozen(t['source_policy'])[0] for t in read(registration/'tasks.json')}
    env = environment()
    pools = env.get('reported_native_pools')
    if pools and any(p.get('num_threads') != 1 for p in pools):
        raise ValueError('A reported numerical thread pool is not limited to one thread')
    seen = set(allocations['forbidden_stream_ids'])
    output.mkdir(parents=True); (output/'rows').mkdir()
    save(output/'launch.json', dict(started_at_utc=utc(), registration=str(registration.relative_to(ROOT)),
        registration_sha256=sha(registration/'manifest.json'), environment=env,
        telemetry=telemetry(), process_id=os.getpid()))
    cpu, wall = time.process_time(), time.monotonic()

    def guard():
        if time.process_time()-cpu > EXPECTED['cpu_limit'] or time.monotonic()-wall > EXPECTED['elapsed_limit']:
            raise RuntimeError('Declared resource cap reached; no completed-sample claim')

    try:
        approximation, setup = build_approximation(bases[5], **EXPECTED['approximation'])
        warm = []
        for rep in range(EXPECTED['warmup_pairs']):
            guard(); warm.append(pair(bases[5], approximation, 0, 5, 4096, rep, seen))
        save(output/'warmup.json', dict(setup=setup, rows=warm))
        arms = []
        for item in schedule():
            case, n = item['case'], item['evaluation_paths']
            model = bycase[case]
            candidate = next(r for r in model['candidates'] if r['evaluation_paths'] == n)
            guard(); before = telemetry()
            arm_cpu, arm_wall = time.process_time(), time.perf_counter()
            approximation, setup = build_approximation(bases[case], **EXPECTED['approximation'])
            if setup['approximation_fingerprint'] != model['approximation_fingerprint']:
                raise ValueError('Value approximation changed')
            save(output/f'setup_{case:02d}_{n}.json', setup)
            rows = []
            for rep in range(candidate['repetitions']):
                guard()
                row = pair(bases[case], approximation, 1, case, n, rep, seen)
                rows.append(row)
                save(output/'rows'/f'case_{case:02d}_n_{n:04d}_rep_{rep:04d}.json', row)
                guard()
                if (rep+1) % 128 == 0:
                    print(f'Case {case}, {n} evaluation paths: {rep+1}/{candidate["repetitions"]} pairs', flush=True)
            # Outer block timers end before exact statistical aggregation.
            outer_cpu, outer_wall = time.process_time()-arm_cpu, time.perf_counter()-arm_wall
            guard()
            result = arm_summary(case, candidate, Fraction(model['budget']), setup, rows)
            result.update(position=item['position'], outer_cpu_seconds=outer_cpu,
                          outer_elapsed_seconds=outer_wall, telemetry_before=before, telemetry_after=telemetry())
            save(output/f'arm_{case:02d}_{n}.json', result); arms.append(result)
            print(f'Completed case {case}, {n} paths: {candidate["repetitions"]} pairs', flush=True)
        comparisons = []
        for case in EXPECTED['cases']:
            reference = next(r for r in arms if (r['case'], r['evaluation_paths']) == (case, 4096))
            for n in (128, 256):
                target = next(r for r in arms if (r['case'], r['evaluation_paths']) == (case, n))
                contrast = compare_summaries(target['statistics'], reference['statistics'])
                v_ratio = contrast['estimated_variance_ratio']
                cost_ratio = target['exact_numerical_cpu']/reference['exact_numerical_cpu']
                comparisons.append(dict(case=case, evaluation_paths=n, reference_paths=4096,
                    **contrast, actual_cpu_ratio=cost_ratio,
                    estimated_variance_cpu_ratio=v_ratio*cost_ratio if v_ratio is not None else None))
        comparison_record = dict(comparisons=comparisons, family_size=12,
            critical_value='13/4', finite_sample_coverage_certified=False,
            measured_mean_squared_error=False, numerical_summary_errors_certified=False)
        primary_cpu = sum(r['exact_numerical_cpu'] for r in arms)
        warm_cpu = Fraction(read(output/'warmup.json')['setup']['cpu_seconds'])
        warm_cpu += sum(Fraction(r['training']['cpu_seconds'])+Fraction(r['evaluation']['cpu_seconds']) for r in warm)
        summary_record = dict(all_registered_counts_observed=True, completion_status_source='manifest.json',
            cases=EXPECTED['cases'], arms=9, primary_pairs=3978,
            warmup_pairs=4, numerical_cpu=primary_cpu, warmup_numerical_cpu=warm_cpu,
            total_numerical_cpu=primary_cpu+warm_cpu,
            planning_charge=sum(Fraction(c['planning_charge']) for m in bycase.values() for c in m['candidates']),
            over_budget_arms=sum(r['over_budget'] for r in arms),
            within_forecast_tolerance=sum(r['within_forecast_tolerance'] for r in arms),
            mean_contrast_status_counts=dict(Counter(r['mean_contrast']['normal_interval_status'] for r in comparisons)),
            variance_contrast_status_counts=dict(Counter(r['variance_contrast']['normal_interval_status'] for r in comparisons)),
            prospective_conditional_on_prior=True, original_financial_models_unchanged=True,
            selected_cases_only=True, new_market_observations=0, publication_clearance=False)
        merge({}, evidence); guard()
        save(output/'comparisons.json', comparison_record)
        save(output/'summary.json', summary_record)
    except Exception as exc:
        save(output/'failure.json', dict(complete=False, exception_type=type(exc).__name__, reason=str(exc),
             recorded_at_utc=utc(), partial_rows_are_not_registered_estimators=True))
        raise
    save(output/'manifest.json', dict(complete=True, completed_at_utc=utc(),
        process_cpu_seconds=time.process_time()-cpu, elapsed_seconds=time.monotonic()-wall,
        registration_sha256=sha(registration/'manifest.json'), evidence_sha256=evidence,
        artifact_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))},
        publication_clearance=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    parser.add_argument('--registration', type=Path, default=BASE/'registration_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'register':
        register(args.output)
    else:
        run(args.registration, args.output)


if __name__ == '__main__':
    main()
