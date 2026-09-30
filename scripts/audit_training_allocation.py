"""Independently reconstruct and reproduce a private allocation pilot."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import study_training_allocation as study


def numerical_record(record):
    """Normalise JSON containers and exclude observed numerical-workload times."""
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if key != 'phases' and not key.endswith('_seconds')}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value
    return strip(json.loads(json.dumps(record, allow_nan=False)))


def _compare(actual, expected, location='record'):
    """Require matching structure and numerical reconstruction to round-off."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'Independent reconstruction keys differ at {location}')
        for key in expected:
            _compare(actual[key], expected[key], f'{location}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'Independent reconstruction length differs at {location}')
        for index, item in enumerate(expected):
            _compare(actual[index], item, f'{location}[{index}]')
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-14)):
            raise ValueError(f'Independent reconstruction differs at {location}')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f'Independent reconstruction differs at {location}')


def reconstruct_model(row, cfg, retained):
    """Rebuild variance components and enumerate every affordable integer R.

    This calculation does not call the allocation selector, timing selector or
    repetition-summary routine. The two 32-ULP allowances are the registered
    rule for floating-point comparison of variable and total CPU costs.
    """
    estimates = [record['evaluation']['gain']['estimate'] for record in row['statistical']]
    means = np.asarray([record['value'] for record in estimates], dtype=float)
    errors = np.asarray([record['standard_error'] for record in estimates], dtype=float)
    observed = float(np.var(means, ddof=1))
    evaluation = float(np.mean(errors**2))
    A, B = observed-evaluation, evaluation*cfg['pilot_paths']
    inference = retained['pilot_inference']
    for name, value in (
            ('repetition_estimate_variance', observed),
            ('mean_conditional_evaluation_variance', evaluation),
            ('training_variance_moment_estimate', A),
            ('evaluation_variance_of_grand_mean', evaluation/len(means)),
            ('evaluation_path_variance', B)):
        _compare(inference[name], value, f'case {row["case"]}.inference.{name}')
    _compare(inference['estimate']['value'], float(np.mean(means)), 'repetition mean')
    _compare(inference['estimate']['standard_error'], math.sqrt(observed/len(means)),
             'repetition standard error')

    timing = {(record['repetition'], record['evaluation']['evaluation_paths']): record
              for record in row['timing']}
    costs = {n: float(np.median([
        row['statistical'][r]['training']['cpu_seconds']+timing[r, n]['evaluation']['cpu_seconds']
        for r in range(cfg['timing_repetitions'])])) for n in cfg['evaluation_grid']}
    setup = float(np.median([record['cpu_seconds'] for record in row['setup']]))
    usable = cfg['reference_repetitions']*costs[cfg['reference_paths']]
    budget = setup+usable
    variable_tolerance, total_tolerance = 32*math.ulp(usable), 32*math.ulp(budget)
    resolved = A > 0 and B > 0

    def affordable(repetitions, cost):
        return (repetitions*cost-usable <= variable_tolerance
                and setup+repetitions*cost-budget <= total_tolerance)

    def candidate(paths, repetitions, capped):
        variable = repetitions*costs[paths]
        total = setup+variable
        feasible = (repetitions >= cfg['minimum_repetitions']
                    and affordable(repetitions, costs[paths]))
        return dict(evaluation_paths=paths, repetitions=repetitions,
                    pair_cpu_seconds=costs[paths], total_cpu_seconds=total,
                    unused_cpu_seconds=max(0., usable-variable),
                    budget_excess_cpu_seconds=max(0., total-budget),
                    predicted_variance=(A/repetitions+B/repetitions/paths
                                        if resolved and feasible else None),
                    feasible=feasible, capped=capped)

    candidates = []
    for paths in sorted(costs):
        # Exhaustive enumeration is independent of the selector's binary search.
        allowed = [r for r in range(cfg['maximum_repetitions']+2)
                   if affordable(r, costs[paths])]
        largest = max(allowed)
        candidates.append(candidate(paths, min(largest, cfg['maximum_repetitions']),
                                    largest > cfg['maximum_repetitions']))
    reference = candidate(cfg['reference_paths'], cfg['reference_repetitions'], False)
    selected = (min((item for item in candidates if item['feasible']),
                    key=lambda item: item['predicted_variance']) if resolved else reference)
    allocation = dict(status='selected' if resolved else 'unresolved_variance',
        raw_variance_components=dict(training=A, evaluation=B),
        budget=dict(cpu_seconds=budget, setup_cpu_seconds=setup, usable_cpu_seconds=usable,
                    tolerance_cpu_seconds=total_tolerance,
                    variable_tolerance_cpu_seconds=variable_tolerance),
        reference=reference, selected=selected, candidates=candidates)
    _compare(retained['allocation'], allocation, f'case {row["case"]}.allocation')
    return dict(case=row['case'], repetition_estimate_variance=observed,
                mean_conditional_evaluation_variance=evaluation,
                independently_reconstructed_allocation=allocation)


def verify(pilot):
    """Check retained evidence and independently reconstruct every model."""
    pilot = study.private(Path(pilot))
    manifest, cfg, tasks, summaries, _ = study.verify_pilot(pilot)
    by_case = {summary['case']: summary for summary in summaries}
    reconstructed = [reconstruct_model(
        study.read(pilot/'blocks'/f'case_{task["case"]:03d}.json'), cfg, by_case[task['case']])
        for task in tasks]
    return manifest, cfg, tasks, reconstructed


def tamper_check(pilot, output):
    """Alter a temporary copy and require rejection before allocation output."""
    pilot, output = study.private(Path(pilot)), study.private(Path(output))
    source = sorted((pilot/'blocks').glob('case_*.json'))[0]
    original_hash = study.sha(source)
    with tempfile.TemporaryDirectory(prefix='alteration_check_', dir=output) as temporary:
        temporary = Path(temporary)
        copied = temporary/'pilot'
        shutil.copytree(pilot, copied)
        target = copied/'blocks'/source.name
        altered = study.read(target)
        altered['statistical'][0]['evaluation']['gain']['estimate']['value'] += 1.
        study.save(target, altered)
        altered_hash = study.sha(target)
        allocation_output = temporary/'must_not_exist'
        try:
            study.allocate(copied, allocation_output)
        except ValueError as error:
            message = str(error)
            if 'checksum' not in message.lower():
                raise ValueError('Altered evidence failed for an unexpected reason') from error
        else:
            raise ValueError('Altered pilot evidence was accepted')
        if allocation_output.exists():
            raise ValueError('Rejected evidence created an allocation directory')
    if study.sha(source) != original_hash:
        raise ValueError('Alteration check changed the original pilot evidence')
    return dict(artifact=str(source.relative_to(pilot)), original_sha256=original_hash,
                altered_sha256=altered_hash, rejected=True, rejection=message,
                allocation_output_created=False, temporary_copy_removed=True)


def audit(pilot, output):
    """Retain reconstruction, four boundary-pair reruns and a tamper check."""
    pilot, output = study.private(Path(pilot)), study.private(Path(output))
    if output.exists():
        raise FileExistsError('Use a fresh audit output directory')
    manifest, cfg, tasks, reconstructed = verify(pilot)
    if any(os.environ.get(key) != '1' for key in study.THREADS):
        raise ValueError('Use the registered numerical-library thread environment')
    input_hash = study.sha(pilot/'manifest.json')
    output.mkdir(parents=True)
    repeated = []
    selected_tasks = {task['case']: task for task in (tasks[0], tasks[-1])}
    repetitions = sorted({0, cfg['pilot_repetitions']-1})
    for case, task in selected_tasks.items():
        base = study.load_frozen(task['source_policy'])[0]
        approximation, setup = study.build_approximation(base, **cfg['approximation'])
        original = study.read(pilot/'blocks'/f'case_{case:03d}.json')
        if numerical_record(setup) != numerical_record(original['setup'][0]):
            raise ValueError('Reproduced value approximation differs')
        for repetition in repetitions:
            policies, training = study.fit(base, cfg, 10, case, repetition)
            evaluation = study.evaluate_pair(policies, approximation, cfg['pilot_paths'],
                                             study.seed(cfg, 11, case, repetition))
            record = dict(repetition=repetition, training=training, evaluation=evaluation)
            if numerical_record(record) != numerical_record(original['statistical'][repetition]):
                raise ValueError(f'Numerical reproduction differs for case {case}, repetition {repetition}')
            filename = f'repeated_case_{case:03d}_rep_{repetition:03d}.json'
            study.save(output/filename, dict(case=case, setup=setup, **record))
            repeated.append(dict(case=case, repetition=repetition, artifact=filename,
                                 exact_numerical_match=True))
    alteration = tamper_check(pilot, output)
    # Recheck retained evidence after reproduction and the temporary-copy probe.
    study.verify_pilot(pilot)
    if study.sha(pilot/'manifest.json') != input_hash:
        raise ValueError('Pilot manifest changed during audit')
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(),
                  models=reconstructed, repeated_pairs=repeated, alteration_check=alteration,
                  full_numerical_rerun=False, timings_expected_to_match=False,
                  main_outcomes_generated=False, publication_clearance=False)
    study.save(output/'verification.json', result)
    files = sorted(output.glob('*.json'))
    inputs = dict(manifest['input_sha256'])
    inputs[str((pilot/'manifest.json').relative_to(study.ROOT))] = input_hash
    study.save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=manifest['source_sha256'],
        auditor_sha256=study.sha(Path(__file__).resolve()),
        artifact_sha256={path.name: study.sha(path) for path in files},
        complete=True, publication_clearance=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.pilot, arguments.output)
    print(f'Independently checked {len(result["models"])} allocations and reproduced '
          f'{len(result["repeated_pairs"])} pilot pairs; altered evidence was rejected.')
