"""Audit a completed, frozen main allocation experiment without amending it."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy.stats import t

import study_training_allocation as study
from audit_training_allocation import _compare, numerical_record
from qf_research.optimal_stopping.funded_policy import _digest

ARMS = ('reference', 'selected')


def _normalise(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _summary(means, errors):
    count = len(means)
    variance = float(np.var(means, ddof=1))
    evaluation = float(np.mean(errors**2))
    mean = float(np.mean(means))
    se = math.sqrt(variance/count)
    radius = float(t.ppf(.975, count-1))*se
    return dict(repetitions=count, mean=mean, standard_error=se,
                lower=mean-radius, upper=mean+radius,
                repetition_variance=variance,
                mean_conditional_evaluation_variance=evaluation,
                training_variance_moment_estimate=variance-evaluation,
                evaluation_share_percent=100*evaluation/variance if variance > 0 else None)


def reconstruct_arm(blocks, retained, cfg, allocation, budget):
    """Independently reconstruct inference, retained costs and deletion checks."""
    count, paths = allocation['repetitions'], allocation['evaluation_paths']
    if len(blocks) != count or [b['repetition'] for b in blocks] != list(range(count)):
        raise ValueError('Main repetition coverage differs from the frozen allocation')
    if count < 3:
        raise ValueError('At least three repetitions are required for the influence audit')
    means = np.asarray([b['evaluation']['gain']['estimate']['value'] for b in blocks])
    errors = np.asarray([b['evaluation']['gain']['estimate']['standard_error'] for b in blocks])
    primary = _summary(means, errors)
    estimate = dict(value=primary['mean'], standard_error=primary['standard_error'],
                    lower=primary['lower'], upper=primary['upper'],
                    observations=count, confidence=.95)
    evaluation = primary['mean_conditional_evaluation_variance']
    inference = dict(estimate=estimate,
        repetition_estimate_variance=primary['repetition_variance'],
        mean_conditional_evaluation_variance=evaluation,
        training_variance_moment_estimate=primary['training_variance_moment_estimate'],
        evaluation_variance_of_grand_mean=evaluation/count,
        approximate_intervals=True, evaluation_path_variance=evaluation*paths)
    deletions = [dict(omitted_repetition=index,
                      **_summary(np.delete(means, index), np.delete(errors, index)))
                 for index in range(count)]
    influence = dict(primary=primary, deletions=deletions,
        minimum_deleted_mean=min(row['mean'] for row in deletions),
        maximum_deleted_mean=max(row['mean'] for row in deletions),
        maximum_absolute_mean_change=max(abs(row['mean']-primary['mean']) for row in deletions),
        positive_deleted_intervals=sum(row['lower'] > 0 for row in deletions),
        all_deleted_means_positive=all(row['mean'] > 0 for row in deletions))
    setup = retained['setup']
    cpu = setup['cpu_seconds']+math.fsum(
        b['training']['cpu_seconds']+b['evaluation']['cpu_seconds'] for b in blocks)
    elapsed = setup['elapsed_seconds']+math.fsum(
        b['training']['elapsed_seconds']+b['evaluation']['elapsed_seconds'] for b in blocks)
    ratio = cpu/budget
    expected = dict(inference=inference, standalone_cpu_seconds=cpu,
        standalone_elapsed_seconds=elapsed,
        zero_gain_sample_pairs=sum(b['evaluation']['nonzero_gain_count'] == 0 for b in blocks),
        influence=influence, setup=setup, allocation=allocation,
        actual_to_budget_ratio=ratio,
        material_budget_mismatch=not cfg['budget_ratio_limits'][0] <= ratio <= cfg['budget_ratio_limits'][1])
    _compare(retained, expected, 'main arm')
    return expected


def _timings(record, location, *, training=False):
    for key in ('cpu_seconds', 'elapsed_seconds'):
        if (isinstance(record[key], bool) or not math.isfinite(record[key])
                or record[key] <= 0):
            raise ValueError(f'Invalid workload timing at {location}.{key}')
    if training:
        phases = record['phases']
        for key in ('simulation_seconds', 'simulation_cpu_seconds',
                    'fitting_seconds', 'fitting_cpu_seconds'):
            if not math.isfinite(phases[key]) or phases[key] < 0:
                raise ValueError('Invalid training phase timing')
        for suffix, total in (('_seconds', 'elapsed_seconds'), ('_cpu_seconds', 'cpu_seconds')):
            value = phases['simulation'+suffix]+phases['fitting'+suffix]
            if value > record[total]+1e-9:
                raise ValueError('Training phase costs exceed the full pair cost')


def _stream(cfg, key, paths):
    sequence = np.random.SeedSequence(cfg['root_seed'], spawn_key=key)
    generator = np.random.Generator(np.random.PCG64(sequence))
    return dict(entropy=sequence.entropy, spawn_key=list(sequence.spawn_key),
                pool_size=sequence.pool_size, bit_generator='PCG64', paths=paths,
                stream_id=_digest(generator.bit_generator.state))


def check_block(block, cfg, base, case, arm, repetition, paths, fingerprint, seen):
    """Check every retained pair against its financial problem and seed role."""
    if (block['case'], block['arm'], block['repetition']) != (case, arm, repetition):
        raise ValueError('Main block identity differs from its filename')
    training, evaluation = block['training'], block['evaluation']
    if training['training_paths'] != cfg['training_paths'] or evaluation['training_paths'] != cfg['training_paths']:
        raise ValueError('Main training path counts differ')
    if evaluation['evaluation_paths'] != paths:
        raise ValueError('Main evaluation path count differs from the frozen allocation')
    if any(len(training[key]) != 2 for key in ('policies', 'policy_fingerprints', 'streams')):
        raise ValueError('Require exactly two retained training policies')
    arm_id = ARMS.index(arm)
    base_record = _normalise(asdict(base))
    for index, count in enumerate(cfg['training_paths']):
        expected_stream = _stream(cfg, (20, case, arm_id, repetition, index), count)
        if training['streams'][index] != expected_stream:
            raise ValueError('Training seed provenance differs from its registered role')
        policy = training['policies'][index]
        for name in ('model', 'option', 'decisions'):
            if policy[name] != base_record[name]:
                raise ValueError('Retained policy differs from its registered financial problem')
        for name, value in (('training_paths', count), ('requested_degree', cfg['degree']),
                            ('minimum_itm_paths', cfg['minimum_itm_paths']),
                            ('condition_limit', cfg['condition_limit']),
                            ('training_stream_id', expected_stream['stream_id'])):
            if policy[name] != value:
                raise ValueError('Retained policy training specification differs')
        if (_digest(policy) != training['policy_fingerprints'][index]
                or len(policy['rules']) != len(base.decisions)
                or policy['rules'][-1]['status'] != 'terminal'):
            raise ValueError('Retained policy fingerprint or decision rules differ')
    if evaluation['policy_fingerprints'] != training['policy_fingerprints']:
        raise ValueError('Main evaluation used different fitted policies')
    expected_evaluation = _stream(cfg, (21, case, arm_id, repetition), paths)
    if evaluation['stream'] != expected_evaluation:
        raise ValueError('Evaluation seed provenance differs from its registered role')
    if evaluation['approximation_fingerprint'] != fingerprint:
        raise ValueError('Main evaluation approximation differs from the pilot construction')
    study.check_streams(training, evaluation, seen)
    study.check_pair(evaluation['gain'], paths)
    if evaluation['gain']['estimate']['confidence'] != .95:
        raise ValueError('Main paired confidence level differs')
    nonzero = evaluation['nonzero_gain_count']
    if type(nonzero) is not int or not 0 <= nonzero <= paths:
        raise ValueError('Invalid nonzero gain count')
    if nonzero == 0 and any(evaluation['gain']['estimate'][key] != 0.
                           for key in ('value', 'standard_error')):
        raise ValueError('An all-zero gain sample has a nonzero estimate')
    gaps = evaluation['minimum_gap_quote']
    if len(gaps) != 2 or any(not math.isfinite(x) or x < 0 for x in gaps):
        raise ValueError('Invalid minimum pathwise gaps')
    path_hash = evaluation['paths_sha256']
    if not isinstance(path_hash, str) or len(path_hash) != 64 or any(c not in '0123456789abcdef' for c in path_hash):
        raise ValueError('Invalid evaluation path evidence hash')
    _timings(training, 'training', training=True)
    _timings(evaluation, 'evaluation')


def verify(run):
    """Verify complete main evidence and reconstruct all retained summaries."""
    run = study.private(Path(run))
    manifest = study.checked(run)
    registration = study.private(study.ROOT/manifest['registration'])
    registered, cfg, tasks = study.stage_registration(registration)
    launch = study.read(run/'launch.json')
    if (manifest.get('complete') is not True or launch.get('stage') != 'main'
            or registered.get('stage') != 'main_allocation'):
        raise ValueError('A completed main execution and frozen allocation are required')
    registration_hash = study.sha(registration/'manifest.json')
    if (manifest['registration_sha256'] != registration_hash
            or launch['registration_sha256'] != registration_hash
            or launch['registration'] != manifest['registration']):
        raise ValueError('Main registration identity differs')
    for key in ('input_sha256', 'source_sha256'):
        if manifest[key] != registered[key]:
            raise ValueError('Main provenance mapping differs from its registration')
    if (launch['python_workers'] != 1
            or launch['thread_environment'] != {key: '1' for key in study.THREADS}):
        raise ValueError('Recorded main execution environment differs')
    pilot = study.private(study.ROOT/registered['pilot'])
    _, pilot_cfg, pilot_tasks, pilot_summaries, prior_streams = study.verify_pilot(pilot)
    allocation_record = study.read(registration/'allocations.json')
    if (cfg != pilot_cfg or tasks != pilot_tasks
            or allocation_record['models'] != pilot_summaries
            or allocation_record['forbidden_streams'] != sorted(prior_streams)
            or allocation_record['main_outcomes_inspected'] is not False):
        raise ValueError('Frozen main allocations differ from verified pilot evidence')
    allocations = {row['case']: row['allocation'] for row in pilot_summaries}
    expected_blocks = {
        f'case_{task["case"]:03d}_{arm}_rep_{rep:04d}.json'
        for task in tasks for arm in ARMS
        for rep in range(allocations[task['case']][arm]['repetitions'])}
    if {path.name for path in (run/'blocks').glob('*.json')} != expected_blocks:
        raise ValueError('Main block-file coverage differs from frozen counts')
    expected_files = {'launch.json', 'warmup.json', 'summary.json'} | {
        f'model_{task["case"]:03d}.json' for task in tasks} | {
        'blocks/'+name for name in expected_blocks}
    actual_files = {str(path.relative_to(run)) for path in run.rglob('*.json')
                    if path != run/'manifest.json'}
    if actual_files != expected_files or set(manifest['artifact_sha256']) != expected_files:
        raise ValueError('Main manifest or model-file coverage differs')
    summary = study.read(run/'summary.json')
    if ([row['case'] for row in summary['models']] != [task['case'] for task in tasks]
            or summary['publication_clearance'] is not False
            or summary['independent_market_validation'] is not False):
        raise ValueError('Main summary coverage or interpretation flags differ')
    seen = set(prior_streams)
    reconstructed = []
    for task, model in zip(tasks, summary['models']):
        case = task['case']
        if study.read(run/f'model_{case:03d}.json') != model or set(model['arms']) != set(ARMS):
            raise ValueError('Main model summary differs from the full summary')
        base = study.load_frozen(task['source_policy'])[0]
        pilot_row = study.read(pilot/'blocks'/f'case_{case:03d}.json')
        expected_fingerprint = pilot_row['setup'][0]['approximation_fingerprint']
        arms = {}
        for arm in ARMS:
            retained = model['arms'][arm]
            setup = retained['setup']
            if (setup['settings'] != cfg['approximation']
                    or setup['approximation_fingerprint'] != expected_fingerprint):
                raise ValueError('Main model setup differs from the frozen approximation')
            _timings(setup, 'setup')
            allocation = allocations[case][arm]
            blocks = []
            for repetition in range(allocation['repetitions']):
                block = study.read(run/'blocks'/f'case_{case:03d}_{arm}_rep_{repetition:04d}.json')
                check_block(block, cfg, base, case, arm, repetition,
                            allocation['evaluation_paths'], expected_fingerprint, seen)
                blocks.append(block)
            arms[arm] = reconstruct_arm(blocks, retained, cfg, allocation,
                                       allocations[case]['budget']['cpu_seconds'])
        left, right = arms['reference'], arms['selected']
        vl = left['inference']['estimate']['standard_error']**2
        vr = right['inference']['estimate']['standard_error']**2
        comparison = dict(estimated_variance_ratio=vr/vl if vl > 0 else None,
            actual_cpu_ratio=right['standalone_cpu_seconds']/left['standalone_cpu_seconds'],
            estimated_variance_cpu_ratio=(vr*right['standalone_cpu_seconds'] /
                (vl*left['standalone_cpu_seconds']) if vl > 0 else None),
            measured_mean_squared_error=False, validated_interval_coverage=False)
        _compare(model['comparison'], comparison, f'case {case}.comparison')
        reconstructed.append(dict(case=case, arms=arms, comparison=comparison))
    return manifest, cfg, tasks, dict(models=reconstructed, blocks=len(expected_blocks),
        independent_training_streams=2*len(expected_blocks),
        independent_evaluation_streams=len(expected_blocks),
        excluded_prior_and_pilot_streams=len(prior_streams))


def _link_or_copy(source, target):
    try:
        return os.link(source, target)
    except OSError:
        return shutil.copy2(source, target)


def tamper_check(run, output):
    """Reject one altered copied block before any audit output is created."""
    source = sorted((run/'blocks').glob('*.json'))[0]
    original_hash = study.sha(source)
    with tempfile.TemporaryDirectory(prefix='alteration_check_', dir=output) as temporary:
        temporary = Path(temporary)
        copied = temporary/'main'
        shutil.copytree(run, copied, copy_function=_link_or_copy)
        target = copied/'blocks'/source.name
        altered = study.read(target)
        estimate = altered['evaluation']['gain']['estimate']
        estimate['value'] = 1. if estimate['value'] == 0. else 0.
        # Break the copied hard link before writing, preserving original evidence.
        target.unlink()
        study.save(target, altered)
        altered_hash = study.sha(target)
        destination = temporary/'must_not_exist'
        try:
            audit(copied, destination)
        except ValueError as error:
            rejection = str(error)
            if 'checksum' not in rejection.lower():
                raise ValueError('Altered main evidence failed for an unexpected reason') from error
        else:
            raise ValueError('Altered main evidence was accepted')
        if destination.exists():
            raise ValueError('Rejected main evidence created an audit output directory')
    if study.sha(source) != original_hash:
        raise ValueError('The alteration check changed original main evidence')
    return dict(artifact=str(source.relative_to(run)), original_sha256=original_hash,
        altered_sha256=altered_hash, rejected=True, rejection=rejection,
        audit_output_created=False, temporary_copy_removed=True)


def audit(run, output):
    """Verify the main run, reproduce eight fixed pairs and retain audit evidence."""
    run, output = study.private(Path(run)), study.private(Path(output))
    if output.exists():
        raise FileExistsError('Use a fresh main audit output directory')
    if output.is_relative_to(run):
        raise ValueError('Keep audit output separate from retained main evidence')
    manifest, cfg, tasks, checks = verify(run)
    if any(os.environ.get(key) != '1' for key in study.THREADS):
        raise ValueError('Use the registered numerical-library thread environment')
    run_hash = study.sha(run/'manifest.json')
    source_path = Path(__file__).resolve()
    source_root = source_path.parents[1]
    test_path = source_root/'tests/test_training_allocation_main_audit.py'
    auditor_hash, test_hash = study.sha(source_path), study.sha(test_path)
    additional_sources = {str(source_path.relative_to(source_root)): auditor_hash,
                          str(test_path.relative_to(source_root)): test_hash}
    audit_sources = dict(manifest['source_sha256'])
    for name in ('scripts/study_training_allocation.py', 'scripts/audit_training_allocation.py',
                 'scripts/audit_training_evaluation.py', 'scripts/validate_funded_dual.py',
                 'scripts/run_conditional_calibration.py', 'scripts/review_training_inference.py',
                 'src/qf_research/optimal_stopping/allocation_workload.py',
                 'src/qf_research/optimal_stopping/funded_policy.py'):
        audit_sources[name] = study.sha(source_root/name)
    audit_sources.update(additional_sources)
    output.mkdir(parents=True)
    models = {row['case']: row for row in checks['models']}
    repeated = []
    for case, task in {task['case']: task for task in (tasks[0], tasks[-1])}.items():
        base = study.load_frozen(task['source_policy'])[0]
        for arm_id, arm in enumerate(ARMS):
            retained = models[case]['arms'][arm]
            allocation = retained['allocation']
            approximation, setup = study.build_approximation(base, **cfg['approximation'])
            if numerical_record(setup) != numerical_record(retained['setup']):
                raise ValueError('Reproduced main value approximation differs')
            for repetition in sorted({0, allocation['repetitions']-1}):
                policies, training = study.fit(base, cfg, 20, case, repetition, arm_id)
                evaluation = study.evaluate_pair(policies, approximation, allocation['evaluation_paths'],
                                                  study.seed(cfg, 21, case, arm_id, repetition))
                record = dict(case=case, arm=arm, repetition=repetition,
                              training=training, evaluation=evaluation)
                original = study.read(run/'blocks'/f'case_{case:03d}_{arm}_rep_{repetition:04d}.json')
                if numerical_record(record) != numerical_record(original):
                    raise ValueError(f'Numerical reproduction differs for case {case}, {arm}, repetition {repetition}')
                filename = f'repeated_case_{case:03d}_{arm}_rep_{repetition:04d}.json'
                study.save(output/filename, dict(setup=setup, **record))
                repeated.append(dict(case=case, arm=arm, repetition=repetition,
                                     artifact=filename, exact_numerical_match=True))
    alteration = tamper_check(run, output)
    study.checked(run)
    if (study.sha(run/'manifest.json') != run_hash or study.sha(source_path) != auditor_hash
            or study.sha(test_path) != test_hash):
        raise ValueError('Main evidence or the new auditor changed during verification')
    for name, expected in audit_sources.items():
        if study.sha(source_root/name) != expected:
            raise ValueError('The separate audit source closure changed during verification')
    verification_plan = dict(
        selection='First and last repetition of both arms for the first and last retained models',
        selection_fixed_for_verification=True, part_of_original_experiment_registration=False,
        new_auditor_prepared_during_main_execution=True,
        main_outcomes_inspected_when_selection_fixed=False, estimation_protocol_changed=False)
    result = dict(verified_at=datetime.now(timezone.utc).isoformat(),
        main_run=str(run.relative_to(study.ROOT)), main_manifest_sha256=run_hash,
        verification_plan=verification_plan, checks=checks,
        audit_environment=dict(python=study.platform.python_version(), numpy=np.__version__,
            scipy=study.scipy.__version__, thread_environment={key: os.environ[key] for key in study.THREADS},
            python_workers=1, effective_native_thread_count=None),
        repeated_pairs=repeated, alteration_check=alteration, full_numerical_rerun=False,
        timings_expected_to_match=False, costs_reconstructed_from_retained_measurements=True,
        approximate_intervals=True, validated_interval_coverage=False, publication_clearance=False)
    study.save(output/'verification.json', result)
    artifacts = sorted(output.glob('*.json'))
    inputs = dict(manifest['input_sha256'])
    inputs[str((run/'manifest.json').relative_to(study.ROOT))] = run_hash
    study.save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, registered_source_sha256=manifest['source_sha256'],
        additional_audit_source_sha256=additional_sources, audit_source_sha256=audit_sources,
        artifact_sha256={path.name: study.sha(path) for path in artifacts},
        registration_unchanged=True, complete=True, publication_clearance=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.run, arguments.output)
    print(f'Checked {result["checks"]["blocks"]} main pairs and reproduced '
          f'{len(result["repeated_pairs"])} fixed pairs; altered evidence was rejected.')
