"""Evaluate eight retained fits on a separately declared larger outer stream.

No training or value-fitting stream is reconstructed. The selected degree-three
exact-terminal polynomial hybrid and both hinge assessors use the same fresh
outer states. A prior local design binds the completed preceding study, frozen
coefficients, counts, sources and near-balanced arm positions before simulation.
The standalone budget combines retained setup measurements with fresh outer
preparation and assessment. Fresh expenditure is also reported separately.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import study_replicated_gap_refinement as original
from certify_multivariate_gap import check_inputs, load_fraction
from study_multivariate_gap_refinement import frozen_model_hash
from certify_max_hinge import interval_record

ARMS = ('cold_uniform', 'adaptive', 'hybrid_exact_terminal')
OUTER_PATHS = 1024
ADDITIONAL_SOURCES = ('scripts/study_frozen_gap_extension.py', 'tests/test_frozen_gap_extension.py')
ORIGINAL_FILES = ('design.json', 'prepared.json', 'preparation_completion.json',
                  'summary.json', 'assessment_completion.json')
ORIGINAL_FIT_FILES = ('inputs.json', 'assessment_attempt.json', 'result.json',
    'assessment_completion.json', 'polynomial_attempt.json', 'polynomial.json',
    'polynomial_completion.json', 'verified_polynomial_attempt.json',
    'verified_polynomial.json', 'verified_polynomial_completion.json')


def clocks():
    return time.process_time(), time.perf_counter()


def elapsed(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def digest(source):
    return hashlib.sha256(source).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def private_directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Retain states, fit exports and detailed certificates privately')
    return directory


def source_hashes():
    sources = original.source_hashes()
    sources.update({name: digest((ROOT/name).read_bytes()) for name in ADDITIONAL_SOURCES})
    return sources


def all_original_files(source):
    return (*ORIGINAL_FILES, *(f'fit_{i:02d}/{name}' for i in range(8) for name in ORIGINAL_FIT_FILES))


def verify_original(source, *, numerical_replay=False):
    """Validate completed records only; the guards prohibit original reruns."""
    source = private_directory(source)
    names = all_original_files(source)
    if any(not (source/name).is_file() for name in names):
        raise ValueError('Every original design, attempt, result and completion must already exist')
    if list(source.rglob('*failure.json')):
        raise ValueError('A failed original attempt cannot be used as a completed source')
    design, manifest, completion, records, sources, design_sha = original.registration(source)
    # Both functions have complete-result branches. All prerequisites above
    # exist before either is called, so neither may start a numerical study.
    original.assessment(source)
    verified = original.certify_polynomial(source)
    if numerical_replay:
        original.polynomial(source)  # Frozen evaluation only; no fitting in its complete branch.
    polynomials = [json.loads((source/f'fit_{i:02d}/polynomial.json').read_bytes()) for i in range(8)]
    inventory = {name: digest((source/name).read_bytes()) for name in names}
    return dict(source_directory=str(source), files=inventory,
                original_source_sha256=sources, original_design_sha256=design_sha), records, polynomials


def fit_designs():
    # Six permutations followed by two declared rotations give position
    # counts two or three per arm and a 4/4 uniform/adaptive relative order.
    orders = list(permutations(ARMS))+[ARMS, ('adaptive', 'hybrid_exact_terminal', 'cold_uniform')]
    return [dict(index=i, root_seed=seed, directory=f'fit_{i:02d}', arm_order=list(orders[i]))
            for i, seed in enumerate(original.SEEDS)]


def design_record(prior, sources):
    return dict(schema_version=1, author='Amanjeet Singh',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), recorded_before_new_outer_sampling=True,
        source_study=prior, source_sha256=sources, fitting_units=8, fits=fit_designs(),
        outer_paths_per_fit=OUTER_PATHS, precision_bits=96, maximum_path_gap_width='1/100',
        adaptive_coarse_width='1', new_outer_spawn_key=[3], old_outer_spawn_key=[2],
        sample_relation='The preceding and new streams are independent outer samples, not nested prefixes. Datewise simulation means that changing the path count does not preserve a common prefix.',
        fitting_spawn_keys=[[0], [1]], training_or_value_streams_rerun=False,
        selected_comparator='Degree-three polynomial continuation with the exact terminal max-call, selected from the completed preceding comparison before drawing this new outer sample.',
        preserved_objects='Original model, exercise policy, hinge functions, and degree-three polynomial coefficients, without refitting or coefficient alteration.',
        order_scope='All six permutations followed by two declared rotations. Each arm occupies every position two or three times; uniform precedes adaptive four times and follows it four times. Position balance is near-exact, not exact.',
        statistical_unit='Eight retained independently fitted policy/value-function pairs. The 8,192 new paths are conditional evaluation observations, not 8,192 training replications.',
        budget_scope='Fresh outer preparation and complete assessment are measured anew. Reconstructed standalone charges add retained setup phase measurements, exclude the preceding outer preparation and every preceding assessment, and do not represent a new fitting execution.',
        sampling_scope='Numerical intervals concern ideal gap arithmetic at supplied binary states. Midpoint standard errors, simulation law and population coverage are not certified.',
        no_new_priority_claim=True, public_preregistration_claim=False)


def validate_design(design, prior, sources):
    expected = design_record(prior, sources)
    for key, value in expected.items():
        if key != 'recorded_at_utc' and design.get(key) != value:
            raise ValueError('The frozen extension design differs: '+key)
    if not isinstance(design.get('recorded_at_utc'), str):
        raise ValueError('The actual prior recording time must be retained')


def fresh_outer(record, *, count=OUTER_PATHS):
    """Replay only a new independent evaluation stream, never a fitting stream."""
    import numpy as np
    from qf_research.optimal_stopping.multivariate_benchmark import (
        MaxCallModel, MaxCallPolicy, MaxCallRule, evaluate_max_call, simulate_max_call)
    if type(count) is not int or count < 2:
        raise ValueError('At least two fresh evaluation paths are required')
    model = MaxCallModel(**record['model'])
    rules = tuple(MaxCallRule(tuple(coefficients), 0, 0, None) for coefficients in record['frozen_policy'])
    policy = MaxCallPolicy(model, record['design']['policy_degree'], rules,
        'retained-policy-training-'+str(record['design']['root_seed']), record['design']['training_paths'])
    stream = np.random.SeedSequence(record['design']['root_seed']).spawn(4)[3]
    paths = simulate_max_call(model, count, np.random.default_rng(stream))
    exercise = evaluate_max_call(policy, paths)
    return paths.states.tolist(), exercise.stopping_indices.tolist(), list(stream.spawn_key)


def frozen_polynomial_identity(polynomial):
    row = polynomial['degrees']['3']
    return canonical_hash(dict(basis_exponents=row['basis_exponents'], coefficients=row['coefficients']))


def setup_charges(old, polynomial, outer_cost):
    """Replace old outer work; do not charge unused fitting or prior assessment."""
    original.validate_cost(outer_cost)
    phases = old['preparation_phases']
    old_outer = phases['outer_evaluation_and_record_construction']['cpu_seconds']
    new_outer = outer_cost['cpu_seconds']
    hinge = old['preparation_cost']['cpu_seconds']-old_outer+new_outer
    poly = polynomial['degrees']['3']
    hybrid = (phases['policy_training']['cpu_seconds']+
        polynomial['shared_value_stream_and_target_cpu_seconds']+
        poly['polynomial_interior_fit_cpu_seconds']+new_outer)
    original.equal(poly['hybrid_setup_pricing_cpu_seconds'], hybrid-new_outer+old_outer,
                   'The retained hybrid setup is not the sum of its exclusive phases')
    if min(hinge, hybrid) < 0:
        raise ValueError('Reconstructed setup charges must be non-negative')
    return dict(cold_uniform=hinge, adaptive=hinge, hybrid_exact_terminal=hybrid)


def adapted_polynomial(polynomial, record, charges, original_sha):
    row = polynomial['degrees']['3']
    return dict(author='Amanjeet Singh', root_seed=record['design']['root_seed'],
        sample_count=len(record['states']), value_fitting_paths=record['design']['value_paths'],
        value_spawn_key=[1], original_export_sha256=original_sha,
        original_outer_sample_count=polynomial['sample_count'],
        frozen_degree_three_polynomial_sha256=frozen_polynomial_identity(polynomial),
        metadata_scope='A coefficient-preserving view for new outer evaluation; no preceding path summary is imported.',
        degrees={'3':dict(basis_exponents=deepcopy(row['basis_exponents']),
            coefficients=deepcopy(row['coefficients']),
            hybrid_setup_pricing_cpu_seconds=charges['hybrid_exact_terminal'])})


def extended_input(old, states, stops, spawn_key, outer_cost, fit, design_sha, sources, old_input_sha):
    record = deepcopy(old)
    phases = deepcopy(old['preparation_phases'])
    phases['outer_evaluation_and_record_construction'] = outer_cost
    record.update(states=states, stopping_indices=stops,
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        extension_design_sha256=design_sha, extension_source_sha256=sources,
        extension_fit_index=fit['index'], original_input_sha256=old_input_sha,
        new_outer_preparation_cost=outer_cost, original_preparation_phases=old['preparation_phases'],
        original_preparation_cost=old['preparation_cost'], preparation_phases=phases,
        preparation_cost={key:sum(row[key] for row in phases.values()) for key in ('cpu_seconds', 'wall_seconds')},
        preparation_cost_scope='Retained original setup phase measurements with the old outer phase replaced by fresh measured outer simulation, frozen-policy evaluation and record construction. No fitting rerun.',
        frozen_model_policy_values_sha256=frozen_model_hash(old))
    record['design'].update(outer_paths=len(states), spawn_keys=[[0], [1], spawn_key],
        original_outer_spawn_key=[2], new_outer_spawn_key=spawn_key,
        evaluation_independent_of_original_outer_stream=True, fitting_streams_not_rerun=True)
    return record


def validate_input(record, old, fit, design_sha, sources, old_input_sha):
    check_inputs(record)
    expected = dict(extension_design_sha256=design_sha, extension_source_sha256=sources,
        extension_fit_index=fit['index'], original_input_sha256=old_input_sha,
        original_preparation_phases=old['preparation_phases'], original_preparation_cost=old['preparation_cost'])
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('New inputs change their design, retained source or fit identity')
    if frozen_model_hash(record) != frozen_model_hash(old):
        raise ValueError('The retained policy or hinge functions changed')
    if record['frozen_model_policy_values_sha256'] != frozen_model_hash(old):
        raise ValueError('The frozen financial identity differs')
    declared = record['design']
    required = dict(root_seed=fit['root_seed'], outer_paths=OUTER_PATHS,
        training_paths=50000, value_paths=50000, policy_degree=old['design']['policy_degree'],
        spawn_keys=[[0], [1], [3]],
        new_outer_spawn_key=[3], original_outer_spawn_key=[2],
        evaluation_independent_of_both_fitting_streams=True,
        evaluation_independent_of_original_outer_stream=True, fitting_streams_not_rerun=True)
    if any(declared.get(key) != value for key, value in required.items()):
        raise ValueError('The follow-up changes its declared counts or independent stream')
    original.validate_cost(record['new_outer_preparation_cost'])
    if set(record['preparation_phases']) != set(old['preparation_phases']):
        raise ValueError('Every exclusive preparation phase must be retained')
    for name, row in record['preparation_phases'].items():
        if name == 'outer_evaluation_and_record_construction':
            if row != record['new_outer_preparation_cost']:
                raise ValueError('The fresh outer preparation measurement changed')
        elif row != old['preparation_phases'].get(name):
            raise ValueError('A retained preparation phase changed')
    for key in ('cpu_seconds', 'wall_seconds'):
        original.equal(record['preparation_cost'][key], sum(row[key] for row in record['preparation_phases'].values()),
                       'The charged hinge setup phases do not sum')
    return True


def preparation(source, directory):
    source, directory = private_directory(source), private_directory(directory)
    if directory.exists():
        raise ValueError('Preparation requires a fresh version; preserve every earlier attempt')
    start = clocks()
    prior, old_records, polynomials = verify_original(source, numerical_replay=True)
    sources = source_hashes()
    design = design_record(prior, sources)
    directory.mkdir(parents=True)
    write_new(directory/'design.json', design)
    design_sha = digest((directory/'design.json').read_bytes())
    retained = []
    for fit, old, polynomial in zip(design['fits'], old_records, polynomials):
        folder = directory/fit['directory']
        phase = 'directory_creation'
        fit_start = clocks()
        try:
            folder.mkdir()
            phase = 'fresh_outer_simulation_and_stopping'
            outer_start = clocks()
            states, stops, key = fresh_outer(old)
            # Input construction is part of preparation, while validation and
            # file I/O below are reported only in the wider workflow timer.
            preliminary_cost = elapsed(outer_start)
            record = extended_input(old, states, stops, key, preliminary_cost, fit,
                design_sha, sources, prior['files'][fit['directory']+'/inputs.json'])
            outer_cost = elapsed(outer_start)
            record['new_outer_preparation_cost'] = outer_cost
            record['preparation_phases']['outer_evaluation_and_record_construction'] = outer_cost
            record['preparation_cost'] = {name:sum(row[name] for row in record['preparation_phases'].values())
                                          for name in ('cpu_seconds', 'wall_seconds')}
            phase = 'input_and_export_validation'
            validate_input(record, old, fit, design_sha, sources,
                           prior['files'][fit['directory']+'/inputs.json'])
            charges = setup_charges(old, polynomial, outer_cost)
            view = adapted_polynomial(polynomial, record, charges,
                prior['files'][fit['directory']+'/polynomial.json'])
            import certify_polynomial_dual as hybrid
            hybrid.validate_export(record, view, (3,))
            phase = 'input_and_export_writing'
            write_new(folder/'inputs.json', record)
            write_new(folder/'polynomial_view.json', view)
            retained.append(dict(index=fit['index'], root_seed=fit['root_seed'],
                input_sha256=digest((folder/'inputs.json').read_bytes()),
                polynomial_view_sha256=digest((folder/'polynomial_view.json').read_bytes()),
                frozen_model_policy_values_sha256=frozen_model_hash(old),
                frozen_degree_three_polynomial_sha256=frozen_polynomial_identity(polynomial),
                original_input_sha256=prior['files'][fit['directory']+'/inputs.json'],
                new_outer_preparation_cost=outer_cost, charged_setup_cpu_seconds_by_arm=charges,
                complete_preparation_workflow_cost=elapsed(fit_start)))
        except Exception as error:
            write_new(directory/'preparation_failure.json', dict(author='Amanjeet Singh',
                design_sha256=design_sha, failed_fit_index=fit['index'], failed_phase=phase,
                completed_fits=len(retained), exception_type=type(error).__name__, message=str(error),
                complete_workflow_cost_through_failure=elapsed(start),
                continuation='Preserve this version and use a new explicitly identified preparation version.'))
            raise
        print(f'Prepared new outer stream for frozen fit {fit["index"]}', flush=True)
    manifest = dict(author='Amanjeet Singh', design_sha256=design_sha, source_sha256=sources, fits=retained)
    write_new(directory/'prepared.json', manifest)
    write_new(directory/'preparation_completion.json', dict(author='Amanjeet Singh',
        design_sha256=design_sha, source_sha256=sources, completed_fits=8,
        prepared_sha256=digest((directory/'prepared.json').read_bytes()),
        complete_preparation_validation_workflow_cost=elapsed(start),
        timing_scope='Original evidence validation, new preparation, validation and artifact I/O; completion-marker serialisation excluded.'))
    return manifest


def registration(source, directory):
    source, directory = private_directory(source), private_directory(directory)
    prior, old_records, polynomials = verify_original(source)
    sources = source_hashes()
    design_source = (directory/'design.json').read_bytes()
    design, design_sha = json.loads(design_source), digest(design_source)
    validate_design(design, prior, sources)
    manifest_source = (directory/'prepared.json').read_bytes()
    manifest = json.loads(manifest_source)
    completion = json.loads((directory/'preparation_completion.json').read_bytes())
    if (manifest.get('design_sha256') != design_sha or manifest.get('source_sha256') != sources
        or completion.get('design_sha256') != design_sha or completion.get('source_sha256') != sources
        or completion.get('prepared_sha256') != digest(manifest_source) or completion.get('completed_fits') != 8
        or len(manifest.get('fits', [])) != 8 or (directory/'preparation_failure.json').exists()):
        raise ValueError('The new preparation is incomplete, failed or changed')
    original.validate_cost(completion['complete_preparation_validation_workflow_cost'])
    records, views = [], []
    for fit, old, polynomial, retained in zip(design['fits'], old_records, polynomials, manifest['fits']):
        folder = directory/fit['directory']
        input_source, view_source = (folder/'inputs.json').read_bytes(), (folder/'polynomial_view.json').read_bytes()
        if (retained['index'] != fit['index'] or retained['root_seed'] != fit['root_seed']
            or digest(input_source) != retained['input_sha256']
            or digest(view_source) != retained['polynomial_view_sha256']):
            raise ValueError('Prepared fit identity or input/export hashes changed')
        record, view = json.loads(input_source), json.loads(view_source)
        validate_input(record, old, fit, design_sha, sources, prior['files'][fit['directory']+'/inputs.json'])
        charges = setup_charges(old, polynomial, record['new_outer_preparation_cost'])
        expected_view = adapted_polynomial(polynomial, record, charges, prior['files'][fit['directory']+'/polynomial.json'])
        if (view != expected_view or retained['charged_setup_cpu_seconds_by_arm'] != charges
            or retained['new_outer_preparation_cost'] != record['new_outer_preparation_cost']
            or retained['frozen_model_policy_values_sha256'] != frozen_model_hash(old)
            or retained['frozen_degree_three_polynomial_sha256'] != frozen_polynomial_identity(polynomial)
            or retained['original_input_sha256'] != prior['files'][fit['directory']+'/inputs.json']):
            raise ValueError('The retained coefficients, costs or adapted export changed')
        original.validate_cost(retained['complete_preparation_workflow_cost'])
        records.append(record); views.append(view)
    return design, manifest, completion, records, views, sources, design_sha


def verify_streams(source, directory):
    """Independently reconstruct the new outer stream without any fitting."""
    directory = private_directory(directory)
    start = clocks()
    design, manifest, _, records, _, sources, design_sha = registration(source, directory)
    checks = []
    for fit, record, retained in zip(design['fits'], records, manifest['fits']):
        states, stops, key = fresh_outer(record)
        if states != record['states'] or stops != record['stopping_indices'] or key != [3]:
            raise ValueError('The complete new outer stream or stopping decisions do not reproduce')
        checks.append(dict(fit_index=fit['index'], root_seed=fit['root_seed'],
            input_sha256=retained['input_sha256'], complete_states_reproduced=True,
            complete_stopping_indices_reproduced=True, spawn_key=key))
    record = dict(author='Amanjeet Singh', design_sha256=design_sha, source_sha256=sources,
        checks=checks, training_and_value_streams_reconstructed=False)
    path = directory/'stream_verification.json'
    completion_path = directory/'stream_verification_completion.json'
    if path.exists() or completion_path.exists():
        if not path.is_file() or not completion_path.is_file():
            raise ValueError('The stream-verification completion is incomplete')
        retained_source = path.read_bytes()
        completion = json.loads(completion_path.read_bytes())
        if (json.loads(retained_source) != record
            or completion.get('verification_sha256') != digest(retained_source)
            or completion.get('design_sha256') != design_sha
            or completion.get('source_sha256') != sources):
            raise ValueError('The retained fresh-stream verification changed')
        original.validate_cost(completion['complete_validation_workflow_cost'])
        return record
    write_new(path, record)
    write_new(completion_path, dict(author='Amanjeet Singh',
        design_sha256=design_sha, source_sha256=sources,
        verification_sha256=digest(path.read_bytes()), complete_validation_workflow_cost=elapsed(start)))
    return record


def decoded(certificate, *, nonnegative=True):
    if any(int(certificate[key]['denominator']) <= 0 for key in ('lower', 'upper', 'width')):
        raise ValueError('Certificate denominators must be positive')
    lower, upper, width = (load_fraction(certificate[key]) for key in ('lower', 'upper', 'width'))
    if upper < lower or width != upper-lower or (nonnegative and lower < 0):
        raise ValueError('An exact interval endpoint/width identity fails')
    for key, value in (('lower', lower), ('upper', upper), ('width', width)):
        if certificate[key]['approximation'] != float(value):
            raise ValueError('A rounded endpoint differs from its exact rational')
    return lower, upper


def aggregate_intervals(intervals):
    count = len(intervals)
    if count < 2:
        raise ValueError('At least two complete matched observations are required')
    lower = sum(lo for lo, hi in intervals)/count
    upper = sum(hi for lo, hi in intervals)/count
    midpoints = [float((lo+hi)/2) for lo, hi in intervals]
    return dict(mean_certificate=interval_record(lower, upper),
        approximate_midpoint_mean=statistics.mean(midpoints),
        approximate_midpoint_mean_standard_error=statistics.stdev(midpoints)/math.sqrt(count),
        midpoint_mean_numerical_error_bound=float((upper-lower)/2),
        maximum_observation_width=float(max(hi-lo for lo, hi in intervals)))


def arm_result(record, view, arm):
    from refine_multivariate_gap import cold_uniform, compare
    import certify_polynomial_dual as hybrid
    start = clocks()
    if arm == 'cold_uniform':
        output = cold_uniform(record, target=Fraction(1, 100), precision=96)
        row = dict(output['strategy'], observations=output['observations'],
            assessment_cpu_seconds=output['actual_assessment_cpu_seconds'],
            assessment_wall_seconds=output['actual_assessment_wall_seconds'],
            input_and_policy_validation_cpu_seconds=output['input_and_policy_validation_cpu_seconds'])
    elif arm == 'adaptive':
        output = compare(record, target=Fraction(1, 100), coarse_tolerance=Fraction(1),
                         precision=96, include_uniform=False)
        row = dict(output['strategies']['adaptive'], observations=output['observations']['adaptive'],
            assessment_cpu_seconds=output['actual_experiment_cpu_seconds'],
            assessment_wall_seconds=output['actual_experiment_wall_seconds'],
            input_and_policy_validation_cpu_seconds=output['input_and_policy_validation_cpu_seconds'])
    elif arm == 'hybrid_exact_terminal':
        output = hybrid.assess(record, view, target=Fraction(1, 100), precision=96,
                              degrees=(3,), methods=('hybrid_exact_terminal',))
        row = deepcopy(output['degrees']['3']['methods']['hybrid_exact_terminal'])
        row['inner_arm_assessment_cpu_seconds'] = row['assessment_cpu_seconds']
        row['inner_arm_assessment_wall_seconds'] = row['assessment_wall_seconds']
        for field in ('standalone_assessment_cpu_seconds_including_validation',
            'standalone_assessment_wall_seconds_including_validation',
            'retained_setup_pricing_cpu_seconds', 'standalone_pricing_cpu_seconds',
            'standalone_verified_workflow_cpu_seconds'):
            row.pop(field, None)
        # Charge the complete API function, including validation, result
        # assembly and method metadata, rather than only its inner arm timer.
        row.update(assessment_cpu_seconds=output['actual_assessment_cpu_seconds'],
            assessment_wall_seconds=output['actual_assessment_wall_seconds'],
            input_and_policy_validation_cpu_seconds=output['input_and_policy_validation_cpu_seconds'])
    else:
        raise ValueError('Unknown extension assessment arm')
    row['kernel_recorded_assessment_cpu_seconds'] = row['assessment_cpu_seconds']
    row['kernel_recorded_assessment_wall_seconds'] = row['assessment_wall_seconds']
    complete_cost = elapsed(start)
    row['assessment_cpu_seconds'] = complete_cost['cpu_seconds']
    row['assessment_wall_seconds'] = complete_cost['wall_seconds']
    row['certificate_phase_cpu_seconds'] = row['assessment_cpu_seconds']-row['input_and_policy_validation_cpu_seconds']
    row['assessment_timing_scope'] = 'Complete kernel execution and normalised result construction, including input and frozen-policy validation; imports and artifact I/O excluded.'
    return row


def paired_hybrid_difference(arms):
    hybrid, adaptive = (arms[name]['observations'] for name in ('hybrid_exact_terminal', 'adaptive'))
    intervals = []
    rows = []
    for h, a in zip(hybrid, adaptive):
        hlo, hhi = decoded(h['signed_gap_certificate'])
        alo, ahi = decoded(a['signed_gap_certificate'])
        lo, hi = hlo-ahi, hhi-alo
        intervals.append((lo, hi))
        rows.append(dict(path_index=h['path_index'], stopping_index=h['stopping_index'],
                         difference_certificate=interval_record(lo, hi)))
    output = aggregate_intervals(intervals)
    output.update(mean_difference_certificate=output.pop('mean_certificate'),
                  observations=rows, direction='Hybrid exact-terminal gap minus adaptive hinge gap')
    return output


def fit_context(fit, retained, design_sha, sources):
    return dict(fit_index=fit['index'], root_seed=fit['root_seed'], arm_order=fit['arm_order'],
        input_sha256=retained['input_sha256'], polynomial_view_sha256=retained['polynomial_view_sha256'],
        design_sha256=design_sha, source_sha256=sources,
        frozen_model_policy_values_sha256=retained['frozen_model_policy_values_sha256'],
        frozen_degree_three_polynomial_sha256=retained['frozen_degree_three_polynomial_sha256'],
        original_input_sha256=retained['original_input_sha256'],
        original_outer_spawn_key=[2], new_outer_spawn_key=[3], precision_bits=96, target='1/100')


def validate_result(result, record, retained, context):
    if any(result.get(key) != value for key, value in context.items()):
        raise ValueError('A frozen extension result changes its context or order')
    if (result.get('outer_paths') != OUTER_PATHS or list(result['arms']) != context['arm_order']
        or result.get('hinge_path_certificates_overlap') is not True
        or result.get('new_outer_preparation_cost') != retained['new_outer_preparation_cost']
        or result.get('charged_setup_cpu_seconds_by_arm') != retained['charged_setup_cpu_seconds_by_arm']):
        raise ValueError('A result changes its arms, sample count, setup or paired-check result')
    intervals_by_arm = {}
    for name, row in result['arms'].items():
        observations = row['observations']
        if len(observations) != OUTER_PATHS or row['all_targets_met'] is not True:
            raise ValueError('Missing paths or an unsuccessful numerical target')
        intervals = []
        for index, observation in enumerate(observations):
            if (type(observation['path_index']) is not int or observation['path_index'] != index
                or observation['stopping_index'] != record['stopping_indices'][index]):
                raise ValueError('A path or stopping identity changed')
            lo, hi = decoded(observation['signed_gap_certificate'])
            if hi-lo > Fraction(1, 100):
                raise ValueError('An actual path gap width exceeds its declared target')
            if name == 'hybrid_exact_terminal':
                if decoded(observation['selected_date_gap_certificate']) != (0, 0):
                    raise ValueError('The selected-date term must cancel exactly')
            else:
                alo, ahi = decoded(observation['absolute_gap_certificate'])
                if not alo <= lo <= hi <= ahi:
                    raise ValueError('The signed certificate leaves its absolute enclosure')
            intervals.append((lo, hi))
        aggregate = aggregate_intervals(intervals)
        if decoded(row['mean_signed_gap_certificate']) != decoded(aggregate['mean_certificate']):
            raise ValueError('A mean does not aggregate every exact path endpoint')
        for field, expected in (('maximum_signed_width', aggregate['maximum_observation_width']),
            ('mean_signed_width', float(decoded(aggregate['mean_certificate'])[1]-decoded(aggregate['mean_certificate'])[0])),
            ('approximate_midpoint_mean', aggregate['approximate_midpoint_mean']),
            ('approximate_midpoint_mean_standard_error', aggregate['approximate_midpoint_mean_standard_error']),
            ('midpoint_mean_numerical_error_bound', aggregate['midpoint_mean_numerical_error_bound'])):
            original.equal(row[field], expected, 'An exact path-derived summary changed: '+field)
        for field in ('conditional_integrals', 'integrand_evaluations'):
            work = [observation[field] if name == 'hybrid_exact_terminal' else observation['total_work'][field]
                    for observation in observations]
            if any(type(value) is not int or value < 0 for value in work) or row[field] != sum(work):
                raise ValueError('A numerical work count omits path work')
        for field in ('assessment_cpu_seconds', 'assessment_wall_seconds', 'input_and_policy_validation_cpu_seconds',
                      'certificate_phase_cpu_seconds'):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError('An assessment timer is invalid')
        original.equal(row['certificate_phase_cpu_seconds'],
            row['assessment_cpu_seconds']-row['input_and_policy_validation_cpu_seconds'],
            'A certificate timer omits or duplicates policy validation')
        standalone = retained['charged_setup_cpu_seconds_by_arm'][name]+row['assessment_cpu_seconds']
        incremental = retained['new_outer_preparation_cost']['cpu_seconds']+row['assessment_cpu_seconds']
        original.equal(result['standalone_pricing_cpu_seconds_by_arm'][name], standalone,
                       'A standalone budget omits or duplicates retained setup')
        original.equal(result['incremental_evaluation_cpu_seconds_by_arm'][name], incremental,
                       'A fresh evaluation budget includes previous fitting work')
        intervals_by_arm[name] = intervals
    for uniform, adaptive in zip(intervals_by_arm['cold_uniform'], intervals_by_arm['adaptive']):
        if max(uniform[0], adaptive[0]) > min(uniform[1], adaptive[1]):
            raise ValueError('Matched hinge certificates do not intersect')
    if result['paired_hybrid_minus_adaptive'] != paired_hybrid_difference(result['arms']):
        raise ValueError('The paired hybrid-minus-hinge differences changed')
    return True


def assessment_fit(directory, fit, record, view, retained, design_sha, sources):
    folder = directory/fit['directory']
    context = fit_context(fit, retained, design_sha, sources)
    names = ('assessment_attempt', 'result', 'assessment_completion', 'assessment_failure')
    paths = {name:folder/(name+'.json') for name in names}
    if paths['result'].exists() or paths['assessment_completion'].exists():
        if not all(paths[name].is_file() for name in names[:-1]) or paths['assessment_failure'].exists():
            raise ValueError('Preserve an incomplete or failed assessment version')
        result_source = paths['result'].read_bytes()
        attempt = json.loads(paths['assessment_attempt'].read_bytes())
        completion = json.loads(paths['assessment_completion'].read_bytes())
        if (any(attempt.get(key) != value or completion.get(key) != value for key, value in context.items())
            or completion.get('result_sha256') != digest(result_source)):
            raise ValueError('A completed fit has altered input, design, source or output')
        original.validate_cost(completion['complete_assessment_validation_workflow_cost'])
        result = json.loads(result_source)
        validate_result(result, record, retained, context)
        return result
    if paths['assessment_attempt'].exists() or paths['assessment_failure'].exists():
        raise ValueError('Do not silently restart an interrupted or failed assessment')
    start = clocks()
    write_new(paths['assessment_attempt'], dict(**context, started_at_utc=datetime.now(timezone.utc).isoformat()))
    arms = {}
    phase = 'arm_assessment'
    try:
        for name in fit['arm_order']:
            phase = name
            arms[name] = arm_result(record, view, name)
            if frozen_model_hash(record) != context['frozen_model_policy_values_sha256']:
                raise ValueError('An assessor changed the retained policy or hinge functions')
        phase = 'paired_aggregation_and_validation'
        result = dict(schema_version=1, author='Amanjeet Singh', **context, outer_paths=len(record['states']),
            arms=arms, hinge_path_certificates_overlap=True,
            new_outer_preparation_cost=retained['new_outer_preparation_cost'],
            charged_setup_cpu_seconds_by_arm=retained['charged_setup_cpu_seconds_by_arm'],
            standalone_pricing_cpu_seconds_by_arm={name:retained['charged_setup_cpu_seconds_by_arm'][name]+
                row['assessment_cpu_seconds'] for name, row in arms.items()},
            incremental_evaluation_cpu_seconds_by_arm={name:retained['new_outer_preparation_cost']['cpu_seconds']+
                row['assessment_cpu_seconds'] for name, row in arms.items()},
            paired_hybrid_minus_adaptive=paired_hybrid_difference(arms),
            budget_scope='Retained setup components plus fresh complete API assessment, including input/policy checks and result construction. Previous outer preparation and previous assessments are excluded.',
            numerical_scope='Ideal gap arithmetic at supplied binary states; mean standard errors are approximate conditional diagnostics.')
        validate_result(result, record, retained, context)
        phase = 'result_writing'
        write_new(paths['result'], result)
        write_new(paths['assessment_completion'], dict(**context,
            result_sha256=digest(paths['result'].read_bytes()),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            complete_assessment_validation_workflow_cost=elapsed(start),
            timing_scope='Attempt recording, all arm executions, exact paired validation and result I/O; completion-marker serialisation excluded.'))
    except Exception as error:
        write_new(paths['assessment_failure'], dict(**context, failed_phase=phase,
            completed_arms=list(arms), completed_arm_results=arms, exception_type=type(error).__name__, message=str(error),
            complete_workflow_cost_through_failure=elapsed(start),
            continuation='Preserve every result and failure; no omitted fit or replacement seed.'))
        raise
    print(f'Completed larger-sample frozen fit {fit["index"]}', flush=True)
    return result


def ratio_summary(values):
    if len(values) != 8 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Eight finite fit-level ratios are required')
    return dict(fitting_units=8, minimum=min(values), median=statistics.median(values), maximum=max(values))


def summarise(results, manifest, design_sha, sources):
    if len(results) != 8:
        raise ValueError('All eight frozen fits are required; no omission or replacement')
    means = {name:aggregate_intervals([decoded(result['arms'][name]['mean_signed_gap_certificate'])
                                    for result in results]) for name in ARMS}
    paired = aggregate_intervals([decoded(result['paired_hybrid_minus_adaptive']['mean_difference_certificate'],
                                         nonnegative=False) for result in results])
    ratios = {}
    for label, field in (('complete_assessment_cpu', None),
                        ('standalone_pricing_cpu', 'standalone_pricing_cpu_seconds_by_arm'),
                        ('incremental_evaluation_cpu', 'incremental_evaluation_cpu_seconds_by_arm')):
        if field is None:
            values = [result['arms']['adaptive']['assessment_cpu_seconds']/result['arms']['cold_uniform']['assessment_cpu_seconds']
                      for result in results]
        else:
            values = [result[field]['adaptive']/result[field]['cold_uniform'] for result in results]
        ratios[label] = ratio_summary(values)
    ratios['conditional_integrals'] = ratio_summary([result['arms']['adaptive']['conditional_integrals']/
        result['arms']['cold_uniform']['conditional_integrals'] for result in results])
    cases = []
    for result in results:
        case = deepcopy(result)
        for row in case['arms'].values():
            row.pop('observations')
        case['paired_hybrid_minus_adaptive'].pop('observations')
        cases.append(case)
    return dict(schema_version=1, author='Amanjeet Singh', design_sha256=design_sha, source_sha256=sources,
        completed_fits=8, fitting_units=8, outer_paths_per_fit=OUTER_PATHS, total_new_outer_paths=8*OUTER_PATHS,
        cases=cases, exact_across_fit_gap_summaries=means, exact_across_fit_paired_difference=paired,
        paired_adaptive_over_cold_uniform_ratios=ratios,
        conditional_mean_gap_standard_error_by_arm={name: math.sqrt(sum(
            result['arms'][name]['approximate_midpoint_mean_standard_error']**2 for result in results))/8
            for name in ARMS},
        conditional_mean_paired_difference_standard_error=(math.sqrt(sum(
            result['paired_hybrid_minus_adaptive']['approximate_midpoint_mean_standard_error']**2
            for result in results))/8),
        conditional_uncertainty_scope='The retained fits and preceding comparator selection are held fixed. Independent fresh path streams supply the variance of their equally weighted mean paired difference. The across-fit mean standard error is a separate descriptive variation measure, not a new unconditional confidence interval.',
        new_outer_preparation_cost_by_fit=[row['new_outer_preparation_cost'] for row in manifest['fits']],
        fitting_and_value_streams_rerun=False, original_setup_measurements_retained=True,
        sampling_unit='Eight retained fits and the preceding comparator selection are held fixed. Fresh paths supply conditional sampling variation. Across-fit dispersion of the eight means is descriptive; the 8,192 paths are not independent fitted-model replications.',
        interpretation='Independent larger-sample evaluation of retained functions and a previously selected degree-three comparator. Same required maximum numerical gap width; actual achieved widths differ. Relative assessor gaps do not identify the optimal option value or policy loss.')


def assessment(source, directory):
    directory = private_directory(directory)
    start = clocks()
    design, manifest, prep_completion, records, views, sources, design_sha = registration(source, directory)
    results = [assessment_fit(directory, fit, record, view, retained, design_sha, sources)
        for fit, record, view, retained in zip(design['fits'], records, views, manifest['fits'])]
    summary = summarise(results, manifest, design_sha, sources)
    summary_path, completion_path = directory/'summary.json', directory/'assessment_completion.json'
    if summary_path.exists() or completion_path.exists():
        if not summary_path.is_file() or not completion_path.is_file():
            raise ValueError('Preserve an incomplete final summary')
        existing = summary_path.read_bytes()
        completion = json.loads(completion_path.read_bytes())
        if (json.loads(existing) != summary or completion.get('summary_sha256') != digest(existing)
            or completion.get('design_sha256') != design_sha or completion.get('source_sha256') != sources
            or completion.get('completed_fits') != 8):
            raise ValueError('The summary differs from every verified fit')
        original.validate_cost(completion['complete_assessment_validation_workflow_cost'])
        print('Verified all completed larger-sample fits without rerunning them.', flush=True)
        return summary
    write_new(summary_path, summary)
    write_new(completion_path, dict(author='Amanjeet Singh', design_sha256=design_sha, source_sha256=sources,
        completed_fits=8, summary_sha256=digest(summary_path.read_bytes()),
        complete_assessment_validation_workflow_cost=elapsed(start),
        preceding_preparation_validation_workflow_cost=prep_completion['complete_preparation_validation_workflow_cost'],
        timing_scope='Original/new registration checks, fit verification or execution, paired summary validation and I/O; completion-marker serialisation excluded.'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'assess', 'verify-streams'))
    parser.add_argument('--source-study', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    {'prepare': preparation, 'assess': assessment, 'verify-streams': verify_streams}[args.mode](
        args.source_study, args.directory)


if __name__ == '__main__':
    main()
