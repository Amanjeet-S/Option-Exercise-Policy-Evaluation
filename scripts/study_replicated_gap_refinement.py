"""Prepare and assess eight independently fitted public-model gap comparisons.

Preparation runs in the research environment and assessment in the separate
python-flint environment. A prior design fixes eight distinct fitting seeds,
256 outer states per fit, a common maximum path-gap width and alternating
strategy order. Established polynomial comparators are declared for separate
execution in their respective research and interval-arithmetic environments.

Completed fits can be resumed after checking their input, design, sources,
exact path certificates, derived statistics and completion hashes. An
interrupted or failed attempt is preserved and cannot be silently restarted.
The fitting unit is a policy/value-function pair, not an individual path.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from certify_multivariate_gap import check_inputs, load_fraction, prepare
from study_multivariate_gap_refinement import (
    case_result, frozen_model_hash, source_hashes as existing_source_hashes)

FITTING_UNITS = 8
SEEDS = tuple(range(2026100410, 2026100418))
ADDITIONAL_SOURCES = ('scripts/study_replicated_gap_refinement.py',
                      'scripts/validate_polynomial_dual.py', 'scripts/certify_polynomial_dual.py',
                      'tests/test_certify_polynomial_dual.py', 'tests/test_replicated_gap_refinement.py')
STRATEGIES = ('cold_uniform', 'adaptive')


def clocks():
    return time.process_time(), time.perf_counter()


def elapsed(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def digest(source):
    return hashlib.sha256(source).hexdigest()


def source_hashes():
    result = existing_source_hashes()
    result.update({name: digest((ROOT/name).read_bytes()) for name in ADDITIONAL_SOURCES})
    return result


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')



def normalise_prepared_record(record):
    """Use the exact JSON container representation required by saved inputs.

    Model dataclass fields are tuples before serialisation, whereas input
    validation compares JSON lists. The round trip changes containers, not
    finite binary floating-point values, coefficients, states or decisions.
    Non-finite values are rejected before input-file creation.
    """
    return json.loads(json.dumps(record, allow_nan=False))


def expected_benchmark_model():
    return dict(spots=[100.]*5, volatilities=[.2]*5, dividends=[.1]*5,
        correlation=[[float(i == j) for j in range(5)] for i in range(5)],
        rate=.05, strike=100., exercise_times=[3.*i/9 for i in range(10)])

def private_directory(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Retain preparation, states and assessment ledgers in a private directory')
    return directory


def fits_design():
    return [dict(index=index, seed=seed, directory=f'fit_{index:02d}',
                 case=dict(name=f'fit_{index:02d}', target='1/100',
                           order=list(STRATEGIES if index % 2 == 0 else STRATEGIES[::-1])),
                 polynomial_degree_order=[2, 3] if index % 2 == 0 else [3, 2],
                 verified_polynomial_method_order=['polynomial_all_dates', 'hybrid_exact_terminal']
                     if index % 2 == 0 else ['hybrid_exact_terminal', 'polynomial_all_dates'])
            for index, seed in enumerate(SEEDS)]


def design_record(sources):
    return dict(schema_version=1, author='Amanjeet Singh',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        recorded_before_any_training=True, fitting_units=FITTING_UNITS,
        dimension=5, initial_spot=100., training_paths=50000, value_paths=50000,
        outer_paths_per_fit=256, precision_bits=96, coarse_conditional_width='1',
        fits=fits_design(),
        mathematical_benchmark='Andersen and Broadie (2004), Table 2: independent five-asset max-call, S=K=100, r=.05, q=.10, sigma=.20, T=3, exercise dates iT/9 including zero.',
        primary_comparison='Cold uniform conditional integration versus fixed-centre adaptive refinement, at maximum path-gap width 1/100.',
        planned_established_comparators=[
            dict(name='unsorted_exact_moment_polynomial_regression_later', degrees=[2, 3],
                 basis='Complete unsorted total-degree monomials in the full state',
                 source='scripts/validate_polynomial_dual.py'),
            dict(name='verified_hybrid_polynomial_with_exact_terminal_call', degrees=[2, 3],
                 conditional_arithmetic='Arb enclosures for polynomial moments and the exact terminal max-call',
                 source='scripts/certify_polynomial_dual.py')],
        established_comparator_execution_modes=['polynomial', 'certify-polynomial'],
        policy='Frozen sorted full-state degree-two Longstaff-Schwartz policy.',
        fitting_streams='Independent policy-training and value-fitting streams; a third stream supplies outer paths within each distinct root seed.',
        statistical_unit='An independently fitted policy and value-function pair. Eight fitting units, each with 256 paths; 2,048 paths are not 2,048 independent fitted-model replications.',
        accuracy_comparison='Same required maximum path-gap width, not necessarily equal achieved width.',
        pricing_budget_scope='Each standalone strategy is charged one pricing preparation_cost plus its complete assessment function cost, including policy checks. Artifact I/O is recorded separately.',
        workflow_timing_scope='Complete preparation and assessment validation workflows include checks and artifact I/O. Completion-marker serialisation is excluded from its own timer.',
        sampling_scope='Interval-midpoint standard errors are approximate diagnostics. Simulator law and statistical coverage are not computer-certified.',
        public_preregistration_claim=False, new_general_theorem_claim=False,
        source_sha256=sources)


def validate_design(design, sources):
    expected = design_record(sources)
    for key in expected:
        if key != 'recorded_at_utc' and design.get(key) != expected[key]:
            raise ValueError('The retained replicated design differs: '+key)
    if not isinstance(design.get('recorded_at_utc'), str):
        raise ValueError('The prior design must retain its actual recording time')


def validate_input(record, fit, design_sha, sources):
    check_inputs(record)
    if (record.get('preparation_design_sha256') != design_sha or
            record.get('preparation_source_sha256') != sources or
            record.get('replicated_fit_index') != fit['index']):
        raise ValueError('Prepared fit identifies a different registration or source')
    declared = record['design']
    expected = dict(root_seed=fit['seed'], training_paths=50000, value_paths=50000, outer_paths=256,
                    spawn_keys=[[0], [1], [2]], evaluation_independent_of_both_fitting_streams=True,
                    policy_degree=2)
    if any(declared.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepared fitting counts, streams or seed differ from the prior design')
    model = record['model']
    expected_model = expected_benchmark_model()
    if model != expected_model or len(record['states']) != 256:
        raise ValueError('Prepared fit uses a different mathematical benchmark')
    snapshot = frozen_model_hash(record)
    if record.get('frozen_model_policy_values_sha256') != snapshot:
        raise ValueError('The frozen model, value functions or policy have changed')
    validate_cost(record['preparation_cost'])
    phases = record.get('preparation_phases')
    if not isinstance(phases, dict) or set(phases) != {'policy_training', 'hinge_value_stream_and_targets',
            'hinge_regression', 'outer_evaluation_and_record_construction', 'preparation_bookkeeping'}:
        raise ValueError('Complete exclusive preparation phases are required for the established comparators')
    for key in ('cpu_seconds', 'wall_seconds'):
        if any(not math.isfinite(row[key]) or row[key] < -1e-9 for row in phases.values()):
            raise ValueError('Invalid preparation phase cost')
        equal(record['preparation_cost'][key], sum(row[key] for row in phases.values()),
              'Preparation phases do not sum to the complete pricing preparation')
    return snapshot


def validate_cost(value):
    if not isinstance(value, dict) or set(value) != {'cpu_seconds', 'wall_seconds'}:
        raise ValueError('A complete CPU/wall cost record is required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
           for x in value.values()):
        raise ValueError('Recorded computation costs must be finite and non-negative')


def preparation(directory):
    directory = private_directory(directory)
    if directory.exists():
        raise ValueError('Prepare a fresh private version; existing attempts cannot be overwritten')
    start = clocks()
    sources = source_hashes()  # All declared comparator sources must exist before recording a design.
    design = design_record(sources)
    directory.mkdir(parents=True)
    write_new(directory/'design.json', design)
    design_sha = digest((directory/'design.json').read_bytes())
    prepared = []
    for fit in design['fits']:
        fit_start = clocks()
        folder = directory/fit['directory']
        phase = 'directory_creation'
        try:
            folder.mkdir()
            phase = 'scientific_preparation'
            record = prepare(dimension=5, training_count=50000, value_count=50000,
                             evaluation_count=256, seed=fit['seed'])
            phase = 'json_container_normalisation'
            record = normalise_prepared_record(record)
            record.update(preparation_design_sha256=design_sha, preparation_source_sha256=sources,
                          replicated_fit_index=fit['index'],
                          frozen_model_policy_values_sha256=frozen_model_hash(record))
            phase = 'input_validation'
            snapshot = validate_input(record, fit, design_sha, sources)
            phase = 'input_writing_and_hashing'
            write_new(folder/'inputs.json', record)
            input_sha = digest((folder/'inputs.json').read_bytes())
            prepared.append(dict(index=fit['index'], seed=fit['seed'],
                input_path=fit['directory']+'/inputs.json', input_sha256=input_sha,
                frozen_model_policy_values_sha256=snapshot,
                pricing_preparation_cost=record['preparation_cost'], pricing_preparation_phases=record['preparation_phases'],
                complete_preparation_workflow_cost=elapsed(fit_start)))
        except Exception as error:
            write_new(directory/'preparation_failure.json', dict(schema_version=1, author='Amanjeet Singh',
                recorded_at_utc=datetime.now(timezone.utc).isoformat(), design_sha256=design_sha,
                source_sha256=sources, failed_fitting_unit_index=fit['index'], root_seed=fit['seed'],
                failed_phase=phase, completed_fits=len(prepared),
                failed_fit_input_saved=(folder/'inputs.json').exists(),
                exception_type=type(error).__name__, message=str(error),
                complete_preparation_workflow_cost_through_failure=elapsed(start),
                continuation='Retain this stopped preparation and use a fresh version. Completed inputs and the original design are preserved.'))
            raise

    manifest = dict(schema_version=1, author='Amanjeet Singh', design_sha256=design_sha,
                    source_sha256=sources, fits=prepared)
    write_new(directory/'prepared.json', manifest)
    completion = dict(schema_version=1, author='Amanjeet Singh', design_sha256=design_sha,
        prepared_sha256=digest((directory/'prepared.json').read_bytes()), source_sha256=sources,
        completed_fits=8, complete_preparation_workflow_cost=elapsed(start),
        timing_scope='From preparation entry through input and preparation-manifest writing/hash verification; completion-marker serialisation excluded.')
    write_new(directory/'preparation_completion.json', completion)
    print(f'Prepared eight frozen fits in {directory}', flush=True)
    return manifest


def registration(directory):
    design_source = (directory/'design.json').read_bytes()
    design = json.loads(design_source)
    sources, design_sha = source_hashes(), digest(design_source)
    validate_design(design, sources)
    manifest_source = (directory/'prepared.json').read_bytes()
    manifest = json.loads(manifest_source)
    completion = json.loads((directory/'preparation_completion.json').read_bytes())
    if (manifest['design_sha256'] != design_sha or manifest['source_sha256'] != sources or
            completion['design_sha256'] != design_sha or completion['source_sha256'] != sources or
            completion['prepared_sha256'] != digest(manifest_source) or completion['completed_fits'] != 8):
        raise ValueError('The preparation manifest or completion record has changed')
    validate_cost(completion['complete_preparation_workflow_cost'])
    if len(manifest['fits']) != 8:
        raise ValueError('The preparation manifest does not contain eight fits')
    records = []
    for fit, retained in zip(design['fits'], manifest['fits']):
        if (retained['index'] != fit['index'] or retained['seed'] != fit['seed'] or
                retained['input_path'] != fit['directory']+'/inputs.json'):
            raise ValueError('The preparation manifest changes a fitting unit')
        source = (directory/retained['input_path']).read_bytes()
        if digest(source) != retained['input_sha256']:
            raise ValueError('A prepared input file has changed')
        record = json.loads(source)
        snapshot = validate_input(record, fit, design_sha, sources)
        if (snapshot != retained['frozen_model_policy_values_sha256'] or
                record['preparation_cost'] != retained['pricing_preparation_cost'] or
                record['preparation_phases'] != retained['pricing_preparation_phases']):
            raise ValueError('Prepared identities or pricing costs differ from their manifest')
        validate_cost(retained['complete_preparation_workflow_cost'])
        records.append(record)
    return design, manifest, completion, records, sources, design_sha


def rational_interval(record):
    try:
        if any(int(record[key]['denominator']) <= 0 for key in ('lower', 'upper', 'width')):
            raise ValueError('Certificate denominators must be positive')
        lo, hi, width = (load_fraction(record[key]) for key in ('lower', 'upper', 'width'))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError('Invalid exact certificate record') from error
    if lo < 0 or hi < lo or width != hi-lo:
        raise ValueError('Certificate endpoint/width identity fails')
    for key, value in (('lower', lo), ('upper', hi), ('width', width)):
        if record[key]['approximation'] != float(value):
            raise ValueError('Certificate display differs from its exact rational')
    return lo, hi


def equal(left, right, message):
    if (isinstance(left, bool) or not isinstance(left, (int, float)) or not math.isfinite(left)
            or not math.isclose(left, right, rel_tol=0., abs_tol=1e-12)):
        raise ValueError(message)


def validate_result(result, record, fit, retained, design_sha, sources):
    identity = dict(input_sha256=retained['input_sha256'], design_sha256=design_sha,
                    source_sha256=sources, fitting_unit_index=fit['index'], root_seed=fit['seed'],
                    frozen_model_policy_values_sha256=frozen_model_hash(record), outer_paths=256,
                    precision_bits=96, name=fit['case']['name'], strategy_order=fit['case']['order'],
                    required_maximum_path_gap_width='1/100', preparation_cost=record['preparation_cost'],
                    preparation_phases=record['preparation_phases'])
    if any(result.get(key) != value for key, value in identity.items()):
        raise ValueError('A retained assessment changes its input, registration, fitting unit or method order')
    if set(result['strategies']) != set(STRATEGIES) or set(result['observations']) != set(STRATEGIES):
        raise ValueError('Both strategies and their path certificates are required')
    if result['all_path_certificates_overlap'] is not True:
        raise ValueError('The paired certificate-overlap check did not succeed')
    path_intervals = {}
    for name in STRATEGIES:
        rows = result['observations'][name]
        if len(rows) != 256:
            raise ValueError('Missing or duplicated assessment paths')
        signed, absolute = [], []
        for index, row in enumerate(rows):
            if type(row['path_index']) is not int or row['path_index'] != index or row['stopping_index'] != record['stopping_indices'][index]:
                raise ValueError('A path index or frozen stopping decision has changed')
            lo, hi = rational_interval(row['signed_gap_certificate'])
            alo, ahi = rational_interval(row['absolute_gap_certificate'])
            if hi-lo > Fraction(1, 100) or not alo <= lo <= hi <= ahi:
                raise ValueError('A path certificate misses its target or leaves its absolute enclosure')
            equal(row['signed_width'], float(hi-lo), 'A path width differs from its certificate')
            signed.append((lo, hi)); absolute.append((alo, ahi))
        path_intervals[name] = signed
        statistics_row = result['strategies'][name]
        if statistics_row['all_targets_met'] is not True or statistics_row['all_signed_inside_absolute'] is not True:
            raise ValueError('A strategy reports an unsuccessful certificate check')
        lower = sum(value[0] for value in signed)/256
        upper = sum(value[1] for value in signed)/256
        if rational_interval(statistics_row['mean_signed_gap_certificate']) != (lower, upper):
            raise ValueError('Mean endpoints differ from the complete exact path aggregation')
        widths = [hi-lo for lo, hi in signed]
        midpoint = [float((hi+lo)/2) for lo, hi in signed]
        if len(statistics_row['mean_signed_gap_interval']) != 2:
            raise ValueError('The mean interval must contain two endpoints')
        for actual, expected in zip(statistics_row['mean_signed_gap_interval'], (lower, upper)):
            equal(actual, float(expected), 'Mean interval display differs from its certificate')
        for field, expected in (
                ('mean_signed_width', float(upper-lower)), ('maximum_signed_width', float(max(widths))),
                ('mean_absolute_width', float(sum(hi-lo for lo, hi in absolute)/256)),
                ('approximate_midpoint_mean', statistics.mean(midpoint)),
                ('approximate_midpoint_mean_standard_error', statistics.stdev(midpoint)/math.sqrt(256)),
                ('midpoint_mean_numerical_error_bound', float((upper-lower)/2))):
            equal(statistics_row[field], expected, 'Retained path-derived statistic differs: '+field)
        for field in ('conditional_integrals', 'integrand_evaluations'):
            total = sum(row['total_work'][field] for row in rows)
            if any(type(row['total_work'][field]) is not int or row['total_work'][field] < 0 for row in rows) or statistics_row[field] != total:
                raise ValueError('Recorded numerical work differs from its complete path ledger')
        for field in ('assessment_cpu_seconds', 'input_and_policy_validation_cpu_seconds', 'certificate_phase_cpu_seconds'):
            if not math.isfinite(statistics_row[field]) or statistics_row[field] < 0:
                raise ValueError('Invalid strategy assessment cost')
        equal(statistics_row['certificate_phase_cpu_seconds'],
              statistics_row['assessment_cpu_seconds']-statistics_row['input_and_policy_validation_cpu_seconds'],
              'The certificate phase omits or duplicates policy validation')
        equal(statistics_row['combined_preparation_and_assessment_cpu_seconds'],
              record['preparation_cost']['cpu_seconds']+statistics_row['assessment_cpu_seconds'],
              'The standalone comparison does not charge pricing preparation exactly once')
    for uniform, adaptive in zip(path_intervals['cold_uniform'], path_intervals['adaptive']):
        if max(uniform[0], adaptive[0]) > min(uniform[1], adaptive[1]):
            raise ValueError('A paired path certificate has no intersection')
    uniform, adaptive = (result['strategies'][name] for name in STRATEGIES)
    equal(result['adaptive_over_cold_uniform_integral_ratio'],
          adaptive['conditional_integrals']/uniform['conditional_integrals'], 'The integral ratio differs')
    equal(result['adaptive_over_cold_uniform_certificate_phase_cpu_ratio'],
          adaptive['certificate_phase_cpu_seconds']/uniform['certificate_phase_cpu_seconds'], 'The certificate-phase ratio differs')
    return True


def assessment_fit(directory, fit, record, retained, design_sha, sources):
    folder = directory/fit['directory']
    paths = {name: folder/(name+'.json') for name in ('assessment_attempt', 'result', 'assessment_completion', 'failure')}
    context = dict(input_sha256=retained['input_sha256'], design_sha256=design_sha,
                   source_sha256=sources, fitting_unit_index=fit['index'], root_seed=fit['seed'])
    if paths['result'].exists() or paths['assessment_completion'].exists():
        if not all(paths[name].exists() for name in ('assessment_attempt', 'result', 'assessment_completion')) or paths['failure'].exists():
            raise ValueError('An assessment is incomplete or failed; retain it and use an explicit new version')
        attempt = json.loads(paths['assessment_attempt'].read_bytes())
        completion = json.loads(paths['assessment_completion'].read_bytes())
        result_source = paths['result'].read_bytes()
        if (any(attempt.get(key) != value or completion.get(key) != value for key, value in context.items())
                or completion['result_sha256'] != digest(result_source)):
            raise ValueError('A completed assessment identifies altered inputs, sources or output')
        validate_cost(completion['complete_fit_validation_workflow_cost'])
        result = json.loads(result_source)
        validate_result(result, record, fit, retained, design_sha, sources)
        return result
    if paths['assessment_attempt'].exists() or paths['failure'].exists():
        raise ValueError('Do not restart an interrupted or failed fit in the same version')
    start = clocks()
    write_new(paths['assessment_attempt'], dict(**context, started_at_utc=datetime.now(timezone.utc).isoformat()))
    try:
        result = case_result(record, fit['case'], precision=96, coarse_tolerance='1')
        result.update(schema_version=1, author='Amanjeet Singh', assessed_at_utc=datetime.now(timezone.utc).isoformat(),
                      **context, preparation_cost=record['preparation_cost'], preparation_phases=record['preparation_phases'],
                      complete_preparation_workflow_cost=retained['complete_preparation_workflow_cost'],
                      pricing_budget_scope='One pricing preparation_cost plus this strategy assessment function; artifact I/O remains a separate validation-workflow cost.')
        validate_result(result, record, fit, retained, design_sha, sources)
        write_new(paths['result'], result)
        result_sha = digest(paths['result'].read_bytes())
        write_new(paths['assessment_completion'], dict(**context, result_sha256=result_sha,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            complete_fit_validation_workflow_cost=elapsed(start),
            timing_scope='Attempt recording, both assessments, exact paired/result validation and result writing/hash verification; completion-marker serialisation excluded.'))
    except Exception as error:
        if not paths['failure'].exists():
            write_new(paths['failure'], dict(**context, failed_at_utc=datetime.now(timezone.utc).isoformat(),
                exception_type=type(error).__name__, message=str(error)))
        raise
    print(f'Completed fitting unit {fit["index"]}, seed {fit["seed"]}', flush=True)
    return result


def ratio_summary(values):
    if len(values) != 8 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Eight finite paired fitting-unit ratios are required')
    return dict(fitting_units=8, minimum=min(values), median=statistics.median(values), maximum=max(values))


def summarise(results, manifest, design_sha, sources):
    ratios = {'integrals': [], 'certificate_phase_cpu': [], 'complete_assessment_cpu': [], 'charged_pricing_budget_cpu': []}
    for result in results:
        uniform, adaptive = (result['strategies'][name] for name in STRATEGIES)
        ratios['integrals'].append(adaptive['conditional_integrals']/uniform['conditional_integrals'])
        for name, field in (('certificate_phase_cpu', 'certificate_phase_cpu_seconds'),
                            ('complete_assessment_cpu', 'assessment_cpu_seconds'),
                            ('charged_pricing_budget_cpu', 'combined_preparation_and_assessment_cpu_seconds')):
            ratios[name].append(adaptive[field]/uniform[field])
    return dict(schema_version=1, author='Amanjeet Singh', design_sha256=design_sha,
        source_sha256=sources, completed_fits=len(results), fitting_units=8, outer_paths_per_fit=256,
        statistical_unit='The independently fitted policy/value-function pair; summaries use eight paired fit ratios, not 2,048 independent fitted-model observations.',
        cases=[{key: value for key, value in result.items() if key != 'observations'} for result in results],
        paired_adaptive_over_cold_uniform_ratios={name: ratio_summary(values) for name, values in ratios.items()},
        pricing_preparation_cost_by_fit=[row['pricing_preparation_cost'] for row in manifest['fits']],
        complete_preparation_workflow_cost_by_fit=[row['complete_preparation_workflow_cost'] for row in manifest['fits']],
        distinct_frozen_model_policy_value_hashes=len({row['frozen_model_policy_values_sha256'] for row in results}),
        established_comparators_executed=False,
        interpretation='Descriptive matched-fit comparisons at a common required path-gap width. Strategy order alternates. Established comparator execution is separate; midpoint standard errors, simulator law and statistical coverage are not computer-certified.')


def assessment(directory):
    directory = private_directory(directory)
    start = clocks()
    design, manifest, preparation_completion, records, sources, design_sha = registration(directory)
    results = [assessment_fit(directory, fit, record, retained, design_sha, sources)
               for fit, record, retained in zip(design['fits'], records, manifest['fits'])]
    expected = summarise(results, manifest, design_sha, sources)
    summary_path, completion_path = directory/'summary.json', directory/'assessment_completion.json'
    if summary_path.exists() or completion_path.exists():
        if not summary_path.exists() or not completion_path.exists():
            raise ValueError('A final summary is incomplete; preserve this version')
        source = summary_path.read_bytes()
        completion = json.loads(completion_path.read_bytes())
        if (json.loads(source) != expected or completion['summary_sha256'] != digest(source)
                or completion['design_sha256'] != design_sha or completion['source_sha256'] != sources):
            raise ValueError('The completed summary differs from its verified fitting units')
        validate_cost(completion['complete_assessment_validation_workflow_cost'])
        print('Verified the eight completed fitting units without rerunning them.', flush=True)
        return expected
    write_new(summary_path, expected)
    summary_sha = digest(summary_path.read_bytes())
    workflow_cost = elapsed(start)
    write_new(completion_path, dict(schema_version=1, author='Amanjeet Singh',
        design_sha256=design_sha, source_sha256=sources, summary_sha256=summary_sha,
        completed_fits=8, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        complete_assessment_validation_workflow_cost=workflow_cost,
        preparation_and_assessment_validation_workflow_cpu_seconds=(
            preparation_completion['complete_preparation_workflow_cost']['cpu_seconds']+workflow_cost['cpu_seconds']),
        timing_scope='Assessment entry through registration checks, completed-fit verification or new execution, result and summary writing/hash checks; final completion-marker serialisation excluded.'))
    print('Saved the eight-fit comparison and complete validation-workflow cost.', flush=True)
    return expected


def polynomial_api():
    # NumPy is required only by this separately selected research mode.
    import validate_polynomial_dual
    return validate_polynomial_dual


def validate_polynomial_result(result, record, hinge, fit, context, api):
    if any(result.get(key) != value for key, value in context.items()):
        raise ValueError('The polynomial comparator identifies different inputs, hinge evidence or sources')
    for field in ('frozen_policy_preserved', 'supplied_outer_stream_reproduced', 'no_clipping_of_fitted_values'):
        if result.get(field) is not True:
            raise ValueError('The polynomial comparator changes its frozen assessment scope')
    if (result.get('numerical_certificate') is not False or result.get('statistical_coverage_certificate') is not False
            or result.get('sample_count') != 256 or result.get('value_fitting_paths') != 50000
            or result.get('value_spawn_key') != [1]
            or list(result['degrees']) != [str(value) for value in fit['polynomial_degree_order']]):
        raise ValueError('The polynomial comparator changes counts, degree order or numerical scope')
    for field in ('shared_value_stream_and_target_cpu_seconds', 'stream_and_policy_reconstruction_cpu_seconds'):
        if not math.isfinite(result[field]) or result[field] < 0:
            raise ValueError('Invalid shared polynomial computation cost')
    model = api.MaxCallModel(**record['model'])
    states = api.np.asarray(record['states'], dtype=float)
    stops = api.np.asarray(record['stopping_indices'], dtype=int)
    common = sum(record['preparation_phases'][key]['cpu_seconds'] for key in
                 ('policy_training', 'outer_evaluation_and_record_construction'))
    for degree, row in result['degrees'].items():
        degree = int(degree)
        expected_powers = [list(value) for value in api.exponent_vectors(5, degree)]
        if row['degree'] != degree or row['basis_exponents'] != expected_powers or row['basis_dimension'] != len(expected_powers):
            raise ValueError('The polynomial comparator changes its complete unsorted basis')
        coefficients = [api.np.asarray(value, dtype=float) for value in row['coefficients']]
        observed = api.dual_paths(states, model, stops, coefficients, degree)
        if row['all_gap_observations_nonnegative'] is not bool(api.np.all(observed['gap'] >= 0)):
            raise ValueError('The polynomial comparator misstates realised gap non-negativity')
        for field in ('gap', 'dual', 'policy_reward', 'stopped_martingale'):
            expected = api.sample_summary(observed[field])
            if row[field]['sample_count'] != 256:
                raise ValueError('A polynomial sampling summary changes its path count')
            for key in ('mean', 'standard_error', 'minimum', 'maximum'):
                equal(row[field][key], expected[key], 'A polynomial summary differs from its frozen-coefficient evaluation')
        paired = api.paired_comparison(observed['gap'], hinge['observations']['adaptive'])
        for key in ('mean', 'standard_error', 'minimum', 'maximum', 'midpoint_mean_numerical_allowance'):
            equal(row['paired_against_finest_hinge'][key], paired[key], 'A paired polynomial comparison differs')
        if len(row['paired_against_finest_hinge']['mean_difference_range_from_hinge_brackets']) != 2:
            raise ValueError('A paired numerical allowance requires two endpoints')
        for actual, expected in zip(row['paired_against_finest_hinge']['mean_difference_range_from_hinge_brackets'],
                                    paired['mean_difference_range_from_hinge_brackets']):
            equal(actual, expected, 'A paired mean difference leaves the retained hinge brackets')
        for field in ('polynomial_fit_cpu_seconds', 'polynomial_interior_fit_cpu_seconds',
                      'terminal_polynomial_fit_cpu_seconds', 'dual_assessment_cpu_seconds'):
            if not math.isfinite(row[field]) or row[field] < -1e-12:
                raise ValueError('A polynomial phase timer is invalid')
        equal(row['polynomial_fit_cpu_seconds'], row['polynomial_interior_fit_cpu_seconds']+row['terminal_polynomial_fit_cpu_seconds'],
              'Polynomial terminal/interior fitting costs do not sum')
        equal(row['common_policy_and_outer_preparation_cpu_seconds'], common,
              'The polynomial comparator includes original hinge preparation in its common cost')
        pricing = (common+result['shared_value_stream_and_target_cpu_seconds']+
                   row['polynomial_fit_cpu_seconds']+row['dual_assessment_cpu_seconds'])
        equal(row['standalone_polynomial_pricing_cpu_seconds'], pricing, 'The standalone polynomial pricing budget differs')
        equal(row['standalone_polynomial_pricing_plus_validation_cpu_seconds'],
              pricing+result['stream_and_policy_reconstruction_cpu_seconds'],
              'The polynomial comparator omits or duplicates reconstruction checks')
        equal(row['hybrid_setup_pricing_cpu_seconds'],
              common+result['shared_value_stream_and_target_cpu_seconds']+row['polynomial_interior_fit_cpu_seconds'],
              'The hybrid setup includes a redundant terminal polynomial fit')
    return True


def polynomial(directory):
    directory = private_directory(directory)
    design, manifest, _, records, sources, design_sha = registration(directory)
    # All primary comparisons must be complete and independently verified first.
    hinges = []
    for fit, record, retained in zip(design['fits'], records, manifest['fits']):
        folder = directory/fit['directory']
        if not (folder/'result.json').exists() or not (folder/'assessment_completion.json').exists():
            raise ValueError('Complete the verified primary assessments before polynomial comparison')
        hinges.append(assessment_fit(directory, fit, record, retained, design_sha, sources))
    api = polynomial_api()
    completed = []
    for fit, record, retained, hinge in zip(design['fits'], records, manifest['fits'], hinges):
        folder = directory/fit['directory']
        names = ('polynomial_attempt', 'polynomial', 'polynomial_completion', 'polynomial_failure')
        paths = {name: folder/(name+'.json') for name in names}
        context = dict(input_sha256=retained['input_sha256'], hinge_sha256=digest((folder/'result.json').read_bytes()),
                       design_sha256=design_sha, source_sha256=sources,
                       fitting_unit_index=fit['index'], root_seed=fit['seed'])
        if paths['polynomial'].exists() or paths['polynomial_completion'].exists():
            if not all(paths[name].exists() for name in names[:-1]) or paths['polynomial_failure'].exists():
                raise ValueError('A polynomial attempt is incomplete or failed; preserve this version')
            source = paths['polynomial'].read_bytes()
            attempt = json.loads(paths['polynomial_attempt'].read_bytes())
            completion = json.loads(paths['polynomial_completion'].read_bytes())
            if (any(attempt.get(key) != value or completion.get(key) != value for key, value in context.items())
                    or completion['polynomial_sha256'] != digest(source)):
                raise ValueError('Completed polynomial evidence has changed')
            validate_cost(completion['complete_validation_workflow_cost'])
            result = json.loads(source)
            validate_polynomial_result(result, record, hinge, fit, context, api)
            completed.append(result)
            continue
        if paths['polynomial_attempt'].exists() or paths['polynomial_failure'].exists():
            raise ValueError('Do not restart a failed or interrupted polynomial attempt')
        start = clocks()
        write_new(paths['polynomial_attempt'], dict(**context, started_at_utc=datetime.now(timezone.utc).isoformat()))
        try:
            result = api.run(record, hinge, degrees=tuple(fit['polynomial_degree_order']))
            result.update(**context, author='Amanjeet Singh', assessed_at_utc=datetime.now(timezone.utc).isoformat())
            validate_polynomial_result(result, record, hinge, fit, context, api)
            write_new(paths['polynomial'], result)
            source_sha = digest(paths['polynomial'].read_bytes())
            write_new(paths['polynomial_completion'], dict(**context, polynomial_sha256=source_sha,
                completed_at_utc=datetime.now(timezone.utc).isoformat(), complete_validation_workflow_cost=elapsed(start),
                timing_scope='Both polynomial fits, reconstruction and same-path validation, then evidence writing/hash verification; completion-marker serialisation excluded.'))
        except Exception as error:
            if not paths['polynomial_failure'].exists():
                write_new(paths['polynomial_failure'], dict(**context, failed_at_utc=datetime.now(timezone.utc).isoformat(),
                    exception_type=type(error).__name__, message=str(error)))
            raise
        completed.append(result)
        print(f'Completed polynomial comparators for fitting unit {fit["index"]}', flush=True)
    return completed


def verified_polynomial_api():
    # This comparator and its validation use Flint and the standard library.
    import certify_polynomial_dual
    return certify_polynomial_dual


def polynomial_export_without_numpy(folder, context, record, fit, api):
    paths = {name: folder/(name+'.json') for name in
             ('polynomial_attempt', 'polynomial', 'polynomial_completion', 'polynomial_failure')}
    if not all(paths[name].exists() for name in ('polynomial_attempt', 'polynomial', 'polynomial_completion')) or paths['polynomial_failure'].exists():
        raise ValueError('Complete polynomial fitting and validation before verified comparison')
    source = paths['polynomial'].read_bytes()
    result = json.loads(source)
    attempt = json.loads(paths['polynomial_attempt'].read_bytes())
    completion = json.loads(paths['polynomial_completion'].read_bytes())
    if (any(attempt.get(key) != value or result.get(key) != value or completion.get(key) != value
            for key, value in context.items()) or completion['polynomial_sha256'] != digest(source)):
        raise ValueError('The polynomial export or its completion identity changed')
    if list(result['degrees']) != [str(value) for value in fit['polynomial_degree_order']]:
        raise ValueError('The exported polynomial execution order changed')
    validate_cost(completion['complete_validation_workflow_cost'])
    api.validate_export(record, result, tuple(fit['polynomial_degree_order']))
    return result, digest(source)


def validate_verified_polynomial_result(result, record, polynomial, fit, context):
    if any(result.get(key) != value for key, value in context.items()):
        raise ValueError('The verified comparator identifies different inputs, fitting outputs or sources')
    if (result.get('outer_paths') != 256 or result.get('precision_bits') != 96 or
            result.get('observed_policy_decisions_checked') is not True or
            result.get('independent_arm_costs') is not True or result.get('exact_terminal_model_discount') is not True or
            result.get('new_general_theorem_claim') is not False or
            list(result['degrees']) != [str(value) for value in fit['polynomial_degree_order']] or
            rational_interval(result['path_gap_width_target']) != (Fraction(1, 100), Fraction(1, 100))):
        raise ValueError('The verified comparator changes its maintained numerical scope')
    validation_cpu = result['input_and_policy_validation_cpu_seconds']
    if not math.isfinite(validation_cpu) or validation_cpu < 0:
        raise ValueError('Invalid verified-comparator validation cost')
    for degree, exported in result['degrees'].items():
        if list(exported['methods']) != fit['verified_polynomial_method_order']:
            raise ValueError('Both independently evaluated established comparators are required')
        for method, row in exported['methods'].items():
            if row['outer_paths'] != 256 or row['degree'] != int(degree) or row['method'] != method or row['all_targets_met'] is not True:
                raise ValueError('The verified comparator changes an arm or path count')
            observations = row['observations']
            if len(observations) != 256:
                raise ValueError('A verified comparator omits assessment paths')
            intervals = []
            for index, observation in enumerate(observations):
                if type(observation['path_index']) is not int or observation['path_index'] != index or observation['stopping_index'] != record['stopping_indices'][index]:
                    raise ValueError('The verified comparator changes its path or stopping identity')
                lower, upper = rational_interval(observation['signed_gap_certificate'])
                if upper-lower > Fraction(1, 100) or rational_interval(observation['selected_date_gap_certificate']) != (0, 0):
                    raise ValueError('The verified comparator misses its width target or same-date cancellation')
                intervals.append((lower, upper))
            lower = sum(value[0] for value in intervals)/256
            upper = sum(value[1] for value in intervals)/256
            if rational_interval(row['mean_signed_gap_certificate']) != (lower, upper):
                raise ValueError('The verified mean does not aggregate its exact path endpoints')
            midpoint = [float((lo+hi)/2) for lo, hi in intervals]
            for field, expected in (('mean_signed_width', float(upper-lower)),
                    ('maximum_signed_width', float(max(hi-lo for lo, hi in intervals))),
                    ('approximate_midpoint_mean', statistics.mean(midpoint)),
                    ('approximate_midpoint_mean_standard_error', statistics.stdev(midpoint)/math.sqrt(256)),
                    ('midpoint_mean_numerical_error_bound', float((upper-lower)/2))):
                equal(row[field], expected, 'The verified midpoint/width statistic differs: '+field)
            for field in ('conditional_integrals', 'integrand_evaluations', 'polynomial_terms_evaluated',
                          'moment_factor_exponentials', 'moment_factor_cache_builds',
                          'moment_factor_cache_hits', 'moment_factors_reused'):
                values = [observation[field] for observation in observations]
                if any(type(value) is not int or value < 0 for value in values) or row[field] != sum(values):
                    raise ValueError('The verified comparator omits charged numerical work')
            assessment_cpu = row['assessment_cpu_seconds']
            if not math.isfinite(assessment_cpu) or assessment_cpu < 0:
                raise ValueError('Invalid verified-comparator assessment cost')
            equal(row['standalone_assessment_cpu_seconds_including_validation'], assessment_cpu+validation_cpu,
                  'The independently charged comparator omits validation')
            ordinary = polynomial['degrees'][degree]
            setup = ordinary['hybrid_setup_pricing_cpu_seconds'] if method == 'hybrid_exact_terminal' else (
                ordinary['standalone_polynomial_pricing_cpu_seconds']-ordinary['dual_assessment_cpu_seconds'])
            equal(row['retained_setup_pricing_cpu_seconds'], setup, 'The verified comparator changes its retained setup charge')
            equal(row['standalone_pricing_cpu_seconds'], setup+assessment_cpu,
                  'The verified comparator omits or duplicates pricing setup')
            equal(row['standalone_verified_workflow_cpu_seconds'], setup+assessment_cpu+validation_cpu,
                  'The verified workflow omits or duplicates input/policy verification')
    return True


def certify_polynomial(directory):
    directory = private_directory(directory)
    design, manifest, _, records, sources, design_sha = registration(directory)
    api = verified_polynomial_api()
    completed = []
    for fit, record, retained in zip(design['fits'], records, manifest['fits']):
        folder = directory/fit['directory']
        if not (folder/'result.json').exists() or not (folder/'assessment_completion.json').exists():
            raise ValueError('Complete primary hinge assessments before verified comparison')
        assessment_fit(directory, fit, record, retained, design_sha, sources)
        original_context = dict(input_sha256=retained['input_sha256'], hinge_sha256=digest((folder/'result.json').read_bytes()),
                                design_sha256=design_sha, source_sha256=sources,
                                fitting_unit_index=fit['index'], root_seed=fit['seed'])
        polynomial_record, polynomial_sha = polynomial_export_without_numpy(folder, original_context, record, fit, api)
        context = dict(**original_context, polynomial_sha256=polynomial_sha,
                       verified_method_order=fit['verified_polynomial_method_order'])
        paths = {name: folder/(name+'.json') for name in ('verified_polynomial_attempt', 'verified_polynomial',
                                                         'verified_polynomial_completion', 'verified_polynomial_failure')}
        if paths['verified_polynomial'].exists() or paths['verified_polynomial_completion'].exists():
            if not all(paths[name].exists() for name in ('verified_polynomial_attempt', 'verified_polynomial', 'verified_polynomial_completion')) or paths['verified_polynomial_failure'].exists():
                raise ValueError('A verified-polynomial attempt is incomplete or failed; preserve this version')
            source = paths['verified_polynomial'].read_bytes()
            attempt = json.loads(paths['verified_polynomial_attempt'].read_bytes())
            completion = json.loads(paths['verified_polynomial_completion'].read_bytes())
            if (any(attempt.get(key) != value or completion.get(key) != value for key, value in context.items())
                    or completion['verified_polynomial_sha256'] != digest(source)):
                raise ValueError('Completed verified-polynomial evidence changed')
            validate_cost(completion['complete_validation_workflow_cost'])
            result = json.loads(source)
            validate_verified_polynomial_result(result, record, polynomial_record, fit, context)
            completed.append(result)
            continue
        if paths['verified_polynomial_attempt'].exists() or paths['verified_polynomial_failure'].exists():
            raise ValueError('Do not silently restart a failed or interrupted verified-polynomial attempt')
        start = clocks()
        write_new(paths['verified_polynomial_attempt'], dict(**context, started_at_utc=datetime.now(timezone.utc).isoformat()))
        try:
            result = api.assess(record, polynomial_record, target=Fraction(1, 100), precision=96,
                                degrees=tuple(fit['polynomial_degree_order']),
                                methods=tuple(fit['verified_polynomial_method_order']))
            result.update(**context, author='Amanjeet Singh', assessed_at_utc=datetime.now(timezone.utc).isoformat())
            validate_verified_polynomial_result(result, record, polynomial_record, fit, context)
            write_new(paths['verified_polynomial'], result)
            source_sha = digest(paths['verified_polynomial'].read_bytes())
            write_new(paths['verified_polynomial_completion'], dict(**context, verified_polynomial_sha256=source_sha,
                completed_at_utc=datetime.now(timezone.utc).isoformat(), complete_validation_workflow_cost=elapsed(start),
                timing_scope='Independent verified arms, same-path aggregation checks and artifact writing/hash verification; completion-marker serialisation excluded.'))
        except Exception as error:
            if not paths['verified_polynomial_failure'].exists():
                write_new(paths['verified_polynomial_failure'], dict(**context, failed_at_utc=datetime.now(timezone.utc).isoformat(),
                    exception_type=type(error).__name__, message=str(error)))
            raise
        completed.append(result)
        print(f'Completed verified polynomial comparators for fitting unit {fit["index"]}', flush=True)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'assess', 'polynomial', 'certify-polynomial'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': preparation, 'assess': assessment, 'polynomial': polynomial,
     'certify-polynomial': certify_polynomial}[args.mode](args.directory)


if __name__ == '__main__':
    main()
