"""Record and assess a fresh 256-path public-model gap refinement design.

Run preparation in the NumPy research environment and assessment in the
separate python-flint environment. Four ordered cases share the same frozen
values and policy: targets 0.1, 0.01, 0.001, then a reversed-order 0.01 timing
repeat. Existing inputs, attempts and outcomes are never overwritten.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

from certify_multivariate_gap import load_fraction, prepare
from refine_multivariate_gap import cold_uniform, compare, intersect

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    'scripts/study_multivariate_gap_refinement.py', 'scripts/refine_multivariate_gap.py',
    'scripts/certify_multivariate_gap.py', 'scripts/certify_max_hinge.py',
    'src/qf_research/optimal_stopping/multivariate_benchmark.py',
    'src/qf_research/optimal_stopping/contracts.py',
)
CASES = (
    dict(name='target_0p1', target='1/10', order=['cold_uniform', 'adaptive']),
    dict(name='target_0p01', target='1/100', order=['cold_uniform', 'adaptive']),
    dict(name='target_0p001', target='1/1000', order=['cold_uniform', 'adaptive']),
    dict(name='target_0p01_reversed_order', target='1/100', order=['adaptive', 'cold_uniform']),
)


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def write_new(path, record):
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')


def frozen_model_hash(record):
    selected = {key: record[key] for key in ('model', 'knots', 'coefficients', 'frozen_policy')}
    return hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def preparation(directory):
    if directory.exists():
        raise ValueError('Fresh preparation requires a new private study directory')
    directory.mkdir(parents=True)
    design = dict(schema_version=1, author='Amanjeet Singh',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        mathematical_benchmark='Andersen and Broadie (2004), Table 2, d=5, S=K=100, r=.05, q=.10, sigma=.20, correlation zero, T=3, dates iT/9.',
        dimension=5, outer_paths=256, training_paths=50000, value_paths=50000, root_seed=2026100408,
        precision_bits=96, coarse_conditional_width='1', cases=CASES,
        policy='Frozen Longstaff-Schwartz, centred sorted full-state degree-2 polynomial basis.',
        value_functions='Frozen hinge regression functions trained on a separate stream.',
        centres='Adaptive centres fixed from tolerance-1 brackets at the preceding state. Cold uniform computes fine centres without coarse initialisation. Both compensate the same ideal v minus conditional expectation increments.',
        accuracy_comparison='Same required maximum path-gap width, not the same achieved width.',
        timing_scope='CPU measured at function entry to derived result, excluding JSON serialisation; validation reported separately. Cold uniform means no coarse phase, not a cold operating-system cache.',
        order_repeat='The final 0.01 case reverses strategy order and changes no financial target, path, value function or policy.',
        sampling_scope='Midpoint standard errors are approximate sampling diagnostics; neither simulator law nor statistical coverage is certified.',
        source_sha256=source_hashes())
    design_path = directory/'design.json'
    write_new(design_path, design)
    record = prepare(dimension=design['dimension'], training_count=design['training_paths'],
        value_count=design['value_paths'], evaluation_count=design['outer_paths'], seed=design['root_seed'])
    record['preparation_design_sha256'] = hashlib.sha256(design_path.read_bytes()).hexdigest()
    record['preparation_source_sha256'] = source_hashes()
    write_new(directory/'inputs_fresh256.json', record)
    print(f'Saved fresh preparation and prior design to {directory}', flush=True)


def decode_interval(record):
    return load_fraction(record['lower']), load_fraction(record['upper'])


def case_result(record, case, *, precision, coarse_tolerance):
    snapshot = frozen_model_hash(record)
    cpu, wall = time.process_time(), time.perf_counter()
    strategies = {}
    observations = {}
    for name in case['order']:
        if name == 'cold_uniform':
            result = cold_uniform(record, target=Fraction(case['target']), precision=precision)
            statistics = result['strategy']
            elapsed = result['actual_assessment_cpu_seconds']
            rows = result['observations']
        else:
            result = compare(record, target=Fraction(case['target']),
                             coarse_tolerance=Fraction(coarse_tolerance), precision=precision,
                             include_uniform=False)
            statistics = result['strategies']['adaptive']
            elapsed = result['actual_experiment_cpu_seconds']
            rows = result['observations']['adaptive']
        validation = result['input_and_policy_validation_cpu_seconds']
        statistics.update(assessment_cpu_seconds=elapsed,
                          input_and_policy_validation_cpu_seconds=validation,
                          certificate_phase_cpu_seconds=elapsed-validation,
                          combined_preparation_and_assessment_cpu_seconds=record['preparation_cost']['cpu_seconds']+elapsed)
        strategies[name] = statistics
        observations[name] = rows
        if frozen_model_hash(record) != snapshot:
            raise AssertionError('A strategy changed the frozen model, value functions or policy')
    for one, two in zip(observations['cold_uniform'], observations['adaptive']):
        if one['path_index'] != two['path_index'] or one['stopping_index'] != two['stopping_index']:
            raise AssertionError('Strategies evaluated different paths or stopping rules')
        intersect(decode_interval(one['signed_gap_certificate']), decode_interval(two['signed_gap_certificate']))
    adaptive, uniform = strategies['adaptive'], strategies['cold_uniform']
    return dict(name=case['name'], required_maximum_path_gap_width=case['target'], strategy_order=case['order'],
        frozen_model_policy_values_sha256=snapshot, outer_paths=len(record['states']), precision_bits=precision,
        strategies=strategies, all_path_certificates_overlap=True,
        adaptive_over_cold_uniform_integral_ratio=adaptive['conditional_integrals']/uniform['conditional_integrals'],
        adaptive_over_cold_uniform_certificate_phase_cpu_ratio=adaptive['certificate_phase_cpu_seconds']/uniform['certificate_phase_cpu_seconds'],
        actual_case_cpu_seconds=time.process_time()-cpu, actual_case_wall_seconds=time.perf_counter()-wall,
        observations=observations)


def assessment(directory):
    design_source = (directory/'design.json').read_bytes()
    input_source = (directory/'inputs_fresh256.json').read_bytes()
    design, record = json.loads(design_source), json.loads(input_source)
    if record['preparation_design_sha256'] != hashlib.sha256(design_source).hexdigest():
        raise ValueError('The prior design has changed since preparation')
    if (len(record['states']) != design['outer_paths'] or record['design']['root_seed'] != design['root_seed']
            or record['design']['training_paths'] != design['training_paths']
            or record['design']['value_paths'] != design['value_paths']):
        raise ValueError('Prepared path counts or seed differ from the prior design')
    if any((directory/(case['name']+'.json')).exists() for case in design['cases']):
        raise ValueError('Use a new preparation directory rather than overwrite any attempt')
    input_hash = hashlib.sha256(input_source).hexdigest()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    cases, centre_hashes = [], None
    for case in design['cases']:
        result = case_result(record, case, precision=design['precision_bits'],
                             coarse_tolerance=design['coarse_conditional_width'])
        result.update(schema_version=1, author='Amanjeet Singh',
                      assessed_at_utc=datetime.now(timezone.utc).isoformat(), input_sha256=input_hash,
                      design_sha256=hashlib.sha256(design_source).hexdigest(), source_sha256=source_hashes(),
                      preparation_cost=record['preparation_cost'],
                      scope='Exact rational gap enclosures for ideal conditional functions at supplied binary states; sampling standard errors and simulator law are not computer-certified.',
                      accuracy_comparison='Same required maximum path width; achieved widths differ.',
                      budget_scope='Preparation is included in each standalone case budget. It is performed once and is charged once in the whole-study strategy budgets.')
        current_hashes = [row['frozen_centres_and_penalty_sha256'] for row in result['observations']['adaptive']]
        if centre_hashes is not None and current_hashes != centre_hashes:
            raise AssertionError('Adaptive coarse centres or penalty changed between targets')
        centre_hashes = current_hashes
        write_new(directory/(case['name']+'.json'), result)
        cases.append({key: value for key, value in result.items() if key != 'observations'})
        print(f'Completed {case["name"]}: cold uniform {result["strategies"]["cold_uniform"]["conditional_integrals"]} integrals, adaptive {result["strategies"]["adaptive"]["conditional_integrals"]}', flush=True)
    budget = {name: record['preparation_cost']['cpu_seconds']+
              sum(case['strategies'][name]['assessment_cpu_seconds'] for case in cases)
              for name in ('cold_uniform', 'adaptive')}
    result = dict(schema_version=1, author='Amanjeet Singh', input_sha256=input_hash,
        design_sha256=hashlib.sha256(design_source).hexdigest(), source_sha256=source_hashes(),
        preparation_cost=record['preparation_cost'],
        frozen_model_policy_values_sha256=frozen_model_hash(record), outer_paths=len(record['states']),
        cases=cases, adaptive_coarse_centres_identical_across_targets=True,
        actual_assessment_cpu_seconds=time.process_time()-cpu_start,
        actual_assessment_wall_seconds=time.perf_counter()-wall_start,
        actual_preparation_plus_whole_assessment_cpu_seconds=record['preparation_cost']['cpu_seconds']+time.process_time()-cpu_start,
        strategy_whole_study_budget_cpu_seconds_including_preparation_once=budget,
        interpretation='Same required maximum width, not equal achieved precision. Two 0.01 passes check ordering sensitivity of timings, not statistical significance of a cost difference.')
    write_new(directory/'summary.json', result)
    size = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    if size >= 20*1024**2:
        raise ArithmeticError('Retained study exceeds the authorised 20 MiB limit')
    print(f'Saved study summary; retained bytes {size}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'assess'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Retain this research study privately')
    (preparation if args.mode == 'prepare' else assessment)(directory)


if __name__ == '__main__':
    main()
