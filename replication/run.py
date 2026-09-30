"""Execute a specified conditional stopping problem using the research kernels.

The public input is a stipulated mathematical example. Authorised financial
inputs use the same interface, with their output confined to private storage.
This experiment does not regenerate the paper's retained historical tables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURE = HERE/'configs/stipulated_fixture.json'
_verifier_spec = importlib.util.spec_from_file_location(
    '_replication_source_verifier', HERE/'verify_sources.py')
source_verifier = importlib.util.module_from_spec(_verifier_spec)
_verifier_spec.loader.exec_module(source_verifier)

import numpy as np
import scipy

from qf_research.optimal_stopping.allocation_workload import (
    build_approximation, evaluate_pair, train_pair,
)
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.expiry_extension import extended_calendar
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, _schedule,
)
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.simulation import estimate_samples
from qf_research.optimal_stopping.terminal_repair import evaluate_terminal_repair
from qf_research.optimal_stopping.training_allocation import choose_allocation
from qf_research.optimal_stopping.training_evaluation import replicate_summary
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale,
)

TOP_FIELDS = {'schema_version', 'classification', 'description', 'model', 'option',
              'decisions', 'experiment'}
EXPERIMENT_FIELDS = {'root_seed', 'training_paths', 'evaluation_paths', 'repetitions',
                    'degree', 'minimum_itm_paths', 'condition_limit', 'approximation',
                    'terminal_extension'}


def read_json(path):
    return source_verifier.read_json_bytes(Path(path).read_bytes())


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer of at least {minimum}')


def validate(specification):
    """Validate the input contract before simulation or output creation."""
    s = specification
    if (not isinstance(s, dict) or set(s) != TOP_FIELDS
            or type(s['schema_version']) is not int or s['schema_version'] != 1):
        raise ValueError('Input fields or schema version differ from the documented contract')
    if s['classification'] not in ('stipulated_mathematical_fixture', 'authorised_model'):
        raise ValueError('A stipulated fixture or authorised model classification is required')
    if not isinstance(s['description'], str) or not s['description'].strip():
        raise ValueError('A description of the model inputs is required')
    # This label is reserved for the precise published example. Changes to its
    # financial inputs must use a separately described authorised specification.
    if s['classification'] == 'stipulated_mathematical_fixture' and s != read_json(FIXTURE):
        raise ValueError('The fixture label is reserved for the published specification')
    if not isinstance(s['model'], dict) or set(s['model']) != {
            'quoted_spot', 'volatility', 'initial_share_factor', 'premium_factor', 'dividends'}:
        raise ValueError('Unexpected or missing model fields')
    if not isinstance(s['option'], dict) or set(s['option']) != {'strike', 'maturity', 'kind'}:
        raise ValueError('Unexpected or missing option fields')
    for name in ('strike', 'maturity'):
        value = s['option'][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Option {name} must be a finite positive number')
    if s['option']['kind'] != 'put':
        raise ValueError('The published value approximation supports puts')
    if not isinstance(s['decisions'], list) or not s['decisions']:
        raise ValueError('An explicit exercise calendar is required')
    if any(not isinstance(d, dict) or set(d) != {'time', 'settlement_factor', 'phase'}
           for d in s['decisions']):
        raise ValueError('Each decision must explicitly specify time, settlement factor and phase')
    model_data = dict(s['model'])
    model_data['dividends'] = tuple(DividendCap(**d) for d in model_data['dividends'])
    model = FundedCashModel(**model_data)
    option = Option(**s['option'])
    decisions = tuple(FundedExercise(**d) for d in s['decisions'])
    rights, _ = _schedule(model, option, decisions)
    if rights != decisions:
        raise ValueError('Exercise decisions must be chronologically ordered')
    e = s['experiment']
    if not isinstance(e, dict) or set(e) != EXPERIMENT_FIELDS:
        raise ValueError('Unexpected or missing experiment fields')
    integer(e['root_seed'], 'root_seed', 0)
    integer(e['repetitions'], 'repetitions', 2)
    integer(e['degree'], 'degree', 0)
    integer(e['minimum_itm_paths'], 'minimum_itm_paths', 2)
    for key in ('training_paths', 'evaluation_paths'):
        values = e[key]
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f'{key} requires at least two increasing counts')
        for n in values:
            integer(n, key, 2)
        if values != sorted(set(values)):
            raise ValueError(f'{key} must increase strictly')
    if len(e['training_paths']) != 2:
        raise ValueError('Exactly two training counts are required')
    if (isinstance(e['condition_limit'], bool) or not math.isfinite(e['condition_limit'])
            or e['condition_limit'] < 1):
        raise ValueError('A finite regression condition limit of at least one is required')
    approximation = e['approximation']
    if not isinstance(approximation, dict) or set(approximation) != {'core_steps', 'wing_steps', 'tolerance'}:
        raise ValueError('Unexpected or missing approximation fields')
    integer(approximation['core_steps'], 'core_steps', 4)
    integer(approximation['wing_steps'], 'wing_steps', 2)
    if (isinstance(approximation['tolerance'], bool)
            or not math.isfinite(approximation['tolerance']) or approximation['tolerance'] < 0):
        raise ValueError('A finite non-negative compression tolerance is required')
    extended_calendar(model, option, decisions, (e['terminal_extension'],))
    return model, option, decisions


def load_specification(path):
    result = read_json(path)
    validate(result)
    return result


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def source_preflight(specification):
    required = tuple(path.relative_to(ROOT).as_posix() for path in
                     (Path(__file__).resolve(), HERE/'verify_sources.py', FIXTURE))
    verification = source_verifier.verify_sources(ROOT, required_paths=required)
    source_verifier.check_import_origins(verification, ROOT)
    validate(specification)
    return dict(verification=verification,
                input_specification_sha256=canonical_digest(specification))


def execution_provenance(specification, before):
    verification = source_verifier.verify_sources(ROOT)
    if verification != before['verification']:
        raise ValueError('Source manifest or listed sources changed during execution')
    if canonical_digest(specification) != before['input_specification_sha256']:
        raise ValueError('Input specification changed during execution')
    origins = source_verifier.check_import_origins(verification, ROOT)
    return dict(input_specification_sha256=before['input_specification_sha256'],
                source_manifest_sha256=verification['source_manifest_sha256'],
                verified_source_files=verification['verified_source_files'],
                project_module_origins=origins, scope=verification['scope'])


def _run_numerics(specification):
    """Run fixed counts; return aggregate results and separate measured costs.

    Counts never depend on realised costs or gains. The allocation calculation
    is a pilot diagnostic and does not select or launch another experiment.
    """
    model, option, decisions = validate(specification)
    e = specification['experiment']
    base = SimpleNamespace(model=model, option=option, decisions=decisions,
                           training_stream_id='configuration-only-no-training')
    approximation, setup = build_approximation(base, **e['approximation'])
    rows, timings, streams = [], [], set()
    for repetition in range(e['repetitions']):
        def seed(role, index=0):
            return np.random.SeedSequence(e['root_seed'], spawn_key=(repetition, role, index))
        policies, training = train_pair(base, tuple(e['training_paths']),
            degree=e['degree'], minimum_itm_paths=e['minimum_itm_paths'],
            condition_limit=e['condition_limit'], seeds=(seed(0), seed(1)))
        identifiers = [row['stream_id'] for row in training['streams']]
        for index, count in enumerate(e['evaluation_paths']):
            evaluation = evaluate_pair(policies, approximation, count, seed(2, index))
            identifiers.append(evaluation['stream']['stream_id'])
            rows.append(dict(repetition=repetition, evaluation_paths=count,
                             gain=evaluation['gain'],
                             minimum_gap_quote=evaluation['minimum_gap_quote']))
            timings.append(dict(repetition=repetition, evaluation_paths=count,
                training_cpu_seconds=training['cpu_seconds'],
                evaluation_cpu_seconds=evaluation['cpu_seconds'],
                pair_cpu_seconds=training['cpu_seconds']+evaluation['cpu_seconds'],
                pair_elapsed_seconds=training['elapsed_seconds']+evaluation['elapsed_seconds']))
        if len(set(identifiers)) != len(identifiers) or streams.intersection(identifiers):
            raise ArithmeticError('A random stream was reused')
        streams.update(identifiers)
    summaries = []
    for count in e['evaluation_paths']:
        estimates = [r['gain']['estimate'] for r in rows if r['evaluation_paths'] == count]
        summaries.append(dict(evaluation_paths=count, **replicate_summary(estimates)))
    reference = summaries[-1]
    component_a = reference['training_variance_moment_estimate']
    component_b = reference['mean_conditional_evaluation_variance']*e['evaluation_paths'][-1]
    costs = {n: median(r['pair_cpu_seconds'] for r in timings if r['evaluation_paths'] == n)
             for n in e['evaluation_paths']}
    allocation = choose_allocation(component_a, component_b, setup['cpu_seconds'], costs,
        reference_repetitions=e['repetitions'], reference_paths=e['evaluation_paths'][-1],
        min_repetitions=2, max_repetitions=4*e['repetitions'])

    # One frozen policy is retained solely for a separate terminal comparison.
    extended_option, extended_rights = extended_calendar(model, option, decisions,
                                                         (e['terminal_extension'],))
    extended_values = build_values(model, extended_option, extended_rights, **e['approximation'])
    terminal_seed = np.random.SeedSequence(e['root_seed'], spawn_key=(e['repetitions'], 3, 0))
    paths = simulate_paths(model, extended_option, observation_rights(extended_values),
                           e['evaluation_paths'][-1], np.random.default_rng(terminal_seed))
    if paths.stream_id in streams:
        raise ArithmeticError('Terminal evaluation reuses a previous stream')
    streams.add(paths.stream_id)
    terminal = evaluate_terminal_repair(policies[-1], value_martingale(extended_values, paths),
                                        e['terminal_extension'])
    discrepancy = (terminal.original.gap_payoffs-terminal.repaired.gap_payoffs
                   -terminal.integrated_gain)
    numerics = dict(pair_results=rows, training_evaluation=summaries,
        variance_components=dict(training=component_a, evaluation=component_b,
                                 negative_training_component_truncated=False),
        terminal_adjustment=dict(
            integrated_gain=asdict(estimate_samples(terminal.integrated_gain/model.premium_factor)),
            realised_gain=asdict(estimate_samples(terminal.realised_gain/model.premium_factor)),
            surviving_fraction=float(terminal.survives.mean()),
            maximum_common_upper_identity_error=float(np.max(np.abs(discrepancy))),
            minimum_integrated_gain=float(terminal.integrated_gain.min()),
            maximum_roundoff_correction=terminal.maximum_roundoff_correction),
        independent_stream_identifiers=len(streams))
    return dict(schema_version=1, classification=specification['classification'],
        scope='Conditional mathematical experiment; no historical-table reproduction or market validation',
        design=dict(training_paths=e['training_paths'], evaluation_paths=e['evaluation_paths'],
                    repetitions=e['repetitions'], degree=e['degree'], counts_fixed_before_execution=True),
        numerics=numerics, numerical_sha256=canonical_digest(numerics),
        costs=dict(setup=setup, pair_timings=timings, allocation_pilot=allocation,
                   selected_counts_executed=False, timing_guarantee=False,
                   combined_timing_excludes_terminal_diagnostic=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            thread_environment={name: os.environ.get(name) for name in
                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}))


def run(specification):
    """Execute the conditional study with source and input provenance checks."""
    before = source_preflight(specification)
    result = _run_numerics(specification)
    result['provenance'] = execution_provenance(specification, before)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=FIXTURE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    specification = read_json(args.config)
    output = args.output.resolve()
    # Keeping even authorised derived outputs private avoids silently exporting
    # an unreviewed model bundle through a convenience command.
    if not output.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Output must remain under data/private until separately reviewed')
    if output.exists():
        raise FileExistsError('Use a fresh output file')
    if args.validate_only:
        before = source_preflight(specification)
        result = dict(valid=True, schema_version=1, classification=specification['classification'],
                      provenance=execution_provenance(specification, before))
    else:
        result = run(specification)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Validated input' if args.validate_only else 'Completed conditional experiment')


if __name__ == '__main__':
    main()
