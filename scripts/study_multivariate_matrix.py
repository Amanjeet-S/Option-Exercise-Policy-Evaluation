"""Retain aggregate public-model benchmark prices and an exercise-margin grid.

The fixed design is saved before outcomes. Identical seeds are retained across
the model cases, so the case estimates are not treated as independent evidence.
No simulated path arrays, fitted coefficients or provider observations are saved.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from qf_research.optimal_stopping.multivariate_benchmark import (
    fit_max_call_policy, published_max_call, simulate_max_call, sorted_polynomial_basis,
)
from validate_multivariate_benchmark import ROOT, SOURCE, validate


CASES = ((1, 100.), *((dimension, spot) for dimension in (2, 3, 5) for spot in (90., 100., 110.)))


def exercise_margin_grid(policy, index=3, *, lower=50., upper=180., points=41):
    """Full-state reward-minus-continuation surface for a frozen two-asset rule."""
    if policy.model.dimension != 2 or not 0 <= index < len(policy.rules)-1:
        raise ValueError('Require a two-asset nonterminal exercise right')
    if points < 2 or not 0 < lower < upper:
        raise ValueError('Require a positive, nonempty state-grid domain')
    rule = policy.rules[index]
    if not rule.coefficients:
        raise ValueError('The selected date has no fitted continuation regression')
    axis = np.linspace(lower, upper, points)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    states = np.column_stack((x.ravel(), y.ravel()))
    design = sorted_polynomial_basis(states, policy.model.strike, policy.degree)
    swapped = sorted_polynomial_basis(states[:, ::-1], policy.model.strike, policy.degree)
    if not np.array_equal(design, swapped):
        raise ArithmeticError('Full-state basis is not permutation invariant')
    continuation = np.maximum(design @ rule.coefficients, 0.)
    counterpart = np.maximum(swapped @ rule.coefficients, 0.)
    discrepancy = float(np.max(np.abs(continuation-counterpart)))
    reward = policy.model.discounted_payoff(states, index)
    margin = reward-continuation
    rows = [dict(asset_1=float(state[0]), asset_2=float(state[1]),
        discounted_exercise_reward=float(h), discounted_continuation_estimate=float(c),
        exercise_margin=float(m), exercise_selected=bool(m > 0.))
        for state, h, c, m in zip(states, reward, continuation, margin)]
    metadata = dict(time_index=index, time=policy.model.exercise_times[index],
        state_grid_lower=lower, state_grid_upper=upper, points_per_asset=points,
        grid_rows=len(rows), currency='Present-value units at time zero',
        permutation_invariance_maximum_absolute_discrepancy=discrepancy,
        exercise_grid_rows=sum(row['exercise_selected'] for row in rows),
        description='Positive margins exercise; zero margins continue. This is a fitted-policy '
            'surface, not an optimal exercise boundary. Grid states can lie outside the dense '
            'training region, where polynomial extrapolation may be inaccurate.')
    return rows, metadata


def run_matrix(output, *, seed=20261004, training_paths=50000,
               evaluation_paths=200000, chunk_size=20000, degree=2):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Benchmark outputs must remain private pending review')
    if (output/'design.json').exists() or (output/'results.json').exists():
        raise FileExistsError('Use a fresh matrix design')
    output.mkdir(parents=True, exist_ok=True)
    design = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        financial_model_source=SOURCE, cases=[dict(dimension=d, spot=s) for d, s in CASES],
        root_seed=seed, seed_unchanged_from_initial_engineering_validation=True,
        training_paths=training_paths, evaluation_paths=evaluation_paths,
        chunk_size=chunk_size, degree=degree, training_evaluation_seed_children=[0, 1],
        case_independence_claim=False, model_cases_share_seed_layout=True,
        training_independent_of_evaluation_within_case=True,
        surface_case=dict(dimension=2, spot=100., time_index=3, lower=50., upper=180., points=41),
        classification='New simulations with published public mathematical inputs; no market observations.',
        pde_reference='Separate one-asset numerical refinement, not a certified price enclosure.')
    (output/'design.json').write_text(json.dumps(design, indent=2, sort_keys=True)+'\n')
    started = time.perf_counter()
    records, table = [], []
    for dimension, spot in CASES:
        record = validate(dimension, spot, training_paths, evaluation_paths, degree, seed, chunk_size)
        records.append(record)
        row = dict(dimension=dimension, spot=spot, training_paths=training_paths,
                   evaluation_paths=evaluation_paths, degree=degree)
        for key, estimate in record['estimates'].items():
            row[key+'_mean'] = estimate['mean']
            row[key+'_standard_error'] = estimate['standard_error']
            row[key+'_lower'] = estimate['pointwise_student_interval'][0]
            row[key+'_upper'] = estimate['pointwise_student_interval'][1]
        row.update(european_quadrature=record['independent_european_reference']['value'],
            terminal_reference_standardised_error=record['terminal_reference_standardised_error'],
            early_exercise_fraction=record['early_exercise_fraction'],
            training_seconds=record['measured_elapsed_seconds']['training'],
            evaluation_seconds=record['measured_elapsed_seconds']['evaluation'])
        interval = record['financial_model_source']['published_bermudan_interval']
        row['published_bermudan_lower'] = interval[0] if interval else None
        row['published_bermudan_upper'] = interval[1] if interval else None
        table.append(row)
        print(f'Completed public model d={dimension}, S={spot:g}', flush=True)
    model = published_max_call(2, 100.)
    training_seed = np.random.SeedSequence(seed).spawn(2)[0]
    training = simulate_max_call(model, training_paths, np.random.default_rng(training_seed))
    policy = fit_max_call_policy(training, degree)
    del training
    surface, surface_metadata = exercise_margin_grid(policy)
    with (output/'state_margin.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(surface[0]))
        writer.writeheader(); writer.writerows(surface)
    with (output/'benchmark_prices.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    result = dict(schema_version=1, design=design, cases=records,
        state_surface=surface_metadata, total_elapsed_seconds=time.perf_counter()-started,
        maximum_absolute_terminal_standardised_error=max(abs(row['terminal_reference_standardised_error'])
                                                        for row in table),
        maximum_matched_gain_mean_cancellation_absolute_error=max(
            r['matched_gain_mean_cancellation_absolute_error'] for r in records),
        scope='Independent out-of-sample evaluation of a fixed Longstaff-Schwartz policy per model. '
            'The sorted full-state polynomial basis differs from the published Andersen-Broadie basis '
            'and no dual upper bound is calculated. Published intervals supply context, not known '
            'optimal values. The policy intervals and European quadrature are not certified.',
        path_arrays_retained=False,
        implementation_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), ROOT/'scripts/validate_multivariate_benchmark.py',
                         ROOT/'src/qf_research/optimal_stopping/multivariate_benchmark.py')})
    (output/'results.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print('Saving the fixed public-model design before running ten benchmark cases.', flush=True)
    result = run_matrix(args.output)
    print(f'Aggregate matrix and state surface saved to {args.output.resolve()}')
