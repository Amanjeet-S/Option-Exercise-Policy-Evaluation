"""Established polynomial regression-later dual baselines on frozen paths.

Glasserman and Yu (2004), 'Simulation for American options: regression now
or regression later?', discuss regression-later constructions with tractable
conditional basis moments, doi:10.1007/978-3-642-18743-8_12. Martingale dual
valuation follows Rogers (2002) and Haugh and Kogan (2004). This script adds
no priority claim and does not reproduce any paper's complete algorithm.

All total-degree-two or three monomials use unsorted coordinates S_i/100.
For an exponent vector a and GBM covariance C_ik=sigma_i sigma_k rho_ik,
E[prod_i (S'_i/100)^a_i | S] equals prod_i (S_i/100)^a_i times
exp(dt * [sum_i a_i(r-q_i-sigma_i^2/2) + a' C a/2]).
Finite-horizon lognormal states have every finite polynomial moment. Frozen
finite coefficients therefore give integrable martingale increments. Fitted
values are never clipped: clipping would require different expectations.

Every date, including terminal payoff, is fitted by a polynomial. Equality
v_T=H_T is unnecessary for martingale validity but approximation can enlarge
dual slack. Reported gaps upper-bound policy suboptimality only in expectation
under the ideal model; realised gaps also contain unknown dual slack.
The moment formula is analytical, but evaluation and least squares here use
ordinary floating point. No numerical or statistical coverage is certified.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations_with_replacement
import json
import math
from pathlib import Path
import time

import numpy as np

from qf_research.optimal_stopping.multivariate_benchmark import (
    MaxCallModel, simulate_max_call, sorted_polynomial_basis,
)

ROOT = Path(__file__).resolve().parents[1]


def exponent_vectors(dimension, degree):
    if type(dimension) is not int or dimension < 1 or type(degree) is not int or not 0 <= degree <= 3:
        raise ValueError('Positive dimension and degree zero to three required')
    exponents = [tuple([0]*dimension)]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(dimension), order):
            powers = [0]*dimension
            for i in indices:
                powers[i] += 1
            exponents.append(tuple(powers))
    return tuple(exponents)


def polynomial_basis(states, degree, *, scale=100.):
    states = np.asarray(states, dtype=float)
    if (states.ndim != 2 or not states.shape[1] or not np.all(np.isfinite(states))
            or np.any(states <= 0) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Finite positive states and scale required')
    powers = exponent_vectors(states.shape[1], degree)
    normalised = states/scale
    with np.errstate(over='raise', invalid='raise'):
        result = np.column_stack([np.prod(normalised**np.asarray(a), axis=1) for a in powers])
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('Polynomial basis overflowed')
    return result


def conditional_basis(states, model, step, degree, *, scale=100.):
    """Analytical multivariate lognormal moments, evaluated in floating point."""
    if not isinstance(model, MaxCallModel) or not math.isfinite(step) or step <= 0:
        raise ValueError('A validated GBM model and positive finite step required')
    basis = polynomial_basis(states, degree, scale=scale)
    if np.asarray(states).shape[1] != model.dimension:
        raise ValueError('State dimension differs from the model')
    powers = np.asarray(exponent_vectors(model.dimension, degree), dtype=float)
    volatility = np.asarray(model.volatilities)
    covariance = np.outer(volatility, volatility)*np.asarray(model.correlation)
    drift = model.rate-np.asarray(model.dividends)-volatility**2/2
    exponent = powers @ drift + np.einsum('ki,ij,kj->k', powers, covariance, powers)/2
    with np.errstate(over='raise', invalid='raise'):
        result = basis*np.exp(step*exponent)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('Conditional moments overflowed')
    return result


def policy_targets(states, model, frozen_rules):
    """Independent fitting-path cash targets under the same frozen policy."""
    states = np.asarray(states, dtype=float)
    if (states.ndim != 3 or states.shape[1:] != (len(model.exercise_times), model.dimension)
            or len(frozen_rules) != len(model.exercise_times) or frozen_rules[-1]):
        raise ValueError('Complete fitting paths and frozen terminal-exercise policy required')
    cash = model.discounted_payoff(states[:, -1], len(model.exercise_times)-1)
    targets = [None]*len(model.exercise_times)
    targets[-1] = cash.copy()
    expected = len(exponent_vectors(model.dimension, 2))
    for j in range(len(model.exercise_times)-2, -1, -1):
        rule = np.asarray(frozen_rules[j], dtype=float)
        if rule.size:
            if rule.shape != (expected,) or not np.all(np.isfinite(rule)):
                raise ValueError('Finite frozen degree-two policy coefficients required')
            immediate = model.discounted_payoff(states[:, j], j)
            continuation = np.maximum(sorted_polynomial_basis(states[:, j], model.strike, 2) @ rule, 0.)
            exercise = immediate > continuation
            cash[exercise] = immediate[exercise]
        targets[j] = cash.copy()
    return targets


def fit_values(states, targets, degree):
    """Unconditional regression-later fits, including the terminal payoff."""
    rights = states.shape[1]
    if len(targets) != rights or rights < 2:
        raise ValueError('Matching nontrivial calendar and fitting targets required')
    coefficients = [np.zeros(len(exponent_vectors(states.shape[2], degree)))]
    diagnostics = []
    for j in range(1, rights):
        date_cpu = time.process_time()
        design = polynomial_basis(states[:, j], degree)
        target = np.asarray(targets[j], dtype=float)
        if target.shape != (len(states),) or not np.all(np.isfinite(target)):
            raise ValueError('Finite independent policy-value targets required')
        fitted, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
        if not np.all(np.isfinite(fitted)):
            raise ArithmeticError('Value coefficients must be finite')
        coefficients.append(fitted)
        residual = design @ fitted-target
        diagnostics.append(dict(date_index=j, rank=int(rank), basis_dimension=design.shape[1],
            condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            fitting_rmse=float(np.sqrt(np.mean(residual**2))),
            fit_cpu_seconds=time.process_time()-date_cpu))
    return tuple(coefficients), diagnostics


def dual_paths(states, model, stopping_indices, coefficients, degree):
    states = np.asarray(states, dtype=float)
    stops = np.asarray(stopping_indices)
    count, rights, dimension = states.shape
    if (rights != len(model.exercise_times) or dimension != model.dimension or len(coefficients) != rights
            or stops.shape != (count,) or stops.dtype.kind not in 'iu'
            or np.any(stops < 0) or np.any(stops >= rights)):
        raise ValueError('Matching paths, frozen functions and integer stopping indices required')
    expected = len(exponent_vectors(dimension, degree))
    if any(np.asarray(c).shape != (expected,) or not np.all(np.isfinite(c)) for c in coefficients):
        raise ValueError('Complete finite polynomial coefficients required')
    rewards = np.column_stack([model.discounted_payoff(states[:, j], j) for j in range(rights)])
    martingale = np.zeros((count, rights))
    for j in range(1, rights):
        realised = polynomial_basis(states[:, j], degree) @ coefficients[j]
        conditional = conditional_basis(states[:, j-1], model,
            model.exercise_times[j]-model.exercise_times[j-1], degree) @ coefficients[j]
        martingale[:, j] = martingale[:, j-1]+realised-conditional
    locations = np.arange(count)
    policy = rewards[locations, stops]
    stopped_martingale = martingale[locations, stops]
    differences = rewards-policy[:, None]-martingale+stopped_martingale[:, None]
    differences[locations, stops] = 0.  # Exact same-date cancellation, not clipping fitted values.
    gap = differences.max(axis=1)
    dual = (rewards-martingale).max(axis=1)
    if not np.all(np.isfinite(gap)) or not np.all(np.isfinite(dual)):
        raise ArithmeticError('Non-finite floating-point dual evaluation')
    return dict(gap=gap, dual=dual, policy_reward=policy,
                stopped_martingale=stopped_martingale, martingale=martingale)


def sample_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError('Two finite observations required')
    return dict(sample_count=len(values), mean=float(values.mean()),
        standard_error=float(values.std(ddof=1)/math.sqrt(len(values))),
        minimum=float(values.min()), maximum=float(values.max()))


def fraction_value(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def paired_comparison(polynomial_gaps, hinge_rows):
    if len(polynomial_gaps) != len(hinge_rows):
        raise ValueError('Both methods must evaluate the same supplied paths')
    lower, upper = [], []
    for i, row in enumerate(hinge_rows):
        if row['path_index'] != i:
            raise ValueError('Hinge path order must match the polynomial evaluation')
        cert = row['signed_gap_certificate']
        lower.append(float(fraction_value(cert['lower'])))
        upper.append(float(fraction_value(cert['upper'])))
    lower, upper = np.asarray(lower), np.asarray(upper)
    midpoints = (lower+upper)/2
    difference = np.asarray(polynomial_gaps)-midpoints
    result = sample_summary(difference)
    result.update(sign_convention='Polynomial gap minus finest adaptive hinge gap midpoint.',
        mean_difference_range_from_hinge_brackets=[float(np.mean(polynomial_gaps-upper)), float(np.mean(polynomial_gaps-lower))],
        midpoint_mean_numerical_allowance=float(np.mean(upper-lower)/2),
        numerical_scope='Only the hinge term has a verified enclosure. Polynomial arithmetic and these paired difference displays are ordinary floating point.',
        sampling_scope='Paired sample standard error on the same paths; approximate, with no coverage certificate.')
    return result


def run(record, hinge, *, degrees=(2, 3)):
    cpu, wall = time.process_time(), time.perf_counter()
    model = MaxCallModel(**record['model'])
    paths = np.asarray(record['states'], dtype=float)
    stops = np.asarray(record['stopping_indices'], dtype=int)
    design = record['design']
    if (design['spawn_keys'] != [[0], [1], [2]] or design['outer_paths'] != len(paths)
            or design['policy_degree'] != 2 or not design['evaluation_independent_of_both_fitting_streams']
            or len(set(degrees)) != len(degrees) or any(d not in (2, 3) for d in degrees)):
        raise ValueError('Require the declared independent three-stream degree-two policy design')
    streams = np.random.SeedSequence(design['root_seed']).spawn(3)
    reconstruction_cpu = time.process_time()
    outer = simulate_max_call(model, design['outer_paths'], np.random.default_rng(streams[2]))
    if not np.array_equal(outer.states, paths):
        raise ArithmeticError('Supplied states differ from the declared outer stream')
    basis_count = len(exponent_vectors(model.dimension, 2))
    reconstructed_stop = np.full(len(paths), len(model.exercise_times)-1)
    alive = np.ones(len(paths), dtype=bool)
    for j, rule in enumerate(record['frozen_policy'][:-1]):
        if rule:
            if len(rule) != basis_count or not np.all(np.isfinite(rule)):
                raise ValueError('Malformed frozen policy')
            reward = model.discounted_payoff(paths[:, j], j)
            continuation = np.maximum(sorted_polynomial_basis(paths[:, j], model.strike, 2) @ rule, 0.)
            selected = alive & (reward > continuation)
            reconstructed_stop[selected] = j
            alive[selected] = False
    if not np.array_equal(reconstructed_stop, stops):
        raise ArithmeticError('Supplied stops differ from the frozen forward policy')
    reconstruction_elapsed = time.process_time()-reconstruction_cpu
    fitting_cpu = time.process_time()
    training = simulate_max_call(model, design['value_paths'], np.random.default_rng(streams[1]))
    targets = policy_targets(training.states, model, record['frozen_policy'])
    shared_fitting_cpu = time.process_time()-fitting_cpu
    hinge_rows = hinge['observations']['adaptive']
    if any(row['stopping_index'] != int(stop) for row, stop in zip(hinge_rows, stops)):
        raise ValueError('Hinge and polynomial methods must use the same stopping rule')
    result = dict(schema_version=1, author='Amanjeet Singh',
        source='Established polynomial regression-later dual construction; no novelty or paper-algorithm replication claim.',
        mathematical_model='Published Andersen-Broadie Table 2 benchmark, not market observations.',
        frozen_policy_preserved=True, supplied_outer_stream_reproduced=True,
        conditional_moment_formula='prod_i (S_i/100)^a_i * exp(dt*(a dot (r-q-sigma^2/2) + a transpose C a/2)), C_ik=sigma_i*sigma_k*rho_ik.',
        integrability='Finite coefficients, finite-horizon GBM and finite polynomial degrees imply all required moments are finite under the ideal model.',
        value_basis='Complete unsorted-state total-degree monomials in S_i/100; different from the sorted centred policy basis.',
        no_clipping_of_fitted_values=True,
        terminal_scope='Terminal payoff is fitted by a polynomial, not represented exactly. An arbitrary integrable terminal value approximation still gives a valid ideal martingale; it can add dual slack.',
        numerical_certificate=False, statistical_coverage_certificate=False,
        slack_scope='Mean gap combines policy suboptimality and unknown dual slack; it does not isolate either quantity.',
        sample_count=len(paths), value_fitting_paths=design['value_paths'], root_seed=design['root_seed'], value_spawn_key=[1],
        approximate_inference_scope='Ordinary sample standard errors conditional on this single frozen fit. The same paths produce all paired comparisons.',
        stream_and_policy_reconstruction_cpu_seconds=reconstruction_elapsed,
        shared_value_stream_and_target_cpu_seconds=shared_fitting_cpu,
        original_policy_and_hinge_preparation_cpu_seconds=record['preparation_cost']['cpu_seconds'],
        timing_scope='Measured process CPU within this function, including fitting and assessment. Environment bootstrap and JSON I/O excluded. Original preparation is a separate sunk cost.',
        degrees={})
    for degree in degrees:
        fitting_cpu = time.process_time()
        coefficients, diagnostics = fit_values(training.states, targets, degree)
        fitting_elapsed = time.process_time()-fitting_cpu
        assessment_cpu = time.process_time()
        observed = dual_paths(paths, model, stops, coefficients, degree)
        assessment_elapsed = time.process_time()-assessment_cpu
        gap = observed['gap']
        aggregate = dict(degree=degree, basis_dimension=len(exponent_vectors(model.dimension, degree)),
            coefficients=[row.tolist() for row in coefficients],
            basis_exponents=[list(row) for row in exponent_vectors(model.dimension, degree)],
            gap=sample_summary(gap), dual=sample_summary(observed['dual']),
            policy_reward=sample_summary(observed['policy_reward']),
            stopped_martingale=sample_summary(observed['stopped_martingale']),
            all_gap_observations_nonnegative=bool(np.all(gap >= 0)),
            gap_dual_identity_maximum_float_residual=float(np.max(np.abs(
                gap-(observed['dual']-observed['policy_reward']+observed['stopped_martingale'])))),
            paired_against_finest_hinge=paired_comparison(gap, hinge_rows),
            fitting_diagnostics=diagnostics,
            polynomial_fit_cpu_seconds=fitting_elapsed,
            polynomial_interior_fit_cpu_seconds=fitting_elapsed-diagnostics[-1]['fit_cpu_seconds'],
            terminal_polynomial_fit_cpu_seconds=diagnostics[-1]['fit_cpu_seconds'],
            dual_assessment_cpu_seconds=assessment_elapsed,
            fitting_and_assessment_cpu_seconds_including_shared_value_work=shared_fitting_cpu+fitting_elapsed+assessment_elapsed,
            total_baseline_cpu_seconds_including_reconstruction=shared_fitting_cpu+fitting_elapsed+assessment_elapsed+reconstruction_elapsed,
            combined_original_preparation_and_baseline_cpu_seconds=record['preparation_cost']['cpu_seconds']+shared_fitting_cpu+fitting_elapsed+assessment_elapsed+reconstruction_elapsed)
        phases = record.get('preparation_phases')
        if phases is not None:
            common = phases['policy_training']['cpu_seconds']+phases['outer_evaluation_and_record_construction']['cpu_seconds']
            aggregate['common_policy_and_outer_preparation_cpu_seconds'] = common
            aggregate['standalone_polynomial_pricing_cpu_seconds'] = common+shared_fitting_cpu+fitting_elapsed+assessment_elapsed
            aggregate['standalone_polynomial_pricing_plus_validation_cpu_seconds'] = aggregate['standalone_polynomial_pricing_cpu_seconds']+reconstruction_elapsed
            aggregate['hybrid_setup_pricing_cpu_seconds'] = common+shared_fitting_cpu+aggregate['polynomial_interior_fit_cpu_seconds']
            aggregate['hybrid_setup_timing_scope'] = 'Component-based setup estimate: common preparation and own value simulation/targets plus interior regression, reconstructed by subtracting the measured terminal regression from the complete fit.'
            aggregate['standalone_timing_scope'] = 'Common policy and outer preparation plus own value simulation, targets, regression and assessment. Original hinge fitting is excluded. Stream/policy reconstruction is verification, separately charged.'
        result['degrees'][str(degree)] = aggregate
    result['actual_two_degree_study_cpu_seconds'] = time.process_time()-cpu
    result['actual_two_degree_study_wall_seconds'] = time.perf_counter()-wall
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--hinge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT/'data/private').resolve()) or destination.exists():
        raise ValueError('Use a new private aggregate output file')
    input_source, hinge_source = args.input.read_bytes(), args.hinge.read_bytes()
    record, hinge = json.loads(input_source), json.loads(hinge_source)
    input_hash = hashlib.sha256(input_source).hexdigest()
    if hinge['input_sha256'] != input_hash:
        raise ValueError('Hinge comparison must use exactly the same input record')
    result = run(record, hinge)
    result.update(assessed_at_utc=datetime.now(timezone.utc).isoformat(), input_sha256=input_hash,
                  hinge_sha256=hashlib.sha256(hinge_source).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                      'scripts/validate_polynomial_dual.py', 'src/qf_research/optimal_stopping/multivariate_benchmark.py',
                      'src/qf_research/optimal_stopping/contracts.py')})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved established polynomial dual baselines to {destination}', flush=True)


if __name__ == '__main__':
    main()
