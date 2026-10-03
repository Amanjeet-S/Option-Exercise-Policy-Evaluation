"""Shared numerical workloads for allocating policy-gain simulation effort.

An evaluation uses one independent path sample and one value martingale for
both frozen policies. Process CPU and elapsed times cover the existing numerical
primitives, their validation, and paired-estimate construction. Record assembly,
policy fingerprints and evidence hashes are outside those timers. The existing
primitives' own provenance checks remain part of their measured workload.
"""

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from .contracts import positive_integer
from .funded_policy import _digest, fit_policy, simulate_paths
from .training_evaluation import policy_gain
from .value_martingale import (
    build_values, evaluate_value_dual, observation_rights, value_martingale,
)


def _json_native(value):
    """Normalise tuples and NumPy scalars before retaining a JSON record."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Unsupported provenance type: {type(item).__name__}')
    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _generator(seed):
    if not isinstance(seed, np.random.SeedSequence):
        raise ValueError('An explicit SeedSequence is required for each stream')
    return np.random.Generator(np.random.PCG64(seed))


def _stream(seed, paths):
    return dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key),
                pool_size=seed.pool_size, bit_generator='PCG64',
                stream_id=paths.stream_id, paths=len(paths.states))


def _problem(policy):
    return policy.model, policy.option, policy.decisions


def train_pair(base_policy, counts=(5000, 20000), *, degree=4,
               minimum_itm_paths=32, condition_limit=1e8, seeds):
    """Fit independent small/large-sample policies and time the complete pair.

    The base policy supplies only the financial model, option and exercise
    calendar. Its fitted coefficients and training paths are not reused.
    """
    if not isinstance(counts, tuple) or len(counts) != 2:
        raise ValueError('Two increasing training counts are required as a tuple')
    counts = tuple(positive_integer(n, 'training count', 2) for n in counts)
    if counts[0] >= counts[1]:
        raise ValueError('Training counts must increase strictly')
    if not isinstance(seeds, tuple) or len(seeds) != 2:
        raise ValueError('Two independent training SeedSequences are required')
    generators = tuple(_generator(seed) for seed in seeds)
    stream_ids = [_digest(rng.bit_generator.state) for rng in generators]
    if len(set(stream_ids)) != 2:
        raise ValueError('Independent training streams must not be reused')
    if base_policy.training_stream_id in stream_ids:
        raise ValueError('Training must not reuse the base policy training stream')

    policies, streams = [], []
    phases = dict(simulation_seconds=0., simulation_cpu_seconds=0.,
                  fitting_seconds=0., fitting_cpu_seconds=0.)
    started, cpu_started = time.perf_counter(), time.process_time()
    for count, seed, rng in zip(counts, seeds, generators):
        mark, cpu_mark = time.perf_counter(), time.process_time()
        paths = simulate_paths(base_policy.model, base_policy.option,
                               base_policy.decisions, count, rng)
        phases['simulation_seconds'] += time.perf_counter()-mark
        phases['simulation_cpu_seconds'] += time.process_time()-cpu_mark
        mark, cpu_mark = time.perf_counter(), time.process_time()
        policy = fit_policy(base_policy.option, paths, degree,
                            minimum_itm_paths=minimum_itm_paths,
                            condition_limit=condition_limit)
        phases['fitting_seconds'] += time.perf_counter()-mark
        phases['fitting_cpu_seconds'] += time.process_time()-cpu_mark
        policies.append(policy)
        # Scalar metadata only; no simulated paths survive the training call.
        streams.append((seed, paths.stream_id, len(paths.states)))
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    record = dict(training_paths=counts, policies=[asdict(p) for p in policies],
                  policy_fingerprints=[p.fingerprint for p in policies],
                  streams=[dict(entropy=s.entropy, spawn_key=list(s.spawn_key),
                                pool_size=s.pool_size, bit_generator='PCG64',
                                stream_id=stream_id, paths=count)
                           for s, stream_id, count in streams],
                  phases=phases, elapsed_seconds=elapsed, cpu_seconds=cpu)
    return tuple(policies), _json_native(record)


def build_approximation(base_policy, *, core_steps=800, wing_steps=40,
                        tolerance=.001):
    """Time a model-level value construction separately from each repetition."""
    started, cpu_started = time.perf_counter(), time.process_time()
    approximation = build_values(base_policy.model, base_policy.option,
                                 base_policy.decisions, core_steps=core_steps,
                                 wing_steps=wing_steps, tolerance=tolerance)
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    record = dict(settings=dict(core_steps=core_steps, wing_steps=wing_steps,
                                tolerance=tolerance),
                  approximation_fingerprint=approximation.fingerprint,
                  elapsed_seconds=elapsed, cpu_seconds=cpu)
    return approximation, _json_native(record)


def evaluate_pair(policies, approximation, count, seed):
    """Evaluate a frozen pair on one common, independently simulated sample.

    Full dual evaluations are retained inside the calculation to check the
    pathwise bounds and common-upper cancellation. Only scalar moments and
    evidence metadata are returned, so memory does not grow with repetitions.
    """
    count = positive_integer(count, 'evaluation count', 2)
    rng = _generator(seed)
    started, cpu_started = time.perf_counter(), time.process_time()
    if not isinstance(policies, tuple) or len(policies) != 2:
        raise ValueError('A tuple of two independently trained policies is required')
    low, high = policies
    if _problem(low) != _problem(high) or _problem(low) != _problem(approximation):
        raise ValueError('Policies and approximation must share model, option and calendar')
    training_ids = {low.training_stream_id, high.training_stream_id}
    if len(training_ids) != 2:
        raise ValueError('Independent training streams must not be reused')
    if _digest(rng.bit_generator.state) in training_ids:
        raise ValueError('Evaluation must not reuse either training stream')
    if low.training_paths >= high.training_paths:
        raise ValueError('Policies must be ordered by increasing training count')
    observations = simulate_paths(low.model, low.option,
                                  observation_rights(approximation), count, rng)
    sample = value_martingale(approximation, observations)
    low_result = evaluate_value_dual(low, sample)
    high_result = evaluate_value_dual(high, sample)
    factor = low.model.premium_factor
    gain = policy_gain(low_result, high_result, factor)
    difference = high_result.adjusted_policy_payoffs-low_result.adjusted_policy_payoffs
    nonzero_gain_count = int(np.count_nonzero(difference))
    minimum_gap_quote = [float(result.gap_payoffs.min()/factor)
                         for result in (low_result, high_result)]
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu_started

    path_hash = hashlib.sha256()
    path_hash.update(observations.states.tobytes())
    path_hash.update(observations.detached_payments.tobytes())
    path_hash.update(json.dumps([asdict(d) for d in observations.decisions],
                                sort_keys=True).encode())
    record = dict(training_paths=[p.training_paths for p in policies],
                  policy_fingerprints=[p.fingerprint for p in policies],
                  evaluation_paths=count, stream=_stream(seed, observations),
                  paths_sha256=path_hash.hexdigest(),
                  approximation_fingerprint=sample.approximation_fingerprint,
                  gain=gain, nonzero_gain_count=nonzero_gain_count,
                  minimum_gap_quote=minimum_gap_quote,
                  elapsed_seconds=elapsed, cpu_seconds=cpu)
    return _json_native(record)
