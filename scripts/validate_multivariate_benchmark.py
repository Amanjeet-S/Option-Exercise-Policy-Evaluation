"""Engineering validation on public multiasset max-call mathematical inputs.

Outputs remain private pending review. This executes an independent LSM
baseline with a declared sorted full-state basis, not the Andersen-Broadie
primal-dual algorithm and not an empirical market-data experiment.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import t

from qf_research.optimal_stopping.multivariate_benchmark import (
    evaluate_max_call, fit_max_call_policy, independent_european_max_call,
    monomial_indices, published_max_call, simulate_max_call,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://business.columbia.edu/sites/default/files-efs/pubfiles/1233/primal_dual_ms_2004.pdf'
PUBLISHED_INTERVALS = {(2, 90.): (8.053, 8.082), (2, 100.): (13.892, 13.934),
    (2, 110.): (21.316, 21.359), (3, 90.): (11.265, 11.308),
    (3, 100.): (18.661, 18.728), (3, 110.): (27.512, 27.663),
    (5, 90.): (16.602, 16.655), (5, 100.): (26.109, 26.292),
    (5, 110.): (36.704, 36.832)}


class Moments:
    """Merge within-chunk centred moments without retaining evaluation paths."""

    def __init__(self):
        self.count, self.mean, self.squares = 0, 0., 0.

    def add(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError('Require nonempty finite payoff values')
        count, mean = len(values), float(values.mean())
        squares = float(np.sum((values-mean)**2))
        delta = mean-self.mean
        total = self.count+count
        self.squares += squares+delta*delta*self.count*count/total
        self.mean += delta*count/total
        self.count = total

    def summary(self):
        if self.count < 2:
            raise ValueError('At least two observations are required')
        error = math.sqrt(self.squares/((self.count-1)*self.count))
        radius = float(t.ppf(.975, self.count-1))*error
        return dict(observations=self.count, mean=self.mean, standard_error=error,
                    pointwise_student_interval=[self.mean-radius, self.mean+radius])


def validate(dimension, spot, training_paths, evaluation_paths, degree, seed, chunk_size):
    if evaluation_paths < 2 or training_paths < 2 or chunk_size < 1:
        raise ValueError('Require positive chunk size and at least two training/evaluation paths')
    model = published_max_call(dimension, spot)
    start = time.perf_counter()
    streams = np.random.SeedSequence(seed).spawn(2)
    training = simulate_max_call(model, training_paths, np.random.default_rng(streams[0]))
    policy = fit_max_call_policy(training, degree)
    train_seconds = time.perf_counter()-start
    del training
    reference = independent_european_max_call(model)
    rng = np.random.default_rng(streams[1])
    moments = {key: Moments() for key in ('policy_value', 'terminal_waiting', 'matched_gain')}
    stops = np.zeros(len(model.exercise_times), dtype=int)
    chunks = 0
    mark = time.perf_counter()
    for begin in range(0, evaluation_paths, chunk_size):
        paths = simulate_max_call(model, min(chunk_size, evaluation_paths-begin), rng)
        evaluation = evaluate_max_call(policy, paths)
        for key, values in [('policy_value', evaluation.discounted_rewards),
                            ('terminal_waiting', evaluation.terminal_rewards),
                            ('matched_gain', evaluation.matched_gains)]:
            moments[key].add(values)
        stops += np.bincount(evaluation.stopping_indices, minlength=len(stops))
        chunks += 1
    estimates = {key: value.summary() for key, value in moments.items()}
    cancellation_error = abs(estimates['policy_value']['mean']-estimates['terminal_waiting']['mean']
                             -estimates['matched_gain']['mean'])
    if cancellation_error > 1e-10:
        raise ArithmeticError('Chunked policy/terminal/gain mean cancellation failed')
    terminal = estimates['terminal_waiting']
    error = terminal['mean']-reference['value']
    published = PUBLISHED_INTERVALS.get((dimension, float(spot)))
    return dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        classification='New public-model mathematical simulation; engineering validation, not market data.',
        financial_model_source=dict(citation='Andersen and Broadie (2004), Management Science 50(9), '
            '1222-1234, equation (19) and Table 2.', doi='10.1287/mnsc.1040.0258', url=SOURCE,
            published_bermudan_interval=published), model=asdict(model),
        design=dict(training_paths=training_paths, evaluation_paths=evaluation_paths, degree=degree,
            basis='Complete total-degree monomials of all asset prices sorted descending, divided by '
                'strike and centred at one. This differs from the published 13-function basis.',
            basis_dimension=len(monomial_indices(dimension, degree)), root_seed=seed,
            stream_spawn_keys=[list(x.spawn_key) for x in streams], chunk_size=chunk_size,
            evaluation_chunks=chunks, same_configuration_reproduces_draws=True,
            chunk_size_is_part_of_random_stream_layout=True,
            training_and_evaluation_independent=True, policy_frozen_before_evaluation=True),
        estimates=estimates, independent_european_reference=reference,
        matched_gain_mean_cancellation_absolute_error=cancellation_error,
        terminal_reference_difference=error,
        terminal_reference_standardised_error=error/terminal['standard_error'] if terminal['standard_error'] else None,
        terminal_reference_absolute_error=abs(error), stopping_counts=stops.tolist(),
        early_exercise_fraction=float(stops[:-1].sum()/evaluation_paths),
        training_rule_diagnostics=[dict(time_index=i, itm_paths=r.training_itm_paths,
            fitted=bool(r.coefficients), rank=r.rank, condition_number=r.condition_number)
            for i, r in enumerate(policy.rules)],
        measured_elapsed_seconds=dict(training=train_seconds, evaluation=time.perf_counter()-mark,
                                      total=time.perf_counter()-start),
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        limitations='Intervals condition on the fitted policy and use approximate Student calibration. '
            'The European quadrature is a numerical point reference. The published Bermudan interval '
            'belongs to another algorithm, basis and simulation budget; it is not an exact optimum '
            'or a certified loss enclosure. This baseline introduces no theoretical or algorithmic novelty.',
        algorithm_attribution='Longstaff and Schwartz (2001), doi:10.1093/rfs/14.1.113. '
            'Full-state basis/sample-size comparisons are related to Glasserman and Yu (2004), '
            'Number of paths versus number of basis functions in American option pricing, '
            'doi:10.1214/105051604000000846.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, choices=(2, 3, 5), default=5)
    parser.add_argument('--spot', type=float, choices=(90., 100., 110.), default=100.)
    parser.add_argument('--training-paths', type=int, default=20000)
    parser.add_argument('--evaluation-paths', type=int, default=100000)
    parser.add_argument('--degree', type=int, choices=(2, 3), default=2)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--chunk-size', type=int, default=10000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Engineering-validation outputs must remain private pending review')
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    print('Running a new mathematical simulation on published public inputs.', flush=True)
    record = validate(args.dimension, args.spot, args.training_paths, args.evaluation_paths,
                      args.degree, args.seed, args.chunk_size)
    source_paths = [Path(__file__).resolve(), ROOT/'src/qf_research/optimal_stopping/multivariate_benchmark.py']
    record['implementation_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in source_paths}
    output.mkdir(parents=True)
    destination = output/'validation.json'
    destination.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(f'Engineering validation saved to {destination}')
