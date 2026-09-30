"""Replay one completed evaluation block without refitting either policy.

The smallest retained evaluation sample, followed by the first repetition,
determines selection. The original random stream is regenerated, not extended.
All financial inputs, paths and numerical traces remain private.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from audit_training_allocation_main import verify as verify_main
from export_residual_certificate import (
    MAIN, BASE, merge_hashes, relative_artifacts, native, validate_document, hexadecimal_leaves,
)
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_training_allocation import THREADS
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise
from qf_research.optimal_stopping.funded_policy import (
    FundedPolicy, ContinuationRule, simulate_paths, realise_policy, _digest,
)
from qf_research.optimal_stopping.training_evaluation import policy_gain
from qf_research.optimal_stopping.value_martingale import (
    LinearValue, ValueNode, ValueApproximation, observation_rights,
    value_martingale, evaluate_value_dual,
)


EXPORTED = BASE / 'certified_residual_inputs_v1'
COMPLETION = BASE / 'certified_residual_completion_v1.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


class CapturingGenerator(np.random.Generator):
    """Retain each original standard-normal call without changing its result."""

    def __init__(self, bit_generator):
        super().__init__(bit_generator)
        self.calls = []

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        value = super().standard_normal(size=size, dtype=dtype, out=out)
        self.calls.append(np.asarray(value).copy())
        return value


def selection_rule(models):
    require(isinstance(models, list) and [row['case'] for row in models] == list(range(13)),
            'All thirteen retained models are required before block selection')
    candidates = []
    for model in models:
        require(set(model['arms']) == {'reference', 'selected'}, 'Both retained arms are required')
        for arm_index, arm in enumerate(('reference', 'selected')):
            allocation = model['arms'][arm]['allocation']
            count = allocation['evaluation_paths']
            require(type(count) is int and count >= 2 and allocation['repetitions'] >= 1,
                    'A non-empty retained evaluation sample is required')
            candidates.append((count, model['case'], arm_index, arm))
    count, case, _, arm = min(candidates)
    return dict(case=case, arm=arm, repetition=0, evaluation_paths=count,
        rule='Smallest retained evaluation-path count; ties use ascending case and reference before selected; first repetition. No gain or arithmetic-error result enters selection.')


def model_from_record(record):
    data = dict(record)
    data['dividends'] = tuple(DividendCap(**event) for event in data['dividends'])
    return FundedCashModel(**data)


def policy_from_record(record):
    data = dict(record)
    data['model'] = model_from_record(data['model'])
    data['option'] = Option(**data['option'])
    data['decisions'] = tuple(FundedExercise(**right) for right in data['decisions'])
    data['rules'] = tuple(ContinuationRule(**{**rule, 'coefficients': tuple(rule['coefficients'])})
                          for rule in data['rules'])
    return FundedPolicy(**data)


def approximation_from_record(record):
    def linear(value):
        return None if value is None else LinearValue(tuple(value['knots']), tuple(value['values']), value['tail_slope'])
    return ValueApproximation(model_from_record(record['model']), Option(**record['option']),
        tuple(FundedExercise(**right) for right in record['decisions']),
        tuple(ValueNode(node['time'], linear(node['before']), linear(node['after'])) for node in record['nodes']),
        record['core_steps'], record['wing_steps'], record['compression_tolerance'], record['initial_value'])


def exact_record_match(actual, expected, path='record'):
    """Require binary64 identity, preserving operation order and signed zero."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), 'Record keys differ: ' + path)
        for key in expected:
            exact_record_match(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Record lengths differ: ' + path)
        for index, (a, b) in enumerate(zip(actual, expected)):
            exact_record_match(a, b, path + '[' + str(index) + ']')
    elif type(expected) is float:
        require(type(actual) is float and math.isfinite(actual) and actual.hex() == expected.hex(),
                'Binary64 record differs: ' + path)
    else:
        require(type(actual) is type(expected) and actual == expected, 'Record value differs: ' + path)


def path_hash(states, payments, rights):
    result = hashlib.sha256()
    result.update(states.tobytes())
    result.update(payments.tobytes())
    result.update(json.dumps(rights, sort_keys=True).encode())
    return result.hexdigest()


def validate_arrays(payload):
    n = payload['selection']['evaluation_paths']
    intervals = payload['innovation_intervals']
    rights = payload['observation_rights']
    model = payload['model_spec']['model']
    maturity = payload['model_spec']['option']['maturity']
    events = [event for event in model['dividends'] if event['ex_time'] <= maturity]
    expected_shapes = {'normals': (n, len(intervals)), 'states': (n, len(rights)),
                       'detached_payments': (n, len(events)), 'stopping_indices': (n, 2),
                       'quoted_gains': (n,)}
    for name, shape in expected_shapes.items():
        values = np.asarray(payload[name])
        require(values.shape == shape and np.all(np.isfinite(values)), 'Malformed replay array: ' + name)
    require(len(intervals) == len(payload['innovation_times'])
            and [row['to_time'] for row in intervals] == payload['innovation_times'],
            'Normal draws are not aligned with diffusion times')
    previous = 0.
    for interval in intervals:
        require(interval['from_time'] == previous and interval['to_time'] > previous,
                'Innovation intervals must cover consecutive positive-time transitions')
        previous = interval['to_time']
    require(previous == maturity and model['volatility'] > 0,
            'This replay requires a positive-volatility calendar ending at maturity')
    stopping = np.asarray(payload['stopping_indices'])
    require(np.issubdtype(stopping.dtype, np.integer) and np.all(stopping >= 0)
            and np.all(stopping < len(payload['policies'][0]['decisions'])), 'Invalid stopping indices')
    require(np.all(np.asarray(payload['states']) >= 0) and np.all(np.asarray(payload['detached_payments']) >= 0),
            'Negative funded states or detached payments')
    actual_hash = path_hash(np.asarray(payload['states'], dtype=np.float64),
        np.asarray(payload['detached_payments'], dtype=np.float64), rights)
    require(actual_hash == payload['record']['evaluation']['paths_sha256'],
            'Exported path bytes differ from the retained evaluation hash')
    return expected_shapes


def run(output):
    output = private(output)
    require(not output.exists(), 'Use a fresh private replay directory')
    require(all(os.environ.get(key) == '1' for key in THREADS), 'The retained thread settings are required')
    main_manifest, cfg, tasks, reconstructed = verify_main(MAIN)
    selected = selection_rule(reconstructed['models'])
    require((selected['case'], selected['arm'], selected['repetition'], selected['evaluation_paths']) ==
            (12, 'selected', 0, 128), 'The agreed first replay selection differs')
    block_path = MAIN / 'blocks' / f"case_{selected['case']:03d}_{selected['arm']}_rep_{selected['repetition']:04d}.json"
    block = read(block_path)
    export_manifest = read(EXPORTED / 'manifest.json')
    require(export_manifest.get('complete') is True, 'The full certified-model export is required')
    completion = read(COMPLETION)
    require(completion.get('complete') is True and completion.get('ideal_model_global_residual_bound_rounding_certified') is True,
            'The completed ideal-model certificate must precede this replay')
    inputs = merge_hashes(main_manifest['input_sha256'], export_manifest['input_sha256'],
        relative_artifacts(MAIN, main_manifest['artifact_sha256']),
        relative_artifacts(EXPORTED, export_manifest['artifact_sha256']), completion['evidence_sha256'],
        {str(path.relative_to(ROOT)): sha(path) for path in
         (MAIN / 'manifest.json', EXPORTED / 'manifest.json', COMPLETION, block_path)})
    sources = merge_hashes(main_manifest['source_sha256'], export_manifest['source_sha256'],
        {name: value for name, value in completion['evidence_sha256'].items()
         if name.startswith(('scripts/', 'src/', 'tests/'))},
        {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__)),
         'tests/test_export_gain_replay.py': sha(ROOT / 'tests/test_export_gain_replay.py')})
    closure = merge_hashes(inputs, sources)
    verify_mapping(closure)
    exported = read(EXPORTED / 'models.json')
    validate_document(exported)
    model_spec = exported['models'][selected['case']]
    approximation = approximation_from_record(model_spec)
    require(approximation.fingerprint == model_spec['approximation_fingerprint'] ==
            block['evaluation']['approximation_fingerprint'], 'The retained interpolant identity differs')
    policies = tuple(policy_from_record(p) for p in block['training']['policies'])
    require([policy.fingerprint for policy in policies] == block['training']['policy_fingerprints'] ==
            block['evaluation']['policy_fingerprints'], 'A retained policy fingerprint differs')
    for policy in policies:
        require((policy.model, policy.option, policy.decisions) ==
                (approximation.model, approximation.option, approximation.decisions),
                'The policies and interpolant describe different financial problems')
    retained_stream = block['evaluation']['stream']
    require(retained_stream['bit_generator'] == 'PCG64', 'The retained PCG64 stream is required')
    seed = np.random.SeedSequence(retained_stream['entropy'], spawn_key=tuple(retained_stream['spawn_key']),
                                  pool_size=retained_stream['pool_size'])
    rng = CapturingGenerator(np.random.PCG64(seed))
    initial_rng_state = native(rng.bit_generator.state)
    require(_digest(initial_rng_state) == retained_stream['stream_id'], 'The retained initial stream differs')
    observations = simulate_paths(policies[0].model, policies[0].option,
        observation_rights(approximation), selected['evaluation_paths'], rng)
    final_rng_state = native(rng.bit_generator.state)
    rights = native([asdict(right) for right in observations.decisions])
    require(path_hash(observations.states, observations.detached_payments, rights) ==
            block['evaluation']['paths_sha256'], 'The replay does not reproduce the complete original path bytes')
    sample = value_martingale(approximation, observations)
    realised = [realise_policy(policy, sample.paths) for policy in policies]
    duals = [evaluate_value_dual(policy, sample) for policy in policies]
    low, high = duals
    factor = policies[0].model.premium_factor
    gain = policy_gain(low, high, factor)
    unscaled_gain = high.adjusted_policy_payoffs - low.adjusted_policy_payoffs
    low_quote, high_quote = low.adjusted_policy_payoffs / factor, high.adjusted_policy_payoffs / factor
    quoted_gain = high_quote - low_quote
    replay = native(dict(training_paths=[p.training_paths for p in policies],
        policy_fingerprints=[p.fingerprint for p in policies], evaluation_paths=selected['evaluation_paths'],
        stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key), pool_size=seed.pool_size,
                    bit_generator='PCG64', stream_id=observations.stream_id, paths=len(observations.states)),
        paths_sha256=path_hash(observations.states, observations.detached_payments, rights),
        approximation_fingerprint=sample.approximation_fingerprint, gain=gain,
        nonzero_gain_count=int(np.count_nonzero(unscaled_gain)),
        minimum_gap_quote=[float(result.gap_payoffs.min()/factor) for result in duals]))
    retained = {key: value for key, value in block['evaluation'].items()
                if key not in ('elapsed_seconds', 'cpu_seconds')}
    exact_record_match(replay, retained)
    times = [node.time for node in approximation.nodes if node.time > 0]
    previous = 0.
    intervals = []
    for time in times:
        require(time > previous and approximation.model.volatility > 0, 'Each captured call requires a positive diffusion interval')
        intervals.append(dict(from_time=previous, to_time=time))
        previous = time
    require(len(rng.calls) == len(intervals) and all(call.shape == (selected['evaluation_paths'],) for call in rng.calls),
            'Captured normal calls differ from the original diffusion schedule')
    normals = np.column_stack(rng.calls)
    payload = dict(schema_version=1, selection=selected, model_spec=model_spec,
        policies=block['training']['policies'], observation_rights=rights,
        innovation_times=times, innovation_intervals=intervals, normals=normals.tolist(),
        states=observations.states.tolist(), detached_payments=observations.detached_payments.tolist(),
        stopping_indices=np.column_stack([r.stopping_indices for r in realised]).tolist(),
        quoted_gains=quoted_gain.tolist(), record=block, replay_evaluation=replay,
        binary64_convention='JSON numbers round-trip to the original binary64 values. Interpret those binary64 inputs as exact dyadic values in the separate certificate. The normal draws are retained finite-generator innovations, not exact Gaussian realisations.',
        array_storage=dict(dtype=observations.states.dtype.str, byte_order=sys.byteorder, order='C',
                           path_hash_fields=['states bytes', 'detached_payments bytes', 'sorted-key JSON observation rights']),
        generator=dict(initial_state=initial_rng_state, final_state=final_rng_state),
        original_arithmetic=dict(
            martingale_at_exercise=sample.values.tolist(),
            adjusted_policy_payoffs=np.column_stack([d.adjusted_policy_payoffs for d in duals]).tolist(),
            separately_scaled_policy_payoffs=np.column_stack([low_quote, high_quote]).tolist(),
            realised_discounted_payoffs=np.column_stack([r.discounted_payoffs for r in realised]).tolist(),
            common_upper=low.upper_payoffs.tolist(), gap_payoffs=np.column_stack([d.gap_payoffs for d in duals]).tolist(),
            unscaled_gain=unscaled_gain.tolist(),
            gap_reduction=(low.gap_payoffs-high.gap_payoffs).tolist(),
            quoted_gain_operation='(high adjusted payoff / premium factor) - (low adjusted payoff / premium factor)',
            policy_diagnostics=[native(list(r.diagnostics)) for r in realised]))
    payload['binary64_hex'] = hexadecimal_leaves(payload)
    shapes = validate_arrays(payload)
    # JSON must preserve all binary64 replay data, including signed zero.
    roundtrip = json.loads(json.dumps(payload, allow_nan=False))
    exact_record_match(roundtrip, payload)
    validate_arrays(roundtrip)
    verify_mapping(closure)
    output.mkdir(parents=True, exist_ok=False)
    try:
        save(output / 'inputs.json', payload)
        exact_record_match(read(output / 'inputs.json'), payload)
        verify_mapping(closure)
        verification = dict(case=selected['case'], arm=selected['arm'], repetition=0,
            original_path_hash_matches=True, original_scalar_record_matches_binary64=True,
            policy_fingerprints_match=True, approximation_fingerprint_matches=True,
            normals_captured_without_changing_generator_results=True,
            replayed_paths=selected['evaluation_paths'], normal_draws=int(normals.size),
            diffusion_intervals=len(intervals), observation_rights=len(rights),
            array_shapes=shapes, policies_refitted=0, new_independent_research_paths=0,
            stopping_disagreements=int(np.count_nonzero(realised[0].stopping_indices != realised[1].stopping_indices)),
            numerical_gain_certificate_produced=False, original_evidence_unchanged=True)
        save(output / 'verification.json', verification)
        save(output / 'manifest.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
            input_sha256=inputs, source_sha256=sources,
            artifact_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            platform=platform.platform(), thread_environment={key: os.environ[key] for key in THREADS},
            selection=selected, original_evidence_unchanged=True, publication_clearance=False))
        verify_mapping(closure)
        return verification
    except Exception as exc:
        save(output / 'incomplete.json', dict(complete=False, exception_type=type(exc).__name__, reason=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), sort_keys=True))
