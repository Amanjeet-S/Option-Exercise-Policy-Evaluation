"""Independently audit component sensitivity using retained paired summaries.

The audit uses standard-library exact rational arithmetic, reconstructs every
full and delete-one component from original pilot blocks, and enumerates every
pairwise affine crossing. It neither imports the analysis implementation nor
uses the production endpoint reduction to establish worst-case regret.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'allocation_variance_sensitivity/2026-09-28'
PREVIOUS = PRIVATE / 'allocation_cost_sensitivity/2026-09-27'
ORIGINAL = PRIVATE / 'training_allocation/2026-09-27'
ACCOUNTS = ('marginal', 'screening_charged')
SCOPES = ('pilot', 'validation', 'joint')
STATUSES = ('positive', 'boundary', 'unresolved_negative', 'degenerate_zero')
REFERENCE_PATHS = 4096


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Keep evidence under data/private')
    return path


def binary64(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected a finite original scalar')
    return Fraction(value)


def rational(value):
    require(isinstance(value, str), 'Expected a canonical rational string')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational string')
    return result


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def same(actual, expected, location):
    expected = encode(expected)
    require(type(actual) is type(expected), 'Record type differs: ' + location)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Record fields differ: ' + location)
        for key, value in expected.items():
            same(actual[key], value, location + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Record length differs: ' + location)
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, location + '[' + str(index) + ']')
    else:
        require(actual == expected, 'Independent value differs: ' + location)


def component_record(pairs, omitted=None):
    """Use uncentred second moments, independently of centred-sum production."""
    kept = [row for row in pairs if row['pair'] != omitted]
    count = len(kept)
    require(count >= 2, 'Too few independent training pairs')
    total = sum((row['mean'] for row in kept), Fraction(0))
    squares = sum((row['mean']**2 for row in kept), Fraction(0))
    observed = (squares - total*total/count)/(count-1)
    evaluation = sum((row['standard_error']**2 for row in kept), Fraction(0))/count
    a, b = observed-evaluation, REFERENCE_PATHS*evaluation
    if a < 0 or b < 0:
        status, weight = 'unresolved_negative', None
    elif a == b == 0:
        status, weight = 'degenerate_zero', None
    else:
        status = 'boundary' if a == 0 or b == 0 else 'positive'
        weight = b/(REFERENCE_PATHS*a+b)
    return dict(omitted_pair=omitted, pairs=count, observed_variance=observed,
                evaluation_mean_variance=evaluation, A=a, B=b, status=status, share=weight)


def candidate_value(row, weight):
    # Direct variance normalisation, rather than intercept-plus-slope evaluation.
    return ((1-weight) + Fraction(REFERENCE_PATHS, row['evaluation_paths'])*weight)/row['repetitions']


def choice(candidates, weight):
    ordered = sorted(candidates, key=lambda row: (row['evaluation_paths'], row['repetitions']))
    values = [(candidate_value(row, weight), row) for row in ordered]
    best = min(value for value, _ in values)
    minimisers = [dict(row) for value, row in values if value == best]
    return dict(**minimisers[0], criterion=best, minimisers=minimisers)


def crossing_geometry(candidates):
    """Recover the lower envelope from every crossing and open-cell midpoint."""
    nodes = {Fraction(0), Fraction(1)}
    pair_crossings = []
    for left, right in combinations(candidates, 2):
        difference_zero = candidate_value(left, Fraction(0))-candidate_value(right, Fraction(0))
        difference_one = candidate_value(left, Fraction(1))-candidate_value(right, Fraction(1))
        if difference_one != difference_zero:
            crossing = difference_zero/(difference_zero-difference_one)
            if 0 <= crossing <= 1:
                nodes.add(crossing)
                pair_crossings.append(crossing)
    nodes = sorted(nodes)
    middle = [(left+right)/2 for left, right in zip(nodes, nodes[1:])]
    samples = sorted(nodes+middle)
    selections = {weight: choice(candidates, weight) for weight in samples}
    intervals, envelope = {}, []
    for row in sorted(candidates, key=lambda item: (item['evaluation_paths'], item['repetitions'])):
        n = row['evaluation_paths']
        weak_nodes = [weight for weight in nodes if row in selections[weight]['minimisers']]
        if not weak_nodes:
            require(all(row not in selections[weight]['minimisers'] for weight in middle),
                    'An affine weak-optimality interval lacks boundary nodes')
            intervals[n] = None
            continue
        lower, upper = min(weak_nodes), max(weak_nodes)
        for weight in samples:
            require((row in selections[weight]['minimisers']) == (lower <= weight <= upper),
                    'Weak-optimality set is not the reconstructed closed interval')
        interval = dict(lower=lower, upper=upper,
            lower_selected=selections[lower]['evaluation_paths'] == n,
            upper_selected=selections[upper]['evaluation_paths'] == n)
        intervals[n] = interval
        envelope.append(dict(**row, region=interval))
    return intervals, envelope, nodes, len(samples), len(pair_crossings)


def exhaustive_regret(candidates, target, lower, upper, nodes):
    target_row = next(row for row in candidates if row['evaluation_paths'] == target)
    weights = sorted({lower, upper} | {weight for weight in nodes if lower <= weight <= upper})
    ratios = {weight: candidate_value(target_row, weight)/choice(candidates, weight)['criterion']
              for weight in weights}
    maximum = max(ratios.values())
    # Each interval between adjacent crossings has a fixed optimal denominator.
    # Its positive affine ratio is monotone or constant, so these nodes exhaust
    # its extrema without assuming that the global endpoints suffice.
    for left, right in zip(weights, weights[1:]):
        middle = (left+right)/2
        value = candidate_value(target_row, middle)/choice(candidates, middle)['criterion']
        require(min(ratios[left], ratios[right]) <= value <= max(ratios[left], ratios[right]),
                'Within-cell fractional-linear monotonicity fails')
    ends = sorted({weight for weight in (lower, upper) if ratios[weight] == maximum})
    require(ends, 'An interior crossing has regret exceeding both declared endpoints')
    return dict(maximum=maximum, lower_ratio=ratios[lower], upper_ratio=ratios[upper],
                worst_endpoints=ends), len(weights)


def audit(inputs, run, output):
    inputs, run, output = map(private, (inputs, run, output))
    require(not output.exists() and not output.is_relative_to(inputs) and not output.is_relative_to(run),
            'Use a fresh, separate audit directory')
    evidence = {}

    def retain(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence path')
        digest = sha(path)
        name = str(path.relative_to(ROOT))
        require(expected is None or digest == expected, 'Evidence checksum differs: ' + name)
        require(name not in evidence or evidence[name] == digest, 'Conflicting evidence checksum')
        evidence[name] = digest
        return path

    def load_manifest(folder):
        record = read(retain(folder/'manifest.json'))
        require(record.get('complete') is True and not (folder/'failure.json').exists(),
                'A completed input or output stage is required')
        for group, anchor in (('evidence_sha256', ROOT), ('artifact_sha256', folder)):
            require(isinstance(record[group], dict) and record[group], 'Missing evidence map')
            for name, digest in record[group].items():
                path = (anchor/name).resolve()
                require(path.is_relative_to(anchor.resolve()), 'Evidence path escapes its declared root')
                retain(path, digest)
        return record

    def bound_read(path):
        require(str(path.resolve().relative_to(ROOT)) in evidence, 'Consumed source is absent from evidence closure')
        return read(path)

    input_manifest, run_manifest = load_manifest(inputs), load_manifest(run)
    expected_files = {'summary.json'} | {f'case_{case:02d}.json' for case in range(13)}
    require(set(input_manifest['artifact_sha256']) == {'inputs.json'}
            and {str(p.relative_to(inputs)) for p in inputs.rglob('*.json')} == {'inputs.json', 'manifest.json'},
            'Input artifact coverage differs')
    require(set(run_manifest['artifact_sha256']) == expected_files
            and {str(p.relative_to(run)) for p in run.rglob('*.json')} == expected_files | {'manifest.json'},
            'Run artifact coverage differs')
    expected_closure = dict(input_manifest['evidence_sha256'])
    for filename in ('inputs.json', 'manifest.json'):
        expected_closure[str((inputs/filename).relative_to(ROOT))] = sha(inputs/filename)
    require(run_manifest['evidence_sha256'] == expected_closure, 'Run and input provenance closures differ')
    for relative in ('scripts/analyse_allocation_variance_sensitivity.py',
                     'src/qf_research/optimal_stopping/allocation_variance_sensitivity.py',
                     'tests/test_allocation_variance_sensitivity.py',
                     str((BASE/'design_v1/protocol.md').relative_to(ROOT)),
                     str((BASE/'theory_v1/component_geometry.md').relative_to(ROOT))):
        require(relative in input_manifest['evidence_sha256'], 'A stage source or design is not bound')
    completion = bound_read(PREVIOUS/'completion_v1.json')
    previous_audit = bound_read(PREVIOUS/'audit_v1/checks.json')
    require(completion['complete'] is True and previous_audit['complete'] is True,
            'The preceding cost study is incomplete')
    for name, digest in {**completion['evidence_sha256'], **previous_audit['evidence_sha256']}.items():
        require(input_manifest['evidence_sha256'].get(name) == digest, 'Prior evidence closure is not retained')
    originals = bound_read(ORIGINAL/'main_registration_v1/allocations.json')['models']
    require(originals == bound_read(ORIGINAL/'pilot_v1/summary.json')['models'],
            'Registered and original pilot moments differ')
    previous_inputs = bound_read(PREVIOUS/'inputs_v1/inputs.json')['models']
    payload = read(inputs/'inputs.json')
    require([model['case'] for model in originals] == [model['case'] for model in previous_inputs]
            == [model['case'] for model in payload['models']] == list(range(13)), 'Case coverage differs')
    reconstructed_inputs, expected_summaries, details = [], [], []
    total_pairs = total_deletions = total_regimes = 0
    total_geometry_points = total_crossings = total_regret_points = 0
    for original, previous, retained_input in zip(originals, previous_inputs, payload['models']):
        case = original['case']
        raw = bound_read(ORIGINAL/f'pilot_v1/blocks/case_{case:03d}.json')
        require(raw['case'] == case and [row['repetition'] for row in raw['statistical']] == list(range(32)),
                'Original whole-pair coverage differs')
        pairs = []
        for row in raw['statistical']:
            estimate = row['evaluation']['gain']['estimate']
            require(estimate['observations'] == REFERENCE_PATHS, 'Original evaluation count differs')
            error = binary64(estimate['standard_error'])
            require(error >= 0, 'Original standard error is negative')
            pairs.append(dict(pair=row['repetition'], mean=binary64(estimate['value']), standard_error=error))
        a = binary64(original['pilot_inference']['training_variance_moment_estimate'])
        b = binary64(original['pilot_inference']['evaluation_path_variance'])
        budget = binary64(original['allocation']['budget']['cpu_seconds'])
        require(a > 0 and b > 0, 'This retained study requires positive baseline components')
        require(a == rational(previous['A']) and b == rational(previous['B'])
                and budget == rational(previous['budget']), 'Original cost-study inputs differ')
        full = component_record(pairs)
        deletions = [component_record(pairs, index) for index in range(32)]
        original_weight = b/(REFERENCE_PATHS*a+b)
        discrepancy = dict(A=full['A']-a, B=full['B']-b)
        require(all(math.isclose(float(full[key]), float(value), rel_tol=2e-12, abs_tol=2e-14)
                    for key, value in (('A', a), ('B', b))), 'Signed reconstruction exceeds stated round-off tolerance')
        valid = [row['share'] for row in deletions if row['share'] is not None]
        diagnostic = valid+[original_weight]+([full['share']] if full['share'] is not None else [])
        lower, upper = min(diagnostic), max(diagnostic)
        prior_case = bound_read(PREVIOUS/f'run_v1/case_{case:02d}.json')
        regimes_in, regimes_out, case_details = [], [], []
        for account in ACCOUNTS:
            for scope in SCOPES:
                previous_choice = prior_case['accounting'][account]['choices'][scope]
                candidates = [{key: row[key] for key in ('evaluation_paths', 'repetitions')}
                              for row in previous_choice['candidates'] if row['feasible']]
                require(candidates and all(type(row['evaluation_paths']) is int
                    and row['evaluation_paths'] in (128, 256, 4096)
                    and type(row['repetitions']) is int and 64 <= row['repetitions'] <= 1024
                    for row in candidates), 'Invalid retained feasible count')
                require(len({row['evaluation_paths'] for row in candidates}) == len(candidates),
                        'Repeated evaluation size')
                frozen = {key: previous_choice['selected'][key] for key in ('evaluation_paths', 'repetitions')}
                regime_in = dict(accounting=account, scope=scope, candidates=candidates, retained_choice=frozen)
                regimes_in.append(regime_in)
                baseline = choice(candidates, original_weight)
                require(all(baseline[key] == frozen[key] for key in frozen), 'Original fixed allocation differs')
                selected = [dict(omitted_pair=row['omitted_pair'], status=row['status'],
                    choice=choice(candidates, row['share']) if row['share'] is not None else None)
                    for row in deletions]
                changes = [row['omitted_pair'] for row in selected if row['choice'] is not None
                           and row['choice']['evaluation_paths'] != baseline['evaluation_paths']]
                intervals, envelope, nodes, geometry_points, crossings = crossing_geometry(candidates)
                target = baseline['evaluation_paths']
                diagnostic_regret, diagnostic_points = exhaustive_regret(candidates, target, lower, upper, nodes)
                entire_regret, entire_points = exhaustive_regret(candidates, target, Fraction(0), Fraction(1), nodes)
                regimes_out.append(dict(**regime_in, retained_share_choice=baseline,
                    reconstructed_full_choice=choice(candidates, full['share']) if full['share'] is not None else None,
                    envelope=envelope, retained_choice_region=intervals[target], deletion_choices=selected,
                    changed_deletion_pairs=changes, diagnostic_hull=dict(lower=lower, upper=upper,
                        coverage_claim=False, unresolved_deletions=32-len(valid)),
                    retained_choice_regret_over_diagnostic_hull=diagnostic_regret,
                    retained_choice_regret_over_nonnegative_components=entire_regret))
                case_details.append(dict(accounting=account, scope=scope, candidate_sizes=len(candidates),
                    pairwise_crossings_in_unit_interval=crossings,
                    distinct_geometry_nodes=len(nodes), geometry_nodes_and_midpoints=geometry_points,
                    diagnostic_regret_nodes=diagnostic_points, nonnegative_regret_nodes=entire_points,
                    weak_optimality_regions=len(envelope), changed_deletions=len(changes)))
                total_regimes += 1
                total_geometry_points += geometry_points
                total_crossings += crossings
                total_regret_points += diagnostic_points+entire_points
        expected_input = dict(case=case, pairs=pairs, retained_A=a, retained_B=b, budget=budget, regimes=regimes_in)
        same(retained_input, expected_input, f'case {case} raw pilot, baseline and fixed cost regimes')
        reconstructed_inputs.append(expected_input)
        expected_case = dict(case=case, retained_A=a, retained_B=b, retained_share=original_weight,
            reconstructed_full=full, reconstruction_difference=discrepancy, deletions=deletions, regimes=regimes_out)
        same(read(run/f'case_{case:02d}.json'), expected_case, f'case {case} full component and geometry output')
        expected_summaries.append(dict(case=case, retained_share=original_weight,
            deletion_statuses={status: sum(row['status'] == status for row in deletions) for status in STATUSES},
            diagnostic_share_range=[lower, upper], regimes=[dict(accounting=row['accounting'], scope=row['scope'],
                retained_choice=row['retained_choice'], changed_deletions=len(row['changed_deletion_pairs']),
                diagnostic_maximum_regret=row['retained_choice_regret_over_diagnostic_hull']['maximum'],
                nonnegative_components_maximum_regret=row['retained_choice_regret_over_nonnegative_components']['maximum'])
                for row in regimes_out]))
        details.append(dict(case=case, original_pairs=len(pairs), signed_deletions=len(deletions),
                            signed_full_reconstruction=True, regimes=case_details))
        total_pairs += len(pairs)
        total_deletions += len(deletions)
        print(f'Case {case}: original moments, all whole-pair deletions and affine geometry verified', flush=True)
    same(payload, dict(models=reconstructed_inputs, reference_paths=4096, pilot_pairs=32,
                       retrospective=True, coverage_claim=False), 'complete inputs')
    expected_summary = dict(complete=True, cases=13, reconstructed_pairs=416, delete_one_diagnostics=416,
        fixed_cost_regimes=78, models=expected_summaries, new_simulations=0, fitted_policies=0,
        confidence_regions_constructed=0, retrospective=True, costs_and_original_allocations_unchanged=True,
        publication_clearance=False)
    same(read(run/'summary.json'), expected_summary, 'complete summary')
    require((total_pairs, total_deletions, total_regimes) == (416, 416, 78), 'Independent coverage differs')
    retain(Path(__file__))
    for name, digest in evidence.items():
        require(sha(ROOT/name) == digest, 'Evidence changed during audit')
    result = dict(complete=True, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        method='Independent standard-library Fraction reconstruction of raw paired summaries, all affine crossings and interval midpoints; no analysis kernel or runner imported.',
        cases=13, original_paired_summaries_reconstructed=total_pairs, signed_full_reconstructions=13,
        signed_delete_one_reconstructions=total_deletions, fixed_cost_regimes=total_regimes,
        pairwise_crossings_in_unit_interval_verified=total_crossings,
        geometry_nodes_and_midpoints_verified=total_geometry_points,
        regret_nodes_verified=total_regret_points,
        all_signed_components_statuses_and_shares_verified=True,
        all_weak_regions_and_endpoint_ownership_verified=True,
        all_baseline_and_deletion_choices_criteria_and_ties_verified=True,
        all_hull_and_unit_interval_regrets_verified=True,
        original_components_and_cost_candidate_counts_unchanged=True,
        full_and_deletion_moments_reconstructed_from_original_blocks=True,
        source_and_artifact_hashes_verified=True, exact_artifact_coverage_verified=True,
        rounding_error_enclosure=False, diagnostic_hull_is_confidence_region=False,
        statistical_coverage_claim=False, retrospective=True, paths_simulated=0, policies_fitted=0,
        publication_clearance=False, python=platform.python_version(), cases_details=details,
        evidence_sha256=evidence, artifact_sha256={})
    output.mkdir(parents=True, exist_ok=False)
    pending = output/'checks.pending.json'
    pending.write_text(json.dumps(encode(result), indent=2, allow_nan=False)+'\n')
    for name, digest in evidence.items():
        require(sha(ROOT/name) == digest, 'Evidence changed before audit completion')
    pending.rename(output/'checks.json')
    return {key: result[key] for key in ('complete', 'cases', 'signed_delete_one_reconstructions',
        'fixed_cost_regimes', 'geometry_nodes_and_midpoints_verified', 'regret_nodes_verified')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=BASE/'inputs_v1')
    parser.add_argument('--run', type=Path, default=BASE/'run_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.inputs, args.run, args.output), sort_keys=True))
