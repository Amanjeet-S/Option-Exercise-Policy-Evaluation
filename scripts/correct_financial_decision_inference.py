"""Reconstruct eligible-suffix inference after the recorded stream exposure.

I preserve the registered computation and its original numerical records.
The dated amendment excludes sixteen previously exposed fit-0 rows solely
from conditional inference. Every generated and assessed row remains charged.
No path simulation, conditional price integration or policy fitting is performed.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


DESIGN_SHA = 'e4a5bf17934f3f4314dcae94ba8e06e3b259c3f051043214a309f5d530ba3149'
AMENDMENT_SHA = 'd4fc21aaeb48d2d3cda0b189a3e9ca27d633d98558834be874d52bd365bca5ce'
LOOKS = (4096, 16384)
ARMS = ('zero', 'european_common')
CAPS = {'zero': Fraction(150), 'european_common': Fraction(50)}
TOLERANCES = (Fraction(1, 2), Fraction(1))
DELTA = Fraction(1, 640)
EXPOSED_FIT = 0
EXPOSED_ROWS = 16
PRECISION = 96
TIMING_FIELDS = {'complete_path_cost', 'complete_query_cost', 'complete_attempt_cost', 'cost'}
SUM_FIELDS = ('sum_lower', 'sum_upper', 'sum_midpoint', 'sum_midpoint_squared', 'sum_radius_squared')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hashed(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clocks():
    return time.process_time(), time.perf_counter()


def elapsed(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def valid_cost(value):
    require(isinstance(value, dict) and set(value) == {'cpu_seconds', 'wall_seconds'}, 'A measured cost requires both clocks')
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in value.values()), 'A measured cost must be finite and non-negative')
    return value


def sum_cost(*values):
    result = dict(cpu_seconds=0., wall_seconds=0.)
    for value in values:
        valid_cost(value)
        for clock in result:
            result[clock] += value[clock]
    return result


def without_timing(value):
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key not in TIMING_FIELDS}
    return value


def load_helper(path):
    specification = importlib.util.spec_from_file_location('eligible_financial_decision_helper', path)
    require(specification is not None and specification.loader is not None, 'The frozen arithmetic helper is unavailable')
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class EligibleStatistics:
    """Partition all ordered records without inspecting outcomes for eligibility."""

    def __init__(self, helper, cap, excluded_prefix):
        require(type(excluded_prefix) is int and excluded_prefix >= 0, 'A non-negative fixed exposure prefix is required')
        self.helper, self.excluded_prefix = helper, excluded_prefix
        self.full = helper.IntervalClippedAccumulator(cap)
        self.eligible = self.full if excluded_prefix == 0 else helper.IntervalClippedAccumulator(cap)
        self.excluded = helper.IntervalClippedAccumulator(cap)
        self.raw_full_lower = self.raw_full_upper = Fraction(0)
        self.raw_eligible_lower = self.raw_eligible_upper = Fraction(0)

    def add(self, index, lower, upper):
        require(type(index) is int and index == self.full.count, 'Observation indices must form the complete original ordered prefix')
        require(0 <= lower <= upper and upper-lower <= Fraction(1, 100), 'A retained gain must be non-negative and meet the registered numerical width')
        self.full.add_interval(lower, upper)
        self.raw_full_lower += lower
        self.raw_full_upper += upper
        if index < self.excluded_prefix:
            self.excluded.add_interval(lower, upper)
        else:
            if self.eligible is not self.full:
                self.eligible.add_interval(lower, upper)
            self.raw_eligible_lower += lower
            self.raw_eligible_upper += upper

    def verify_partition(self):
        require(self.full.count == self.eligible.count+self.excluded.count, 'The eligibility partition loses an original observation')
        for field in SUM_FIELDS:
            require(getattr(self.full, field) == getattr(self.eligible, field)+getattr(self.excluded, field), 'Exact clipped statistics do not partition: '+field)
        for field in ('maximum_raw_width', 'maximum_clipped_width'):
            require(getattr(self.full, field) == max(getattr(self.eligible, field), getattr(self.excluded, field)), 'Maximum widths do not partition')


def validate_amendment(amendment):
    require(amendment['schema_version'] == 1 and amendment['study_design_sha256'] == DESIGN_SHA, 'An amendment must refer to the original registered design')
    exclusion = amendment['exclusion']
    require(exclusion['fit_index'] == EXPOSED_FIT and exclusion['path_indices_start'] == 0 and
        exclusion['path_indices_stop_exclusive'] == EXPOSED_ROWS, 'The reviewed exclusion cannot change')
    require(exclusion['root_entropy'] == 2026100410 and exclusion['dimensions_per_vector'] == 5 and
        exclusion['dtype'] == 'float64' and exclusion['spawn_keys'] == [[4, j] for j in range(1, 10)], 'The reviewed exposure stream identity cannot change')
    expected_looks = [dict(costed_paths_per_fit=n, eligible_paths_fit_0=n-EXPOSED_ROWS,
        eligible_paths_other_fits=n, eligible_paths_across_fits=8*n-EXPOSED_ROWS) for n in LOOKS]
    require(amendment['fixed_looks'] == expected_looks, 'The original costed looks and eligible counts cannot change')
    require(amendment['probability_allocation'] == dict(alpha='1/20', delta='1/640', events=32), 'The simultaneous probability allocation cannot change')
    require(amendment['unchanged']['caps'] == {'zero': '150', 'european_common': '50'} and
        amendment['unchanged']['tolerances'] == ['1/2', '1'], 'Caps and financial tolerances cannot change')
    for name in ('all_generated_and_assessed_paths_charged', 'original_raw_observations_and_numerical_records_preserved', 'original_source_registration_preserved'):
        require(amendment['unchanged'][name] is True, 'Original records and costs must remain preserved')


def source_snapshot(root, retained, design, amendment, amendment_path):
    repository_sources = {name: digest(root/name) for name in design['source_sha256']}
    retained_inputs = {name: digest(retained/name) for name in design['retained_input_sha256']}
    require(repository_sources == design['source_sha256'] and retained_inputs == design['retained_input_sha256'], 'A registered source or retained policy input has changed')
    require({name: repository_sources[name] for name in amendment['source_sha256']} == amendment['source_sha256'], 'The exposure amendment source inventory has changed')
    inventories = {name: digest(amendment_path.parent/name) for name in amendment['inventory_sha256']}
    require(inventories == amendment['inventory_sha256'], 'A reviewed exposure inventory has changed')
    return dict(source_sha256=repository_sources, retained_input_sha256=retained_inputs,
        exposure_inventory_sha256=inventories, amendment_sha256=digest(amendment_path),
        correction_program_sha256=digest(Path(__file__)))


def verify_original_look(look, partition, helper, tail_upper):
    full = partition.full
    require(look['paths'] == full.count and look['upper_limit']['statistics'] == full.snapshot(), 'Original sufficient statistics do not reconstruct from their complete prefix')
    expected = full.upper_limit(delta=DELTA, tail_upper=tail_upper, precision=PRECISION)
    require({key: value for key, value in expected.items() if key != 'work'} ==
        {key: value for key, value in look['upper_limit'].items() if key != 'work'}, 'The original outward expression does not replay exactly')
    require(helper.decode_interval(look['raw_gain_mean_certificate']) ==
        (partition.raw_full_lower/full.count, partition.raw_full_upper/full.count), 'The original raw mean does not reconstruct')
    require(helper.decode_interval(look['clipped_gain_mean_certificate']) ==
        (full.sum_lower/full.count, full.sum_upper/full.count), 'The original clipped mean does not reconstruct')
    authoritative = helper.decode_rational(look['upper_limit']['upper_limit'])
    require(look['acceptance'] == {str(tolerance): authoritative <= tolerance for tolerance in TOLERANCES}, 'Original acceptance does not use its exact upper endpoint')
    valid_cost(look['reconstructed_standalone_prefix_cost'])
    return expected


def corrected_look(look, partition, helper, tail_upper):
    partition.verify_partition()
    n = partition.eligible.count
    require(n >= 2, 'The eligible suffix must support empirical Bernstein')
    if partition.excluded_prefix:
        limit = partition.eligible.upper_limit(delta=DELTA, tail_upper=tail_upper, precision=PRECISION)
        origin = 'Eligible-suffix reconstruction after the recorded amendment'
    else:
        limit = look['upper_limit']
        origin = 'Unchanged original arithmetic record, independently replayed'
    authoritative = helper.decode_rational(limit['upper_limit'])
    return dict(costed_paths=look['paths'], eligible_paths=n, excluded_paths=partition.excluded.count,
        upper_limit=limit, limit_origin=origin,
        acceptance={str(tolerance): authoritative <= tolerance for tolerance in TOLERANCES},
        clipped_gain_mean_certificate=helper.interval_record(partition.eligible.sum_lower/n, partition.eligible.sum_upper/n),
        raw_gain_mean_certificate=helper.interval_record(partition.raw_eligible_lower/n, partition.raw_eligible_upper/n),
        excluded_statistics=partition.excluded.snapshot(),
        original_reconstructed_standalone_prefix_cost=look['reconstructed_standalone_prefix_cost'],
        original_observation_prefix_sha256=look['observation_prefix_sha256'],
        original_upper_limit_retained_as_arithmetic_record=look['upper_limit'],
        conditional_inference_scope='The eligible suffix is independent of the pre-run exposure information under the declared ideal independent-innovation premise. Eligibility follows solely from the pinned exposure inventory. Numerical endpoints do not certify the floating-point sampling law or unrounded ideal paths.')


def choose_decisions(looks, helper):
    require([look['costed_paths'] for look in looks] == list(LOOKS), 'Both original registered looks must be retained')
    decisions = {}
    for tolerance in TOLERANCES:
        accepted = [look for look in looks if look['acceptance'][str(tolerance)]]
        chosen = accepted[0] if accepted else looks[-1]
        decisions[str(tolerance)] = dict(accepted=bool(accepted), required_costed_paths=chosen['costed_paths'],
            required_eligible_paths=chosen['eligible_paths'], upper_limit=chosen['upper_limit']['upper_limit'],
            original_required_prefix_cost=chosen['original_reconstructed_standalone_prefix_cost'],
            full_budget_charged_if_unresolved=not bool(accepted))
    return decisions


def reconstruct_arm(folder, original, tail, fit_index, helper):
    start = clocks()
    require(original['completed'] is True and original['fit_index'] == fit_index and original['paths'] == LOOKS[-1], 'A complete original assessor is required')
    arm = original['assessor']
    require(arm in ARMS and original['design_sha256'] == DESIGN_SHA, 'An original assessor identity changed')
    require(original['tail_certificate_sha256'] == hashed(tail), 'The original assessor tail certificate changed')
    require(tail['ideal_model_tail_bound_certified'] is True, 'A deterministic ideal-model population tail is required')
    cap = helper.decode_rational(tail['cap'] if arm == 'zero' else tail['cap_split']['B'])
    require(cap == CAPS[arm], 'The original clipping cap changed')
    tail_lower, tail_upper = helper.decode_interval(tail['clipping_tail_allowance'])
    require(tail_lower == 0 and tail_upper >= 0, 'A non-negative population allowance is required')
    original_looks = {look['paths']: look for look in original['look_records']}
    require(list(original_looks) == list(LOOKS), 'The original look schedule changed')
    partition = EligibleStatistics(helper, cap, EXPOSED_ROWS if fit_index == EXPOSED_FIT else 0)
    prefix_hash = hashlib.sha256()
    corrected = []
    for count, look in original_looks.items():
        require(read(folder/f'look_{count}.json') == look, 'An original standalone look differs from its completion record')
    with gzip.open(folder/'observations.jsonl.gz', 'rb') as source:
        for encoded in source:
            row = json.loads(encoded)
            require(encoded == canonical(row)+b'\n', 'An observation must retain its canonical lossless encoding')
            index = row['path_index']
            require(type(index) is int and index < LOOKS[-1], 'An original path index exceeds the registered maximum')
            require(type(row['stopping_index']) is int and 1 <= row['stopping_index'] <= 9, 'A retained stopping index leaves the calendar')
            native = row['assessment']
            require(native['completed'] is True and native['error'] is None and
                native['stopping_index'] == row['stopping_index'] and native['rights'] == 10 and
                native['dimension'] == 5 and native['precision_bits'] == PRECISION and
                native['arm'] == ('zero_martingale' if arm == 'zero' else 'coarse_common'),
                'A native ledger changes completion, geometry, arithmetic or the stopping decision')
            require(row['native_ledger_sha256'] == hashed(native) and
                row['numerical_ledger_sha256'] == hashed(without_timing(native)), 'A native or timing-free numerical ledger identity changed')
            require(row['gain_certificate'] == native['gain_certificate'], 'A compact gain differs from its full native ledger')
            lower, upper = helper.decode_interval(row['gain_certificate'])
            partition.add(index, lower, upper)
            prefix_hash.update(encoded)
            count = partition.full.count
            if count in original_looks:
                look = original_looks[count]
                require(prefix_hash.hexdigest() == look['observation_prefix_sha256'], 'An original observation prefix identity changed')
                verify_original_look(look, partition, helper, tail_upper)
                amended = corrected_look(look, partition, helper, tail_upper)
                expected_count = count-(EXPOSED_ROWS if fit_index == EXPOSED_FIT else 0)
                require(amended['eligible_paths'] == expected_count, 'The corrected look has an unexpected eligible count')
                corrected.append(amended)
    require(partition.full.count == LOOKS[-1] and prefix_hash.hexdigest() == original['uncompressed_observation_sha256'], 'The complete original observation stream is missing or changed')
    require(digest(folder/'observations.jsonl.gz') == original['compressed_observation_sha256'], 'The original compressed observation artifact changed')
    require((folder/'observations.jsonl.gz').stat().st_size == original['compressed_observation_bytes'], 'The compressed observation length changed')
    partition.verify_partition()
    result = dict(fit_index=fit_index, assessor=arm, original_completed_paths=partition.full.count,
        eligible_completed_paths=partition.eligible.count, excluded_completed_paths=partition.excluded.count,
        looks=corrected, decisions=choose_decisions(corrected, helper),
        original_actual_complete_assessment_cost=original['actual_complete_assessment_cost'],
        exact_full_statistics=partition.full.snapshot(), exact_eligible_statistics=partition.eligible.snapshot(),
        exact_excluded_statistics=partition.excluded.snapshot(),
        original_compressed_observation_sha256=original['compressed_observation_sha256'],
        original_uncompressed_observation_sha256=original['uncompressed_observation_sha256'],
        correction_and_reconstruction_cost=elapsed(start))
    valid_cost(result['original_actual_complete_assessment_cost'])
    return result


def assessment_snapshot(assessment):
    names = ['summary.json', 'attempt.json', 'zero_tail.json', 'european_common_tail.json']
    names.extend(f'fit_{index:02d}/{arm}/{filename}' for index in range(8) for arm in ARMS
        for filename in ('completion.json', 'look_4096.json', 'look_16384.json', 'observations.jsonl.gz'))
    return {name: digest(assessment/name) for name in names}


def correct(args):
    root, study, assessed, output, amendment_path = (Path(value).resolve() for value in
        (args.root, args.study, args.assessment, args.output, args.amendment))
    private = (root/'data/private').resolve()
    require(all(path.is_relative_to(private) for path in (study, assessed, output, amendment_path)), 'Inputs and output must remain in the local private project')
    require(not output.exists(), 'An inference correction cannot overwrite an existing attempt')
    require(not any(output.is_relative_to(path) or path.is_relative_to(output) for path in (study, assessed, amendment_path.parent)), 'Correction output must be separate from the original experiment and amendment')
    require(args.amendment_sha256 == AMENDMENT_SHA and digest(amendment_path) == AMENDMENT_SHA, 'The independently recorded amendment hash must match exactly')
    require(digest(study/'design.json') == DESIGN_SHA, 'The original registered financial design changed')
    amendment, design, summary = read(amendment_path), read(study/'design.json'), read(assessed/'summary.json')
    validate_amendment(amendment)
    require(not list(assessed.rglob('*failure.json')), 'A failed original attempt cannot be corrected as a completed study')
    require(summary['completed'] is True and summary['completed_fits'] == 8 and summary['design_sha256'] == DESIGN_SHA and
        summary['fixed_upper_events'] == 32 and summary['family_failure_probability'] == '1/20', 'The complete original study is required')
    require(design['root'] == str(root) and [fit['index'] for fit in design['fits']] == list(range(8)) and
        [fit['root_seed'] for fit in design['fits']] == list(range(2026100410, 2026100418)), 'The frozen fit cohort changed')
    retained = Path(design['retained_study']).resolve()
    require(retained.is_relative_to(private), 'Retained policy inputs must remain private')
    before = source_snapshot(root, retained, design, amendment, amendment_path)
    require(before['amendment_sha256'] == AMENDMENT_SHA, 'The amendment changed during initial verification')
    inputs_before = assessment_snapshot(assessed)
    require(summary['source_snapshot_before'] == summary['source_snapshot_after'] ==
        {'source_sha256': before['source_sha256'], 'retained_input_sha256': before['retained_input_sha256']}, 'The original assessment source snapshots do not match the sealed inputs')
    helper = load_helper(root/'scripts/certify_financial_decision.py')
    tails = {arm: read(assessed/(arm+'_tail.json')) for arm in ARMS}
    results = []
    require([fit['fit_index'] for fit in summary['fits']] == list(range(8)), 'Original fit results are missing or reordered')
    for fit in summary['fits']:
        index = fit['fit_index']
        require(set(fit['assessors']) == set(ARMS), 'Both original assessors are required')
        for arm in ARMS:
            folder = assessed/f'fit_{index:02d}'/arm
            original = read(folder/'completion.json')
            require(original == fit['assessors'][arm], 'A completion record differs from the original workflow summary')
            results.append(reconstruct_arm(folder, original, tails[arm], index, helper))
        print(f'Reconstructed eligible inference for fit {index}', flush=True)
    after = source_snapshot(root, retained, design, amendment, amendment_path)
    inputs_after = assessment_snapshot(assessed)
    require(after == before and inputs_after == inputs_before and digest(study/'design.json') == DESIGN_SHA,
        'A registered source, amendment, policy input or original numerical record changed during correction')
    return dict(schema_version=1, author='Amanjeet Singh', completed=True,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        status='Post-registration corrected conditional inference with original numerical records retained',
        design_sha256=DESIGN_SHA, amendment_sha256=AMENDMENT_SHA,
        amendment_recorded_at_utc=amendment['recorded_at_utc'], original_assessment_summary_sha256=inputs_before['summary.json'],
        correction_source_snapshot_before=before, correction_source_snapshot_after=after,
        original_assessment_artifact_sha256_before=inputs_before, original_assessment_artifact_sha256_after=inputs_after,
        excluded_fit_index=EXPOSED_FIT, excluded_prefix_rows=EXPOSED_ROWS,
        fixed_costed_looks=list(LOOKS), eligible_paths_across_fits=[8*n-EXPOSED_ROWS for n in LOOKS],
        fixed_upper_events=32, failure_probability_per_event='1/640', family_failure_probability='1/20',
        original_observations_reconstructed=8*2*LOOKS[-1], eligible_observations_at_final_look=2*(8*LOOKS[-1]-EXPOSED_ROWS),
        excluded_observations_at_final_look=2*EXPOSED_ROWS, corrected_look_records=32,
        changed_look_records=4, unchanged_look_records=28, assessments=results,
        original_actual_preparation_workflow_cost=summary['actual_preparation_workflow_cost'],
        original_actual_assessment_workflow_cost=summary['actual_assessment_workflow_cost'],
        originals_changed=False, paths_simulated=False, policies_fitted=False,
        conditional_pricing_integrations_performed=0, original_computational_auditor_changed=False,
        inference_scope=amendment['scope'], selection_basis=amendment['selection_basis'],
        numerical_scope='I reconstruct exact complete-prefix statistics, eligible-suffix statistics and the frozen 96-bit outward confidence construction. Native and numerical hashes are checked. This correction does not replace the separate stopped-gain, pricing-trace and bridge audit.',
        sampling_law_certified=False, unrounded_ideal_path_enclosure_established=False,
        market_validation_established=False, policy_improved=False,
        substantive_original_research_contribution_established=False)


def attach_costs(result, correction_cost):
    """Keep original charges and add the full amendment workflow conservatively."""
    valid_cost(correction_cost)
    result['correction_workflow_cost'] = correction_cost
    result['original_plus_correction_workflow_cost'] = sum_cost(
        result['original_actual_preparation_workflow_cost'], result['original_actual_assessment_workflow_cost'], correction_cost)
    for assessment in result['assessments']:
        for decision in assessment['decisions'].values():
            decision['conservative_after_run_cost'] = sum_cost(decision['original_required_prefix_cost'], correction_cost)
            decision['correction_workflow_charge'] = correction_cost.copy()
    result['cost_scope'] = ('All original generated and assessed rows remain charged, including the exposed prefix. '
        'Original required-prefix charges are preserved verbatim. I add the entire measured correction workflow '
        'to each decision as a conservative after-run reconstruction, rather than a chronological stopping cost '
        'or an allocation of shared verification costs. The whole-study cost counts the original preparation, '
        'original assessment and correction workflows once each; historical training remains in the original '
        'standalone charges. Independent audit costs and final summary serialisation are not included.')
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ('root', 'study', 'assessment', 'amendment', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--amendment-sha256', required=True)
    args = parser.parse_args()
    start = clocks()
    output = args.output.resolve()
    try:
        result = attach_costs(correct(args), elapsed(start))
        output.mkdir(parents=True, exist_ok=False)
        with (output/'corrected_summary.json').open('x') as destination:
            json.dump(result, destination, sort_keys=True, indent=2, allow_nan=False)
            destination.write('\n')
        print(json.dumps(dict(completed=True, eligible_observations=result['eligible_observations_at_final_look'],
            amendment_sha256=AMENDMENT_SHA, corrected_summary_sha256=digest(output/'corrected_summary.json'))), flush=True)
    except Exception as error:
        private = (args.root.resolve()/'data/private').resolve()
        excluded = (args.study.resolve(), args.assessment.resolve(), args.amendment.resolve().parent)
        safe = output.is_relative_to(private) and not any(output.is_relative_to(path) or path.is_relative_to(output) for path in excluded)
        if safe and not output.exists():
            output.mkdir(parents=True)
            with (output/'correction_failure.json').open('x') as destination:
                json.dump(dict(author='Amanjeet Singh', completed=False, amendment_sha256=AMENDMENT_SHA,
                    correction_program_sha256=digest(Path(__file__)), correction_cost=elapsed(start),
                    exception_type=type(error).__name__, message=str(error), originals_changed=False), destination,
                    sort_keys=True, indent=2, allow_nan=False)
                destination.write('\n')
        raise


if __name__ == '__main__':
    main()
