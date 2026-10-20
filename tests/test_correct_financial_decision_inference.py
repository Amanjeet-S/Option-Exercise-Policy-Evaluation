"""Deterministic fixtures for the historical exposure-inference correction.

I use exact literal endpoint intervals and small private temporary ledgers.
No random-number generator, model path or financial-price oracle is called.
"""
from copy import deepcopy
from fractions import Fraction
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(os.environ.get('QF_TEST_REPOSITORY', Path(__file__).resolve().parents[1])).resolve()
if not (ROOT/'scripts/certify_financial_decision.py').is_file():
    ROOT = Path.cwd().resolve()
if not (ROOT/'scripts/certify_financial_decision.py').is_file():
    raise RuntimeError('Run the fixtures from the local repository or set QF_TEST_REPOSITORY')
SOURCE = ROOT/'scripts/correct_financial_decision_inference.py'
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name('correct_financial_decision_inference.py')
specification = importlib.util.spec_from_file_location('financial_eligibility_correction_fixture', SOURCE)
correction = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = correction
specification.loader.exec_module(correction)
helper = correction.load_helper(ROOT/'scripts/certify_financial_decision.py')


def measured(cpu=1., wall=2.):
    return dict(cpu_seconds=cpu, wall_seconds=wall)


def amendment():
    return dict(schema_version=1, study_design_sha256=correction.DESIGN_SHA,
        exclusion=dict(fit_index=0, path_indices_start=0, path_indices_stop_exclusive=16,
            root_entropy=2026100410, dimensions_per_vector=5, dtype='float64',
            spawn_keys=[[4, j] for j in range(1, 10)]),
        fixed_looks=[dict(costed_paths_per_fit=n, eligible_paths_fit_0=n-16,
            eligible_paths_other_fits=n, eligible_paths_across_fits=8*n-16) for n in correction.LOOKS],
        probability_allocation=dict(alpha='1/20', delta='1/640', events=32),
        unchanged=dict(caps={'zero': '150', 'european_common': '50'}, tolerances=['1/2', '1'],
            all_generated_and_assessed_paths_charged=True,
            original_raw_observations_and_numerical_records_preserved=True,
            original_source_registration_preserved=True))


class ExactPartitionChecks(unittest.TestCase):
    def test_excluded_prefix_partitions_all_exact_statistics(self):
        records = [(Fraction(149), Fraction(149)), (Fraction(150), Fraction(150))]
        records += [(Fraction(x), Fraction(x)) for x in (1, 2, 3, 4)]
        accumulator = correction.EligibleStatistics(helper, Fraction(10), 2)
        for index, bounds in enumerate(records):
            accumulator.add(index, *bounds)
        accumulator.verify_partition()
        self.assertEqual((accumulator.full.count, accumulator.eligible.count, accumulator.excluded.count), (6, 4, 2))
        self.assertEqual(accumulator.full.sum_lower, 30)
        self.assertEqual(accumulator.eligible.sum_lower, 10)
        self.assertEqual(accumulator.excluded.sum_lower, 20)
        self.assertEqual(accumulator.raw_full_upper, 309)
        self.assertEqual(accumulator.raw_eligible_upper, 10)
        self.assertEqual(accumulator.eligible.sum_midpoint_squared, 30)
        self.assertEqual(accumulator.eligible.sum_radius_squared, 0)

    def test_clipping_boundary_retains_endpoint_uncertainty(self):
        accumulator = correction.EligibleStatistics(helper, Fraction(10), 1)
        accumulator.add(0, Fraction(0), Fraction(1, 1000))
        accumulator.add(1, Fraction(9999, 1000), Fraction(10001, 1000))
        accumulator.verify_partition()
        self.assertEqual(accumulator.eligible.sum_upper, 10)
        self.assertEqual(accumulator.eligible.sum_lower, Fraction(9999, 1000))
        self.assertEqual(accumulator.eligible.sum_radius_squared, Fraction(1, 4000000))
        self.assertEqual(accumulator.eligible.maximum_raw_width, Fraction(1, 500))
        self.assertEqual(accumulator.eligible.maximum_clipped_width, Fraction(1, 1000))

    def test_other_fit_retains_one_accumulator_without_exclusion(self):
        accumulator = correction.EligibleStatistics(helper, Fraction(50), 0)
        for index, value in enumerate((1, 2, 3, 4)):
            accumulator.add(index, Fraction(value), Fraction(value))
        accumulator.verify_partition()
        self.assertIs(accumulator.full, accumulator.eligible)
        self.assertEqual(accumulator.excluded.count, 0)
        self.assertEqual(accumulator.raw_full_lower, accumulator.raw_eligible_lower)

    def test_duplicate_reordered_and_invalid_gain_records_are_rejected(self):
        accumulator = correction.EligibleStatistics(helper, Fraction(50), 0)
        with self.assertRaises(ValueError):
            accumulator.add(1, Fraction(1), Fraction(1))
        with self.assertRaises(ValueError):
            accumulator.add(0, Fraction(-1), Fraction(0))
        with self.assertRaises(ValueError):
            accumulator.add(0, Fraction(0), Fraction(1, 50))
        accumulator.add(0, Fraction(1), Fraction(1))
        with self.assertRaises(ValueError):
            accumulator.add(0, Fraction(1), Fraction(1))
        self.assertEqual(accumulator.full.count, 1)


class AmendmentAndCostChecks(unittest.TestCase):
    def test_amendment_rejects_changed_mask_stream_caps_and_event_count(self):
        correction.validate_amendment(amendment())
        for section, key, value in [('exclusion', 'path_indices_stop_exclusive', 15),
            ('exclusion', 'root_entropy', 2026100411), ('probability_allocation', 'events', 16),
            ('unchanged', 'caps', {'zero': '150', 'european_common': '51'})]:
            changed = amendment()
            changed[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(ValueError):
                    correction.validate_amendment(changed)

    def test_first_acceptable_look_and_unresolved_maximum_are_charged(self):
        looks = []
        for raw, eligible, upper in [(4096, 4080, Fraction(3, 5)), (16384, 16368, Fraction(2, 5))]:
            looks.append(dict(costed_paths=raw, eligible_paths=eligible,
                upper_limit=dict(upper_limit=helper.rational_record(upper)),
                acceptance={str(t): upper <= t for t in correction.TOLERANCES},
                original_reconstructed_standalone_prefix_cost=measured(float(raw), float(raw)+1)))
        decisions = correction.choose_decisions(looks, helper)
        self.assertEqual(decisions['1/2']['required_costed_paths'], 16384)
        self.assertEqual(decisions['1']['required_costed_paths'], 4096)
        self.assertEqual(decisions['1']['required_eligible_paths'], 4080)
        for look in looks:
            look['acceptance'] = {'1/2': False, '1': False}
        unresolved = correction.choose_decisions(looks, helper)
        self.assertTrue(all(d['required_costed_paths'] == 16384 and d['full_budget_charged_if_unresolved'] for d in unresolved.values()))

    def test_after_run_cost_adds_correction_without_discarding_original_work(self):
        result = dict(original_actual_preparation_workflow_cost=measured(2., 3.),
            original_actual_assessment_workflow_cost=measured(7., 11.),
            assessments=[dict(decisions={'1': dict(original_required_prefix_cost=measured(5., 8.))})])
        original_cost = deepcopy(result['assessments'][0]['decisions']['1']['original_required_prefix_cost'])
        corrected = correction.attach_costs(result, measured(13., 17.))
        self.assertEqual(corrected['original_plus_correction_workflow_cost'], measured(22., 31.))
        decision = corrected['assessments'][0]['decisions']['1']
        self.assertEqual(decision['original_required_prefix_cost'], original_cost)
        self.assertEqual(decision['conservative_after_run_cost'], measured(18., 25.))


class StreamingFixtureChecks(unittest.TestCase):
    def fixture(self, folder, fit_index=0, arm='zero'):
        cap = correction.CAPS[arm]
        tail = dict(cap=helper.rational_record(cap), cap_split=dict(B=helper.rational_record(cap)),
            ideal_model_tail_bound_certified=True,
            clipping_tail_allowance=helper.interval_record(Fraction(0), Fraction(1, 100)))
        accumulator = correction.EligibleStatistics(helper, cap, 0)
        hasher = correction.hashlib.sha256()
        looks = []
        stream = bytearray()
        for index, value in enumerate((200, 200, 1, 2, 3, 4, 5, 6)):
            certificate = helper.interval_record(Fraction(value), Fraction(value))
            native = dict(completed=True, error=None, stopping_index=9, rights=10, dimension=5,
                precision_bits=96, arm='zero_martingale' if arm == 'zero' else 'coarse_common',
                gain_certificate=certificate, cost=dict(cpu_seconds=.1, wall_seconds=.2))
            row = dict(path_index=index, stopping_index=9, assessment=native, gain_certificate=certificate,
                native_ledger_sha256=correction.hashed(native),
                numerical_ledger_sha256=correction.hashed(correction.without_timing(native)))
            encoded = correction.canonical(row)+b'\n'
            stream.extend(encoded)
            hasher.update(encoded)
            accumulator.add(index, Fraction(value), Fraction(value))
            n = index+1
            if n in (4, 8):
                limit = accumulator.full.upper_limit(delta=correction.DELTA, tail_upper=Fraction(1, 100), precision=96)
                upper = helper.decode_rational(limit['upper_limit'])
                look = dict(paths=n, upper_limit=limit,
                    acceptance={str(t): upper <= t for t in correction.TOLERANCES},
                    raw_gain_mean_certificate=helper.interval_record(accumulator.raw_full_lower/n, accumulator.raw_full_upper/n),
                    clipped_gain_mean_certificate=helper.interval_record(accumulator.full.sum_lower/n, accumulator.full.sum_upper/n),
                    reconstructed_standalone_prefix_cost=measured(float(n), float(n)+1),
                    observation_prefix_sha256=hasher.hexdigest())
                looks.append(look)
                (folder/f'look_{n}.json').write_text(json.dumps(look, sort_keys=True))
        with (folder/'observations.jsonl.gz').open('xb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as compressed:
                compressed.write(stream)
        original = dict(completed=True, fit_index=fit_index, paths=8, assessor=arm,
            design_sha256=correction.DESIGN_SHA, tail_certificate_sha256=correction.hashed(tail),
            look_records=looks, uncompressed_observation_sha256=hasher.hexdigest(),
            compressed_observation_sha256=correction.digest(folder/'observations.jsonl.gz'),
            compressed_observation_bytes=(folder/'observations.jsonl.gz').stat().st_size,
            actual_complete_assessment_cost=measured(10., 15.))
        return original, tail, bytes(stream)

    def test_complete_stream_suffix_and_original_records_are_preserved(self):
        with TemporaryDirectory() as temporary, patch.object(correction, 'LOOKS', (4, 8)), patch.object(correction, 'EXPOSED_ROWS', 2):
            folder = Path(temporary)
            original, tail, original_stream = self.fixture(folder)
            original_bytes = correction.canonical(original)
            evidence_before = correction.digest(folder/'observations.jsonl.gz')
            result = correction.reconstruct_arm(folder, original, tail, 0, helper)
            self.assertEqual(result['original_completed_paths'], 8)
            self.assertEqual(result['eligible_completed_paths'], 6)
            self.assertEqual([look['eligible_paths'] for look in result['looks']], [2, 6])
            self.assertEqual(helper.decode_interval(result['looks'][1]['raw_gain_mean_certificate']), (Fraction(7, 2), Fraction(7, 2)))
            self.assertEqual(result['looks'][0]['original_reconstructed_standalone_prefix_cost'], original['look_records'][0]['reconstructed_standalone_prefix_cost'])
            self.assertEqual(correction.canonical(original), original_bytes)
            self.assertEqual(correction.digest(folder/'observations.jsonl.gz'), evidence_before)
            self.assertEqual(gzip.decompress((folder/'observations.jsonl.gz').read_bytes()), original_stream)
            for look in result['looks']:
                self.assertEqual(helper.decode_rational(look['upper_limit']['failure_probability']), Fraction(1, 640))
                self.assertEqual(look['upper_limit']['statistics']['count'], look['eligible_paths'])

    def test_other_fit_keeps_each_original_limit_verbatim(self):
        with TemporaryDirectory() as temporary, patch.object(correction, 'LOOKS', (4, 8)):
            folder = Path(temporary)
            original, tail, _ = self.fixture(folder, fit_index=1, arm='european_common')
            result = correction.reconstruct_arm(folder, original, tail, 1, helper)
            self.assertEqual(result['eligible_completed_paths'], 8)
            self.assertEqual(result['excluded_completed_paths'], 0)
            for original_look, corrected_look in zip(original['look_records'], result['looks']):
                self.assertEqual(corrected_look['upper_limit'], original_look['upper_limit'])

    def test_truncated_extra_reordered_and_changed_native_streams_fail(self):
        for mode in ('truncated', 'extra', 'reordered', 'changed_native'):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary, patch.object(correction, 'LOOKS', (4, 8)), patch.object(correction, 'EXPOSED_ROWS', 2):
                folder = Path(temporary)
                original, tail, stream = self.fixture(folder)
                rows = [json.loads(line) for line in stream.splitlines()]
                if mode == 'truncated':
                    rows.pop()
                elif mode == 'extra':
                    extra = deepcopy(rows[-1]); extra['path_index'] = 8; rows.append(extra)
                elif mode == 'reordered':
                    rows[0], rows[1] = rows[1], rows[0]
                else:
                    rows[0]['assessment']['stopping_index'] = 8
                with gzip.open(folder/'observations.jsonl.gz', 'wb') as destination:
                    destination.write(b''.join(correction.canonical(row)+b'\n' for row in rows))
                with self.assertRaises(ValueError):
                    correction.reconstruct_arm(folder, original, tail, 0, helper)


if __name__ == '__main__':
    unittest.main()
