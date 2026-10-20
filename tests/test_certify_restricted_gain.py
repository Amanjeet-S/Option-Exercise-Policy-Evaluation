"""Arithmetic and domain checks for the retained-calendar gain calculation."""
from collections.abc import Sequence
from fractions import Fraction as F
from itertools import product
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from certify_restricted_gain import gain_interval_without_zero_initial_right as gain


class UnreadInitialConditionals(Sequence):
    def __init__(self, tail):
        self.tail = tuple(tail)
        self.reads = []

    def __len__(self):
        return 2 + len(self.tail)

    def __getitem__(self, index):
        if index in (0, 1):
            raise AssertionError('An initial conditional placeholder was accessed')
        self.reads.append(index)
        return self.tail[index - 2]


def full_span_interval(rewards, values, conditionals, stopping):
    """Independent full-calendar reference via cancelled cumulative intervals."""
    result = []
    count = len(rewards)
    # Prefix sums are formed separately, then shared intervals are cancelled
    # before any endpoint subtraction. This reference includes initial date 0.
    v_prefix = [F(0)]
    lo_prefix = [F(0)]
    hi_prefix = [F(0)]
    for index in range(1, count):
        v_prefix.append(v_prefix[-1] + values[index])
        lo_prefix.append(lo_prefix[-1] + conditionals[index][0])
        hi_prefix.append(hi_prefix[-1] + conditionals[index][1])
    for date in range(count):
        if date == stopping:
            result.append((F(0), F(0)))
            continue
        lo = rewards[date][0] - rewards[stopping][1]
        hi = rewards[date][1] - rewards[stopping][0]
        if date > stopping:
            realised = v_prefix[date] - v_prefix[stopping]
            lo += -realised + lo_prefix[date] - lo_prefix[stopping]
            hi += -realised + hi_prefix[date] - hi_prefix[stopping]
        else:
            realised = v_prefix[stopping] - v_prefix[date]
            lo += realised - hi_prefix[stopping] + hi_prefix[date]
            hi += realised - lo_prefix[stopping] + lo_prefix[date]
        result.append((lo, hi))
    return max(lo for lo, _ in result), max(hi for _, hi in result)


def exact_truth(rewards, values, conditionals, stopping, *, first_candidate=1):
    martingale = [F(0)]
    for index in range(1, len(rewards)):
        martingale.append(martingale[-1] + values[index] - conditionals[index])
    return max(rewards[date] - martingale[date] for date in range(first_candidate, len(rewards))) - rewards[stopping] + martingale[stopping]


class RetainedCalendarGainTests(unittest.TestCase):
    def setUp(self):
        self.rewards = [(F(0), F(0)), (F(2), F(3)), (F(6), F(7)), (F(1), F(2))]
        self.values = [F(0), F(-7), F(4), F(-2)]
        self.conditionals = [None, None, (F(3), F(5)), (F(-3), F(-1))]

    def test_first_middle_and_last_stop_have_literal_span_bounds(self):
        self.assertEqual(gain(self.rewards, self.values, self.conditionals, 1), (F(2), F(6)))
        self.assertEqual(gain(self.rewards, self.values, self.conditionals, 2), (F(0), F(0)))
        self.assertEqual(gain(self.rewards, self.values, self.conditionals, 3), (F(3), F(7)))

    def test_initial_conditional_placeholders_are_never_read(self):
        for tau in (1, 2, 3):
            rows = UnreadInitialConditionals(self.conditionals[2:])
            self.assertEqual(gain(self.rewards, self.values, rows, tau),
                             gain(self.rewards, self.values, self.conditionals, tau))
            self.assertEqual(rows.reads, [2, 3])

    def test_arbitrary_first_increment_has_no_effect(self):
        for tau in (1, 2, 3):
            expected = gain(self.rewards, self.values, self.conditionals, tau)
            for first in (None, object(), (float('-inf'), float('inf')), (float('nan'), 0),
                          (F(-10**12), F(10**12)), (F(9), F(-9))):
                rows = [None, first, *self.conditionals[2:]]
                self.assertEqual(gain(self.rewards, self.values, rows, tau), expected)

    def test_first_realised_increment_cancels_from_retained_differences(self):
        for tau in (1, 2, 3):
            expected = gain(self.rewards, self.values, self.conditionals, tau)
            for first in (F(-10**9), F(0), F(3, 17), F(10**9)):
                values = [self.values[0], first, *self.values[2:]]
                self.assertEqual(gain(self.rewards, values, self.conditionals, tau), expected)

    def test_one_retained_right_has_exact_zero_gain(self):
        rows = UnreadInitialConditionals([])
        self.assertEqual(gain([(0, 0), (0, 100)], [99, -31], rows, 1), (F(0), F(0)))
        self.assertEqual(rows.reads, [])

    def test_later_zero_reward_right_is_retained(self):
        self.assertEqual(gain([(0, 0), (0, 0), (0, 0)], [0, 0, -2], [None, None, (0, 0)], 1),
                         (F(2), F(2)))

    def test_required_second_conditional_row_is_mandatory(self):
        for bad in (None, (), (0,), (3, 2), (float('-inf'), 3), (0, float('nan'))):
            rows = [None, None, bad, self.conditionals[3]]
            with self.subTest(row=bad), self.assertRaises(ValueError):
                gain(self.rewards, self.values, rows, 1)

    def test_later_required_conditional_row_cannot_be_omitted(self):
        rows = list(self.conditionals)
        rows[3] = None
        for tau in (1, 2, 3):
            with self.subTest(stop=tau), self.assertRaises(ValueError):
                gain(self.rewards, self.values, rows, tau)

    def test_positive_or_nonzero_initial_reward_is_rejected(self):
        for initial in ((0, 1), (1, 1), (-1, 0), (-1, -1), (F(1, 10**30), F(1, 10**30))):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                gain([initial, *self.rewards[1:]], self.values, self.conditionals, 1)

    def test_negative_reward_lower_endpoint_is_rejected(self):
        for index in (1, 2, 3):
            rewards = list(self.rewards)
            rewards[index] = (F(-1, 100), F(2))
            with self.subTest(index=index), self.assertRaises(ValueError):
                gain(rewards, self.values, self.conditionals, 2)

    def test_invalid_stopping_indices_are_rejected(self):
        for tau in (0, -1, 4, 100, True, False, 1., F(1), '1', None):
            with self.subTest(stop=tau), self.assertRaises(ValueError):
                gain(self.rewards, self.values, self.conditionals, tau)

    def test_calendar_and_full_array_lengths_are_checked(self):
        for rewards, values, rows in (([], [], []), ([(0, 0)], [0], [None]),
                                     (self.rewards, self.values[:-1], self.conditionals),
                                     (self.rewards, self.values, self.conditionals[:-1]),
                                     (self.rewards, self.values, self.conditionals + [None])):
            with self.subTest(lengths=(len(rewards), len(values), len(rows))), self.assertRaises(ValueError):
                gain(rewards, values, rows, 1)
        for calendar in ((0, 1, 2), (0, 1, 1, 3), (0, 2, 1, 3), (-1, 0, 1, 2),
                         (0, 1, 2, float('inf')), (0, True, 2, 3)):
            with self.subTest(calendar=calendar), self.assertRaises(ValueError):
                gain(self.rewards, self.values, self.conditionals, 1, exercise_calendar=calendar)
        self.assertEqual(gain(self.rewards, self.values, self.conditionals, 1,
                              exercise_calendar=(0, F(1, 3), F(2, 3), 1)), (F(2), F(6)))

    def test_nonfinite_realised_values_and_invalid_reward_rows_are_rejected(self):
        for value in (float('inf'), float('nan'), True, '3'):
            values = list(self.values)
            values[2] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                gain(self.rewards, values, self.conditionals, 1)
        for row in (None, (1,), (3, 2), (0, float('inf')), (0, True)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                gain([self.rewards[0], row, *self.rewards[2:]], self.values, self.conditionals, 1)

    def test_binary_float_inputs_have_exact_fraction_semantics(self):
        self.assertEqual(gain([(0., 0.), (.1, .1), (.2, .2)], [0., 0., .05],
                              [None, None, (.03, .03)], 1),
                         (F(.2) - F(.1) - F(.05) + F(.03),) * 2)

    def test_full_calendar_endpoints_dominate_subset_endpoints(self):
        for tau in (1, 2, 3):
            restricted = gain(self.rewards, self.values, self.conditionals, tau)
            for first in ((F(-50), F(-49)), (F(0), F(0)), (F(48), F(52))):
                rows = [(F(0), F(0)), first, *self.conditionals[2:]]
                full = full_span_interval(self.rewards, self.values, rows, tau)
                self.assertLessEqual(restricted[0], full[0])
                self.assertLessEqual(restricted[1], full[1])

    def test_removing_initial_candidate_can_strictly_lower_the_dual_gain(self):
        rewards = [(F(0), F(0)), (F(1), F(1)), (F(2), F(2))]
        values = [F(0), F(100), F(1)]
        rows = [(F(0), F(0)), (F(0), F(0)), (F(1), F(1))]
        self.assertEqual(gain(rewards, values, rows, 1), (F(1), F(1)))
        self.assertEqual(full_span_interval(rewards, values, rows, 1), (F(99), F(99)))

    def test_endpoint_dominance_does_not_imply_width_dominance(self):
        rewards = [(F(0), F(0)), (F(1), F(1)), (F(1), F(21))]
        values = [F(0), F(101), F(0)]
        rows = [(F(0), F(0)), (F(0), F(0)), (F(0), F(0))]
        restricted = gain(rewards, values, rows, 1)
        complete = full_span_interval(rewards, values, rows, 1)
        self.assertEqual(restricted, (F(0), F(20)))
        self.assertEqual(complete, (F(100), F(100)))
        self.assertLessEqual(restricted[0], complete[0])
        self.assertLessEqual(restricted[1], complete[1])
        self.assertGreater(restricted[1] - restricted[0], complete[1] - complete[0])

    def test_every_enumerated_feasible_truth_is_enclosed(self):
        for tau in (1, 2, 3):
            lower, upper = gain(self.rewards, self.values, self.conditionals, tau)
            for realised_rewards in product(*self.rewards):
                for tail_conditionals in product(*self.conditionals[2:]):
                    for first in (F(-100), F(0), F(100)):
                        conditionals = [F(0), first, *tail_conditionals]
                        truth = exact_truth(realised_rewards, self.values, conditionals, tau)
                        self.assertLessEqual(lower, truth)
                        self.assertLessEqual(truth, upper)

    def test_general_signed_polynomial_and_hinge_values_are_accepted(self):
        state = (F(80), F(140))
        polynomial = F(3, 2) - F(2, 5) * state[0]**2 + F(1, 1000) * state[0] * state[1]**2
        hinge = F(-7) + F(3, 10) * max(max(state) - 100, F(0))
        values = [F(0), F(19), polynomial, hinge]
        rows = [None, None, (polynomial - F(1, 4), polynomial + F(1, 4)),
                (hinge - F(1, 4), hinge + F(1, 4))]
        result = gain(self.rewards, values, rows, 1)
        self.assertEqual(result, (F(11, 4), F(21, 4)))
        self.assertTrue(all(type(value) is F for value in result))


class RetainedAuditTests(unittest.TestCase):
    """Small exact ledgers exercise validation, rather than market evidence."""
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        import os
        repository = Path(os.environ.get('QF_REPOSITORY', str(Path(__file__).resolve().parents[1])))
        path = repository / 'paper/verification/revision_results.py'
        spec = importlib.util.spec_from_file_location('restricted_gain_test_helpers', path)
        cls.helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helpers)

    def fixture(self):
        import certify_restricted_gain as module
        import hashlib
        import json
        certificate = self.helpers._replicated_interval_record
        record = {
            'model': {'spots': [100.] * 5, 'exercise_times': [0., 1., 2.]},
            'design': {'root_seed': 7, 'outer_paths': 1, 'spawn_keys': [[0], [1], [3]]},
            'knots': [80., 100., 120., 140.],
            'coefficients': [[0.] * 10, [10.] + [0.] * 9, [0.] * 10],
            'frozen_policy': [[], [], []],
            'states': [[[100.] * 5 for _ in range(3)]], 'stopping_indices': [1],
        }
        frozen = {key: record[key] for key in ('model', 'knots', 'coefficients', 'frozen_policy')}
        # Arithmetic fixture only. Its point conditional inputs test retained
        # algebra and do not represent a conditional oracle or model result.
        fingerprint = hashlib.sha256(json.dumps(
            [[str(value), '1'] for value in (0, 0, 0, 0, 10, 10)],
            separators=(',', ':')).encode()).hexdigest()
        queries = [dict(increment=index, requested_width_certificate=certificate(F(1), F(1)),
            returned_certificate=certificate(F(0), F(0)),
            work={'integrals': 1, 'integrand_evaluations': 4},
            complete_query_cost={'cpu_seconds': .01, 'wall_seconds': .01}) for index in (1, 2)]
        observation = dict(path_index=0, stopping_index=1,
            initial_common_width_certificate=certificate(F(1), F(1)),
            reward_certificates=[certificate(F(0), F(0)) for _ in range(3)],
            reward_difference_width_allowance_certificate=certificate(F(0), F(0)),
            maximum_stopping_span_length=1, initial_queries=queries, global_rounds=[],
            frozen_centres_and_penalty_sha256=fingerprint,
            signed_gap_certificate=certificate(F(10), F(10)),
            absolute_gap_certificate=certificate(F(10), F(10)),
            coarse_work={'conditional_integrals': 2, 'integrand_evaluations': 8},
            refinement_work={'conditional_integrals': 0, 'integrand_evaluations': 0},
            total_work={'conditional_integrals': 2, 'integrand_evaluations': 8},
            complete_path_cost={'cpu_seconds': .03, 'wall_seconds': .03})
        ledger = dict(fit_index=0, root_seed=7, outer_paths=1,
            arms={'coarse_common': {'outer_paths': 1, 'observations': [observation],
                'mean_signed_gap_certificate': certificate(F(10), F(10))}})
        retained = dict(fit_index=0, root_seed=7,
            frozen_model_policy_values_sha256=module._canonical_digest(frozen),
            arms={'adaptive': {'observations': [dict(path_index=0, stopping_index=1,
                signed_gap_certificate=certificate(F(10), F(10)))]}})
        return record, ledger, retained

    def run_fit(self, record, ledger, retained):
        from certify_restricted_gain import _audit_fit
        return _audit_fit(record, ledger, retained, self.helpers, fit_index=0, root_seed=7,
            expected_paths=1, expected_rights=3)

    def test_one_path_ledger_reconstructs_full_and_restricted_exact_means(self):
        result = self.run_fit(*self.fixture())
        decode = self.helpers._replicated_interval
        self.assertEqual(decode(result['mean_full_gain_certificate']), (F(10), F(10)))
        self.assertEqual(decode(result['mean_restricted_gain_certificate']), (F(0), F(0)))
        self.assertEqual(decode(result['mean_gain_reduction_certificate']), (F(10), F(10)))
        self.assertEqual(result['unchanged_full_interval_count'], 0)
        self.assertEqual(result['retained_adaptive_full_interval_intersections'], 1)
        self.assertEqual(result['restricted_old_adaptive_interval_intersections'], 0)
        self.assertEqual(result['restricted_paths_above_original_width_target'], 0)
        self.assertEqual(len(result['ordered_proof_sha256']), 64)

    def test_changed_conditional_trace_is_rejected(self):
        record, ledger, retained = self.fixture()
        observation = ledger['arms']['coarse_common']['observations'][0]
        observation['initial_queries'][1]['returned_certificate'] = self.helpers._replicated_interval_record(F(1), F(1))
        with self.assertRaisesRegex(ValueError, 'centres'):
            self.run_fit(record, ledger, retained)

    def test_missing_initial_conditional_is_rejected_by_full_trace_audit(self):
        record, ledger, retained = self.fixture()
        ledger['arms']['coarse_common']['observations'][0]['initial_queries'].pop(0)
        with self.assertRaisesRegex(ValueError, 'initialisation'):
            self.run_fit(record, ledger, retained)

    def test_initial_exercise_policy_and_mismatched_stopping_indices_are_rejected(self):
        for where in ('input', 'coarse', 'adaptive'):
            record, ledger, retained = self.fixture()
            if where == 'input':
                record['stopping_indices'][0] = 0
            elif where == 'coarse':
                ledger['arms']['coarse_common']['observations'][0]['stopping_index'] = 2
            else:
                retained['arms']['adaptive']['observations'][0]['stopping_index'] = 2
            with self.subTest(where=where), self.assertRaises(ValueError):
                self.run_fit(record, ledger, retained)

    def test_model_or_value_mutation_is_rejected_before_algebra(self):
        for key in ('model', 'coefficients', 'frozen_policy'):
            record, ledger, retained = self.fixture()
            if key == 'model':
                record['model']['exercise_times'][2] = 3.
            elif key == 'coefficients':
                record['coefficients'][1][0] = 11.
            else:
                record['frozen_policy'][1] = [1.] * 21
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'changed'):
                self.run_fit(record, ledger, retained)

    def test_old_adaptive_overlap_is_required_only_for_original_full_evaluator(self):
        record, ledger, retained = self.fixture()
        retained['arms']['adaptive']['observations'][0]['signed_gap_certificate'] = self.helpers._replicated_interval_record(F(9), F(9))
        with self.assertRaisesRegex(ValueError, 'do not intersect'):
            self.run_fit(record, ledger, retained)

    def test_reconstructed_mean_cannot_be_substituted_by_altered_summary(self):
        record, ledger, retained = self.fixture()
        ledger['arms']['coarse_common']['mean_signed_gap_certificate'] = self.helpers._replicated_interval_record(F(9), F(9))
        with self.assertRaisesRegex(ValueError, 'sample mean'):
            self.run_fit(record, ledger, retained)

    def test_file_tracker_rejects_declared_hash_and_detects_later_mutation(self):
        from certify_restricted_gain import _AccessedFiles
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'inputs.json'
            path.write_text('{"value":1}')
            tracker = _AccessedFiles()
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                tracker.read(path, '0' * 64)
            self.assertEqual(tracker.read(path), {'value': 1})
            path.write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                tracker.verify_unchanged()

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(self):
        from certify_restricted_gain import _AccessedFiles
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            for index, text in enumerate(('{"value":1,"value":2}', '{"value":NaN}', '{"value":Infinity}')):
                path = Path(folder) / (str(index) + '.json')
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    _AccessedFiles().read(path)

    def test_existing_audit_is_preserved_before_any_source_access(self):
        from certify_restricted_gain import audit_directory
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            existing = Path(folder) / 'audit.json'
            existing.write_text('retained prior audit')
            with self.assertRaises(FileExistsError):
                audit_directory(Path(folder) / 'missing', existing, repository=Path(folder) / 'absent')
            self.assertEqual(existing.read_text(), 'retained prior audit')

    def test_public_output_location_and_source_path_traversal_are_rejected(self):
        from certify_restricted_gain import audit_directory, _source_path
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            repository = Path(folder)
            with self.assertRaisesRegex(ValueError, 'data/private'):
                audit_directory(repository / 'data/private/study', repository / 'published.json', repository=repository)
            for name in ('../outside.py', '/outside.py', 'scripts/../outside.py', 'scripts//file.py', 'scripts/file:stream', 'scripts\\file.py'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _source_path(repository, name)
            self.assertEqual(_source_path(repository, 'scripts/certify_restricted_gain.py'),
                             repository / 'scripts/certify_restricted_gain.py')


if __name__ == '__main__':
    unittest.main()
