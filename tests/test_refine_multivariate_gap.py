"""Information-order and interval checks for fixed-centre gap refinement."""

from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('refine_multivariate_gap', SCRIPTS/'refine_multivariate_gap.py')
component = importlib.util.module_from_spec(spec)
spec.loader.exec_module(component)


class FixedCentreChecks(unittest.TestCase):
    def test_largest_width_selection_breaks_ties_closest_to_stop(self):
        rewards = [(F(-1000), F(-1000)), (F(4), F(4)), (F(3), F(3)), (F(10), F(10))]
        drifts = [(F(0), F(0)), (F(-99), F(99)), (F(0), F(4)), (F(0), F(4))]
        increment, upper_date = component.select_increment(rewards, (F(0),)*4, drifts, 1)
        self.assertEqual((increment, upper_date), (2, 3))
        # The wider increment at the stop does not enter a future-date gap.
        self.assertNotEqual(increment, 1)

    def test_past_date_selection_uses_reversed_signed_increment_span(self):
        rewards = [(F(10), F(10)), *[(F(0), F(0))]*3]
        drifts = [(F(-99), F(99)), (F(-1), F(1)), (F(-5), F(5)), (F(-2), F(2))]
        self.assertEqual(component.select_increment(rewards, (F(0),)*4, drifts, 3), (2, 0))
        candidates = component.candidate_bounds(rewards, (F(0),)*4, drifts, 3)
        self.assertEqual(candidates[0], (F(2), F(18)))
        self.assertEqual(candidates[3], (F(0), F(0)))

    def test_refinement_leaves_centres_penalty_and_oracle_target_unchanged(self):
        rewards = [(F(0), F(0)), (F(4), F(4)), (F(3), F(3)), (F(10), F(10))]
        penalty = (F(0), F(2), F(-1), F(1))
        centres = (F(0), F(3), F(7), F(-2))
        brackets = ((F(0), F(0)), (F(2), F(4)), (F(5), F(9)), (F(-4), F(0)))
        actual = (F(0), F(3), F(7), F(-2))
        frozen = component.frozen_hash(centres, penalty)
        calls = []

        def oracle(j, target):
            calls.append((j, target))
            return actual[j], actual[j], dict(integrals=1, integrand_evaluations=7)

        refined, certificate, history = component.refine_gap(
            rewards, penalty, centres, brackets, 1, oracle, target=F(1, 100))
        self.assertEqual(calls[0][0], 2)
        self.assertLessEqual(certificate[1]-certificate[0], F(1, 100))
        exact_drifts = component.drift_intervals(tuple((x, x) for x in actual), centres)
        exact_certificate = component.signed_gap(rewards, penalty, exact_drifts, 1)
        self.assertLessEqual(certificate[0], exact_certificate[0])
        self.assertGreaterEqual(certificate[1], exact_certificate[1])
        self.assertEqual(component.frozen_hash(centres, penalty), frozen)
        self.assertEqual(brackets[2], (F(5), F(9)))
        self.assertEqual(len(history), len(calls))
        self.assertTrue(all(old[0] <= new[0] <= new[1] <= old[1]
                            for old, new in zip(brackets, refined)))

    def test_zero_gap_needs_no_integrals_and_intersection_fails_closed(self):
        def forbidden(*_):
            raise AssertionError('The exact zero path needs no refinement')
        _, certificate, history = component.refine_gap(
            [(F(0), F(0))]*2, (F(0), F(0)), (F(0), F(0)),
            ((F(0), F(0)), (F(0), F(0))), 1, forbidden)
        self.assertEqual(certificate, (0, 0, 0, 0))
        self.assertFalse(history)
        with self.assertRaises(ArithmeticError):
            component.intersect((F(0), F(1)), (F(2), F(3)))
        self.assertEqual(component.intersect((F(0), F(2)), (F(1), F(3))), (F(1), F(2)))

    def test_unproductive_refinement_and_invalid_target_fail_closed(self):
        rewards = [(F(10), F(10)), (F(0), F(0))]
        args = (rewards, (F(0), F(0)), (F(0), F(0)),
                ((F(0), F(0)), (F(0), F(1))), 1)
        with self.assertRaises(ArithmeticError):
            component.refine_gap(*args, lambda *_: (F(0), F(1), {}))
        with self.assertRaises(ValueError):
            component.refine_gap(*args, lambda *_: None, target=F(0))


@unittest.skipIf(component.ctx is None, 'The separate python-flint environment is required')
class VerifiedComparisonChecks(unittest.TestCase):
    def test_both_strategies_enclose_same_fixed_model_gap_without_input_mutation(self):
        record = dict(model=dict(spots=[100., 100.], volatilities=[.2, .2], dividends=[.1, .1],
            correlation=[[1., 0.], [0., 1.]], rate=.05, strike=100., exercise_times=[0., 1.]),
            states=[[[100., 100.], [110., 95.]], [[100., 100.], [90., 105.]]],
            stopping_indices=[1, 1], coefficients=[[0.]*10, [0., 0., 0., 100., *[0.]*6]],
            knots=[80., 100., 120., 140.], frozen_policy=[[], []],
            design=dict(training_paths=64, value_paths=64, outer_paths=2,
                spawn_keys=[[0], [1], [2]], root_seed=1,
                evaluation_independent_of_both_fitting_streams=True))
        before = deepcopy(record)
        result = component.compare(record)
        self.assertEqual(record, before)
        self.assertTrue(result['observed_policy_decisions_checked'])
        self.assertTrue(result['all_method_certificates_overlap'])
        for name in ('uniform', 'adaptive'):
            self.assertTrue(result['strategies'][name]['all_targets_met'])
            self.assertGreater(result['strategies'][name]['conditional_integrals'], 0)
        for one, two in zip(result['observations']['uniform'], result['observations']['adaptive']):
            self.assertEqual(one['frozen_centres_and_penalty_sha256'], two['frozen_centres_and_penalty_sha256'])
        cold = component.cold_uniform(record)
        self.assertFalse(cold['coarse_initialisation_performed'])
        self.assertTrue(cold['strategy']['all_targets_met'])
        self.assertGreater(cold['strategy']['conditional_integrals'], 0)
        self.assertTrue(all(row['coarse_work']['conditional_integrals'] == 0 for row in cold['observations']))
        decode = lambda interval: tuple(F(int(interval[key]['numerator']), int(interval[key]['denominator']))
                                        for key in ('lower', 'upper'))
        for fine, adaptive in zip(cold['observations'], result['observations']['adaptive']):
            component.intersect(decode(fine['signed_gap_certificate']), decode(adaptive['signed_gap_certificate']))
        midpoint_mean = sum(sum(decode(row['signed_gap_certificate']))/2 for row in cold['observations'])/2
        self.assertAlmostEqual(float(midpoint_mean), cold['strategy']['approximate_midpoint_mean'], places=14)
        self.assertIsNotNone(cold['strategy']['approximate_midpoint_mean_standard_error'])
        self.assertNotIn('states', result)
        self.assertNotIn('coefficients', result)
        self.assertNotIn('frozen_policy', result)


if __name__ == '__main__':
    unittest.main()
