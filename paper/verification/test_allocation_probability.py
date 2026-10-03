"""Finite mathematical examples for conditional allocation and stopped means."""

from fractions import Fraction as Q
from itertools import product
import unittest

from allocation_probability import (
    FiniteLaw, cost_stopped_signs, independent_average, pilot_selected_mean,
    repeated_training_mean, training_fixture, variance_components,
)


class AllocationProbabilityTests(unittest.TestCase):
    def test_exhaustive_variance_for_general_joint_training_laws(self):
        expected = {
            'coupled': (Q(1, 3), Q(8, 9), Q(10, 3)),
            'shared': (Q(1, 2), Q(1, 4), Q(5)),
            'independent': (Q(1, 2), Q(5, 4), Q(9, 2)),
        }
        for kind, (mean, a, beta) in expected.items():
            training, gains = training_fixture(kind)
            components = variance_components(training, gains)
            self.assertEqual((components.mean, components.a, components.beta), (mean, a, beta))
            for m, n in product((1, 2, 3), (1, 2, 3)):
                with self.subTest(kind=kind, m=m, n=n):
                    enumerated = repeated_training_mean(training, gains, m, n)
                    self.assertEqual(enumerated.expectation(), mean)
                    self.assertEqual(enumerated.variance(), a/m + beta/(m*n))

    def test_shared_training_is_not_an_independent_gain_sample(self):
        training, gains = training_fixture()
        a = variance_components(training, gains).a
        m, n = 2, 3
        correct = repeated_training_mean(training, gains, m, n)
        marginal = repeated_training_mean(training, gains, 1, 1)
        mistaken = independent_average(marginal, m*n)
        self.assertEqual(correct.variance() - mistaken.variance(), a*(n-1)/(m*n))
        self.assertGreater(correct.variance(), mistaken.variance())

    def test_pilot_mixture_uses_mean_inverse_counts(self):
        training, gains = training_fixture()
        components = variance_components(training, gains)
        pilot = FiniteLaw((((1, 1), Q(1, 3)), ((2, 3), Q(2, 3))))
        enumerated = pilot_selected_mean(training, gains, pilot)
        expected = (
            components.a * pilot.expectation(lambda counts: Q(1, counts[0]))
            + components.beta * pilot.expectation(lambda counts: Q(1, counts[0]*counts[1]))
        )
        self.assertEqual(enumerated.expectation(), Q(1, 3))
        self.assertEqual(enumerated.variance(), Q(56, 27))
        self.assertEqual(enumerated.variance(), expected)
        inverse_mean_product = (
            components.a / pilot.expectation(lambda counts: counts[0])
            + components.beta / pilot.expectation(lambda counts: counts[0]*counts[1])
        )
        product_inverse_means = (
            components.a / pilot.expectation(lambda counts: counts[0])
            + components.beta / (pilot.expectation(lambda counts: counts[0])
                                 * pilot.expectation(lambda counts: counts[1]))
        )
        self.assertNotEqual(expected, inverse_mean_product)
        self.assertNotEqual(expected, product_inverse_means)

    def test_deterministic_pilot_matches_fixed_counts(self):
        training, gains = training_fixture('shared')
        pilot = FiniteLaw((((2, 2), 1),))
        self.assertEqual(pilot_selected_mean(training, gains, pilot),
                         repeated_training_mean(training, gains, 2, 2))

    def test_cost_stopping_preserves_sum_mean_but_biases_average(self):
        law = cost_stopped_signs()
        self.assertEqual(dict(law.atoms), {
            (1, Q(1), Q(2)): Q(1, 2),
            (2, Q(-2), Q(2)): Q(1, 4),
            (2, Q(0), Q(3)): Q(1, 4),
        })
        self.assertEqual(law.expectation(lambda state: state[1]), 0)
        self.assertEqual(law.expectation(lambda state: state[1] / state[0]), Q(1, 4))

    def test_zero_variance_components_and_singleton_counts(self):
        training = FiniteLaw((('a', Q(1, 3)), ('b', Q(2, 3))))
        constant = FiniteLaw(((Q(7, 3), 1),))
        gains = {'a': constant, 'b': constant}
        components = variance_components(training, gains)
        self.assertEqual((components.a, components.beta), (0, 0))
        self.assertEqual(repeated_training_mean(training, gains, 1, 1), constant)
        self.assertEqual(repeated_training_mean(training, gains, 3, 3), constant)

    def test_invalid_probability_laws_are_rejected(self):
        cases = (
            (), ((0, 0), (1, 1)), ((0, -1), (1, 2)),
            ((0, Q(1, 2)),), ((0, Q(1, 2)), (0, Q(1, 2))),
            ((0, 1.0),), ((0, True),), (([], 1),),
        )
        for atoms in cases:
            with self.subTest(atoms=atoms), self.assertRaises(ValueError):
                FiniteLaw(atoms)

    def test_invalid_counts_are_rejected(self):
        training, gains = training_fixture()
        for bad in (0, -1, True, 2.0, Q(2), '2'):
            for m, n in ((bad, 2), (2, bad)):
                with self.subTest(m=m, n=n), self.assertRaises(ValueError):
                    repeated_training_mean(training, gains, m, n)
            with self.assertRaises(ValueError):
                independent_average(gains[(0, 0)], bad)
        for allocation in ((1,), '12', (1, 0)):
            with self.assertRaises(ValueError):
                pilot_selected_mean(training, gains, FiniteLaw(((allocation, 1),)))

    def test_incomplete_extra_or_nonrational_conditionals_are_rejected(self):
        training, gains = training_fixture()
        malformed = [
            {state: law for state, law in gains.items() if state != (0, 0)},
            dict(gains, extra=FiniteLaw(((0, 1),))),
            {**gains, (0, 0): FiniteLaw(((0.5, 1),))},
            {**gains, (0, 0): 'invalid'},
        ]
        for conditionals in malformed:
            with self.assertRaises(ValueError):
                repeated_training_mean(training, conditionals, 1, 1)
            with self.assertRaises(ValueError):
                variance_components(training, conditionals)


if __name__ == '__main__':
    unittest.main()
