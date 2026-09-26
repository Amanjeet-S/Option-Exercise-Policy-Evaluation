"""Boundary and failure cases for finite-tree compensation."""

from fractions import Fraction as Q
import unittest

from conditional_drift import Audit, Tree, counterexample, fixtures


class CompensationTests(unittest.TestCase):
    def test_zero_unconditional_mean_is_insufficient(self):
        a, waiting, optimal = counterexample()
        t = a.tree
        for j in range(1, 3):
            unconditional = sum((t.probability(n) * (a.penalty[n] - a.penalty[n[:-1]])
                                 for n in t.nodes(j)), Q(0))
            self.assertEqual(unconditional, 0)
        self.assertEqual(a.drift[(-1,)], -1)
        self.assertEqual(a.drift[(1,)], 1)
        w, o = a.metrics(waiting), a.metrics(optimal)
        self.assertEqual(w.gap, 0)
        self.assertEqual(a.optimal_value - w.value, Q(1, 2))
        self.assertEqual(w.correction, Q(1, 2))
        self.assertEqual(o.value - w.value, Q(1, 2))
        self.assertEqual(w.gap - o.gap, 0)
        a.verify()

    def test_first_drift_cancels_with_only_one_exercise_right(self):
        t = Tree((Q(1, 3),))
        h = {(): Q(0), (-1,): Q(2), (1,): Q(-3)}
        a = {(): Q(0), (-1,): Q(9), (1,): Q(12)}
        audit = Audit(t, h, a)
        self.assertEqual(audit.drift[()], 10)
        self.assertEqual(audit.norm_budget, 0)
        self.assertEqual(audit.metrics(t.policies()[0]).correction, 0)
        audit.verify()

    def test_true_martingale_needs_no_correction(self):
        t = Tree((Q(1, 2),) * 3)
        h = {n: Q(max(2 - sum(n), 0))
             for j in range(4) for n in t.nodes(j)}
        a = {n: Q(sum(n)) for n in h}
        audit = Audit(t, h, a)
        self.assertTrue(all(mu == 0 for mu in audit.drift.values()))
        self.assertEqual(audit.norm_budget, 0)
        audit.verify()

    def test_nonuniform_measure_and_signed_rewards(self):
        cases = list(fixtures())
        self.assertEqual(len(cases), 162)
        for a in cases:
            self.assertEqual(a.verify(), (25, 625))

    def test_non_stopping_rule_is_rejected(self):
        a, _, _ = counterexample()
        # Stops both at a node and at its descendant, so is not a stopping rule.
        with self.assertRaises(ValueError):
            a.metrics(frozenset(((-1,), (-1, -1), (1, -1), (1, 1))))
        # The other branch is uncovered, equivalent to an incomplete rule.
        with self.assertRaises(ValueError):
            a.metrics(frozenset(((-1,),)))
        with self.assertRaises(ValueError):
            a.metrics(frozenset(((),)))

    def test_invalid_measure_and_incomplete_node_values_are_rejected(self):
        for probabilities in ((), (Q(0),), (Q(1),), (0.5,)):
            with self.assertRaises(ValueError):
                Tree(probabilities)
        a, _, _ = counterexample()
        broken = dict(a.penalty)
        del broken[(1,)]
        with self.assertRaises(ValueError):
            Audit(a.tree, a.reward, broken)
        broken = dict(a.penalty)
        broken[()] = Q(1)
        with self.assertRaises(ValueError):
            Audit(a.tree, a.reward, broken)


if __name__ == '__main__':
    unittest.main()
