"""Exact corner checks and conditional integration checks for gap enclosures."""

import importlib.util
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('certify_multivariate_gap', SCRIPTS/'certify_multivariate_gap.py')
component = importlib.util.module_from_spec(spec)
spec.loader.exec_module(component)


def prepared_fixture():
    """Small mathematical input, with no retained research or provider data."""
    return dict(model=dict(spots=[100., 100.], volatilities=[.2, .2],
                           dividends=[.1, .1], correlation=[[1., 0.], [0., 1.]],
                           rate=.05, strike=100., exercise_times=[0., 1.]),
                states=[[[100., 100.], [110., 95.]], [[100., 100.], [90., 105.]]],
                stopping_indices=[1, 1], coefficients=[[0.]*10, [0.]*10],
                knots=[80., 100., 120., 140.],
                design=dict(training_paths=64, value_paths=64, outer_paths=2,
                            spawn_keys=[[0], [1], [2]], root_seed=1,
                            evaluation_independent_of_both_fitting_streams=True),
                frozen_policy=[[], []])


class PreparedInputChecks(unittest.TestCase):
    def test_fixture_and_zero_frozen_coefficients_are_valid(self):
        model, paths, stops, values, times = component.check_inputs(prepared_fixture())
        self.assertEqual((len(model['spots']), len(paths), stops, len(values)), (2, 2, [1, 1], 2))
        self.assertEqual(times, [Fraction(0), Fraction(1)])

    def test_asset_vector_and_correlation_shapes_are_checked(self):
        changes = [('volatilities', [.2]), ('dividends', [.1]),
                   ('correlation', []), ('correlation', [[1., 0.]]),
                   ('correlation', [[1., 0., 0.], [0., 1., 0.]])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                record = prepared_fixture(); record['model'][key] = value
                with self.assertRaises(ValueError):
                    component.check_inputs(record)

    def test_nonfinite_and_unsupported_model_inputs_are_rejected(self):
        changes = [('spots', [float('nan'), 100.]), ('spots', [0., 100.]),
                   ('rate', float('inf')), ('volatilities', [0., 0.]),
                   ('volatilities', [.2, .3]), ('dividends', [.1, .2]),
                   ('correlation', [[1., .01], [.01, 1.]])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                record = prepared_fixture(); record['model'][key] = value
                with self.assertRaises(ValueError):
                    component.check_inputs(record)

    def test_calendar_state_origin_stops_and_coefficients_are_checked(self):
        changes = []
        for times in ([], [0.], [1., 2.], [0., 0.], [0., float('nan')]):
            record = prepared_fixture(); record['model']['exercise_times'] = times
            changes.append(record)
        record = prepared_fixture(); record['states'][0][0][0] = 99.; changes.append(record)
        record = prepared_fixture(); record['states'][0][1] = [110.]; changes.append(record)
        record = prepared_fixture(); record['stopping_indices'][0] = True; changes.append(record)
        record = prepared_fixture(); record['coefficients'][1][2] = float('nan'); changes.append(record)
        record = prepared_fixture(); record['coefficients'][1] = [0.]*9; changes.append(record)
        for index, record in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                component.check_inputs(record)

    def test_metadata_cannot_claim_a_different_outer_count(self):
        record = deepcopy(prepared_fixture())
        record['design']['outer_paths'] = 3
        with self.assertRaises(ValueError):
            component.check_inputs(record)


class ExactGapChecks(unittest.TestCase):
    def test_all_interval_corners_are_enclosed(self):
        rewards = [(Fraction(2), Fraction(3)), (Fraction(1), Fraction(2)),
                   (Fraction(5), Fraction(6)), (Fraction(3), Fraction(4))]
        penalties = list(map(Fraction, [0, 2, -1, 4]))
        drifts = [(Fraction(-99), Fraction(99)), (Fraction(-1), Fraction(2)),
                  (Fraction(-2), Fraction(-1)), (Fraction(0), Fraction(3))]
        for tau in range(4):
            lo, hi, alo, ahi = component.signed_gap(rewards, penalties, drifts, tau)
            self.assertLessEqual(alo, lo)
            self.assertLessEqual(hi, ahi)
            for h, increments in product(product(*rewards), product(*drifts)):
                b = [increments[0]]
                for delta in increments[1:]:
                    b.append(b[-1]+delta)
                value = max(h[j]-penalties[j]-h[tau]+penalties[tau]+b[j]-b[tau] for j in range(4))
                self.assertLessEqual(lo, value)
                self.assertGreaterEqual(hi, value)

    def test_initial_increment_cancels_and_selected_date_is_zero(self):
        lo, hi, alo, ahi = component.signed_gap([(Fraction(1), Fraction(2))],
            [Fraction(10)], [(Fraction(-10**100), Fraction(10**100))], 0)
        self.assertEqual((lo, hi, alo, ahi), (0, 0, 0, 0))

    def test_rational_hinge_value_matches_definition(self):
        coefficients = [Fraction(1, 3), Fraction(-2), *[Fraction(i, 7) for i in range(8)]]
        states = [90., 120., 101.]
        knots = [80., 100., 120., 140.]
        result = component.realised_value(states, coefficients, knots)
        features = [Fraction(1), Fraction(311, 300), Fraction(40, 100), Fraction(20, 100), 0, 0,
                    Fraction(71, 300), Fraction(21, 300), 0, 0]
        self.assertEqual(result, sum(c*f for c, f in zip(coefficients, features)))
        self.assertEqual(result, component.realised_value(states[::-1], coefficients, knots))

    def test_invalid_gain_or_value_inputs_rejected(self):
        with self.assertRaises(ValueError):
            component.signed_gap([(Fraction(2), Fraction(1))], [Fraction(0)], [(Fraction(0), Fraction(0))], 0)
        for state in ([], [0], [float('nan')]):
            with self.assertRaises(ValueError):
                component.realised_value(state, [0]*10, [80, 100, 120, 140])


@unittest.skipIf(component.ctx is None, 'The separate python-flint environment is required')
class ConditionalFunctionChecks(unittest.TestCase):
    def test_frozen_policy_recheck_uses_current_state_and_strict_exercise(self):
        record = prepared_fixture()
        record['model']['exercise_times'] = [0., 1., 2.]
        record['coefficients'] = [[0.]*10 for _ in range(3)]
        record['states'] = [[[100., 100.], [110., 95.], [90., 90.]],
                            [[100., 100.], [110., 95.], [10000., 10000.]]]
        record['stopping_indices'] = [1, 1]
        record['frozen_policy'] = [[], [5., 0., 0., 0., 0., 0.], []]
        self.assertTrue(component.check_policy(record))
        record['states'][0][-1] = [1e10, 1e10]
        self.assertTrue(component.check_policy(record))
        record['stopping_indices'][0] = 2
        with self.assertRaises(ArithmeticError):
            component.check_policy(record)

    def test_zero_payoff_tie_continues_and_policy_coefficients_are_checked(self):
        record = prepared_fixture()
        record['frozen_policy'] = [[0.]*6, []]
        self.assertTrue(component.check_policy(record))
        for policy in ([[0.]*5, []], [[0., 0., float('nan'), 0., 0., 0.], []], [[], [0.]*6]):
            record['frozen_policy'] = policy
            with self.assertRaises(ValueError):
                component.check_policy(record)

    def test_linear_component_has_analytic_expectation(self):
        model = dict(rate=.05, dividends=[.1]*2, volatilities=[.2]*2)
        coefficients = [3., -2., *[0.]*8]
        lo, hi, work = component.conditional_value([90., 110.], coefficients, [80, 100, 120, 140],
                                                  model, Fraction(1, 3), .001)
        with component.ctx.workprec(192):
            value = component.ball(3)-component.ball(2)*((component.ball(.05)-component.ball(.1))/3).exp()
            rlo, rhi = component.endpoints(value)
        self.assertLessEqual(lo, rlo)
        self.assertGreaterEqual(hi, rhi)
        self.assertEqual(work['integrals'], 0)

    def test_negative_max_coefficient_reverses_endpoints(self):
        model = dict(rate=.05, dividends=[.1]*2, volatilities=[.2]*2)
        coefficients = [0., 0., 0., -2., *[0.]*6]
        lo, hi, _ = component.conditional_value([100., 100.], coefficients, [80, 100, 120, 140],
                                                model, Fraction(1, 3), .0001)
        self.assertLessEqual(lo, hi)
        self.assertLess(hi, 0)
        self.assertLessEqual(hi-lo, Fraction.from_float(.0001))


@unittest.skipIf(importlib.util.find_spec('numpy') is None, 'The separate research environment is required')
class PreparationScaleChecks(unittest.TestCase):
    def test_terminal_basis_coefficient_recovers_discounted_cash_payoff(self):
        import math
        import json
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root/'src'))
        prepared = component.prepare(dimension=2, training_count=64, value_count=64,
                                     evaluation_count=2, seed=7821)
        terminal = prepared['coefficients'][-1]
        value = component.realised_value([120., 80.], terminal, prepared['knots'])
        self.assertAlmostEqual(float(value), 20*math.exp(-.05*3), places=12)
        json.dumps(prepared, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
