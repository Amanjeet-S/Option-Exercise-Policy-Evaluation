"""Stipulated mathematical fixtures for finite-innovation arithmetic replay.

These cases are analytical checks. They are not market observations, fitted
empirical policies or additional Monte Carlo samples.
"""

import copy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

try:
    from flint import arb, ctx
except ModuleNotFoundError:
    arb = ctx = None


if arb is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import certify_gain_replay as replay
else:
    replay = None


def constant(value=4.):
    return dict(knots=[0., 32.], values=[value, value], tail_slope=0.)


def fixture(*, sigma=0., cash=True):
    rights = ([dict(time=.5, settlement_factor=1., phase='before'),
               dict(time=.5, settlement_factor=1., phase='after')]
              if cash else [dict(time=.5, settlement_factor=1., phase='after')])
    rights.append(dict(time=1., settlement_factor=1., phase='after'))
    return dict(model=dict(quoted_spot=8., volatility=sigma,
                          initial_share_factor=1., premium_factor=2.,
                          dividends=[dict(ex_time=.5, discounted_cash=3.)] if cash else []),
                option=dict(strike=10., maturity=1., kind='put'), decisions=rights,
                nodes=[dict(time=0., before=None, after=constant()),
                       dict(time=.5, before=constant() if cash else None, after=constant()),
                       dict(time=1., before=None, after=constant())])


def policy(spec, coefficients):
    result = {key: copy.deepcopy(spec[key]) for key in ('model', 'option', 'decisions')}
    result['rules'] = [dict(centre=0., scale=1., coefficients=list(c), status='full_degree')
                       for c in coefficients]
    result['rules'].append(dict(centre=0., scale=1., coefficients=[], status='terminal'))
    return result


def payload():
    spec = fixture()
    return dict(schema_version=1, model_spec=spec,
                policies=[policy(spec, [[0.], []]), policy(spec, [[], [0.]])],
                normals=[[], []], innovation_times=[],
                observation_rights=copy.deepcopy(spec['decisions']),
                states=[[8., 5., 5.], [8., 5., 5.]],
                stopping_indices=[[0, 1], [0, 1]], quoted_gains=[1.5, 1.5],
                record=dict(evaluation=dict(gain=dict(estimate=dict(value=1.5)))))


@unittest.skipIf(arb is None, 'Run these arithmetic checks in the separate qf-certification environment')
class ReplayArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.precision = ctx.workprec(192)
        self.precision.__enter__()

    def tearDown(self):
        self.precision.__exit__(None, None, None)

    def assertContains(self, interval, exact):
        exact = Fraction(exact)
        self.assertLessEqual(interval[0], exact)
        self.assertGreaterEqual(interval[1], exact)

    def test_enclose_contains_non_dyadic_endpoints(self):
        value = replay.enclose(Fraction(1, 3), Fraction(2, 3))
        self.assertLessEqual(Fraction(str(value.lower().fmpq())), Fraction(1, 3))
        self.assertGreaterEqual(Fraction(str(value.upper().fmpq())), Fraction(2, 3))

    def test_invalid_intervals_are_rejected(self):
        for value in ((2, 1), (0,), (0, 1, 2), (False, 1), (0., float('inf'))):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replay.checked_interval(value)

    def test_piecewise_linear_interval_crosses_knots_and_constant_ray(self):
        value = replay.RationalValue([0., 1., 3.], [3., 2., 0.])
        self.assertEqual(replay.linear_interval(value, (Fraction(1, 2), Fraction(4))),
                         (Fraction(0), Fraction(5, 2)))
        self.assertEqual(replay.expectation_interval(value, (Fraction(1, 2), Fraction(4)), arb(0)),
                         (Fraction(0), Fraction(5, 2)))

    def test_nonmonotone_values_and_negative_states_are_rejected(self):
        value = replay.RationalValue([0., 1., 2.], [2., 0., 1.])
        with self.assertRaises(ValueError):
            replay.linear_interval(value, (0, 2))
        with self.assertRaises(ValueError):
            replay.expectation_interval(value, (0, 2), arb(1))
        with self.assertRaises(ValueError):
            replay.linear_interval(replay.RationalValue([0., 1.], [1., 0.]), (-1, 1))

    def test_constant_expectation_is_exact(self):
        value = replay.RationalValue([0., 1., 20.], [7., 7., 7.])
        self.assertEqual(replay.expectation_interval(value, (Fraction(0), Fraction(100)), arb(2)),
                         (Fraction(7), Fraction(7)))

    def test_put_expectation_against_independent_erf_identity(self):
        value = replay.RationalValue([0., 1.], [1., 0.])
        result = replay.expectation_interval(value, (Fraction(1), Fraction(1)), arb(1))
        # E[(1-exp(-1/2+Z))+] = erf(1/(2 sqrt(2))) for Z standard normal.
        oracle = (arb(1) / (2 * arb(2).sqrt())).erf()
        self.assertLessEqual(result[0], Fraction(str(oracle.lower().fmpq())))
        self.assertGreaterEqual(result[1], Fraction(str(oracle.upper().fmpq())))
        self.assertLess(result[1] - result[0], Fraction(1, 2**75))
        ranged = replay.expectation_interval(value, (Fraction(0), Fraction(1)), arb(1))
        self.assertEqual(ranged[0], result[0])
        self.assertEqual(ranged[1], 1)

    def test_initial_product_uses_exact_binary64_factors(self):
        spec = fixture(cash=False)
        spec['model'].update(quoted_spot=1.1, initial_share_factor=1.1)
        exact = Fraction(1.1) ** 2
        self.assertNotEqual(exact, Fraction(1.1 * 1.1))
        states = replay.propagate(spec, [])
        self.assertTrue(all(interval == (exact, exact) for interval in states.values()))

    def test_exact_cash_jump_and_liquidation_atom(self):
        spec = fixture()
        states = replay.propagate(spec, [])
        self.assertEqual(states[Fraction(1, 2), 'before'], (8, 8))
        self.assertEqual(states[Fraction(1, 2), 'after'], (5, 5))
        spec['model']['dividends'][0]['discounted_cash'] = 9.
        states = replay.propagate(spec, [])
        self.assertEqual(states[Fraction(1, 2), 'after'], (0, 0))
        self.assertEqual(states[Fraction(1), 'after'], (0, 0))

    def test_normal_drift_against_independent_exponential(self):
        spec = fixture(sigma=1., cash=False)
        states = replay.propagate(spec, [0., 0.])
        for time in (Fraction(1, 2), Fraction(1)):
            oracle = 8 * (-replay.ball(time) / 2).exp()
            result = states[time, 'after']
            self.assertLessEqual(result[0], Fraction(str(oracle.lower().fmpq())))
            self.assertGreaterEqual(result[1], Fraction(str(oracle.upper().fmpq())))
            self.assertLess(result[1] - result[0], Fraction(1, 2**70))

    def test_exact_time_difference_is_not_a_rounded_float_difference(self):
        spec = fixture(sigma=1., cash=False)
        spec['nodes'][1]['time'] = 2.**-60
        spec['decisions'][0]['time'] = 2.**-60
        states = replay.propagate(spec, [0., 0.])
        oracle = 8 * arb(-.5).exp()
        lo, hi = states[Fraction(1), 'after']
        self.assertLessEqual(lo, Fraction(str(oracle.lower().fmpq())))
        self.assertGreaterEqual(hi, Fraction(str(oracle.upper().fmpq())))
        self.assertLess(hi - lo, Fraction(1, 2**70))

    def test_wrong_innovation_count_and_nonfinite_innovations_are_rejected(self):
        for spec, normals in ((fixture(), [0.]), (fixture(sigma=1.), []),
                              (fixture(sigma=1.), [0., float('nan')])):
            with self.subTest(normals=normals), self.assertRaises(ValueError):
                replay.propagate(spec, normals)

    def test_payoff_threshold_is_an_exact_product(self):
        spec = fixture()
        spec['option']['strike'] = 1.1
        right = dict(time=.5, settlement_factor=1.1, phase='before')
        threshold = Fraction(1.1) ** 2
        self.assertEqual(replay.reward_interval(spec, right, (0, 0)), (threshold, threshold))

    def test_polynomial_standardisation_horner_and_clipping(self):
        rule = dict(centre=2., scale=2., coefficients=[1., 2., 3.])
        self.assertEqual(replay.prediction_interval(rule, (4, 4)), (6, 6))
        rule['coefficients'] = [-3.]
        self.assertEqual(replay.prediction_interval(rule, (4, 4)), (0, 0))

    def test_strict_payoff_continuation_ties_continue(self):
        spec = fixture(cash=False)
        states = replay.propagate(spec, [])
        self.assertEqual(replay.stopping_candidates(policy(spec, [[2.]]), states), (1,))
        self.assertEqual(replay.stopping_candidates(policy(spec, [[1.]]), states), (0,))

    def test_unfitted_rule_continues_and_negative_prediction_exercises(self):
        spec = fixture(cash=False)
        states = replay.propagate(spec, [])
        self.assertEqual(replay.stopping_candidates(policy(spec, [[]]), states), (1,))
        self.assertEqual(replay.stopping_candidates(policy(spec, [[-1.]]), states), (0,))

    def test_out_of_money_state_does_not_exercise_against_zero(self):
        spec = fixture(cash=False)
        states = replay.propagate(spec, [])
        states[Fraction(1, 2), 'after'] = (10, 10)
        self.assertEqual(replay.stopping_candidates(policy(spec, [[-1.]]), states), (1,))

    def test_ambiguous_branch_keeps_both_stopping_possibilities(self):
        spec = fixture(cash=False)
        states = replay.propagate(spec, [])
        states[Fraction(1, 2), 'after'] = (7, 9)
        self.assertEqual(replay.stopping_candidates(policy(spec, [[2.]]), states), (0, 1))

    def test_earlier_ambiguity_survives_later_certain_exercise(self):
        spec = fixture()
        states = replay.propagate(spec, [])
        states[Fraction(1, 2), 'before'] = (7, 9)
        self.assertEqual(replay.stopping_candidates(policy(spec, [[2.], [0.]]), states), (0, 1))

    def test_same_time_cash_rights_have_zero_martingale_difference(self):
        spec = fixture()
        states = replay.propagate(spec, [])
        cache = {}
        self.assertEqual(replay.gain_interval(spec, states, 0, 1, cache), (Fraction(3, 2), Fraction(3, 2)))
        self.assertEqual(cache, {})
        self.assertEqual(replay.gain_interval(spec, states, 1, 0), (Fraction(-3, 2), Fraction(-3, 2)))

    def test_identical_stop_is_exactly_zero_without_expectation_work(self):
        spec = fixture(sigma=1.)
        states = replay.propagate(spec, [0., 0.])
        cache = {}
        self.assertEqual(replay.gain_interval(spec, states, 2, 2, cache), (0, 0))
        self.assertEqual(cache, {})

    def test_only_increments_between_distinct_stops_are_cached(self):
        spec = fixture()
        states = replay.propagate(spec, [])
        cache = {}
        self.assertEqual(replay.gain_interval(spec, states, 0, 2, cache), (Fraction(3, 2), Fraction(3, 2)))
        self.assertEqual(set(cache), {Fraction(1)})
        self.assertEqual(replay.gain_interval(spec, states, 2, 0, cache), (Fraction(-3, 2), Fraction(-3, 2)))

    def test_nonzero_martingale_increment_and_gain_sign_against_erf(self):
        spec = fixture(sigma=1., cash=False)
        for node in spec['nodes']:
            node['after'] = dict(knots=[0., 8.], values=[8., 0.], tail_slope=0.)
        # These are stipulated states for an algebraic functional, not draws.
        states = {(Fraction(0), 'after'): (8, 8),
                  (Fraction(1, 2), 'after'): (8, 8),
                  (Fraction(1), 'after'): (8, 8)}
        # Identical exercise rewards cancel. Over the half-year interval,
        # -Delta M / premium = 8 erf(1/4) / 2.
        oracle = 4 * arb(.25).erf()
        gain = replay.gain_interval(spec, states, 0, 1)
        self.assertLessEqual(gain[0], Fraction(str(oracle.lower().fmpq())))
        self.assertGreaterEqual(gain[1], Fraction(str(oracle.upper().fmpq())))
        self.assertLess(gain[1] - gain[0], Fraction(1, 2**75))
        self.assertEqual(replay.gain_interval(spec, states, 1, 0), (-gain[1], -gain[0]))

    def test_bad_stopping_indices_are_rejected(self):
        spec = fixture()
        states = replay.propagate(spec, [])
        for indices in ((-1, 0), (0, 3), (True, 0)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                replay.gain_interval(spec, states, *indices)

    def test_complete_stipulated_replay_and_scope_flags(self):
        result = replay.certify_paths(payload())
        self.assertEqual(result['mean_interval'], (Fraction(3, 2), Fraction(3, 2)))
        self.assertEqual(result['mean_error_interval'], (0, 0))
        self.assertEqual(result['maximum_state_error_bound'], 0)
        self.assertTrue(result['all_stopping_indices_match_on_exported_paths'])
        self.assertFalse(result['ideal_gaussian_sampling_law_certified'])
        self.assertFalse(result['statistical_coverage_certified'])
        self.assertFalse(result['training_algorithm_error_certified'])
        self.assertEqual(result['new_independent_paths'], 0)

    def test_mean_error_uses_original_scalar_not_recomputed_path_mean(self):
        data = payload()
        data['record']['evaluation']['gain']['estimate']['value'] = 1.75
        result = replay.certify_paths(data)
        self.assertEqual(result['recorded_path_mean_exact'], Fraction(3, 2))
        self.assertEqual(result['recorded_mean_aggregation_error'], Fraction(1, 4))
        self.assertEqual(result['mean_error_interval'], (Fraction(-1, 4), Fraction(-1, 4)))

    def test_path_error_retains_both_signs(self):
        data = payload()
        data['quoted_gains'] = [1., 2.]
        result = replay.certify_paths(data)
        self.assertEqual(result['rows'][0]['error_interval'], (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(result['rows'][1]['error_interval'], (Fraction(-1, 2), Fraction(-1, 2)))
        self.assertEqual(result['maximum_path_error_bound'], Fraction(1, 2))

    def test_report_identifies_different_implemented_stopping_indices(self):
        data = payload()
        data['stopping_indices'][0] = [1, 0]
        result = replay.certify_paths(data)
        self.assertFalse(result['all_stopping_indices_match_on_exported_paths'])
        self.assertFalse(result['rows'][0]['stopping_decisions_match'])
        self.assertEqual(result['proved_stopping_index_difference_count'], 2)
        self.assertEqual(result['mean_interval'], (Fraction(3, 2), Fraction(3, 2)))

    def test_malformed_replay_dimensions_are_rejected(self):
        for key, replacement in (('normals', [[]]), ('states', [[8., 5., 5.]]),
                                 ('stopping_indices', [[0, 1]]), ('quoted_gains', [1.5]),
                                 ('innovation_times', [.5]), ('observation_rights', [])):
            data = payload()
            data[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.certify_paths(data)

    def test_missing_observation_is_rejected_despite_matching_row_widths(self):
        data = payload()
        del data['observation_rights'][1]
        for row in data['states']:
            del row[1]
        with self.assertRaises(ValueError):
            replay.certify_paths(data)

    def test_reordered_observations_and_duplicate_vertices_are_rejected(self):
        for duplicate in (False, True):
            data = payload()
            if duplicate:
                data['observation_rights'][1] = copy.deepcopy(data['observation_rights'][0])
            else:
                data['observation_rights'][0], data['observation_rights'][1] = (
                    data['observation_rights'][1], data['observation_rights'][0])
                for row in data['states']:
                    row[0], row[1] = row[1], row[0]
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                replay.certify_paths(data)


if __name__ == '__main__':
    unittest.main()
