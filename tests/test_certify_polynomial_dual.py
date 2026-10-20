"""Independent moment, terminal-payoff and stopped-gap checks."""

from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import certify_polynomial_dual as component


def fixture():
    record = dict(model=dict(spots=[100., 100.], volatilities=[.2, .2],
        dividends=[.1, .1], correlation=[[1., 0.], [0., 1.]], rate=.05,
        strike=100., exercise_times=[0., 1., 2.]),
        states=[[[100., 100.], [120., 90.], [110., 80.]],
                [[100., 100.], [95., 90.], [105., 80.]]],
        stopping_indices=[1, 2], coefficients=[[0.]*10 for _ in range(3)],
        frozen_policy=[[], [10., 0., 0., 0., 0., 0.], []],
        knots=[80., 100., 120., 140.],
        design=dict(root_seed=71, training_paths=64, value_paths=64, outer_paths=2,
            spawn_keys=[[0], [1], [2]], policy_degree=2,
            evaluation_independent_of_both_fitting_streams=True))
    polynomial = dict(root_seed=71, sample_count=2, value_fitting_paths=64,
        value_spawn_key=[1], degrees={'2':dict(
            basis_exponents=[list(row) for row in component.exponents(2, 2)],
            coefficients=[[0.]*6, [1., -2., 0., .4, .3, -.1], [2., -1., .2, -.3, .1, .4]])})
    return record, polynomial


class ExactChecks(unittest.TestCase):
    def test_signed_realised_polynomial_is_exact_and_uses_unsorted_assets(self):
        powers = component.exponents(2, 2)
        result = component.realised_polynomial([125., 80.], [1., -2., 3., .5, -.25, .125], powers)
        expected = F(1)-2*F(5, 4)+3*F(4, 5)+F(1, 2)*F(25, 16)-F(1, 4)+F(1, 8)*F(16, 25)
        self.assertEqual(result, expected)
        self.assertNotEqual(result, component.realised_polynomial([80., 125.], [1., -2., 3., .5, -.25, .125], powers))

    def test_stopped_gap_encloses_independent_interval_corners_at_every_stop(self):
        rewards = [(F(0), F(1)), (F(2), F(3)), (F(3), F(4))]
        increments = [(F(-10**30), F(10**30)), (F(-1), F(2)), (F(-3), F(4))]
        for stop in range(3):
            lower, upper = component.gain_enclosure(rewards, increments, stop)
            for h, delta in product(product(*rewards), product(*increments)):
                m = [delta[0], delta[0]+delta[1], delta[0]+delta[1]+delta[2]]
                actual = max(h[j]-h[stop]-m[j]+m[stop] for j in range(3))
                self.assertLessEqual(lower, actual)
                self.assertGreaterEqual(upper, actual)

    def test_selected_date_and_arbitrary_initial_increment_cancel_exactly(self):
        self.assertEqual(component.gain_enclosure([(F(-9), F(10))],
            [(F(-10**100), F(10**100))], 0), (0, 0))

    def test_malformed_complete_exponents_and_coefficients_are_rejected(self):
        record, polynomial = fixture()
        altered = deepcopy(polynomial)
        altered['degrees']['2']['basis_exponents'][1][0] = True
        with self.assertRaises(ValueError):
            component.validate_export(record, altered, (2,))
        altered = deepcopy(polynomial)
        altered['degrees']['2']['coefficients'][1].pop()
        with self.assertRaises(ValueError):
            component.validate_export(record, altered, (2,))
        altered = deepcopy(polynomial)
        altered['degrees']['2']['coefficients'][1][0] = float('nan')
        with self.assertRaises(ValueError):
            component.validate_export(record, altered, (2,))

    def test_fit_metadata_and_correlation_domain_are_checked(self):
        record, polynomial = fixture()
        for key, value in [('root_seed', 72), ('sample_count', 3), ('value_fitting_paths', 65), ('value_spawn_key', [0])]:
            altered = deepcopy(polynomial); altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                component.validate_export(record, altered, (2,))
        record['model']['correlation'] = [[1., .1], [.1, 1.]]
        with self.assertRaises(ValueError):
            component.validate_export(record, polynomial, (2,))

    def test_cli_rejects_a_fit_export_bound_to_different_input_bytes(self):
        record, polynomial = fixture()
        polynomial['input_sha256'] = 'different_input'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            input_path, polynomial_path = path/'input.json', path/'polynomial.json'
            input_path.write_text(json.dumps(record))
            polynomial_path.write_text(json.dumps(polynomial))
            output = component.ROOT/'data/private'/('unwritten_'+path.name+'.json')
            with patch.object(sys, 'argv', ['certify_polynomial_dual', '--input', str(input_path),
                    '--polynomial', str(polynomial_path), '--output', str(output)]):
                with self.assertRaisesRegex(ValueError, 'different complete input'):
                    component.main()
            self.assertFalse(output.exists())


@unittest.skipIf(component.ctx is None, 'The separate python-flint environment is required')
class VerifiedChecks(unittest.TestCase):
    def test_one_asset_polynomial_moment_matches_independent_closed_moments(self):
        model = dict(rate=.05, dividends=[.1], volatilities=[.2], correlation=[[1.]])
        lower, upper, work = component.conditional_polynomial([125.], [1., -2., 3.],
            component.exponents(1, 2), model, F(1, 3), precision=96)
        with component.ctx.workprec(192):
            r, q, sigma = map(component.ball, [.05, .1, .2])
            first = component.ball(F(5, 4))*((r-q)/3).exp()
            second = component.ball(F(25, 16))*((2*(r-q)+sigma**2)/3).exp()
            oracle_lower, oracle_upper = component.endpoints(1-2*first+3*second)
        self.assertLessEqual(lower, oracle_lower)
        self.assertGreaterEqual(upper, oracle_upper)
        self.assertEqual(work['polynomial_terms_evaluated'], 3)

    def test_zero_and_signed_moments_have_finite_enclosures(self):
        record, _ = fixture(); model = record['model']
        powers = component.exponents(2, 2)
        lower, upper, work = component.conditional_polynomial([90., 120.], [0.]*6, powers, model, F(1))
        self.assertEqual((lower, upper), (0, 0))
        self.assertEqual(work['polynomial_terms_evaluated'], 0)
        lower, upper, _ = component.conditional_polynomial([90., 120.], [-3., -1., 0., 0., 0., 0.], powers, model, F(1))
        self.assertLessEqual(lower, upper)
        self.assertLess(upper, 0)

    def test_cached_moments_overlap_uncached_and_build_only_per_model_step_precision(self):
        record, _ = fixture(); model = record['model']
        powers = component.exponents(2, 2)
        coefficients = [1., -2., 3., -.1, .2, .3]
        cache = {}
        builds = exponentials = hits = 0
        for states in ([90., 110.], [120., 95.], [105., 88.]):
            lo, hi, work = component.conditional_polynomial(states, coefficients, powers, model,
                F(1), factor_cache=cache)
            reference_lo, reference_hi, _ = component.conditional_polynomial(states, coefficients,
                powers, model, F(1))
            self.assertLessEqual(max(lo, reference_lo), min(hi, reference_hi))
            builds += work['moment_factor_cache_builds']
            exponentials += work['moment_factor_exponentials']
            hits += work['moment_factor_cache_hits']
        self.assertEqual((len(cache), builds, exponentials, hits), (1, 1, 5, 2))
        _, _, work = component.conditional_polynomial([100., 100.], coefficients, powers, model,
            F(1, 2), factor_cache=cache)
        self.assertEqual(work['moment_factor_cache_builds'], 1)
        _, _, work = component.conditional_polynomial([100., 100.], coefficients, powers, model,
            F(1), precision=192, factor_cache=cache)
        self.assertEqual(work['moment_factor_cache_builds'], 1)
        altered = deepcopy(model); altered['rate'] = .06
        _, _, work = component.conditional_polynomial([100., 100.], coefficients, powers, altered,
            F(1), factor_cache=cache)
        self.assertEqual(work['moment_factor_cache_builds'], 1)
        self.assertEqual(len(cache), 4)

    def test_one_asset_terminal_integral_encloses_closed_call_with_model_discount(self):
        model = dict(rate=.05, dividends=[.1], volatilities=[.2], correlation=[[1.]],
                     strike=100., exercise_times=[0., F(1, 3)])
        lower, upper, work = component.conditional_terminal([100.], model, F(1, 3), F(1, 10000), precision=96)
        with component.ctx.workprec(192):
            state, strike = component.ball(100), component.ball(100)
            r, q, sigma, step = map(component.ball, [.05, .1, .2, F(1, 3)])
            sd = sigma*step.sqrt()
            d2 = ((state/strike).log()+(r-q-sigma**2/2)*step)/sd
            normal = lambda d: (1+(d/component.ball(2).sqrt()).erf())/2
            closed = state*((r-q)*step).exp()*normal(d2+sd)-strike*normal(d2)
            oracle_lower, oracle_upper = component.endpoints((-r*step).exp()*closed)
        self.assertLessEqual(lower, oracle_lower)
        self.assertGreaterEqual(upper, oracle_upper)
        self.assertLessEqual(upper-lower, F(1, 10000))
        self.assertGreater(work['conditional_integrals'], 0)
        self.assertGreater(work['integrand_evaluations'], 0)

    def test_terminal_domain_rejects_correlation_and_heterogeneity(self):
        record, _ = fixture(); model = record['model']
        for key, value in [('correlation', [[1., .1], [.1, 1.]]), ('volatilities', [.2, .3])]:
            altered = deepcopy(model); altered[key] = value
            with self.assertRaises(ValueError):
                component.conditional_terminal([100., 100.], altered, F(1), F(1, 100))

    def test_both_full_comparators_attain_width_and_preserve_input(self):
        record, polynomial = fixture()
        before = deepcopy((record, polynomial))
        result = component.assess(record, polynomial, target=F(1, 100))
        self.assertEqual((record, polynomial), before)
        self.assertTrue(result['observed_policy_decisions_checked'])
        self.assertTrue(result['exact_terminal_model_discount'])
        for name, row in result['degrees']['2']['methods'].items():
            self.assertTrue(row['all_targets_met'])
            self.assertLessEqual(row['maximum_signed_width'], .01)
            self.assertEqual(len(row['observations']), 2)
            self.assertEqual(row['conditional_integrals'] == 0, name == 'polynomial_all_dates')
            self.assertGreaterEqual(row['standalone_assessment_cpu_seconds_including_validation'], row['assessment_cpu_seconds'])
            self.assertEqual(row['moment_factor_cache_builds'], 1)
            self.assertEqual(row['moment_factor_exponentials'], 5)
            self.assertGreater(row['moment_factor_cache_hits'], 0)
            for observation in row['observations']:
                self.assertEqual(observation['selected_date_gap_certificate']['width']['numerator'], '0')
        json.dumps(result, allow_nan=False)

    def test_zero_polynomial_functions_enclose_direct_zero_martingale_gap(self):
        record, polynomial = fixture()
        polynomial['degrees']['2']['coefficients'] = [[0.]*6 for _ in range(3)]
        result = component.assess(record, polynomial, methods=('polynomial_all_dates',))
        rows = result['degrees']['2']['methods']['polynomial_all_dates']['observations']
        with component.ctx.workprec(192):
            for path, stop, row in zip(record['states'], record['stopping_indices'], rows):
                payoffs = [(-component.ball(.05)*component.ball(t)).exp()*component.ball(max(max(s)-100, 0))
                           for s, t in zip(path, record['model']['exercise_times'])]
                intervals = [(F(0), F(0)) if j == stop else component.endpoints(value-payoffs[stop])
                             for j, value in enumerate(payoffs)]
                oracle_lower = max(F(0), *(lo for lo, _ in intervals))
                oracle_upper = max(F(0), *(hi for _, hi in intervals))
                observed = row['signed_gap_certificate']
                self.assertLessEqual(component.load_fraction(observed['lower']), oracle_lower)
                self.assertGreaterEqual(component.load_fraction(observed['upper']), oracle_upper)

    def test_wrong_stopping_instruction_and_invalid_target_fail_closed(self):
        record, polynomial = fixture(); record['stopping_indices'][0] = 2
        with self.assertRaises(ArithmeticError):
            component.assess(record, polynomial)
        record, polynomial = fixture()
        with self.assertRaises(ValueError):
            component.assess(record, polynomial, target=F(0))
        with self.assertRaises(ValueError):
            component.assess(record, polynomial, methods=('unsupported',))


if __name__ == '__main__':
    unittest.main()
