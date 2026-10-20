"""Checks of ideal-model threshold losses, root allowances and analytic bounds."""

from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/certify_policy_loss_reference.py'
SPEC = importlib.util.spec_from_file_location('certify_policy_loss_reference', SCRIPT)
component = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(component)


def number(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def bounds(interval):
    return number(interval['lower']), number(interval['upper'])


@unittest.skipIf(component.flint is None, 'The separate python-flint environment is required')
class PolicyLossCertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = component.certify()

    def test_default_root_has_strict_correct_signs_and_requested_width(self):
        root_lo, root_hi = bounds(self.reference['boundary'])
        self.assertLessEqual(root_hi-root_lo, Fraction(1, 10**14))
        self.assertLess(bounds(self.reference['boundary_work']['lower_advantage'])[1], 0)
        self.assertGreater(bounds(self.reference['boundary_work']['upper_advantage'])[0], 0)
        self.assertTrue(root_lo > 110)
        self.assertTrue(root_hi < 111)

    def test_threshold_loss_classifications_have_separated_enclosures(self):
        expected = {
            100: ['exceeds_tolerance', 'exceeds_tolerance', 'within_tolerance'],
            110: ['within_tolerance']*3,
            120: ['exceeds_tolerance', 'within_tolerance', 'within_tolerance'],
            130: ['exceeds_tolerance', 'exceeds_tolerance', 'within_tolerance'],
            None: ['exceeds_tolerance']*3,
        }
        for row in self.reference['policies']:
            theta = None if row['threshold'] is None else number(row['threshold'])
            self.assertEqual([decision['classification'] for decision in row['decisions']], expected[theta])
            lower, upper = bounds(row['loss'])
            self.assertGreaterEqual(lower, 0)
            self.assertLess(upper-lower, Fraction(1, 10**15))
            self.assertEqual(number(row['loss']['width']), upper-lower)

    def test_analytic_bound_can_certify_without_population_loss_quadrature(self):
        theta110 = self.reference['policies'][1]
        upper = number(theta110['deterministic_gap_upper'])
        self.assertGreater(upper, Fraction(110, 1000))
        self.assertLess(upper, Fraction(111, 1000))
        self.assertEqual([row['classification'] for row in theta110['analytic_bound_decisions']],
                         ['unresolved', 'within_tolerance', 'within_tolerance'])
        for row in self.reference['policies'][:-1]:
            self.assertLessEqual(bounds(row['loss'])[1], number(row['deterministic_gap_upper']))
            self.assertGreaterEqual(row['work']['standalone_analytic_bound_cpu_seconds'],
                                    row['work']['analytic_bound']['cpu_seconds'])
        self.assertLessEqual(self.reference['root_free_model_setup']['cpu_seconds'],
                             self.reference['shared_model_and_boundary_setup']['cpu_seconds'])

    def test_exact_martingale_gap_is_bounded_on_both_sides_of_threshold(self):
        with component.ctx.workprec(192):
            params = {name: number(value) for name, value in self.reference['model'].items()}
            model = component.TwoRightModel(**params)
            for row in self.reference['policies'][:-1]:
                theta = number(row['threshold'])
                upper = number(row['deterministic_gap_upper'])
                for stock in (100, 101, 105, 110, 111, 115, 120, 130, 150, 200, 1000):
                    lo, hi = component.endpoints(model.exercise_advantage(component.ball(stock)))
                    gap_upper = max(Fraction(0), hi) if stock <= theta else max(Fraction(0), -lo)
                    self.assertLessEqual(gap_upper, upper)

    def test_alternative_standard_normal_integral_overlaps_stock_reference(self):
        # Independent variable transformation and explicit Black-Scholes formula.
        # Integrating only to b_l uses the same rigorous root-strip remainder.
        with component.ctx.workprec(192):
            a = component.ball
            first, horizon = a(1.5), a(1.5)
            sigma, r, q, strike = a(.2), a(.05), a(.1), a(100)
            sd = sigma*first.sqrt()
            mu = a(100).log()+(r-q-sigma**2/2)*first
            root_lo = bounds(self.reference['boundary'])[0]
            lower = (strike.log()-mu)/sd
            upper = (a(root_lo).log()-mu)/sd
            def callback(z, analytic):
                state = (mu+sd*z).exp()
                future_sd = sigma*horizon.sqrt()
                d2 = (state.log(analytic=analytic)-strike.log()+(r-q-sigma**2/2)*horizon)/future_sd
                cdf = lambda value: (-value/a(2).sqrt()).erfc()/2
                call = state*(-q*horizon).exp()*cdf(d2+future_sd)-strike*(-r*horizon).exp()*cdf(d2)
                advantage = (-r*first).exp()*(state-strike-call)
                return -advantage*(-z*z/2).exp()/(2*component.arb.pi()).sqrt()
            value = component.acb.integral(callback, component.acb(lower), component.acb(upper),
                                           abs_tol=a(Fraction(1, 10**25)),
                                           rel_tol=a(Fraction(1, 10**25)), use_heap=True)
            lo, hi = component.endpoints(value.real)
            hi += bounds(self.reference['policies'][0]['root_strip_allowance'])[1]
        target_lo, target_hi = bounds(self.reference['policies'][0]['loss'])
        self.assertLessEqual(lo, target_hi)
        self.assertGreaterEqual(hi, target_lo)

    def test_tail_refinement_preserves_reference_and_reduces_allowance(self):
        coarse = component.certify(cutoff=160)
        fine = self.reference
        coarse_wait, fine_wait = coarse['policies'][-1], fine['policies'][-1]
        self.assertGreater(bounds(coarse_wait['tail_allowance'])[1], bounds(fine_wait['tail_allowance'])[1])
        coarse_lo, coarse_hi = bounds(coarse_wait['loss'])
        fine_lo, fine_hi = bounds(fine_wait['loss'])
        self.assertLessEqual(coarse_lo, fine_hi)
        self.assertGreaterEqual(coarse_hi, fine_lo)
        self.assertGreater(coarse_hi-coarse_lo, fine_hi-fine_lo)

    def test_threshold_inside_root_bracket_uses_safe_strip(self):
        root_lo, root_hi = bounds(self.reference['boundary'])
        near_root = component.certify(thresholds=[(root_lo+root_hi)/2])
        row = near_root['policies'][0]
        self.assertEqual(bounds(row['finite_integral']), (0, 0))
        self.assertEqual(bounds(row['loss'])[0], 0)
        self.assertLess(bounds(row['loss'])[1], Fraction(1, 10**25))

    def test_exact_binary_inputs_and_reference_price_identity(self):
        self.assertEqual(number(self.reference['model']['rate']), Fraction.from_float(.05))
        self.assertEqual(number(self.reference['model']['dividend_yield']), Fraction.from_float(.1))
        expiry_lo, expiry_hi = bounds(self.reference['expiry_only_value'])
        loss_lo, loss_hi = bounds(self.reference['policies'][-1]['loss'])
        self.assertEqual(bounds(self.reference['optimal_two_right_value']), (expiry_lo+loss_lo, expiry_hi+loss_hi))
        self.assertTrue(self.reference['ideal_model_population_loss_enclosed'])
        self.assertFalse(self.reference['realised_simulation_law_certified'])
        self.assertFalse(self.reference['statistical_coverage_certified'])

    def test_endpoint_and_classification_boundaries(self):
        self.assertEqual(component.classify(Fraction(1), Fraction(2), Fraction(2)), 'within_tolerance')
        self.assertEqual(component.classify(Fraction(1), Fraction(2), Fraction(1)), 'unresolved')
        self.assertEqual(component.classify(Fraction(1), Fraction(2), Fraction(0)), 'exceeds_tolerance')
        with self.assertRaises(ArithmeticError):
            component.interval_record(Fraction(2), Fraction(1))
        with self.assertRaises(ArithmeticError):
            component.endpoints(component.arb('nan'))

    def test_analytic_branch_cut_is_rejected(self):
        with component.ctx.workprec(128):
            params = {name: number(value) for name, value in self.reference['model'].items()}
            integrand = component.SignedLossIntegrand(component.TwoRightModel(**params), 1)
            self.assertFalse(integrand(component.acb(-1), True).is_finite())
            self.assertTrue(integrand(component.acb(150), True).is_finite())

    def test_insufficient_precision_or_unsupported_domain_fails_closed(self):
        for kwargs in ({'spot': 0}, {'strike': -1}, {'rate': -.1}, {'dividend_yield': 0},
                       {'volatility': 0}, {'first_right': 0}, {'maturity': 1.5},
                       {'thresholds': []}, {'thresholds': [99]}, {'thresholds': [110, 110]},
                       {'tolerances': []}, {'tolerances': [0]}, {'tolerances': [.1, .1]},
                       {'precision': 63}, {'precision': True}, {'evaluation_limit': 0},
                       {'root_width': 0}, {'absolute_tolerance': 0}, {'relative_tolerance': 0},
                       {'rate': float('nan')}, {'spot': True}, {'cutoff': 101}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                component.certify(**kwargs)
        with self.assertRaises(ArithmeticError):
            component.certify(precision=64, root_width=Fraction(1, 10**30))

    def test_root_free_route_never_calls_root_search_or_quadrature(self):
        with patch.object(component, 'bracket_boundary', side_effect=AssertionError('No root search permitted')), \
             patch.object(component, 'finite_integral', side_effect=AssertionError('No finite integration permitted')), \
             patch.object(component.TwoRightModel, 'tail_upper', side_effect=AssertionError('No tail truncation permitted')):
            result = component.certify_analytic_bounds()
        self.assertFalse(result['boundary_search_executed'])
        self.assertFalse(result['numerical_quadrature_executed'])
        self.assertFalse(result['simulation_executed'])
        self.assertTrue(result['population_loss_upper_only'])
        for analytic, full in zip(result['policies'], self.reference['policies'][:-1]):
            self.assertEqual(analytic['deterministic_gap_upper'], full['deterministic_gap_upper'])
            self.assertEqual(analytic['analytic_gap_absolute_advantage'], full['analytic_gap_absolute_advantage'])
            self.assertEqual(analytic['decisions'], full['analytic_bound_decisions'])
            self.assertEqual(bounds(analytic['loss']), (0, number(analytic['deterministic_gap_upper'])))
            self.assertGreaterEqual(result['work']['cpu_seconds'], analytic['work']['cpu_seconds'])

    def test_root_free_decision_uses_rational_endpoint_not_display_float(self):
        result = component.certify_analytic_bounds(thresholds=[110])
        endpoint = number(result['policies'][0]['deterministic_gap_upper'])
        # The tiny offset is invisible in binary64 but must alter the decision.
        below = endpoint-Fraction(1, 10**100)
        above = endpoint+Fraction(1, 10**100)
        self.assertEqual(float(below), float(above))
        refined = component.certify_analytic_bounds(thresholds=[110], tolerances=[below, endpoint, above])
        self.assertEqual([row['classification'] for row in refined['policies'][0]['decisions']],
                         ['unresolved', 'within_tolerance', 'within_tolerance'])

    def test_root_free_upper_bound_failure_is_unresolved_not_rejection(self):
        row = component.certify_analytic_bounds(thresholds=[110], tolerances=[Fraction(1, 10)])['policies'][0]
        self.assertEqual(row['decisions'][0]['classification'], 'unresolved')
        # The independent integral shows this policy's actual loss is below .1.
        self.assertLess(bounds(self.reference['policies'][1]['loss'])[1], Fraction(1, 10))

    def test_root_free_route_rejects_unsupported_or_ambiguous_inputs(self):
        for kwargs in ({'spot': 0}, {'strike': -1}, {'rate': -.1}, {'dividend_yield': 0},
                       {'volatility': 0}, {'first_right': 0}, {'maturity': 1.5},
                       {'thresholds': []}, {'thresholds': [99]}, {'thresholds': [110, 110]},
                       {'thresholds': None}, {'tolerances': None},
                       {'tolerances': []}, {'tolerances': [0]}, {'tolerances': [.1, .1]},
                       {'precision': 63}, {'precision': True}, {'rate': float('nan')}, {'spot': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                component.certify_analytic_bounds(**kwargs)
        with patch.object(component, 'threshold_gap_bound', side_effect=ArithmeticError('Non-finite formula evaluation')):
            with self.assertRaises(ArithmeticError):
                component.certify_analytic_bounds()

    def test_nonfinite_quadrature_fails_closed(self):
        with patch.object(component, 'finite_integral', side_effect=ArithmeticError('Non-finite quadrature')):
            with self.assertRaises(ArithmeticError):
                component.certify()


if __name__ == '__main__':
    unittest.main()
