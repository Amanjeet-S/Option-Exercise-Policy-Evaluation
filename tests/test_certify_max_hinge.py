"""Independent mathematical checks for the conditional max-hinge component."""

import importlib.util
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/certify_max_hinge.py'
SPEC = importlib.util.spec_from_file_location('certify_max_hinge', SCRIPT)
component = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(component)


def number(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def bounds(certificate, name='expectation'):
    interval = certificate[name]
    return number(interval['lower']), number(interval['upper'])


@unittest.skipIf(component.flint is None, 'The optional python-flint dependency is required')
class MaxHingeCertificationTests(unittest.TestCase):
    def test_single_asset_encloses_closed_form_call(self):
        # This analytic value is computed at higher precision independently of
        # the quadrature. It checks finite integration plus the missing tail.
        result = component.certify([100], cutoff=170, precision=96)
        with component.ctx.workprec(192):
            b = component.ball
            sd = b(.2)*b(1/3).sqrt()
            d2 = ((b(.05)-b(.10)-b(.2)**2/2)*b(1/3))/sd
            value = b(100)*(b(.05-.10)*b(1/3)).exp()*component.normal_cdf(d2+sd)-b(100)*component.normal_cdf(d2)
            analytic_lo, analytic_hi = component.endpoints(value)
        lo, hi = bounds(result)
        self.assertLessEqual(lo, analytic_lo)
        self.assertGreaterEqual(hi, analytic_hi)

    def test_dimension_and_state_ordering(self):
        one = component.certify([100], cutoff=220)
        five = component.certify([100]*5, cutoff=220)
        higher = component.certify([110]*5, cutoff=220)
        self.assertLess(bounds(one)[1], bounds(five)[0])
        self.assertLess(bounds(five)[1], bounds(higher)[0])

    def test_cutoff_reduces_tail_and_expectation_width(self):
        coarse = component.certify([110]*5, cutoff=150, precision=96)
        fine = component.certify([110]*5, cutoff=220, precision=96)
        self.assertLess(bounds(fine, 'tail_allowance')[1], bounds(coarse, 'tail_allowance')[1])
        self.assertLess(number(fine['expectation']['width']), number(coarse['expectation']['width']))
        self.assertLessEqual(bounds(coarse)[0], bounds(fine)[1])
        self.assertLessEqual(bounds(fine)[0], bounds(coarse)[1])

    def test_precision_refinement_has_consistent_finite_integral(self):
        coarse = component.certify([90, 95, 100, 105, 110], precision=64,
                                   absolute_tolerance=1e-8, relative_tolerance=1e-8)
        fine = component.certify([90, 95, 100, 105, 110], precision=160,
                                 absolute_tolerance=1e-18, relative_tolerance=1e-18)
        self.assertLess(number(fine['finite_integral']['width']), number(coarse['finite_integral']['width']))
        self.assertLessEqual(bounds(coarse, 'finite_integral')[0], bounds(fine, 'finite_integral')[1])
        self.assertLessEqual(bounds(fine, 'finite_integral')[0], bounds(coarse, 'finite_integral')[1])

    def test_binary_input_record_is_exact(self):
        result = component.certify([100]*5)
        self.assertEqual(number(result['model']['step']), Fraction.from_float(1/3))
        self.assertEqual(number(result['model']['rate']), Fraction.from_float(.05))
        self.assertEqual(number(result['expectation']['width']), bounds(result)[1]-bounds(result)[0])
        self.assertFalse(result['statistical_coverage_certified'])

    def test_branch_cut_rejected_in_analytic_callback(self):
        with component.ctx.workprec(96):
            integrand = component.SurvivalIntegrand((component.ball(100).log(),),
                                                   component.ball(0), component.ball(.2))
            self.assertFalse(integrand(component.acb(-1), True).is_finite())
            self.assertTrue(integrand(component.acb(100), True).is_finite())

    def test_nonfinite_integral_fails_closed(self):
        class FakeAcb:
            def __call__(self, x):
                return component.arb(x)

            def integral(self, *args, **kwargs):
                return component.arb('nan')
        with patch.object(component, 'acb', FakeAcb()):
            with self.assertRaises(ArithmeticError):
                component.certify([100]*5)

    def test_unsupported_or_invalid_inputs_rejected(self):
        for kwargs in ({'volatility': 0}, {'volatility': [.2]*5},
                       {'correlation': .1}, {'step': 0}, {'strike': 0},
                       {'cutoff': 100}, {'rate': float('nan')},
                       {'precision': 31}, {'evaluation_limit': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                component.certify([100]*5, **kwargs)
        for states in ([], [0], [-1], [float('inf')]):
            with self.subTest(states=states), self.assertRaises(ValueError):
                component.certify(states)


if __name__ == '__main__':
    unittest.main()
