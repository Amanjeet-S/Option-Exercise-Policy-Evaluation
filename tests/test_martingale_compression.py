"""Literal-coefficient checks of projection and actual-law energy enclosures.

No random generator is called. NumPy and outward-arithmetic checks execute
in their separate environments. Direct source loading leaves the original
package initialiser and registered numerical components unchanged.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest


ROOT = Path(os.environ.get('QF_TEST_REPOSITORY', Path(__file__).resolve().parents[1])).resolve()
SOURCE = ROOT/'src/qf_research/optimal_stopping/martingale_compression.py'
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name('martingale_compression.py')
spec = importlib.util.spec_from_file_location('compression_fixture_source', SOURCE)
compression = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compression
spec.loader.exec_module(compression)
NUMPY = importlib.util.find_spec('numpy') is not None
FLINT = importlib.util.find_spec('flint') is not None


def parameters(d=3, **changes):
    result = dict(rate=Fraction(1, 20), dividends=(Fraction(1, 10),)*d,
        volatilities=(Fraction(1, 5),)*d, correlation=Fraction(0))
    result.update(changes)
    return result


def coefficients(d=3, degree=3):
    return tuple(Fraction((i%7)-3, 5) for i in range(len(compression.complete_monomial_indices(d, degree))))


class InputAndAlgebraChecks(unittest.TestCase):
    def test_law_validation_is_exact_and_does_not_require_numpy(self):
        initial, t0, t1, law, signatures = compression._setup((1, 1, 1), 0, 1, degree=3,
            **parameters(correlation=Fraction(-1, 2)))
        self.assertEqual(law.correlation, Fraction(-1, 2))
        self.assertEqual(len(signatures), 20)
        self.assertEqual(compression._cross((0,), (1,), law), Fraction(-1, 50))
        self.assertEqual(compression._cross((0,), (0,), law), Fraction(1, 25))

    def test_rejected_times_parameters_and_coefficient_lengths(self):
        for state, t0, t1, updates in [((1, 1, 1), -1, 1, {}),
            ((1, 1, 1), 2, 1, {}), ((1, 1, 1), 0, 1, {'correlation': Fraction(-3, 5)}),
            ((1, 1, 1), 0, 1, {'volatilities': (.1, .2, .3), 'correlation': .2}),
            ((1, float('nan'), 1), 0, 1, {})]:
            with self.subTest(state=state, t0=t0, t1=t1, updates=updates):
                with self.assertRaises(ValueError):
                    compression._setup(state, t0, t1, degree=3, **parameters(**updates))
        with self.assertRaises(ValueError):
            compression._coefficients([1], 2, 'Coefficients')
        with self.assertRaises(ValueError):
            compression._coefficients([True], 1, 'Coefficients')

    def test_psi_cross_identity_including_negative_covariance(self):
        _, _, _, law, _ = compression._setup((1, 1, 1), 0, 1, degree=3,
            **parameters(dividends=(Fraction(1, 50), Fraction(1, 10), Fraction(9, 50)), correlation=Fraction(-1, 2)))
        first, second = (0, 0, 1), (1, 2)
        combined = tuple(sorted(first+second))
        self.assertEqual(compression._psi(combined, law)-compression._psi(first, law)-compression._psi(second, law),
            compression._cross(first, second, law))

    def test_rational_records_do_not_round_authoritative_values(self):
        value = Fraction(1, 3)
        self.assertEqual(compression.decode_rational(compression.rational_record(value)), value)
        self.assertEqual(compression.decode_interval(compression.interval_record(0, value)), (0, value))
        huge = compression.rational_record(Fraction(10)**400)
        self.assertIsNone(huge['approximation'])
        invalid = deepcopy(compression.interval_record(0, value)); invalid['width'] = compression.rational_record(0)
        with self.assertRaises(ValueError):
            compression.decode_interval(invalid)

    def test_compressed_higher_degree_terms_cannot_be_omitted_from_a_certificate(self):
        for degree, forbidden_orbit in [(0, 1), (1, 2), (2, 4)]:
            retained = (0,)*len(compression.complete_monomial_indices(1, degree))
            compressed = [0]*7; compressed[forbidden_orbit] = 1
            with self.subTest(degree=degree, forbidden_orbit=forbidden_orbit):
                with self.assertRaises(ValueError):
                    compression.certify_residual_energy(retained, compressed, (1,), 0, 1,
                        degree=degree, **parameters(1))


@unittest.skipUnless(NUMPY, 'The nominal SVD tests require the research NumPy environment')
class FloatingProjectionChecks(unittest.TestCase):
    def test_nominal_gram_matches_direct_second_moment_difference(self):
        import numpy as np
        initial, t0, t1, law, signatures = compression._setup((Fraction(9, 10), 1, Fraction(11, 10)), Fraction(1, 2), Fraction(3, 2), degree=2,
            **parameters(dividends=(.02, .1, .18), volatilities=(.1, .2, .3)))
        actual, _ = compression._nominal_gram(initial, t0, law, signatures, 10000)
        reference = np.zeros_like(actual)
        for i, first in enumerate(signatures):
            for j, second in enumerate(signatures):
                combined = tuple(sorted(first+second))
                initial_value = float(compression._initial_monomial(initial, combined))
                first_moment = initial_value*math.exp(float(t1*compression._psi(combined, law)))
                centres = initial_value*math.exp(float(t0*compression._psi(combined, law)+law.step*(compression._psi(first, law)+compression._psi(second, law))))
                reference[i, j] = first_moment-centres
        np.testing.assert_allclose(actual, reference, rtol=2e-11, atol=2e-15)
        np.testing.assert_array_equal(actual, actual.T)

    def test_exchangeable_energy_projection_matches_exact_orbit_average(self):
        projected = compression.fit_energy_projection(coefficients(), (1, 1, 1), Fraction(1, 2), Fraction(3, 2), **parameters())
        averaged = compression.project_coefficients(coefficients(), 3)
        for expected, actual in zip(averaged, projected.coefficients):
            self.assertAlmostEqual(float(expected), float(actual), places=8)
        self.assertFalse(projected.diagnostics['exact_optimiser_certified'])
        self.assertFalse(projected.diagnostics['negative_eigenvalues_clipped'])

    def test_heterogeneous_carry_and_volatility_change_the_projection(self):
        averaged = compression.project_coefficients(coefficients(), 3)
        settings = [parameters(dividends=(.02, .1, .18)), parameters(volatilities=(.1, .2, .3))]
        for setting in settings:
            projected = compression.fit_energy_projection(coefficients(), (1, 1, 1), Fraction(1, 2), Fraction(3, 2), **setting)
            self.assertGreater(max(abs(float(x-y)) for x, y in zip(projected.coefficients, averaged)), 1e-5)
            self.assertLessEqual(projected.diagnostics['nominal_residual_energy'], projected.diagnostics['nominal_averaging_energy']+1e-10)
            self.assertTrue(all(isinstance(value, Fraction) for value in projected.coefficients))
            self.assertEqual(len(projected.record()['frozen_orbit_coefficients']), 7)

    def test_singular_correlations_and_absent_orbits_have_explicit_rank(self):
        perfect = compression.fit_energy_projection(coefficients(), (1, 1, 1), 0, 1, **parameters(correlation=1))
        self.assertEqual(perfect.diagnostics['nominal_rank'], 3)
        for d, expected in [(1, [1, 2, 4]), (2, [1, 2, 3, 4, 5])]:
            result = compression.fit_energy_projection(coefficients(d), (1,)*d, 0, 1, **parameters(d))
            self.assertEqual(result.diagnostics['present_orbit_indices'], expected)
            self.assertEqual(result.coefficients[6], 0)
        lower = compression.fit_energy_projection(coefficients(), (1, 1, 1), 0, 1, **parameters(correlation=Fraction(-1, 2)))
        self.assertTrue(lower.diagnostics['nominal_rank'] > 0)
        constant = compression.fit_energy_projection((Fraction(7, 8),), (1,), 0, 1, degree=0, **parameters(1))
        self.assertEqual(constant.diagnostics['present_orbit_indices'], [])
        self.assertEqual(constant.diagnostics['nominal_rank'], 0)
        self.assertEqual(constant.coefficients, (Fraction(7, 8),)+(Fraction(0),)*6)

    def test_deterministic_steps_and_dense_resource_limits(self):
        for t0, t1, setting in [(0, 0, parameters()), (0, 1, parameters(volatilities=0))]:
            result = compression.fit_energy_projection(coefficients(), (1, 1, 1), t0, t1, **setting)
            self.assertEqual(result.diagnostics['nominal_rank'], 0)
            self.assertEqual(result.diagnostics['nominal_residual_energy'], 0.)
        with self.assertRaises(ValueError):
            compression.fit_energy_projection(coefficients(), (1, 1, 1), 0, 1, max_gram_entries=1, **parameters())
        with self.assertRaises(ValueError):
            compression.fit_energy_projection(coefficients(), (1, 1, 1), 0, 1, rcond=0, **parameters())

    def test_underflowing_nominal_moments_fail_instead_of_becoming_zero(self):
        with self.assertRaises(ArithmeticError):
            compression._finite_float(Fraction(1, 10**500), 'Tiny covariance')
        with self.assertRaises(ArithmeticError):
            compression.fit_energy_projection((0, 1), (1,), 0, 1, degree=1,
                **parameters(1, dividends=1000))


@unittest.skipUnless(FLINT, 'The certificate tests require the separate outward-arithmetic environment')
class OutwardCertificateChecks(unittest.TestCase):
    def independent_expression(self, expression):
        from flint import arb, ctx, fmpq
        def ball(value):
            value = Fraction(value)
            return arb(fmpq(value.numerator, value.denominator))
        with ctx.workprec(256):
            value = expression(ball)
            lo, hi = value.lower().fmpq(), value.upper().fmpq()
            return Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q))

    def assert_contains(self, record, expected):
        lower, upper = compression.decode_interval(record)
        self.assertLessEqual(lower, expected[0])
        self.assertGreaterEqual(upper, expected[1])

    def test_linear_energy_contains_independent_256bit_moment_difference(self):
        result = compression.certify_residual_energy((0, 1), (0,)*7, (1,), 1, 2, degree=1, **parameters(1))
        mu, variance = Fraction(-1, 20), Fraction(1, 25)
        expected = self.independent_expression(lambda b: (b(4*mu+2*variance)).exp()-(b(4*mu+variance)).exp())
        self.assert_contains(result['energy_certificate'], expected)
        self.assertGreater(compression.decode_rational(result['single_increment_expected_dual_gap_difference_upper']), 0)

    def test_negative_offdiagonal_covariances_are_retained(self):
        result = compression.certify_residual_energy((0, 1, -1), (0,)*7, (1, 1), 0, 1, degree=1,
            **parameters(2, correlation=-1))
        mu, variance = Fraction(-1, 20), Fraction(1, 25)
        expected = self.independent_expression(lambda b: b(2)*(b(2*mu+variance).exp()-b(2*mu-variance).exp()))
        self.assert_contains(result['energy_certificate'], expected)
        self.assertGreater(compression.decode_interval(result['energy_certificate'])[0], 0)

    def test_near_zero_residual_is_not_obtained_by_subtracting_large_energies(self):
        residual = Fraction(1, 2**40)
        compressed = (Fraction(0), 1-residual)+(Fraction(0),)*5
        result = compression.certify_residual_energy((0, 1), compressed, (1,), 1, 2, degree=1, **parameters(1))
        mu, variance = Fraction(-1, 20), Fraction(1, 25)
        expected = self.independent_expression(lambda b: b(residual**2)*(b(4*mu+2*variance).exp()-b(4*mu+variance).exp()))
        self.assert_contains(result['energy_certificate'], expected)
        lower, upper = compression.decode_interval(result['energy_certificate'])
        self.assertGreater(lower, 0)
        self.assertLess(upper, Fraction(1, 10**20))

    def test_exact_symmetric_residual_is_zero_under_heterogeneous_actual_law(self):
        signatures = compression.complete_monomial_indices(3)
        retained = tuple(Fraction(compression.monomial_orbit_index(signature), 16) for signature in signatures)
        averaged = compression.project_coefficients(retained, 3)
        result = compression.certify_residual_energy(retained, averaged, (.9, 1, 1.1), 0, 2,
            **parameters(dividends=(.02, .1, .18), volatilities=(.1, .2, .3)))
        self.assertEqual(compression.decode_interval(result['energy_certificate']), (0, 0))
        self.assertTrue(result['exact_zero_martingale_residual'])
        self.assertEqual(result['work']['monomial_pairs_evaluated'], 0)

    def test_constant_kernel_and_deterministic_increments_are_exact_zero(self):
        for values, t0, t1, setting in [((1, 0, 0, 0), 0, 1, parameters(1)),
            ((1, 2, 3, 4), 0, 0, parameters(1)), ((1, 2, 3, 4), 0, 1, parameters(1, volatilities=0))]:
            result = compression.certify_residual_energy(values, (0,)*7, (1,), t0, t1, **setting)
            self.assertEqual(compression.decode_interval(result['energy_certificate']), (0, 0))
            self.assertEqual(compression.decode_interval(result['square_root_energy_certificate']), (0, 0))
            self.assertTrue(result['exact_zero_martingale_residual'])

    def test_valid_constant_and_quadratic_degree_certificates(self):
        constant = compression.certify_residual_energy((1,), (0,)*7, (1,), 0, 1, degree=0, **parameters(1))
        self.assertEqual(compression.decode_interval(constant['energy_certificate']), (0, 0))
        quadratic = compression.certify_residual_energy((0, 0, 1), (0,)*7, (1,), 0, 1, degree=2, **parameters(1))
        expected = self.independent_expression(lambda b: b(Fraction(1, 25)).exp()-b(Fraction(-3, 25)).exp())
        self.assert_contains(quadratic['energy_certificate'], expected)

    def test_calendar_combination_sums_energy_before_taking_square_root(self):
        first = compression.certify_residual_energy((0, 1), (0,)*7, (1,), 0, 1, degree=1, **parameters(1))
        second = compression.certify_residual_energy((0, 1), (0,)*7, (1,), 1, 2, degree=1, **parameters(1))
        combined = compression.combine_energy_certificates([first, second])
        expected_upper = sum(compression.decode_interval(x['energy_certificate'])[1] for x in (first, second))
        self.assertEqual(compression.decode_interval(combined['terminal_energy_certificate'])[1], expected_upper)
        self.assertEqual(combined['maximal_inequality_constant'], 1)
        self.assertTrue(combined['bound_is_difference_of_expectations'])
        self.assertFalse(combined['bound_is_pathwise'])
        self.assertFalse(combined['sampling_coverage_certified'])

    def test_overlapping_models_false_scope_and_malformed_intervals_fail(self):
        first = compression.certify_residual_energy((0, 1), (0,)*7, (1,), 0, 1, degree=1, **parameters(1))
        overlap = compression.certify_residual_energy((0, 1), (0,)*7, (1,), Fraction(1, 2), Fraction(3, 2), degree=1, **parameters(1))
        changed = deepcopy(first); changed['model']['rate'] = compression.rational_record(0)
        false = deepcopy(first); false['population_energy_arithmetic_certified'] = False
        for second in (overlap, changed, false):
            with self.assertRaises(ValueError):
                compression.combine_energy_certificates([first, second])
        with self.assertRaises(ValueError):
            compression.combine_energy_certificates([])
        with self.assertRaises(ValueError):
            compression.certify_residual_energy((0, 1), (0,)*7, (1,), 0, 1, degree=1, precision=16, **parameters(1))


if __name__ == '__main__':
    unittest.main()
