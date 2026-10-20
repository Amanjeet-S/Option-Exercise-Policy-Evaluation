"""Independent finite algebra and outward checks for the positive energy draft.

Author: Amanjeet Singh. No random generator is used. Standard-library checks
also run without NumPy or Flint; outward checks use the separate Flint runtime.
"""
from collections import Counter
from fractions import Fraction
import importlib.util
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1]/'src/qf_research/optimal_stopping/positive_martingale_energy.py'
SPEC = importlib.util.spec_from_file_location('positive_energy_fixture', SOURCE)
energy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = energy
SPEC.loader.exec_module(energy)
FLINT = importlib.util.find_spec('flint') is not None


def parameters(dimension, **changes):
    answer = dict(rate=Fraction(1, 20), dividends=tuple(Fraction(i+1, 30) for i in range(dimension)),
        volatilities=tuple(Fraction(i+1, 10) for i in range(dimension)), correlation=0)
    answer.update(changes)
    return answer


def moment(mean, q, order):
    return mean**order*q**(order*(order-1)//2)


def coefficients(mean, q):
    return ((Fraction(1),), (mean, Fraction(1)),
        (mean**2*q, mean*q*(q+1), Fraction(1)),
        (mean**3*q**3, mean**2*q**3*(q*q+q+1), mean*q*q*(q*q+q+1), Fraction(1)))


def norm(mean, q, order):
    answer = mean**(2*order)*q**(3*order*(order-1)//2)
    for ell in range(1, order+1):
        answer *= q**ell-1
    return answer


def polynomial_inner(first, second, mean, q):
    return sum((a*b*moment(mean, q, i+j) for i, a in enumerate(first)
        for j, b in enumerate(second)), Fraction(0))


def orthogonal_polynomials_from_moments(mean, q, degree):
    """Gram-Schmidt from moments, independently of displayed C coefficients."""
    polys, norms = [], []
    for order in range(degree+1):
        current = [Fraction(0)]*order+[Fraction(1)]
        for previous, previous_norm in zip(polys, norms):
            projection = polynomial_inner(current, previous, mean, q)/previous_norm
            for k, coefficient in enumerate(previous):
                current[k] -= projection*coefficient
        polys.append(tuple(current))
        norms.append(polynomial_inner(current, current, mean, q))
    return tuple(polys), tuple(norms)


def polynomials_from_triangular_coefficients(mean, q):
    polys = []
    for order, row in enumerate(coefficients(mean, q)):
        current = [Fraction(0)]*order+[Fraction(1)]
        for previous, multiplier in zip(polys, row):
            for k, coefficient in enumerate(previous):
                current[k] -= multiplier*coefficient
        polys.append(tuple(current))
    return tuple(polys)


def determinant(matrix):
    matrix = [list(row) for row in matrix]
    result = Fraction(1)
    for i in range(len(matrix)):
        pivot = next((j for j in range(i, len(matrix)) if matrix[j][i]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != i:
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            result = -result
        result *= matrix[i][i]
        for j in range(i+1, len(matrix)):
            scale = matrix[j][i]/matrix[i][i]
            for k in range(i+1, len(matrix)):
                matrix[j][k] -= scale*matrix[i][k]
    return result


def dense_rational_gram(first, second, previous_mean, previous_q, return_mean, return_q):
    a, b = Counter(first), Counter(second)
    previous_product = return_joint = return_first = return_second = Fraction(1)
    for i in set(a)|set(b):
        previous_product *= moment(previous_mean[i], previous_q[i], a[i]+b[i])
        return_joint *= moment(return_mean[i], return_q[i], a[i]+b[i])
        return_first *= moment(return_mean[i], return_q[i], a[i])
        return_second *= moment(return_mean[i], return_q[i], b[i])
    return previous_product*(return_joint-return_first*return_second)


def positive_rational_gram(first, second, structure, previous_mean, previous_q, return_mean, return_q):
    """Use independent Gram-Schmidt C coefficients for non-degenerate laws."""
    def expanded(mean, q):
        if mean > 0 and q > 1:
            polys, norms = orthogonal_polynomials_from_moments(mean, q, 3)
            return tuple(tuple(polynomial_inner((Fraction(0),)*n+(Fraction(1),), poly, mean, q)/h
                for poly, h in zip(polys[:n+1], norms[:n+1])) for n in range(4))
        return coefficients(mean, q)
    r_c = [expanded(y, q) for y, q in zip(return_mean, return_q)]
    tilted_c = {(i, a): expanded(y*q**(2*a), q)
        for i, (y, q) in enumerate(zip(previous_mean, previous_q)) for a in range(4)}
    answer = Fraction(0)
    first_column = structure['signatures'].index(first)
    second_column = structure['signatures'].index(second)
    for (alpha, gamma), entries in structure['rows'].items():
        a_counts, g_counts = Counter(alpha), Counter(gamma)
        weight = Fraction(1)
        for i, a in a_counts.items():
            weight *= norm(return_mean[i], return_q[i], a)*moment(previous_mean[i], previous_q[i], 2*a)
        for i, g in g_counts.items():
            weight *= norm(previous_mean[i]*previous_q[i]**(2*a_counts[i]), previous_q[i], g)
        if not weight:
            continue
        left = right = Fraction(0)
        for column, local in entries:
            if column not in (first_column, second_column):
                continue
            factor = Fraction(1)
            for i, n, a, g in local:
                factor *= r_c[i][n][a]*tilted_c[i, a][n-a][g]
            if column == first_column:
                left = factor
            if column == second_column:
                right = factor
        answer += weight*left*right
    return answer


class ExactAlgebraChecks(unittest.TestCase):
    def test_coefficients_orthogonality_and_norms_against_exact_moments_through_six(self):
        for mean, q in [(Fraction(2, 3), Fraction(3, 2)), (Fraction(7, 5), Fraction(9, 8)),
                (Fraction(1), Fraction(1)), (Fraction(0), Fraction(5, 4)), (Fraction(0), Fraction(1))]:
            with self.subTest(mean=mean, q=q):
                polys = polynomials_from_triangular_coefficients(mean, q)
                for i, first in enumerate(polys):
                    self.assertEqual(first[-1], 1)
                    for j, second in enumerate(polys):
                        expected = norm(mean, q, i) if i == j else 0
                        self.assertEqual(polynomial_inner(first, second, mean, q), expected)
                if mean > 0 and q > 1:
                    independent, independent_norms = orthogonal_polynomials_from_moments(mean, q, 3)
                    self.assertEqual(polys, independent)
                    self.assertEqual(independent_norms, tuple(norm(mean, q, i) for i in range(4)))

    def test_hankel_vandermonde_norm_derivation_is_exact(self):
        mean, q = Fraction(7, 6), Fraction(5, 4)
        previous = Fraction(1)
        for order in range(4):
            actual = determinant([[moment(mean, q, i+j) for j in range(order+1)] for i in range(order+1)])
            expected = mean**(order*(order+1))*q**(order*(order-1)*(order+1)//2)
            for ell in range(1, order+1):
                expected *= (q**ell-1)**(order+1-ell)
            self.assertEqual(actual, expected)
            self.assertEqual(actual/previous, norm(mean, q, order))
            previous = actual

    def test_all_small_gram_entries_equal_independent_rational_moments(self):
        equalities = 0
        for dimension in range(1, 4):
            signatures = energy.complete_monomial_indices(dimension)
            structure = energy.factor_structure(signatures, dimension=dimension)
            laws = [
                ([Fraction(i+2, 3) for i in range(dimension)], [Fraction(i+4, i+3) for i in range(dimension)],
                 [Fraction(i+3, 4) for i in range(dimension)], [Fraction(i+7, i+5) for i in range(dimension)]),
                ([Fraction(0)]+[Fraction(i+2, 3) for i in range(1, dimension)], [Fraction(1)]*dimension,
                 [Fraction(i+3, 4) for i in range(dimension)], [Fraction(i+7, i+5) for i in range(dimension)]),
                ([Fraction(i+2, 3) for i in range(dimension)], [Fraction(i+4, i+3) for i in range(dimension)],
                 [Fraction(i+3, 4) for i in range(dimension)], [Fraction(1)]+[Fraction(i+7, i+5) for i in range(1, dimension)]),
            ]
            for setting, law in enumerate(laws):
                for first, second in product(signatures, repeat=2):
                    with self.subTest(dimension=dimension, law=setting, first=first, second=second):
                        self.assertEqual(positive_rational_gram(first, second, structure, *law),
                            dense_rational_gram(first, second, *law))
                    equalities += 1
        self.assertEqual(equalities, 1548)

    def test_sparse_patterns_rows_and_storage_counts(self):
        for signature, expected in [((), 0), ((0,), 1), ((0, 0), 3), ((0, 1), 5),
                ((0, 0, 0), 6), ((0, 0, 1), 12), ((0, 1, 2), 19)]:
            result = energy.factor_structure((signature,), dimension=3)
            self.assertEqual(result['factor_entries'], expected)
            self.assertEqual(energy._entry_count(signature), expected)
            self.assertEqual(result['factor_rows'], expected)
            for (alpha, gamma), entries in result['rows'].items():
                self.assertTrue(alpha)
                self.assertLessEqual(len(alpha)+len(gamma), 3)
                for _, local in entries:
                    self.assertLessEqual(len(local), 3)
        structure = energy.factor_structure(energy.complete_monomial_indices(5), dimension=5)
        self.assertEqual((len(structure['signatures']), structure['factor_rows'], structure['factor_entries']), (56, 230, 530))
        for dimension in range(1, 7):
            structure = energy.factor_structure(energy.complete_monomial_indices(dimension), dimension=dimension)
            self.assertLessEqual(structure['factor_rows'], 7*len(structure['signatures']))
        # One sparse cubic has nineteen rows, so 7 times its non-zero count
        # would be a false bound. Coordinates can be large without a d scan.
        sparse = energy.factor_structure(((2, 900, 999),), dimension=1000)
        self.assertEqual(sparse['factor_rows'], 19)

    def test_downward_closed_dictionary_has_seven_m_row_bound(self):
        dictionary = ((), (0,), (2,), (0, 0), (0, 2), (2, 2), (0, 0, 2))
        structure = energy.factor_structure(dictionary, dimension=4)
        self.assertLessEqual(structure['factor_rows'], 7*len(dictionary))

    def test_resource_guard_runs_before_factor_entries_are_allocated(self):
        with patch.object(energy, '_factor_patterns', side_effect=AssertionError('Allocated a factor before its guard')):
            with self.assertRaisesRegex(ValueError, 'entry budget'):
                energy.factor_structure(((0, 1, 2),), dimension=3, max_factor_entries=18)
            with self.assertRaisesRegex(ValueError, 'entry budget'):
                energy.certify_independent_polynomial_energy({(0, 1, 2): 1}, (1, 1, 1), 0, 1,
                    max_factor_entries=18, **parameters(3))
        for maximum in (True, 0, -1, 2.5):
            with self.assertRaises(ValueError):
                energy.factor_structure(((0,),), dimension=1, max_factor_entries=maximum)

    def test_endpoint_square_preserves_non_negative_upper_bound(self):
        for endpoints, expected in [((-2, 3), (0, 9)), ((-3, -2), (4, 9)), ((2, 3), (4, 9)), ((0, 0), (0, 0))]:
            self.assertEqual(energy._square_endpoints(*map(Fraction, endpoints)), expected)
        with self.assertRaises(ArithmeticError):
            energy._square_endpoints(Fraction(1), Fraction(0))

    def test_weight_endpoint_intersection_preserves_upper_and_rejects_negative_upper(self):
        self.assertEqual(energy._positive_endpoints(None, lambda value: (Fraction(-1), Fraction(2))), (0, 2))
        self.assertEqual(energy._positive_endpoints(None, lambda value: (Fraction(1), Fraction(2))), (1, 2))
        with self.assertRaises(ArithmeticError):
            energy._positive_endpoints(None, lambda value: (Fraction(-2), Fraction(-1)))

    def test_validation_rejects_covariance_nonfinite_bool_and_malformed_signatures(self):
        changes = [dict(correlation=Fraction(1, 2)), dict(correlation=-1),
            dict(volatilities=0, correlation=1), dict(rate=True), dict(dividends=float('nan')),
            dict(volatilities=float('inf')), dict(volatilities=-1)]
        for setting in changes:
            with self.subTest(setting=setting):
                with self.assertRaises(ValueError):
                    energy.certify_independent_polynomial_energy((0, 1), (1,), 0, 1, degree=1, **parameters(1, **setting))
        for x0, t0, t1, coefficient in [((True,), 0, 1, 1), ((-1,), 0, 1, 1),
                ((1,), True, 1, 1), ((1,), 2, 1, 1), ((1,), -1, 1, 1),
                ((1,), 0, 1, True), ((1,), 0, 1, float('nan'))]:
            with self.assertRaises(ValueError):
                energy.certify_independent_polynomial_energy((0, coefficient), x0, t0, t1, degree=1, **parameters(1))
        for signature in ((True,), (1, 0), (4,), (-1,), (0, 0, 0, 0)):
            with self.assertRaises(ValueError):
                energy.certify_independent_polynomial_energy({signature: 1}, (1, 1, 1), 0, 1, **parameters(3))
        for precision in (True, 0, 31, 32.5):
            with self.assertRaises(ValueError):
                energy.certify_independent_polynomial_energy((0, 1), (1,), 0, 1, degree=1,
                    precision=precision, **parameters(1))


@unittest.skipUnless(FLINT, 'Outward checks require the separate python-flint environment')
class OutwardCertificateChecks(unittest.TestCase):
    def dense_moment_reference(self, coeffs, initial, t0, t1, model, degree=3, precision=256):
        from flint import arb, ctx, fmpq
        def ball(value):
            value = Fraction(value)
            return arb(fmpq(value.numerator, value.denominator))
        def psi(signature):
            counts = Counter(signature)
            return sum((n*(Fraction(model['rate'])-Fraction(model['dividends'][i]))
                +Fraction(n*(n-1), 2)*Fraction(model['volatilities'][i])**2
                for i, n in counts.items()), Fraction(0))
        signatures = energy.complete_monomial_indices(len(initial), degree)
        with ctx.workprec(precision):
            reference = arb(0)
            for i, first in enumerate(signatures):
                if not first or not coeffs[i]:
                    continue
                for j, second in enumerate(signatures):
                    if not second or not coeffs[j]:
                        continue
                    combined = tuple(sorted(first+second))
                    prefactor = Fraction(coeffs[i])*Fraction(coeffs[j])
                    for coordinate in combined:
                        prefactor *= Fraction(initial[coordinate])
                    if not prefactor:
                        continue
                    # A dense direct moment difference, with independent
                    # moment exponents, rather than either factor backend.
                    reference += ball(prefactor)*(ball(t1*psi(combined)).exp()
                        -ball(t0*psi(combined)+(t1-t0)*(psi(first)+psi(second))).exp())
            lo, hi = reference.lower().fmpq(), reference.upper().fmpq()
            return Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q))

    def assert_contains(self, record, reference):
        lo, hi = energy.decode_interval(record)
        self.assertGreaterEqual(lo, 0)
        self.assertLessEqual(lo, reference[0])
        self.assertGreaterEqual(hi, reference[1])

    def test_outward_scalar_C_and_norm_match_independent_exact_values(self):
        from flint import arb, ctx, fmpq
        def ball(value):
            value = Fraction(value)
            return arb(fmpq(value.numerator, value.denominator))
        def endpoints(value):
            lo, hi = value.lower().fmpq(), value.upper().fmpq()
            return Fraction(int(lo.p), int(lo.q)), Fraction(int(hi.p), int(hi.q))
        with ctx.workprec(128):
            mean = Fraction(2, 3)
            # v=log(3/2) is outward, so it cannot be supplied as an exact
            # variance to the API; here test C directly with the log ball.
            q = Fraction(3, 2)
            v = ball(q).log()
            matrix = energy._lognormal_coefficients(ball(mean), v, arb, lambda x: x if hasattr(x, 'exp') else ball(x))
            for n, row in enumerate(coefficients(mean, q)):
                for k, expected in enumerate(row):
                    lo, hi = endpoints(matrix[n][k])
                    self.assertLessEqual(lo, expected)
                    self.assertGreaterEqual(hi, expected)
            # Norm calculation uses exact v in production; independent
            # 256-bit exp moments in the energy tests verify that path.

    def test_dense_256bit_moments_contained_for_signed_heterogeneous_cubic_coefficients(self):
        for dimension, degree, initial, t0, t1, changes in [
                (1, 3, (Fraction(9, 10),), Fraction(1, 2), Fraction(3, 2), {}),
                (2, 3, (Fraction(9, 10), Fraction(11, 10)), Fraction(3, 4), Fraction(7, 4), {}),
                (3, 3, (Fraction(9, 10), 1, Fraction(11, 10)), 0, 1, {}),
                (3, 3, (0, 1, Fraction(11, 10)), Fraction(1, 2), Fraction(3, 2), {}),
                (3, 3, (Fraction(9, 10), 1, Fraction(11, 10)), Fraction(1, 2), Fraction(3, 2), {'volatilities': (0, Fraction(1, 5), Fraction(3, 10))}),
                (4, 2, (1, Fraction(4, 5), Fraction(6, 5), 1), Fraction(1, 3), Fraction(4, 3), {}),
                (2, 1, (1, 2), Fraction(1, 2), Fraction(3, 2), {})]:
            model = parameters(dimension, **changes)
            coeffs = tuple(Fraction((i%7)-3, 11) for i in range(len(energy.complete_monomial_indices(dimension, degree))))
            with self.subTest(dimension=dimension, degree=degree, initial=initial, changes=changes):
                result = energy.certify_independent_polynomial_energy(coeffs, initial, t0, t1, degree=degree, **model)
                reference = self.dense_moment_reference(coeffs, initial, t0, t1, model, degree)
                self.assert_contains(result['energy_certificate'], reference)
                self.assertEqual(result['work']['monomial_pairs_evaluated'], 0)
                self.assertTrue(result['population_energy_arithmetic_certified'])
                self.assertFalse(result['sampling_coverage_certified'])
                self.assertFalse(result['exact_optimiser_certified'])

    def test_sparse_and_complete_coefficient_inputs_have_same_certificate(self):
        dimension = 3
        signatures = energy.complete_monomial_indices(dimension)
        mapping = {(): Fraction(9, 8), (0,): -2, (1, 2): Fraction(7, 5), (0, 1, 2): Fraction(-1, 7)}
        complete = tuple(mapping.get(s, 0) for s in signatures)
        first = energy.certify_independent_polynomial_energy(complete, (1, 2, 3), Fraction(1, 2), Fraction(3, 2), **parameters(dimension))
        second = energy.certify_independent_polynomial_energy(mapping, (1, 2, 3), Fraction(1, 2), Fraction(3, 2), **parameters(dimension))
        self.assertEqual(first['energy_certificate'], second['energy_certificate'])
        self.assertEqual(first['work'], second['work'])
        self.assertFalse(second['complete_dictionary'])

    def test_constants_zero_initial_states_zero_time_and_zero_volatility_are_exact(self):
        for coeffs, initial, t0, t1, changes, degree in [
                ((9,), (1,), 0, 1, {}, 0),
                ((1, 2, 3, 4), (1,), 0, 0, {}, 3),
                ((1, 2, 3, 4), (1,), 0, 1, {'volatilities': 0}, 3),
                ((1, 2, 3, 4), (0,), 0, 1, {}, 3),
                ({(0, 1, 2): 1}, (0, 1, 2), 0, 1, {}, 3),
                ({(0,): 1}, (1, 1, 1), 0, 1, {'volatilities': (0, Fraction(1, 5), Fraction(1, 2))}, 3)]:
            dimension = len(initial)
            result = energy.certify_independent_polynomial_energy(coeffs, initial, t0, t1, degree=degree, **parameters(dimension, **changes))
            self.assertEqual(energy.decode_interval(result['energy_certificate']), (0, 0))
            self.assertEqual(energy.decode_interval(result['square_root_energy_certificate']), (0, 0))
            self.assertTrue(result['exact_zero_martingale_residual'])

    def test_tiny_positive_energy_and_norms_do_not_become_zero(self):
        result = energy.certify_independent_polynomial_energy((0, 1), (1,), 0, 1, degree=1,
            rate=0, dividends=1000, volatilities=Fraction(1, 5))
        lo, hi = energy.decode_interval(result['energy_certificate'])
        self.assertGreater(lo, 0)
        self.assertGreater(hi, 0)
        self.assertEqual(result['energy_certificate']['upper']['approximation'], 0.)
        self.assertGreater(energy.decode_interval(result['square_root_energy_certificate'])[0], 0)
        tiny_variance = energy.certify_independent_polynomial_energy((0, 1), (1,), 0, 1,
            degree=1, rate=0, dividends=0, volatilities=Fraction(1, 10**250))
        self.assertGreater(energy.decode_interval(tiny_variance['energy_certificate'])[0], 0)
        self.assertFalse(tiny_variance['exact_zero_martingale_residual'])

    def test_uncertain_row_crossing_zero_keeps_sound_positive_energy_bounds(self):
        # At tiny variance the alpha=1 row of R**2-2R has an amplitude
        # smaller than its 128-bit ball radius. Its square must start at zero.
        model = parameters(1, rate=0, dividends=(0,), volatilities=(Fraction(1, 10**100),))
        crossing = []
        square = energy._square_endpoints
        def observed(lower, upper):
            crossing.append(lower <= 0 <= upper and lower < upper)
            return square(lower, upper)
        with patch.object(energy, '_square_endpoints', side_effect=observed):
            result = energy.certify_independent_polynomial_energy((0, -2, 1), (1,), 0, 1, degree=2, **model)
        self.assertTrue(any(crossing))
        reference = self.dense_moment_reference((0, -2, 1), (1,), 0, 1, model, degree=2, precision=2048)
        self.assert_contains(result['energy_certificate'], reference)
        self.assertGreater(energy.decode_interval(result['energy_certificate'])[0], 0)

    def test_unsupported_covariance_is_rejected_even_for_zero_time(self):
        for model in [parameters(3, correlation=Fraction(1, 2)),
                parameters(3, correlation=((1, 0, 0), (0, 1, Fraction(1, 4)), (0, Fraction(1, 4), 1))),
                parameters(3, correlation=((2, 0, 0), (0, 1, 0), (0, 0, 1)))]:
            with self.assertRaises(ValueError):
                energy.certify_independent_polynomial_energy({(0,): 1}, (1, 1, 1), 0, 0, **model)

    def test_law_dictionary_validated_law_and_dense_identity_correlation(self):
        model = parameters(2)
        model['correlation'] = ((1, 0), (0, 1))
        coeffs = (0, 1, -2)
        expected = energy.certify_independent_polynomial_energy(coeffs, (1, 2), 0, 1, degree=1, **model)
        dictionary = energy.certify_independent_polynomial_energy(coeffs, (1, 2), 0, 1, model, degree=1)
        law = energy._core.validate_moment_law(2, 1, **model)
        validated = energy.certify_independent_polynomial_energy(coeffs, (1, 2), 0, 1, law, degree=1)
        self.assertEqual(expected['energy_certificate'], dictionary['energy_certificate'])
        self.assertEqual(expected['energy_certificate'], validated['energy_certificate'])
        for bad in [dict(model, step=1), dict(rate=0), energy._core.validate_moment_law(2, 2, **model)]:
            with self.assertRaises(ValueError):
                energy.certify_independent_polynomial_energy(coeffs, (1, 2), 0, 1, bad, degree=1)
        with self.assertRaises(ValueError):
            energy.certify_independent_polynomial_energy(coeffs, (1, 2), 0, 1, model, degree=1, rate=0)

    def test_records_combine_in_unchanged_core_and_sqrt_endpoints_are_sound(self):
        first = energy.certify_independent_polynomial_energy((0, 1), (1,), 0, 1, degree=1, **parameters(1))
        second = energy.certify_independent_polynomial_energy((0, -2), (1,), 1, 2, degree=1, **parameters(1))
        combined = energy._core.combine_energy_certificates((first, second))
        lo1, hi1 = energy.decode_interval(first['energy_certificate'])
        lo2, hi2 = energy.decode_interval(second['energy_certificate'])
        self.assertEqual(energy.decode_interval(combined['terminal_energy_certificate']), (lo1+lo2, hi1+hi2))
        for result in (first, second):
            lo, hi = energy.decode_interval(result['energy_certificate'])
            root_lo, root_hi = energy.decode_interval(result['square_root_energy_certificate'])
            self.assertLessEqual(root_lo**2, lo)
            self.assertGreaterEqual(root_hi**2, hi)
            self.assertEqual(result['author'], 'Amanjeet Singh')


if __name__ == '__main__':
    unittest.main(verbosity=2)
