"""Same-function value and delta checks with independent monomial references."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sparse_pricing_kernel_fixture',
    ROOT/'src/qf_research/optimal_stopping/sparse_pricing_kernel.py')
kernel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kernel
SPEC.loader.exec_module(kernel)
FLINT = importlib.util.find_spec('flint') is not None


def candidate(dimension, degree=3, *, identity=False, labels=True):
    signatures = kernel.symmetry.complete_monomial_indices(dimension, degree)
    present = {kernel.symmetry.monomial_orbit_index(s) for s in signatures}
    orbit = tuple(Fraction((-1)**j*(j+1), 7) if j in present else Fraction(0) for j in range(7))
    additions = [(i, Fraction((-1)**i*(i+2), 13)) for i in (1, len(signatures)-1)
        if labels and 0 < i < len(signatures)]
    additions = list(dict(additions).items())
    expanded = [orbit[kernel.symmetry.monomial_orbit_index(s)] for s in signatures]
    for i, c in additions:
        expanded[i] += c
    if identity:
        orbit = (expanded[0],)+(Fraction(0),)*6
        additions = [(i, c) for i, c in enumerate(expanded) if i and c]
    return dict(compression=not identity,
        orbit_coefficients=[kernel.core.rational_record(c) for c in orbit],
        labelled_coefficients=[dict(monomial_index=i, coefficient=kernel.core.rational_record(c)) for i, c in additions],
        expanded_coefficients=[kernel.core.rational_record(c) for c in expanded])


def parameters(dimension, *, heterogeneous=False, correlation=0):
    return dict(rate=Fraction(1, 20), dividends=tuple(Fraction(i+1, 30) for i in range(dimension)),
        volatilities=tuple(Fraction(i+1, 10) if heterogeneous else Fraction(1, 5) for i in range(dimension)),
        correlation=correlation)


class InputChecks(unittest.TestCase):
    def test_tampered_expansion_rejected_before_compilation(self):
        value = candidate(3)
        value['expanded_coefficients'][4] = kernel.core.rational_record(99)
        with self.assertRaisesRegex(ValueError, 'identical polynomial'):
            kernel.prepare_kernel(value, 3, Fraction(1, 3), **parameters(3))

    def test_duplicate_constant_and_out_of_range_labels_rejected(self):
        for position in (0, -1, 100, True):
            value = candidate(3)
            value['labelled_coefficients'][0]['monomial_index'] = position
            with self.subTest(position=position), self.assertRaises(ValueError):
                kernel.prepare_kernel(value, 3, 1, **parameters(3))
        value = candidate(3)
        value['labelled_coefficients'].append(deepcopy(value['labelled_coefficients'][0]))
        with self.assertRaises(ValueError):
            kernel.prepare_kernel(value, 3, 1, **parameters(3))

    def test_absent_orbits_cannot_be_silently_ignored(self):
        for dimension, degree, orbit_index in ((1, 3, 3), (5, 1, 6)):
            value = candidate(dimension, degree)
            value['orbit_coefficients'][orbit_index] = kernel.core.rational_record(1)
            with self.subTest(dimension=dimension, degree=degree), self.assertRaises(ValueError):
                kernel.prepare_kernel(value, dimension, 1, degree=degree, **parameters(dimension))

    def test_model_and_candidate_domain_validation(self):
        for value in (None, {}, dict(candidate(3), compression=1)):
            with self.assertRaises(ValueError):
                kernel.prepare_kernel(value, 3, 1, **parameters(3))
        with self.assertRaises(ValueError):
            kernel.prepare_kernel(candidate(3), 3, -1, **parameters(3))
        with self.assertRaises(ValueError):
            kernel.prepare_kernel(candidate(3), 3, 1, **parameters(3, heterogeneous=True, correlation=Fraction(1, 3)))


@unittest.skipUnless(FLINT, 'Requires the separate outward-arithmetic environment')
class OutwardQueryChecks(unittest.TestCase):
    def test_values_and_all_deltas_across_supported_covariance_families(self):
        for dimension in (1, 2, 5, 8):
            models = [parameters(dimension, heterogeneous=True), parameters(dimension),
                parameters(dimension, correlation=1)]
            if dimension > 1:
                models.append(parameters(dimension, correlation=-Fraction(1, dimension-1)))
            for model in models:
                description = kernel.prepare_kernel(candidate(dimension), dimension, Fraction(1, 3), **model)
                state = tuple(Fraction(i+2, 3) for i in range(dimension))
                with self.subTest(dimension=dimension, model=model):
                    result = kernel.compare_query(description, state)
                    self.assertTrue(result['passed'])
                    self.assertEqual(len(result['common_intersections']), dimension+1)

    def test_zero_stock_coordinates_and_zero_time_are_finite_without_division(self):
        for state in ((0, 1, 2), (0, 0, 0)):
            for step in (0, Fraction(2, 7)):
                description = kernel.prepare_kernel(candidate(3), 3, step, **parameters(3, heterogeneous=True))
                with self.subTest(state=state, step=step):
                    self.assertTrue(kernel.compare_query(description, state)['passed'])

    def test_zero_volatility_and_signed_carries(self):
        model = dict(rate=-Fraction(1, 10), dividends=(Fraction(0), -Fraction(1, 5)),
            volatilities=(Fraction(0), Fraction(0)), correlation=1)
        description = kernel.prepare_kernel(candidate(2), 2, Fraction(2, 3), **model)
        self.assertTrue(kernel.compare_query(description, (Fraction(2, 3), Fraction(5, 4)))['passed'])

    def test_lower_degrees_and_constant_only_function(self):
        for degree in (0, 1, 2, 3):
            description = kernel.prepare_kernel(candidate(3, degree), 3, 1, degree=degree, **parameters(3))
            self.assertTrue(kernel.compare_query(description, (1, 2, 3))['passed'])

    def test_identity_fallback_uses_identical_dense_evaluation(self):
        description = kernel.prepare_kernel(candidate(3, identity=True), 3, Fraction(1, 4), **parameters(3))
        compact = kernel.compile_kernel(description, 'compact')
        dense = kernel.compile_kernel(description, 'dense')
        self.assertEqual(compact.representation, 'dense')
        self.assertEqual(kernel.result_records(kernel.evaluate_kernel(compact, (0, 2, 3)), 128),
            kernel.result_records(kernel.evaluate_kernel(dense, (0, 2, 3)), 128))

    def test_zero_time_exact_value_and_derivatives_by_rational_algebra(self):
        from collections import Counter
        description = kernel.prepare_kernel(candidate(3), 3, 0, **parameters(3))
        state = tuple(map(Fraction, (0, 2, 3)))
        value, gradient = Fraction(0), [Fraction(0)]*3
        for signature, coefficient in description.expanded_terms:
            powers = Counter(signature)
            term = coefficient
            for i, power in powers.items():
                term *= state[i]**power
            value += term
            for i, power in powers.items():
                derivative = coefficient*power
                for j, count in powers.items():
                    derivative *= state[j]**(count-(i == j))
                gradient[i] += derivative
        result = kernel.compare_query(description, state)
        for interval, exact in zip(result['common_intersections'], (value, *gradient)):
            lower, upper = kernel.core.decode_interval(interval)
            self.assertLessEqual(lower, exact)
            self.assertGreaterEqual(upper, exact)

    def test_orbit_kernel_matches_unchanged_seven_orbit_implementation(self):
        for heterogeneous in (False, True):
            model = parameters(5, heterogeneous=heterogeneous)
            description = kernel.prepare_kernel(candidate(5, labels=False), 5, Fraction(1, 3), **model)
            state = (Fraction(0), Fraction(2, 3), Fraction(5, 7), Fraction(4, 3), Fraction(7, 4))
            compiled = kernel.compile_kernel(description, 'compact')
            first = kernel.result_records(kernel.evaluate_kernel(compiled, state), 128)
            other = kernel.symmetry.certify_conditional_polynomial(description.orbit_coefficients,
                state, Fraction(1, 3), precision=256, **model)
            for a, b in zip(first, (other['value_certificate'], *other['gradient_certificates'])):
                al, au = kernel.core.decode_interval(a)
                bl, bu = kernel.core.decode_interval(b)
                self.assertLessEqual(max(al, bl), min(au, bu))

    def test_width_failure_is_not_relaxed_to_pass(self):
        description = kernel.prepare_kernel(candidate(3), 3, Fraction(1, 3), **parameters(3))
        with self.assertRaisesRegex(ArithmeticError, 'width'):
            kernel.compare_query(description, (1, 2, 3), precision=32)

    def test_unsupported_states_precision_and_modes_fail(self):
        description = kernel.prepare_kernel(candidate(3), 3, 1, **parameters(3))
        compiled = kernel.compile_kernel(description, 'dense')
        for state in ((1, 2), (-1, 2, 3), (True, 2, 3), (float('nan'), 2, 3)):
            with self.assertRaises(ValueError):
                kernel.evaluate_kernel(compiled, state)
        for precision in (31, True):
            with self.assertRaises(ValueError):
                kernel.compile_kernel(description, 'dense', precision=precision)
        with self.assertRaises(ValueError):
            kernel.compile_kernel(description, 'unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
