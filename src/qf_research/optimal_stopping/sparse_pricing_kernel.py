"""Compiled conditional polynomial prices and deltas for one frozen function.

I compare an orbit-plus-labelled representation with its exact expanded
ordinary polynomial. Both use the actual GBM law and identical supplied
states. Dense and compact kernels cache coefficient and growth factors at
one declared precision. Query timing excludes certificate transcription and
independent verification. Derivatives are with respect to normalised stock
coordinates, including at zero, with no division by a stock value.
"""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys

if __package__:
    from . import martingale_compression as core
else:
    _path = Path(__file__).with_name('martingale_compression.py')
    _spec = importlib.util.spec_from_file_location('qf_sparse_kernel_core', _path)
    core = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = core
    _spec.loader.exec_module(core)

symmetry = core._symmetry


@dataclass(frozen=True)
class KernelDescription:
    law: object
    degree: int
    orbit_coefficients: tuple
    labelled_terms: tuple
    expanded_terms: tuple
    compression: bool
    monomial_count: int


@dataclass(frozen=True)
class CompiledKernel:
    description: KernelDescription
    representation: str
    precision: int
    terms: tuple
    orbit_coefficients: tuple
    first_growth: tuple
    second_growth: tuple
    third_growth: tuple
    orbit_factors: tuple


def prepare_kernel(candidate, dimension, step, *, rate, dividends,
        volatilities, correlation=0, degree=3):
    """Check exact expansion before compiling either representation."""
    if not isinstance(candidate, dict) or type(candidate.get('compression')) is not bool:
        raise ValueError('A candidate with an explicit compression flag is required')
    signatures = symmetry.complete_monomial_indices(dimension, degree)
    law = symmetry.validate_moment_law(dimension, step, rate=rate,
        dividends=dividends, volatilities=volatilities, correlation=correlation)
    orbit = tuple(core.decode_rational(x) for x in candidate['orbit_coefficients'])
    dense = tuple(core.decode_rational(x) for x in candidate['expanded_coefficients'])
    if len(orbit) != 7 or len(dense) != len(signatures):
        raise ValueError('Candidate coefficients must match the declared dictionary')
    present = {symmetry.monomial_orbit_index(s) for s in signatures}
    if any(coefficient and index not in present for index, coefficient in enumerate(orbit)):
        raise ValueError('An absent orbit cannot have a non-zero coefficient')
    seen, labelled, expanded = set(), [], [orbit[symmetry.monomial_orbit_index(s)] for s in signatures]
    for record in candidate['labelled_coefficients']:
        if not isinstance(record, dict) or set(record) != {'monomial_index', 'coefficient'}:
            raise ValueError('A labelled monomial record is required')
        index = record['monomial_index']
        if type(index) is not int or not 0 < index < len(signatures) or index in seen:
            raise ValueError('Labelled indices must be distinct nonconstant dictionary positions')
        coefficient = core.decode_rational(record['coefficient'])
        seen.add(index)
        expanded[index] += coefficient
        if coefficient:
            labelled.append((signatures[index], coefficient))
    if tuple(expanded) != dense:
        raise ValueError('The compact candidate must expand to the identical polynomial')
    return KernelDescription(law, degree, orbit, tuple(labelled),
        tuple((s, c) for s, c in zip(signatures, dense) if c),
        candidate['compression'], len(signatures))


def _flint(precision):
    if type(precision) is not int or precision < 32:
        raise ValueError('Precision must be an integer of at least 32 bits')
    from flint import arb, ctx, fmpq
    def ball(x):
        return arb(fmpq(x.numerator, x.denominator))
    return arb, ctx, ball


def compile_kernel(description, representation, *, precision=128):
    """Cache exact-input ball factors once, charging this setup separately."""
    if not isinstance(description, KernelDescription) or representation not in ('dense', 'compact'):
        raise ValueError('A checked description and dense or compact representation are required')
    arb, ctx, ball = _flint(precision)
    actual = 'dense' if representation == 'dense' or not description.compression else 'compact'
    law = description.law
    with ctx.workprec(precision):
        coefficients = description.expanded_terms if actual == 'dense' else description.labelled_terms
        terms = tuple((signature, ball(c)*ball(law.step*core._psi(signature, law)).exp(),
            tuple((i, n, signature[:signature.index(i)]+signature[signature.index(i)+1:])
                for i, n in Counter(signature).items()))
            for signature, c in coefficients)
        if actual == 'dense':
            orbit = first = second = third = factors = ()
        else:
            orbit = tuple(ball(c) for c in description.orbit_coefficients)
            first = tuple(ball((law.rate-q)*law.step).exp() for q in law.dividends)
            if law.family == 'common_volatility_exchangeable':
                second = third = ()
                v = law.volatilities[0]**2*law.step
                factors = tuple(ball(k*v).exp() for k in
                    (Fraction(0), Fraction(0), Fraction(1), law.correlation,
                     Fraction(3), 1+2*law.correlation, 3*law.correlation))
            else:
                second = tuple(ball((2*(law.rate-q)+sigma*sigma)*law.step).exp()
                    for q, sigma in zip(law.dividends, law.volatilities))
                third = tuple(ball((3*(law.rate-q)+3*sigma*sigma)*law.step).exp()
                    for q, sigma in zip(law.dividends, law.volatilities))
                factors = tuple(arb(1) for _ in range(7))
        return CompiledKernel(description, actual, precision, terms, orbit, first, second, third, factors)


def _state(kernel, state, ball):
    exact = symmetry._vector(state, 'State', nonnegative=True)
    if len(exact) != kernel.description.law.dimension:
        raise ValueError('State dimension must match the compiled model')
    return tuple(ball(x) for x in exact)


def _add_terms(terms, state, value, gradients, arb):
    products = {(): arb(1)}
    def product(signature):
        if signature not in products:
            products[signature] = product(signature[:-1])*state[signature[-1]]
        return products[signature]
    for signature, coefficient, derivatives in terms:
        value += coefficient*product(signature)
        for index, exponent, reduced in derivatives:
            gradients[index] += coefficient*exponent*product(reduced)
    return value, tuple(gradients)


def evaluate_kernel(kernel, state):
    """Return an outward value and all coordinate derivatives for one query."""
    if not isinstance(kernel, CompiledKernel):
        raise ValueError('A compiled kernel is required')
    arb, ctx, ball = _flint(kernel.precision)
    with ctx.workprec(kernel.precision):
        x = _state(kernel, state, ball)
        d = len(x)
        if kernel.representation == 'dense':
            return _add_terms(kernel.terms, x, arb(0), [arb(0) for _ in x], arb)
        if not any(kernel.description.orbit_coefficients[1:]):
            return _add_terms(kernel.terms, x, kernel.orbit_coefficients[0], [arb(0) for _ in x], arb)
        first = tuple(a*g for a, g in zip(x, kernel.first_growth))
        common = kernel.description.law.family == 'common_volatility_exchangeable'
        if common:
            second = tuple(a*a for a in first)
            third = tuple(a*a*a for a in first)
        else:
            second = tuple(a*a*g for a, g in zip(x, kernel.second_growth))
            third = tuple(a*a*a*g for a, g in zip(x, kernel.third_growth))
        s1 = s2 = s11 = s3 = s21 = s111 = arb(0)
        for a, b, c in zip(first, second, third):
            s111 = s111+a*s11
            s21 = s21+b*s1+a*s2
            s11 = s11+a*s1
            s1, s2, s3 = s1+a, s2+b, s3+c
        features = (arb(1), s1, s2, s11, s3, s21, s111)
        value = sum((c*f*a for c, f, a in zip(kernel.orbit_coefficients,
            kernel.orbit_factors, features)), arb(0))
        p1, p2, p11 = [[arb(0) for _ in range(d+1)] for _ in range(3)]
        r1, r2, r11 = [[arb(0) for _ in range(d+1)] for _ in range(3)]
        for i, (a, b) in enumerate(zip(first, second)):
            p11[i+1], p1[i+1], p2[i+1] = p11[i]+a*p1[i], p1[i]+a, p2[i]+b
        for i in range(d-1, -1, -1):
            a, b = first[i], second[i]
            r11[i], r1[i], r2[i] = r11[i+1]+a*r1[i+1], r1[i+1]+a, r2[i+1]+b
        gradients = []
        for i, current in enumerate(x):
            other1, other2 = p1[i]+r1[i+1], p2[i]+r2[i+1]
            other11 = p11[i]+r11[i+1]+p1[i]*r1[i+1]
            if common:
                a = first[i]
                derivatives = tuple(kernel.first_growth[i]*f for f in
                    (arb(0), arb(1), 2*a, other1, 3*a*a, 2*a*other1+other2, other11))
            else:
                g1, g2, g3 = kernel.first_growth[i], kernel.second_growth[i], kernel.third_growth[i]
                derivatives = (arb(0), g1, 2*current*g2, g1*other1, 3*current*current*g3,
                    2*current*g2*other1+g1*other2, g1*other11)
            gradients.append(sum((c*f*a for c, f, a in zip(kernel.orbit_coefficients,
                kernel.orbit_factors, derivatives)), arb(0)))
        return _add_terms(kernel.terms, x, value, gradients, arb)


def independent_reference(description, state, *, precision=256):
    """Rebuild moments from exponent multiplicities, without cached factors."""
    arb, ctx, ball = _flint(precision)
    exact = symmetry._vector(state, 'State', nonnegative=True)
    law = description.law
    if len(exact) != law.dimension:
        raise ValueError('State dimension must match the model')
    with ctx.workprec(precision):
        x = tuple(ball(a) for a in exact)
        value, gradients = arb(0), [arb(0) for _ in x]
        for signature, coefficient in description.expanded_terms:
            exponents = Counter(signature)
            drift = sum((n*(law.rate-law.dividends[i]-law.volatilities[i]**2/2)
                for i, n in exponents.items()), Fraction(0))
            variance = sum((a*b*law.volatilities[i]*law.volatilities[j]*
                (1 if i == j else law.correlation)
                for i, a in exponents.items() for j, b in exponents.items()), Fraction(0))/2
            weight = ball(coefficient)*ball(law.step*(drift+variance)).exp()
            term = arb(1)
            for i, n in exponents.items():
                term *= x[i]**n
            value += weight*term
            for i, n in exponents.items():
                derivative = arb(n)
                for j, m in exponents.items():
                    derivative *= x[j]**(m-(i == j))
                gradients[i] += weight*derivative
        return value, tuple(gradients)


def result_records(result, precision):
    _, ctx, _ = _flint(precision)
    value, gradients = result
    with ctx.workprec(precision):
        return tuple(symmetry._arb_interval_record(x) for x in (value, *gradients))


def compare_query(description, state, *, precision=128, reference_precision=256,
        width_target=Fraction(1, 10**20)):
    """Check common interval intersections and a declared absolute width."""
    width_target = symmetry._exact(width_target, 'Absolute interval width target')
    if width_target <= 0 or reference_precision <= precision:
        raise ValueError('A positive width and strictly higher reference precision are required')
    records = []
    for representation in ('dense', 'compact'):
        kernel = compile_kernel(description, representation, precision=precision)
        records.append(result_records(evaluate_kernel(kernel, state), precision))
    records.append(result_records(independent_reference(description, state,
        precision=reference_precision), reference_precision))
    widths, intersections = [], []
    for components in zip(*records):
        intervals = [core.decode_interval(x) for x in components]
        lower, upper = max(x[0] for x in intervals), min(x[1] for x in intervals)
        if lower > upper:
            raise ArithmeticError('Compiled representations and independent reference disagree')
        if any(b-a > width_target for a, b in intervals):
            raise ArithmeticError('A declared absolute enclosure width was exceeded')
        widths.append([core.rational_record(b-a) for a, b in intervals])
        intersections.append(core.interval_record(lower, upper))
    return dict(schema_version=1, author='Amanjeet Singh', dimension=description.law.dimension,
        representations=['dense', 'compact' if description.compression else 'dense_identity', 'independent_reference'],
        precision_bits=precision, reference_precision_bits=reference_precision,
        width_target=core.rational_record(width_target), component_widths=widths,
        common_intersections=intersections, passed=True, same_polynomial_verified=True,
        derivative_coordinates='Normalised stock coordinates; apply the declared stock scale and discount factor for model hedge holdings.')
