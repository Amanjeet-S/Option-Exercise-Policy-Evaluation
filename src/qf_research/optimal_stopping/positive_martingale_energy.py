"""Positive sparse population energy for independent GBM polynomials.

Author: Amanjeet Singh.

The finite ordinary stock polynomial has total degree at most three. Its
actual-law martingale increment energy is evaluated as a sum of non-negative
weighted squares using finite lognormal orthogonal polynomials. This is a
finite-dimensional identity; no completeness claim in lognormal L2 is made.
There are at most nineteen stored factor entries per non-zero coefficient.
No dense monomial Gram matrix is formed. See positive_martingale_energy_proof.md.

Only independent GBM coordinates are supported. Input binary floats are
frozen as exact rational values. python-flint is loaded only when an outward
certificate is requested. The original package and numerical sources need
not be imported to load this private revision draft directly.
"""
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from itertools import combinations_with_replacement, product
import importlib.util
import math
import os
from pathlib import Path
import sys


if __package__:
    from . import martingale_compression as _core
else:
    _here = Path(__file__).resolve()
    _candidates = [_here.with_name('martingale_compression.py')]
    if os.environ.get('QF_TEST_REPOSITORY'):
        _candidates.append(Path(os.environ['QF_TEST_REPOSITORY'])/'src/qf_research/optimal_stopping/martingale_compression.py')
    _candidates.extend(parent/'src/qf_research/optimal_stopping/martingale_compression.py' for parent in _here.parents)
    _source = next((path for path in _candidates if path.is_file()), None)
    if _source is None:
        raise RuntimeError('The unchanged martingale compression source is unavailable')
    _spec = importlib.util.spec_from_file_location('qf_positive_energy_core', _source)
    if _spec is None or _spec.loader is None:
        raise RuntimeError('The unchanged martingale compression source cannot be loaded')
    _core = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _core
    _spec.loader.exec_module(_core)


DEFAULT_MAX_FACTOR_ENTRIES = 4_000_000
_exact = _core._exact
rational_record = _core.rational_record
interval_record = _core.interval_record
decode_rational = _core.decode_rational
decode_interval = _core.decode_interval
complete_monomial_indices = _core.complete_monomial_indices


def _signature(value, dimension, degree):
    value = _core._values(value, 'Monomial signature')
    if len(value) > degree or any(type(i) is not int or i < 0 or i >= dimension for i in value) or tuple(sorted(value)) != value:
        raise ValueError('A monomial signature must be ordered, within the state dimension and declared degree')
    return value


def _factor_patterns(signature):
    """Yield sparse alpha, gamma and local (coordinate,n,a,g) factors.

    Each tuple stores at most three coordinate powers; no scan over the full
    dimension occurs for an entry. Alpha is non-zero and alpha+gamma<=beta.
    """
    support = tuple(Counter(signature).items())
    for powers in product(*(range(n+1) for _, n in support)):
        if not any(powers):
            continue
        alpha = tuple(i for (i, _), a in zip(support, powers) for _ in range(a))
        for gamma_powers in product(*(range(n-a+1) for (_, n), a in zip(support, powers))):
            gamma = tuple(i for (i, _), g in zip(support, gamma_powers) for _ in range(g))
            local = tuple((i, n, a, g) for (i, n), a, g in zip(support, powers, gamma_powers))
            yield alpha, gamma, local


def _entry_count(signature):
    # Includes all non-zero alpha; there are no factor entries for constants.
    support = tuple(Counter(signature).items())
    return sum(math.prod(n-a+1 for (_, n), a in zip(support, powers))
        for powers in product(*(range(n+1) for _, n in support)) if any(powers))


def factor_structure(signatures, *, dimension, degree=3,
        max_factor_entries=DEFAULT_MAX_FACTOR_ENTRIES):
    """Build the structural sparse factor for an explicitly supplied dictionary.

    The guard runs before row and entry allocation. Dictionary monomials are
    unique, ordinary signatures. Rows are (alpha,gamma); entries additionally
    give a dictionary column and its at-most-three local scalar factors.
    No coefficient values or law-dependent arithmetic is performed here.
    """
    dimension = _core._symmetry._integer(dimension, 'Dimension', 1)
    degree = _core._symmetry._integer(degree, 'Degree', 0)
    if degree > 3:
        raise ValueError('Only degrees zero to three are supported')
    if type(max_factor_entries) is not int or max_factor_entries < 1:
        raise ValueError('The factor-entry budget must be a positive integer')
    signatures = tuple(_signature(signature, dimension, degree) for signature in _core._values(signatures, 'Monomial dictionary'))
    if len(set(signatures)) != len(signatures):
        raise ValueError('A monomial dictionary cannot contain duplicate signatures')
    entries = sum(_entry_count(signature) for signature in signatures)
    if entries > max_factor_entries:
        raise ValueError('The sparse positive factor exceeds its explicit factor-entry budget')
    rows = {}
    for column, signature in enumerate(signatures):
        for alpha, gamma, local in _factor_patterns(signature):
            rows.setdefault((alpha, gamma), []).append((column, local))
    return dict(signatures=signatures, rows=rows, factor_entries=entries,
        factor_rows=len(rows), maximum_entries_per_column=max((_entry_count(s) for s in signatures), default=0))


def _setup(initial_state, t0, t1, law, *, rate, dividends,
        volatilities, correlation, degree):
    initial = _core._vector(initial_state, 'Initial scaled state', nonnegative=True)
    t0, t1 = _exact(t0, 'Previous time'), _exact(t1, 'Current time')
    if t0 < 0 or t1 < t0:
        raise ValueError('Ordered non-negative calendar times are required')
    degree = _core._symmetry._integer(degree, 'Degree', 0)
    if degree > 3:
        raise ValueError('Only degrees zero to three are supported')
    if law is not None:
        if any(value is not None for value in (rate, dividends, volatilities, correlation)):
            raise ValueError('Supply either a law or model keyword parameters, without conflicts')
        if isinstance(law, Mapping):
            if set(law) - {'rate', 'dividends', 'volatilities', 'correlation'}:
                raise ValueError('A law dictionary accepts rate, dividends, volatilities and correlation only')
            if not {'rate', 'dividends', 'volatilities'} <= set(law):
                raise ValueError('A law dictionary requires rate, dividends and volatilities')
            kwargs = dict(law)
        elif isinstance(law, _core._symmetry.MomentLaw) or all(hasattr(law, name) for name in ('dimension', 'step', 'rate', 'dividends', 'volatilities', 'correlation')):
            if type(law.dimension) is not int or law.dimension != len(initial) or _exact(law.step, 'Law step') != t1-t0:
                raise ValueError('The supplied law dimension and step must match the state and times')
            kwargs = dict(rate=law.rate, dividends=law.dividends, volatilities=law.volatilities, correlation=law.correlation)
        else:
            raise ValueError('Supply a validated moment law or a model parameter dictionary')
    else:
        if any(value is None for value in (rate, dividends, volatilities)):
            raise ValueError('Rate, dividends and volatilities are required')
        kwargs = dict(rate=rate, dividends=dividends, volatilities=volatilities,
            correlation=0 if correlation is None else correlation)
    actual = _core.validate_moment_law(len(initial), t1-t0, **kwargs)
    # Independence is required even for deterministic data: unsupported
    # covariance requests are never silently accepted by an exact-zero branch.
    if actual.correlation != 0:
        raise ValueError('The positive polynomial factor supports independent GBM coordinates only (correlation=0)')
    return initial, t0, t1, actual, degree


def _coefficient_dictionary(coefficients, dimension, degree, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('The factor-entry budget must be a positive integer')
    full_count = math.comb(dimension+degree, degree)
    active = []
    dictionary_count = planned_entries = 0
    if isinstance(coefficients, Mapping):
        iterator = coefficients.items()
        complete = False
    else:
        values = _core._values(coefficients, 'Polynomial coefficients')
        if len(values) != full_count:
            raise ValueError('Coefficients must match the complete declared basis or be a sparse signature mapping')
        signatures = (signature for order in range(degree+1)
            for signature in combinations_with_replacement(range(dimension), order))
        iterator = zip(signatures, values)
        complete = True
    for signature, coefficient in iterator:
        signature = _signature(signature, dimension, degree)
        coefficient = _exact(coefficient, 'Polynomial coefficient')
        dictionary_count += 1
        if coefficient and signature:
            planned_entries += _entry_count(signature)
            if planned_entries > maximum:
                raise ValueError('The sparse positive factor exceeds its explicit factor-entry budget')
            active.append((signature, coefficient))
    if isinstance(coefficients, Mapping):
        # Mapping keys are unique, but tuple conversion must not merge keys.
        if dictionary_count != len({_signature(s, dimension, degree) for s in coefficients}):
            raise ValueError('A monomial dictionary cannot contain duplicate signatures')
        complete = dictionary_count == full_count
    return tuple(active), dictionary_count, full_count, complete, planned_entries


def _lognormal_coefficients(mean, variance, arb, ball):
    """C[n][k] in Y**n=sum_k C[n][k]P_k, n<=3; no divisions."""
    q = ball(variance).exp()
    one = arb(1)
    zero = arb(0)
    y2, q2 = mean*mean, q*q
    geometric = q2+q+one
    return ((one, zero, zero, zero),
        (mean, one, zero, zero),
        (y2*q, mean*q*(q+one), one, zero),
        (y2*mean*q2*q, y2*q2*q*geometric, mean*q2*geometric, one))


def _lognormal_norm(order, mean, variance, arb, ball):
    """h_k=y**(2k) exp(3k(k-1)v/2) prod_l expm1(l*v)."""
    if order == 0:
        return arb(1)
    if variance == 0 or mean.is_zero():
        return arb(0)
    value = mean**(2*order)*ball(Fraction(3*order*(order-1), 2)*variance).exp()
    for ell in range(1, order+1):
        value *= ball(ell*variance).expm1()
    return value


def _square_endpoints(lower, upper):
    """Exact endpoint square; a crossing interval has lower square zero."""
    if lower > upper:
        raise ArithmeticError('Square interval endpoints cannot be reversed')
    return (Fraction(0) if lower <= 0 <= upper else min(lower*lower, upper*upper),
        max(lower*lower, upper*upper))


def _positive_endpoints(value, endpoints):
    lower, upper = endpoints(value)
    if upper < 0:
        raise ArithmeticError('A non-negative factor weight has a negative outward upper endpoint')
    return max(Fraction(0), lower), upper


def certify_independent_polynomial_energy(coefficients, initial_state, t0, t1, law=None,
        *, rate=None, dividends=None, volatilities=None, correlation=None,
        degree=3, precision=128, max_factor_entries=DEFAULT_MAX_FACTOR_ENTRIES):
    """Enclose E[(f(X1)-E[f(X1)|X0])**2] by positive sparse factors.

    Complete coefficient sequences use the unchanged ordinary monomial order.
    A mapping from ordered repeated-coordinate signatures to coefficients is
    also accepted, with absent monomials equal to zero. The explicit entry
    guard runs before factor rows are allocated. Records share model, time,
    energy and premise fields with the unchanged core energy combiner.
    """
    start = _core._clocks()
    initial, t0, t1, actual, degree = _setup(initial_state, t0, t1, law,
        rate=rate, dividends=dividends, volatilities=volatilities,
        correlation=correlation, degree=degree)
    active, dictionary_count, full_count, complete, planned_entries = _coefficient_dictionary(
        coefficients, len(initial), degree, max_factor_entries)
    arb, ctx, ball, endpoints, flint_version = _core._outward_tools(precision)
    deterministic = actual.step == 0 or not any(actual.volatilities)
    evaluated_rows = zero_weight_rows = factor_rows = factor_entries = 0
    lower = upper = Fraction(0)
    if not deterministic and active:
        structure = factor_structure(tuple(signature for signature, _ in active),
            dimension=len(initial), degree=degree, max_factor_entries=max_factor_entries)
        factor_rows, factor_entries = structure['factor_rows'], structure['factor_entries']
        with ctx.workprec(precision):
            y_return, y_previous, v_return, v_previous = [], [], [], []
            for i in range(len(initial)):
                carry = actual.rate-actual.dividends[i]
                y_return.append(ball(carry*actual.step).exp())
                y_previous.append(arb(0) if initial[i] == 0 else ball(initial[i])*ball(carry*t0).exp())
                v_return.append(actual.volatilities[i]**2*actual.step)
                v_previous.append(actual.volatilities[i]**2*t0)
            return_c, previous_c, return_h, previous_h, previous_m = {}, {}, {}, {}, {}
            def rcoeff(i, n, a):
                if i not in return_c:
                    return_c[i] = _lognormal_coefficients(y_return[i], v_return[i], arb, ball)
                return return_c[i][n][a]
            def pcoeff(i, a, n, g):
                key = i, a
                if key not in previous_c:
                    tilted = y_previous[i]*ball(2*a*v_previous[i]).exp()
                    previous_c[key] = _lognormal_coefficients(tilted, v_previous[i], arb, ball)
                return previous_c[key][n][g]
            def weight(alpha, gamma):
                value = arb(1)
                a_counts, g_counts = Counter(alpha), Counter(gamma)
                for i, a in a_counts.items():
                    if v_return[i] == 0 or initial[i] == 0:
                        return arb(0)
                    if (i, a) not in return_h:
                        return_h[i, a] = _lognormal_norm(a, y_return[i], v_return[i], arb, ball)
                        previous_m[i, a] = y_previous[i]**(2*a)*ball(a*(2*a-1)*v_previous[i]).exp()
                    value *= return_h[i, a]*previous_m[i, a]
                for i, g in g_counts.items():
                    a = a_counts.get(i, 0)
                    if v_previous[i] == 0 or initial[i] == 0:
                        return arb(0)
                    if (i, a, g) not in previous_h:
                        tilted = y_previous[i]*ball(2*a*v_previous[i]).exp()
                        previous_h[i, a, g] = _lognormal_norm(g, tilted, v_previous[i], arb, ball)
                    value *= previous_h[i, a, g]
                return value
            for (alpha, gamma), entries in structure['rows'].items():
                w = weight(alpha, gamma)
                if w.is_zero():
                    zero_weight_rows += 1
                    continue
                w_lower, w_upper = _positive_endpoints(w, endpoints)
                amplitude = arb(0)
                for column, local in entries:
                    factor = ball(active[column][1])
                    for i, n, a, g in local:
                        factor *= rcoeff(i, n, a)*pcoeff(i, a, n-a, g)
                    amplitude += factor
                amplitude_lower, amplitude_upper = endpoints(amplitude)
                square_lower, square_upper = _square_endpoints(amplitude_lower, amplitude_upper)
                # Endpoints here are exact rational numbers. Positive product
                # and summation require no further rounding or cancellation.
                lower += w_lower*square_lower
                upper += w_upper*square_upper
                evaluated_rows += 1
    with ctx.workprec(precision):
        sqrt_lower = max(Fraction(0), endpoints(ball(lower).sqrt())[0])
        sqrt_upper = endpoints(ball(upper).sqrt())[1]
    return dict(schema_version=1, author='Amanjeet Singh',
        backend='positive_sparse_independent_polynomial',
        model=_core._model_record(initial, actual), previous_time=rational_record(t0), current_time=rational_record(t1),
        degree=degree, monomial_count=full_count, dictionary_count=dictionary_count,
        complete_dictionary=complete,
        nonzero_nonconstant_coefficients=[dict(signature=list(s), coefficient=rational_record(c)) for s, c in active],
        energy_certificate=interval_record(lower, upper),
        square_root_energy_certificate=interval_record(sqrt_lower, sqrt_upper),
        single_increment_expected_dual_gap_difference_upper=rational_record(sqrt_upper),
        precision_bits=precision, python_flint_version=flint_version,
        deterministic_increment=deterministic, exact_zero_martingale_residual=upper == 0,
        work=dict(planned_factor_entries=planned_entries, factor_entries=factor_entries,
            factor_rows=factor_rows, evaluated_factor_rows=evaluated_rows,
            exact_zero_weight_rows=zero_weight_rows, monomial_pairs_evaluated=0,
            maximum_entries_per_nonzero_coefficient=19, max_factor_entries=max_factor_entries),
        cost=_core._elapsed(start), actual_law_centres_required=True,
        population_energy_arithmetic_certified=True, exact_optimiser_certified=False,
        sampling_coverage_certified=False, simulator_law_certified=False,
        scope='Outward population increment energy for finite ordinary stock polynomials of total degree at most three under independent GBM and actual-law conditional centres. Positive sparse finite polynomial factors are used; completeness in lognormal L2 is not asserted. The square-root bound concerns the absolute difference of expected dual gaps for one unchanged adapted policy. For disjoint increments add their energies before taking the square root. Fixed-degree preprocessing is O(d)+O(m) arithmetic operations for a complete m=C(d+degree,degree) coefficient vector; bit complexity and coefficient storage still depend on m and precision.')
