"""Actual-law increment projection and outward dual-error certificates.

I project frozen ordinary monomial functions into seven orbit sums using the
martingale-increment seminorm. Increment projection is an established method;
see Alfonsi, Kebaier and Lelong (2025), Mathematical Finance,
doi:10.1111/mafi.12460. The selected
floating SVD coefficients are frozen as exact binary rationals. Their actual
residual energy is then evaluated outwards, without claiming exact optimality.

The absolute difference between expected dual gaps is at most the square root
of the terminal residual energy. This uses the established one-sided maximum
inequality of Dubins and Schwarz (1988), Astérisque, 157–158, pp. 129–145:
https://www.numdam.org/item/AST_1988__157-158__129_0.pdf . It is not a pathwise
error bound, a sampling certificate or a new maximal inequality. Both pricing
martingales must use the actual-law conditional centres and the same policy.

Dense Gram construction is quadratic in the complete monomial count. It is
intended for small retained functions; linear orbit evaluation does not make
dense high-dimensional preprocessing linear. No production sampling occurs.
"""
from dataclasses import dataclass
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import sys
import time

if __package__:
    from . import symmetry as _symmetry
else:
    # Direct source loading avoids the unchanged package's NumPy imports in
    # the separate outward-arithmetic environment.
    _spec = importlib.util.spec_from_file_location('qf_compression_symmetry', Path(__file__).with_name('symmetry.py'))
    if _spec is None or _spec.loader is None:
        raise RuntimeError('The sibling symmetry source is unavailable')
    _symmetry = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _symmetry
    _spec.loader.exec_module(_symmetry)

ORBIT_NAMES = _symmetry.ORBIT_NAMES
_exact, _values, _vector = _symmetry._exact, _symmetry._values, _symmetry._vector
complete_monomial_indices = _symmetry.complete_monomial_indices
monomial_orbit_index = _symmetry.monomial_orbit_index
project_coefficients = _symmetry.project_coefficients
validate_moment_law = _symmetry.validate_moment_law


DEFAULT_MAX_GRAM_ENTRIES = 4_000_000
ORBIT_DEGREES = (0, 1, 2, 2, 3, 3, 3)


def _clocks():
    return time.process_time(), time.perf_counter()


def _elapsed(start):
    return dict(cpu_seconds=time.process_time()-start[0], wall_seconds=time.perf_counter()-start[1])


def rational_record(value):
    value = _exact(value, 'Rational value')
    try:
        approximate = float(value)
    except OverflowError:
        approximate = None
    if approximate is not None and not math.isfinite(approximate):
        approximate = None
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=approximate)


def decode_rational(record):
    if not isinstance(record, dict) or set(record) != {'numerator', 'denominator', 'approximation'}:
        raise ValueError('An exact rational record is required')
    if not isinstance(record['numerator'], str) or not isinstance(record['denominator'], str):
        raise ValueError('Rational numerator and denominator must be integer strings')
    numerator, denominator = int(record['numerator']), int(record['denominator'])
    if denominator <= 0:
        raise ValueError('An exact denominator must be positive')
    return Fraction(numerator, denominator)


def interval_record(lower, upper):
    lower, upper = _exact(lower, 'Lower endpoint'), _exact(upper, 'Upper endpoint')
    if lower > upper:
        raise ArithmeticError('Interval endpoints cannot be reversed')
    return dict(lower=rational_record(lower), upper=rational_record(upper), width=rational_record(upper-lower))


def decode_interval(record):
    if not isinstance(record, dict) or set(record) != {'lower', 'upper', 'width'}:
        raise ValueError('An exact endpoint interval is required')
    lower, upper = decode_rational(record['lower']), decode_rational(record['upper'])
    if lower > upper or decode_rational(record['width']) != upper-lower:
        raise ValueError('Interval order and width must agree')
    return lower, upper


def _setup(initial_state, t0, t1, *, rate, dividends, volatilities, correlation, degree):
    initial = _vector(initial_state, 'Initial scaled state', nonnegative=True)
    t0, t1 = _exact(t0, 'Previous time'), _exact(t1, 'Current time')
    if t0 < 0 or t1 < t0:
        raise ValueError('Ordered non-negative calendar times are required')
    signatures = complete_monomial_indices(len(initial), degree)
    law = validate_moment_law(len(initial), t1-t0, rate=rate, dividends=dividends,
        volatilities=volatilities, correlation=correlation)
    return initial, t0, t1, law, signatures


def _coefficients(values, count, name):
    values = _values(values, name)
    if len(values) != count:
        raise ValueError(name+' must match the declared basis')
    return tuple(_exact(value, name) for value in values)


def _covariance(law, i, j):
    return law.volatilities[i]*law.volatilities[j]*(1 if i == j else law.correlation)


def _psi(signature, law):
    # A signature repeats each coordinate according to its exponent.
    drift = sum((law.rate-law.dividends[i]-law.volatilities[i]**2/2 for i in signature), Fraction(0))
    variance = sum((_covariance(law, i, j) for i in signature for j in signature), Fraction(0))/2
    return drift+variance


def _cross(first, second, law):
    return sum((_covariance(law, i, j) for i in first for j in second), Fraction(0))


def _initial_monomial(initial, signature):
    return math.prod((initial[i] for i in signature), start=Fraction(1))


def _model_record(initial, law):
    return dict(dimension=law.dimension, initial_scaled_state=[rational_record(x) for x in initial],
        rate=rational_record(law.rate), dividends=[rational_record(x) for x in law.dividends],
        volatilities=[rational_record(x) for x in law.volatilities],
        correlation=rational_record(law.correlation), family=law.family)


def _finite_float(value, name):
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ArithmeticError(name+' is not representable as a finite float') from error
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ArithmeticError(name+' is not a finite non-underflowing float')
    return result


def _resource_guard(count, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('The Gram-entry budget must be a positive integer')
    if count*count > maximum:
        raise ValueError('The complete dense Gram exceeds its explicit entry budget; use a reviewed sparse or aggregated moment implementation')


def _nominal_gram(initial, t0, law, signatures, maximum):
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError('The floating projection requires NumPy in the research environment') from error
    count = len(signatures)
    _resource_guard(count, maximum)
    matrix = np.zeros((count, count), dtype=float)
    psi = [_psi(signature, law) for signature in signatures]
    initial_values = [_initial_monomial(initial, signature) for signature in signatures]
    evaluations = 0
    for i, first in enumerate(signatures):
        for j in range(i, count):
            second = signatures[j]
            cross = _cross(first, second, law)
            prefactor = initial_values[i]*initial_values[j]
            if law.step == 0 or cross == 0 or prefactor == 0:
                continue
            exponent = t0*_psi(tuple(sorted(first+second)), law)+law.step*(psi[i]+psi[j])
            try:
                value = _finite_float(prefactor, 'Initial monomial product')*math.exp(_finite_float(exponent, 'Moment exponent'))*math.expm1(_finite_float(law.step*cross, 'Covariance exponent'))
            except OverflowError as error:
                raise ArithmeticError('A nominal increment moment overflows') from error
            if not math.isfinite(value) or value == 0:
                raise ArithmeticError('A non-zero nominal increment Gram entry is non-finite or underflows')
            matrix[i, j] = matrix[j, i] = value
            evaluations += 1
    return matrix, evaluations


@dataclass(frozen=True)
class EnergyProjection:
    coefficients: tuple[Fraction, ...]
    diagnostics: dict

    def record(self):
        return dict(schema_version=1, author='Amanjeet Singh',
            orbit_names=list(ORBIT_NAMES), frozen_orbit_coefficients=[rational_record(x) for x in self.coefficients],
            diagnostics=self.diagnostics,
            scope='Finite binary coefficients selected by a nominal actual-law increment projection. No exact optimiser, residual-energy or pricing-error certificate is asserted by this floating solve.')


def fit_energy_projection(coefficients, initial_state, t0, t1, *, rate, dividends,
        volatilities, correlation=0, degree=3, rcond=1e-12,
        max_gram_entries=DEFAULT_MAX_GRAM_ENTRIES):
    """Select seven frozen binary coefficients using an equilibrated SVD.

    The constant is inherited because it is in the increment kernel. Absent
    and nominal zero-energy orbits retain their averaged coefficients. Rank,
    conditioning and objective diagnostics are floating quantities only. The
    input Gram is never altered by clipping negative eigenvalues to zero.
    """
    start = _clocks()
    initial, t0, t1, law, signatures = _setup(initial_state, t0, t1,
        rate=rate, dividends=dividends, volatilities=volatilities, correlation=correlation, degree=degree)
    coefficients = _coefficients(coefficients, len(signatures), 'Retained coefficients')
    rcond = _finite_float(_exact(rcond, 'SVD relative cutoff'), 'SVD relative cutoff')
    if not 0 < rcond < 1:
        raise ValueError('The SVD relative cutoff must lie strictly between zero and one')
    import numpy as np
    matrix, evaluations = _nominal_gram(initial, t0, law, signatures, max_gram_entries)
    categories = [monomial_orbit_index(signature) for signature in signatures]
    buckets = [[i for i, category in enumerate(categories) if category == orbit] for orbit in range(7)]
    present = [orbit for orbit in range(1, 7) if buckets[orbit]]
    gram = np.asarray([[math.fsum(float(matrix[i, j]) for i in buckets[left] for j in buckets[right])
        for right in present] for left in present], dtype=float).reshape(len(present), len(present))
    numeric = [_finite_float(value, 'Retained coefficient') for value in coefficients]
    rhs = np.asarray([math.fsum(float(matrix[i, j])*numeric[j] for i in buckets[orbit] for j in range(len(signatures)))
        for orbit in present], dtype=float)
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise ArithmeticError('The nominal small projection system is non-finite')
    diagonal = np.diag(gram)
    if np.any(diagonal < 0):
        raise ArithmeticError('A nominal orbit-energy diagonal is negative; the Gram has not been projected onto a PSD cone')
    active_positions = [i for i, value in enumerate(diagonal) if value > 0]
    averaging = project_coefficients(coefficients, len(initial), degree)
    selected = [_finite_float(value, 'Averaging coefficient') for value in averaging]
    rank, condition, minimum_eigenvalue = 0, None, None
    singular_values, scales = [], []
    if active_positions:
        active = np.ix_(active_positions, active_positions)
        scales_array = np.sqrt(diagonal[active_positions])
        equilibrated = gram[active]/scales_array[:, None]/scales_array[None, :]
        right = rhs[active_positions]/scales_array
        if not np.isfinite(equilibrated).all() or not np.isfinite(right).all():
            raise ArithmeticError('The equilibrated nominal system is non-finite')
        u, singular, vt = np.linalg.svd(equilibrated, full_matrices=False)
        minimum_eigenvalue = float(np.linalg.eigvalsh(equilibrated)[0])
        if minimum_eigenvalue < -100*np.finfo(float).eps*max(1., float(singular[0])):
            raise ArithmeticError('The nominal small Gram has a substantial negative eigenvalue')
        keep = singular > rcond*singular[0]
        rank = int(np.count_nonzero(keep))
        solution = vt[keep].T@((u[:, keep].T@right)/singular[keep])
        solution /= scales_array
        for position, value in zip(active_positions, solution):
            selected[present[position]] = float(value)
        singular_values = [float(value) for value in singular]
        scales = [float(value) for value in scales_array]
        condition = float(singular[0]/singular[keep][-1]) if rank else None
    if any(not math.isfinite(value) for value in selected):
        raise ArithmeticError('A selected coefficient is non-finite')
    frozen = tuple(Fraction(value) for value in selected)
    residual = [_finite_float(value-frozen[category], 'Residual coefficient') for value, category in zip(coefficients, categories)]
    averaging_residual = [_finite_float(value-averaging[category], 'Averaging residual') for value, category in zip(coefficients, categories)]
    def nominal_energy(values):
        try:
            answer = math.fsum(values[i]*float(matrix[i, j])*values[j] for i in range(len(values)) for j in range(len(values)))
        except (OverflowError, ValueError) as error:
            raise ArithmeticError('A nominal residual-energy diagnostic overflows') from error
        if not math.isfinite(answer):
            raise ArithmeticError('A nominal residual-energy diagnostic is non-finite')
        return answer
    diagnostics = dict(model=_model_record(initial, law), previous_time=rational_record(t0), current_time=rational_record(t1),
        degree=degree, monomial_count=len(signatures), present_orbit_indices=present,
        active_orbit_indices=[present[i] for i in active_positions], nominal_rank=rank,
        rcond=rcond, singular_values=singular_values, equilibration_scales=scales,
        nominal_condition_number=condition, nominal_minimum_eigenvalue=minimum_eigenvalue,
        nominal_residual_energy=nominal_energy(residual), nominal_averaging_energy=nominal_energy(averaging_residual),
        nominal_increment_gram_evaluations=evaluations, numpy_version=np.__version__,
        exact_optimiser_certified=False, negative_eigenvalues_clipped=False,
        zero_energy_orbits_inherited=True, cost=_elapsed(start))
    return EnergyProjection(frozen, diagnostics)


def _outward_tools(precision):
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')
    try:
        from flint import arb, ctx, fmpq, __version__ as flint_version
    except ImportError as error:
        raise RuntimeError('Residual certificates require the separate python-flint environment') from error
    def ball(value):
        return arb(fmpq(value.numerator, value.denominator))
    def endpoints(value):
        if not value.is_finite():
            raise ArithmeticError('A non-finite ball cannot provide an energy certificate')
        lower, upper = value.lower().fmpq(), value.upper().fmpq()
        return Fraction(int(lower.p), int(lower.q)), Fraction(int(upper.p), int(upper.q))
    return arb, ctx, ball, endpoints, flint_version


def certify_residual_energy(coefficients, orbit_coefficients, initial_state, t0, t1,
        *, rate, dividends, volatilities, correlation=0, degree=3, precision=128,
        max_gram_entries=DEFAULT_MAX_GRAM_ENTRIES):
    """Enclose the residual increment energy for exact supplied coefficients.

    Signed off-diagonal Gram contributions are retained. Only the final lower
    energy endpoint may be intersected with known non-negativity; a negative
    upper endpoint is an error. A zero residual or deterministic step returns
    exactly zero without treating floating cancellation as a proof.
    """
    start = _clocks()
    initial, t0, t1, law, signatures = _setup(initial_state, t0, t1,
        rate=rate, dividends=dividends, volatilities=volatilities, correlation=correlation, degree=degree)
    coefficients = _coefficients(coefficients, len(signatures), 'Retained coefficients')
    compressed = _coefficients(orbit_coefficients, 7, 'Orbit coefficients')
    if any(value and order > degree for value, order in zip(compressed, ORBIT_DEGREES)):
        raise ValueError('Compressed coefficients cannot introduce an orbit above the declared polynomial degree')
    categories = [monomial_orbit_index(signature) for signature in signatures]
    residual = tuple(value-compressed[category] for value, category in zip(coefficients, categories))
    arb, ctx, ball, endpoints, flint_version = _outward_tools(precision)
    active = [i for i, value in enumerate(residual) if value and signatures[i]]
    deterministic = law.step == 0 or not any(law.volatilities)
    pairs = exponentials = expm1_calls = exact_zero_pairs = 0
    if deterministic or not active:
        lower = upper = Fraction(0)
    else:
        _resource_guard(len(active), max_gram_entries)
        psi = {i: _psi(signatures[i], law) for i in active}
        initial_values = {i: _initial_monomial(initial, signatures[i]) for i in active}
        with ctx.workprec(precision):
            energy = arb(0)
            for position, i in enumerate(active):
                first = signatures[i]
                for j in active[position:]:
                    second = signatures[j]
                    cross = _cross(first, second, law)
                    prefactor = initial_values[i]*initial_values[j]
                    pairs += 1
                    if cross == 0 or prefactor == 0:
                        exact_zero_pairs += 1
                        continue
                    exponent = t0*_psi(tuple(sorted(first+second)), law)+law.step*(psi[i]+psi[j])
                    moment = ball(prefactor)*ball(exponent).exp()*ball(law.step*cross).expm1()
                    multiplier = residual[i]*residual[j]*(1 if i == j else 2)
                    energy += ball(multiplier)*moment
                    exponentials += 1
                    expm1_calls += 1
            lower, upper = endpoints(energy)
        if upper < 0:
            raise ArithmeticError('The outward energy upper endpoint contradicts the PSD increment form')
        lower = max(Fraction(0), lower)
    with ctx.workprec(precision):
        sqrt_lower = endpoints(ball(lower).sqrt())[0]
        sqrt_upper = endpoints(ball(upper).sqrt())[1]
    return dict(schema_version=1, author='Amanjeet Singh',
        model=_model_record(initial, law), previous_time=rational_record(t0), current_time=rational_record(t1),
        degree=degree, monomial_count=len(signatures), orbit_names=list(ORBIT_NAMES),
        retained_coefficients=[rational_record(value) for value in coefficients],
        compressed_orbit_coefficients=[rational_record(value) for value in compressed],
        residual_coefficients=[rational_record(value) for value in residual],
        energy_certificate=interval_record(lower, upper),
        square_root_energy_certificate=interval_record(sqrt_lower, sqrt_upper),
        single_increment_expected_dual_gap_difference_upper=rational_record(sqrt_upper),
        precision_bits=precision, python_flint_version=flint_version, deterministic_increment=deterministic,
        exact_zero_martingale_residual=upper == 0, work=dict(monomial_pairs_evaluated=pairs,
            exact_zero_pairs=exact_zero_pairs, arb_exponentials=exponentials, arb_expm1=expm1_calls),
        cost=_elapsed(start), actual_law_centres_required=True,
        population_energy_arithmetic_certified=True, exact_optimiser_certified=False,
        sampling_coverage_certified=False, simulator_law_certified=False,
        scope='Outward population increment energy for frozen ordinary monomial coefficients and actual GBM centres. The square-root expression is an upper bound on the absolute difference of expected dual gaps for the same adapted policy, not a pathwise or sampling error bound. For multiple dates, combine their disjoint increment energies before taking the square root.')


def combine_energy_certificates(records, *, precision=128):
    """Add ordered disjoint increment energies before bounding expected prices."""
    start = _clocks()
    records = _values(records, 'Energy certificates')
    if not records:
        raise ValueError('At least one increment certificate is required')
    model = records[0]['model']
    lower = upper = Fraction(0)
    previous_end = None
    for record in records:
        if record['model'] != model or record['population_energy_arithmetic_certified'] is not True or record['actual_law_centres_required'] is not True:
            raise ValueError('All increments require the same actual model and population energy premise')
        t0, t1 = decode_rational(record['previous_time']), decode_rational(record['current_time'])
        if t0 < 0 or t1 < t0 or (previous_end is not None and t0 < previous_end):
            raise ValueError('Increment intervals must be ordered and disjoint')
        lo, hi = decode_interval(record['energy_certificate'])
        if lo < 0:
            raise ValueError('An increment energy certificate must be non-negative')
        lower += lo
        upper += hi
        previous_end = t1
    arb, ctx, ball, endpoints, flint_version = _outward_tools(precision)
    with ctx.workprec(precision):
        sqrt_lower = endpoints(ball(lower).sqrt())[0]
        sqrt_upper = endpoints(ball(upper).sqrt())[1]
    return dict(schema_version=1, author='Amanjeet Singh', model=model, increments=len(records),
        terminal_energy_certificate=interval_record(lower, upper),
        square_root_energy_certificate=interval_record(sqrt_lower, sqrt_upper),
        absolute_expected_dual_gap_difference_upper=rational_record(sqrt_upper),
        maximal_inequality_constant=1,
        maximal_inequality_source='Dubins and Schwarz (1988), Astérisque, 157–158, pp. 129–145',
        precision_bits=precision, python_flint_version=flint_version, cost=_elapsed(start),
        policy_unchanged_required=True, actual_law_centres_required=True,
        bound_is_difference_of_expectations=True, bound_is_pathwise=False,
        sampling_coverage_certified=False, simulator_law_certified=False,
        scope='The actual-law residual increments must be disjoint square-integrable martingale increments on one filtration. Their terminal energy is the sum of the supplied population energies. The result bounds the absolute difference of expected dual upper prices and, for one unchanged adapted policy, expected dual gaps. No exercise-policy improvement or sampling guarantee is asserted.')
