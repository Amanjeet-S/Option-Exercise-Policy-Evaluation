"""Budgeted frozen martingale compression with independent outward acceptance.

I use established greedy orthogonal least squares in the nominal increment
seminorm. Six nonconstant orbit columns form the initial span; labelled
monomial atoms refine it. The ranking is a floating heuristic, with no global
optimality or new matching-pursuit claim. Every coefficient is then frozen as
an exact binary rational. Outward certification of the full residual is the
acceptance authority. The exact original polynomial is always available.

The certificate inherits the actual-law, same-policy expected dual-price
interpretation of martingale_compression. It does not certify sampling or a
simulator. Dense preprocessing has a declared quadratic entry budget. Linear input,
basis and identity records are still materialised on a resource fallback.
"""
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

if __package__:
    from . import martingale_compression as core
else:
    _source = Path(__file__).with_name('martingale_compression.py')
    if not _source.is_file():
        _source = next((parent/'src/qf_research/optimal_stopping/martingale_compression.py'
            for parent in Path(__file__).resolve().parents
            if (parent/'src/qf_research/optimal_stopping/martingale_compression.py').is_file()), None)
    if _source is None:
        raise RuntimeError('The unchanged martingale-compression source is unavailable')
    _spec = importlib.util.spec_from_file_location('qf_adaptive_compression_core', _source)
    if _spec is None or _spec.loader is None:
        raise RuntimeError('The unchanged martingale-compression source is unavailable')
    core = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = core
    _spec.loader.exec_module(core)

DEFAULT_MAX_GRAM_ENTRIES = core.DEFAULT_MAX_GRAM_ENTRIES
ALGORITHM = 'greedy orthogonal least squares after an orbit-span SVD'


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=True, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ValueError('A finite canonical JSON record is required') from error


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def compression_path_sha256(path):
    """Hash the complete path, excluding only its own hash field."""
    if not isinstance(path, dict):
        raise ValueError('A compression path dictionary is required')
    return _digest({key: value for key, value in path.items() if key != 'path_sha256'})


def _source_identity():
    return {name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
        for name, source in [('adaptive_martingale_compression.py', __file__),
            ('martingale_compression.py', core.__file__), ('symmetry.py', core._symmetry.__file__)]}


def _law_parameters(law, rate, dividends, volatilities, correlation):
    if law is None:
        if rate is None or dividends is None or volatilities is None:
            raise ValueError('Supply a supported law or rate, dividends and volatilities')
        return dict(rate=rate, dividends=dividends, volatilities=volatilities, correlation=correlation)
    if rate is not None or dividends is not None or volatilities is not None or correlation != 0:
        raise ValueError('Supply the law once, without competing parameter keywords')
    if isinstance(law, core._symmetry.MomentLaw):
        return dict(rate=law.rate, dividends=law.dividends,
            volatilities=law.volatilities, correlation=law.correlation)
    if not isinstance(law, Mapping) or not {'rate', 'dividends', 'volatilities'} <= set(law):
        raise ValueError('A law requires rate, dividends and volatilities')
    if set(law)-{'rate', 'dividends', 'volatilities', 'correlation'}:
        raise ValueError('Unknown law parameters are unsupported')
    return dict(rate=law['rate'], dividends=law['dividends'],
        volatilities=law['volatilities'], correlation=law.get('correlation', 0))


def _validated(coefficients, initial_state, t0, t1, degree, parameters, supplied_law):
    initial = core._vector(initial_state, 'Initial scaled state', nonnegative=True)
    if type(degree) is not int or not 0 <= degree <= 3:
        raise ValueError('The polynomial degree must be an integer from zero to three')
    # Reject arity without first allocating a combinatorial signature tuple.
    count = math.comb(len(initial)+degree, degree)
    coefficients = core._coefficients(coefficients, count, 'Teacher coefficients')
    initial, t0, t1, law, signatures = core._setup(initial, t0, t1, degree=degree, **parameters)
    if isinstance(supplied_law, core._symmetry.MomentLaw):
        if supplied_law != law:
            raise ValueError('The supplied moment law must match dimension and time step')
    return coefficients, initial, t0, t1, law, signatures


def _candidate(orbit, labelled, signatures, *, identity=False, diagnostics=None):
    categories = [core.monomial_orbit_index(signature) for signature in signatures]
    expanded = [orbit[category] for category in categories]
    for index, value in labelled:
        expanded[index] += value
    return dict(kind='identity' if identity else 'refinement', compression=not identity,
        orbit_coefficients=[core.rational_record(value) for value in orbit],
        labelled_coefficients=[dict(monomial_index=index, coefficient=core.rational_record(value))
            for index, value in labelled],
        expanded_coefficients=[core.rational_record(value) for value in expanded],
        structural_coefficient_count=sum(value != 0 for value in orbit)+sum(value != 0 for _, value in labelled),
        diagnostics={} if diagnostics is None else diagnostics)


def _identity(coefficients, signatures):
    return _candidate((coefficients[0],)+(Fraction(0),)*6,
        [(index, value) for index, value in enumerate(coefficients) if index and value],
        signatures, identity=True, diagnostics=dict(exact_original_polynomial=True))


class NominalCompressionError(ArithmeticError):
    """A floating solve could not safely propose another frozen candidate."""


def _solve_span(matrix, numeric, columns, rcond):
    import numpy as np
    small = columns.T @ matrix @ columns
    right = columns.T @ matrix @ numeric
    if not np.isfinite(small).all() or not np.isfinite(right).all():
        raise NominalCompressionError('The nominal small Gram or right side is non-finite')
    diagonal = np.diag(small)
    if np.any(diagonal < 0):
        raise NominalCompressionError('The nominal small Gram has a negative energy diagonal')
    active = np.flatnonzero(diagonal > 0)
    weights = np.zeros(columns.shape[1], dtype=float)
    inverse = np.zeros_like(small)
    singular_values, scales = [], []
    rank = 0
    minimum_eigenvalue = condition = None
    if active.size:
        scales_array = np.sqrt(diagonal[active])
        system = small[np.ix_(active, active)]/scales_array[:, None]/scales_array[None, :]
        target = right[active]/scales_array
        if not np.isfinite(system).all() or not np.isfinite(target).all():
            raise NominalCompressionError('The equilibrated nominal system is non-finite')
        u, singular, vt = np.linalg.svd(system, full_matrices=False)
        minimum_eigenvalue = float(np.linalg.eigvalsh(system)[0])
        if minimum_eigenvalue < -100*np.finfo(float).eps*max(1., float(singular[0])):
            raise NominalCompressionError('The nominal small Gram has a substantial negative eigenvalue')
        keep = singular > rcond*singular[0]
        rank = int(np.count_nonzero(keep))
        active_inverse = (vt[keep].T/singular[keep]) @ u[:, keep].T
        active_inverse /= scales_array[:, None]*scales_array[None, :]
        inverse[np.ix_(active, active)] = active_inverse
        weights[active] = active_inverse @ right[active]
        singular_values = [float(value) for value in singular]
        scales = [float(value) for value in scales_array]
        condition = float(singular[0]/singular[keep][-1]) if rank else None
    if not np.isfinite(weights).all() or not np.isfinite(inverse).all():
        raise NominalCompressionError('The selected nominal coefficients are non-finite')
    return weights, inverse, dict(nominal_rank=rank, singular_values=singular_values,
        equilibration_scales=scales, nominal_condition_number=condition,
        nominal_minimum_eigenvalue=minimum_eigenvalue, negative_eigenvalues_clipped=False)


def prepare_compression_path(coefficients, initial_state, t0, t1, *, law=None,
        rate=None, dividends=None, volatilities=None, correlation=0, degree=3,
        rcond=1e-12, max_gram_entries=DEFAULT_MAX_GRAM_ENTRIES):
    """Propose a finite orbit-plus-labelled hierarchy and append exact identity.

    Input-domain errors raise. Valid-domain resource, numerical and dependency
    failures are recorded and leave exact identity available. Nominal atom
    gains q_m**2/s_m describe greedy orthogonal least-squares refinement only.
    With a changing SVD rank cutoff they remain approximate heuristic gains.
    """
    start = core._clocks()
    parameters = _law_parameters(law, rate, dividends, volatilities, correlation)
    coefficients, initial, t0, t1, actual, signatures = _validated(
        coefficients, initial_state, t0, t1, degree, parameters, law)
    rcond = core._finite_float(core._exact(rcond, 'SVD relative cutoff'), 'SVD relative cutoff')
    if not 0 < rcond < 1:
        raise ValueError('The SVD relative cutoff must lie strictly between zero and one')
    if type(max_gram_entries) is not int or max_gram_entries < 1:
        raise ValueError('The Gram-entry budget must be a positive integer')
    identity = _identity(coefficients, signatures)
    path = dict(schema_version=1, author='Amanjeet Singh', algorithm=ALGORITHM,
        degree=degree, monomial_count=len(signatures), monomial_signatures=[list(x) for x in signatures],
        teacher_coefficients=[core.rational_record(x) for x in coefficients],
        model=core._model_record(initial, actual), previous_time=core.rational_record(t0),
        current_time=core.rational_record(t1), source_identity=_source_identity(),
        candidates=[], preparation=dict(rcond=rcond, max_gram_entries=max_gram_entries,
            exact_optimiser_certified=False, global_optimality_claimed=False, production_rng_used=False))
    path['teacher_sha256'] = _digest(path['teacher_coefficients'])
    preparation = path['preparation']
    if len(signatures)**2 > max_gram_entries:
        preparation['failure'] = dict(type='resource_limit', message='The complete dense Gram exceeds its explicit entry budget')
    else:
        try:
            import numpy as np
            matrix, evaluations = core._nominal_gram(initial, t0, actual, signatures, max_gram_entries)
            preparation.update(numpy_version=np.__version__, nominal_increment_gram_evaluations=evaluations)
            # The constant is in the increment kernel and never converted to float.
            numeric = np.asarray([0.]+[core._finite_float(x, 'Nonconstant teacher coefficient') for x in coefficients[1:]])
            categories = [core.monomial_orbit_index(x) for x in signatures]
            present = [orbit for orbit in range(1, 7) if orbit in categories]
            columns = np.asarray([[float(category == orbit) for orbit in present]
                for category in categories], dtype=float).reshape(len(signatures), len(present))
            labels = []
            previous_atom = None
            for iteration in range(len(signatures)):
                weights, inverse, diagnostics = _solve_span(matrix, numeric, columns, rcond)
                orbit = [coefficients[0]]+[Fraction(0)]*6
                for orbit_index, value in zip(present, weights):
                    orbit[orbit_index] = Fraction(float(value))
                labelled = [(index, Fraction(float(value)))
                    for index, value in zip(labels, weights[len(present):])]
                proposed = _candidate(tuple(orbit), labelled, signatures, diagnostics=diagnostics)
                frozen = [core.decode_rational(value) for value in proposed['expanded_coefficients']]
                residual = np.asarray([0.]+[core._finite_float(c-g, 'Frozen residual coefficient')
                    for c, g in zip(coefficients[1:], frozen[1:])])
                energy = float(residual @ matrix @ residual)
                if not math.isfinite(energy):
                    raise NominalCompressionError('The nominal frozen residual energy is non-finite')
                diagnostics.update(nominal_residual_energy=energy, selected_atom=previous_atom,
                    labelled_monomial_indices=list(labels))
                path['candidates'].append(proposed)
                if frozen == list(coefficients):
                    preparation['termination'] = 'exact_teacher_in_refined_span'
                    break
                cross = columns.T @ matrix
                projected = np.einsum('ij,ij->j', cross, inverse @ cross)
                diagonal = np.diag(matrix)
                schur = diagonal-projected
                q = matrix @ (numeric-columns @ weights)
                threshold = 100*np.finfo(float).eps*np.maximum(np.abs(diagonal), np.abs(projected))
                eligible = []
                for index in range(1, len(signatures)):
                    if index in labels:
                        continue
                    if not math.isfinite(float(schur[index])) or not math.isfinite(float(q[index])):
                        raise NominalCompressionError('A nominal atom energy or residual correlation is non-finite')
                    if schur[index] < -threshold[index]:
                        raise NominalCompressionError('An orthogonalised nominal atom has a substantially negative energy')
                    if schur[index] <= threshold[index]:
                        continue
                    score = abs(float(q[index]))/math.sqrt(float(schur[index]))
                    if not math.isfinite(score):
                        raise NominalCompressionError('A nominal atom gain is non-finite')
                    if score:
                        eligible.append((score, index))
                if not eligible:
                    preparation['termination'] = 'no_resolved_positive_atom_gain'
                    break
                # Lower complete-basis index wins an exactly tied nominal score.
                score, index = max(eligible, key=lambda item: (item[0], -item[1]))
                gain = score*score
                if not math.isfinite(gain):
                    raise NominalCompressionError('A nominal energy reduction overflows')
                previous_atom = dict(monomial_index=index, q=float(q[index]), s=float(schur[index]),
                    nominal_energy_reduction=gain)
                labels.append(index)
                atom = np.zeros((len(signatures), 1)); atom[index, 0] = 1.
                columns = np.concatenate((columns, atom), axis=1)
            else:
                preparation['termination'] = 'finite_complete_basis_limit'
        except (ArithmeticError, RuntimeError, ImportError) as error:
            preparation['failure'] = dict(type=type(error).__name__, message=str(error))
        except Exception as error:
            # NumPy solver failures are typed numerical failures; programming and
            # input errors remain visible rather than silently becoming identity.
            if error.__class__.__name__ != 'LinAlgError':
                raise
            preparation['failure'] = dict(type='LinAlgError', message=str(error))
    path['candidates'].append(identity)
    preparation['cost'] = core._elapsed(start)
    path['path_sha256'] = compression_path_sha256(path)
    return path


def _decode_exact(record):
    try:
        value = core.decode_rational(record)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError('A canonical exact rational record is required') from error
    if (record['approximation'] is not None and type(record['approximation']) is not float) or record != core.rational_record(value):
        raise ValueError('Exact rational records must use canonical numerator, denominator and approximation')
    return value


def _validate_candidates(path, coefficients, signatures):
    candidates = path.get('candidates')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('A finite non-empty candidate hierarchy is required')
    previous_labels = []
    decoded = []
    required = {'kind', 'compression', 'orbit_coefficients', 'labelled_coefficients',
        'expanded_coefficients', 'structural_coefficient_count', 'diagnostics'}
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ValueError('A candidate must have the declared complete schema')
        orbit_records = candidate['orbit_coefficients']
        if not isinstance(orbit_records, list) or len(orbit_records) != 7:
            raise ValueError('Every candidate requires seven exact orbit coefficients')
        orbit = tuple(_decode_exact(x) for x in orbit_records)
        if orbit[0] != coefficients[0] or any(value and order > path['degree'] for value, order in zip(orbit, core.ORBIT_DEGREES)):
            raise ValueError('The teacher constant and declared polynomial degree must be preserved')
        categories = [core.monomial_orbit_index(x) for x in signatures]
        if any(orbit[index] and index not in categories for index in range(1, 7)):
            raise ValueError('Absent orbits must have zero coefficients')
        entries = candidate['labelled_coefficients']
        if not isinstance(entries, list):
            raise ValueError('Labelled coefficients require an ordered list')
        labels, labelled = [], []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'monomial_index', 'coefficient'}:
                raise ValueError('A labelled atom requires its index and exact coefficient')
            index = entry['monomial_index']
            if type(index) is not int or not 1 <= index < len(signatures) or index in labels:
                raise ValueError('Labelled atoms must have distinct nonconstant complete-basis indices')
            labels.append(index); labelled.append((index, _decode_exact(entry['coefficient'])))
        reconstructed = _candidate(orbit, labelled, signatures)
        if candidate['expanded_coefficients'] != reconstructed['expanded_coefficients']:
            raise ValueError('The full expansion must exactly equal orbit plus labelled coefficients')
        if type(candidate['structural_coefficient_count']) is not int or candidate['structural_coefficient_count'] != reconstructed['structural_coefficient_count']:
            raise ValueError('The structural coefficient count must match its exact representation')
        expanded = tuple(_decode_exact(x) for x in candidate['expanded_coefficients'])
        if position == len(candidates)-1:
            identity = _identity(coefficients, signatures)
            if candidate['compression'] is not False or candidate != identity:
                raise ValueError('The last candidate must be the declared exact original identity')
        else:
            if candidate['kind'] != 'refinement' or candidate['compression'] is not True:
                raise ValueError('Nonfinal candidates must be declared refinement candidates')
            if position == 0 and labels:
                raise ValueError('The initial candidate must use the orbit span only')
            if position and (labels[:-1] != previous_labels or len(labels) != len(previous_labels)+1):
                raise ValueError('Each refinement must append exactly one new labelled atom')
            previous_labels = labels
        decoded.append(expanded)
    if len(candidates) > len(signatures)+1:
        raise ValueError('The hierarchy exceeds its finite complete-basis limit')
    return decoded


def certify_compression_path(path, coefficients, initial_state, t0, t1, *, epsilon,
        law, precision=128, expected_path_sha256, degree=None,
        max_gram_entries=DEFAULT_MAX_GRAM_ENTRIES):
    """Accept the first structurally smaller candidate with an outward budget.

    Bindings and exact expansions are verified before any candidate is tested.
    Certificate failures are preserved in attempts. The final identity has an
    exact zero residual under every validated supported law.
    """
    start = core._clocks()
    epsilon = core._exact(epsilon, 'Per-increment square-root energy budget')
    if epsilon < 0:
        raise ValueError('The energy budget must be non-negative')
    if type(precision) is not int or precision < 32:
        raise ValueError('At least 32 precision bits are required')
    if type(max_gram_entries) is not int or max_gram_entries < 1:
        raise ValueError('The Gram-entry budget must be a positive integer')
    if not isinstance(path, dict) or type(path.get('schema_version')) is not int or path.get('schema_version') != 1 or path.get('algorithm') != ALGORITHM:
        raise ValueError('A supported frozen compression path is required')
    declared_degree = path.get('degree')
    if degree is not None and degree != declared_degree:
        raise ValueError('The independently supplied degree must match the path')
    parameters = _law_parameters(law, None, None, None, 0)
    coefficients, initial, t0, t1, actual, signatures = _validated(
        coefficients, initial_state, t0, t1, declared_degree, parameters, law)
    if not isinstance(expected_path_sha256, str) or len(expected_path_sha256) != 64 or any(x not in '0123456789abcdef' for x in expected_path_sha256):
        raise ValueError('An independently retained canonical path SHA256 is required')
    digest = compression_path_sha256(path)
    if digest != expected_path_sha256 or path.get('path_sha256') != digest:
        raise ValueError('The frozen path does not match its independently retained SHA256')
    expected = dict(teacher_coefficients=[core.rational_record(x) for x in coefficients],
        model=core._model_record(initial, actual), previous_time=core.rational_record(t0),
        current_time=core.rational_record(t1), monomial_count=len(signatures),
        monomial_signatures=[list(x) for x in signatures], source_identity=_source_identity())
    if any(path.get(key) != value for key, value in expected.items()) or path.get('teacher_sha256') != _digest(expected['teacher_coefficients']):
        raise ValueError('The frozen teacher, model, time, basis or source identity is incompatible')
    decoded = _validate_candidates(path, coefficients, signatures)
    original_count = path['candidates'][-1]['structural_coefficient_count']
    attempts = []
    selected = certificate = None
    for position, (candidate, expanded) in enumerate(zip(path['candidates'], decoded)):
        if candidate['compression'] and candidate['structural_coefficient_count'] >= original_count:
            attempts.append(dict(candidate_index=position, status='not_structurally_smaller',
                structural_coefficient_count=candidate['structural_coefficient_count']))
            continue
        residual = tuple(c-g for c, g in zip(coefficients, expanded))
        try:
            result = core.certify_residual_energy(residual, (Fraction(0),)*7,
                initial, t0, t1, degree=declared_degree, precision=precision,
                max_gram_entries=max_gram_entries, **parameters)
            upper = core.decode_interval(result['square_root_energy_certificate'])[1]
            accepted = upper <= epsilon
            attempts.append(dict(candidate_index=position, status='accepted' if accepted else 'over_budget',
                structural_coefficient_count=candidate['structural_coefficient_count'], certificate=result))
            if accepted:
                selected, certificate = position, result
                break
        except (ArithmeticError, RuntimeError, ValueError) as error:
            if not candidate['compression']:
                raise
            attempts.append(dict(candidate_index=position, status='certificate_failed',
                error_type=type(error).__name__, message=str(error)))
    if selected is None:
        raise ArithmeticError('The exact original identity certificate was unavailable')
    candidate = path['candidates'][selected]
    return deepcopy(dict(schema_version=1, author='Amanjeet Singh', path_sha256=digest,
        teacher_sha256=path['teacher_sha256'], source_identity=path['source_identity'],
        model=path['model'], previous_time=path['previous_time'], current_time=path['current_time'],
        degree=declared_degree, epsilon=core.rational_record(epsilon),
        selected_candidate_index=selected, selected_candidate=candidate,
        selected_certificate=certificate, compression=candidate['compression'],
        original_structural_coefficient_count=original_count, attempts=attempts,
        population_energy_arithmetic_certified=True, first_acceptable_smaller_candidate=True,
        exact_optimiser_certified=False, global_optimality_claimed=False, cost=core._elapsed(start),
        scope='The full exact residual is certified under the supplied actual GBM law. The budget bounds the absolute difference of expected dual gaps for the same adapted policy. Sum disjoint increment energies before taking the terminal square root; no sampling or simulator certificate is asserted.'))
