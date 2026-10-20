"""Labelled cubic LSM rules for two independent max-call models.

I freeze binary64 training coefficients as exact polynomial constants. The
interval oracle applies that mathematical rule to stock enclosures. Ordinary
floating simulation and diagnostic decisions do not certify the ideal law.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import struct

MODEL_IDS = ('homogeneous', 'heterogeneous')
MONOMIALS = ((), (0,), (1,), (0, 0), (0, 1), (1, 1),
             (0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))
PARAMETERS = dict(strike='100', spots=['100', '100'], rate='1/20',
    dividends=['1/10', '1/10'], exercise_times=['0']+[str(Fraction(j, 3)) for j in range(1, 10)],
    correlation=[['1', '0'], ['0', '1']], asset_order=['asset_1', 'asset_2'])
FIT_PARAMETERS = dict(degree=3, minimum_itm_paths=64, ridge=1e-8,
    ridge_penalty='Nonconstant coefficients only', svd_rcond=1e-12,
    objective='Mean squared strike-normalised discounted cash error plus ridge penalty',
    solver='Augmented SVD least squares', insufficient_itm='Continue',
    time_zero='Continue; zero immediate reward', final_right='Exercise every survivor',
    feature_order=[list(term) for term in MONOMIALS],
    coordinates='Labelled x_i=S_i/100-1; asset identities are preserved',
    target='Time-zero discounted payoff divided by strike',
    exercise='Strict discounted payoff/strike > max(exact polynomial,0); ties continue')


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def exact_model(model_id):
    if model_id not in MODEL_IDS:
        raise ValueError('A declared two-asset model is required')
    return dict(PARAMETERS, model_id=model_id,
        volatilities=['1/5', '1/5' if model_id == 'homogeneous' else '3/10'],
        law='Independent risk-neutral GBM on exact dates j/3, j=1,...,9')


def floating_model(model_id):
    from .multivariate_benchmark import MaxCallModel
    record = exact_model(model_id)
    return MaxCallModel(spots=(100., 100.),
        volatilities=tuple(float(Fraction(s)) for s in record['volatilities']),
        dividends=(.1, .1), correlation=((1., 0.), (0., 1.)),
        rate=.05, strike=100., exercise_times=tuple(float(Fraction(j, 3)) for j in range(10)))


def labelled_basis(states):
    import numpy as np
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2 or not np.all(np.isfinite(states)) or np.any(states <= 0):
        raise ValueError('Both labelled stocks require finite positive states')
    coordinates = states/100.-1.
    result = np.ones((len(states), len(MONOMIALS)))
    for column, term in enumerate(MONOMIALS[1:], 1):
        result[:, column] = np.prod(coordinates[:, term], axis=1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError('Training features overflowed; no state is clipped or censored')
    return result


def validate_paths(states):
    import numpy as np
    states = np.asarray(states, dtype=np.float64)
    if (states.ndim != 3 or states.shape[1:] != (10, 2) or len(states) < 1
            or not np.all(np.isfinite(states)) or np.any(states <= 0)
            or not np.array_equal(states[:, 0], np.full((len(states), 2), 100.))):
        raise ValueError('Complete finite positive two-asset paths are required')
    return states


@dataclass(frozen=True)
class FrozenTwoAssetPolicy:
    model_id: str
    coefficients: tuple[tuple[float, ...], ...]
    training_stream_sha256: str
    training_paths: int
    diagnostics: tuple[dict, ...]

    def __post_init__(self):
        exact_model(self.model_id)
        if (type(self.training_paths) is not int or self.training_paths < 1
                or not isinstance(self.training_stream_sha256, str)
                or len(self.training_stream_sha256) != 64 or len(self.coefficients) != 10
                or len(self.diagnostics) != 10 or self.coefficients[0] or self.coefficients[-1]):
            raise ValueError('A complete frozen policy and training identity are required')
        bytes.fromhex(self.training_stream_sha256)
        for row in self.coefficients:
            if row and (len(row) != 10 or any(not math.isfinite(c) for c in row)):
                raise ValueError('Every active rule needs ten finite labelled cubic coefficients')

    def coefficient_bytes(self):
        return b''.join(struct.pack('<10d', *(row if row else (0.,)*10)) for row in self.coefficients)

    def record(self):
        payload = self.coefficient_bytes()
        return dict(schema_version=1, model=exact_model(self.model_id), fit_parameters=FIT_PARAMETERS,
            training_stream_sha256=self.training_stream_sha256, training_paths=self.training_paths,
            coefficients_hex=[[value.hex() for value in row] for row in self.coefficients],
            coefficient_bytes_layout='Little-endian 10x10 binary64; inactive rows all zero; hex rows mark activity',
            coefficient_bytes_sha256=hashlib.sha256(payload).hexdigest(), diagnostics=list(self.diagnostics),
            coefficient_interpretation='Each finite binary64 coefficient is an exact rational constant of the mathematical labelled polynomial',
            bootstrap_variability_assessed=False, assessor_martingale_fitted=False)


def policy_from_record(record, coefficient_payload=None):
    model_id = record['model']['model_id']
    if record.get('schema_version') != 1 or record['model'] != exact_model(model_id) or record['fit_parameters'] != FIT_PARAMETERS:
        raise ValueError('The exact model or predeclared fitting rule changed')
    policy = FrozenTwoAssetPolicy(model_id,
        tuple(tuple(float.fromhex(c) for c in row) for row in record['coefficients_hex']),
        record['training_stream_sha256'], record['training_paths'], tuple(record['diagnostics']))
    payload = policy.coefficient_bytes()
    if hashlib.sha256(payload).hexdigest() != record['coefficient_bytes_sha256']:
        raise ValueError('Frozen coefficient bytes differ from their identity')
    if coefficient_payload is not None and payload != coefficient_payload:
        raise ValueError('The separate frozen coefficient file differs')
    return policy


def fit_policy(states, model_id, training_stream_sha256):
    """Fit once using only the complete training paths and fixed ridge rule."""
    import numpy as np
    states = validate_paths(states)
    model = floating_model(model_id)
    rows = [() for _ in range(10)]
    diagnostics = [dict(date=j, itm_paths=0, training_exercise_paths=0,
                        design_rank=0, design_condition_number=None, rule='Continue') for j in range(10)]
    cash = model.discounted_payoff(states[:, 9], 9)/100.
    stopping = np.full(len(states), 9, dtype=np.uint8)
    terminal = cash.copy()
    diagnostics[9] = dict(date=9, itm_paths=int((cash > 0).sum()),
        training_exercise_paths=len(states), design_rank=0, design_condition_number=None, rule='Terminal')
    for date in range(8, 0, -1):
        immediate = model.discounted_payoff(states[:, date], date)/100.
        itm = immediate > 0.
        count = int(itm.sum())
        row = diagnostics[date]
        row['itm_paths'] = count
        if count < FIT_PARAMETERS['minimum_itm_paths']:
            row['rule'] = 'Continue: insufficient in-the-money training paths'
            continue
        design = labelled_basis(states[itm, date])
        scale = math.sqrt(count)
        penalty = np.diag([0.]+[math.sqrt(FIT_PARAMETERS['ridge'])]*9)
        augmented = np.vstack((design/scale, penalty))
        target = np.concatenate((cash[itm]/scale, np.zeros(10)))
        coefficient, _, rank, singular = np.linalg.lstsq(augmented, target, rcond=FIT_PARAMETERS['svd_rcond'])
        if not np.all(np.isfinite(coefficient)) or rank != 10:
            raise ArithmeticError('The fixed augmented regression did not return a finite complete rule')
        condition=float(np.linalg.cond(design))
        row.update(design_rank=int(np.linalg.matrix_rank(design, tol=FIT_PARAMETERS['svd_rcond']*np.linalg.norm(design, 2))),
            augmented_rank=int(rank), augmented_condition_number=float(singular[0]/singular[-1]),
            design_condition_number=condition if math.isfinite(condition) else None, rule='Labelled cubic ridge')
        rows[date] = tuple(float(c) for c in coefficient)
        locations = np.flatnonzero(itm)
        exercise = immediate[itm] > np.maximum(design @ coefficient, 0.)
        selected = locations[exercise]
        cash[selected] = immediate[selected]
        stopping[selected] = date
        row['training_exercise_paths'] = len(selected)
    diagnostics[0]['rule'] = 'Continue: dominated zero initial reward'
    diagnostics[9]['surviving_training_paths'] = int((stopping == 9).sum())
    diagnostics[0]['final_training_stopping_counts'] = np.bincount(stopping, minlength=10).tolist()
    diagnostics[0]['terminal_positive_training_payoffs'] = int((terminal > 0).sum())
    return FrozenTwoAssetPolicy(model_id, tuple(rows), training_stream_sha256, len(states), tuple(diagnostics))


def diagnostic_stopping(policy, states):
    """Pathwise binary64 screening only; exact decisions use the interval oracle."""
    import numpy as np
    states = validate_paths(states)
    model = floating_model(policy.model_id)
    discounts = tuple(math.exp(-model.rate*float(Fraction(j, 3))) for j in range(10))
    stopping = np.full(len(states), 9, dtype=np.uint8)
    for path_index, path in enumerate(states):
        for date, row in enumerate(policy.coefficients[1:9], 1):
            if not row:
                continue
            coordinates = tuple(float(s)/100.-1. for s in path[date])
            features = (1.,)+tuple(math.prod(coordinates[i] for i in term) for term in MONOMIALS[1:])
            continuation = math.fsum(c*x for c, x in zip(row, features))
            immediate = discounts[date]*max(max(path[date])-100., 0.)/100.
            if not math.isfinite(continuation) or not math.isfinite(immediate):
                raise ArithmeticError('Diagnostic arithmetic failed; preserve the entire sample')
            if immediate > max(continuation, 0.):
                stopping[path_index] = date
                break
    return stopping


def _interval(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('Every quantity needs an explicit rational endpoint pair')
    result = tuple(Fraction(x) for x in value)
    if result[0] > result[1]:
        raise ValueError('Interval endpoints must be ordered')
    return result


def _add(left, right):
    return left[0]+right[0], left[1]+right[1]


def _multiply(left, right):
    products = tuple(a*b for a in left for b in right)
    return min(products), max(products)


def interval_decision(policy, date, stock_intervals, discount_interval):
    """Decide on exact stocks; unresolved cells require refinement, never censoring.

    The caller supplies outward stock and exp(-r*j/3) enclosures for the ideal
    model. No stock clipping or floating continuation approximation occurs.
    """
    if type(date) is not int or not 1 <= date <= 9:
        raise ValueError('The policy oracle accepts positive exercise dates only')
    if not isinstance(stock_intervals, (tuple, list)) or len(stock_intervals) != 2:
        raise ValueError('Both labelled stock intervals are required')
    stocks = tuple(_interval(value) for value in stock_intervals)
    discount = _interval(discount_interval)
    if any(pair[0] <= 0 for pair in stocks) or discount[0] <= 0:
        raise ValueError('Stocks and discount need strictly positive enclosures')
    if date == 9:
        return dict(decision='exercise', continuation=None, immediate=None, terminal=True)
    row = policy.coefficients[date]
    if not row:
        return dict(decision='continue', continuation=None, immediate=None, terminal=False)
    coordinates = tuple((lower/100-1, upper/100-1) for lower, upper in stocks)
    polynomial = (Fraction(0), Fraction(0))
    for coefficient, term in zip(row, MONOMIALS):
        product = (Fraction(coefficient), Fraction(coefficient))
        for index in term:
            product = _multiply(product, coordinates[index])
        polynomial = _add(polynomial, product)
    continuation = (max(polynomial[0], 0), max(polynomial[1], 0))
    intrinsic = (max(max(s[0] for s in stocks)-100, 0)/100,
                 max(max(s[1] for s in stocks)-100, 0)/100)
    immediate = _multiply(discount, intrinsic)
    decision = ('exercise' if immediate[0] > continuation[1] else
                'continue' if immediate[1] <= continuation[0] else 'unresolved')
    return dict(decision=decision, continuation=continuation, immediate=immediate,
                margin=(immediate[0]-continuation[1], immediate[1]-continuation[0]), terminal=False)
