"""Enclose selected lognormal interpolant expectations using Arb arithmetic.

Run with the separate qf-certification interpreter. Inputs are retained binary64
knots, values, tail slopes, states and volatility-time products. This script does
not certify a global residual supremum, the construction of those interpolants,
or the computation of sigma times sqrt(dt) before its supplied binary64 value.
"""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from flint import arb, fmpq, ctx
import flint


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = (ROOT / 'data/private').resolve()
CERTIFICATION_ENV = (Path.home() / '.venvs/qf-certification').resolve()


def binary64(value, label):
    """Interpret a JSON number as the explicitly specified finite binary64 input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a binary64 numeric input')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{label} must be finite') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite')
    if isinstance(value, int) and int(result) != value:
        raise ValueError(f'{label}: integer input is not exactly representable in binary64')
    return result


def exact(value):
    numerator, denominator = value.as_integer_ratio()
    return arb(fmpq(numerator, denominator))


def endpoint(value):
    rational = value.fmpq()
    return str(rational)


def enclosure(value):
    if not value.is_finite():
        raise ArithmeticError('The ball enclosure is not finite')
    lower, upper = value.lower(), value.upper()
    return dict(lower_rational=endpoint(lower), upper_rational=endpoint(upper),
                display=value.str(30), exact=value.is_exact())


def normal_cdf(value):
    return (-value / arb(2).sqrt()).erfc() / 2


class BallLinearValue:
    """Reconstruct all slopes from exact binary64 knots and values in Arb."""

    def __init__(self, knots, values, tail_slope=0.):
        if not isinstance(knots, list) or not isinstance(values, list):
            raise ValueError('Knots and values must be JSON arrays')
        self.knots = [binary64(x, 'knot') for x in knots]
        raw_values = [binary64(y, 'value') for y in values]
        self.tail_input = binary64(tail_slope, 'tail slope')
        if len(self.knots) < 2 or len(raw_values) != len(self.knots):
            raise ValueError('Equal arrays with at least two knots are required')
        if self.knots[0] != 0. or any(a >= b for a, b in zip(self.knots, self.knots[1:])):
            raise ValueError('Knots must start at zero and increase strictly')
        self.x = [exact(x) for x in self.knots]
        self.y = [exact(y) for y in raw_values]
        self.tail = exact(self.tail_input)
        self.slopes = [(b - a) / (v - u) for a, b, u, v in
                       zip(self.y, self.y[1:], self.x, self.x[1:])]
        extended = self.slopes + [self.tail]
        self.weights = [b - a for a, b in zip(extended, extended[1:])]

    def value(self, state):
        state = binary64(state, 'state')
        if state < 0:
            raise ValueError('State must be non-negative')
        point = exact(state)
        if state >= self.knots[-1]:
            return self.y[-1] + self.tail * (point - self.x[-1])
        index = bisect_right(self.knots, state) - 1
        if state == self.knots[index]:
            return self.y[index]
        return self.y[index] + self.slopes[index] * (point - self.x[index])

    def expectation(self, state, sd):
        state, sd = binary64(state, 'state'), binary64(sd, 'volatility-time product')
        if state < 0 or sd < 0:
            raise ValueError('State and volatility-time product must be non-negative')
        if state == 0.:
            return self.y[0]
        if sd == 0.:
            return self.value(state)
        point, scale = exact(state), exact(sd)
        result = self.y[-1] + self.tail * (point - self.x[-1])
        for strike, weight in zip(self.x[1:], self.weights):
            if weight.is_zero():
                continue
            z = (strike.log() - point.log()) / scale
            put = strike * normal_cdf(z + scale / 2) - point * normal_cdf(z - scale / 2)
            result += weight * put
        if not result.is_finite():
            raise ArithmeticError('Increase precision or inspect the finite input domain')
        return result


def enclose_expectation(specification, state, sd, *, precision_bits=128, reference_float=None):
    """Return rational endpoints for one selected exact-input expectation.

    An optional reference is a supplied numerical implementation's output, not
    another exact expectation. Its difference enclosure quantifies pointwise
    floating-point discrepancy; containment of that float is not required.
    """
    if type(precision_bits) is not int or precision_bits < 64:
        raise ValueError('Precision must be an integer of at least 64 bits')
    with ctx.workprec(precision_bits):
        function = BallLinearValue(specification['knots'], specification['values'],
                                   specification.get('tail_slope', 0.))
        state, sd = binary64(state, 'state'), binary64(sd, 'volatility-time product')
        expected = function.expectation(state, sd)
        result = dict(state_hex=state.hex(), sd_hex=sd.hex(), precision_bits=precision_bits,
                      expectation=enclosure(expected))
        if reference_float is not None:
            reference = binary64(reference_float, 'reference output')
            difference = exact(reference) - expected
            result.update(reference_hex=reference.hex(),
                          reference_inside_ball=expected.contains(exact(reference)),
                          reference_minus_exact=enclosure(difference),
                          absolute_reference_error_upper_rational=endpoint(difference.abs_upper()))
        return result


def self_test():
    """Use analytical identities and exact rational fixtures, without market data."""
    checks = []
    with ctx.workprec(160):
        constant = BallLinearValue([0., 1., 4.], [7., 7., 7.])
        affine = BallLinearValue([0., 1., 4.], [3., 5., 11.], 2.)
        for state in [0., .125, 2., 1000.]:
            for sd in [0., .25, 2.]:
                assert constant.expectation(state, sd).contains(7)
                assert affine.expectation(state, sd).contains(3 + 2 * exact(state))
        checks.append('constant and affine martingale identities over twelve pairs')
        put = BallLinearValue([0., 2., 8.], [2., 0., 0.])
        scale = exact(.5)
        # At-the-money put: k [2 Phi(sd/2)-1] = k erf(sd/(2 sqrt(2))).
        analytical_put = 2 * (scale / (2 * arb(2).sqrt())).erf()
        assert put.expectation(2., .5).overlaps(analytical_put)
        checks.append('at-the-money put against independent erf identity')
        irregular = BallLinearValue([0., 1., 2., 4., 8.], [3., -2., 5., -1., 3.])
        for state, expected in [(0., 3), (.5, Fraction(1, 2)),
                                (1.5, Fraction(3, 2)), (3., 2), (16., 3)]:
            q = fmpq(expected.numerator, expected.denominator) if isinstance(expected, Fraction) else expected
            assert irregular.expectation(state, 0.).contains(q)
        checks.append('non-monotone zero-volatility interpolation and constant upper tail')
        shifted = BallLinearValue([0., 1., 3., 9.], [2., 2., 0., 0.])
        # f((X-1)+) = (3-X)+ - (1-X)+ for f(x)=(2-x)+.
        put3 = BallLinearValue([0., 3., 9.], [3., 0., 0.])
        put1 = BallLinearValue([0., 1., 9.], [1., 0., 0.])
        for state in [0., .25, 1., 5.]:
            a = shifted.expectation(state, .75)
            b = put3.expectation(state, .75) - put1.expectation(state, .75)
            assert a.overlaps(b)
        checks.append('cash plateau expectation against a difference of put payoffs')
        # The small hump sits on a large exact baseline and produces cancelling
        # signed hinge terms. The high-precision enclosure checks precision use.
        spec = dict(knots=[0., 1., 2., 4., 8.],
                    values=[2.**40, 2.**40+1., 2.**40-1., 2.**40+2., 2.**40])
        low = enclose_expectation(spec, 2., .75, precision_bits=80)
        high = enclose_expectation(spec, 2., .75, precision_bits=224)
        ll, lu = (Fraction(low['expectation'][k]) for k in ('lower_rational', 'upper_rational'))
        hl, hu = (Fraction(high['expectation'][k]) for k in ('lower_rational', 'upper_rational'))
        assert max(ll, hl) <= min(lu, hu) and hu - hl < lu - ll
        checks.append('signed-hinge cancellation at 80 and 224 bits; narrower high-precision enclosure')
    return dict(passed=True, checks=checks, market_observations=False,
                global_residual_certificate=False)


def private_path(path):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE):
        raise ValueError('Input and output records must remain under data/private')
    return path


def run(input_path, output_path):
    input_path, output_path = private_path(input_path), private_path(output_path)
    if output_path.exists():
        raise FileExistsError('Use a new output record')
    raw = input_path.read_bytes()
    payload = json.loads(raw)
    if payload.get('schema_version') != 1 or not isinstance(payload.get('functions'), list):
        raise ValueError('Require schema_version 1 and a functions array')
    precision = payload.get('precision_bits', 128)
    if type(precision) is not int or precision < 64:
        raise ValueError('Require integer precision_bits >= 64')
    results, identifiers = [], set()
    for specification in payload['functions']:
        identifier = specification.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('Function identifiers must be distinct non-empty strings')
        identifiers.add(identifier)
        queries = specification.get('queries')
        if not isinstance(queries, list) or not queries:
            raise ValueError('Each function requires a non-empty queries array')
        evaluated = []
        for index, query in enumerate(queries):
            record = enclose_expectation(specification, query['state'], query['sd'],
                precision_bits=precision, reference_float=query.get('reference_float'))
            record.update(query_index=index, query_id=query.get('id'))
            evaluated.append(record)
        results.append(dict(id=identifier, queries=evaluated))
    if not results:
        raise ValueError('At least one function is required')
    record = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Selected full-support expectation enclosures for exact binary64 inputs',
        input_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, platform=platform.platform(), python_flint=flint.__version__,
        flint_version=flint.__FLINT_VERSION__, precision_bits=precision,
        functions=results, exact_binary64_inputs=True, slopes_rebuilt_in_ball_arithmetic=True,
        supplied_sd_is_exact_binary64=True, sigma_sqrt_dt_rounding_certified=False,
        global_residual_supremum_certified=False, market_model_validity_certified=False,
        retained_interpolant_construction_error_certified=False,
        input_data_redistribution_authorised=False)
    if input_path.read_bytes() != raw:
        raise ValueError('Input changed during evaluation')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Enclosed {sum(len(x["queries"]) for x in results)} selected expectations; no global supremum certified.')


def main():
    if Path(sys.prefix).resolve() != CERTIFICATION_ENV:
        raise RuntimeError('Use the separate ~/.venvs/qf-certification/bin/python interpreter')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.input or args.output:
            parser.error('Self-test is separate from input evaluation')
        print(json.dumps(self_test(), indent=2))
    else:
        if args.input is None or args.output is None:
            parser.error('--input and --output are required')
        run(args.input, args.output)


if __name__ == '__main__':
    main()
