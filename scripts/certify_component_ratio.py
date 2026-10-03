"""Enclose component-ratio endpoints and a normal-tail calibration in Arb.

This calculation certifies numerical enclosures only. It does not certify
normal approximation, finite-sample coverage or the underlying moment data.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import flint
from flint import arb, ctx, fmpq

ROOT = Path(__file__).resolve().parents[1]
SOURCES = frozenset(('pilot', 'reference', 'selected'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value, name):
    require(isinstance(value, str), f'{name} must be a rational string')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{name} must be a finite rational') from exc


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def endpoint_parameters(spec):
    require(isinstance(spec, dict), 'An endpoint must be an object')
    if spec.get('kind') == 'rational':
        require(set(spec) == {'kind', 'value'}, 'Invalid rational endpoint fields')
        return rational(spec['value'], 'endpoint value'), Fraction(0), 1
    require(spec.get('kind') == 'root', 'Unknown endpoint kind')
    require(set(spec) == {'kind', 'centre', 'radius_squared', 'sign'},
            'Invalid root endpoint fields')
    centre = rational(spec['centre'], 'root centre')
    radius_squared = rational(spec['radius_squared'], 'squared root radius')
    sign = spec['sign']
    require(radius_squared >= 0, 'A squared root radius cannot be negative')
    require(type(sign) is int and sign in (-1, 1), 'Root sign must be -1 or 1')
    return centre, radius_squared, sign


def validate(payload):
    require(isinstance(payload, dict), 'Input must be a JSON object')
    require(set(payload) == {'family_size', 'critical_value', 'family_alpha', 'sets'},
            'Unexpected input fields')
    require(type(payload['family_size']) is int and payload['family_size'] == 39,
            'The registered family contains exactly 39 sets')
    critical = rational(payload['critical_value'], 'critical value')
    alpha = rational(payload['family_alpha'], 'family alpha')
    require(critical == Fraction(13, 4) and alpha == Fraction(1, 20),
            'Require the registered cutoff 13/4 and family alpha 1/20')
    sets = payload['sets']
    require(isinstance(sets, list) and len(sets) == 39,
            'Require exactly 39 component sets')
    seen = set()
    for row in sets:
        require(isinstance(row, dict) and set(row) == {'case', 'source', 'components'},
                'Invalid component-set fields')
        case, source = row['case'], row['source']
        require(type(case) is int and 0 <= case < 13, 'Case must be an integer from 0 to 12')
        require(isinstance(source, str) and source in SOURCES, 'Unknown evidence source')
        require((case, source) not in seen, 'Duplicate case/source pair')
        seen.add((case, source))
        components = row['components']
        require(isinstance(components, list) and len(components) <= 2,
                'Require zero, one or two interval components')
        for component in components:
            require(isinstance(component, dict) and set(component) == {'lower', 'upper'},
                    'Invalid interval component fields')
            endpoint_parameters(component['lower'])
            endpoint_parameters(component['upper'])
    require(seen == {(case, source) for case in range(13) for source in SOURCES},
            'Incomplete registered case/source family')
    return critical, alpha


def ball(value):
    return arb(fmpq(value.numerator, value.denominator))


def enclosure(value):
    if not value.is_finite():
        raise ArithmeticError('A finite numerical enclosure could not be established')
    lower = Fraction(str(value.lower().fmpq()))
    upper = Fraction(str(value.upper().fmpq()))
    if lower > upper:
        raise ArithmeticError('Reversed numerical enclosure')
    return dict(lower=str(lower), upper=str(upper), display=value.str(25))


def enclose_endpoint(spec):
    centre, radius_squared, sign = endpoint_parameters(spec)
    value = ball(centre) + sign * ball(radius_squared).sqrt()
    return dict(spec=spec, **enclosure(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--precision-bits', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, target = args.input.resolve(), args.output.resolve()
    private = ROOT / 'data/private'
    require(source.is_relative_to(private) and source.is_file(),
            'Require an existing private input file')
    require(target.is_relative_to(private) and not target.exists(),
            'Require a fresh private output file')
    require(args.precision_bits >= 64, 'Precision must be at least 64 bits')
    raw = source.read_bytes()
    payload = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    critical, alpha = validate(payload)
    with ctx.workprec(args.precision_bits):
        tail = enclosure(payload['family_size'] * (ball(critical) / arb(2).sqrt()).erfc())
        if not Fraction(tail['upper']) < alpha:
            raise ArithmeticError('The normal family-tail upper bound does not lie below alpha')
        tail['upper_below_family_alpha'] = True
        rows = [dict(case=row['case'], source=row['source'], components=[
            dict(lower=enclose_endpoint(part['lower']), upper=enclose_endpoint(part['upper']))
            for part in row['components']]) for row in payload['sets']]
    result = dict(
        schema_version=1, family_size=payload['family_size'], critical_value=str(critical),
        family_alpha=str(alpha), normal_family_tail=tail, sets=rows,
        input_path=str(source.relative_to(ROOT)), input_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        precision_bits=args.precision_bits, python=platform.python_version(),
        python_flint=flint.__version__, flint_version=flint.__FLINT_VERSION__,
        statistical_coverage_certified=False, allocation_decisions_certified=False,
        scope='Numerical endpoint enclosures and normal-tail calibration only',
        tail_formula='family_size * erfc(critical_value / sqrt(2))',
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
