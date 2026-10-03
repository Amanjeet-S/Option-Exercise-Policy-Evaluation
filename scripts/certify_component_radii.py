"""Enclose two Hoeffding radius factors in the separate Arb environment."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import flint
from flint import arb, ctx, fmpq

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, required=True)
    parser.add_argument('--paths', type=int, required=True)
    parser.add_argument('--tail-probability', type=Fraction, required=True)
    parser.add_argument('--precision-bits', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to(ROOT/'data/private') or path.exists():
        raise ValueError('Require a fresh private output file')
    if min(args.pairs, args.paths) < 2 or args.precision_bits < 64:
        raise ValueError('Require counts at least two and precision at least 64 bits')
    delta = args.tail_probability
    if not 0 < delta < 1:
        raise ValueError('Tail probability must be strictly between zero and one')
    with ctx.workprec(args.precision_bits):
        inverse_delta = arb(fmpq(delta.denominator, delta.numerator))
        logarithm = inverse_delta.log()
        factors = dict(variance_factor=2*(logarithm/(2*(args.pairs//2))).sqrt(),
                       evaluation_factor=(logarithm/(2*args.pairs)).sqrt()/(args.paths-1))
        result = {}
        for name, value in factors.items():
            if not value.is_finite() or not value > 0:
                raise ArithmeticError('Unresolved positive concentration radius')
            result[name] = dict(lower=str(value.lower().fmpq()), upper=str(value.upper().fmpq()))
    result.update(precision_bits=args.precision_bits, pairs=args.pairs, paths=args.paths,
                  tail_probability=str(delta), upper_endpoints_used=True,
                  python=platform.python_version(), python_flint=flint.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
