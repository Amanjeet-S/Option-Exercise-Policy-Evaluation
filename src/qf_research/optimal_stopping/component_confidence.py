"""Exact geometry for a bounded-cluster variance-component confidence procedure.

The probability argument assumes ideal independent clusters and valid supplied
summary-error bounds. Rational geometry alone does not verify those assumptions.
"""

from fractions import Fraction
from itertools import combinations
import math

from .allocation_variance_sensitivity import envelope, regret, select


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f'{name} must be finite')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return Fraction(value)


def region(observed_variance, evaluation_mean_variance, variance_radius,
           evaluation_radius, bound_squared, *, pilot_paths=4096,
           tail_probability=Fraction(1, 1040), variance_error=0, evaluation_error=0):
    """Intersect four one-sided moment limits with the parameter support.

    v=A+B/n and u=B/n satisfy v>=u>=0 and v+(n-1)u<=L^2.
    Markov gives each lower limit; externally justified Hoeffding radii give
    the upper limits. Error allowances widen both limits outwards.
    """
    if type(pilot_paths) is not int or pilot_paths < 2:
        raise ValueError('pilot_paths must be an integer at least two')
    names = ('observed_variance', 'evaluation_mean_variance', 'variance_radius',
             'evaluation_radius', 'bound_squared', 'variance_error', 'evaluation_error')
    values = [_number(x, name) for name, x in zip(names, (
        observed_variance, evaluation_mean_variance, variance_radius,
        evaluation_radius, bound_squared, variance_error, evaluation_error))]
    if any(x < 0 for x in values) or values[4] == 0:
        raise ValueError('Require nonnegative inputs and a positive squared bound')
    s, u, rv, ru, cap, ev, eu = values
    delta = _number(tail_probability, 'tail_probability')
    if not 0 < delta < 1:
        raise ValueError('tail_probability must lie strictly between zero and one')
    vl, vu = delta*max(Fraction(0), s-ev), s+ev+rv
    ul, uu = delta*max(Fraction(0), u-eu), u+eu+ru
    # A*v+B*u <= C. The lower limits already impose nonnegativity.
    constraints = [(-1, 0, -vl), (1, 0, vu), (0, -1, -ul),
                   (0, 1, uu), (-1, 1, 0), (1, pilot_paths-1, cap)]
    vertices = set()
    for (a, b, c), (d, e, f) in combinations(constraints, 2):
        determinant = a*e-b*d
        if determinant == 0:
            continue
        v = Fraction(c*e-b*f, determinant)
        q = Fraction(a*f-c*d, determinant)
        if all(x*v+y*q <= z for x, y, z in constraints):
            vertices.add((v, q))
    rows = [dict(v=v, u=q, A=v-q, B=pilot_paths*q) for v, q in sorted(vertices)]
    shares = [q/v for v, q in vertices if v > 0]
    return dict(intervals={'v': [vl, vu], 'u': [ul, uu]}, vertices=rows,
                share_bounds={'lower': min(shares), 'upper': max(shares)} if shares else None,
                zero_variance_possible=(Fraction(0), Fraction(0)) in vertices,
                status='empty' if not vertices else 'nondegenerate' if shares else 'degenerate_zero')


def decision_summary(candidates, lower, upper, retained_N):
    """Map a component interval to weak optima, actual choices and regret."""
    lower, upper = _number(lower, 'lower'), _number(upper, 'upper')
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Require an ordered share interval in [0, 1]')
    geometry = envelope(candidates)
    weak, selected = [], []
    for row in geometry:
        n, interval = row['evaluation_paths'], row['region']
        lo, hi = max(lower, interval['lower']), min(upper, interval['upper'])
        if lo > hi:
            continue
        weak.append(n)
        if lo < hi or select(candidates, lo)['evaluation_paths'] == n:
            selected.append(n)
    uniform = selected[0] if len(selected) == 1 else None
    return dict(weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
                uniformly_selected_size=uniform,
                retained_choice_regret=regret(candidates, retained_N, lower, upper))
