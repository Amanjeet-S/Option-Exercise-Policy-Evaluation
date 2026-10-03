"""Joint moment uncertainty and exact quadratic inversion for component ratios.

Normal calibration is asymptotic. Exact algebra on saved summaries does not
certify their original observation errors or finite-sample statistical coverage.
"""

from fractions import Fraction
from math import isfinite, isqrt

from .allocation_variance_sensitivity import envelope


def _number(x, name):
    if isinstance(x, bool) or not isinstance(x, (int, float, Fraction)):
        raise ValueError(f'{name} must be a finite real number')
    if isinstance(x, float) and not isfinite(x):
        raise ValueError(f'{name} must be finite')
    return Fraction(x)


def _integer(x, name):
    if type(x) is not int or x < 2:
        raise ValueError(f'{name} must be an integer at least two')


def moment_covariance(means, standard_errors, paths, reference_paths=4096):
    """Retain signed moments and an empirical joint influence covariance."""
    _integer(paths, 'paths'); _integer(reference_paths, 'reference_paths')
    w = [_number(x, 'mean') for x in means]
    errors = [_number(x, 'standard error') for x in standard_errors]
    if len(w) < 3 or len(w) != len(errors) or any(x < 0 for x in errors):
        raise ValueError('Require at least three paired means and nonnegative errors')
    m = len(w)
    mean = sum(w)/m
    u = [x*x for x in errors]; ubar = sum(u)/m
    squares = [(x-mean)**2 for x in w]; dbar = sum(squares)/m
    observed = sum(squares)/(m-1)
    kappa = Fraction(paths, reference_paths)
    d_hat, e_hat = observed+(kappa-1)*ubar, kappa*ubar
    psi_d = [d-dbar+(kappa-1)*(q-ubar) for d, q in zip(squares, u)]
    psi_e = [kappa*(q-ubar) for q in u]
    scale = m*(m-1)
    dd = sum(x*x for x in psi_d)/scale
    de = sum(x*y for x, y in zip(psi_d, psi_e))/scale
    ee = sum(x*x for x in psi_e)/scale
    determinant = dd*ee-de*de
    if determinant < 0:
        raise ArithmeticError('Influence covariance is not positive semidefinite')
    point = e_hat/d_hat if d_hat > 0 else None

    def concentration(values):
        squared = [x*x for x in values]
        total = sum(squared)
        return max(squared)/total if total > 0 else None

    return dict(pairs=m, paths=paths, reference_paths=reference_paths,
                observed_variance=observed, evaluation_mean_variance=ubar,
                A=observed-ubar, B=paths*ubar, denominator=d_hat, numerator=e_hat,
                covariance={'dd': dd, 'de': de, 'ee': ee},
                covariance_rank=2 if determinant > 0 else 1 if dd+ee > 0 else 0,
                point_share=point, point_in_parameter_domain=point is not None and 0 <= point <= 1,
                largest_squared_influence_share={
                    'denominator': concentration(psi_d), 'numerator': concentration(psi_e),
                    'ratio_contrast': concentration([y-point*x for x, y in zip(psi_d, psi_e)])
                    if point is not None else None})


def fieller_coefficients(denominator, numerator, covariance, critical_value=Fraction(13, 4)):
    d, e, z = (_number(x, name) for x, name in (
        (denominator, 'denominator'), (numerator, 'numerator'), (critical_value, 'critical_value')))
    dd, de, ee = (_number(covariance[k], k) for k in ('dd', 'de', 'ee'))
    if z <= 0 or min(dd, ee) < 0 or dd*ee-de*de < 0:
        raise ValueError('Require a positive cutoff and positive semidefinite covariance')
    return dict(a=d*d-z*z*dd, b=-2*d*e+2*z*z*de, c=e*e-z*z*ee)


def _polynomial(a, b, c):
    return tuple(_number(x, name) for x, name in zip((a, b, c), ('a', 'b', 'c')))


def intersects(a, b, c, lower, upper, *, include_lower=True, include_upper=True):
    """Whether q(w)<=0 holds anywhere in a possibly open rational interval."""
    a, b, c = _polynomial(a, b, c)
    lower, upper = _number(lower, 'lower'), _number(upper, 'upper')
    if lower > upper or type(include_lower) is not bool or type(include_upper) is not bool:
        raise ValueError('Require ordered endpoints and Boolean ownership flags')
    q = lambda x: a*x*x+b*x+c
    if lower == upper:
        return include_lower and include_upper and q(lower) <= 0
    # A strict negative boundary value also supplies interior accepted points.
    if q(lower) < 0 or q(upper) < 0:
        return True
    if include_lower and q(lower) == 0 or include_upper and q(upper) == 0:
        return True
    if q((lower+upper)/2) <= 0:
        return True
    if a > 0:
        vertex = -b/(2*a)
        if lower < vertex < upper and q(vertex) <= 0:
            return True
    return False


def _rational(x):
    return dict(kind='rational', value=Fraction(x))


def _root(centre, radius_squared, sign):
    n, d = isqrt(radius_squared.numerator), isqrt(radius_squared.denominator)
    if n*n == radius_squared.numerator and d*d == radius_squared.denominator:
        return _rational(centre+sign*Fraction(n, d))
    return dict(kind='root', centre=centre, radius_squared=radius_squared, sign=sign)


def endpoint_compare(endpoint, rational):
    """Compare an algebraic endpoint with a rational without taking a root."""
    rational = _number(rational, 'comparison point')
    if endpoint['kind'] == 'rational':
        difference = endpoint['value']-rational
    else:
        t = endpoint['centre']-rational
        r = endpoint['radius_squared']
        if endpoint['sign'] == 1:
            if t >= 0:
                return 1
            difference = r-t*t
        else:
            if t <= 0:
                return -1
            difference = t*t-r
    return (difference > 0)-(difference < 0)


def quadratic_set(a, b, c):
    """Complete quadratic acceptance set intersected with [0,1].

    Irrational endpoints remain algebraic. No tolerance decides the degree,
    discriminant, disconnectedness or inclusion of an equality endpoint.
    """
    a, b, c = _polynomial(a, b, c)
    zero, one = _rational(0), _rational(1)
    parts = []

    def clip(left, right):
        if endpoint_compare(left, 1) > 0 or endpoint_compare(right, 0) < 0:
            return
        if endpoint_compare(left, 0) <= 0:
            left = zero
        if endpoint_compare(right, 1) >= 0:
            right = one
        parts.append(dict(lower=left, upper=right))

    discriminant = b*b-4*a*c
    if a == 0:
        if b == 0:
            unrestricted = 'all_real' if c <= 0 else 'empty'
            if c <= 0: parts.append(dict(lower=zero, upper=one))
        else:
            unrestricted = 'half_line'
            root = _rational(-c/b)
            if b > 0: clip(zero, root)
            else: clip(root, one)
    elif discriminant < 0:
        unrestricted = 'all_real' if a < 0 else 'empty'
        if a < 0: parts.append(dict(lower=zero, upper=one))
    elif discriminant == 0:
        unrestricted = 'all_real' if a < 0 else 'singleton'
        if a < 0: parts.append(dict(lower=zero, upper=one))
        else:
            root = _rational(-b/(2*a)); clip(root, root)
    else:
        centre, radius_squared = -b/(2*a), discriminant/(4*a*a)
        lower = _root(centre, radius_squared, -1)
        upper = _root(centre, radius_squared, 1)
        if a > 0:
            unrestricted = 'bounded_interval'; clip(lower, upper)
        else:
            unrestricted = 'two_rays'; clip(zero, lower); clip(upper, one)
    if not parts: kind = 'empty'
    elif len(parts) > 1: kind = 'two_intervals'
    elif parts[0] == dict(lower=zero, upper=one): kind = 'full_domain'
    elif parts[0]['lower'] == parts[0]['upper']: kind = 'singleton'
    else: kind = 'interval'
    return dict(unrestricted_kind=unrestricted, physical_kind=kind,
                discriminant=discriminant, components=parts)


def allocation_choices(a, b, c, candidates):
    """Intersect the exact quadratic set with every allocation's own region."""
    a, b, c = _polynomial(a, b, c)
    weak, selected = [], []
    for row in envelope(candidates):
        interval = row['region']; n = row['evaluation_paths']
        if intersects(a, b, c, interval['lower'], interval['upper']):
            weak.append(n)
        if intersects(a, b, c, interval['lower'], interval['upper'],
                      include_lower=interval['lower_selected'], include_upper=interval['upper_selected']):
            selected.append(n)
    return dict(weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
                uniformly_selected_size=selected[0] if len(selected) == 1 else None)
