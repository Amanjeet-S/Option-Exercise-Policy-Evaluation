"""Variance-adaptive sequential width diagnostics for bounded pair means.

The construction combines Howard, Ramdas, McAuliffe and Sekhon (2021),
Theorem 4 and equation (4.1), with their stitched boundary in equation (3.1)
and Theorem 1. Journal pages 1062--1063 and 1069--1070 were consulted in the
published paper, doi:10.1214/20-AOS1991. The sub-gamma boundary applies because
psi_E,c(lambda) <= psi_G,c(lambda) for 0 <= lambda < 1/c.

For observations in [a,b], the scale is the support diameter c=b-a. Each
tail uses (1-confidence)/2; Theorem 4's two-sided statement requires this
factor of two. Predictable squared deviations replace a known variance.

The mathematical coverage result requires a valid deterministic support,
predictable choices and the stated conditional-mean target. A support or
tuning rule chosen after inspecting outcomes is not automatically covered.
These functions calculate floating-point widths; they do not certify a
support proof, conditional centring, arithmetic error or empirical coverage.
"""

import math
from numbers import Real

import numpy as np
from scipy.special import zeta


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite real number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return value


def _settings(support_width, confidence, eta, s, m):
    width = _finite(support_width, 'support_width')
    confidence = _finite(confidence, 'confidence')
    eta, s = _finite(eta, 'eta'), _finite(s, 's')
    if width <= 0 or not 0 < confidence < 1 or eta <= 1 or s <= 1:
        raise ValueError('Require positive support width, 0 < confidence < 1, eta > 1 and s > 1')
    m = width * width if m is None else _finite(m, 'm')
    if not math.isfinite(m) or m <= 0:
        raise ValueError('m must be representable as a finite positive number')
    zeta_s = float(zeta(s, 1.))
    if not math.isfinite(zeta_s) or zeta_s < 1:
        raise ValueError('The stitching weights are not representable')
    return dict(support_width=width, confidence=confidence,
                alpha_side=(1-confidence)/2, eta=eta, s=s, m=m,
                log_zeta_s=math.log(zeta_s))


def _boundary(intrinsic_variance, settings):
    variance = _finite(intrinsic_variance, 'intrinsic_variance')
    if variance < 0:
        raise ValueError('intrinsic_variance must be non-negative')
    c, eta, s, m = (settings[key] for key in ('support_width', 'eta', 's', 'm'))
    v = max(variance, m)
    # Logarithms avoid an overflowing v/m ratio. At v=m the epoch is zero.
    epoch = max(0., (math.log(v)-math.log(m))/math.log(eta))
    ell = (s * math.log1p(epoch) + settings['log_zeta_s']
           - math.log(settings['alpha_side']))
    k1 = (eta**.25 + eta**(-.25))/math.sqrt(2.)
    k2 = (math.sqrt(eta)+1.)/2.
    linear = k2*c*ell
    root_term = k1*math.sqrt(v)*math.sqrt(ell)
    boundary = math.hypot(root_term, linear)+linear
    if not math.isfinite(boundary) or boundary <= 0:
        raise ValueError('The stitched boundary is not a finite positive number')
    return boundary


def stitched_empirical_bernstein_boundary(intrinsic_variance, support_width, *,
                                         confidence=.95, eta=2., s=1.4, m=None):
    """Return the two-sided construction's boundary for the centred sum.

    The one-sided stitched boundary has crossing probability
    ``alpha_side=(1-confidence)/2``. Its spending function is
    ``h(k)=zeta(s)*(k+1)**s``. The returned sum boundary must be divided by
    the completed row count to obtain a mean half-width.

    ``intrinsic_variance`` is the sum of squared deviations from predictable
    observations, not an estimated variance of a sample mean. ``m`` has units
    of squared observation and defaults to ``support_width**2``. No correction
    across cases or arms is hidden in ``confidence``.
    """
    settings = _settings(support_width, confidence, eta, s, m)
    return _boundary(intrinsic_variance, settings)


def empirical_bernstein_halfwidth(intrinsic_variance, repetitions, support_width, *,
                                 confidence=.95, eta=2., s=1.4, m=None):
    """Return the mean half-width at one supplied completed-prefix length."""
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be a positive integer')
    boundary = stitched_empirical_bernstein_boundary(
        intrinsic_variance, support_width, confidence=confidence, eta=eta, s=s, m=m)
    try:
        halfwidth = boundary/repetitions
    except OverflowError as exc:
        raise ValueError('The repetition count is not numerically representable') from exc
    if not math.isfinite(halfwidth) or halfwidth <= 0:
        raise ValueError('The half-width is not a finite positive number')
    return halfwidth


def empirical_bernstein_sequence(row_means, support_lower, support_upper, *,
                                initial_prediction=None, confidence=.95,
                                eta=2., s=1.4, m=None):
    """Calculate predictable-variance diagnostics for every completed prefix.

    ``row_means`` must be a finite, nonempty, one-dimensional ordered sequence
    of completed independent pair means for the intended common-mean target.
    More generally Theorem 4 targets the average conditional mean; dependence
    cannot be discarded merely by giving the data this name.

    The first prediction is the support midpoint, or a supplied fixed pilot
    prediction in the support. Later predictions use only preceding row means.
    The current observation never enters its own prediction. The supplied
    support and any pilot choices must be independent of the main sequence for
    the conditional fixed-target interpretation used in this project.

    Returned dictionaries are JSON-native. Intersected endpoints use the
    deterministic support, which also bounds the mean; untruncated endpoints
    and widths are retained. Finite-sample coverage is a mathematical property
    under assumptions, not something this retrospective diagnostic validates.
    """
    lower = _finite(support_lower, 'support_lower')
    upper = _finite(support_upper, 'support_upper')
    if lower >= upper:
        raise ValueError('Require support_lower < support_upper')
    settings = _settings(upper-lower, confidence, eta, s, m)
    midpoint = lower+settings['support_width']/2.
    initial = midpoint if initial_prediction is None else _finite(initial_prediction, 'initial_prediction')
    if not lower <= initial <= upper:
        raise ValueError('initial_prediction must belong to the deterministic support')
    try:
        values = np.asarray(row_means, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError('Require a finite one-dimensional sequence of real row means') from exc
    if values.ndim != 1 or len(values) == 0:
        raise ValueError('Require a nonempty one-dimensional sequence of row means')
    # Reject Boolean, complex and string observations instead of silently casting.
    observations = [_finite(value, 'row mean') for value in values]
    if any(value < lower or value > upper for value in observations):
        raise ValueError('A row mean falls outside the supplied deterministic support')
    mean, variance, compensation = 0., 0., 0.
    rows = []
    for index, value in enumerate(observations):
        count = index+1
        prediction = initial if index == 0 else min(upper, max(lower, mean))
        try:
            squared_error = (value-prediction)**2
        except OverflowError as exc:
            raise ValueError('A squared prediction error is not numerically representable') from exc
        # Compensated accumulation limits lost small terms. It is not a proof
        # of a floating-point enclosure of the theoretical variance process.
        adjusted = squared_error-compensation
        updated = variance+adjusted
        compensation = (updated-variance)-adjusted
        variance = updated
        mean = value if index == 0 else mean+(value-mean)/count
        if not math.isfinite(variance) or not math.isfinite(mean):
            raise ValueError('Prefix moments are not numerically representable')
        width = _boundary(variance, settings)/count
        untruncated_lower, untruncated_upper = mean-width, mean+width
        if not all(math.isfinite(x) for x in (width, untruncated_lower, untruncated_upper)):
            raise ValueError('Prefix interval endpoints are not numerically representable')
        rows.append(dict(repetitions=count, mean=mean, prediction=prediction,
            intrinsic_variance=variance, halfwidth=width,
            untruncated_lower=untruncated_lower, untruncated_upper=untruncated_upper,
            lower=max(lower, untruncated_lower), upper=min(upper, untruncated_upper)))
    return dict(settings={key: value for key, value in settings.items() if key != 'log_zeta_s'},
        support_lower=lower, support_upper=upper, initial_prediction=initial,
        rows=rows, floating_point_bound_certified=False, coverage_validated=False,
        scope='Numerical sequential-width diagnostic; mathematical coverage requires valid support, predictability and the stated conditional-mean target')
