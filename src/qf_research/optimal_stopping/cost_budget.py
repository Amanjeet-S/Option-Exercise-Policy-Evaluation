"""Timing-only allocation and independent-sample comparisons of dual gaps.

Time allocations are predictions from a separate pilot, not hard deadlines.
The difference interval uses independent evaluation streams; it must not be
applied to paired methods or to policies sharing the same simulated paths.
"""

import math
from statistics import median

from scipy.stats import t


def timing_model(sizes, seconds):
    """Fit a non-negative setup cost and positive marginal cost at two sizes."""
    if (len(sizes) != 2 or any(type(n) is not int or n < 2 for n in sizes)
            or sizes[0] >= sizes[1] or len(seconds) != 2
            or any(len(row) < 3 for row in seconds)
            or len(seconds[0]) != len(seconds[1])
            or any(not math.isfinite(x) or x <= 0 for row in seconds for x in row)):
        raise ValueError('Require two increasing sizes and at least three positive timings each')
    low, high = map(median, seconds)
    slope = (high-low)/(sizes[1]-sizes[0])
    if slope <= 0:
        raise ValueError('Pilot does not identify a positive marginal cost')
    raw_setup = low-slope*sizes[0]
    return dict(setup_seconds=max(0., raw_setup), seconds_per_path=slope,
                unconstrained_setup_seconds=raw_setup, median_seconds=[low, high],
                timing_ranges=[[min(row), max(row)] for row in seconds])


def allocate_paths(model, budget, *, minimum=2000, maximum=300000, quantum=1000):
    """Round down a pilot prediction; never silently exceed it to reach a floor."""
    setup, slope = model['setup_seconds'], model['seconds_per_path']
    if (not all(math.isfinite(x) for x in (setup, slope, budget)) or setup < 0
            or slope <= 0 or budget <= 0 or any(type(x) is not int for x in (minimum, maximum, quantum))
            or minimum < 2 or maximum < minimum or quantum < 1):
        raise ValueError('Invalid allocation inputs')
    affordable = math.floor((budget-setup)/slope/quantum)*quantum
    cap = maximum//quantum*quantum
    count = min(affordable, cap)
    if count < minimum:
        return dict(status='infeasible', paths=None, predicted_seconds=None, capped=False)
    return dict(status='allocated', paths=count, predicted_seconds=setup+slope*count,
                capped=affordable > cap)


def independent_difference(right, left, *, confidence=.95):
    """Welch interval for independent sample means, in the supplied units."""
    if not 0 < confidence < 1:
        raise ValueError('Invalid confidence level')
    for estimate in (right, left):
        if (type(estimate['observations']) is not int or estimate['observations'] < 2
                or not math.isfinite(estimate['value'])
                or not math.isfinite(estimate['standard_error']) or estimate['standard_error'] < 0):
            raise ValueError('Invalid mean or standard error')
    vr, vl = right['standard_error']**2, left['standard_error']**2
    value = right['value']-left['value']
    variance = vr+vl
    if variance == 0:
        return dict(value=value, standard_error=0., lower=value, upper=value,
                    degrees_of_freedom=None, confidence=confidence)
    df = variance**2/(vr**2/(right['observations']-1)+vl**2/(left['observations']-1))
    se = math.sqrt(variance)
    radius = float(t.ppf((1+confidence)/2, df))*se
    return dict(value=value, standard_error=se, lower=value-radius, upper=value+radius,
                degrees_of_freedom=df, confidence=confidence)
