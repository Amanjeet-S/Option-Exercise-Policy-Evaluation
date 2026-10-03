"""Fixed prospective allocation rules and independent-arm moment comparisons.

Exact arithmetic applies to retained summaries. Normal contrast intervals are
asymptotic; neither clock accuracy nor simulation rounding is certified here.
"""

from fractions import Fraction
from math import isfinite

from .allocation_variance_sensitivity import regret, select


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError('Require a finite real number')
    if isinstance(value, float) and not isfinite(value):
        raise ValueError('Require a finite real number')
    return Fraction(value)


def minimise_relative_variance(candidates, components):
    """Minimise worst relative variance on a declared union of closed intervals.

    Each ratio of positive affine criteria is monotone or constant. Its maximum
    on each component lies at an endpoint, including a singleton component.
    Ties select the smaller evaluation count, as in the retained selector.
    """
    if not components:
        raise ValueError('A nonempty declared component set is required')
    intervals = []
    for row in components:
        low, high = _number(row['lower']), _number(row['upper'])
        if not 0 <= low <= high <= 1:
            raise ValueError('Require ordered component endpoints in [0,1]')
        intervals.append((low, high))
    # The existing selector validates distinct positive integer counts.
    select(candidates, intervals[0][0])
    scores = []
    for candidate in sorted(candidates, key=lambda row: row['evaluation_paths']):
        maxima = [regret(candidates, candidate['evaluation_paths'], low, high)['maximum']
                  for low, high in intervals]
        scores.append(dict(evaluation_paths=candidate['evaluation_paths'],
                           repetitions=candidate['repetitions'], maximum_relative_variance=max(maxima)))
    best = min(r['maximum_relative_variance'] for r in scores)
    minimisers = [r['evaluation_paths'] for r in scores if r['maximum_relative_variance'] == best]
    chosen = next(r for r in scores if r['evaluation_paths'] == minimisers[0])
    return dict(selected=dict(chosen), minimisers=minimisers, scores=scores)


def sample_summary(means):
    """Unbiased estimated variance of a fixed-count mean and moment diagnostics."""
    values = [_number(x) for x in means]
    m = len(values)
    if m < 3:
        raise ValueError('At least three independent rows are required')
    total, total_squared = sum(values), sum(x*x for x in values)
    mean = total/m
    deviations = [x-mean for x in values]
    squares = [x*x for x in deviations]
    sum_squared = sum(squares)
    observed = sum_squared/(m-1)
    dbar = sum_squared/m
    fourth_influences = [d-dbar for d in squares]
    fourth_total = sum(x*x for x in fourth_influences)
    estimate_variance = observed/m
    variance_of_variance = fourth_total/(m**3*(m-1))
    covariance = sum(x*y for x, y in zip(deviations, fourth_influences))/(m*m*(m-1))
    if covariance*covariance > estimate_variance*variance_of_variance:
        raise ArithmeticError('Moment covariance is not positive semidefinite')
    leave_one = [(total_squared-x*x-(total-x)**2/(m-1))/((m-2)*m) for x in values]
    return dict(pairs=m, mean=mean, row_variance=observed,
                variance_of_mean=estimate_variance,
                variance_of_variance=variance_of_variance,
                covariance_mean_variance=covariance,
                mean_variance_degenerate=estimate_variance == 0,
                variance_influence_degenerate=variance_of_variance == 0,
                largest_squared_mean_influence_share=max(squares)/sum_squared if sum_squared else None,
                largest_squared_variance_influence_share=max(x*x for x in fourth_influences)/fourth_total if fourth_total else None,
                leave_one_variance_at_fixed_count=dict(lower=min(leave_one), upper=max(leave_one),
                    lower_row=leave_one.index(min(leave_one)), upper_row=leave_one.index(max(leave_one))))


def _contrast(difference, sampling_variance, critical):
    if sampling_variance < 0:
        raise ValueError('Sampling variance cannot be negative')
    radius_squared = critical*critical*sampling_variance
    if sampling_variance == 0:
        direction = 'degenerate'
    elif difference*difference <= radius_squared:
        direction = 'includes_zero'
    else:
        direction = 'negative' if difference < 0 else 'positive'
    return dict(difference=difference, estimated_sampling_variance=sampling_variance,
                radius_squared=radius_squared, normal_interval_status=direction,
                asymptotic_only=True)


def compare_summaries(target, reference, critical_value=Fraction(13, 4)):
    """Two marginal contrast intervals for independently sampled complete arms."""
    critical = _number(critical_value)
    if critical <= 0:
        raise ValueError('A positive normal cutoff is required')
    keys = ('mean', 'variance_of_mean', 'variance_of_variance')
    target = {k: _number(target[k]) for k in keys}
    reference = {k: _number(reference[k]) for k in keys}
    if min(target['variance_of_mean'], target['variance_of_variance'],
           reference['variance_of_mean'], reference['variance_of_variance']) < 0:
        raise ValueError('Require nonnegative sampling variance estimates')
    v0, v1 = reference['variance_of_mean'], target['variance_of_mean']
    return dict(mean_contrast=_contrast(target['mean']-reference['mean'], v1+v0, critical),
                variance_contrast=_contrast(v1-v0,
                    target['variance_of_variance']+reference['variance_of_variance'], critical),
                estimated_variance_ratio=v1/v0 if v0 > 0 else None,
                ratio_has_confidence_interval=False, critical_value=critical)
