"""Nested core-session exercise rights under unchanged dated funding factors."""

from datetime import datetime, timedelta
import math

import numpy as np

from .event_calendar import NEW_YORK, act365f
from ..optimal_stopping.funded_cash import FundedExercise


def core_session_schedules(calendar, observation, expiry, baseline, intervals=(60, 30)):
    """Enlarge an exact core-close schedule without inventing settlement factors.

    These are restricted research calendars. Opening rights use the post-event
    phase and never supply a pre-dividend left-limit exercise opportunity.
    """
    closes = calendar.research_decisions(observation, expiry)
    if not isinstance(baseline, tuple) or len(baseline) != len(closes):
        raise ValueError('Baseline must contain every dated core close exactly once')
    expected = tuple(act365f(observation, t) for t in closes)
    if any(not isinstance(r, FundedExercise) or r.phase != 'after' or r.time != t
           for r, t in zip(baseline, expected)):
        raise ValueError('Baseline time or phase differs from the dated core closes')
    if not intervals or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in intervals):
        raise ValueError('Positive integer minute intervals required')
    if any(a <= b or a % b for a, b in zip(intervals, intervals[1:])):
        raise ValueError('Intervals must strictly decrease and divide their predecessor')
    schedules = {'terminal_only': (baseline[-1],), 'core_close': baseline}
    local = observation.astimezone(NEW_YORK)
    for minutes in intervals:
        rights = {r.time: r for r in baseline}
        for close, source in zip(closes, baseline):
            session = calendar.session(close.date())
            stamp = session.opens_at
            while stamp < session.closes_at:
                if stamp > local:
                    t = act365f(observation, stamp)
                    rights[t] = FundedExercise(t, source.settlement_factor, 'after')
                stamp += timedelta(minutes=minutes)
        schedules[f'core_{minutes}_minutes'] = tuple(rights[t] for t in sorted(rights))
    previous = set()
    for rights in schedules.values():
        if not previous.issubset(set(rights)):
            raise ValueError('Calendar rights and funding factors are not nested')
        previous = set(rights)
    return schedules


def descriptive_strike_slope(strikes, residuals):
    """Within-date/side descriptive OLS slope, without inferential statistics."""
    x, y = np.asarray(strikes, float), np.asarray(residuals, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError('At least two matched strikes and residuals are required')
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or len(np.unique(x)) != len(x):
        raise ValueError('Finite values and distinct strikes are required')
    x = x - x.mean()
    return float(x @ (y-y.mean()) / (x @ x))


def paired_quote_decomposition(call, put):
    """Decompose a matched call-minus-put midpoint discrepancy algebraically.

    Inputs are explicit records with common economic state and dated factors.
    The quoted difference interval is a descriptive comparison, not execution.
    """
    identity = ('date', 'strike', 'expiry', 'dividend_model', 'volatility',
                'quoted_spot', 'initial_share_factor', 'premium_factor',
                'terminal_settlement_factor', 'dividend_caps')
    if call['kind'] != 'call' or put['kind'] != 'put' or any(call[k] != put[k] for k in identity):
        raise ValueError('Call and put must share the complete dated economic specification')
    fields = ('bid', 'ask', 'model_price', 'european_price')
    for row in (call, put):
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in fields) or row['ask'] <= row['bid']:
            raise ValueError('Finite non-negative prices and positive quote widths required')
    observed = (call['bid']+call['ask']-put['bid']-put['ask'])/2
    european = call['european_price']-put['european_price']
    early = (call['model_price']-call['european_price'])-(put['model_price']-put['european_price'])
    model = call['model_price']-put['model_price']
    return {'observed_midpoint_difference': observed,
            'observed_difference_lower': call['bid']-put['ask'],
            'observed_difference_upper': call['ask']-put['bid'],
            'european_difference': european, 'early_exercise_premium_difference': early,
            'model_difference': model, 'model_minus_observed_difference': model-observed,
            'decomposition_rounding_residual': model-(european+early)}
