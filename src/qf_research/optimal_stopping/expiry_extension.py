"""A later expiry decision with unchanged preceding exercise instructions.

The added interval has unchanged settlement, no cash event and the same
driftless lognormal state. Conditional Jensen reduces its exercise problem to
its endpoint. The public methods note derives the terminal-reward comparison.
"""

from dataclasses import replace
import math

import numpy as np
from scipy.special import erf

from .funded_cash import FundedExercise, _apply, _result, european_reference
from .funded_policy import FundedPaths, _validate_paths, realise_policy
from .funded_dual import DualEvaluation
from .value_martingale import LinearValue
from .simulation import estimate_samples
from qf_research.computational.funded_kernel import advance_scipy, price_prepared


def check_extension(model, option, extension):
    if isinstance(extension, bool) or not math.isfinite(extension) or extension < 0:
        raise ValueError('A finite non-negative expiry extension is required')
    if any(option.maturity < d.ex_time <= option.maturity+extension and d.discounted_cash > 0 for d in model.dividends):
        raise ValueError('The expiry extension must contain no cash transfer')


def tail_put(states, settlement_strike, volatility, extension):
    """Conditional value at the old expiry of the later put exercise reward."""
    if not math.isfinite(settlement_strike) or settlement_strike <= 0:
        raise ValueError('A positive finite settlement strike is required')
    if not math.isfinite(volatility) or volatility < 0:
        raise ValueError('Finite non-negative volatility required')
    if not math.isfinite(extension) or extension < 0:
        raise ValueError('Finite non-negative expiry extension required')
    if volatility == 0 or extension == 0:
        values = np.asarray(states, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Finite non-negative states required')
        return np.maximum(settlement_strike-values, 0.)
    payoff = LinearValue((0., settlement_strike), (settlement_strike, 0.))
    return payoff.expectation(states, volatility*math.sqrt(extension))


def price_terminal_extension(prepared, extension, advance=advance_scipy):
    """Change only the terminal data on an unchanged prepared PDE clock.

    The extra interval is integrated analytically. The preceding grid, damping,
    boundary arrays and cash-event maps are identical for every extension.
    """
    p = prepared
    check_extension(p.model, p.option, extension)
    if p.option.kind != 'put':
        raise ValueError('Terminal replacement currently supports puts only')
    terminal = p.rights[-1]
    values = tail_put(p.grid, p.option.strike*terminal.settlement_factor, p.model.volatility, extension)
    # Events at the original expiry require separate before/after treatment.
    if p.option.maturity in p.events:
        raise ValueError('A cash event at the original expiry is unsupported')
    if p.model.volatility == 0:
        return price_prepared(p, advance)
    for block in p.blocks:
        values = advance(p.diffusion, values, block.widths, block.theta, block.left, block.right)
        values = _apply(p.model, p.option, p.grid, values, block.earlier, p.rights, p.events)
    return _result(float(np.interp(p.model.initial_state, p.grid, values)), p.model,
                   p.grid, p.time_intervals, p.rights, p.events)


def extended_calendar(model, option, original, extensions):
    """Preserve every original right and add ordered later terminal rights."""
    if not original or original[-1].time != option.maturity or original[-1].phase != 'after':
        raise ValueError('Original terminal post-event right required')
    if not extensions or any(a >= b for a,b in zip((0., *extensions[:-1]), extensions)):
        raise ValueError('Strictly increasing positive extensions required')
    for extension in extensions:
        check_extension(model, option, extension)
    rights = original+tuple(FundedExercise(option.maturity+d, original[-1].settlement_factor) for d in extensions)
    return replace(option, maturity=rights[-1].time), rights


def restrict_paths(paths, option, decisions):
    """Restrict one common path sample to a nested clock without new draws."""
    if option.maturity > paths.maturity:
        raise ValueError('Cannot extend observed paths')
    lookup = {d: i for i,d in enumerate(paths.decisions)}
    if any(d not in lookup for d in decisions):
        raise ValueError('Every restricted right must have unchanged terms in the sample')
    states = paths.states[:, [lookup[d] for d in decisions]].copy()
    event_count = sum(d.ex_time <= option.maturity for d in paths.model.dividends)
    payments = paths.detached_payments[:, :event_count].copy()
    states.setflags(write=False); payments.setflags(write=False)
    result = FundedPaths(paths.model, option.maturity, decisions, states, payments, paths.stream_id)
    _validate_paths(result, option)
    return result


def evaluate_embedded_policy(policy, sample):
    """Bound loss on the larger clock while the old policy stops by old expiry."""
    if sample.paths.model != policy.model:
        raise ValueError('Embedding requires the unchanged financial model')
    paths = restrict_paths(sample.paths, policy.option, policy.decisions)
    realised = realise_policy(policy, paths)
    lookup = {d:i for i,d in enumerate(sample.paths.decisions)}
    indices = np.array([lookup[d] for d in policy.decisions])[realised.stopping_indices]
    martingale = sample.values
    if martingale.shape != sample.paths.states.shape or not np.all(np.isfinite(martingale)):
        raise ValueError('Finite martingale values on the expanded clock required')
    rewards = np.column_stack([policy.model.reward(policy.option, d, sample.paths.states[:,j])
                               for j,d in enumerate(sample.paths.decisions)])
    upper = (rewards-martingale).max(axis=1)
    stopped = martingale[np.arange(len(indices)), indices]
    lower = realised.discounted_payoffs-stopped
    gap = upper-lower
    if np.any(gap < 0):
        raise ArithmeticError('Negative pathwise dual gap')
    lo, hi = estimate_samples(lower), estimate_samples(upper)
    for a in (lower, upper, gap):
        a.setflags(write=False)
    return DualEvaluation(lo, hi, estimate_samples(gap), estimate_samples(stopped),
        estimate_samples(martingale[:,-1]), lo.lower, hi.upper, lower, upper, gap), realised


def european_increment(model, option, terminal, extension):
    """Numerically evaluate the analytical upper bound on calendar value.

    The quadrature error field is a diagnostic, not a rounding certificate.
    The exact inequality uses exact European expectations, not their estimates.
    """
    check_extension(model, option, extension)
    base = european_reference(model, option, terminal)
    later = replace(option, maturity=option.maturity+extension)
    end = replace(terminal, time=later.maturity)
    expanded = european_reference(model, later, end)
    return dict(baseline=base, expanded=expanded,
        increment_quote=expanded.premium_quote-base.premium_quote,
        reference_error_diagnostic_quote=(base.quadrature_error_estimate+base.omitted_tail_bound+
            expanded.quadrature_error_estimate+expanded.omitted_tail_bound)/model.premium_factor,
        uniform_reward_bound_quote=option.strike*terminal.settlement_factor*
            erf(model.volatility*math.sqrt(extension)/(2*math.sqrt(2)))/model.premium_factor)
