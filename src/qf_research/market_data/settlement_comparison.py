"""Funding-path construction and matched accounting for conditional repricing."""
from dataclasses import dataclass, replace
from datetime import date
import math

from .funding_sensitivity import component_changes
from .settlement_funding import SettlementFunding
from .valuation_inputs import DiscountCurve
from ..optimal_stopping.contracts import Option
from ..optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise


@dataclass(frozen=True)
class RepricingInputs:
    """One development quote and its model; no training/holdout partition implied."""
    bid: float
    ask: float
    model: FundedCashModel
    option: Option
    decisions: tuple[FundedExercise, ...]

    def __post_init__(self):
        if any(isinstance(x, bool) or not math.isfinite(x) for x in (self.bid, self.ask)) or not 0 <= self.bid < self.ask:
            raise ValueError('Require finite non-negative bid and positive quote width')


def restore_problem(raw):
    model = dict(raw['model'])
    model['dividends'] = tuple(DividendCap(**e) for e in model['dividends'])
    return RepricingInputs(raw['bid'], raw['ask'], FundedCashModel(**model), Option(**raw['option']),
                           tuple(FundedExercise(**d) for d in raw['decisions']))


def spot_relative_curve(curve, spot_date):
    """Restrict and renormalise an existing log-linear curve without refitting."""
    if type(spot_date) is not date or not curve.anchor <= spot_date < curve.dates[-1]:
        raise ValueError('Swap spot must lie inside the retained curve domain')
    future = tuple(d for d in curve.dates if d > spot_date)
    return DiscountCurve(spot_date, (spot_date,) + future,
                         (1.,) + tuple(curve.between(spot_date, d) for d in future))


def funding_paths(curve, settlement_date, spot_date, candidate_bridge, nodes):
    """Construct declared paths; lower/upper names concern node factors only.

    Recalibrated option prices need not attain extrema at these two paths.
    """
    baseline = spot_relative_curve(curve, spot_date)
    paths = {'retained_lagged': SettlementFunding(settlement_date, baseline,
                curve.between(settlement_date, spot_date)),
             'front_replacement': SettlementFunding(settlement_date, baseline, candidate_bridge)}
    if not nodes:
        raise ValueError('Missing short-swap nodes')
    dates = (spot_date,) + tuple(date.fromisoformat(n['accrual_end']) for n in nodes)
    if any(not n['lower'] <= n['midpoint'] <= n['upper'] for n in nodes):
        raise ValueError('Reversed short-swap node factors')
    for side in ('lower', 'midpoint', 'upper'):
        term = DiscountCurve(spot_date, dates, (1.,) + tuple(n[side] for n in nodes))
        paths['hybrid_' + side] = SettlementFunding(settlement_date, term, candidate_bridge)
    return paths


def check_normalised_equivalence(full, normalised):
    """Check exact clocks and scaled factors before comparing monetary results."""
    if full.option != normalised.option or full.deliveries != normalised.deliveries:
        raise ValueError('Normalisation changed a contract or dated exercise right')
    scale = full.model.premium_factor
    expected = replace(full.model, initial_share_factor=full.model.initial_share_factor / scale,
        premium_factor=1., dividends=tuple(replace(e, discounted_cash=e.discounted_cash / scale)
                                          for e in full.model.dividends))
    actual = normalised.model
    numbers = [(expected.initial_share_factor, actual.initial_share_factor),
               (expected.premium_factor, actual.premium_factor)]
    if (expected.quoted_spot != actual.quoted_spot or expected.volatility != actual.volatility or
        len(expected.dividends) != len(actual.dividends) or len(full.decisions) != len(normalised.decisions)):
        raise ValueError('Normalisation changed state, volatility or event coverage')
    for e, a in zip(expected.dividends, actual.dividends):
        if e.ex_time != a.ex_time:
            raise ValueError('Normalisation changed a cash-event clock')
        numbers.append((e.discounted_cash, a.discounted_cash))
    for e, a in zip(full.decisions, normalised.decisions):
        if (e.time, e.phase) != (a.time, a.phase):
            raise ValueError('Normalisation changed the exercise calendar')
        numbers.append((e.settlement_factor / scale, a.settlement_factor))
    if any(not math.isclose(e, a, rel_tol=1e-13, abs_tol=1e-13) for e, a in numbers):
        raise ValueError('Normalised funding factors differ')
    return max(abs(e-a) for e, a in numbers)


def matched_changes(rows):
    """Pair only the same quote cutoff, strike and nominal cash rule."""
    lookup = {}
    for r in rows:
        key = (r['cutoff'], r['strike'], r['dividend_model'], r['variant'])
        if key in lookup:
            raise ValueError('Duplicate repricing scenario')
        lookup[key] = r
    comparisons = []
    pairs = [('front_component', 'retained_lagged', 'front_replacement'),
             ('term_component', 'front_replacement', 'hybrid_midpoint'),
             ('total', 'retained_lagged', 'hybrid_midpoint')]
    groups = sorted({key[:3] for key in lookup})
    for group in groups:
        for label, before, after in pairs:
            a, b = lookup.get(group + (before,)), lookup.get(group + (after,))
            row = dict(cutoff=group[0], strike=group[1], dividend_model=group[2], comparison=label)
            if a is None or b is None or 'decomposition' not in a or 'decomposition' not in b:
                comparisons.append(dict(row, status='unresolved_missing_valuation'))
                continue
            quote_keys = ('underlying_bid', 'underlying_ask', 'call_bid', 'call_ask', 'put_bid', 'put_ask')
            if any(a[k] != b[k] for k in quote_keys):
                raise ValueError('Matched comparison changed an observed quote')
            changes = component_changes(a['decomposition'], b['decomposition'])
            gap_change = b['lower_minus_put_ask'] - a['lower_minus_put_ask']
            if not math.isclose(gap_change, changes['lower_bound_change'], abs_tol=2e-12):
                raise ValueError('Gap and lower-bound changes disagree')
            comparisons.append(dict(row, status='computed',
                both_numerical_screens_pass=all(a['numerical_screens'].values()) and all(b['numerical_screens'].values()),
                gap_change=gap_change,
                volatility_lower_change=b['rectangle']['volatility_lower']-a['rectangle']['volatility_lower'],
                volatility_upper_change=b['rectangle']['volatility_upper']-a['rectangle']['volatility_upper'],
                accounting=changes))
    return comparisons
