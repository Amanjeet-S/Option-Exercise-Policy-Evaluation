"""Sufficient redundancy test for additional post-event exercise rights."""
from bisect import bisect_right

from .funded_cash import FundedExercise


def same_settlement_redundancy(model, baseline, expanded):
    """Map each added right to an unchanged later right without a cash event.

    With identical settlement factors and no intervening cash transfer, the
    transformed state is a martingale and both exercise rewards are convex.
    Conditional Jensen therefore dominates immediate exercise by waiting.
    Failure of this sufficient check is inconclusive, not a price prediction.
    """
    for rights in (baseline,expanded):
        if not isinstance(rights,tuple) or not rights or any(not isinstance(d,FundedExercise) or d.phase!='after' for d in rights):
            raise ValueError('Nonempty tuples of post-event rights required')
        if any(a.time>=b.time for a,b in zip(rights,rights[1:])):
            raise ValueError('Strictly increasing exercise times required')
    if not set(baseline).issubset(set(expanded)) or baseline[-1]!=expanded[-1]:
        raise ValueError('Preserve every original right and the terminal right')
    times=[d.time for d in baseline];original=set(baseline);checks=[]
    for right in expanded:
        if right in original:continue
        later=baseline[bisect_right(times,right.time)]
        same=right.settlement_factor==later.settlement_factor
        clear=not any(e.discounted_cash>0 and right.time<e.ex_time<=later.time for e in model.dividends)
        checks.append(dict(added_time=right.time,retained_time=later.time,
            same_settlement_factor=same,no_intervening_cash_transfer=clear,
            sufficient_condition_satisfied=same and clear))
    return dict(added_rights=len(checks),checks=checks,
        sufficient_condition_satisfied=all(c['sufficient_condition_satisfied'] for c in checks),
        applies_to=['call','put'],numerical_price_tolerance_used=False)
