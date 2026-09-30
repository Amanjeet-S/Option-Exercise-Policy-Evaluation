"""Conditional short-OIS discount sets with an explicit, unidentified front end.

Historical quote numbers are mapped to percentage par rates by a maintained
convention. Deterministic single-curve accumulation and coincident leg payment
dates are required. The construction is not a stochastic OIS bootstrap.
"""

from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime
import math

from .valuation_inputs import DiscountCurve


def _day(value):
    if type(value) is not date:
        raise ValueError('Require a date without a time component')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class OisSchedule:
    ric: str
    trade_date: date
    start: date
    end: date
    payment: date
    displayed_end: date

    def __post_init__(self):
        for d in (self.trade_date, self.start, self.end, self.payment, self.displayed_end):
            _day(d)
        if not self.ric or not self.trade_date < self.start < self.end <= self.payment:
            raise ValueError('Require a future, single-period accrual and non-early payment')

    @property
    def alpha(self):
        return (self.end - self.start).days / 360


def inspect_contracts(payload, observation_date, rics):
    """Resolve accrual dates from cash flows, never from displayed maturity.

    Provider-calculated rates/amounts check schedule arithmetic only. They are
    not historical observed quotes. Current template use does not verify its
    historical vintage or the mapping from OTC contributions to these terms.
    """
    _day(observation_date)
    names = [h['name'] for h in payload['headers']]
    if len(names) != len(set(names)) or len(set(rics)) != len(rics):
        raise ValueError('Duplicate columns or requested instruments')
    groups = {ric: [] for ric in rics}
    rows = payload['data']
    statuses = payload.get('statuses')
    if statuses is None or len(statuses) != len(rows):
        raise ValueError('Missing field-status evidence')
    for cells, status in zip(rows, statuses):
        if len(cells) != len(names) or len(status) != len(names) or 4 in status:
            raise ValueError('Malformed row or provider field error')
        row = dict(zip(names, cells))
        if row.get('ErrorCode') or row.get('ErrorMessage'):
            raise ValueError('Provider contract error')
        if row['InstrumentTag'] not in groups:
            raise ValueError('Unexpected instrument')
        groups[row['InstrumentTag']].append(row)
    result, diagnostics = [], []
    def day(value):
        # Provider date fields are not intraday quote clocks.
        return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
    for ric, legs in groups.items():
        if len(legs) != 2 or {r['InterestType'] for r in legs} != {'Fixed', 'Float'}:
            raise ValueError('Require exactly one fixed and one floating leg')
        if {r['LegIndex'] for r in legs} != {0, 1}:
            raise ValueError('Ambiguous leg indices')
        events = []
        for row in legs:
            expected = dict(InterestCalculationMethod='Dcb_Actual_360',
                InterestPaymentFrequency='Annual', InterestPaymentDelay=2,
                PaymentBusinessDayConvention='NextBusinessDay', PaymentBusinessDays='USA',
                PaymentRollConvention='Same')
            if any(row.get(k) != v for k, v in expected.items()):
                raise ValueError('Provider convention differs from the reviewed scope')
            if day(row['TradeDate']) != observation_date:
                raise ValueError('Trade date differs')
            cash = row['CashFlows']
            if cash['instrumentType'] != 'Swap' or len(cash['payments']) != 1:
                raise ValueError('Require one cash flow on each leg')
            pay = cash['payments'][0]
            if pay['event'] != 'Interest' or pay['type'] != row['InterestType'] or pay['currency'] != 'USD':
                raise ValueError('Unexpected cash-flow type or currency')
            s = OisSchedule(ric, observation_date, day(pay['startDate']), day(pay['endDate']),
                            day(pay['date']), day(row['EndDate']))
            if day(row['StartDate']) != s.start:
                raise ValueError('Displayed and accrual starts differ')
            n, k, amount = row['NotionalAmount'], pay['annualRatePercent'], pay['amount']
            if not all(_number(x) for x in (n, k, amount)) or n <= 0:
                raise ValueError('Invalid provider cash-flow numbers')
            expected_amount = n * k / 100 * s.alpha
            sign = -1 if row['InterestType'] == 'Fixed' else 1
            if abs(amount - sign * expected_amount) > 1e-8 * max(1., abs(expected_amount)):
                raise ValueError('ACT/360 cash-flow amount does not reconcile')
            if row['InterestType'] == 'Fixed':
                if not _number(row['FixedRatePercent']) or not math.isclose(row['FixedRatePercent'], k, abs_tol=1e-12):
                    raise ValueError('Fixed coupon differs from cash-flow rate')
            else:
                if any(row.get(k) != v for k, v in dict(IndexName='SOFR', IndexTenor='ON',
                    IndexFixingRic='USDSOFR=', IndexFixingLag=1, IndexLockoutPeriod=0).items()):
                    raise ValueError('Floating convention differs')
            events.append(s)
        if events[0] != events[1]:
            raise ValueError('Fixed and floating accrual or payment dates differ')
        result.append(events[0])
        diagnostics.append(dict(ric=ric, accrual_days=round(events[0].alpha * 360),
            displayed_maturity_difference_days=(events[0].displayed_end - events[0].end).days,
            current_provider_schedule_resolved=True, historical_quote_mapping_verified=False))
    if len({r.start for r in result}) != 1:
        raise ValueError('Short swaps do not share a common start')
    return tuple(sorted(result, key=lambda r: r.end)), diagnostics


@dataclass(frozen=True)
class DiscountBand:
    lower: DiscountCurve
    midpoint: DiscountCurve
    upper: DiscountCurve

    def __post_init__(self):
        if not (self.lower.dates == self.midpoint.dates == self.upper.dates):
            raise ValueError('Discount bands require identical date grids')
        if any(not a <= b <= c for a, b, c in zip(
                self.lower.factors, self.midpoint.factors, self.upper.factors)):
            raise ValueError('Reversed factor interval')

    def _weights(self, target):
        _day(target)
        ds = self.lower.dates
        if not ds[0] <= target <= ds[-1]:
            raise ValueError('Unidentified front end or prohibited extrapolation')
        i = bisect_left(ds, target)
        w = [0.] * len(ds)
        if ds[i] == target:
            w[i] = 1.
        else:
            t = (target - ds[i-1]).days / (ds[i] - ds[i-1]).days
            w[i-1], w[i] = 1-t, t
        return w

    def ratio_bounds(self, start, end):
        """Exact extrema over a rectangular set of log-linear node factors.

        The two discount factors share nodes. Dividing separate marginal extrema
        generally gives a wider interval and is not the implemented calculation.
        """
        if end < start:
            raise ValueError('Discounting end precedes start')
        coefficients = [b-a for a, b in zip(self._weights(start), self._weights(end))]
        lo, hi = map(lambda c: tuple(map(math.log, c.factors)), (self.lower, self.upper))
        bottom = math.fsum(c * (l if c >= 0 else h) for c, l, h in zip(coefficients, lo, hi))
        top = math.fsum(c * (h if c >= 0 else l) for c, l, h in zip(coefficients, lo, hi))
        return math.exp(bottom), math.exp(top)


def build_spot_band(schedules, quotes):
    """Fit all short-swap intervals under the stated deterministic convention.

    A common delayed payment discounts both legs equally and cancels at par.
    Each quote therefore restricts D(start)/D(end) - 1 to alpha*[bid, ask]/100.
    The unidentified observation-to-spot discount is not assigned a number.
    """
    schedules = tuple(sorted(schedules, key=lambda r: r.end))
    if not schedules or len({s.ric for s in schedules}) != len(schedules):
        raise ValueError('Need unique short-swap instruments')
    if set(quotes) != {s.ric for s in schedules} or len({s.start for s in schedules}) != 1:
        raise ValueError('Quote coverage or common spot start differs')
    start = schedules[0].start
    if len({s.trade_date for s in schedules}) != 1:
        raise ValueError('Mixed observation dates')
    ds, lower, mid, upper = [start], [1.], [1.], [1.]
    for s in schedules:
        bid, ask = quotes[s.ric]
        if not all(_number(x) for x in (bid, ask)) or bid > ask:
            raise ValueError('Invalid par-rate interval')
        denominator = [1 + s.alpha * k / 100 for k in (ask, (bid+ask)/2, bid)]
        if min(denominator) <= 0 or not all(math.isfinite(x) for x in denominator):
            raise ValueError('Par quote implies non-positive accumulation')
        ds.append(s.end)
        for values, x in zip((lower, mid, upper), denominator):
            values.append(1/x)
    return DiscountBand(*(DiscountCurve(start, tuple(ds), tuple(v)) for v in (lower, mid, upper)))
