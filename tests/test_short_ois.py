"""Mathematical and record-layout fixtures, not simulated empirical observations."""
from copy import deepcopy
from datetime import date, timedelta
from itertools import product
import math

import pytest

from qf_research.market_data.short_ois import OisSchedule, inspect_contracts, build_spot_band
from qf_research.market_data.valuation_inputs import DiscountCurve


START = date(2024, 1, 4)


def schedule(i, days, *, display_extra=0):
    end = START + timedelta(days=days)
    return OisSchedule(str(i), date(2024,1,2), START, end, end+timedelta(days=2),
                       end+timedelta(days=display_extra))


def test_single_period_negative_and_zero_par_rates_reprice():
    ss = [schedule(1,7), schedule(2,30), schedule(3,90)]
    q = {'1': (-1.,-0.5), '2': (0.,0.), '3': (2.,3.)}
    b = build_spot_band(ss,q)
    assert b.lower.factors[1] > 1 and b.midpoint.factors[2] == 1
    for s in ss:
        for curve,k in [(b.lower,q[s.ric][1]), (b.upper,q[s.ric][0]),
                        (b.midpoint,sum(q[s.ric])/2)]:
            assert (1/curve.discount(s.end)-1)/s.alpha*100 == pytest.approx(k, abs=2e-12)


def test_interval_ratios_equal_vertex_extrema_with_shared_interpolation_nodes():
    ss = [schedule(1,7), schedule(2,30), schedule(3,90)]
    b = build_spot_band(ss, {'1':(1.,3.),'2':(2.,5.),'3':(3.,7.)})
    # Both dates share interpolation nodes, making independent marginal division wrong.
    a, z = START+timedelta(days=40), START+timedelta(days=60)
    values=[]
    for factors in product(*zip(b.lower.factors[1:],b.upper.factors[1:])):
        c=DiscountCurve(START,b.lower.dates,(1.,)+factors)
        values.append(c.between(a,z))
    lo,hi=b.ratio_bounds(a,z)
    assert lo == pytest.approx(min(values), abs=3e-15)
    assert hi == pytest.approx(max(values), abs=3e-15)
    marginal_lo=b.lower.discount(z)/b.upper.discount(a)
    marginal_hi=b.upper.discount(z)/b.lower.discount(a)
    assert lo > marginal_lo and hi < marginal_hi
    assert b.ratio_bounds(a,a) == (1.,1.)


def test_payment_lag_cancels_only_as_a_common_deterministic_factor():
    ss=[schedule(1,30)]
    b=build_spot_band(ss,{'1':(4.,4.)})
    alpha=ss[0].alpha
    accumulation=1/b.midpoint.factors[1]
    for payment_discount in (0.8,1.,1.2):
        assert payment_discount*(accumulation-1-0.04*alpha) == pytest.approx(0., abs=3e-16)
    # Separate leg payment discounts do not cancel.
    assert abs(0.8*(accumulation-1)-0.9*0.04*alpha) > 1e-4


def test_spot_scaling_leaves_ratios_unidentified_in_absolute_levels():
    b=build_spot_band([schedule(1,30)],{'1':(4.,4.1)})
    g=b.midpoint.factors[-1]
    assert 0.99*g != g
    assert (0.99*g)/0.99 == pytest.approx(g)
    with pytest.raises(ValueError,match='front end'):
        b.ratio_bounds(START-timedelta(days=1),START+timedelta(days=7))
    with pytest.raises(ValueError,match='extrapolation'):
        b.ratio_bounds(START,START+timedelta(days=31))


@pytest.mark.parametrize('q',[{'1':(4.,3.)},{'1':(True,4.)},{'1':(0.,math.nan)},
                              {'1':(-6000.,-5000.)},{'2':(3.,4.)}])
def test_invalid_quotes_do_not_generate_discount_inputs(q):
    with pytest.raises(ValueError):
        build_spot_band([schedule(1,7)],q)


def test_duplicate_maturities_and_mixed_dates_are_rejected():
    a,b=schedule(1,30),schedule(2,30)
    with pytest.raises(ValueError):
        build_spot_band([a,b],{'1':(2.,3.),'2':(2.,3.)})
    b=OisSchedule('2',date(2024,1,3),START,START+timedelta(days=60),START+timedelta(days=62),START+timedelta(days=60))
    with pytest.raises(ValueError,match='Mixed'):
        build_spot_band([a,b],{'1':(2.,3.),'2':(2.,3.)})


def payload():
    # Declared layout fixture with displayed maturity differing from coupon end.
    rows=[]
    for i,kind in enumerate(('Fixed','Float')):
        r=dict(InstrumentTag='TEST',LegIndex=i,TradeDate='2024-01-02',StartDate='2024-01-04',
            EndDate='2024-02-05',InterestType=kind,InterestCalculationMethod='Dcb_Actual_360',
            InterestPaymentFrequency='Annual',InterestPaymentDelay=2,
            PaymentBusinessDayConvention='NextBusinessDay',PaymentBusinessDays='USA',
            PaymentRollConvention='Same',NotionalAmount=1000.,FixedRatePercent=3.6 if i==0 else None,
            IndexName=None if i==0 else 'SOFR',IndexTenor=None if i==0 else 'ON',
            IndexFixingRic=None if i==0 else 'USDSOFR=',IndexFixingLag=None if i==0 else 1,
            IndexLockoutPeriod=None if i==0 else 0,ErrorCode='',ErrorMessage='')
        r['CashFlows']=dict(instrumentType='Swap',payments=[dict(startDate='2024-01-04',endDate='2024-02-03',
            date='2024-02-06',annualRatePercent=3.6,amount=-3. if i==0 else 3.,
            event='Interest',type=kind,currency='USD')])
        rows.append(r)
    names=list(rows[0])
    return dict(headers=[dict(name=n) for n in names],data=[[r[n] for n in names] for r in rows],
                statuses=[[3]*len(names) for _ in rows])


def alter(p,row,field,value):
    names=[h['name'] for h in p['headers']]
    p['data'][row][names.index(field)]=value


def test_provider_cashflow_dates_take_precedence_over_displayed_maturity():
    p=payload(); original=deepcopy(p)
    ss,ds=inspect_contracts(p,date(2024,1,2),['TEST'])
    assert ss[0].alpha == 30/360 and ss[0].displayed_end != ss[0].end
    assert ds[0]['displayed_maturity_difference_days']==2 and p==original


def test_amount_reconciliation_detects_wrong_accrual_end():
    p=payload();names=[h['name'] for h in p['headers']]
    p['data'][0][names.index('CashFlows')]['payments'][0]['endDate']='2024-02-05'
    with pytest.raises(ValueError,match='reconcile'):
        inspect_contracts(p,date(2024,1,2),['TEST'])


def test_distinct_leg_payment_dates_cannot_use_common_payment_identity():
    p=payload();names=[h['name'] for h in p['headers']]
    p['data'][1][names.index('CashFlows')]['payments'][0]['date']='2024-02-07'
    with pytest.raises(ValueError,match='dates differ'):
        inspect_contracts(p,date(2024,1,2),['TEST'])


def test_provider_errors_missing_legs_and_changed_conventions_are_rejected():
    for kind in ('status','leg','convention','date'):
        p=payload()
        if kind=='status':p['statuses'][0][0]=4
        elif kind=='leg':p['data'].pop();p['statuses'].pop()
        elif kind=='convention':alter(p,0,'InterestPaymentDelay',0)
        else:alter(p,0,'TradeDate','2024-01-03')
        with pytest.raises(ValueError):inspect_contracts(p,date(2024,1,2),['TEST'])
