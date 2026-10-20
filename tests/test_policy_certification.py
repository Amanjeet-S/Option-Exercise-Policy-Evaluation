"""Independent payoff, envelope, inference and stream checks for the decision study."""

from fractions import Fraction
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_policy_certification import (
    GapProcess, affine_truncated_mean, allocated_probability, empirical_upper, first_reward, gains,
    mean_gap_reference, parameter_check, rational_value, reference_lookup, simulate, stock_envelope, tail,
)
from validate_gap_coverage import numerical_oracle


@pytest.mark.parametrize('threshold',[100.,110.,130.,None])
@pytest.mark.parametrize('fraction',[0.,.5,1.])
def test_stopped_gain_and_global_envelope_against_direct_payoff(threshold,fraction):
    p = GapProcess()
    k = fraction*math.exp(-p.dividend_yield*p.maturity)
    rng = np.random.default_rng(7319)
    s1 = np.exp(rng.uniform(math.log(.01),math.log(1e5),4000))
    s2 = np.exp(rng.uniform(math.log(.01),math.log(1e5),4000))
    h1 = first_reward(s1,p)
    h2 = math.exp(-p.rate*p.maturity)*np.maximum(s2-p.strike,0.)
    m1 = k*(math.exp(-(p.rate-p.dividend_yield)*p.first_right)*s1-p.spot)
    m2 = k*(math.exp(-(p.rate-p.dividend_yield)*p.maturity)*s2-p.spot)
    exercise = np.zeros(len(s1),dtype=bool) if threshold is None else s1>threshold
    direct = np.maximum(h1-m1,h2-m2)-np.where(exercise,h1-m1,h2-m2)
    calculated = gains(s1,s2,p,threshold,'stock',k)
    np.testing.assert_allclose(calculated,direct,atol=3e-11,rtol=2e-12)
    env = stock_envelope(p,threshold,k)
    bound = env['constant']+env['scale']*s1 if threshold is None else \
        np.maximum(env['constant'],np.maximum(env['scale']*s2-env['shift'],0.))
    assert np.all(calculated<=bound+3e-10)
    if k == 0:
        np.testing.assert_array_equal(calculated,gains(s1,s2,p,threshold,'zero'))


@pytest.mark.parametrize('threshold',[100.,120.,None])
@pytest.mark.parametrize('cap',[.05,8.,80.])
def test_tail_formula_integrates_the_global_envelope(threshold,cap):
    p = GapProcess()
    k = .4
    env = stock_envelope(p,threshold,k)
    t = env['horizon']
    sd = p.volatility*math.sqrt(t)
    mu = math.log(p.spot)+(p.rate-p.dividend_yield-.5*p.volatility**2)*t
    def integrand(z):
        s = math.exp(mu+sd*z)
        bound = env['constant']+env['scale']*s if threshold is None else \
            max(env['constant'],max(env['scale']*s-env['shift'],0.))
        return max(bound-cap,0.)*math.exp(-z*z/2)/math.sqrt(2*math.pi)
    # Split at the envelope and clipping kinks rather than trusting quadrature
    # to discover an arbitrarily small far-tail region.
    splits = [-12.,12.]
    for level in (env['constant'],cap):
        state = (level+env['shift'])/env['scale'] if threshold is not None else \
            (level-env['constant'])/env['scale']
        if state>0:
            z = (math.log(state)-mu)/sd
            if -12<z<12:
                splits.append(z)
    splits = sorted(set(splits))
    expected = sum(quad(integrand,l,r,epsabs=1e-10)[0] for l,r in zip(splits,splits[1:]))
    assert tail(p,threshold,'stock',k,cap)==pytest.approx(expected,rel=2e-9,abs=1e-10)


def test_known_zero_gap_mean_reproduces_separate_conditional_call_identity():
    p = GapProcess()
    observed = mean_gap_reference(p,None,'zero',0.)
    oracle = numerical_oracle(p)
    assert observed['value']==pytest.approx(oracle['expected_gaps']['zero']['value'],abs=2e-9)
    assert observed['omitted_tail_upper']>=0
    assert observed['rigorous_enclosure'] is False


def test_affine_conditional_mean_matches_direct_normal_integration():
    p = GapProcess()
    s = 93.
    h = p.maturity-p.first_right
    sd = p.volatility*math.sqrt(h)
    mu = math.log(s)+(p.rate-p.dividend_yield-.5*p.volatility**2)*h
    for intercept,slope,lo,hi in [(20.,-.3,0.,100.),(-20.,.3,100.,math.inf),(2.,0.,0.,math.inf)]:
        zl = -12. if lo==0 else (math.log(lo)-mu)/sd
        zu = 12. if math.isinf(hi) else (math.log(hi)-mu)/sd
        root = -intercept/slope if slope else -1
        splits = [zl,zu]
        if root>0 and zl<(math.log(root)-mu)/sd<zu:
            splits.append((math.log(root)-mu)/sd)
        splits.sort()
        integrand = lambda z:max(intercept+slope*math.exp(mu+sd*z),0.)*math.exp(-z*z/2)/math.sqrt(2*math.pi)
        direct = sum(quad(integrand,l,r,epsabs=1e-10)[0] for l,r in zip(splits,splits[1:]))
        assert affine_truncated_mean(s,intercept,slope,lo,hi,p)==pytest.approx(direct,abs=2e-10)


def test_empirical_bernstein_constants_and_pairwise_variance():
    x = np.array([0.,.1,.3,1.])
    clipped = np.minimum(x,.5)
    variance = sum((a-b)**2 for i,a in enumerate(clipped) for b in clipped[i+1:])/(len(x)*(len(x)-1))
    alpha = .025/75
    expected = float(clipped.mean())+math.sqrt(2*variance*math.log(2/alpha)/4)+7*.5*math.log(2/alpha)/9+.2
    assert empirical_upper(x,.5,.2,alpha)==pytest.approx(expected)
    assert empirical_upper(np.zeros(4),0.,0.,alpha)==0.


def test_ties_continue_and_zero_yield_call_never_has_waiting_shortfall():
    p = GapProcess()
    s = np.array([110.])
    last = np.array([200.])
    assert gains(s,last,p,110.,'zero')[0]==0.
    zero = GapProcess(dividend_yield=0.)
    np.testing.assert_array_equal(gains(np.geomspace(.01,1e6,100),None,zero,None,'terminal_call'),0.)


def test_nested_stream_prefixes_reproduce_and_pilot_is_separate():
    p = GapProcess()
    def generator(key):
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(911,spawn_key=key)))
    one = simulate(generator((1,0,0)),generator((1,0,1)),32,p,terminal=True)
    a,b = generator((1,0,0)),generator((1,0,1))
    two = [simulate(a,b,n,p,terminal=True) for n in (8,24)]
    for j in (0,1):
        np.testing.assert_array_equal(one[j],np.concatenate([row[j] for row in two]))
    first_only,_ = simulate(generator((1,0,0)),generator((1,0,1)),32,p,terminal=False)
    np.testing.assert_array_equal(first_only,one[0])
    pilot,_ = simulate(generator((0,0)),generator((0,1)),32,p,terminal=True)
    assert not np.array_equal(pilot,one[0])


@pytest.mark.parametrize('x,cap,allowance,alpha',[
    ([-1.,0.],1.,0.,.1),([float('nan'),0.],1.,0.,.1),([0.],1.,0.,.1),
    ([0.,1.],-1.,0.,.1),([0.,1.],1.,-1.,.1),([0.,1.],1.,0.,0.),
])
def test_invalid_inference_inputs_are_rejected(x,cap,allowance,alpha):
    with pytest.raises(ValueError):
        empirical_upper(x,cap,allowance,alpha)


def test_authoritative_rational_endpoint_is_used_instead_of_display():
    assert rational_value(dict(numerator='1',denominator='3',approximation=.34))==Fraction(1,3)
    with pytest.raises(ValueError):
        parameter_check(GapProcess(),99.,0.)
    with pytest.raises(ValueError):
        parameter_check(GapProcess(),100.,.8)


def test_far_tail_positive_probability_is_not_lost_to_cdf_cancellation():
    from scipy.special import ndtr
    p = GapProcess()
    h = p.maturity-p.first_right
    mu = math.log(100.)+(p.rate-p.dividend_yield-.5*p.volatility**2)*h
    sd = p.volatility*math.sqrt(h)
    result = affine_truncated_mean(100.,1.,0.,math.exp(mu+9*sd),math.exp(mu+10*sd),p)
    assert result>0
    assert result==pytest.approx(float(ndtr(-9)-ndtr(-10)),rel=3e-13,abs=0.)


def test_finite_policy_changes_preserve_the_martingale_dual_slack():
    losses = [(.353488510414744,100.),(.000139441969724896,110.),
              (.161904571434292,120.),(.456311228484810,130.),(1.156991108872084,None)]
    p = GapProcess()
    for method,k in [('zero',0.),('stock',.3)]:
        slacks = [mean_gap_reference(p,theta,method,k)['value']-loss for loss,theta in losses]
        assert max(slacks)-min(slacks)<2e-9
    with pytest.raises(ValueError):
        mean_gap_reference(p,110.,'terminal_call',0.)


def test_different_exact_threshold_cannot_hide_in_a_rounded_label():
    from dataclasses import asdict
    def endpoint(x):
        f = Fraction.from_float(float(x))
        return dict(numerator=str(f.numerator),denominator=str(f.denominator),approximation=float(x))
    malformed = dict(ideal_model_population_loss_enclosed=True,
        model={key:endpoint(value) for key,value in asdict(GapProcess()).items()},
        shared_model_and_boundary_setup={},policies=[dict(threshold=endpoint(100.000001))])
    with pytest.raises(ValueError,match='exact policy threshold'):
        reference_lookup(malformed,GapProcess())


def test_family_probability_budget_holds_in_exact_binary_arithmetic():
    for count in (1,3,75,10001):
        alpha = allocated_probability(.025,count)
        assert Fraction.from_float(alpha)*count<=Fraction.from_float(.025)
        assert alpha>0
    for budget,members in [(.025,0),(.025,True),(0.,75),(float('nan'),75)]:
        with pytest.raises(ValueError):
            allocated_probability(budget,members)
