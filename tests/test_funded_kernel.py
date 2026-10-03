"""Independent operator identities and cross-language funded-price checks.

All inputs are declared mathematical fixtures; no market observations are used.
"""

import importlib.util
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from qf_research.computational.funded_kernel import NativeKernel, advance_scipy, prepare, price_prepared
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import (
    FundedCashModel, DividendCap, FundedExercise, finite_difference_price, european_reference,
)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not shutil.which(os.environ.get("CXX", "c++")):
        pytest.skip("Optional C++ compiler unavailable")
    path=Path(__file__).resolve().parents[1]/"scripts/build_funded_kernel.py"
    spec=importlib.util.spec_from_file_location("build_funded_kernel",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return NativeKernel(module.build(tmp_path_factory.mktemp("native_parent")/"build"))


def test_native_step_matches_independently_assembled_dense_equations(native):
    d=np.array([.01,2.,.3,12.]);initial=np.array([1.,3.,-2.,4.,5.,2.])
    h,w=.07,.63;left,right=.4,2.1
    a=w*h*d
    matrix=np.diag(1+2*a)+np.diag(-a[:-1],1)+np.diag(-a[1:],-1)
    explicit=np.diag(1-2*(1-w)*h*d)
    explicit+=np.diag((1-w)*h*d[:-1],1)+np.diag((1-w)*h*d[1:],-1)
    rhs=explicit@initial[1:-1]
    rhs[0]+=h*d[0]*((1-w)*initial[0]+w*left)
    rhs[-1]+=h*d[-1]*((1-w)*initial[-1]+w*right)
    expected=np.r_[left,np.linalg.solve(matrix,rhs),right]
    actual=native(d,initial,[h],[w],[left],[right])
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)
    np.testing.assert_array_equal(initial,[1.,3.,-2.,4.,5.,2.])


def test_linear_stationary_solution_over_irregular_steps(native):
    y=np.linspace(0,3,65);values=1+2*y
    result=native(.7*np.arange(1,64)**2,values,[.01,.03,.02],[1.,.5,.5],[1.]*3,[7.]*3)
    np.testing.assert_allclose(result,values,atol=3e-13,rtol=3e-13)


def test_quadratic_manufactured_solution_with_exact_discrete_growth(native):
    y=np.linspace(0,3,80);sigma=.3
    widths=np.array([.01,.02,.004]);weights=np.array([1.,.5,1.])
    growth=np.cumprod((1+(1-weights)*widths*sigma**2)/(1-weights*widths*sigma**2))
    result=native(.5*sigma**2*np.arange(1,79)**2,y*y,widths,weights,np.zeros(3),9*growth)
    np.testing.assert_allclose(result,y*y*growth[-1],atol=5e-13,rtol=5e-13)


def test_cn_is_not_implicitly_claimed_to_preserve_positivity(native):
    result=native([10.],[0.,1.,0.],[1.],[.5],[0.],[0.])
    assert result[1] == pytest.approx(-9/11)


@pytest.mark.parametrize("bad",["negative_diffusion","zero_width","weight","shape","nan","empty"])
def test_native_adapter_rejects_invalid_input(native,bad):
    args=[[1.],[0.,1.,0.],[.1],[.5],[0.],[0.]]
    if bad=="negative_diffusion":args[0]=[-1.]
    elif bad=="zero_width":args[2]=[0.]
    elif bad=="weight":args[3]=[.49]
    elif bad=="shape":args[1]=[0.,1.]
    elif bad=="nan":args[4]=[float('nan')]
    else:args[2]=[]
    with pytest.raises(ValueError):native(*args)


def test_overflow_is_reported_without_returning_a_price(native):
    with pytest.raises(FloatingPointError):native([1e308],[0.,1.,0.],[1e308],[1.],[0.],[0.])


@pytest.mark.parametrize("kind",["call","put"])
@pytest.mark.parametrize("case",["european","negative_funding","cash_phases","expiry_cash","zero_volatility","short_maturity"])
def test_prepared_backends_preserve_reference_price_and_event_counts(native,kind,case):
    maturity=.4 if case!="short_maturity" else 2/365
    vol=0. if case=="zero_volatility" else .27
    cash=(DividendCap(maturity/2,5.),)
    if case=="expiry_cash":cash=(DividendCap(maturity,7.),)
    if case in {"european","short_maturity"}:cash=()
    model=FundedCashModel(100.37,vol,1.0001,.9995,cash)
    option=Option(103.11,maturity,kind)
    sign=1 if case=="negative_funding" else -1
    rights=tuple(FundedExercise(t,1+sign*.03*t) for t in [maturity/4,maturity/2,maturity])
    if case in {"european","short_maturity"}:rights=rights[-1:]
    if case=="cash_phases":rights=(*rights,FundedExercise(maturity/2,.998,"before"),FundedExercise(0.,1.))
    if case=="expiry_cash":rights=(*rights,FundedExercise(maturity,.999,"before"))
    extra=(maturity*.137,maturity*.69)
    args=(model,option,401,93,430.,rights)
    original=finite_difference_price(*args,propagation_times=extra)
    prepared=prepare(*args,propagation_times=extra)
    for backend in (advance_scipy,native):
        result=price_prepared(prepared,backend)
        assert result.present_value==pytest.approx(original.present_value,abs=1e-10,rel=1e-11)
        assert result.premium_quote==pytest.approx(original.premium_quote,abs=1e-10,rel=1e-11)
        assert (result.time_intervals,result.decision_count,result.event_count)==(original.time_intervals,original.decision_count,original.event_count)


def test_native_european_convergence_to_independent_formula(native):
    # Keep payoff kink and evaluation state aligned to isolate the grid order.
    # Off-grid strikes can change interpolation-error coefficients on refinement.
    model=FundedCashModel(100.,.25,1.,1.)
    option=Option(100.,1.,"put");terminal=FundedExercise(1.,1.)
    exact=european_reference(model,option,terminal).premium_quote
    errors=[]
    for n in (400,800,1600):
        result=price_prepared(prepare(model,option,n,n//2,400.,(terminal,)),native)
        errors.append(abs(result.premium_quote-exact))
    assert errors[-1]<.0002 and errors[-1]<errors[0]/8


def test_nested_rights_on_common_propagation_grid(native):
    model=FundedCashModel(95.,.3,1.,1.,(DividendCap(.3,4.),))
    option=Option(100.,1.,"put")
    full=tuple(FundedExercise(t,1-.05*t) for t in (.1,.3,.5,.7,1.))
    values=[]
    for rights in (full[-1:],full[::2],full):
        p=prepare(model,option,800,500,400.,rights,propagation_times=tuple(d.time for d in full))
        values.append(price_prepared(p,native).premium_quote)
    assert values[0]<=values[1]+1e-10<=values[2]+2e-10
