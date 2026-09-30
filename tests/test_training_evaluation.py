"""Analytical fixtures for crossed training/evaluation inference and execution."""

from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import ttest_rel

from qf_research.optimal_stopping.training_evaluation import paired_difference, policy_gain, replicate_summary
from qf_research.optimal_stopping.simulation import estimate_samples
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise, DividendCap
from qf_research.optimal_stopping.funded_policy import simulate_paths, fit_policy, calibrate_control

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from study_training_evaluation import execute_block, numerical_record, summarise, seed
from audit_training_evaluation import check_pair


def test_paired_covariance_and_interval_match_independent_reference():
    x=np.array([1.,4.,9.,16.,25.]);y=np.array([.5,3.,8.,12.,27.])
    r=paired_difference(x,y);reference=ttest_rel(x,y).confidence_interval(.95)
    assert r['estimate']['lower']==pytest.approx(reference.low)
    assert r['estimate']['upper']==pytest.approx(reference.high)
    check_pair(r,len(x))
    assert r['estimate']['standard_error']**2 < (x.var(ddof=1)+y.var(ddof=1))/len(x)


def test_common_upper_identifies_gain_and_rejects_changed_benchmark():
    upper=np.array([5.,6.,9.,8.]);lo=np.array([2.,3.,4.,2.]);hi=np.array([3.,2.,6.,1.])
    a=SimpleNamespace(upper_payoffs=upper,adjusted_policy_payoffs=lo,gap_payoffs=upper-lo)
    b=SimpleNamespace(upper_payoffs=upper.copy(),adjusted_policy_payoffs=hi,gap_payoffs=upper-hi)
    r=policy_gain(a,b,2.)
    assert r['estimate']['value']==pytest.approx(((hi-lo)/2).mean())
    b.upper_payoffs[0]+=1
    with pytest.raises(ValueError):policy_gain(a,b,2.)
    with pytest.raises(ValueError):policy_gain(a,a,0.)


def test_repetition_inference_retains_negative_variance_component():
    es=[dict(value=x,standard_error=1.,observations=100) for x in [1.,1.1,.9,1.]]
    r=replicate_summary(es)
    assert r['estimate']==asdict(estimate_samples(np.array([e['value'] for e in es])))
    assert r['training_variance_moment_estimate']<0
    assert r['evaluation_variance_of_grand_mean']==.25
    with pytest.raises(ValueError):replicate_summary(es[:1])


@pytest.mark.parametrize('x,y',[([1.],[1.]),([1.,2.],[1.]),([1.,np.nan],[1.,2.]),([[1.,2.]],[[2.,3.]])])
def test_invalid_paired_samples_rejected(x,y):
    with pytest.raises(ValueError):paired_difference(x,y)


def fixture():
    model=FundedCashModel(100.,.25,1.,.98,(DividendCap(.4,3.),))
    option=Option(100.,1.,'put')
    rights=(FundedExercise(.2,.99),FundedExercise(.4,.98,'before'),FundedExercise(1.,.96))
    paths=simulate_paths(model,option,rights,500,np.random.default_rng(821))
    policy=fit_policy(option,paths,2)
    pilot=simulate_paths(model,option,rights,500,np.random.default_rng(822))
    control=calibrate_control(policy,pilot)
    task=dict(case=0,source_policy=dict(policy=asdict(policy),fingerprint=policy.fingerprint,control=asdict(control)),
              prior_streams=[paths.stream_id,pilot.stream_id])
    c=dict(training_paths=[300,600],evaluation_paths=[400,800],degree=2,training_repetitions=2,
           regression=dict(minimum_itm_paths=32,condition_limit=1e8),root_seed=823,
           approximations=dict(coarse=dict(core_steps=30,wing_steps=8,tolerance=.02),
                               fine=dict(core_steps=60,wing_steps=8,tolerance=.01)))
    return task,c


def test_complete_crossed_block_pairing_coverage_and_reproduction():
    task,c=fixture();first=execute_block(task,c,0);again=execute_block(task,c,0)
    assert numerical_record(first)==numerical_record(again)
    assert len(first['workloads'])==8
    for size in c['evaluation_paths']:
        selected=[r for r in first['workloads'] if r['evaluation_paths']==size]
        assert len({r['paths_sha256'] for r in selected})==1
        for name in ('policy_gains','evaluation_effects','gain_consistency'):
            for r in first[name]:check_pair(r,r['evaluation_paths'])
    assert len({r['stream']['stream_id'] for r in first['workloads']})==2
    assert {r['stream']['stream_id'] for r in first['workloads']}.isdisjoint(
        {r['stream']['stream_id'] for r in first['training']})
    for r in first['workloads']:
        trained=next(t for t in first['training'] if t['training_paths']==r['training_paths'])
        assert r['policy_fingerprint']==trained['fingerprint']
        assert r['total_elapsed_seconds']==r['elapsed_seconds']+trained['elapsed_seconds']
    second=execute_block(task,c,1)
    summary=summarise([first,second],c,[0])
    assert summary['workloads']==16 and not summary['equal_time_design']
    with pytest.raises(ValueError):summarise([first],c,[0])
    with pytest.raises(ValueError):summarise([first,first],c,[0])


def test_seed_roles_are_distinct_and_evaluation_is_deliberately_shared():
    _,c=fixture()
    states=[np.random.PCG64(seed(c,role,case,rep,index)).random_raw()
            for role in (1,2) for case in (0,1) for rep in (0,1) for index in (0,1)]
    assert len(set(states))==len(states)
