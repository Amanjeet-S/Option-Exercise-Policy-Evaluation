"""Deterministic mechanical and analytical fixtures, not empirical quote data."""

from copy import deepcopy
from dataclasses import replace

import pytest

from qf_research.market_data.conditional_calibration import (
    CalibrationQuote, select_october_pairs, scaled_midpoint_loss,
    fit_shared_volatility, quote_diagnostics,
)
from qf_research.optimal_stopping.contracts import Option, Market
from qf_research.optimal_stopping.funded_cash import FundedCashModel, FundedExercise
from qf_research.optimal_stopping.analytical import black_scholes


def mechanical_panel():
    cfg={'train_dates':['2026-09-10'],'heldout_dates':['2026-09-14'],'expiry':'2026-10-16',
         'strikes':[100.],'kinds':['put'],'age_threshold_seconds':5}
    refs=[{'RIC':'MECHANICAL_OPTION','ExpiryDate':'2026-10-16','StrikePrice':100.,'CallPutOption':'Put'}]
    events=[];pairs=[]
    for day in cfg['train_dates']+cfg['heldout_dates']:
        cutoff=day+'T19:45:00+00:00'
        for ric,bid,ask in [('MECHANICAL_OPTION',4.,6.),('MECHANICAL_UNDERLYING',99.,101.)]:
            events.append({'ric':ric,'cutoff_utc':cutoff,'quote_state_resolved':True,
                           'price_consensus':{'BID':bid,'ASK':ask}})
        pairs.append({'option_ric':'MECHANICAL_OPTION','underlying_ric':'MECHANICAL_UNDERLYING',
                      'cutoff_utc':cutoff,'price_and_age_checks':{'5':True},
                      'unique_state_and_age_checks':{'5':True},'empirical_acceptance':False})
    return {'events':events,'pairs':pairs},refs,cfg


def analytical_quote(strike=100., split='train', identity='case'):
    option=Option(strike,1.,'put')
    exact=black_scholes(Market(100.,0.,.2),option)
    return CalibrationQuote(identity,split,exact-.1,exact+.1,
                            FundedCashModel(100.,0.,1.,1.),option,(FundedExercise(1.,1.),))


def config():
    return {'volatility_bounds':[.01,.8],'search_grid_points':13,
            'volatility_tolerance':1e-9,'domain_multiple':4.}


def analytical_pricer(r,sigma,grid,domain):
    return black_scholes(Market(r.model.quoted_spot,0.,sigma),r.option)


def test_selection_keeps_split_and_original_acceptance_false():
    panel,refs,cfg=mechanical_panel();before=deepcopy(panel)
    selected,excluded=select_october_pairs(panel,refs,cfg)
    assert len(selected)==2 and not excluded
    assert {r['split'] for r in selected}=={'train','heldout'}
    assert all(r['original_empirical_acceptance'] is False for r in selected)
    assert panel==before


@pytest.mark.parametrize('failure', ['timing','tied','locked','outside'])
def test_every_failed_candidate_has_an_explicit_exclusion(failure):
    panel,refs,cfg=mechanical_panel()
    # Add a second training date so the failed row does not empty its split.
    cfg['train_dates'].append('2026-09-11')
    extra=deepcopy(panel['pairs'][0]);extra['cutoff_utc']='2026-09-11T19:45:00+00:00';panel['pairs'].append(extra)
    for e in panel['events'][:2]:
        e=deepcopy(e);e['cutoff_utc']=extra['cutoff_utc'];panel['events'].append(e)
    if failure=='timing':
        panel['pairs'][0]['price_and_age_checks']['5']=False
        panel['pairs'][0]['unique_state_and_age_checks']['5']=False
    elif failure=='tied':
        panel['events'][0]['quote_state_resolved']=False
        panel['pairs'][0]['unique_state_and_age_checks']['5']=False
    elif failure=='locked':panel['events'][0]['price_consensus']['ASK']=4.
    else:
        r=deepcopy(refs[0]);r['RIC']='OUTSIDE';r['ExpiryDate']='2026-12-18';refs.append(r)
        extra=deepcopy(panel['pairs'][0]);extra['option_ric']='OUTSIDE';panel['pairs'].append(extra)
    selected,excluded=select_october_pairs(panel,refs,cfg)
    assert len(excluded)==1 and excluded[0]['reasons']
    assert len(selected)+len(excluded)==len(panel['pairs'])


@pytest.mark.parametrize('change', ['overlap','future_train','duplicate_pair','duplicate_event','duplicate_reference'])
def test_rejects_ambiguous_splits_and_duplicate_evidence(change):
    panel,refs,cfg=mechanical_panel()
    if change=='overlap':cfg['train_dates']+=cfg['heldout_dates']
    elif change=='future_train':cfg['train_dates']=['2026-09-16']
    elif change=='duplicate_pair':panel['pairs'].append(panel['pairs'][0])
    elif change=='duplicate_event':panel['events'].append(panel['events'][0])
    else:refs.append(refs[0])
    with pytest.raises(ValueError):select_october_pairs(panel,refs,cfg)


def test_midpoint_scaling_and_interval_distance_are_distinct():
    r=replace(analytical_quote(),bid=4.,ask=6.)
    assert scaled_midpoint_loss([r],[5.5])==.25
    d=quote_diagnostics(r,5.5)
    assert d['inside_quote_interval'] and d['outside_quote_distance']==0.
    d=quote_diagnostics(r,7.)
    assert d['midpoint_residual']==2. and d['outside_quote_distance']==1.
    assert d['outside_quote_distance_in_full_spreads']==.5


def test_recovers_common_analytical_volatility_without_using_heldout_rows():
    rows=[analytical_quote(strike=k,identity=str(k)) for k in (90.,100.,110.)]
    result=fit_shared_volatility(rows,config(),{},pricer=analytical_pricer)
    assert result['best']['volatility']==pytest.approx(.2,abs=1e-7)
    assert result['best']['loss']<1e-8
    assert result['training_count']==3 and result['all_local_searches_converged']
    assert not result['boundary_selected'] and not result['global_optimality_certified']
    assert all(r.model.volatility==0. for r in rows)
    with pytest.raises(ValueError,match='training observations only'):
        fit_shared_volatility(rows+[analytical_quote(split='heldout',identity='later')],config(),{},pricer=analytical_pricer)


def test_retains_parameter_boundary_and_flat_objective():
    r=analytical_quote()
    result=fit_shared_volatility([r],config(),{},pricer=lambda row,*_:row.midpoint)
    assert result['boundary_selected']
    assert result['best']['volatility']==config()['volatility_bounds'][0]
    assert all(s['loss']==0. for s in result['scan'])


@pytest.mark.parametrize('bid,ask', [(1.,1.),(2.,1.),(-1.,1.),(float('nan'),2.),(1.,float('inf'))])
def test_rejects_unweightable_or_invalid_quote(bid,ask):
    with pytest.raises(ValueError):replace(analytical_quote(),bid=bid,ask=ask)


@pytest.mark.parametrize('price', [-1.,float('nan'),float('inf')])
def test_rejects_invalid_pricer_result(price):
    with pytest.raises(ValueError):scaled_midpoint_loss([analytical_quote()],[price])


def test_rejects_repeated_training_record():
    r=analytical_quote()
    with pytest.raises(ValueError,match='Duplicate training'):
        fit_shared_volatility([r,r],config(),{},pricer=analytical_pricer)
