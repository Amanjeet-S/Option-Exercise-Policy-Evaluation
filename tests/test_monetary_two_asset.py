"""Bounded fixtures for labelled training, exact decisions and sample storage."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qf_research.optimal_stopping import monetary_two_asset as monetary
spec=importlib.util.spec_from_file_location('prepare_monetary_two_asset_fixture',ROOT/'scripts/prepare_monetary_two_asset.py')
preparation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def fixed_policy(row, model='heterogeneous'):
    rows=((),tuple(float(x) for x in row))+tuple(() for _ in range(8))
    return monetary.FrozenTwoAssetPolicy(model,rows,'a'*64,100,tuple({} for _ in range(10)))


def test_labelled_cubic_features_do_not_erase_heterogeneous_asset_identity():
    values=monetary.labelled_basis(np.array([[120.,90.],[90.,120.]]))
    assert values.shape==(2,10)
    assert values[0,1]==pytest.approx(.2)
    assert values[0,2]==pytest.approx(-.1)
    assert values[1,1]==pytest.approx(-.1)
    assert values[1,2]==pytest.approx(.2)
    assert monetary.exact_model('heterogeneous')['volatilities']==['1/5','3/10']
    assert monetary.exact_model('homogeneous')['exercise_times'][7]=='7/3'


def test_ridge_handles_rank_deficiency_and_coefficient_point_bytes_are_frozen():
    states=np.full((128,10,2),110.);states[:,0]=100.
    policy=monetary.fit_policy(states,'homogeneous','b'*64)
    assert all(row['augmented_rank']==10 for row in policy.diagnostics[1:9])
    assert all(row['design_rank']==1 for row in policy.diagnostics[1:9])
    record=json.loads(json.dumps(policy.record(),allow_nan=False))
    rebuilt=monetary.policy_from_record(record,policy.coefficient_bytes())
    assert rebuilt.coefficients==policy.coefficients
    assert len(policy.coefficient_bytes())==800
    damaged=deepcopy(record);damaged['coefficients_hex'][1][0]=float(123).hex()
    with pytest.raises(ValueError,match='coefficient bytes'):
        monetary.policy_from_record(damaged)
    with pytest.raises(ValueError,match='coefficient file'):
        monetary.policy_from_record(record,b'0'*800)


def test_exact_oracle_respects_strict_ties_and_keeps_ambiguous_cells():
    policy=fixed_policy([.125]+[0.]*9)
    exact=((Fraction(125),Fraction(125)),(Fraction(90),Fraction(90)))
    assert monetary.interval_decision(policy,1,exact,(Fraction(1,2),Fraction(1,2)))['decision']=='continue'
    assert monetary.interval_decision(policy,1,exact,(Fraction(3,4),Fraction(3,4)))['decision']=='exercise'
    wide=((Fraction(124),Fraction(126)),exact[1])
    result=monetary.interval_decision(policy,1,wide,(Fraction(1,2),Fraction(1,2)))
    assert result['decision']=='unresolved'
    assert result['margin'][0]<0<result['margin'][1]
    assert monetary.interval_decision(policy,9,wide,(1,1))['decision']=='exercise'
    assert monetary.interval_decision(policy,2,wide,(1,1))['decision']=='continue'


def test_labelled_oracle_and_forward_screen_are_adapted():
    policy=fixed_policy([0.,1.]+[0.]*8)
    first=((120,120),(110,110));swapped=first[::-1]
    assert monetary.interval_decision(policy,1,first,(1,1))['decision']=='continue'
    assert monetary.interval_decision(policy,1,swapped,(1,1))['decision']=='exercise'
    states=np.full((2,10,2),100.);states[:,1]=(110.,120.)
    stops=monetary.diagnostic_stopping(policy,states)
    states[:,2:]=10000.
    assert np.array_equal(stops,monetary.diagnostic_stopping(policy,states))
    assert np.all(stops==1)


def test_preregistered_role_and_date_streams_do_not_overlap_or_get_sampled():
    identities=[]
    for model in monetary.MODEL_IDS:
        for role in preparation.ROLES:
            streams,records=preparation.construct_streams(model,role)
            assert all(value['generator']=='PCG64DXSM' and not value['sampled'] for value in records.values())
            assert records['3']['spawn_key']==[monetary.MODEL_IDS.index(model),preparation.ROLES.index(role),3]
            identities.extend(row['initial_state_sha256'] for row in records.values())
            assert preparation.state_record(streams[3])==records['3']['initial_state']
    assert len(set(identities))==54


def test_fixed_batches_replay_prefix_words_and_have_exact_constant_simulator_steps():
    streams,initial=preparation.construct_streams('heterogeneous','evaluation')
    first=preparation.generate_batch('heterogeneous',streams,4,role='evaluation')
    second=preparation.generate_batch('heterogeneous',streams,4,role='evaluation')
    replica,_=preparation.construct_streams('heterogeneous','evaluation')
    copy=preparation.generate_batch('heterogeneous',replica,4,role='evaluation')
    assert np.array_equal(first['uniform_prefixes'],copy['uniform_prefixes'])
    assert np.array_equal(first['states'],copy['states'])
    assert first['after']==second['before']
    assert first['before']['1']==initial['1']['initial_state']
    model=monetary.floating_model('heterogeneous')
    sigma=np.array(model.volatilities);dt=float(Fraction(1,3))
    expected=first['states'][:,7]*np.exp((.05-.1-.5*sigma*sigma)*dt+first['normals'][:,7]*(sigma*dt**.5))
    assert np.array_equal(expected,first['states'][:,8])
    extremes=preparation.midpoint_normals(np.array([[0,0],[(1<<64)-1,(1<<64)-1]],dtype=np.uint64))
    assert np.all(np.isfinite(extremes)) and extremes[0]==-extremes[1]


def test_registration_precedes_draws_and_every_fit_precedes_assessment(tmp_path):
    folder=tmp_path/'phase'
    registered=preparation.register(folder)
    assert registered['draws']==0
    assert not list(folder.rglob('*.bin'))
    with pytest.raises(ValueError,match='complete monetary assessment'):
        preparation.checked_assessment_design(folder)
    with pytest.raises(ValueError,match='Freeze every model'):
        preparation.prepare_role(folder,'homogeneous','evaluation',count=8,batch=4,fixture=True)
    for model in monetary.MODEL_IDS:
        result=preparation.prepare_role(folder,model,'training',count=128,batch=32,fixture=True)
        assert result['policy_fits_executed']==1
    before=hashlib_snapshot(folder)
    result=preparation.prepare_role(folder,'heterogeneous','evaluation',count=8,batch=4,fixture=True)
    assert result['monetary_inference_executed'] is False
    assert result['policy_fits_executed']==0
    target=folder/'heterogeneous'/'evaluation'
    assert (target/'states.bin').stat().st_size==8*160
    assert (target/'normals.bin').stat().st_size==8*144
    assert (target/'uniform_prefixes.bin').stat().st_size==8*288
    assert (target/'diagnostic_stops.bin').stat().st_size==8
    assert before==hashlib_snapshot(folder)
    with pytest.raises(ValueError,match='fresh role folder'):
        preparation.prepare_role(folder,'heterogeneous','evaluation',count=8,batch=4,fixture=True)


def test_registration_accepts_only_existing_tools_sources_and_never_overwrites(tmp_path):
    folder=tmp_path/'tools_only';(folder/'tools').mkdir(parents=True)
    source=folder/'tools'/'bridge.py';source.write_text('x = 1\n')
    preparation.register(folder)
    assert source.read_text()=='x = 1\n'
    with pytest.raises(ValueError,match='tools-only'):
        preparation.register(folder)
    rejected=tmp_path/'draws';rejected.mkdir();(rejected/'states.bin').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='tools-only'):
        preparation.register(rejected)
    assert (rejected/'states.bin').read_bytes()==b'preserve'
    rejected2=tmp_path/'bad_tools';(rejected2/'tools').mkdir(parents=True)
    (rejected2/'tools'/'draws.bin').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='tools-only'):
        preparation.register(rejected2)


def test_separate_assessment_seal_checks_sources_and_frozen_coefficients(tmp_path):
    folder=tmp_path/'phase';preparation.register(folder)
    for model in monetary.MODEL_IDS:
        preparation.prepare_role(folder,model,'training',count=128,batch=32,fixture=True)
    source='src/qf_research/optimal_stopping/monetary_two_asset.py'
    record=dict(schema_version=1,author='Amanjeet Singh',complete=True,protocol={'bounded_fixture':True},
        preparation_design_sha256=preparation.hash_file(folder/'design.json'),
        frozen_policy_sha256=hashlib_snapshot_by_model(folder),
        source_sha256={source:preparation.hash_file(ROOT/source)})
    preparation.write_new(folder/'assessment_design.json',record)
    assert len(preparation.checked_assessment_design(folder)['sha256'])==64
    record['source_sha256'][source]='0'*64
    (folder/'assessment_design.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='numerical source changed'):
        preparation.checked_assessment_design(folder)


def hashlib_snapshot_by_model(folder):
    return {model:preparation.hash_file(folder/model/'training'/'freeze.json') for model in monetary.MODEL_IDS}


def hashlib_snapshot(folder):
    return {str(path):preparation.hash_file(path) for path in folder.glob('*/training/freeze.json')}
