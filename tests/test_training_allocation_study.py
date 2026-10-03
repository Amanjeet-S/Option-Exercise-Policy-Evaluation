"""Finite declared fixtures for registration and paired-allocation orchestration."""

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import study_training_allocation as study
from run_conditional_calibration import save,sha,read
from qf_research.optimal_stopping.funded_policy import simulate_paths,calibrate_control
from test_allocation_workload import fixture


def registration(tmp_path,monkeypatch):
    # Only declared mathematical inputs enter this temporary fixture root.
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'private',lambda p:Path(p).resolve())
    original_verify=study.verify_mapping
    monkeypatch.setattr(study,'verify_mapping',lambda mapping,base=None:
                        original_verify(mapping,tmp_path if base is None else base))
    monkeypatch.setattr(study,'validate_config',lambda cfg:None)
    for key in study.THREADS:monkeypatch.setenv(key,'1')
    base,_=fixture()
    paths=simulate_paths(base.model,base.option,base.decisions,200,np.random.default_rng(989))
    control=calibrate_control(base,paths)
    entry=dict(policy=asdict(base),fingerprint=base.fingerprint,control=asdict(control))
    cfg=dict(training_paths=[128,256],degree=2,minimum_itm_paths=32,condition_limit=1e8,
        approximation=dict(core_steps=30,wing_steps=6,tolerance=.02),
        pilot_repetitions=3,pilot_paths=128,timing_repetitions=3,evaluation_grid=[64,128],
        reference_repetitions=3,reference_paths=128,minimum_repetitions=3,maximum_repetitions=8,
        root_seed=982343,budget_ratio_limits=[.9,1.1],pilot_cpu_limit=120.,pilot_elapsed_limit=300.,
        main_cpu_limit=120.,main_elapsed_limit=300.,publication_clearance=False)
    folder=tmp_path/'registration';folder.mkdir()
    save(folder/'configuration.json',cfg)
    save(folder/'tasks.json',[dict(case=i,source_policy=entry,prior_streams=[base.training_stream_id,paths.stream_id]) for i in range(13)])
    (folder/'protocol.md').write_text('Declared mathematical fixture, not market observations.\n')
    save(folder/'manifest.json',dict(stage='pilot_design',input_sha256={},source_sha256={},
        artifact_sha256={n:sha(folder/n) for n in ['configuration.json','tasks.json','protocol.md']}))
    return folder,cfg


def test_complete_fixture_workflow_and_tamper_rejection(tmp_path,monkeypatch):
    reg,cfg=registration(tmp_path,monkeypatch)
    pilot=tmp_path/'pilot';study.run_pilot(reg,pilot)
    _,_,tasks,summaries,seen=study.verify_pilot(pilot)
    assert len(tasks)==len(summaries)==13
    assert read(pilot/'summary.json')['statistical_pairs']==39
    plan=tmp_path/'main_registration';study.allocate(pilot,plan)
    assert not (tmp_path/'main').exists()
    study.run_main(plan,tmp_path/'main')
    result=read(tmp_path/'main/summary.json');assert len(result['models'])==13
    for model in result['models']:
        for arm in ('reference','selected'):
            row=model['arms'][arm]
            assert row['inference']['estimate']['observations']==row['allocation']['repetitions']
            assert row['standalone_cpu_seconds']>row['setup']['cpu_seconds']
    # An extra or altered block cannot be accepted under the retained manifest.
    target=pilot/'blocks/case_000.json';target.write_text(target.read_text()+' ')
    rejected=tmp_path/'rejected'
    with pytest.raises(ValueError,match='checksum'):
        study.allocate(pilot,rejected)
    assert not rejected.exists()


def test_wrong_stage_and_changed_registration_fail_before_output(tmp_path,monkeypatch):
    reg,_=registration(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='stage'):
        study.run_main(reg,tmp_path/'invalid_main')
    assert not (tmp_path/'invalid_main').exists()
    (reg/'configuration.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        study.run_pilot(reg,tmp_path/'invalid_pilot')
    assert not (tmp_path/'invalid_pilot').exists()


def test_resource_limit_preserves_completed_checkpoints(tmp_path,monkeypatch):
    reg,_=registration(tmp_path,monkeypatch);calls=[]
    def interrupt_after_one(*args):
        calls.append(1)
        if len(calls)>1:raise RuntimeError('Declared fixture interruption')
    monkeypatch.setattr(study,'guard',interrupt_after_one)
    out=tmp_path/'partial'
    with pytest.raises(RuntimeError,match='interruption'):
        study.run_pilot(reg,out)
    assert (out/'checkpoints/case_000_stat_000.json').exists()
    assert not (out/'manifest.json').exists()


def test_completion_rejects_changed_registered_artifact(tmp_path,monkeypatch):
    reg,_=registration(tmp_path,monkeypatch)
    out=tmp_path/'execution';m,cfg,tasks,cpu,wall=study.launch(reg,out,'pilot')
    (reg/'tasks.json').write_text('[]')
    with pytest.raises(ValueError,match='checksum'):
        study.finish(reg,out,m,cpu,wall)
    assert not (out/'manifest.json').exists()
