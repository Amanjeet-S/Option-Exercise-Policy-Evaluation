"""Reconstruct joint-family inference, financial diagnostics and frozen policies."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import math

import numpy as np
from scipy.stats import t

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_joint_input_expiry import evaluate, numerical_record, summarise, YEAR
from study_policy_input_sensitivity import restore_scenario
from validate_funded_dual import load_frozen
from qf_research.optimal_stopping.funded_policy import _digest
from qf_research.optimal_stopping.expiry_extension import extended_calendar, european_increment
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=3e-12, atol=2e-12)


def check_estimate(e, count):
    if e['observations'] != count or e['confidence'] != .95 or e['standard_error'] < 0:
        raise ValueError('Incorrect estimate sample or confidence')
    radius = t.ppf(.975, count-1)*e['standard_error']
    close([e['lower'],e['upper']], [e['value']-radius,e['value']+radius])


def check_difference(r, left, right):
    # r estimates left minus right; left and right are independent mean records.
    v = left['standard_error']**2+right['standard_error']**2
    value = left['value']-right['value']
    if v:
        df = v*v/sum(e['standard_error']**4/(e['observations']-1) for e in (left,right))
        close(r['degrees_of_freedom'],df)
        radius=t.ppf(.975,df)*math.sqrt(v)
    else:
        if r['degrees_of_freedom'] is not None: raise ValueError('Zero-variance degrees of freedom differs')
        radius=0.
    close([r['value'],r['standard_error'],r['lower'],r['upper']], [value,math.sqrt(v),value-radius,value+radius])
    if r['confidence'] != .95: raise ValueError('Wrong difference confidence')


def quote(e, factor):
    return dict(e, value=e['value']/factor, standard_error=e['standard_error']/factor)


def audit(folder, output):
    folder,output=private(folder),private(output)
    if output.exists(): raise FileExistsError('Use a fresh private audit directory')
    manifest=read(folder/'manifest.json')
    for key in ('input_sha256','source_sha256'): verify_mapping(manifest[key])
    verify_mapping(manifest['artifact_sha256'],folder)
    registration=ROOT/manifest['registration']
    if sha(registration/'manifest.json') != manifest['registration_sha256']:
        raise ValueError('Registration changed')
    registered=read(registration/'manifest.json')
    verify_mapping(registered['artifact_sha256'],registration)
    tasks, rows=read(registration/'tasks.json'),read(folder/'rows.json')
    cfg=read(registration/'configuration.json')
    if [r['case'] for r in rows] != [task['case'] for task in tasks]:
        raise ValueError('Local-case coverage changed')
    if len(rows) != len({r['case'] for r in rows}): raise ValueError('Duplicated local case')
    streams=set(); policy_count=references=approximations=0
    full_counts={}
    for task,row in zip(tasks,rows):
        if [s['scenario'] for s in row['scenarios']] != [s['scenario'] for s in task['scenarios']]:
            raise ValueError('Financial family coverage changed')
        if len(row['scenarios']) != cfg['family_size']: raise ValueError('Wrong nominal family size')
        state=np.random.PCG64(np.random.SeedSequence(cfg['root_seed'],spawn_key=(task['case'],))).state
        if (row['stream'] != dict(entropy=cfg['root_seed'],spawn_key=[task['case']],stream_id=_digest(state),paths=cfg['evaluation_paths'])
                or row['stream']['stream_id'] in task['previous_streams']):
            raise ValueError('Invalid stream identity or sample size')
        streams.add(row['stream']['stream_id'])
        frozen=[(r['repetition'],p) for r in task['policy_rows'] for p in r['policies']]
        identity=[(rep,p['training_paths'],p['degree'],p['fingerprint']) for rep,p in frozen]
        clock=None
        for record,previous,s in zip(task['scenarios'],task['old_financial_results'],row['scenarios']):
            if record['fit']['status'] != 'conditional_numerical_witness':
                if s['status'] != 'no_original_screened_construction' or s['policies']:
                    raise ValueError('Failed original construction admitted')
                continue
            if s['status'] != 'evaluated' or s['stream_id'] != row['stream']['stream_id']:
                raise ValueError('Missing scenario or changed common-shock stream')
            model,option,rights=restore_scenario(record)
            contract,calendar=extended_calendar(model,option,rights,tuple(x/YEAR for x in cfg['covered_extensions_seconds'] if x))
            extra=max(cfg['covered_extensions_seconds'])/YEAR
            if s['extension_seconds'] != max(cfg['covered_extensions_seconds']): raise ValueError('Wrong endpoint')
            if s['premium_factor'] != model.premium_factor or s['spread'] != record['fit']['put_ask']-record['fit']['put_bid']:
                raise ValueError('Financial units changed')
            approximation=build_values(model,contract,calendar,**cfg['value_approximation'])
            if (s['approximation_fingerprint'] != approximation.fingerprint
                    or s['approximation'] != read_roundtrip(asdict(approximation))):
                raise ValueError('Value approximation differs')
            current=[(d.time,d.phase) for d in observation_rights(approximation)]
            if clock is not None and current != clock: raise ValueError('Common observation clock differs')
            clock=current; approximations+=1
            condition=terminal_call_dominance(model,replace(contract,kind='call'),calendar)
            if s['terminal_call_condition'] != read_roundtrip(asdict(condition)): raise ValueError('Call dominance differs')
            for label, opt in (('european',option),('call',replace(option,kind='call'))):
                expected=european_increment(model,opt,rights[-1],extra)
                expected={k:asdict(v) if k in ('baseline','expanded') else v for k,v in expected.items()}
                if read_roundtrip(expected) != s[label]: raise ValueError('Independent European reference differs')
                references+=1
            fitted=record['fit']; eta=fitted['numerical_allowance']; tol=fitted['call_tolerance']
            native_tol=task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(s['original_fine_price'])
            if abs(s['original_fine_price']-fitted['put_prices'][1]) > native_tol:
                raise ValueError('Original numerical price differs')
            close(s['pde_increment_quote'],s['prices']['fine']-s['original_fine_price'])
            screens=dict(joint_space_time=abs(s['prices']['fine']-s['prices']['base'])<=eta,
                domain=abs(s['prices']['wide']-s['prices']['fine'])<=eta,
                european_increment=-2*eta<=s['pde_increment_quote']<=s['european']['increment_quote']+2*eta,
                call_reference=(s['call']['expanded']['quadrature_error_estimate']+s['call']['expanded']['omitted_tail_bound'])/model.premium_factor<=tol)
            call_value=s['call']['expanded']['premium_quote']
            callfit=bool(condition.sufficient_condition_satisfied and fitted['call_bid']<=call_value-tol and call_value+tol<=fitted['call_ask'])
            putfit=bool(fitted['put_bid']<=s['prices']['fine']-eta and s['prices']['fine']+eta<=fitted['put_ask'])
            if (s['numerical_screens'] != screens or s['call_quote_with_margin'] != callfit
                    or s['put_quote_with_margin'] != putfit or s['joint_quote_compatible'] != (callfit and putfit and all(screens.values()))):
                raise ValueError('Numerical or quote classification differs')
            expected_repeat=task['repeat_python'] and s['scenario']==task['scenarios'][0]['scenario']
            if bool(s['python_repeat']) != expected_repeat: raise ValueError('Python-repeat coverage differs')
            if expected_repeat:
                repeat=s['python_repeat']
                if abs(repeat['native_quote']-repeat['python_quote'])>repeat['tolerance']:
                    raise ValueError('Python/native comparison failed')
            actual=[(p['repetition'],p['training_paths'],p['degree'],p['fingerprint']) for p in s['policies']]
            if actual != identity: raise ValueError('Frozen policy coverage differs')
            for j, (p, old) in enumerate(zip(s['policies'],previous['policies'])):
                source,_=load_frozen(frozen[j][1])
                if [(d.time,d.phase) for d in rights] != [(d.time,d.phase) for d in source.decisions]:
                    raise ValueError('Original decision clock changed')
                result=p['result']; count=cfg['evaluation_paths']; factor=model.premium_factor
                for name in ('policy_value','upper_value','gap_bound','stopped_martingale','terminal_martingale'):
                    check_estimate(result[name],count)
                close(result['gap_bound']['value'],result['upper_value']['value']-result['policy_value']['value'])
                close(result['loss_upper_97_5_quote'],result['gap_bound']['upper']/factor)
                expected=(result['gap_bound']['value']+t.ppf(1-cfg['family_error_probability']/cfg['family_size'],count-1)*result['gap_bound']['standard_error'])/factor
                close(p['joint_family_upper_quote'],expected)
                if result['sample_min_gap_pv']<0 or result['upper_value']!=s['policies'][0]['result']['upper_value']:
                    raise ValueError('Invalid gap or non-shared upper payoff')
                if len(p['stopping_counts'])!=len(rights) or min(p['stopping_counts'])<0 or sum(p['stopping_counts'])!=count:
                    raise ValueError('Stopping count or old-expiry coverage differs')
                effect=p['paired_financial_effect_quote']; check_estimate(effect,count)
                base=row['scenarios'][0]
                base_gap=base['policies'][j]['result']['gap_bound']
                close(effect['value'],result['gap_bound']['value']/factor-base_gap['value']/base['premium_factor'])
                a=result['gap_bound']['standard_error']/factor; b=base_gap['standard_error']/base['premium_factor']
                covariance=(a*a+b*b-effect['standard_error']**2)/2
                if abs(covariance)>a*b+1e-12: raise ValueError('Impossible paired variance')
                check_difference(p['independent_expiry_gap_change_quote'],quote(result['gap_bound'],factor),quote(old['result']['gap_bound'],factor))
                check_difference(p['interaction_quote'],effect,old['paired_gap_change'])
                policy_count+=1
        evaluated=[s for s in row['scenarios'] if s['status']=='evaluated']
        complete=len(evaluated)==cfg['family_size']
        compatible=complete and all(s['joint_quote_compatible'] for s in evaluated)
        if len(row['families'])!=len(frozen): raise ValueError('Family policy count differs')
        for j,family in enumerate(row['families']):
            maximum, scenario=max((s['policies'][j]['joint_family_upper_quote'],s['scenario']) for s in evaluated)
            if (family['fingerprint']!=frozen[j][1]['fingerprint'] or family['full_family']!=complete
                    or family['all_joint_quotes_compatible']!=compatible or family['maximum_upper_quote']!=maximum
                    or family['maximising_scenario']!=scenario or family['spread']!=evaluated[0]['spread']
                    or family['below_spread']!=(complete and maximum<family['spread'])
                    or family['compatible_family_below_spread']!=(compatible and maximum<family['spread'])
                    or family['old_financial_family_upper_quote']!=task['old_families'][j]['maximum_family_adjusted_upper_quote']):
                raise ValueError('Joint family maximum or interpretation differs')
            key=(family['training_paths'],family['degree'])
            c=full_counts.setdefault(key,dict(policies=0,complete_families=0,loss_bounds_below_spread=0,compatible_families=0,compatible_family_bounds_below_spread=0,bounds=[]))
            for field,flag in (('policies',True),('complete_families',complete),('loss_bounds_below_spread',family['below_spread']),
                    ('compatible_families',compatible),('compatible_family_bounds_below_spread',family['compatible_family_below_spread'])):
                c[field]+=int(flag)
            c['bounds'].append(maximum)
    if len(streams)!=len(rows): raise ValueError('Local-case streams reused')
    summary=read(folder/'summary.json')
    if summarise(rows)!=summary: raise ValueError('Summary reconstruction differs')
    for v in summary['families']:
        c=full_counts[v['training_paths'],v['degree']]
        for field,value in c.items():
            if field!='bounds' and v[field]!=value: raise ValueError('Independent family summary differs')
        if v['maximum_bound_range']!=[min(c['bounds']),max(c['bounds'])]: raise ValueError('Family range differs')
    repeated=evaluate(tasks[0])
    # Dataclass tuples become lists in the retained JSON representation.
    # Canonicalise that representation without rounding any numerical values.
    encoded=read_roundtrip(numerical_record(repeated))
    output.mkdir(parents=True)
    save(output/'rerun_case.json',encoded)
    if encoded!=numerical_record(rows[0]):
        raise ValueError('Prescribed complete local-case numerical rerun differs')
    save(output/'verification.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(folder/'manifest.json'),auditor_sha256=sha(Path(__file__)),
        rebuilt_value_functions=approximations,recomputed_european_pairs=references,checked_policy_evaluations=policy_count,
        reconstructed_families=sum(len(r['families']) for r in rows),
        rerun_case=tasks[0]['case'],rerun_financial_scenarios=len(tasks[0]['scenarios']),exact_numerical_match=True,
        rerun_sha256=sha(output/'rerun_case.json'),full_experiment_repeated=False,publication_clearance=False))
    print(f'Verified {policy_count} policy evaluations and repeated the complete first local case',flush=True)


def read_roundtrip(value):
    import json
    return json.loads(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.experiment,args.output)
