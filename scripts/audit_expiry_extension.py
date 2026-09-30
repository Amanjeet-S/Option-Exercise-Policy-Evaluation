"""Reconstruct expiry clocks, terminal bounds and embedded policy inference."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np

from run_conditional_calibration import private, read, save
from study_expiry_extension import planned_inputs, verified_results, scenario_calendars, YEAR
from study_policy_input_sensitivity import restore_scenario
from audit_fitted_policy_gain import check_estimate, close
from qf_research.market_data.calendar_robustness import mesh_changes
from qf_research.optimal_stopping.value_martingale import build_values
from qf_research.optimal_stopping.expiry_extension import european_increment
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import Estimate


def reconstruct(folder):
    folder=private(folder);rows,_=verified_results(folder)
    tasks,inputs,sources=planned_inputs();m=read(folder/'manifest.json')
    if tasks!=read(folder/'tasks.json') or inputs!=m['input_sha256'] or sources!=m['source_sha256']:
        raise ValueError('Planned inputs or source identities differ')
    values=duals=repeats=0
    for task,row in zip(tasks,rows):
        cfg=task['configuration'];model,option,rights=restore_scenario(task['baseline'])
        fit=task['baseline']['fit'];n=cfg['evaluation_paths'];stream=row['stream']
        if stream['entropy']!=cfg['root_seed'] or stream['spawn_key']!=[row['case']] or stream['paths']!=n or stream['stream_id'] in task['previous_streams']:
            raise ValueError('New common sample differs')
        expected=[(r['repetition'],p['training_paths'],p['degree'],p['fingerprint']) for r in task['policy_rows'] for p in r['policies']]
        if [s['extension_seconds'] for s in row['scenarios']]!=cfg['extensions_seconds']:
            raise ValueError('Expiry coverage differs')
        for (seconds,contract,calendar),s in zip(scenario_calendars(model,option,rights,cfg),row['scenarios']):
            if s['decisions']!=[asdict(d) for d in calendar] or s['maturity']!=contract.maturity:
                raise ValueError('Expiry clock or settlement terms differ')
            close(s['prices']['fine']-row['scenarios'][0]['prices']['fine'],s['increment_prices']['fine'],'calendar price increment')
            if mesh_changes(s['prices'])!=s['mesh_changes']:
                raise ValueError('Mesh diagnostics differ')
            for name,kind in [('european','put'),('call','call')]:
                result=european_increment(model,replace(option,kind=kind),rights[-1],seconds/YEAR)
                serial={k:asdict(v) if k in ('baseline','expanded') else v for k,v in result.items()}
                if serial!=s[name]:raise ValueError('European reference or terminal upper bound differs')
            close(s['call']['increment_quote'],s['european']['increment_quote'],'unchanged cash parity across the extension')
            condition=terminal_call_dominance(model,replace(contract,kind='call'),calendar)
            if json.loads(json.dumps(asdict(condition)))!=s['terminal_call_condition']:
                raise ValueError('Terminal-call condition differs')
            approximation=build_values(model,contract,calendar,**cfg['value_approximation'])
            if json.loads(json.dumps(asdict(approximation)))!=s['approximation'] or approximation.fingerprint!=s['approximation_fingerprint']:
                raise ValueError('Retained value functions do not reproduce')
            values+=1
            if task['repeat_python']:
                ref=s['python_repeat'];close(ref['difference'],ref['native_price']-ref['python_price'],'Python/native difference')
                if abs(ref['difference'])>ref['tolerance']:raise ValueError('Native parity failed')
                repeats+=1
            actual=[(p['repetition'],p['training_paths'],p['degree'],p['fingerprint']) for p in s['policies']]
            if actual!=expected:raise ValueError('Frozen policy coverage differs')
            eta,tol=fit['numerical_allowance'],fit['call_tolerance'];value=s['call']['expanded']['premium_quote']
            callfit=fit['call_bid']<=value-tol and value+tol<=fit['call_ask']
            putfit=fit['put_bid']<=s['prices']['fine']-eta and s['prices']['fine']+eta<=fit['put_ask']
            c=s['mesh_changes'];increments=s['increment_prices']
            for mesh,value in increments.items():
                close(value,s['prices'][mesh]-row['scenarios'][0]['prices'][mesh],'matched-mesh increment')
            screens=dict(space=all(abs(c[k])<=eta for k in ('space_at_coarse_time','space_at_fine_time')),
                time=all(abs(c[k])<=eta for k in ('time_at_coarse_space','time_at_fine_space')),
                domain=abs(c['domain'])<=eta,
                increment_refinement=all(abs(v-increments['fine'])<=eta for v in increments.values()),
                european_increment=all(-2*eta<=v<=s['european']['increment_quote']+2*eta for v in increments.values()),
                call_reference=(s['call']['expanded']['quadrature_error_estimate']+s['call']['expanded']['omitted_tail_bound'])/model.premium_factor<=tol)
            if s['numerical_screens']!=screens or (callfit,putfit,callfit and putfit and all(screens.values()))!=(s['call_quote_with_margin'],s['put_quote_with_margin'],s['conditional_joint_compatibility']):
                raise ValueError('Numerical or quote classification differs')
            for i,p in enumerate(s['policies']):
                r=p['result'];base=row['scenarios'][0]['policies'][i]
                for key in ('policy_value','upper_value','gap_bound','stopped_martingale','terminal_martingale'):
                    check_estimate(r[key],n)
                close(r['gap_bound']['value'],r['upper_value']['value']-r['policy_value']['value'],'dual accounting')
                close(r['loss_upper_97_5_quote'],r['gap_bound']['upper']/model.premium_factor,'quoted pointwise upper bound')
                close(p['family_adjusted_upper_quote'],family_upper_limit(Estimate(**r['gap_bound']),cfg['family_size'],cfg['family_error_probability'])/model.premium_factor,'family limit')
                if r['sample_min_gap_pv']<0 or r['upper_value']!=s['policies'][0]['result']['upper_value']:
                    raise ValueError('Non-negative gap or common upper payoff failed')
                difference=p['paired_gap_change'];check_estimate(difference,n)
                close(difference['value'],(r['gap_bound']['value']-base['result']['gap_bound']['value'])/model.premium_factor,'paired difference')
                a=r['gap_bound']['standard_error']/model.premium_factor;b=base['result']['gap_bound']['standard_error']/model.premium_factor
                covariance=.5*(a*a+b*b-difference['standard_error']**2)
                if abs(covariance)>a*b+1e-12:raise ValueError('Impossible paired variance')
                if p['stopping_counts']!=base['stopping_counts'] or sum(p['stopping_counts'])!=n or min(p['stopping_counts'])<0:
                    raise ValueError('Original stopping decisions changed')
                if seconds==0:close([difference['value'],difference['standard_error']],[0,0],'baseline comparison')
                duals+=1
        for i,f in enumerate(row['families']):
            bound,seconds=max((s['policies'][i]['family_adjusted_upper_quote'],s['extension_seconds']) for s in row['scenarios'])
            if f['maximum_upper_quote']!=bound or f['maximising_extension_seconds']!=seconds or f['below_spread']!=(bound<row['spread']):
                raise ValueError('Family comparison differs')
    return dict(models=len(rows),rebuilt_value_functions=values,dual_evaluations=duals,
        recomputed_european_pairs=2*values,retained_python_native_comparisons=repeats,
        unchanged_policy_stopping_allocations=True,full_simulation_repeated=False,
        empirical_calibration_accepted=False,publication_clearance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=private(args.output)
    if output.exists():raise FileExistsError('Use a fresh private output file')
    result=reconstruct(args.run);output.parent.mkdir(parents=True,exist_ok=True);save(output,result);print(result)
