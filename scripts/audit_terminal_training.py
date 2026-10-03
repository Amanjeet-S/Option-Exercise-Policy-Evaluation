"""Reconstruct matched training, shared-upper comparisons and family limits."""
import argparse
import json
from pathlib import Path
from run_conditional_calibration import private,read,save
from study_terminal_training import planned_inputs,verified_results,train_case,training_record
from study_expiry_extension import scenario_calendars
from study_policy_input_sensitivity import restore_scenario
from audit_fitted_policy_gain import check_estimate,close
from qf_research.optimal_stopping.value_martingale import build_values
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import Estimate


def reconstruct(folder):
    folder=private(folder);rows,_=verified_results(folder)
    tasks,inputs,sources=planned_inputs();manifest=read(folder/'manifest.json')
    if tasks!=read(folder/'tasks.json') or inputs!=manifest['input_sha256'] or sources!=manifest['source_sha256']:
        raise ValueError('Planned evidence differs')
    fits=values=comparisons=0
    for task,row in zip(tasks,rows):
        cfg=task['configuration'];n=cfg['evaluation_paths'];model,option,rights=restore_scenario(task['baseline']);factor=model.premium_factor
        training,_=train_case(task)
        if json.loads(json.dumps(training_record(training)))!=training_record(row['training']):
            raise ValueError('Matched regression coefficients do not reproduce')
        fits+=sum(len(r['scenarios']) for r in training)
        stream=row['stream']
        if stream['entropy']!=cfg['root_seed'] or stream['spawn_key']!=[row['case']] or stream['paths']!=n or stream['stream_id'] in task['previous_streams']:
            raise ValueError('Evaluation stream differs')
        close(row['premium_factor'],factor,'premium factor')
        close(row['spread'],task['baseline']['fit']['put_ask']-task['baseline']['fit']['put_bid'],'spread')
        if row['refit'] or [s['extension_seconds'] for s in row['scenarios']]!=cfg['extensions_seconds']:
            raise ValueError('Financial scope or scenario coverage differs')
        for (seconds,contract,calendar),ref,scenario in zip(scenario_calendars(model,option,rights,cfg),task['references'],row['scenarios']):
            if any(scenario[k]!=v for k,v in ref.items()):raise ValueError('Reference changed')
            approx=build_values(model,contract,calendar,**cfg['value_approximation'])
            if approx.fingerprint!=ref['approximation_fingerprint']:raise ValueError('Approximate values changed')
            values+=1
            if len(scenario['policies'])!=len(training):raise ValueError('Policy coverage differs')
            for index,(entry,p) in enumerate(zip(training,scenario['policies'])):
                if any(p[k]!=entry[k] for k in ('repetition','training_paths','degree','original_fingerprint')):
                    raise ValueError('Policy identity differs')
                trained=next(s for s in entry['scenarios'] if s['extension_seconds']==seconds)
                if p['retrained_fingerprint']!=trained['fingerprint']:raise ValueError('Changed training fingerprint')
                for name,result in p['duals'].items():
                    for key in ('policy_value','upper_value','gap_bound','stopped_martingale','terminal_martingale'):check_estimate(result[key],n)
                    close(result['gap_bound']['value'],result['upper_value']['value']-result['policy_value']['value'],'dual accounting')
                    close(result['loss_upper_97_5_quote'],result['gap_bound']['upper']/factor,'pointwise limit')
                    close(p['family_upper_quote'][name],family_upper_limit(Estimate(**result['gap_bound']),cfg['family_size'],cfg['family_error_probability'])/factor,'family limit')
                    if result['sample_min_gap_pv']<0 or result['upper_value']!=scenario['policies'][0]['duals'][name]['upper_value']:
                        raise ValueError('Common upper payoff or non-negative gap failed')
                old,new=p['duals']['terminal_only'],p['duals']['retrained'];gain=p['paired_gain'];check_estimate(gain,n)
                if old['upper_value']!=new['upper_value']:raise ValueError('Policy comparison changed the upper payoff')
                close(gain['value'],(old['gap_bound']['value']-new['gap_bound']['value'])/factor,'paired loss change')
                close(gain['value'],(new['policy_value']['value']-old['policy_value']['value'])/factor,'paired value change')
                a,b=old['gap_bound']['standard_error']/factor,new['gap_bound']['standard_error']/factor
                covariance=.5*(a*a+b*b-gain['standard_error']**2)
                if abs(covariance)>a*b+1e-12:raise ValueError('Impossible paired covariance')
                if not 0<=p['changed_stops']==p['moved_earlier']+p['moved_later']<=n:raise ValueError('Invalid stopping changes')
                for counts in p['stopping_counts'].values():
                    if len(counts)!=len(rights) or sum(counts)!=n or min(counts)<0:raise ValueError('Invalid stopping allocation')
                base=row['scenarios'][0]['policies'][index]
                if p['stopping_counts']['terminal_only']!=base['stopping_counts']['terminal_only']:
                    raise ValueError('Original earlier stopping instructions changed')
                if seconds==0 and (old!=new or p['changed_stops'] or gain['value'] or gain['standard_error']):
                    raise ValueError('Zero-extension comparison differs')
                comparisons+=1
        if len(row['families'])!=len(training):raise ValueError('Family coverage differs')
        for index,(entry,family) in enumerate(zip(training,row['families'])):
            if any(family[k]!=entry[k] for k in ('repetition','training_paths','degree')):raise ValueError('Family identity differs')
            for name in ('terminal_only','retrained'):
                maximum=max(s['policies'][index]['family_upper_quote'][name] for s in row['scenarios'])
                if family['maximum_upper_quote'][name]!=maximum or family['below_spread'][name]!=(maximum<row['spread']):
                    raise ValueError('Family comparison differs')
    return dict(models=len(rows),reproduced_training_policies=fits,rebuilt_value_functions=values,
        paired_comparisons=comparisons,reconstructed_dual_records=2*comparisons,
        original_training_fingerprints_match=True,full_evaluation_repeated=False,new_pde_valuations=False,
        empirical_calibration_accepted=False,publication_clearance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();output=private(args.output)
    if output.exists():raise FileExistsError('Use a fresh private audit file')
    result=reconstruct(args.run);output.parent.mkdir(parents=True,exist_ok=True);save(output,result);print(result)
