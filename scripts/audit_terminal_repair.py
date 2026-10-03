"""Reconstruct the terminal-repair evidence and paired statistical identities."""

import argparse
from pathlib import Path
import numpy as np
from run_conditional_calibration import private, read, save
from study_terminal_repair import planned_inputs, verified_results
from study_expiry_extension import scenario_calendars
from study_policy_input_sensitivity import restore_scenario
from audit_fitted_policy_gain import check_estimate, close
from qf_research.optimal_stopping.value_martingale import build_values
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import Estimate


def reconstruct(folder):
    folder = private(folder); rows, _ = verified_results(folder)
    tasks, inputs, sources = planned_inputs(); manifest = read(folder/'manifest.json')
    if tasks != read(folder/'tasks.json') or inputs != manifest['input_sha256'] or sources != manifest['source_sha256']:
        raise ValueError('Planned inputs or source identities differ')
    values = policies = duals = 0
    for task, row in zip(tasks, rows):
        cfg = task['configuration']; n = cfg['evaluation_paths']
        model, option, rights = restore_scenario(task['baseline']); factor = model.premium_factor
        fit = task['baseline']['fit']; close(row['spread'], fit['put_ask']-fit['put_bid'], 'spread')
        if row['premium_factor'] != factor or row['refit'] or row['retraining']:
            raise ValueError('Model or training scope differs')
        stream = row['stream']
        if (stream['entropy'] != cfg['root_seed'] or stream['spawn_key'] != [row['case']]
                or stream['paths'] != n or stream['stream_id'] in task['previous_streams']):
            raise ValueError('Fresh sample protocol differs')
        expected = [(r['repetition'], p['training_paths'], p['degree'], p['fingerprint'])
                    for r in task['policy_rows'] for p in r['policies']]
        if [s['extension_seconds'] for s in row['scenarios']] != cfg['extensions_seconds']:
            raise ValueError('Scenario coverage differs')
        for (seconds, contract, calendar), ref, scenario in zip(scenario_calendars(model, option, rights, cfg), task['references'], row['scenarios']):
            if any(scenario[k] != v for k, v in ref.items()):
                raise ValueError('Retained PDE or financial reference changed')
            approximation = build_values(model, contract, calendar, **cfg['value_approximation'])
            if approximation.fingerprint != ref['approximation_fingerprint']:
                raise ValueError('Deterministic value approximation differs')
            values += 1
            if [(p['repetition'], p['training_paths'], p['degree'], p['fingerprint']) for p in scenario['policies']] != expected:
                raise ValueError('Frozen policy coverage differs')
            for index, p in enumerate(scenario['policies']):
                base = row['scenarios'][0]['policies'][index]
                for name, result in p['duals'].items():
                    for key in ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale'):
                        check_estimate(result[key], n)
                    close(result['gap_bound']['value'], result['upper_value']['value']-result['policy_value']['value'], 'dual identity')
                    close(result['loss_upper_97_5_quote'], result['gap_bound']['upper']/factor, 'pointwise loss limit')
                    if result['sample_min_gap_pv'] < 0 or result['upper_value'] != scenario['policies'][0]['duals'][name]['upper_value']:
                        raise ValueError('Pathwise gap or shared upper payoff failed')
                    duals += 1
                for name in ('original', 'repaired'):
                    close(p['family_upper_quote'][name], family_upper_limit(Estimate(**p['duals'][name]['gap_bound']),
                        cfg['family_size'], cfg['family_error_probability'])/factor, 'family limit')
                old, new, direct = [p['duals'][name] for name in ('original', 'repaired', 'direct_repaired')]
                if old['upper_value'] != new['upper_value']:
                    raise ValueError('Policy comparison changed the upper payoff')
                gain = p['integrated_gain']; realised = p['realised_gain']; noise = p['conditional_noise']
                for estimate in (gain, realised, noise, p['direct_minus_collapsed_gap']):
                    check_estimate(estimate, n)
                close((old['gap_bound']['value']-new['gap_bound']['value'])/factor, gain['value'], 'gain equals loss-bound reduction')
                close((new['policy_value']['value']-old['policy_value']['value'])/factor, gain['value'], 'gain equals lower-payoff increase')
                close(noise['value'], realised['value']-gain['value'], 'conditional residual')
                close(p['direct_minus_collapsed_gap']['value'], (direct['gap_bound']['value']-new['gap_bound']['value'])/factor, 'direct gap comparison')
                a, b = realised['standard_error'], gain['standard_error']
                covariance = .5*(a*a+b*b-noise['standard_error']**2)
                if abs(covariance) > a*b+1e-12:
                    raise ValueError('Impossible paired covariance')
                if b:
                    close(p['direct_to_integrated_variance_ratio'], a*a/(b*b), 'variance ratio')
                elif p['direct_to_integrated_variance_ratio'] is not None:
                    raise ValueError('Undefined variance ratio must remain missing')
                if (p['minimum_integrated_gain'] < 0 or p['maximum_accounting_error'] > 1e-11
                        or p['maximum_roundoff_correction'] > 64*np.finfo(float).eps*max(1., option.strike*rights[-1].settlement_factor)):
                    raise ValueError('Gain or numerical accounting failed')
                if (p['stopping_counts'] != base['stopping_counts'] or sum(p['stopping_counts']) != n
                        or p['terminal_survivors'] != p['stopping_counts'][-1]
                        or not 0 <= p['realised_negative_gain_count'] <= p['terminal_survivors']):
                    raise ValueError('Stopping allocation or negative-gain count differs')
                if seconds:
                    close(p['gain_to_pde_increment'], gain['value']/ref['pde_increment'], 'descriptive gain/increment ratio')
                elif (gain['value'] != 0 or gain['standard_error'] != 0 or realised['value'] != 0
                        or old != new or new != direct or p['gain_to_pde_increment'] is not None):
                    raise ValueError('Zero-extension identity failed')
                policies += 1
        if len(row['families']) != len(expected): raise ValueError('Family coverage differs')
        for index, family in enumerate(row['families']):
            if (family['repetition'], family['training_paths'], family['degree']) != expected[index][:3]:
                raise ValueError('Family identity differs')
            for name in ('original', 'repaired'):
                maximum = max(s['policies'][index]['family_upper_quote'][name] for s in row['scenarios'])
                if family['maximum_upper_quote'][name] != maximum or family['below_spread'][name] != (maximum < row['spread']):
                    raise ValueError('Family classification differs')
    return dict(models=len(rows), rebuilt_value_functions=values, policy_scenarios=policies,
        reconstructed_dual_records=duals, paired_accounting_checked=True,
        full_simulation_repeated=False, new_pde_valuations=False,
        empirical_calibration_accepted=False, publication_clearance=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); output = private(args.output)
    if output.exists(): raise FileExistsError('Use a fresh private audit file')
    result = reconstruct(args.run); output.parent.mkdir(parents=True, exist_ok=True); save(output, result); print(result)
