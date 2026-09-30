"""Reconstruct frozen inputs, stopping records and paired sampling identities."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from run_conditional_calibration import private, read, save
from study_fitted_policy_gain import planned_inputs, verified_results
from diagnose_fitted_calendars import task_problem
from validate_funded_dual import load_frozen


def close(a, b, label):
    if not np.allclose(a, b, rtol=2e-11, atol=2e-12):
        raise ValueError(f'Inconsistent {label}')


def check_estimate(estimate, count):
    if estimate['observations'] != count or estimate['confidence'] != .95 or estimate['standard_error'] < 0:
        raise ValueError('Invalid sampling estimate')
    radius = student_t.ppf(.975, count-1)*estimate['standard_error']
    close([estimate['lower'], estimate['upper']],
          [estimate['value']-radius, estimate['value']+radius], 'sampling interval')


def reconstruct(folder):
    rows, _ = verified_results(folder)
    tasks, inputs, sources = planned_inputs()
    manifest = read(folder/'manifest.json')
    if tasks != read(folder/'tasks.json') or inputs != manifest['input_sha256'] or sources != manifest['source_sha256']:
        raise ValueError('Original planned inputs differ')
    policy_count = contrasts = duals = 0
    for task, row in zip(tasks, rows):
        model, problem, _ = task_problem(task['original'])
        cfg = task['configuration']
        option = replace(problem.option, kind='put')
        by_role = {s['role']: s for s in row['streams']}
        for role in cfg['stream_roles']:
            stream = by_role[role]
            expected = [task['case'], task['repetition'], cfg['stream_roles'].index(role)]
            if stream['spawn_key'] != expected or stream['entropy'] != cfg['root_seed']:
                raise ValueError('Random-stream allocation differs')
        expected_variants = [(n, d) for n in cfg['training_paths'] for d in cfg['degrees']]
        if [(p['training_paths'], p['degree']) for p in row['policies']] != expected_variants:
            raise ValueError('Policy coverage differs')
        n = cfg['evaluation_paths']
        for kind in ('raw', 'controlled'):
            covariance = np.asarray(row[f'{kind}_gain_covariance_quote'])
            close(covariance, covariance.T, 'covariance symmetry')
            if np.linalg.eigvalsh(covariance).min() < -1e-10:
                raise ValueError('Invalid covariance matrix')
            close(np.diag(covariance)/n, [p['gain'][f'{kind}_quote']['standard_error']**2 for p in row['policies']],
                  'marginal covariance')
            for contrast in row['contrasts']:
                a, b = contrast['left'], contrast['right']
                if contrast['direction'] != 'right_minus_left':
                    raise ValueError('Contrast direction differs')
                check_estimate(contrast[kind], n)
                pa, pb = row['policies'][a]['gain'][f'{kind}_quote'], row['policies'][b]['gain'][f'{kind}_quote']
                close(contrast[kind]['value'], pb['value']-pa['value'], 'paired mean')
                close(contrast[kind]['standard_error']**2,
                      (covariance[a, a]+covariance[b, b]-2*covariance[a, b])/n, 'paired variance')
        if {(p['left'], p['right']) for p in row['contrasts']} != {(i, j) for i in range(4) for j in range(i+1, 4)}:
            raise ValueError('Incomplete contrasts')
        contrasts += len(row['contrasts'])
        for entry in row['policies']:
            policy, control = load_frozen(entry)
            if (policy.model != model or policy.option != option or policy.decisions != problem.decisions
                    or policy.training_stream_id != by_role['training']['stream_id']):
                raise ValueError('Frozen policy problem or training provenance differs')
            for coefficient in (entry['gain_control'], entry['control']):
                if (coefficient['policy_fingerprint'] != policy.fingerprint
                        or coefficient['pilot_stream_id'] != by_role['pilot']['stream_id']
                        or coefficient['pilot_paths'] != cfg['pilot_paths']):
                    raise ValueError('Control provenance differs')
            counts = entry['stopping_counts']
            if len(counts) != len(problem.decisions) or any(c < 0 for c in counts) or sum(counts) != n:
                raise ValueError('Stopping counts differ')
            close(entry['early_exercise_fraction'], 1-counts[-1]/n, 'early exercise fraction')
            if [d['exercises'] for d in entry['decision_diagnostics']] != counts[:-1]:
                raise ValueError('Decision diagnostics differ')
            for estimate in entry['gain'].values():
                check_estimate(estimate, n)
            gain = entry['gain']['controlled_quote']
            close(entry['pde_shortfall'], row['pde_quote']-row['european_quote']-gain['value'], 'numerical shortfall')
            close(entry['pde_shortfall_sampling_interval'],
                  [row['model_exercise_premium']-gain['upper'], row['model_exercise_premium']-gain['lower']],
                  'shortfall interval')
            for hedge, dual in entry['duals'].items():
                for key in ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale'):
                    check_estimate(dual[key], cfg['dual_paths'])
                close(dual['gap_bound']['value'], dual['upper_value']['value']-dual['policy_value']['value'], 'dual mean identity')
                close(dual['loss_upper_97_5_quote'], dual['gap_bound']['upper']/model.premium_factor, 'dual quote units')
                if dual['sample_min_gap_pv'] < 0:
                    raise ValueError('Negative retained pathwise dual gap')
                if hedge == 'zero':
                    close([dual['stopped_martingale']['value'], dual['terminal_martingale']['value']], [0., 0.], 'zero hedge')
                duals += 1
            policy_count += 1
    return dict(tasks=len(tasks), policies=policy_count, paired_contrasts=contrasts, dual_estimates=duals,
        reconstructed_inputs_match=True, sampling_and_covariance_identities_match=True,
        full_simulation_repeated=False, empirical_calibration_accepted=False, publication_clearance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder, output = private(args.run), private(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh audit output')
    result = reconstruct(folder)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print(result)
