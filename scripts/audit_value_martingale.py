"""Rebuild retained value functions and check all policy-bound identities."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from run_conditional_calibration import private, read, save
from study_value_martingale import planned_inputs, verified_results
from validate_funded_dual import load_frozen
from audit_fitted_policy_gain import check_estimate, close
from qf_research.optimal_stopping.value_martingale import build_values


def reconstruct(folder):
    rows, _ = verified_results(folder)
    tasks, inputs, sources = planned_inputs()
    manifest = read(folder/'manifest.json')
    if tasks != read(folder/'tasks.json') or inputs != manifest['input_sha256'] or sources != manifest['source_sha256']:
        raise ValueError('Planned input reconstruction differs')
    count = 0
    for task, row in zip(tasks, rows):
        cfg = task['configuration']
        old = task['prior_rows'][0]
        policy, _ = load_frozen(old['policies'][0])
        if row['stream']['paths'] != cfg['evaluation_paths'] or row['stream']['spawn_key'] != [row['case']]:
            raise ValueError('Evaluation sample allocation differs')
        for name, settings in cfg['approximations'].items():
            approximation = build_values(policy.model, policy.option, policy.decisions, **settings)
            saved = row['approximations'][name]
            if approximation.fingerprint != saved['fingerprint'] or json.loads(json.dumps(asdict(approximation))) != saved['value']:
                raise ValueError('Approximate value functions do not reproduce')
        expected = [(r['repetition'], p['training_paths'], p['degree'], p['fingerprint'])
                    for r in task['prior_rows'] for p in r['policies']]
        actual = [(p['repetition'], p['training_paths'], p['degree'], p['fingerprint']) for p in row['policies']]
        if actual != expected:
            raise ValueError('Frozen policy identities differ')
        for entry in row['policies']:
            if set(entry['results']) != {'baseline', *cfg['approximations']}:
                raise ValueError('Missing prescribed martingale')
            for method, result in entry['results'].items():
                for key in ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale'):
                    check_estimate(result[key], cfg['evaluation_paths'])
                close(result['gap_bound']['value'], result['upper_value']['value']-result['policy_value']['value'], 'dual expectation accounting')
                close(result['loss_upper_97_5_quote'], result['gap_bound']['upper']/row['premium_factor'], 'quoted loss bound')
                if result['sample_min_gap_pv'] < 0:
                    raise ValueError('Negative dual pathwise gap')
                if method != 'baseline' and result['upper_value'] != row['policies'][0]['results'][method]['upper_value']:
                    raise ValueError('Common value martingale has policy-dependent upper payoff')
                if entry['evaluation_seconds'][method] < 0:
                    raise ValueError('Invalid measured cost')
                count += 1
            if set(entry['paired_gap_comparisons']) != set(cfg['comparisons']):
                raise ValueError('Missing paired comparison')
            for comparison, estimate in entry['paired_gap_comparisons'].items():
                right, left = comparison.split('_minus_')
                check_estimate(estimate, cfg['evaluation_paths'])
                close(estimate['value'], (entry['results'][right]['gap_bound']['value']-
                    entry['results'][left]['gap_bound']['value'])/row['premium_factor'], 'paired gap difference')
                # Recover the paired covariance from the three sample variances.
                a, b = entry['results'][right]['gap_bound'], entry['results'][left]['gap_bound']
                se_a, se_b = a['standard_error']/row['premium_factor'], b['standard_error']/row['premium_factor']
                covariance_mean = .5*(se_a**2+se_b**2-estimate['standard_error']**2)
                if abs(covariance_mean) > se_a*se_b+1e-12:
                    raise ValueError('Paired variance violates covariance bounds')
    return dict(models=len(rows), reconstructed_value_approximations=2*len(rows), dual_evaluations=count,
        common_upper_payoff_identity=True, paired_sampling_identities=True,
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
