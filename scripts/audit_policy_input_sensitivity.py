"""Reconstruct scenario inputs, value functions and finite-family loss limits."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from run_conditional_calibration import private, read, save
from calibrate_policy_sensitivity import planned_inputs as calibration_inputs, verified_results as calibration_results
from study_policy_input_sensitivity import planned_inputs, verified_results, restore_scenario
from cover_joint_cash_funding import model_at
from qf_research.market_data.settlement_comparison import restore_problem
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_cash import european_reference
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import Estimate
from audit_fitted_policy_gain import check_estimate, close
from validate_funded_dual import load_frozen


def check_plan(folder, planned):
    tasks, inputs, sources = planned
    manifest = read(folder/'manifest.json')
    if tasks != read(folder/'tasks.json') or inputs != manifest['input_sha256'] or sources != manifest['source_sha256']:
        raise ValueError('Prescribed inputs or source identities differ')
    return tasks


def reconstruct(calibration, folder):
    calibration, folder = private(calibration), private(folder)
    fitted, _ = calibration_results(calibration)
    fit_tasks = check_plan(calibration, calibration_inputs())
    rows, _ = verified_results(folder)
    tasks = check_plan(folder, planned_inputs(calibration))
    fits = repeats = 0
    for task, record in zip(fit_tasks, fitted):
        old, result = task['fit_task'], record['fit']
        close(result['nominal_cash'], old['component'][0]+task['fraction']*np.diff(old['component'])[0], 'cash scenario')
        if result['status'] != 'conditional_numerical_witness':
            continue
        model, option, rights = restore_scenario(record)
        raw = restore_problem(old['variants']['midpoint']['raw_call'])
        expected = model_at(old, 'midpoint', result['nominal_cash'], result['state_at_cutoff'], result['volatility'])
        if model != expected or option != replace(raw.option, kind='put') or rights != raw.decisions:
            raise ValueError('Calibrated financial inputs differ')
        condition = terminal_call_dominance(model, raw.option, rights)
        if json.loads(json.dumps(asdict(condition))) != result['terminal_call_condition'] or not condition.sufficient_condition_satisfied:
            raise ValueError('Call exercise reduction differs')
        if not all(result['numerical_screens'].values()):
            raise ValueError('Accepted calibration fails its numerical screens')
        coarse, fine, wide = result['put_prices']
        eta, tol = result['numerical_allowance'], result['call_tolerance']
        call_reference = european_reference(model, raw.option, rights[-1])
        close(call_reference.premium_quote, result['call_value'], 'recomputed call reference')
        if (call_reference.quadrature_error_estimate+call_reference.omitted_tail_bound)/model.premium_factor > tol:
            raise ValueError('Call reference exceeds its numerical allowance')
        close(result['call_margin'], min(result['call_value']-result['call_bid'], result['call_ask']-result['call_value']), 'call margin')
        close(result['put_margin'], min(fine-result['put_bid'], result['put_ask']-fine), 'put margin')
        if (result['call_margin'] < tol or result['put_margin'] < eta
                or abs(fine-coarse) > eta or abs(wide-fine) > eta):
            raise ValueError('Quote fit or grid/domain screen differs')
        interval = result['call_interval']
        if not interval['lower'] <= model.volatility <= interval['upper']:
            raise ValueError('Volatility outside the retained call bracket')
        if task['fit_task']['repeat_python']:
            repeat = result['python_repeat']
            close(repeat['difference'], repeat['native_price']-repeat['original_python_price'], 'independent implementation difference')
            if not repeat['passed'] or abs(repeat['difference']) > repeat['tolerance']:
                raise ValueError('Independent implementation check failed')
            repeats += 1
        if record['baseline_reproduced']:
            for key in ('nominal_cash', 'state_at_cutoff', 'volatility', 'call_value', 'put_prices'):
                if result[key] != task['baseline_retained'][key]:
                    raise ValueError('Baseline calibration changed')
        fits += 1
    approximations = comparisons = families = 0
    for task, row in zip(tasks, rows):
        cfg, stream = task['configuration'], row['stream']
        if (stream['entropy'] != cfg['root_seed'] or stream['spawn_key'] != [task['case']]
                or stream['paths'] != cfg['evaluation_paths'] or stream['stream_id'] in task['previous_streams']):
            raise ValueError('Sample size, seed or stream separation differs')
        originals = [(r['repetition'], p) for r in task['policy_rows'] for p in r['policies']]
        expected = [(rep, p['training_paths'], p['degree'], p['fingerprint']) for rep, p in originals]
        if [s['scenario'] for s in row['scenarios']] != [s['scenario'] for s in task['scenarios']]:
            raise ValueError('Prescribed scenario coverage differs')
        clock = None
        baseline = row['scenarios'][0]
        for record, scenario in zip(task['scenarios'], row['scenarios']):
            if record['fit']['status'] != 'conditional_numerical_witness':
                if scenario['status'] != 'no_screened_construction' or scenario['policies']:
                    raise ValueError('Unscreened construction entered policy inference')
                continue
            if scenario['status'] != 'evaluated' or scenario['stream_id'] != stream['stream_id']:
                raise ValueError('Scenario omitted or common-shock stream differs')
            model, option, rights = restore_scenario(record)
            if scenario['observed_spread'] != record['fit']['put_ask']-record['fit']['put_bid']:
                raise ValueError('Economic comparison threshold changed')
            approximation = build_values(model, option, rights, **cfg['value_approximation'])
            if (scenario['approximation_fingerprint'] != approximation.fingerprint
                    or scenario['approximation'] != json.loads(json.dumps(asdict(approximation)))):
                raise ValueError('Scenario value approximation does not reproduce')
            current = [(d.time, d.phase) for d in observation_rights(approximation)]
            if clock is not None and current != clock:
                raise ValueError('Common-shock observation clock changed')
            clock = current
            approximations += 1
            actual = [(p['repetition'], p['training_paths'], p['degree'], p['source_fingerprint']) for p in scenario['policies']]
            if actual != expected:
                raise ValueError('Original policy coverage differs')
            for index, entry in enumerate(scenario['policies']):
                source, _ = load_frozen(originals[index][1])
                if [(d.time, d.phase) for d in source.decisions] != [(d.time, d.phase) for d in rights]:
                    raise ValueError('Original decision clock changed')
                result = entry['result']
                n, factor = cfg['evaluation_paths'], model.premium_factor
                for key in ('policy_value', 'upper_value', 'gap_bound', 'stopped_martingale', 'terminal_martingale'):
                    check_estimate(result[key], n)
                close(result['gap_bound']['value'], result['upper_value']['value']-result['policy_value']['value'], 'dual accounting')
                close(result['loss_upper_97_5_quote'], result['gap_bound']['upper']/factor, 'pointwise loss upper bound')
                close(entry['family_adjusted_upper_quote'], family_upper_limit(Estimate(**result['gap_bound']),
                    cfg['family_size'], cfg['family_error_probability'])/factor, 'family tail allocation')
                if result['sample_min_gap_pv'] < 0 or result['upper_value'] != scenario['policies'][0]['result']['upper_value']:
                    raise ValueError('Pathwise inequality or shared upper payoff differs')
                difference = entry['paired_gap_change']
                check_estimate(difference, n)
                reference = baseline['policies'][index]['result']['gap_bound']
                close(difference['value'], result['gap_bound']['value']/factor-reference['value']/baseline['premium_factor'], 'paired mean')
                se_a, se_b = result['gap_bound']['standard_error']/factor, reference['standard_error']/baseline['premium_factor']
                covariance = .5*(se_a**2+se_b**2-difference['standard_error']**2)
                if abs(covariance) > se_a*se_b+1e-12:
                    raise ValueError('Paired variance violates covariance bounds')
                counts = entry['stopping_counts']
                if len(counts) != len(rights) or min(counts) < 0 or sum(counts) != n:
                    raise ValueError('Invalid stopping allocation')
                if [d['exercises'] for d in entry['diagnostics']] != counts[:-1]:
                    raise ValueError('Stopping diagnostics differ')
                close(entry['stopping_disagreement_fraction'], entry['earlier_fraction']+entry['later_fraction'], 'changed stopping decisions')
                if not all(0 <= entry[k] <= 1 for k in ('stopping_disagreement_fraction', 'earlier_fraction', 'later_fraction')):
                    raise ValueError('Invalid stopping comparison fraction')
                if scenario is baseline:
                    close([difference['value'], difference['standard_error'], entry['stopping_disagreement_fraction']], [0, 0, 0], 'baseline pairing')
                comparisons += 1
        accepted = [s for s in row['scenarios'] if s['status'] == 'evaluated']
        complete = len(accepted) == cfg['family_size']
        if len(row['families']) != len(originals):
            raise ValueError('Incomplete policy-family records')
        for index, family in enumerate(row['families']):
            maximum, name = max((s['policies'][index]['family_adjusted_upper_quote'], s['scenario']) for s in accepted)
            if (family['source_fingerprint'] != originals[index][1]['fingerprint']
                    or family['maximum_family_adjusted_upper_quote'] != maximum
                    or family['maximising_scenario'] != name or family['full_family'] != complete
                    or family['evaluated_scenarios'] != len(accepted)
                    or family['full_family_bound_below_spread'] != (complete and maximum < baseline['observed_spread'])):
                raise ValueError('Finite-family inference differs')
            families += 1
    return dict(models=len(rows), screened_calibrations=fits, recomputed_call_references=fits,
        retained_independent_python_checks=repeats,
        rebuilt_value_approximations=approximations, policy_evaluations=comparisons, policy_families=families,
        quote_coordinate_policy_identities=True, family_tail_allocation_checked=True,
        paired_sampling_identities=True, full_simulation_repeated=False,
        empirical_calibration_accepted=False, publication_clearance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = private(args.output)
    if output.exists():
        raise FileExistsError('Use a fresh audit output')
    result = reconstruct(args.calibration, args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, result)
    print(result)
