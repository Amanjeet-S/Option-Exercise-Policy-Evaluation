"""Check and render the released, rounded research-summary tables.

This verifies presentation and arithmetic within the supplement. It neither
recreates the simulations nor verifies findings against restricted observations.
Use --write to refresh marked tables after an authorised summary revision.
"""

import argparse
import json
import math
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(row) + ' |' for row in rows],
    ])


def render(data):
    tables = {}
    tables['measurement'] = table(
        ['Training paths', 'Degree', 'Slope gap', 'Coarse value gap', 'Fine value gap', 'Reduction (per cent)'],
        [[f"{r['training_paths']:,}", str(r['degree']), *[f'{r[k]:.6f}' for k in ('slope', 'coarse', 'fine')], f"{100*r['reduction']:.2f}"] for r in data['measurement']],
    )
    tables['terminal_adjustment'] = table(
        ['Training paths', 'Degree', 'Unadjusted gap', 'Adjusted gap', 'Paired policy gain'],
        [[f"{r['training_paths']:,}", str(r['degree']), *[f'{r[k]:.8f}' for k in ('original', 'adjusted', 'gain')]] for r in data['terminal_adjustment']],
    )
    tables['training_evaluation'] = table(
        ['Case', 'Mean gain', 'Pointwise 95 per cent interval', 'Evaluation share (per cent)'],
        [[str(r['case']), f"{r['gain']:.5f}", f"[{r['lower']:.5f}, {r['upper']:.5f}]", f"{r['evaluation_share_percent']:.3f}"] for r in data['training_evaluation']],
    )
    tables['allocation_comparison'] = table(
        ['Case', 'Selected pairs', 'Paths per pair', 'Pilot variance ratio', 'Main variance ratio', 'CPU ratio', 'Variance × CPU ratio'],
        [[str(r[k]) for k in ('case', 'selected_repetitions', 'selected_paths')] + [f'{r[k]:.3f}' for k in ('pilot_predicted_variance_ratio', 'estimated_variance_ratio', 'actual_cpu_ratio', 'estimated_variance_cpu_ratio')] for r in data['allocation_comparison']],
    )
    tables['allocation_gains'] = table(
        ['Case', 'Reference mean', 'Reference interval', 'Selected mean', 'Selected interval'],
        [[str(r['case']), f"{r['reference_mean']:.5f}", f"[{r['reference_lower']:.5f}, {r['reference_upper']:.5f}]", f"{r['selected_mean']:.5f}", f"[{r['selected_lower']:.5f}, {r['selected_upper']:.5f}]"] for r in data['allocation_gains']],
    )
    tables['allocation_costs'] = table(
        ['Component', 'Reference predicted', 'Reference actual', 'Selected predicted', 'Selected actual'],
        [[r['component'].capitalize()] + [f'{r[k]:.3f}' for k in ('reference_predicted_cpu_seconds', 'reference_actual_cpu_seconds', 'selected_predicted_cpu_seconds', 'selected_actual_cpu_seconds')] for r in data['allocation_costs']],
    )
    tables['prospective_allocation'] = table(
        ['Case', 'Pairs, 128 paths', 'Pairs, 256 paths', 'Pairs, 4,096 paths', 'Variance ratio, 128', 'Variance ratio, 256', 'CPU ratio, 256', 'Variance × CPU ratio, 256'],
        [[str(r[k]) for k in ('case', 'pairs_128', 'pairs_256', 'pairs_4096')] + [f'{r[k]:.6f}' for k in ('variance_ratio_128', 'variance_ratio_256', 'cpu_ratio_256', 'variance_cpu_ratio_256')] for r in data['prospective_allocation']],
    )
    tables['prospective_contrasts'] = table(
        ['Case', 'Evaluation paths', 'Mean-difference interval', 'Variance-difference interval'],
        [[str(r['case']), f"{r['evaluation_paths']:,}", f"[{r['mean_lower']:.7g}, {r['mean_upper']:.7g}]", f"[{r['variance_lower']:.7g}, {r['variance_upper']:.7g}]"] for r in data['prospective_contrasts']],
    )
    labels = {'historical': 'Historical', 'pass_start': 'Pass-start calibration', 'immediate': 'Immediate calibration'}
    tables['cpu_prediction'] = table(
        ['Forecast', 'Within ten per cent', 'Within upper allowance', 'Mean absolute relative error (per cent)', 'Actual / forecast range'],
        [[labels[r['predictor']], f"{r['within_tolerance']}/18", f"{r['within_allowance']}/18", f"{r['mean_absolute_relative_error_percent']:.3f}", f"[{r['actual_to_forecast_min']:.4f}, {r['actual_to_forecast_max']:.4f}]"] for r in data['cpu_prediction']],
    )
    return tables


def require(condition, message):
    if not condition:
        raise ValueError(message)


# Explicit field sets prevent additions of unreviewed input bundles to this summary.
# Their presence is a content constraint, not a general confidentiality proof.
TABLE_FIELDS = {
    'measurement': 'training_paths degree policies slope coarse fine reduction',
    'terminal_adjustment': 'training_paths degree policies original adjusted gain',
    'training_evaluation': 'case gain lower upper repetitions evaluation_share_percent',
    'allocation_comparison': 'case selected_repetitions selected_paths pilot_predicted_variance_ratio estimated_variance_ratio actual_cpu_ratio estimated_variance_cpu_ratio',
    'allocation_gains': 'case reference_mean reference_lower reference_upper selected_mean selected_lower selected_upper',
    'allocation_costs': 'component reference_predicted_cpu_seconds reference_actual_cpu_seconds selected_predicted_cpu_seconds selected_actual_cpu_seconds',
    'prospective_allocation': 'case pairs_128 pairs_256 pairs_4096 variance_ratio_128 variance_ratio_256 cpu_ratio_256 variance_cpu_ratio_256',
    'prospective_contrasts': 'case evaluation_paths mean_lower mean_upper variance_lower variance_upper',
    'cpu_prediction': 'predictor within_tolerance within_allowance mean_absolute_relative_error_percent actual_to_forecast_min actual_to_forecast_max forecast_affordable allowance_affordable falsely_forecast_affordable falsely_allowance_affordable',
}
DESIGN_FIELDS = {
    'allocation_design': 'training_paths degree pilot_pairs_per_case pilot_evaluation_paths reference_pairs_per_case reference_evaluation_paths measured_evaluation_path_grid pairs_allowed budget_tolerance reference_pairs_total selected_pairs_total actual_to_pilot_reference_budget_range arms_outside_budget_tolerance pilot_cpu_seconds pilot_elapsed_seconds ratio_confidence_intervals validated_equal_realised_budget interval_type',
    'prospective_design': 'cases evaluation_paths primary_arms primary_pairs warmup_pairs training_paths degree registered_choice contrast_family_size critical_value mean_intervals_including_zero negative_variance_intervals variance_ratio_confidence_intervals finite_sample_coverage_verified equal_realised_cost fixed_counts new_market_observations arms_within_budget arms_within_forecast_tolerance forecast_tolerance actual_to_forecast_range primary_numerical_cpu_seconds predicted_numerical_cpu_seconds largest_squared_variance_influence_percent',
    'cpu_prediction_design': 'validation_arms validation_pairs calibration_pairs warmup_pairs passes calibration_pairs_per_arm_per_scheme calibration_schemes tolerance allowance_multiplier fixed_counts prediction_coverage_claim causal_order_effect_identified new_financial_gain_inference new_market_observations relative_error_denominator averaging validation_within_original_budget within_budget_with_pass_start_calibration within_budget_with_immediate_calibration within_budget_with_both_calibration_schemes validation_numerical_cpu_seconds pass_start_calibration_cpu_seconds immediate_calibration_cpu_seconds warmup_numerical_cpu_seconds total_numerical_cpu_seconds process_cpu_seconds process_elapsed_seconds',
}


def check_fields(data):
    metadata = set('schema_version title date scope monetary_unit rounding'.split())
    require(set(data) == metadata | set(TABLE_FIELDS) | set(DESIGN_FIELDS), 'Unexpected or missing summary field')
    for name, fields in TABLE_FIELDS.items():
        require(isinstance(data[name], list) and data[name], f'{name}: expected non-empty rows')
        for row in data[name]:
            require(isinstance(row, dict) and set(row) == set(fields.split()), f'{name}: unexpected or missing row field')
    for name, fields in DESIGN_FIELDS.items():
        require(isinstance(data[name], dict) and set(data[name]) == set(fields.split()), f'{name}: unexpected or missing design field')
    influence = data['prospective_design']['largest_squared_variance_influence_percent']
    require(isinstance(influence, list) and len(influence) == 2, 'Influence summary coverage mismatch')
    for row in influence:
        require(set(row) == {'case', 'evaluation_paths', 'percent'}, 'Unexpected influence field')
        require(0 <= row['percent'] <= 100, 'Influence share is not a percentage')
    def finite(value):
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
        elif isinstance(value, (int, float)):
            require(math.isfinite(value), 'Non-finite released number')
    finite(data)


def check_extensions(data):
    allocation = data['prospective_allocation']
    design = data['prospective_design']
    require([r['case'] for r in allocation] == design['cases'] == [5, 6, 10], 'Prospective case coverage mismatch')
    require(design['evaluation_paths'] == [128, 256, 4096], 'Prospective path grid mismatch')
    require(design['training_paths'] == [5000, 20000] and design['degree'] == 4, 'Training procedures changed')
    require(design['registered_choice'] == 256, 'Registered choice changed')
    require(design['primary_arms'] == 9 and design['primary_pairs'] == 3978 and design['warmup_pairs'] == 4, 'Prospective count mismatch')
    require(sum(r[k] for r in allocation for k in ('pairs_128', 'pairs_256', 'pairs_4096')) == design['primary_pairs'], 'Prospective pair sum mismatch')
    expected_counts = [(776, 687, 91), (674, 632, 141), (438, 399, 140)]
    require([tuple(r[k] for k in ('pairs_128', 'pairs_256', 'pairs_4096')) for r in allocation] == expected_counts, 'Registered per-arm counts changed')
    for r in allocation:
        require(all(r[k] > 0 for k in ('variance_ratio_128', 'variance_ratio_256', 'cpu_ratio_256', 'variance_cpu_ratio_256')), 'Non-positive prospective ratio')
        require(abs(r['variance_ratio_256'] * r['cpu_ratio_256'] - r['variance_cpu_ratio_256']) < 2e-6, 'Rounded prospective variance-cost identity mismatch')
    contrasts = data['prospective_contrasts']
    require([(r['case'], r['evaluation_paths']) for r in contrasts] == [(c, n) for c in (5, 6, 10) for n in (128, 256)], 'Registered contrast coverage mismatch')
    require(design['contrast_family_size'] == 2*len(contrasts) == 12 and design['critical_value'] == 3.25, 'Contrast family mismatch')
    require(all(r['mean_lower'] <= r['mean_upper'] and r['variance_lower'] <= r['variance_upper'] for r in contrasts), 'Reversed contrast interval')
    require(sum(r['mean_lower'] <= 0 <= r['mean_upper'] for r in contrasts) == design['mean_intervals_including_zero'] == 6, 'Mean interval classification mismatch')
    require(sum(r['variance_upper'] < 0 for r in contrasts) == design['negative_variance_intervals'] == 6, 'Variance interval classification mismatch')
    require(design['forecast_tolerance'] == [0.9, 1.1] and design['arms_within_budget'] == 9 and design['arms_within_forecast_tolerance'] == 0, 'Prospective CPU classification mismatch')
    require(0 < design['actual_to_forecast_range'][0] <= design['actual_to_forecast_range'][1] < .9, 'Prospective forecast range mismatch')
    require(design['fixed_counts'] is True and design['new_market_observations'] == 0, 'Prospective scope changed')
    require(all(design[k] is False for k in ('variance_ratio_confidence_intervals', 'finite_sample_coverage_verified', 'equal_realised_cost')), 'Unsupported prospective inference claim')
    timing = data['cpu_prediction_design']
    require(timing['passes'] == 2 and timing['validation_arms'] == 2*design['primary_arms'] == 18, 'Timing pass coverage mismatch')
    require(timing['validation_pairs'] == 2*design['primary_pairs'] == 7956 and timing['warmup_pairs'] == 4, 'Timing validation count mismatch')
    require(timing['calibration_pairs_per_arm_per_scheme'] == 16 and timing['calibration_schemes'] == 2, 'Calibration design changed')
    require(timing['calibration_pairs'] == timing['validation_arms']*timing['calibration_schemes']*timing['calibration_pairs_per_arm_per_scheme'] == 576, 'Calibration count mismatch')
    require(timing['tolerance'] == [.9, 1.1] and timing['allowance_multiplier'] == 1.1, 'Timing tolerance changed')
    require(timing['relative_error_denominator'] == 'forecast', 'Relative error denominator changed')
    require(timing['fixed_counts'] is True and timing['new_market_observations'] == 0, 'Timing scope changed')
    require(all(timing[k] is False for k in ('prediction_coverage_claim', 'causal_order_effect_identified', 'new_financial_gain_inference')), 'Unsupported timing inference claim')
    rows = data['cpu_prediction']
    require([r['predictor'] for r in rows] == ['historical', 'pass_start', 'immediate'], 'Predictor coverage mismatch')
    count_fields = ('within_tolerance', 'within_allowance', 'forecast_affordable', 'allowance_affordable', 'falsely_forecast_affordable', 'falsely_allowance_affordable')
    for r in rows:
        require(all(type(r[k]) is int and 0 <= r[k] <= 18 for k in count_fields), 'Invalid forecast count')
        require(r['within_tolerance'] <= r['within_allowance'], 'Tolerance count exceeds upper-allowance count')
        require(r['falsely_forecast_affordable'] <= r['forecast_affordable'] and r['falsely_allowance_affordable'] <= r['allowance_affordable'], 'False affordability count exceeds declarations')
        require(r['actual_to_forecast_min'] > 0 and r['actual_to_forecast_max'] >= r['actual_to_forecast_min'] and r['mean_absolute_relative_error_percent'] >= 0, 'Invalid forecast ratio or error')
    require([(r['within_tolerance'], r['within_allowance']) for r in rows] == [(1, 18), (16, 16), (17, 17)], 'Forecast classification changed')
    require([r['forecast_affordable'] for r in rows] == [18, 18, 18] and [r['allowance_affordable'] for r in rows] == [0, 18, 17], 'Affordability declarations changed')
    require([r['falsely_forecast_affordable'] for r in rows] == [1, 1, 1] and [r['falsely_allowance_affordable'] for r in rows] == [0, 1, 1], 'False affordability declarations changed')
    require([timing[k] for k in ('validation_within_original_budget', 'within_budget_with_pass_start_calibration', 'within_budget_with_immediate_calibration', 'within_budget_with_both_calibration_schemes')] == [17, 17, 17, 14], 'Calibration expenditure categories changed')
    components = ('validation_numerical_cpu_seconds', 'pass_start_calibration_cpu_seconds', 'immediate_calibration_cpu_seconds', 'warmup_numerical_cpu_seconds')
    require(all(timing[k] >= 0 for k in components), 'Negative CPU component')
    require(abs(sum(timing[k] for k in components) - timing['total_numerical_cpu_seconds']) < .003, 'Rounded numerical CPU total mismatch')
    require(timing['process_cpu_seconds'] >= timing['total_numerical_cpu_seconds'] and timing['process_elapsed_seconds'] > 0, 'Process accounting mismatch')


def check(data):
    check_fields(data)
    require(data['schema_version'] == 2, 'Unsupported summary schema')
    for key in ('training_evaluation', 'allocation_comparison', 'allocation_gains'):
        require([r['case'] for r in data[key]] == list(range(13)), f'{key}: missing or reordered case')
    for key in ('measurement', 'terminal_adjustment'):
        require([(r['training_paths'], r['degree']) for r in data[key]] == [(5000, 2), (5000, 4), (20000, 2), (20000, 4)], f'{key}: configuration mismatch')
        require(all(r['policies'] == 52 for r in data[key]), f'{key}: policy denominator mismatch')
    for r in data['measurement']:
        require(abs(1-r['fine']/r['slope']-r['reduction']) < 3e-6, 'Rounded gap reduction mismatch')
    for r in data['terminal_adjustment']:
        require(abs(r['original']-r['adjusted']-r['gain']) < 2e-8, 'Rounded common-upper cancellation mismatch')
    design = data['allocation_design']
    require(sum(r['selected_repetitions'] for r in data['allocation_comparison']) == design['selected_pairs_total'], 'Selected pair total mismatch')
    require(13*design['reference_pairs_per_case'] == design['reference_pairs_total'], 'Reference pair total mismatch')
    for r in data['allocation_comparison']:
        require(r['selected_paths'] in design['measured_evaluation_path_grid'], 'Unmeasured path allocation')
        require(design['pairs_allowed'][0] <= r['selected_repetitions'] <= design['pairs_allowed'][1], 'Pair allocation outside design')
        # Each of the three factors is rounded to three decimal places.
        require(abs(r['estimated_variance_ratio']*r['actual_cpu_ratio']-r['estimated_variance_cpu_ratio']) < .002, 'Rounded variance-cost identity mismatch')
    for r in data['training_evaluation']:
        require(r['lower'] <= r['gain'] <= r['upper'], 'Training interval does not contain its estimate')
        require(0 <= r['evaluation_share_percent'] <= 100, 'Variance share is not a percentage')
    for r in data['allocation_gains']:
        for arm in ('reference', 'selected'):
            require(r[f'{arm}_lower'] <= r[f'{arm}_mean'] <= r[f'{arm}_upper'], 'Allocation interval does not contain its estimate')
    require([r['component'] for r in data['allocation_costs']] == ['setup', 'training', 'evaluation'], 'Cost component mismatch')
    for r in data['allocation_costs']:
        require(all(v >= 0 for k, v in r.items() if k != 'component'), 'Negative CPU charge')
    check_extensions(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Refresh only the nine marked Markdown tables')
    args = parser.parse_args()
    data = json.loads((PAPER/'numerical_results.json').read_text())
    check(data)
    path = PAPER/'numerical_results.md'
    text = path.read_text()
    updated = text
    for key, contents in render(data).items():
        start, end = f'<!-- table:{key} -->', f'<!-- endtable:{key} -->'
        require(updated.count(start) == updated.count(end) == 1, f'Missing or repeated table marker: {key}')
        before, tail = updated.split(start)
        _, after = tail.split(end)
        updated = before + start + '\n\n' + contents + '\n\n' + end + after
    if args.write:
        path.write_text(updated)
    else:
        require(text == updated, 'Published tables differ from the released summaries; inspect before using --write')
    print('Verified nine summary tables, declared fields, study counts and rounded arithmetic identities. No experiment rerun.')


if __name__ == '__main__':
    main()
