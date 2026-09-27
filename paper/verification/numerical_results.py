"""Check and render the released, rounded research-summary tables.

This verifies presentation and arithmetic within the supplement. It neither
recreates the simulations nor verifies findings against restricted observations.
Use --write to refresh marked tables after an authorised summary revision.
"""

import argparse
import json
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
    return tables


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(data):
    require(data['schema_version'] == 1, 'Unsupported summary schema')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Refresh only the six marked Markdown tables')
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
    print('Verified six summary tables, case coverage and rounded arithmetic identities. No experiment rerun.')


if __name__ == '__main__':
    main()
