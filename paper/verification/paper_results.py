"""Check the standalone paper's tables against the released aggregate summary.

Use --write to update only marked table blocks. This checks their transcription,
not the financial validity or statistical coverage of the retained experiments.
"""

import argparse
import json
from pathlib import Path
import re

from numerical_results import check

PAPER = Path(__file__).resolve().parents[1]


def number(value, precision=6):
    text = f'{value:.{precision}g}'
    if 'e' in text:
        mantissa, exponent = text.split('e')
        return mantissa + r'\times10^{' + str(int(exponent)) + '}'
    return text


def interval(lower, upper):
    return '$[' + number(lower, 7) + ',' + number(upper, 7) + ']$'


def table(caption, label, headers, rows, note, align=None):
    align = align or 'l' + 'r' * (len(headers)-1)
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{' + caption + r'}\label{tab:' + label + '}',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{' + align + '}',
        r'\toprule', ' & '.join(headers) + r' \\', r'\midrule',
        *[' & '.join(row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize ' + note + r'\end{minipage}',
        r'\end{table}',
    ])


def render(data):
    check(data)
    result = {}
    stack = lambda *parts: r'\shortstack{' + r'\\'.join(parts) + '}'
    result['measurement'] = table('Expected-gap estimates for unchanged exercise policies', 'measurement',
        [stack('Training','paths'), 'Degree', 'Slope', 'Coarse value', 'Fine value', stack('Reduction','(per cent)')],
        [[f"{r['training_paths']:,}", str(r['degree']), *[f'{r[k]:.6f}' for k in ('slope','coarse','fine')], f"{100*r['reduction']:.2f}"] for r in data['measurement']],
        'Each entry averages 52 policies. Gaps are USD per share in premium-quote units and contain both policy loss and dual slack. Reduction compares the fine value construction with the slope construction.')
    result['terminal_adjustment'] = table('Terminal adjustment under a common upper payoff', 'terminal-adjustment',
        [stack('Training','paths'), 'Degree', 'Unadjusted gap', 'Adjusted gap', 'Policy gain'],
        [[f"{r['training_paths']:,}", str(r['degree']), *[f'{r[k]:.8f}' for k in ('original','adjusted','gain')]] for r in data['terminal_adjustment']],
        'Means across 52 original policies per configuration, in premium-quote units. Both instructions are assessed on the expanded calendar. The common upper term cancels in their paired difference.')
    result['training_evaluation'] = table('Mean gains across independent training pairs', 'training-evaluation',
        ['Case', 'Mean gain', 'Pointwise interval', stack('Evaluation share','(per cent)')],
        [[str(r['case']), f"{r['gain']:.5f}", f"$[{r['lower']:.5f},{r['upper']:.5f}]$", f"{r['evaluation_share_percent']:.3f}"] for r in data['training_evaluation']],
        'Eight pairs per case; 80,000 common evaluation paths per pair. Intervals use a pointwise 95 per cent Student approximation across pair means. Finite-sample coverage is unverified; the thirteen intervals are not simultaneous.')
    result['allocation_comparison'] = table('Independent allocation comparison', 'allocation-comparison',
        ['Case', 'Pairs', 'Paths', stack('Pilot','variance'), stack('Main','variance'), 'CPU', stack('Variance','times CPU')],
        [[str(r[k]) for k in ('case','selected_repetitions','selected_paths')] + [f'{r[k]:.3f}' for k in ('pilot_predicted_variance_ratio','estimated_variance_ratio','actual_cpu_ratio','estimated_variance_cpu_ratio')] for r in data['allocation_comparison']],
        'Counts describe the selected allocation. Ratios are selected relative to the independent reference, which uses 64 pairs and 4,096 paths. Variance refers to the estimated variance of the mean gain. Ratios are descriptive and have no ratio confidence intervals.')
    result['allocation_gains'] = table('Training-procedure gains under the two allocations', 'allocation-gains',
        ['Case', 'Reference mean', 'Reference interval', 'Selected mean', 'Selected interval'],
        [[str(r['case']), f"{r['reference_mean']:.5f}", f"$[{r['reference_lower']:.5f},{r['reference_upper']:.5f}]$", f"{r['selected_mean']:.5f}", f"$[{r['selected_lower']:.5f},{r['selected_upper']:.5f}]$"] for r in data['allocation_gains']],
        'Premium-quote units. Intervals use a pointwise 95 per cent Student approximation over independent pair means. Positive intervals concern the fixed conditional specifications, not independent market confirmations.')
    result['allocation_costs'] = table('Predicted and actual allocation CPU expenditure', 'allocation-costs',
        ['Component', stack('Reference','predicted'), stack('Reference','actual'), stack('Selected','predicted'), stack('Selected','actual')],
        [[r['component'].capitalize()] + [f'{r[k]:.3f}' for k in ('reference_predicted_cpu_seconds','reference_actual_cpu_seconds','selected_predicted_cpu_seconds','selected_actual_cpu_seconds')] for r in data['allocation_costs']],
        'Aggregate seconds. Component predictions retain the pilot observation supplying the median total pair cost. The separate pilot expense is excluded from this table and reported in the text.')
    result['prospective_allocation'] = table('Prespecified computational allocation follow-up', 'prospective-allocation',
        ['Case', stack('Pairs','128'), stack('Pairs','256'), stack('Pairs','4,096'), stack('Variance','128'), stack('Variance','256'), stack('CPU','256'), stack('Variance','times CPU','256')],
        [[str(r[k]) for k in ('case','pairs_128','pairs_256','pairs_4096')] + [f'{r[k]:.6f}' for k in ('variance_ratio_128','variance_ratio_256','cpu_ratio_256','variance_cpu_ratio_256')] for r in data['prospective_allocation']],
        'Column suffixes give evaluation paths per pair. Ratios use the fresh within-case 4,096-path reference. The prespecified choice is 256 paths. Ratios and comparisons between 128 and 256 paths remain descriptive; realised costs differ.')
    result['prospective_contrasts'] = table('Registered mean and estimator-variance difference intervals', 'prospective-contrasts',
        ['Case', 'Paths', 'Mean difference', 'Variance difference'],
        [[str(r['case']), f"{r['evaluation_paths']:,}", interval(r['mean_lower'],r['mean_upper']), interval(r['variance_lower'],r['variance_upper'])] for r in data['prospective_contrasts']],
        'Each contrast subtracts the independent 4,096-path reference. Mean differences use premium-quote units and variance differences their square. The common cutoff 13/4 supplies asymptotic family calibration for twelve contrasts; finite-sample coverage is unverified.')
    labels = {'historical':'Historical','pass_start':'Pass start','immediate':'Immediate'}
    result['cpu_prediction'] = table('Historical and locally calibrated CPU forecasts', 'cpu-prediction',
        ['Forecast', stack('Within','ten per cent'), stack('Within upper','allowance'), stack('Mean absolute error','(per cent of forecast)'), stack('Actual / forecast','range')],
        [[labels[r['predictor']], f"{r['within_tolerance']}/18", f"{r['within_allowance']}/18", f"{r['mean_absolute_relative_error_percent']:.3f}", f"$[{r['actual_to_forecast_min']:.4f},{r['actual_to_forecast_max']:.4f}]$"] for r in data['cpu_prediction']],
        'Each forecast is assessed on the same eighteen complete validation charges. The error denominator is the forecast; arms receive equal weight. The upper allowance is 1.1 times the forecast and has no statistical coverage guarantee.')
    return result


def synchronise(source, tables, write=False):
    for name, expected in tables.items():
        start, end = f'% Begin result-table:{name}', f'% End result-table:{name}'
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError(f'Missing or repeated table marker: {name}')
        before, rest = source.split(start); found, after = rest.split(end)
        if not write and found.strip() != expected:
            raise ValueError(f'Paper table differs from the released summary: {name}')
        if write:
            source = before + start + '\n' + expected + '\n' + end + after
    return source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    data = json.loads((PAPER/'numerical_results.json').read_text())
    path = PAPER/'main.tex'
    source = synchronise(path.read_text(), render(data), args.write)
    if args.write:
        path.write_text(source)
    print('Verified all nine paper tables against the released summaries; no experiment rerun.')
