"""Build private paper assets from verified allocation reports and timing evidence."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from report_training_allocation import report_values

ARMS = ('reference', 'selected')
COMPONENTS = ('setup', 'training', 'evaluation')


def _close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-10):
        raise ValueError(f'Allocation asset evidence differs: {name}')


def _checked(folder, groups=('input_sha256', 'source_sha256')):
    manifest = read(folder/'manifest.json')
    for group in groups:
        verify_mapping(manifest[group])
    verify_mapping(manifest['artifact_sha256'], folder)
    return manifest


def preflight(report, audit, timing):
    """Bind all retained inputs and sources before creating any asset directory."""
    report, audit, timing = map(private, map(Path, (report, audit, timing)))
    report_manifest = _checked(report)
    audit_groups = ('input_sha256', 'registered_source_sha256',
                    'additional_audit_source_sha256', 'audit_source_sha256')
    audit_manifest = _checked(audit, audit_groups)
    timing_manifest = _checked(timing)
    if (report_manifest.get('complete') is not True or audit_manifest.get('complete') is not True
            or audit_manifest.get('registration_unchanged') is not True
            or timing_manifest.get('complete') is not True):
        raise ValueError('Completed report, audit and timing evidence are required')
    for folder, manifest, names in (
            (report, report_manifest, ('values.json', 'allocation_comparison.png')),
            (audit, audit_manifest, ('verification.json',)),
            (timing, timing_manifest, ('diagnostic.json',))):
        if not set(names).issubset(manifest['artifact_sha256']):
            raise ValueError(f'Required asset input is not bound by {folder.name} manifest')
    main = private(ROOT/report_manifest['main_run'])
    if report_manifest['audit'] != str(audit.relative_to(ROOT)):
        raise ValueError('Report identifies a different main audit')
    main_manifest = _checked(main)
    registration = private(ROOT/main_manifest['registration'])
    registered = _checked(registration)
    if main_manifest.get('complete') is not True or registered.get('stage') != 'main_allocation':
        raise ValueError('A completed main run and frozen allocation are required')
    if main_manifest['registration_sha256'] != sha(registration/'manifest.json'):
        raise ValueError('Main registration identity differs')
    pilot = private(ROOT/registered['pilot'])
    pilot_manifest = _checked(pilot)
    if pilot_manifest.get('complete') is not True:
        raise ValueError('A completed pilot is required')
    cfg = read(registration/'configuration.json')
    verification = read(audit/'verification.json')
    main_name = str(main.relative_to(ROOT))
    main_hash = sha(main/'manifest.json')
    if (verification['main_run'] != main_name or verification['main_manifest_sha256'] != main_hash
            or audit_manifest['input_sha256'].get(main_name+'/manifest.json') != main_hash):
        raise ValueError('Retained audit does not identify the supplied main report')
    values = read(report/'values.json')
    reconstructed = report_values(read(main/'summary.json'), cfg)
    reconstructed['main_stage_execution'] = dict(process_cpu_seconds=main_manifest['process_cpu_seconds'],
                                                elapsed_seconds=main_manifest['elapsed_seconds'])
    reconstructed['evidence'] = dict(main_run=main_name, main_manifest_sha256=main_hash,
        audit=str(audit.relative_to(ROOT)), audit_manifest_sha256=sha(audit/'manifest.json'))
    if reconstructed != values:
        raise ValueError('Report values differ from their retained main evidence')
    diagnostic = read(timing/'diagnostic.json')
    if diagnostic['input_sha256'] != timing_manifest['input_sha256']:
        raise ValueError('Timing diagnostic input mapping differs')
    verify_mapping(diagnostic['source_sha256'])
    for folder, manifest in ((main, main_manifest), (pilot, pilot_manifest)):
        name = str((folder/'manifest.json').relative_to(ROOT))
        if timing_manifest['input_sha256'].get(name) != sha(folder/'manifest.json'):
            raise ValueError('Timing diagnostic belongs to a different execution')
    if any(record.get('publication_clearance') is not False for record in
           (report_manifest, audit_manifest, timing_manifest, values, diagnostic)):
        raise ValueError('The asset builder requires the retained private publication status')
    inputs = {}
    for folder in (report, audit, timing, main, pilot, registration):
        inputs[str((folder/'manifest.json').relative_to(ROOT))] = sha(folder/'manifest.json')
    for path in (report/'values.json', report/'allocation_comparison.png', audit/'verification.json',
                 timing/'diagnostic.json', main/'summary.json', registration/'configuration.json'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    sources = {}
    for manifest, groups in ((report_manifest, ('source_sha256',)),
                             (audit_manifest, audit_groups[1:]),
                             (timing_manifest, ('source_sha256',))):
        for group in groups:
            for name, digest in manifest[group].items():
                if name in sources and sources[name] != digest:
                    raise ValueError('Asset source closure contains inconsistent hashes')
                sources[name] = digest
    source = Path(__file__).resolve()
    sources[str(source.relative_to(ROOT))] = sha(source)
    verify_mapping(inputs)
    verify_mapping(sources)
    evidence = dict(report=str(report.relative_to(ROOT)), audit=str(audit.relative_to(ROOT)),
                    timing=str(timing.relative_to(ROOT)), main_run=main_name,
                    pilot=str(pilot.relative_to(ROOT)), main_manifest_sha256=main_hash)
    return values, diagnostic, cfg, pilot_manifest, main_manifest, inputs, sources, evidence


def derive(values, diagnostic, cfg, pilot_manifest, main_manifest):
    """Derive table entries and macro values without inserting empirical constants."""
    models = values['models']
    if len(models) != 13 or [model['case'] for model in models] != list(range(13)):
        raise ValueError('The paper requires all thirteen retained cases in order')
    rows, gain_rows = [], []
    for model in models:
        selected = model['arms']['selected']
        rows.append(dict(case=model['case'], selected_repetitions=selected['repetitions'],
                         selected_paths=selected['evaluation_paths'], **model['comparison']))
        gain_rows.append(dict(case=model['case'], **{
            f'{arm}_{name}': model['arms'][arm]['gain_'+name]
            for arm in ARMS for name in ('mean', 'lower', 'upper')}))
    components = {(row['case'], row['arm']): row for row in diagnostic['components']}
    if len(components) != len(diagnostic['components']) or set(components) != {
            (model['case'], arm) for model in models for arm in ARMS}:
        raise ValueError('Timing decomposition coverage differs')
    for model in models:
        for arm in ARMS:
            part = components[model['case'], arm]
            for category, report_key in (('predicted_cpu_seconds', 'pilot_predicted_cpu_seconds'),
                                          ('actual_cpu_seconds', 'actual_cpu_seconds')):
                _close(sum(part[category][name] for name in COMPONENTS),
                       model['arms'][arm][report_key], f'case {model["case"]} {arm} {category}')
    cost_rows = []
    for component in COMPONENTS:
        row = dict(component=component)
        for arm in ARMS:
            for category in ('predicted', 'actual'):
                value = sum(components[model['case'], arm][category+'_cpu_seconds'][component]
                            for model in models)
                _close(value, diagnostic['totals'][arm][category+'_cpu_seconds'][component],
                       f'{arm} {category} {component} total')
                row[f'{arm}_{category}_cpu_seconds'] = value
        cost_rows.append(row)
    totals = {arm: sum(model['arms'][arm]['repetitions'] for model in models) for arm in ARMS}
    positive = {arm: sum(model['arms'][arm]['gain_lower'] > 0 for model in models) for arm in ARMS}
    budget_ratios = [model['arms'][arm]['actual_to_budget_ratio'] for model in models for arm in ARMS]
    failures = sum(model['arms'][arm]['material_budget_mismatch'] for model in models for arm in ARMS)
    for arm in ARMS:
        if (totals[arm] != values['totals'][arm]['repetitions']
                or positive[arm] != values['descriptive_counts']['positive_pointwise_lower_bounds'][arm]['positive']):
            raise ValueError('Derived paper counts differ from the audited report')
    for model in models:
        reference = model['arms']['reference']
        if (reference['repetitions'] != cfg['reference_repetitions']
                or reference['evaluation_paths'] != cfg['reference_paths']):
            raise ValueError('Reference counts differ from the frozen configuration')
    macros = dict(AllocationModels=len(models),
        AllocationPilotRepetitions=cfg['pilot_repetitions'], AllocationPilotPaths=cfg['pilot_paths'],
        AllocationReferenceRepetitions=cfg['reference_repetitions'], AllocationReferencePaths=cfg['reference_paths'],
        AllocationSelectedRepetitionsMin=min(row['selected_repetitions'] for row in rows),
        AllocationSelectedRepetitionsMax=max(row['selected_repetitions'] for row in rows),
        AllocationSelectedPathsMin=min(row['selected_paths'] for row in rows),
        AllocationSelectedPathsMax=max(row['selected_paths'] for row in rows),
        AllocationReferencePairs=totals['reference'], AllocationSelectedPairs=totals['selected'],
        AllocationTotalPairs=sum(totals.values()), AllocationBudgetRatioMin=min(budget_ratios),
        AllocationBudgetRatioMax=max(budget_ratios), AllocationBudgetFailures=failures,
        AllocationReferenceCpu=values['totals']['reference']['actual_cpu_seconds'],
        AllocationSelectedCpu=values['totals']['selected']['actual_cpu_seconds'],
        AllocationPredictedReferenceCpu=values['totals']['reference']['pilot_predicted_cpu_seconds'],
        AllocationPredictedSelectedCpu=values['totals']['selected']['pilot_predicted_cpu_seconds'],
        AllocationPositiveReferenceIntervals=positive['reference'],
        AllocationPositiveSelectedIntervals=positive['selected'],
        AllocationPilotCpu=pilot_manifest['process_cpu_seconds'],
        AllocationPilotElapsed=pilot_manifest['elapsed_seconds'],
        AllocationMainCpu=main_manifest['process_cpu_seconds'],
        AllocationMainElapsed=main_manifest['elapsed_seconds'])
    for stem, key in (('PredictedVariance', 'pilot_predicted_variance_ratio'),
                      ('Variance', 'estimated_variance_ratio'), ('Cpu', 'actual_cpu_ratio'),
                      ('VarianceCpu', 'estimated_variance_cpu_ratio')):
        ratios = [row[key] for row in rows if row[key] is not None]
        if not ratios:
            raise ValueError('A requested paper ratio has no defined observations')
        macros[f'Allocation{stem}RatioMin'] = min(ratios)
        macros[f'Allocation{stem}RatioMax'] = max(ratios)
        if stem in ('Variance', 'VarianceCpu'):
            macros[f'Allocation{stem}BelowOne'] = sum(value < 1 for value in ratios)
    for row in cost_rows:
        for arm in ARMS:
            for category in ('predicted', 'actual'):
                macros[f'Allocation{arm.title()}{category.title()}{row["component"].title()}Cpu'] = row[f'{arm}_{category}_cpu_seconds']
    if any(not re.fullmatch(r'Allocation[A-Za-z]+', name) for name in macros):
        raise ValueError('Allocation macro names must be unique alphabetic control sequences')
    return dict(macros=macros, comparison_rows=rows, cost_rows=cost_rows, gain_rows=gain_rows,
                descriptive_ratio_estimates=True, ratio_confidence_intervals=False,
                validated_equal_budget=False, global_optimum_claim=False,
                publication_clearance=False)


def number(value):
    if value is None:
        return r'\textnormal{n/a}'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Only finite numerical values can enter the paper assets')
    return str(value) if type(value) is int else f'{value:.3f}'


def comparison_table(rows):
    lines = [r'\begin{tabular}{rrrrrrr}', r'\toprule',
        r'Case & $R_{\mathrm{sel}}$ & $N_{\mathrm{sel}}$ & \shortstack{Pilot variance\\ratio} & \shortstack{Main variance\\ratio} & \shortstack{Actual CPU\\ratio} & \shortstack{Variance $\times$ CPU\\ratio} \\',
        r'\midrule']
    keys = ('case', 'selected_repetitions', 'selected_paths', 'pilot_predicted_variance_ratio',
            'estimated_variance_ratio', 'actual_cpu_ratio', 'estimated_variance_cpu_ratio')
    lines += [' & '.join(number(row[key]) for key in keys)+r' \\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', ''])


def cost_table(rows):
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
        r' & \multicolumn{2}{c}{Reference} & \multicolumn{2}{c}{Selected} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'Component & Predicted & Actual & Predicted & Actual \\',
        r'\midrule']
    keys = ('reference_predicted_cpu_seconds', 'reference_actual_cpu_seconds',
            'selected_predicted_cpu_seconds', 'selected_actual_cpu_seconds')
    lines += [row['component'].title()+' & '+' & '.join(number(row[key]) for key in keys)+r' \\' for row in rows]
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', ''])


def gain_table(rows):
    lines = [r'\begin{tabular}{rrrrr}', r'\toprule',
        r'Case & \shortstack{Reference\\mean} & \shortstack{Reference\\95\% interval} & \shortstack{Selected\\mean} & \shortstack{Selected\\95\% interval} \\',
        r'\midrule']
    for row in rows:
        fields = [str(row['case'])]
        for arm in ARMS:
            fields.extend((f'{row[arm+"_mean"]:.5f}',
                           f'$[{row[arm+"_lower"]:.5f}, {row[arm+"_upper"]:.5f}]$'))
        lines.append(' & '.join(fields)+r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}', ''])


def paper_figure(values, path):
    """Plot verified ratios in three vertical panels for a paper-width figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = [row['case'] for row in values['models']]
    blue, orange = '#17618d', '#c45b28'
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8}):
        figure, axes = plt.subplots(3, 1, figsize=(6.4, 7.5), layout='constrained')
        for axis, title in zip(axes, ('(a) Variance ratios', '(b) Actual CPU ratio',
                                     '(c) Actual cost relative to the predicted budget')):
            axis.set_title(title, loc='left')
            axis.set_xticks(cases)
            axis.set_xlim(min(cases)-.5, max(cases)+.5)
            axis.set_xlabel('Retained model (case)')
            axis.axhline(1., color='#5c6268', linewidth=.8, linestyle='--', zorder=1)
            axis.grid(axis='y', color='#e4e7eb', linewidth=.6)
            axis.spines[['top', 'right']].set_visible(False)
        for key, label, colour, marker, offset in (
                ('pilot_predicted_variance_ratio', 'Pilot prediction', blue, '^', -.07),
                ('estimated_variance_ratio', 'Main estimate', orange, 'o', .07)):
            points = [(row['case']+offset, row['comparison'][key]) for row in values['models']
                      if row['comparison'][key] is not None]
            axes[0].scatter([x for x, _ in points], [y for _, y in points],
                            s=27, color=colour, marker=marker, label=label, zorder=3)
        axes[0].set_ylabel('Selected / reference variance')
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc='upper right', frameon=False, ncol=2)
        axes[1].scatter(cases, [row['comparison']['actual_cpu_ratio'] for row in values['models']],
                        s=27, color=blue, zorder=3)
        axes[1].set_ylabel('Selected / reference CPU')
        axes[1].set_ylim(bottom=0)
        lower, upper = values['budget_ratio_limits']
        axes[2].axhspan(lower, upper, color='#dfece3', zorder=0, label='Registered tolerance')
        for arm, label, colour, marker, offset in (
                ('reference', 'Reference', blue, '^', -.07),
                ('selected', 'Selected', orange, 'o', .07)):
            axes[2].scatter([case+offset for case in cases],
                [row['arms'][arm]['actual_to_budget_ratio'] for row in values['models']],
                s=27, color=colour, marker=marker, label=label, zorder=3)
        axes[2].set_ylabel('Actual / predicted budget')
        axes[2].set_ylim(0, max(upper*1.12, axes[2].get_ylim()[1]))
        axes[2].legend(loc='lower right', frameon=False, ncol=3)
        figure.savefig(path, dpi=220)
        plt.close(figure)
    return matplotlib.__version__


def build(report, audit, timing, output):
    """Write tables, macros and an unchanged figure into a fresh private folder."""
    report, audit, timing, output = map(private, map(Path, (report, audit, timing, output)))
    if output.exists():
        raise FileExistsError('Use a fresh private paper-asset directory')
    if any(output.is_relative_to(folder) for folder in (report, audit, timing)):
        raise ValueError('Keep generated assets separate from retained evidence')
    values, diagnostic, cfg, pilot, main, inputs, sources, evidence = preflight(report, audit, timing)
    assets = derive(values, diagnostic, cfg, pilot, main)
    assets['evidence'] = evidence
    assets['macro_text'] = {name: number(value) for name, value in assets['macros'].items()}
    comparison = comparison_table(assets['comparison_rows'])
    costs = cost_table(assets['cost_rows'])
    gains = gain_table(assets['gain_rows'])
    macro_text = '\n'.join('\\newcommand{\\'+name+'}{'+value+'}'
                           for name, value in sorted(assets['macro_text'].items()))+'\n'
    # Every retained hash and derived value above is checked before mkdir.
    output.mkdir(parents=True)
    (output/'tables').mkdir()
    (output/'figures').mkdir()
    (output/'tables/allocation_comparison.tex').write_text(comparison)
    (output/'tables/allocation_costs.tex').write_text(costs)
    (output/'tables/allocation_gains.tex').write_text(gains)
    (output/'allocation_values.tex').write_text(macro_text)
    save(output/'allocation_values.json', assets)
    shutil.copyfile(report/'allocation_comparison.png', output/'figures/allocation_comparison.png')
    if sha(output/'figures/allocation_comparison.png') != sha(report/'allocation_comparison.png'):
        raise ValueError('The copied private comparison figure differs')
    matplotlib_version = paper_figure(values, output/'figures/allocation_paper.png')
    verify_mapping(inputs)
    verify_mapping(sources)
    files = sorted(path for path in output.rglob('*') if path.is_file())
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={str(path.relative_to(output)): sha(path) for path in files},
        complete=True, numerical_reruns=False, figure_copied_without_changes=True,
        matplotlib=matplotlib_version, paper_figure_generated_from_verified_values=True,
        publication_clearance=False))
    return assets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'audit', 'timing', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    arguments = parser.parse_args()
    result = build(arguments.report, arguments.audit, arguments.timing, arguments.output)
    print(f'Built private allocation assets for {result["macros"]["AllocationModels"]} retained cases.')
