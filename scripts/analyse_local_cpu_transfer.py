"""Summarise audited local CPU predictions without selecting or refitting cells."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_analysis(run, audit, records, output):
    run, audit, records, output = (p.resolve() for p in (run,audit,records,output))
    if output.exists() or not all(p.is_relative_to(ROOT/'data/private') for p in (run,audit,records,output)):
        raise ValueError('Fresh private output and private evidence are required')
    evidence = {}
    for path in (audit/'checks.json',records/'checks.json'):
        record=read(path)
        if record['complete'] is not True:
            raise ValueError('Completed audits are required')
        evidence[str(path.relative_to(ROOT))]=sha(path)
        for name,digest in record['evidence_sha256'].items():
            p=(ROOT/name).resolve()
            if not p.is_relative_to(ROOT) or sha(p)!=digest:
                raise ValueError('Changed audit evidence')
            if name in evidence and evidence[name]!=digest:
                raise ValueError('Conflicting evidence')
            evidence[name]=digest
    summary,manifest=read(run/'summary.json'),read(run/'manifest.json')
    if evidence.get(str((run/'summary.json').relative_to(ROOT))) != sha(run/'summary.json'):
        raise ValueError('Audits refer to another run')
    cells=summary['cells']
    forecasts=[read(run/f"cell_{c['position']:02d}"/'forecast.json') for c in cells]
    groups=[]
    for n in (128,256,4096):
        selected=[c for c in cells if c['evaluation_paths']==n]
        actual=sum((Fraction(c['exact_actual_block_cpu']) for c in selected),Fraction(0))
        predicted=sum((Fraction(c['exact_predicted_block_cpu']) for c in selected),Fraction(0))
        groups.append(dict(evaluation_paths=n,cells=len(selected),
            within_tolerance=sum(c['within_registered_tolerance'] for c in selected),
            below_tolerance=sum(Fraction(c['exact_ratio'])<Fraction(9,10) for c in selected),
            above_tolerance=sum(Fraction(c['exact_ratio'])>Fraction(11,10) for c in selected),
            pooled_ratio=float(actual/predicted),actual_cpu=float(actual),predicted_cpu=float(predicted),
            minimum_ratio=min(c['actual_to_predicted_ratio'] for c in selected),
            maximum_ratio=max(c['actual_to_predicted_ratio'] for c in selected)))
    predicted=sum((Fraction(c['exact_predicted_block_cpu']) for c in cells),Fraction(0))
    actual=sum((Fraction(c['exact_actual_block_cpu']) for c in cells),Fraction(0))
    phase_changes={k:float(sum((Fraction(c['exact_phase_actual_cpu'][k])-Fraction(p['exact_phase_predictions_cpu'][k])
                      for c,p in zip(cells,forecasts)),Fraction(0))) for k in cells[0]['exact_phase_actual_cpu']}
    warmup=read(audit/'checks.json')['warmup_numerical_cpu']
    result=dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        cells=39,pilot_pairs=312,validation_pairs=2496,warmup_pairs=4,
        within_tolerance=sum(c['within_registered_tolerance'] for c in cells),
        below_tolerance=sum(Fraction(c['exact_ratio'])<Fraction(9,10) for c in cells),
        above_tolerance=sum(Fraction(c['exact_ratio'])>Fraction(11,10) for c in cells),
        minimum_ratio=summary['minimum_ratio'],maximum_ratio=summary['maximum_ratio'],
        median_ratio=summary['median_ratio'],pooled_ratio=float(actual/predicted),
        predicted_validation_cpu=float(predicted),actual_validation_cpu=float(actual),
        pilot_numerical_cpu=summary['pilot_numerical_cpu'],
        pilot_as_fraction_of_validation=summary['pilot_numerical_cpu']/float(actual),
        warmup_numerical_cpu=warmup,
        all_numerical_cpu=read(records/'checks.json')['total_numerical_cpu_seconds'],
        process_cpu=manifest['process_cpu_seconds'],process_elapsed=manifest['elapsed_seconds'],
        validation_record_writing_cpu=summary['validation_record_writing_cpu'],
        validation_record_writing_elapsed=summary['validation_record_writing_elapsed'],
        pilot_cv_minimum=min(p['pilot_cv_pair_cpu'] for p in forecasts),
        pilot_cv_median=statistics.median(p['pilot_cv_pair_cpu'] for p in forecasts),
        pilot_cv_maximum=max(p['pilot_cv_pair_cpu'] for p in forecasts),
        last_to_first_eight_minimum=min(c['last_eight_mean_pair_cpu']/c['first_eight_mean_pair_cpu'] for c in cells),
        last_to_first_eight_maximum=max(c['last_eight_mean_pair_cpu']/c['first_eight_mean_pair_cpu'] for c in cells),
        phase_changes_cpu=phase_changes,by_evaluation_paths=groups,
        statistical_prediction_intervals=False,causal_comparison_to_previous_study=False,
        reliable_budget_control_established=False,publication_clearance=False)
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10.2,4.7),layout='constrained')
    ax.axhspan(.9,1.1,color='#e9edf0',label='Registered tolerance')
    ax.axhline(1.,color='#535960',lw=1.)
    for bound in (.9,1.1):
        ax.axhline(bound,color='#8a949e',lw=.8,ls='--')
    for n,colour,marker in ((128,'#006991','o'),(256,'#c06800','s'),(4096,'#713984','^')):
        group=[c for c in cells if c['evaluation_paths']==n]
        ax.scatter([c['position']+1 for c in group], [c['actual_to_predicted_ratio'] for c in group],
                   s=45,c=colour,marker=marker,label=f'{n:,} evaluation paths',zorder=3)
    ax.set(xlabel='Registered execution position',ylabel='Actual CPU / predicted CPU',
           title='Local cost pilots and fixed-count validation',xlim=(0,40))
    ax.set_xticks([1,5,10,15,20,25,30,35,39])
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,ncols=2,loc='upper right',fontsize=9)
    fig.savefig(output/'validation_ratios.png',dpi=180)
    fig.savefig(output/'validation_ratios.svg')
    plt.close(fig)
    evidence[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Evidence changed during analysis')
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,evidence_sha256=evidence,
        artifact_sha256={p.name:sha(p) for p in output.iterdir()},all_cells_retained=True),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','audit','records','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run_analysis(args.run,args.audit,args.records,args.output)
