"""Render audited finite-budget frontiers and compare retained allocation choices."""

import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def f(value):return float(Fraction(value))


def plot(inputs,run,audit,output):
    inputs,run,audit,output=(p.resolve() for p in (inputs,run,audit,output))
    if output.exists() or not all(p.is_relative_to(ROOT/'data/private') for p in (inputs,run,audit,output)):
        raise ValueError('Fresh private output required')
    checks=read(audit/'checks.json')
    if checks['complete'] is not True:raise ValueError('Completed independent audit required')
    evidence=dict(checks['evidence_sha256'])
    for path in (run/'summary.json',inputs/'inputs.json'):
        if evidence.get(str(path.relative_to(ROOT)))!=sha(path):raise ValueError('Audit refers to different evidence')
    for name,digest in evidence.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:raise ValueError('Audited evidence changed')
    evidence[str((audit/'checks.json').relative_to(ROOT))]=sha(audit/'checks.json')
    data=read(inputs/'inputs.json')['models'];models=read(run/'summary.json')['models']
    results=[]
    for accounting in ('marginal','screening_charged'):
        rows=[]
        for original,model in zip(data,models):
            if model['status']!='resolved':raise ValueError('Plot expects all retained cases resolved')
            a=model['accounting'][accounting];p=a['choices']['pilot']['selected'];v=a['choices']['validation']['selected'];j=a['choices']['joint']['selected']
            if any(x is None for x in (p,v,j)):raise ValueError('Plot requires feasible candidates')
            cross=a['cross_evaluation']
            rows.append(dict(case=model['case'],original_selected=original['original']['selected'],
                pilot=dict(N=p['evaluation_paths'],R=p['repetitions']),
                validation=dict(N=v['evaluation_paths'],R=v['repetitions']),
                joint=dict(N=j['evaluation_paths'],R=j['repetitions']),
                pilot_choice_affordable_under_validation=cross['pilot_choice']['scenario_feasible']['validation'],
                validation_choice_affordable_under_pilot=cross['validation_choice']['scenario_feasible']['pilot'],
                original_reference_jointly_feasible=cross['original_reference']['jointly_feasible'],
                original_selected_jointly_feasible=cross['original_selected']['jointly_feasible'],
                joint_to_reference_variance_ratio=f(a['joint_to_original_reference_variance_ratio']),
                joint_to_budget_matched_reference_variance_ratio=(f(a['joint_to_budget_matched_reference_variance_ratio']) if a['joint_to_budget_matched_reference_variance_ratio'] is not None else None),
                joint_to_pilot_variance_ratio=f(a['joint_to_single_scenario_variance_ratio']['pilot']),
                joint_to_validation_variance_ratio=f(a['joint_to_single_scenario_variance_ratio']['validation']),
                joint_budget_fraction=max(f(x) for x in cross['joint_choice']['scenario_costs'].values())/f(original['budget']),
                overhead=f(a['overhead']),joint_cap_binds=j['capped']))
        results.append(dict(accounting=accounting,rows=rows,
            pilot_validation_preferred_N_differences=sum(r['pilot']['N']!=r['validation']['N'] for r in rows),
            pilot_choice_validation_budget_failures=sum(not r['pilot_choice_affordable_under_validation'] for r in rows),
            validation_choice_pilot_budget_failures=sum(not r['validation_choice_affordable_under_pilot'] for r in rows),
            joint_choice_differs_from_original_N=sum(r['joint']['N']!=r['original_selected']['evaluation_paths'] for r in rows),
            original_selected_jointly_feasible=sum(r['original_selected_jointly_feasible'] for r in rows),
            original_reference_jointly_feasible=sum(r['original_reference_jointly_feasible'] for r in rows),
            maximum_joint_to_pilot_variance_ratio=max(r['joint_to_pilot_variance_ratio'] for r in rows),
            maximum_joint_to_validation_variance_ratio=max(r['joint_to_validation_variance_ratio'] for r in rows),
            joint_reference_ratio_range=[min(r['joint_to_reference_variance_ratio'] for r in rows),max(r['joint_to_reference_variance_ratio'] for r in rows)]))
    charges=[f(m['screening_charge'])/f(m['budget']) for m in data]
    summary=dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),accounting=results,
        screening_budget_fraction_range=[min(charges),max(charges)],
        screening_changes_joint_N=sum(x['joint']['N']!=y['joint']['N'] for x,y in zip(results[0]['rows'],results[1]['rows'])),
        charged_to_marginal_joint_variance_ratio=[f(m['accounting']['screening_charged']['choices']['joint']['selected']['variance'])/f(m['accounting']['marginal']['choices']['joint']['selected']['variance']) for m in models],
        new_financial_estimates=0,retrospective=True,publication_clearance=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap,BoundaryNorm
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    # Case zero is selected by index as an illustration, not by its outcome.
    case=data[0];C=f(case['budget']);H=f(case['screening_charge'])
    reference=f(models[0]['accounting']['marginal']['cross_evaluation']['original_reference']['variance'])
    fig,axes=plt.subplots(2,1,figsize=(9,7.4),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
    for name,label,colour,style in [('marginal','Existing cost information','#006991','-'),('screening_charged','Three-size pilot charged','#ad5700','--')]:
        curve=read(run/f'case_00_{name}_frontiers.json')['joint']
        selected=[r for r in curve if f(r['required_budget'])<=C]
        xs=[f(r['required_budget']) for r in selected]+[C]
        ys=[f(r['variance'])/reference for r in selected]+[f(selected[-1]['variance'])/reference]
        axes[0].step(xs,ys,where='post',label=label,color=colour,ls=style,lw=1.6)
        axes[0].scatter([C],[ys[-1]],color=colour,s=28,zorder=4)
        ns=[r['repetitions'] for r in selected]+[selected[-1]['repetitions']]
        axes[1].step(xs,ns,where='post',color=colour,ls=style,lw=1.4)
    for ax in axes:
        ax.axvline(C,color='#6b7177',lw=.8,ls=':')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.18)
    axes[0].set(title='Joint feasibility under both observed cost scenarios: case 0',
                ylabel='Pilot variance criterion / original reference criterion')
    axes[0].legend(frameon=False,title=f'Common screening charge: {H:.3f} CPU seconds',loc='upper right')
    axes[1].set(xlabel='Numerical CPU budget (seconds)',ylabel='Selected R',xlim=(0,C*1.015))
    fig.savefig(output/'case_00_budget_frontier.png',dpi=180);fig.savefig(output/'case_00_budget_frontier.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,5.4),layout='constrained')
    cmap=ListedColormap(['#e0eff5','#78b6cf','#dda86a']);norm=BoundaryNorm([-.5,.5,1.5,2.5],cmap.N)
    for ax,record in zip(axes,results):
        matrix=np.array([[{128:0,256:1,4096:2}[r[q]['N']] for r in record['rows']] for q in ('pilot','validation','joint')])
        ax.imshow(matrix,cmap=cmap,norm=norm,aspect='auto')
        for row,q in enumerate(('pilot','validation','joint')):
            for col,r in enumerate(record['rows']):
                ax.text(col,row,f"{r[q]['N']:,}\n({r[q]['R']})",ha='center',va='center',fontsize=8,color='#142b36')
        ax.set(xticks=range(13),xticklabels=range(13),yticks=range(3),yticklabels=['Pilot costs','Validation costs','Both scenarios'],
               title='Existing cost information' if record['accounting']=='marginal' else 'Three-size pilot charged')
        ax.set_xlabel('Retained financial case')
    fig.suptitle('Chosen evaluation paths N, with fitted-pair repetitions R in brackets',fontsize=12)
    fig.savefig(output/'allocation_choices.png',dpi=180);fig.savefig(output/'allocation_choices.svg');plt.close(fig)
    evidence[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    for name,digest in evidence.items():
        if sha(ROOT/name)!=digest:raise ValueError('Evidence changed during rendering')
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,evidence_sha256=evidence,
        artifact_sha256={p.name:sha(p) for p in output.iterdir()},new_financial_estimates=0),indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='accounting'},indent=2))
    for r in results:print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','run','audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();plot(args.inputs,args.run,args.audit,args.output)
