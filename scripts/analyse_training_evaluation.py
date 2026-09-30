"""Private tables and figures for the completed crossed experiment."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_conditional_calibration import private, read, save, sha
from audit_training_evaluation import verify


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyse(run,output):
    run,output=private(run),private(output)
    if output.exists():raise FileExistsError('Use a fresh analysis directory')
    blocks,cfg,tasks,checks=verify(run)
    summary=read(run/'summary.json')
    output.mkdir(parents=True)
    primary=[];evaluation=[];repetitions=[];regression=[];costs=[]
    contention=run.parent/'host_contention_v1.json'
    for model in summary['models']:
        r=model['primary_training_gain_quote'];e=r['estimate'];times=model['additional_training_seconds']
        primary.append(dict(case=model['case'],gain=e['value'],standard_error=e['standard_error'],
            lower=e['lower'],upper=e['upper'],repetitions=e['observations'],
            training_variance_moment_estimate=r['training_variance_moment_estimate'],
            evaluation_variance_of_grand_mean=r['evaluation_variance_of_grand_mean'],
            median_extra_training_seconds=float(np.median(times)),
            median_extra_training_cpu_seconds=float(np.median([b['training'][1]['cpu_seconds']-b['training'][0]['cpu_seconds'] for b in blocks if b['case']==model['case']]))))
        for block in (b for b in blocks if b['case']==model['case']):
            gain=next(g['estimate'] for g in block['policy_gains'] if
                      g['method']=='fine' and g['evaluation_paths']==cfg['evaluation_paths'][-1])
            repetitions.append(dict(case=model['case'],repetition=block['repetition'],
                gain=gain['value'],conditional_standard_error=gain['standard_error']))
            for trained in block['training']:
                rules=[r for r in trained['policy']['rules'] if r['status']!='terminal']
                statuses=sorted({r['status'] for r in rules})
                for status in statuses:
                    selected=[r for r in rules if r['status']==status]
                    conditions=[r['condition_number'] for r in selected if r['condition_number'] is not None]
                    regression.append(dict(case=model['case'],repetition=block['repetition'],
                        training_paths=trained['training_paths'],status=status,decisions=len(selected),
                        largest_condition_number=max(conditions) if conditions else None))
        for workload in model['workloads']:
            selected=[w for b in blocks if b['case']==model['case'] for w in b['workloads']
                if (w['training_paths'],w['method'],w['evaluation_paths'])==
                   (workload['training_paths'],workload['method'],workload['evaluation_paths'])]
            costs.append(dict(case=model['case'],training_paths=workload['training_paths'],
                method=workload['method'],evaluation_paths=workload['evaluation_paths'],
                median_total_elapsed_seconds=float(np.median([w['total_elapsed_seconds'] for w in selected])),
                median_total_cpu_seconds=float(np.median([w['total_cpu_seconds'] for w in selected])),
                minimum_total_elapsed_seconds=min(w['total_elapsed_seconds'] for w in selected),
                maximum_total_elapsed_seconds=max(w['total_elapsed_seconds'] for w in selected),
                median_martingale_fraction=float(np.median([
                    w['phases']['martingale_seconds']/w['elapsed_seconds'] for w in selected]))))
        for n in cfg['training_paths']:
            for size in cfg['evaluation_paths']:
                rows=[next(x for x in b['evaluation_effects'] if (x['training_paths'],x['evaluation_paths'])==(n,size))
                      for b in blocks if b['case']==model['case']]
                evaluation.append(dict(case=model['case'],training_paths=n,evaluation_paths=size,
                    mean_fine_minus_coarse_gap=float(np.mean([x['estimate']['value'] for x in rows])),
                    conditional_sampling_variance_of_mean=float(sum(x['estimate']['standard_error']**2 for x in rows)/len(rows)**2)))
    write_csv(output/'primary_training_gain.csv',primary)
    write_csv(output/'evaluation_effects.csv',evaluation)
    write_csv(output/'repetition_gains.csv',repetitions)
    write_csv(output/'regression_status.csv',regression)
    write_csv(output/'workload_costs.csv',costs)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),layout='constrained')
    indices=np.arange(len(primary));g=np.array([r['gain'] for r in primary])
    err=np.array([[r['gain']-r['lower'] for r in primary],[r['upper']-r['gain'] for r in primary]])
    axes[0].errorbar(g,indices,xerr=err,fmt='o',color='#195277',capsize=3,markersize=4)
    axes[0].axvline(0,color='.4',lw=.8)
    axes[0].set(yticks=indices,yticklabels=[str(r['case']) for r in primary],
        xlabel='Expected reward gain, quoted premium units',ylabel='Retained model',
        title='Additional training: signed policy-value gain')
    axes[0].invert_yaxis()
    axes[1].plot([r['median_extra_training_seconds'] for r in primary],indices-.1,'o',color='#9d5e24',markersize=4,label='Elapsed')
    axes[1].plot([r['median_extra_training_cpu_seconds'] for r in primary],indices+.1,'x',color='#195277',markersize=4,label='Process CPU')
    axes[1].set(yticks=indices,yticklabels=[str(r['case']) for r in primary],
        xlabel='Additional training time, median seconds',title='Observed expenditure on training')
    axes[1].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.suptitle('Fixed financial models and exercise opportunities',fontsize=12)
    fig.savefig(output/'training_gain_and_cost.png',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(13,11),layout='constrained')
    colours={cfg['training_paths'][0]:'#195277',cfg['training_paths'][1]:'#a35124'}
    markers={'coarse':'o','fine':'^'}
    for ax,model in zip(axes.flat,summary['models']):
        for w in model['workloads']:
            colour=colours[w['training_paths']]
            filled=w['evaluation_paths']==cfg['evaluation_paths'][-1]
            ax.scatter(w['total_elapsed_seconds']['median'],w['mean_upper_limit_quote'],
                       color=colour if filled else 'none',edgecolors=colour,
                       marker=markers[w['method']],s=38)
        ax.set_title(f'Model {model["case"]}',fontsize=10)
        ax.set_xlabel('Median total seconds',fontsize=8)
        ax.set_ylabel('Mean reported loss limit',fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(summary['models']):]:ax.set_visible(False)
    handles=[Line2D([0],[0],color=colours[n],marker='o',ls='',label=f'{n:,} training paths') for n in cfg['training_paths']]
    handles += [Line2D([0],[0],color='.25',marker=marker,ls='',label=method.capitalize()) for method,marker in markers.items()]
    handles += [Line2D([0],[0],color='.25',marker='s',mfc='none',ls='',label=f'{cfg["evaluation_paths"][0]:,} evaluation paths'),
                Line2D([0],[0],color='.25',marker='s',ls='',label=f'{cfg["evaluation_paths"][1]:,} evaluation paths')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Observed workload cost and conditional loss assessment'+(' on a shared host' if contention.is_file() else ''),fontsize=12)
    fig.savefig(output/'cost_and_assessment.png',dpi=200)
    plt.close(fig)
    save(output/'tables.json',dict(primary=primary,evaluation=evaluation,repetitions=repetitions,
        regression_status=regression,workload_costs=costs,
        descriptive_only=dict(positive_primary_pointwise_intervals=sum(r['lower']>0 for r in primary),
            negative_primary_pointwise_intervals=sum(r['upper']<0 for r in primary),
            negative_training_variance_estimates=sum(r['training_variance_moment_estimate']<0 for r in primary)),
        publication_clearance=False))
    files=[p.name for p in output.iterdir() if p.is_file()]
    save(output/'manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        experiment_manifest_sha256=sha(run/'manifest.json'),script_sha256=sha(Path(__file__)),
        host_contention_record_sha256=sha(contention) if contention.is_file() else None,
        artifact_sha256={name:sha(output/name) for name in files},publication_clearance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyse(a.run,a.output)
