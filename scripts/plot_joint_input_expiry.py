"""Private joint-family bounds, secondary interactions and quote diagnostics."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_conditional_calibration import private, read, save, sha, verify_mapping


def plot(folder, audit, output):
    folder,audit,output=map(private,(folder,audit,output))
    if output.exists(): raise FileExistsError('Use a fresh private figure directory')
    manifest=read(folder/'manifest.json'); check=read(audit/'verification.json')
    verify_mapping(manifest['artifact_sha256'],folder)
    if check['experiment_manifest_sha256']!=sha(folder/'manifest.json'):
        raise ValueError('Audit does not match the experiment')
    rows=read(folder/'rows.json')
    variants=sorted({(f['training_paths'],f['degree']) for r in rows for f in r['families']})
    labels=[f'{n//1000}k, degree {d}' for n,d in variants]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,6.6),gridspec_kw={'width_ratios':[1,1,1.2]})
    plotted=dict(families=[],interactions=[],compatibility=[])
    for j,variant in enumerate(variants):
        selected=[(r['case'],f) for r in rows for f in r['families'] if (f['training_paths'],f['degree'])==variant]
        x=j+np.linspace(-.18,.18,len(selected))
        old=[f['old_financial_family_upper_quote']/f['spread'] for _,f in selected]
        new=[f['maximum_upper_quote']/f['spread'] for _,f in selected]
        axes[0].vlines(x,old,new,color='#789399',lw=.6,alpha=.5)
        axes[0].scatter(x,old,s=13,color='#90969D',alpha=.7,label='Original expiry' if j==0 else None)
        axes[0].scatter(x,new,s=17,color='#147D83',alpha=.8,label='Joint family' if j==0 else None)
        for (case,f),a,b in zip(selected,old,new):
            plotted['families'].append(dict(case=case,fingerprint=f['fingerprint'],original_ratio=a,joint_ratio=b))
        effects=[(r['case'],s['scenario'],p) for r in rows for s in r['scenarios'][1:] if s['status']=='evaluated'
                 for p in s['policies'] if (p['training_paths'],p['degree'])==variant]
        x=j+np.linspace(-.2,.2,len(effects))
        e=[p['interaction_quote'] for _,_,p in effects]
        axes[1].vlines(x,[v['lower'] for v in e],[v['upper'] for v in e],color='#176D93',alpha=.14,lw=.5)
        axes[1].scatter(x,[v['value'] for v in e],color='#176D93',alpha=.35,s=5)
        for case,scenario,p in effects:
            plotted['interactions'].append(dict(case=case,scenario=scenario,fingerprint=p['fingerprint'],**p['interaction_quote']))
    axes[0].axhline(1.,color='#303030',ls='--',lw=1)
    axes[0].set(title='Per-policy finite-family loss limits',ylabel='Upper limit / quoted spread',xticks=range(len(labels)),xticklabels=labels)
    axes[0].legend(fontsize=9,loc='upper right')
    axes[1].axhline(0.,color='#303030',ls='--',lw=1)
    axes[1].set(title='Financial effect: extended minus original',ylabel='Interaction in expected gap, quote units',xticks=range(len(labels)),xticklabels=labels)
    for ax in axes[:2]:
        ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.15)
    matrix=np.array([[float(s['joint_quote_compatible']) if s['status']=='evaluated' else np.nan for s in r['scenarios']] for r in rows])
    axes[2].imshow(matrix,vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(['#DA9276','#53A397']),aspect='auto')
    names=[s['scenario'].replace('cash_','').replace('_funding_',' / ') for s in rows[0]['scenarios']]
    for i,row in enumerate(rows):
        for j,s in enumerate(row['scenarios']):
            value=matrix[i,j]
            axes[2].text(j,i,'?' if np.isnan(value) else ('Y' if value else 'N'),ha='center',va='center',fontsize=8,color='#14242E')
            plotted['compatibility'].append(dict(case=row['case'],scenario=s['scenario'],compatible=None if np.isnan(value) else bool(value)))
    axes[2].set(title='Extended joint quote diagnostic',ylabel='Original local case',xticks=range(len(names)),xticklabels=names,
                yticks=range(len(rows)),yticklabels=[str(r['case']) for r in rows])
    axes[2].tick_params(axis='x',rotation=65)
    fig.suptitle('Joint financial-input and expiry sensitivity',fontsize=16,y=.98)
    fig.text(.02,.018,'Family limits concern one frozen policy. Interaction intervals are pointwise. '
        'Quote failures remain in the conditional experiment; no new market sample is introduced.',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.94))
    output.mkdir(parents=True)
    for suffix in ('png','svg'): fig.savefig(output/f'joint_robustness.{suffix}',dpi=170)
    plt.close(fig)
    save(output/'plotted_values.json',plotted)
    save(output/'manifest.json',dict(experiment_manifest_sha256=sha(folder/'manifest.json'),
        audit_sha256=sha(audit/'verification.json'),source_sha256=sha(Path(__file__)),
        artifact_sha256={n:sha(output/n) for n in ('joint_robustness.png','joint_robustness.svg','plotted_values.json')},
        publication_clearance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','audit','output'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); plot(args.experiment,args.audit,args.output)
