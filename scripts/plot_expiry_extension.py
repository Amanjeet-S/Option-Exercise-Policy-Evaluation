"""Private figures for the expiry opportunity value and frozen-policy losses."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_conditional_calibration import private,save,sha
from study_expiry_extension import verified_results


def plot(folder,output):
    folder,output=private(folder),private(output)
    if output.exists():raise FileExistsError('Use a fresh private figure directory')
    rows,summary=verified_results(folder);output.mkdir(parents=True)
    colours={0:'#8899a6',900:'#2984a5',5400:'#b95e3c'}
    labels={0:'Original close',900:'+15 minutes',5400:'+90 minutes'}
    fig,axes=plt.subplots(1,2,figsize=(12.5,5),constrained_layout=True)
    x=[r['case']+1 for r in rows]
    for seconds in (900,5400):
        selected=[next(s for s in r['scenarios'] if s['extension_seconds']==seconds) for r in rows]
        axes[0].plot(x,[s['increment_prices']['fine'] for s in selected],'o-',color=colours[seconds],label=f'{labels[seconds]}: PDE increment',lw=1.3)
        axes[0].plot(x,[s['european']['increment_quote'] for s in selected],'--',color=colours[seconds],label=f'{labels[seconds]}: European upper comparison',lw=1.3)
    axes[0].set(xlabel='Retained model',ylabel='USD per share',title='Added expiry opportunity with fixed financial inputs')
    axes[0].legend(fontsize=8)
    variants=[(f['training_paths'],f['degree']) for f in summary['families']]
    for scenario in summary['scenarios']:
        seconds=scenario['extension_seconds']
        axes[1].plot(range(len(variants)),[v['mean_gap_quote'] for v in scenario['variants']],
            'o-',color=colours[seconds],label=labels[seconds],lw=1.3)
    axes[1].set(xticks=range(len(variants)),xticklabels=[f'{n//1000}k, degree {d}' for n,d in variants],
        ylabel='Mean dual gap (USD per share)',title='Same original stopping instructions')
    axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('Expiry-day horizon sensitivity | neither a refit nor contract certification',fontsize=13)
    fig.savefig(output/'expiry_value.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5),constrained_layout=True)
    for j,variant in enumerate(variants):
        selected=[(r,f) for r in rows for f in r['families'] if (f['training_paths'],f['degree'])==variant]
        axes[0].scatter(j+np.linspace(-.15,.15,len(selected)),[f['maximum_upper_quote']/r['spread'] for r,f in selected],s=18,alpha=.7,color='#2984a5')
    axes[0].axhline(1.,color='#333333',ls='--',lw=1)
    axes[0].set(xticks=range(len(variants)),xticklabels=[f'{n//1000}k, degree {d}' for n,d in variants],
        ylabel='Largest adjusted loss limit / observed spread',title='Every original policy and all prescribed endpoints')
    names=['Original','+15 min','+90 min']
    matrix=np.array([[int(s['call_quote_with_margin'])+2*int(s['put_quote_with_margin']) for s in r['scenarios']] for r in rows])
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#b14545','#e2ab60','#8fb6cc','#2e7c69']);norm=BoundaryNorm(np.arange(-.5,4.5),cmap.N)
    axes[1].imshow(matrix,cmap=cmap,norm=norm,aspect='auto')
    for i in range(len(rows)):
        for j in range(3):
            axes[1].text(j,i,['Neither','Call only','Put only','Both'][matrix[i,j]],ha='center',va='center',fontsize=9,color='white' if matrix[i,j] in (0,3) else '#172531')
    axes[1].set(xticks=range(3),xticklabels=names,yticks=range(len(rows)),yticklabels=x,
        ylabel='Retained model',title='Original quote-margin screens after extending expiry')
    axes[0].grid(alpha=.18)
    fig.suptitle('Conditional loss and financial compatibility are separate checks',fontsize=13)
    fig.savefig(output/'expiry_bounds_and_fit.png',dpi=180);plt.close(fig)
    save(output/'manifest.json',dict(experiment_manifest_sha256=sha(folder/'manifest.json'),source_sha256=sha(Path(__file__).resolve()),
        figures={p.name:sha(p) for p in sorted(output.iterdir())},publication_clearance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.run,args.output)
