"""Private figures retaining signed retraining effects and training repetitions."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_conditional_calibration import private,save,sha
from study_terminal_training import verified_results


def plot(folder,output):
    folder,output=private(folder),private(output)
    if output.exists():raise FileExistsError('Use a fresh private figure directory')
    rows,summary=verified_results(folder);output.mkdir(parents=True)
    variants=[(v['training_paths'],v['degree']) for v in summary['families']]
    palette=['#2984a5','#b95e3c','#2e7c69','#8899a6']
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),constrained_layout=True)
    for ax,(n,d) in zip(axes.flat,variants):
        for rep in range(4):
            chosen=[(r,next(p for p in r['scenarios'][-1]['policies'] if (p['training_paths'],p['degree'],p['repetition'])==(n,d,rep))) for r in rows]
            x=np.array([r['case']+1 for r,_ in chosen])+(rep-1.5)*.16
            y=np.array([p['paired_gain']['value'] for _,p in chosen])*100
            error=np.array([[p['paired_gain']['value']-p['paired_gain']['lower'] for _,p in chosen],
                            [p['paired_gain']['upper']-p['paired_gain']['value'] for _,p in chosen]])*100
            ax.errorbar(x,y,yerr=error,fmt='o',ms=3,capsize=2,lw=.8,color=palette[rep],label=f'Repetition {rep+1}')
        ax.axhline(0,color='#333333',lw=1);ax.grid(alpha=.18)
        ax.set(title=f'{n:,} training paths, degree {d}',xlabel='Retained model (one-based)',ylabel='Retraining gain (cents per share)',xticks=[1,3,5,7,9,11,13])
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('+90 minutes: retrained policy relative to terminal-only adjustment\nPointwise 95% sampling intervals; vertical scales differ',fontsize=13)
    fig.savefig(output/'retraining_effect.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5),constrained_layout=True)
    for j,(seconds,colour) in enumerate([(900,'#2984a5'),(5400,'#b95e3c')]):
        scenario=next(s for s in summary['scenarios'] if s['extension_seconds']==seconds)
        axes[0].bar(np.arange(4)+(j-.5)*.3,[100*v['mean_gain'] for v in scenario['variants']],width=.29,color=colour,label=f'+{seconds//60} minutes')
    axes[0].axhline(0,color='#333333',lw=1)
    axes[0].set(ylabel='Mean paired gain (cents per share)',title='Descriptive average over all cases and repetitions')
    axes[0].legend(fontsize=9)
    for j,(name,colour,label) in enumerate([('terminal_only','#8899a6','Terminal-only'),('retrained','#2984a5','Retrained')]):
        axes[1].bar(np.arange(4)+(j-.5)*.3,[100*v['mean_gap_quote'][name] for v in summary['scenarios'][-1]['variants']],width=.29,color=colour,label=label)
    axes[1].set(ylabel='Mean dual gap (cents per share)',title='Residual loss bounds at +90 minutes');axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set(xticks=range(4),xticklabels=[f'{n//1000}k, degree {d}' for n,d in variants]);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Matched training samples and a common dual upper payoff',fontsize=13)
    fig.savefig(output/'retraining_summary.png',dpi=180);plt.close(fig)
    save(output/'manifest.json',dict(experiment_manifest_sha256=sha(folder/'manifest.json'),source_sha256=sha(Path(__file__).resolve()),
        figures={p.name:sha(p) for p in sorted(output.iterdir())},publication_clearance=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.run,args.output)
