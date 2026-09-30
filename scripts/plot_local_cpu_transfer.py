"""Render all local timing ratios with an unobstructed legend."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(analysis,run,output):
    analysis,run,output=(p.resolve() for p in (analysis,run,output))
    if output.exists() or not all(p.is_relative_to(ROOT/'data/private') for p in (analysis,run,output)):
        raise ValueError('Use a fresh private figure directory')
    manifest=json.loads((analysis/'manifest.json').read_text())
    evidence=dict(manifest['evidence_sha256'])
    for name,digest in manifest['artifact_sha256'].items():
        evidence[str((analysis/name).relative_to(ROOT))]=digest
    evidence[str((analysis/'manifest.json').relative_to(ROOT))]=sha(analysis/'manifest.json')
    if evidence.get(str((run/'summary.json').relative_to(ROOT)))!=sha(run/'summary.json'):
        raise ValueError('The analysis refers to another run')
    for name,digest in evidence.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=digest:
            raise ValueError('Evidence changed')
    cells=json.loads((run/'summary.json').read_text())['cells']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10.2,5.4))
    fig.subplots_adjust(left=.1,right=.99,bottom=.27,top=.9)
    ax.axhspan(.9,1.1,color='#e9edf0',label='Registered tolerance')
    ax.axhline(1.,color='#535960',lw=1.)
    for bound in (.9,1.1):
        ax.axhline(bound,color='#8a949e',lw=.8,ls='--')
    for n,colour,marker in ((128,'#006991','o'),(256,'#c06800','s'),(4096,'#713984','^')):
        group=[c for c in cells if c['evaluation_paths']==n]
        ax.scatter([c['position']+1 for c in group],[c['actual_to_predicted_ratio'] for c in group],
                   s=40,c=colour,marker=marker,label=f'{n:,} evaluation paths',zorder=3)
    ax.set(xlabel='Registered execution position',ylabel='Actual CPU / predicted CPU',
           title='Local cost pilots and fixed-count validation',xlim=(0,40))
    ax.set_xticks([1,5,10,15,20,25,30,35,39])
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,ncols=2,loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=9)
    output.mkdir(parents=True)
    fig.savefig(output/'validation_ratios.png',dpi=180)
    fig.savefig(output/'validation_ratios.svg')
    plt.close(fig)
    evidence[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,
        scope='All 39 registered ratios; legend placed outside the data area.',
        evidence_sha256=evidence,artifact_sha256={p.name:sha(p) for p in output.iterdir()}),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('analysis','run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    render(args.analysis,args.run,args.output)
