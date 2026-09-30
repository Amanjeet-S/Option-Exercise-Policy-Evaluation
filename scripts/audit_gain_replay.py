"""Check the retained replay with a separate 192-bit implementation.

No replay-engine functions are imported. The implementation uses the same Arb
library, so it is not an independent verification of that numerical library.
"""

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from flint import arb, fmpq, ctx


ROOT=Path(__file__).resolve().parents[1]


def require(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(value):
    require(type(value) in (int,float),'An integer or finite binary64 input is required')
    return F(value)


def rational(value):
    require(isinstance(value,str),'A rational string is required')
    result=F(value)
    require(str(result)==value,'A canonical rational string is required')
    return result


def ball(value):
    value=F(value)
    return arb(fmpq(value.numerator,value.denominator))


def endpoints(value):
    require(value.is_finite(),'Non-finite independent enclosure')
    return F(str(value.lower().fmpq())),F(str(value.upper().fmpq()))


def add(a,b):
    return a[0]+b[0],a[1]+b[1]


def sub(a,b):
    return a[0]-b[1],a[1]-b[0]


def multiply(a,b):
    products=[x*y for x in a for y in b]
    return min(products),max(products)


def scaled(a,c):
    return multiply(a,(c,c))


def contained(inner,outer):
    return outer[0]<=inner[0]<=inner[1]<=outer[1]


class Linear:
    def __init__(self,spec):
        self.x=tuple(map(exact,spec['knots']))
        self.y=tuple(map(exact,spec['values']))
        require(exact(spec['tail_slope'])==0 and self.x[0]==0
                and len(self.x)==len(self.y)>=2
                and all(b>a for a,b in zip(self.x,self.x[1:]))
                and all(b<=a for a,b in zip(self.y,self.y[1:])), 'Invalid retained value function')
        slopes=[(self.y[j+1]-self.y[j])/(self.x[j+1]-self.x[j]) for j in range(len(self.x)-1)]+[F(0)]
        self.hinges=[(self.x[j],slopes[j]-slopes[j-1]) for j in range(1,len(self.x))
                     if slopes[j]!=slopes[j-1]]

    def point(self,y):
        if y>=self.x[-1]:
            return self.y[-1]
        j=bisect_left(self.x,y)
        if self.x[j]==y:
            return self.y[j]
        w=(y-self.x[j-1])/(self.x[j]-self.x[j-1])
        return (1-w)*self.y[j-1]+w*self.y[j]

    def interval(self,state):
        return self.point(state[1]),self.point(state[0])

    def mean_point(self,y,s):
        if y==0:
            return ball(self.y[0])
        if s.is_zero():
            return ball(self.point(y))
        total=ball(self.y[-1]); yy=ball(y); root_two=arb(2).sqrt()
        for k,w in self.hinges:
            kk=ball(k)
            d=(kk/yy).log()/s
            first=1+((d+s/2)/root_two).erf()
            second=1+((d-s/2)/root_two).erf()
            total+=ball(w)*(kk*first-yy*second)/2
        return total

    def mean_interval(self,state,s):
        return endpoints(self.mean_point(state[1],s))[0],endpoints(self.mean_point(state[0],s))[1]


def propagate(spec,z):
    model=spec['model']; nodes=spec['nodes']; sigma=ball(exact(model['volatility']))
    initial=exact(model['quoted_spot'])*exact(model['initial_share_factor'])
    state=(initial,initial); values={(F(0),'after'):state}
    events={exact(e['ex_time']):exact(e['discounted_cash']) for e in model['dividends']}
    require(len(z)==len(nodes)-1 and sigma>0,'The retained positive-volatility replay is required')
    for previous,node,innovation in zip(nodes,nodes[1:],z):
        t0,t1=exact(previous['time']),exact(node['time'])
        s=sigma*ball(t1-t0).sqrt()
        factor=(s*(ball(exact(innovation))-s/2)).exp()
        state=(max(F(0),endpoints(ball(state[0])*factor)[0]),endpoints(ball(state[1])*factor)[1])
        if t1 in events:
            values[t1,'before']=state
            state=tuple(max(F(0),y-events[t1]) for y in state)
        values[t1,'after']=state
    return values


def reward(spec,right,state):
    k=exact(spec['option']['strike'])*exact(right['settlement_factor'])
    return max(F(0),k-state[1]),max(F(0),k-state[0])


def candidates(policy,states):
    answer=[]
    for j,(right,rule) in enumerate(zip(policy['decisions'][:-1],policy['rules'][:-1])):
        if not rule['coefficients']:
            continue
        state=states[exact(right['time']),right['phase']]
        immediate=reward(policy,right,state)
        centre,scale=exact(rule['centre']),exact(rule['scale'])
        require(scale>0,'Policy scale must be positive')
        x=((state[0]-centre)/scale,(state[1]-centre)/scale)
        coefficients=list(map(exact,rule['coefficients']))
        prediction=(coefficients[-1],coefficients[-1])
        for c in reversed(coefficients[:-1]):
            prediction=add(multiply(prediction,x),(c,c))
        prediction=tuple(max(F(0),a) for a in prediction)
        if immediate[0]>prediction[1]:
            return tuple((*answer,j))
        if immediate[1]>prediction[0]:
            answer.append(j)
    return tuple((*answer,len(policy['decisions'])-1))


def gain(spec,states,stops,functions):
    a,b=stops
    if a==b:
        return (F(0),F(0)),0
    low,high=spec['decisions'][a],spec['decisions'][b]
    difference=sub(reward(spec,high,states[exact(high['time']),high['phase']]),
                   reward(spec,low,states[exact(low['time']),low['phase']]))
    lo,hi=sorted((exact(low['time']),exact(high['time'])))
    increment=(F(0),F(0)); count=0
    for previous,node in zip(spec['nodes'],spec['nodes'][1:]):
        t0,t1=exact(previous['time']),exact(node['time'])
        if not lo<t1<=hi:
            continue
        phase='before' if node['before'] is not None else 'after'
        value=functions[t1]
        s=ball(exact(spec['model']['volatility']))*ball(t1-t0).sqrt()
        delta=sub(value.interval(states[t1,phase]),value.mean_interval(states[t0,'after'],s))
        increment=add(increment,delta);count+=1
    if b<a:
        increment=scaled(increment,F(-1))
    return scaled(sub(difference,increment),1/exact(spec['model']['premium_factor'])),count


def run(folder,output):
    folder,output=folder.resolve(),output.resolve()
    private=ROOT/'data/private'
    require(folder.is_relative_to(private) and output.is_relative_to(private) and not output.exists(),
            'Use existing private evidence and a fresh private audit directory')
    manifest=read(folder/'manifest.json')
    require(manifest['complete'] and not (folder/'incomplete.json').exists(),'A complete replay is required')
    hashes={str((folder/'manifest.json').relative_to(ROOT)):sha(folder/'manifest.json'),
            str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for key in ('input','export_manifest','source'):
        hashes[manifest[key]]=manifest[key+'_sha256']
    for name,digest in manifest['artifact_sha256'].items():
        hashes[str((folder/name).relative_to(ROOT))]=digest
    input_path=ROOT/manifest['input']; export=read(ROOT/manifest['export_manifest'])
    require(export['complete'] and export['original_evidence_unchanged'],'Replay export is incomplete')
    for group in ('input_sha256','source_sha256'):
        for name,digest in export[group].items():
            require(name not in hashes or hashes[name]==digest,'Conflicting evidence checksums')
            hashes[name]=digest
    for name,digest in export['artifact_sha256'].items():
        hashes[str((input_path.parent/name).relative_to(ROOT))]=digest
    def verify():
        for name,digest in hashes.items():
            path=(ROOT/name).resolve()
            require(path.is_relative_to(ROOT) and sha(path)==digest,'Changed audit evidence: '+name)
    verify()
    source=read(input_path); result=read(folder/'enclosures.json'); summary=read(folder/'summary.json')
    require({k:v for k,v in result.items() if k!='rows'}==summary,'Replay summary differs from full result')
    require(result['paths']==len(source['normals'])==128 and len(result['rows'])==128,
            'The complete first replay block is required')
    require(result['all_stopping_indices_match_on_exported_paths'] and result['ambiguous_policy_count']==0
            and result['proved_stopping_index_difference_count']==0,
            'This audit targets the resolved stopping indices of the first replay')
    spec=source['model_spec']; checked=[]; total=(F(0),F(0)); reference_total=(F(0),F(0))
    original_sum=F(0);state_count=0;increment_count=0
    with ctx.workprec(192):
        functions={exact(n['time']):Linear(n['before'] or n['after']) for n in spec['nodes'][1:]}
        for i,(z,row) in enumerate(zip(source['normals'],result['rows'])):
            require(row['path']==i,'Incomplete or reordered replay paths')
            states=propagate(spec,z)
            stored={(rational(s['time']),s['phase']):(rational(s['lower']),rational(s['upper']))
                    for s in row['state_enclosures']}
            require(len(stored)==len(row['state_enclosures']) and set(states)==set(stored),
                    'State vertex coverage differs')
            require(all(contained(states[key],stored[key]) for key in states),'Independent states escape recorded enclosures')
            state_count+=len(states)
            stops=source['stopping_indices'][i]
            require(row['implemented_stopping_indices']==stops and row['stopping_decisions_match']
                    and row['ambiguous_policy_count']==row['proved_stopping_index_difference_count']==0,
                    'Path stopping metadata differs')
            for policy,retained_stop,recorded_set in zip(source['policies'],stops,row['stopping_candidates']):
                require(candidates(policy,stored)==(retained_stop,) and recorded_set==[retained_stop],
                        'Independent rational exercise comparison differs')
            reference,count=gain(spec,states,stops,functions)
            recorded=tuple(map(rational,row['gain_interval']))
            require(contained(reference,recorded),'Independent gain enclosure escapes recorded interval')
            require(row['evaluated_martingale_increments']==count,'Increment count differs')
            increment_count+=count
            original=exact(source['quoted_gains'][i]);error=sub(recorded,(original,original))
            require(rational(row['recorded_gain'])==original
                    and tuple(map(rational,row['error_interval']))==error
                    and rational(row['absolute_error_bound'])==max(map(abs,error)), 'Path error assembly differs')
            total=add(total,recorded);reference_total=add(reference_total,reference);original_sum+=original
            checked.append(dict(path=i,states_checked=len(states),stopping_indices=stops,
                independent_gain_interval=[str(x) for x in reference],contained=True))
    mean=scaled(total,F(1,128));reference_mean=scaled(reference_total,F(1,128))
    retained_mean=exact(source['record']['evaluation']['gain']['estimate']['value'])
    error=sub(mean,(retained_mean,retained_mean))
    require(tuple(map(rational,result['mean_interval']))==mean
            and tuple(map(rational,result['mean_error_interval']))==error
            and rational(result['mean_absolute_error_bound'])==max(map(abs,error))
            and rational(result['recorded_path_mean_exact'])==original_sum/128
            and rational(result['recorded_mean_aggregation_error'])==retained_mean-original_sum/128,
            'Reported mean assembly differs')
    require(contained(reference_mean,mean) and increment_count==result['evaluated_martingale_increments'],
            'Independent mean or increment coverage differs')
    verify()
    report=dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),paths=128,
        state_enclosures_checked=state_count,stopping_indices_checked=256,
        independent_martingale_increments=increment_count,independent_precision_bits=192,
        all_independent_state_and_gain_enclosures_contained=True,
        independent_mean_interval=[str(x) for x in reference_mean],exact_mean_assembly_verified=True,
        implementation='Separate propagation, exact rational polynomial decisions, and erf-based hinge integration; no replay-engine imports.',
        same_underlying_arb_library=True,independent_arithmetic_library_verification=False,
        gaussian_sampling_law_certified=False,statistical_coverage_certified=False,
        original_evidence_unchanged=True,evidence_sha256=hashes,publication_clearance=False)
    output.mkdir(parents=True)
    (output/'independent_paths.json').write_text(json.dumps(checked,indent=2)+'\n')
    report['artifact_sha256']={'independent_paths.json':sha(output/'independent_paths.json')}
    pending=output/'checks.pending.json';pending.write_text(json.dumps(report,indent=2)+'\n')
    verify();pending.rename(output/'checks.json')
    return {k:v for k,v in report.items() if k not in ('evidence_sha256','artifact_sha256','independent_mean_interval')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.run,args.output),indent=2))
