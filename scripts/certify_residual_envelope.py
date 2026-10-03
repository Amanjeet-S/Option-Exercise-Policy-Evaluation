"""Certify full-domain residual ranges for retained put interpolants.

Use the separate qf-certification environment. Binary64 inputs define exact
dyadic values; all time scaling is formed within Arb. Cash maps, payoff kinks
and interval aggregation use exact rational arithmetic. The resulting bound
concerns the ideal-model martingale, not recorded finite-precision path gains.
"""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

from flint import arb, fmpq, ctx
import flint


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
ENDPOINT_BITS = 80
APPROXIMATION_FIELDS = ('model','option','decisions','nodes','core_steps',
                        'wing_steps','compression_tolerance','initial_value')


def q(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('An exact rational or finite binary64 input is required')
    if isinstance(value, int):
        return Fraction(value)
    if not math.isfinite(value):
        raise ValueError('A finite input is required')
    return Fraction.from_float(value)


def ball(value):
    value = q(value)
    return arb(fmpq(value.numerator, value.denominator))


def outward(value):
    """Outward dyadic endpoints, with fixed absolute resolution 2**(-80)."""
    if not value.is_finite():
        raise ArithmeticError('A non-finite ball cannot certify a range')
    scale = 1 << ENDPOINT_BITS
    lower = int((value.lower().fmpq()*scale).floor())
    upper = int((value.upper().fmpq()*scale).ceil())
    return Fraction(lower, scale), Fraction(upper, scale)


def normal_cdf(value):
    return (-value/arb(2).sqrt()).erfc()/2


class RationalValue:
    """A constant-tail interpolant with exact rational knots and values."""

    def __init__(self, knots, values):
        self.knots = tuple(q(x) for x in knots)
        self.values = tuple(q(y) for y in values)
        if (len(self.knots) < 2 or len(self.knots) != len(self.values)
                or self.knots[0] != 0
                or any(b <= a for a, b in zip(self.knots, self.knots[1:]))):
            raise ValueError('Increasing knots beginning at zero and equal value lengths required')
        self.slopes = tuple((b-a)/(v-u) for a,b,u,v in
                            zip(self.values,self.values[1:],self.knots,self.knots[1:]))
        extended = (*self.slopes, Fraction(0))
        self.hinges = tuple((strike, b-a) for strike,a,b in
                            zip(self.knots[1:], extended, extended[1:]) if b != a)
        self.arb_hinges = tuple((ball(k), ball(k).log(), ball(w), ball(abs(w)))
                               for k,w in self.hinges)

    def value(self, state):
        state = q(state)
        if state < 0:
            raise ValueError('Non-negative states are required')
        if state >= self.knots[-1]:
            return self.values[-1]
        index = bisect_right(self.knots,state)-1
        return self.values[index]+self.slopes[index]*(state-self.knots[index])

    def expectation(self, state, sd):
        state = q(state)
        if state < 0:
            raise ValueError('Non-negative states are required')
        if state == 0:
            return ball(self.values[0])
        if sd.is_zero():
            return ball(self.value(state))
        if not sd.is_finite() or not sd > 0:
            raise ArithmeticError('A provably positive finite volatility-time ball is required')
        y = ball(state)
        log_y = y.log()
        result = ball(self.values[-1])
        for strike,log_strike,weight,_ in self.arb_hinges:
            z = (log_strike-log_y)/sd
            result += weight*(strike*normal_cdf(z+sd/2)-y*normal_cdf(z-sd/2))
        if not result.is_finite():
            raise ArithmeticError('Conditional expectation enclosure is not finite')
        return result


def transition_sd(sigma,t0,t1):
    sigma,t0,t1 = q(sigma),q(t0),q(t1)
    if sigma < 0 or t0 < 0 or t1 <= t0:
        raise ValueError('Non-negative volatility and increasing non-negative times required')
    return ball(sigma)*ball(t1-t0).sqrt()


def linear_range(left,right,cap=Fraction(0)):
    """Exact range of left(y)-right((y-cap)+), including the constant tail."""
    cap = q(cap)
    if cap < 0:
        raise ValueError('A non-negative cash cap is required')
    grid = sorted(set(left.knots) | {x+cap for x in right.knots} | {Fraction(0)})
    values = [left.value(x)-right.value(max(Fraction(0),x-cap)) for x in grid]
    return dict(lower=min(values),upper=max(values),knots=grid,values=values,
                tail=left.values[-1]-right.values[-1])


def defect_range(value,strike,factor):
    threshold = q(strike)*q(factor)
    if threshold <= 0:
        raise ValueError('A positive exercise threshold is required')
    payoff = RationalValue((Fraction(0),threshold),(threshold,Fraction(0)))
    result = linear_range(payoff,value)
    result['threshold'] = threshold
    return result


class Curvature:
    """Certified hinge-gamma bounds, with conservative undecided-mode branches."""

    def __init__(self,value,sd):
        self.sd,self.s2 = sd,sd*sd
        self.log_sd = sd.log()
        self.half_log_two_pi = (2*arb.pi()).log()/2
        self.hinges = []
        for strike,log_strike,_,abs_weight in value.arb_hinges:
            global_gamma = self.s2.exp()/(strike*sd*(2*arb.pi()).sqrt())
            self.hinges.append((log_strike,abs_weight,global_gamma))

    def upper(self,a,b):
        log_a = ball(a).log() if a else None
        log_b = ball(b).log()
        total = arb(0)
        for log_k,weight,global_gamma in self.hinges:
            # log(y/k)+3s²/2 locates y relative to the gamma maximiser.
            if a and log_a-log_k+3*self.s2/2 > 0:
                log_y = log_a
            elif log_b-log_k+3*self.s2/2 < 0:
                log_y = log_b
            else:
                total += weight*global_gamma
                continue
            z = (log_k-log_y)/self.sd-self.sd/2
            gamma = (-z*z/2-log_y-self.log_sd-self.half_log_two_pi).exp()
            total += weight*gamma
        return max(Fraction(0),outward(total)[1])


def residual_range(left,right,sd,*,tolerance=Fraction(1,1000),max_rounds=24,max_points=200000):
    tolerance = q(tolerance)
    if tolerance <= 0 or type(max_rounds) is not int or max_rounds < 0:
        raise ValueError('Positive tolerance and non-negative integer refinement count required')
    if type(max_points) is not int or max_points < 2:
        raise ValueError('At least two state points are required')
    if not sd.is_finite() or not (sd.is_zero() or sd > 0):
        raise ValueError('A non-negative resolved volatility-time ball is required')
    if sd.is_zero():
        result = linear_range(left,right)
        return dict(lower=result['lower'],upper=result['upper'],
                    candidate_inf_upper=result['lower'],candidate_sup_lower=result['upper'],
                    tolerance_met=True,coverage_complete=True,tolerance=tolerance,
                    zero_volatility=True,exact_linear_range=result)
    if any(b>a for v in (left,right) for a,b in zip(v.values,v.values[1:])):
        raise ValueError('Positive-volatility enclosure requires non-increasing value functions')
    grid = sorted(set(left.knots)|set(right.knots))
    if len(grid)>max_points:
        raise ValueError('Point limit cannot hold the initial full-domain partition')
    samples = {}
    def evaluate(x):
        mean_lo,mean_hi = outward(right.expectation(x,sd))
        f = left.value(x)
        samples[x] = (f,mean_lo,mean_hi,f-mean_hi,f-mean_lo)
    for x in grid:
        evaluate(x)
    curvature = Curvature(right,sd)
    limit = left.values[-1]-right.values[-1]
    cells_cache = {}
    for iteration in range(max_rounds+1):
        inf_upper = min([v[4] for v in samples.values()]+[limit])
        sup_lower = max([v[3] for v in samples.values()]+[limit])
        cells = []
        for a,b in zip(grid,grid[1:]):
            if (a,b) not in cells_cache:
                fa,ga_lo,ga_hi,ra_lo,ra_hi = samples[a]
                fb,gb_lo,gb_hi,rb_lo,rb_hi = samples[b]
                gamma = curvature.upper(a,b)
                allowance = gamma*(b-a)**2/8
                lower = max(fb-ga_hi,min(ra_lo,rb_lo)-allowance)
                upper = min(fa-gb_lo,max(ra_hi,rb_hi)+allowance)
                if lower>upper:
                    raise ArithmeticError('Disjoint mathematical cell enclosures')
                cells_cache[a,b] = dict(a=a,b=b,lower=lower,upper=upper,
                                        curvature_upper=gamma,chord_allowance=allowance)
            cells.append(cells_cache[a,b])
        tail_lo = left.values[-1]-samples[grid[-1]][2]
        tail_hi = limit
        # Intersecting with the exact limit also handles a loose mean ball.
        tail_lo = min(tail_lo,limit)
        lower = min([c['lower'] for c in cells]+[tail_lo])
        upper = max([c['upper'] for c in cells]+[tail_hi])
        tolerance_met = inf_upper-lower<=tolerance and upper-sup_lower<=tolerance
        finished = tolerance_met or iteration==max_rounds
        additions = [] if finished else [(c['a']+c['b'])/2 for c in cells
                        if c['lower']<inf_upper-tolerance or c['upper']>sup_lower+tolerance]
        if not additions or len(grid)+len(additions)>max_points:
            return dict(lower=lower,upper=upper,candidate_inf_upper=inf_upper,
                candidate_sup_lower=sup_lower,tolerance_met=tolerance_met,
                coverage_complete=True,tolerance=tolerance,rounds=iteration,
                points=[dict(state=x,value=samples[x][0],mean_lower=samples[x][1],
                    mean_upper=samples[x][2],residual_lower=samples[x][3],
                    residual_upper=samples[x][4]) for x in grid],
                cells=cells,tail=dict(start=grid[-1],lower=tail_lo,upper=tail_hi,limit=limit),
                sd_enclosure=outward(sd),zero_volatility=False)
        for x in additions:
            evaluate(x)
        grid=sorted((*grid,*additions))
    raise AssertionError('Unreachable refinement state')


def _value(spec):
    if not isinstance(spec,dict) or q(spec.get('tail_slope',0.)) != 0:
        raise ValueError('A constant-tail value specification is required')
    return RationalValue(spec['knots'],spec['values'])


def certify_model(spec,*,tolerance=Fraction(1,1000),max_rounds=24,max_points=200000):
    model,option = spec['model'],spec['option']
    if option['kind']!='put' or q(option['strike'])<=0 or q(model['premium_factor'])<=0:
        raise ValueError('Positive-strike put and positive premium factor required')
    nodes=spec['nodes']
    if not nodes or q(nodes[0]['time'])!=0 or nodes[0]['before'] is not None:
        raise ValueError('An initial post-event node at time zero is required')
    declared_events={q(e['ex_time']):q(e['discounted_cash']) for e in model['dividends']}
    if (len(declared_events)!=len(model['dividends'])
            or any(t<=0 or c<0 for t,c in declared_events.items())):
        raise ValueError('Distinct positive cash dates and non-negative caps required')
    events={t:c for t,c in declared_events.items() if t<=q(option['maturity'])}
    represented_events={q(node['time']) for node in nodes if node['before'] is not None}
    if represented_events!=set(events):
        raise ValueError('Every active cash event must have a before-event node')
    previous_time,previous_value=Fraction(0),_value(nodes[0]['after'])
    vertices={(Fraction(0),'after'):(0,previous_value)}
    edges=[]
    for node in nodes[1:]:
        time=q(node['time'])
        after=_value(node['after'])
        before=_value(node['before']) if node['before'] is not None else None
        if (before is not None)!=(time in events):
            raise ValueError('Cash vertices must match the declared event schedule')
        sd=transition_sd(model['volatility'],previous_time,time)
        result=residual_range(previous_value,before or after,sd,tolerance=tolerance,
                              max_rounds=max_rounds,max_points=max_points)
        edges.append(dict(kind='diffusion',from_time=previous_time,to_time=time,**result))
        if before is not None:
            vertices[time,'before']=(len(edges),before)
            result=linear_range(before,after,events[time])
            edges.append(dict(kind='cash',from_time=time,to_time=time,**result,
                              tolerance_met=True,coverage_complete=True))
        vertices[time,'after']=(len(edges),after)
        previous_time,previous_value=time,after
    if previous_time!=q(option['maturity']):
        raise ValueError('Final node must equal option maturity')
    rights=[]
    for right in spec['decisions']:
        key=(q(right['time']),right['phase'])
        if key not in vertices or q(right['settlement_factor'])<=0:
            raise ValueError('A declared vertex and positive settlement factor are required')
        index,value=vertices[key]
        defect=defect_range(value,option['strike'],right['settlement_factor'])
        rights.append(dict(vertex=index,time=key[0],phase=key[1],**defect))
    if (not rights or (rights[-1]['time'],rights[-1]['phase'])!=(q(option['maturity']),'after')
            or any(b['vertex']<=a['vertex'] for a,b in zip(rights,rights[1:]))):
        raise ValueError('Ordered distinct exercise rights ending at post-event maturity required')
    pairs=[]
    bound=Fraction(0)
    for i,a in enumerate(rights):
        for j in range(i+1,len(rights)):
            b=rights[j]
            between=edges[a['vertex']:b['vertex']]
            lo=b['lower']-a['upper']-sum((e['upper'] for e in between),Fraction(0))
            hi=b['upper']-a['lower']-sum((e['lower'] for e in between),Fraction(0))
            bound=max(bound,abs(lo),abs(hi))
            pairs.append(dict(earlier=i,later=j,lower=lo,upper=hi))
    return dict(case=spec.get('case'),approximation_fingerprint=spec.get('approximation_fingerprint'),
        gain_bound=bound/q(model['premium_factor']),edges=edges,rights=rights,pairs=pairs,
        tolerance_met=all(e['tolerance_met'] for e in edges),coverage_complete=True,
        ideal_model_bound_certified=True,recorded_floating_point_gains_certified=False)


def serialise(value):
    if isinstance(value,Fraction):
        return str(value)
    if isinstance(value,dict):
        return {k:serialise(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [serialise(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private(path):
    path=Path(path).resolve()
    if not path.is_relative_to(PRIVATE.resolve()):
        raise ValueError('Evidence must remain under data/private')
    return path


def verify_hash_mapping(mapping,base):
    if not isinstance(mapping,dict) or not mapping:
        raise ValueError('A non-empty evidence hash mapping is required')
    for name,digest in mapping.items():
        target=(base/name).resolve()
        if not target.is_relative_to(base.resolve()) or sha(target)!=digest:
            raise ValueError('Evidence hash or path differs: '+name)


def _hexadecimal_leaves(value,prefix=''):
    if isinstance(value,dict):
        result={}
        for name,item in value.items():
            result.update(_hexadecimal_leaves(item,prefix+'/'+name.replace('~','~0').replace('/','~1')))
        return result
    if isinstance(value,list):
        result={}
        for i,item in enumerate(value):
            result.update(_hexadecimal_leaves(item,prefix+'/'+str(i)))
        return result
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Non-finite exported input')
        return {prefix:value.hex()}
    return {}


def verify_export(input_path):
    """Check the complete export closure before constructing any output."""
    input_path=private(input_path)
    manifest_path=input_path.parent/'manifest.json'
    manifest_raw=manifest_path.read_bytes()
    manifest=json.loads(manifest_raw)
    if manifest.get('complete') is not True or manifest.get('original_evidence_unchanged') is not True:
        raise ValueError('A completed unchanged-evidence export is required')
    for key in ('input_sha256','source_sha256'):
        verify_hash_mapping(manifest[key],ROOT)
    verify_hash_mapping(manifest['artifact_sha256'],input_path.parent)
    if input_path.name not in manifest['artifact_sha256']:
        raise ValueError('The model input is not an exported artifact')
    raw=input_path.read_bytes()
    payload=json.loads(raw)
    if (payload.get('schema_version')!=1 or payload.get('approximation_fields')!=list(APPROXIMATION_FIELDS)
            or not isinstance(payload.get('models'),list)):
        raise ValueError('A version-one complete model export is required')
    if payload.get('exact_input_convention')!=manifest.get('exact_input_convention'):
        raise ValueError('Exported exact-input conventions differ')
    if [m['case'] for m in payload['models']]!=list(range(13)):
        raise ValueError('All thirteen ordered retained cases are required')
    for row in payload['models']:
        spec={key:row[key] for key in APPROXIMATION_FIELDS}
        digest=hashlib.sha256(json.dumps(spec,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if (row['approximation_fingerprint']!=digest
                or row['matching_arm_fingerprints']!={'reference':digest,'selected':digest}
                or row['residual_fine_fingerprint']!=digest
                or row['exact_binary64_hex']!=_hexadecimal_leaves(spec)):
            raise ValueError('Retained function identity or binary64 representation differs')
    return raw,payload,manifest_raw


def run(input_path,output,*,precision=128,tolerance=Fraction(1,1000)):
    input_path,output=private(input_path),private(output)
    if output.exists():
        raise FileExistsError('Use a fresh certificate directory')
    if type(precision) is not int or precision<96:
        raise ValueError('At least 96 bits of Arb precision required')
    raw,payload,export_manifest_raw=verify_export(input_path)
    source_sha=sha(Path(__file__))
    output.mkdir(parents=True)
    results=[]
    try:
        with ctx.workprec(precision):
            for spec in payload['models']:
                result=certify_model(spec,tolerance=tolerance)
                if not result['tolerance_met']:
                    raise ArithmeticError('Full-domain enclosure did not meet the requested tolerance')
                record=serialise(result)
                path=output/f"case_{spec['case']:03d}.json"
                path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
                results.append(dict(case=spec['case'],gain_bound=record['gain_bound'],
                    approximation_fingerprint=spec['approximation_fingerprint'],
                    diffusion_edges=sum(e['kind']=='diffusion' for e in result['edges']),
                    cash_edges=sum(e['kind']=='cash' for e in result['edges']),
                    cells=sum(len(e.get('cells',[])) for e in result['edges']),
                    points=sum(len(e.get('points',[])) for e in result['edges'])))
                print(f"Certified ideal-model residual envelope for case {spec['case']}",flush=True)
        final_raw,_,final_manifest_raw=verify_export(input_path)
        if final_raw!=raw or final_manifest_raw!=export_manifest_raw or sha(Path(__file__))!=source_sha:
            raise ValueError('Input or certifier changed during execution')
        summary=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),precision_bits=precision,
            endpoint_quantisation_bits=ENDPOINT_BITS,tolerance=str(tolerance),cases=results,
            exact_binary64_financial_inputs=True,exact_cash_and_payoff_rationals=True,
            volatility_time_formed_in_ball_arithmetic=True,
            ideal_model_bound_certified=True,recorded_floating_point_gains_certified=False,
            coverage_of_reported_intervals_certified=False,market_model_validity_certified=False,
            python=platform.python_version(),python_flint=flint.__version__,flint_version=flint.__FLINT_VERSION__)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        manifest=dict(complete=True,input=str(input_path.relative_to(ROOT)),
            input_sha256=hashlib.sha256(raw).hexdigest(),source=str(Path(__file__).relative_to(ROOT)),
            export_manifest=str((input_path.parent/'manifest.json').relative_to(ROOT)),
            export_manifest_sha256=hashlib.sha256(export_manifest_raw).hexdigest(),
            source_sha256=source_sha,artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return summary
    except Exception as exc:
        (output/'incomplete.json').write_text(json.dumps(dict(complete=False,
            exception_type=type(exc).__name__,reason=str(exc)),indent=2)+'\n')
        raise


if __name__=='__main__':
    if Path(sys.prefix).resolve()!=(Path.home()/'.venvs/qf-certification').resolve():
        raise RuntimeError('Use the separate qf-certification interpreter')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',type=int,default=128)
    args=parser.parse_args()
    run(args.input,args.output,precision=args.precision)
