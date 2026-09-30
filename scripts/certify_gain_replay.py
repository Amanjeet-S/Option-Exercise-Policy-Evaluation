"""Enclose deterministic replay of retained simulation innovations and policies.

This calculation certifies arithmetic for the exported finite innovation array.
It does not establish the generator's sampling law or statistical coverage.
Run in the separate qf-certification environment.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import flint
from flint import arb, ctx

from certify_residual_envelope import (
    RationalValue, q, ball, outward, transition_sd, serialise,
    verify_hash_mapping, private, ROOT, ENDPOINT_BITS,
    _hexadecimal_leaves, APPROXIMATION_FIELDS,
)


ZERO = (Fraction(0), Fraction(0))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_interval(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            'A pair of interval endpoints is required')
    lo, hi = map(q, value)
    require(lo <= hi, 'Interval endpoints must be ordered')
    return lo, hi


def enclose(lower, upper):
    lower, upper = checked_interval((lower, upper))
    return ball(lower).union(ball(upper))


def add(left, right):
    return left[0]+right[0], left[1]+right[1]


def subtract(left, right):
    return left[0]-right[1], left[1]-right[0]


def scale(value, factor):
    factor = q(factor)
    endpoints = (value[0]*factor, value[1]*factor)
    return min(endpoints), max(endpoints)


def _monotone(value):
    require(all(b <= a for a, b in zip(value.values, value.values[1:])),
            'Non-increasing retained values are required')


def linear_interval(value, state):
    lo, hi = checked_interval(state)
    require(lo >= 0, 'A non-negative state interval is required')
    _monotone(value)
    return value.value(hi), value.value(lo)


def expectation_interval(value, state, sd):
    lo, hi = checked_interval(state)
    require(lo >= 0, 'A non-negative state interval is required')
    _monotone(value)
    if sd.is_zero():
        return linear_interval(value, state)
    lower = outward(value.expectation(hi, sd))[0]
    upper = outward(value.expectation(lo, sd))[1]
    require(lower <= upper, 'Conditional-expectation enclosure is reversed')
    return lower, upper


def _validate_spec(spec):
    model, option, nodes = spec['model'], spec['option'], spec['nodes']
    require(option['kind'] == 'put' and q(option['strike']) > 0
            and q(model['premium_factor']) > 0 and q(model['volatility']) >= 0,
            'A positive-strike put with a positive premium factor is required')
    require(q(model['quoted_spot']) > 0 and q(model['initial_share_factor']) > 0,
            'Positive initial state factors are required')
    times = [q(n['time']) for n in nodes]
    require(len(times) >= 2 and times[0] == 0 and times[-1] == q(option['maturity'])
            and all(b > a for a, b in zip(times, times[1:])), 'Invalid node calendar')
    declared = [(q(d['ex_time']), q(d['discounted_cash'])) for d in model['dividends']]
    require(all(t > 0 and c >= 0 for t, c in declared)
            and all(b[0] > a[0] for a, b in zip(declared, declared[1:])),
            'Distinct ordered cash dates and non-negative caps are required')
    events = {t:c for t,c in declared if t <= times[-1]}
    require({t for t,n in zip(times,nodes) if n['before'] is not None} == set(events)
            and nodes[0]['before'] is None, 'Cash node coverage differs')
    keys = [(q(d['time']), d['phase']) for d in spec['decisions']]
    vertices = {(t,'after') for t in times} | {(t,'before') for t in events}
    require(keys and len(keys) == len(set(keys)) and all(k in vertices for k in keys)
            and keys == sorted(keys, key=lambda k:(k[0], k[1] == 'after'))
            and keys[-1] == (times[-1],'after'), 'Invalid exercise calendar')
    require(all(q(d['settlement_factor']) > 0 for d in spec['decisions']),
            'Positive exercise settlement factors are required')
    return times, events


def propagate(spec, normals):
    times, events = _validate_spec(spec)
    model = spec['model']
    sigma = q(model['volatility'])
    require(len(normals) == (len(times)-1 if sigma else 0),
            'One retained innovation per positive-volatility diffusion interval is required')
    innovations = [q(z) for z in normals]
    initial = q(model['quoted_spot'])*q(model['initial_share_factor'])
    current = (initial, initial)
    states = {(Fraction(0),'after'):current}
    for j,(t0,t1) in enumerate(zip(times,times[1:])):
        if sigma and current != ZERO:
            dt = t1-t0
            exponent = ball(-sigma*sigma*dt/2) + transition_sd(sigma,t0,t1)*ball(innovations[j])
            factor = exponent.exp()
            lo = outward(ball(current[0])*factor)[0]
            hi = outward(ball(current[1])*factor)[1]
            current = (max(Fraction(0),lo),hi)
            require(current[0] <= current[1], 'Invalid propagated state enclosure')
        if t1 in events:
            states[t1,'before'] = current
            cap = events[t1]
            current = tuple(max(Fraction(0),x-cap) for x in current)
        states[t1,'after'] = current
    return states


def reward_interval(spec, right, state):
    lo, hi = checked_interval(state)
    require(lo >= 0, 'A non-negative exercise state is required')
    threshold = q(spec['option']['strike'])*q(right['settlement_factor'])
    return max(Fraction(0),threshold-hi), max(Fraction(0),threshold-lo)


def prediction_interval(rule, state):
    coefficients = rule['coefficients']
    require(coefficients and q(rule['scale']) > 0, 'A fitted rule with positive scale is required')
    x = (enclose(*state)-ball(rule['centre']))/ball(rule['scale'])
    result = ball(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result*x+ball(coefficient)
    lo, hi = outward(result)
    return max(Fraction(0),lo), max(Fraction(0),hi)


def stopping_candidates(policy, states):
    rights, rules = policy['decisions'], policy['rules']
    require(len(rights) == len(rules) and rights and rules[-1]['status'] == 'terminal',
            'Each exercise right requires a rule and the last rule must be terminal')
    candidates = []
    for j,(right,rule) in enumerate(zip(rights[:-1],rules[:-1])):
        if not rule['coefficients']:
            continue
        state = states[q(right['time']),right['phase']]
        immediate = reward_interval(policy,right,state)
        if immediate[1] == 0:
            continue
        prediction = prediction_interval(rule,state)
        if immediate[0] > prediction[1]:
            candidates.append(j)
            return tuple(candidates)
        if immediate[1] > prediction[0]:
            candidates.append(j)
    return tuple((*candidates,len(rights)-1))


def _value(record):
    require(q(record['tail_slope']) == 0, 'Constant-tail functions are required')
    value = RationalValue(record['knots'],record['values'])
    _monotone(value)
    return value


def gain_interval(spec, states, low_index, high_index, cache=None):
    rights = spec['decisions']
    require(type(low_index) is int and type(high_index) is int
            and 0 <= low_index < len(rights) and 0 <= high_index < len(rights),
            'Valid stopping indices are required')
    if low_index == high_index:
        return ZERO
    cache = {} if cache is None else cache
    low, high = rights[low_index],rights[high_index]
    rewards = subtract(reward_interval(spec,high,states[q(high['time']),high['phase']]),
                       reward_interval(spec,low,states[q(low['time']),low['phase']]))
    low_time, high_time = q(low['time']),q(high['time'])
    start,end = min(low_time,high_time),max(low_time,high_time)
    martingale_difference = ZERO
    nodes = spec['nodes']
    for previous,node in zip(nodes,nodes[1:]):
        t0,t1 = q(previous['time']),q(node['time'])
        if not start < t1 <= end:
            continue
        if t1 not in cache:
            phase = 'before' if node['before'] is not None else 'after'
            value = _value(node['before'] or node['after'])
            endpoint = linear_interval(value,states[t1,phase])
            sd = transition_sd(spec['model']['volatility'],t0,t1)
            mean = expectation_interval(value,states[t0,'after'],sd)
            cache[t1] = subtract(endpoint,mean)
        martingale_difference = add(martingale_difference,cache[t1])
    if high_time < low_time:
        martingale_difference = scale(martingale_difference,-1)
    return scale(subtract(rewards,martingale_difference),1/q(spec['model']['premium_factor']))


def certify_paths(payload):
    require(payload.get('schema_version') == 1, 'Unknown replay schema')
    spec, policies = payload['model_spec'],payload['policies']
    times,_ = _validate_spec(spec)
    require(len(policies) == 2, 'Two retained policies are required')
    for policy in policies:
        require(all(policy[key] == spec[key] for key in ('model','option','decisions')),
                'Policy and approximation financial specifications differ')
    n = len(payload['normals'])
    require(n >= 2 and all(len(payload[key]) == n for key in
            ('states','stopping_indices','quoted_gains')), 'Replay row coverage differs')
    expected_times = times[1:] if q(spec['model']['volatility']) else []
    require([q(t) for t in payload['innovation_times']] == expected_times,
            'Innovation columns differ from chronological diffusion times')
    observation_keys = [(q(d['time']),d['phase']) for d in payload['observation_rights']]
    expected_observations = {(q(d['time']),d['phase']) for d in spec['decisions']}
    expected_observations.update((q(node['time']),'after') for node in spec['nodes'][1:])
    expected_observations.update((q(node['time']),'before') for node in spec['nodes'][1:]
                                 if node['before'] is not None)
    require(observation_keys == sorted(expected_observations,key=lambda k:(k[0],k[1]=='after')),
            'Complete ordered observation columns are required')
    rows=[]
    total=ZERO
    original_sum=Fraction(0)
    largest_state_error=Fraction(0)
    for index,innovations in enumerate(payload['normals']):
        states=propagate(spec,innovations)
        require(set(observation_keys).issubset(states), 'An observation has no propagated vertex')
        original_states=payload['states'][index]
        require(len(original_states) == len(observation_keys), 'Original state row dimension differs')
        state_errors=[]
        for key,observed in zip(observation_keys,original_states):
            observed=q(observed)
            require(observed >= 0, 'Retained states must be non-negative')
            state_errors.append(subtract(states[key],(observed,observed)))
        state_error=max(abs(x) for interval in state_errors for x in interval)
        largest_state_error=max(largest_state_error,state_error)
        candidates=[stopping_candidates(policy,states) for policy in policies]
        stops=payload['stopping_indices'][index]
        require(len(stops)==2 and all(type(s) is int and 0<=s<len(spec['decisions']) for s in stops),
                'Two valid implemented stopping indices are required')
        cache={}
        intervals=[gain_interval(spec,states,a,b,cache) for a in candidates[0] for b in candidates[1]]
        gain=(min(x[0] for x in intervals),max(x[1] for x in intervals))
        original=q(payload['quoted_gains'][index])
        error=subtract(gain,(original,original))
        total=add(total,gain)
        original_sum+=original
        rows.append(dict(path=index,stopping_candidates=candidates,
            implemented_stopping_indices=stops,
            stopping_decisions_match=all(c==(s,) for c,s in zip(candidates,stops)),
            ambiguous_policy_count=sum(len(c)>1 for c in candidates),
            proved_stopping_index_difference_count=sum(len(c)==1 and c[0]!=s
                                                       for c,s in zip(candidates,stops)),
            gain_interval=gain,recorded_gain=original,error_interval=error,
            absolute_error_bound=max(map(abs,error)),state_error_bound=state_error,
            state_enclosures=[dict(time=t,phase=phase,lower=interval[0],upper=interval[1])
                for (t,phase),interval in sorted(states.items(),key=lambda row:(row[0][0],row[0][1]=='after'))],
            evaluated_martingale_increments=len(cache)))
    mean=scale(total,Fraction(1,n))
    recorded=q(payload['record']['evaluation']['gain']['estimate']['value'])
    error=subtract(mean,(recorded,recorded))
    return dict(complete=True,paths=n,rows=rows,mean_interval=mean,
        recorded_mean=recorded,recorded_path_mean_exact=original_sum/n,
        recorded_mean_aggregation_error=recorded-original_sum/n,
        mean_error_interval=error,mean_absolute_error_bound=max(map(abs,error)),
        maximum_path_error_bound=max(r['absolute_error_bound'] for r in rows),
        maximum_state_error_bound=largest_state_error,
        ambiguous_policy_count=sum(r['ambiguous_policy_count'] for r in rows),
        proved_stopping_index_difference_count=sum(r['proved_stopping_index_difference_count'] for r in rows),
        all_stopping_indices_match_on_exported_paths=all(r['stopping_decisions_match'] for r in rows),
        evaluated_martingale_increments=sum(r['evaluated_martingale_increments'] for r in rows),
        exact_binary64_innovation_replay_certified=True,
        ideal_gaussian_sampling_law_certified=False,statistical_coverage_certified=False,
        training_algorithm_error_certified=False,new_independent_paths=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_input(path):
    path=private(path)
    manifest_path=path.parent/'manifest.json'
    manifest_raw=manifest_path.read_bytes()
    manifest=json.loads(manifest_raw)
    require(manifest.get('complete') is True and manifest.get('original_evidence_unchanged') is True,
            'A complete unchanged-evidence replay export is required')
    for group in ('input_sha256','source_sha256'):
        verify_hash_mapping(manifest[group],ROOT)
    verify_hash_mapping(manifest['artifact_sha256'],path.parent)
    require(path.name in manifest['artifact_sha256'], 'Replay input is not an exported artifact')
    raw=path.read_bytes()
    payload=json.loads(raw)
    require(payload.get('binary64_hex') == _hexadecimal_leaves(
            {key:value for key,value in payload.items() if key!='binary64_hex'}),
            'Replay binary64 hexadecimal identities differ')
    spec=payload['model_spec']
    fingerprint=hashlib.sha256(json.dumps({key:spec[key] for key in APPROXIMATION_FIELDS},
                                        sort_keys=True,allow_nan=False).encode()).hexdigest()
    record=payload['record']
    require(fingerprint == spec['approximation_fingerprint']
            == record['evaluation']['approximation_fingerprint'], 'Approximation identity differs')
    policy_fingerprints=[hashlib.sha256(json.dumps(p,sort_keys=True,allow_nan=False).encode()).hexdigest()
                         for p in payload['policies']]
    require(policy_fingerprints == record['training']['policy_fingerprints']
            == record['evaluation']['policy_fingerprints'], 'Retained policy identity differs')
    n=len(payload['normals'])
    require(n == payload['selection']['evaluation_paths'] == record['evaluation']['evaluation_paths']
            == record['evaluation']['gain']['estimate']['observations'], 'Retained evaluation count differs')
    require(payload['selection'] == manifest['selection'], 'Exported block selection differs')
    return raw,payload,manifest_raw


def run(input_path,output,precision=128):
    input_path,output=private(input_path),private(output)
    require(not output.exists(), 'Use a fresh arithmetic replay directory')
    require(type(precision) is int and precision>=96, 'At least 96 bits of precision required')
    raw,payload,manifest_raw=verify_input(input_path)
    source_sha=sha(Path(__file__))
    output.mkdir(parents=True)
    try:
        with ctx.workprec(precision):
            result=certify_paths(payload)
        final_raw,_,final_manifest=verify_input(input_path)
        require(raw==final_raw and manifest_raw==final_manifest and sha(Path(__file__))==source_sha,
                'Replay input or source changed during execution')
        result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),precision_bits=precision,
            endpoint_quantisation_bits=ENDPOINT_BITS,python=platform.python_version(),
            python_flint=flint.__version__,flint_version=flint.__FLINT_VERSION__)
        (output/'enclosures.json').write_text(json.dumps(serialise(result),indent=2,allow_nan=False)+'\n')
        summary={k:v for k,v in result.items() if k!='rows'}
        (output/'summary.json').write_text(json.dumps(serialise(summary),indent=2,allow_nan=False)+'\n')
        manifest=dict(complete=True,input=str(input_path.relative_to(ROOT)),input_sha256=hashlib.sha256(raw).hexdigest(),
            export_manifest=str((input_path.parent/'manifest.json').relative_to(ROOT)),
            export_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            source=str(Path(__file__).relative_to(ROOT)),source_sha256=source_sha,
            artifact_sha256={p.name:sha(p) for p in output.iterdir()},publication_clearance=False)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        return serialise(summary)
    except Exception as exc:
        (output/'incomplete.json').write_text(json.dumps(dict(complete=False,
            exception_type=type(exc).__name__,reason=str(exc)),indent=2)+'\n')
        raise


if __name__=='__main__':
    require(Path(sys.prefix).resolve()==(Path.home()/'.venvs/qf-certification').resolve(),
            'Use the separate qf-certification environment')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',type=int,default=128)
    args=parser.parse_args()
    print(json.dumps(run(args.input,args.output,args.precision),indent=2))
