"""Audit bounded-cluster component inference without importing its implementation.

I reconstruct the pilot statistics from pairwise mean differences and saved
standard errors, solve the polygon in population-component coordinates, and
partition every allocation comparison at all affine crossings. Radius checks
use the same Arb library at higher precision, not an independent library.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

import flint
from flint import arb, ctx, fmpq

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT/'data/private'
BASE = PRIVATE/'component_confidence/2026-09-28'
PREVIOUS = PRIVATE/'allocation_variance_sensitivity/2026-09-28'
CERTIFIED = PRIVATE/'cpu_budget_transfer/2026-09-27'
PILOT = PRIVATE/'training_allocation/2026-09-27/pilot_v1'
N0, M, K = 4096, 32, 13


def require(test, message):
    if not test:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON value: '+value)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(value):
    require(type(value) is str, 'Expected rational string')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational string')
    return result


def scalar(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid saved scalar')
    return Fraction(value)


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def save(path, record):
    path.write_text(json.dumps(encode(record), indent=2, allow_nan=False)+'\n')


def same(actual, expected, label):
    expected = encode(expected)
    require(type(actual) is type(expected), 'Record type differs: '+label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Record fields differ: '+label)
        for key, value in expected.items():
            same(actual[key], value, label+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Record length differs: '+label)
        for i, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label+'['+str(i)+']')
    else:
        require(actual == expected, 'Independent value differs: '+label)


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Require private storage')
    return path


def rectangle_polygon(s, u, hv, hu, cap, delta):
    """Enumerate the feasible vertices directly in (A, B) coordinates."""
    vl, vu, ul, uu = delta*s, s+hv, delta*u, u+hu
    # a*A+b*B <= c, with separate non-negativity restrictions.
    constraints = [(-1, 0, 0), (0, -1, 0), (1, 1, cap),
                   (-1, Fraction(-1, N0), -vl), (1, Fraction(1, N0), vu),
                   (0, Fraction(-1, N0), -ul), (0, Fraction(1, N0), uu)]
    vertices = set()
    for (a, b, c), (d, e, f) in combinations(constraints, 2):
        determinant = a*e-b*d
        if determinant:
            A = Fraction(c*e-b*f, determinant)
            B = Fraction(a*f-c*d, determinant)
            if all(x*A+y*B <= z for x, y, z in constraints):
                vertices.add((A, B))
    transformed = sorted((A+B/N0, B/N0, A, B) for A, B in vertices)
    shares = [B/(N0*A+B) for A, B in vertices if N0*A+B > 0]
    return dict(intervals={'v': [vl, vu], 'u': [ul, uu]},
                vertices=[dict(v=v, u=w, A=A, B=B) for v, w, A, B in transformed],
                share_bounds=dict(lower=min(shares), upper=max(shares)) if shares else None,
                zero_variance_possible=(Fraction(0), Fraction(0)) in vertices,
                status='empty' if not vertices else 'nondegenerate' if shares else 'degenerate_zero')


def value(row, weight):
    return ((1-weight)+Fraction(N0, row['evaluation_paths'])*weight)/row['repetitions']


def comparison(candidates, low, high, target):
    """Partition at all pairwise crossings, including non-optimal crossings."""
    rows = sorted(candidates, key=lambda r: (r['evaluation_paths'], r['repetitions']))
    require(rows and len({r['evaluation_paths'] for r in rows}) == len(rows), 'Candidate grid differs')
    for row in rows:
        require(set(row) == {'evaluation_paths', 'repetitions'} and
                all(type(row[k]) is int and row[k] > 0 for k in row), 'Invalid candidate')
    nodes = {low, high}
    for left, right in combinations(rows, 2):
        dz = value(left, 0)-value(right, 0)
        do = value(left, 1)-value(right, 1)
        if dz != do:
            crossing = dz/(dz-do)
            if low <= crossing <= high:
                nodes.add(crossing)
    nodes = sorted(nodes)
    middle = [(left+right)/2 for left, right in zip(nodes, nodes[1:])]
    weak, selected = set(), set()
    minimisers, criterion = {}, {}
    for w in nodes+middle:
        numbers = [value(row, w) for row in rows]
        best = min(numbers)
        ties = [row for row, number in zip(rows, numbers) if number == best]
        weak.update(row['evaluation_paths'] for row in ties)
        selected.add(ties[0]['evaluation_paths'])
        minimisers[w], criterion[w] = ties, best
    target_row = next(row for row in rows if row['evaluation_paths'] == target)
    ratios = {w: value(target_row, w)/criterion[w] for w in nodes}
    for left, right in zip(nodes, nodes[1:]):
        w = (left+right)/2
        r = value(target_row, w)/criterion[w]
        require(min(ratios[left], ratios[right]) <= r <= max(ratios[left], ratios[right]),
                'Fractional-linear within-cell monotonicity differs')
    maximum = max(ratios.values())
    ends = sorted({w for w in (low, high) if ratios[w] == maximum})
    require(ends, 'An interior crossing exceeds both endpoint regrets')
    return dict(weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
                uniformly_selected_size=next(iter(selected)) if len(selected) == 1 else None,
                retained_choice_regret=dict(maximum=maximum, lower_ratio=ratios[low],
                    upper_ratio=ratios[high], worst_endpoints=ends)), len(nodes), len(middle)


def higher_precision_factors(delta):
    with ctx.workprec(192):
        log_inverse = arb(fmpq(delta.denominator, delta.numerator)).log()
        # Algebraically reordered relative to production's nested divisions.
        factors = {'variance_factor': (2*log_inverse/(M//2)).sqrt(),
                   'evaluation_factor': (log_inverse/(2*M*(N0-1)**2)).sqrt()}
        return {key: dict(lower=Fraction(str(ball.lower().fmpq())),
                          upper=Fraction(str(ball.upper().fmpq())))
                for key, ball in factors.items()}


def audit(inputs, run, output, tamper=True):
    inputs, run, output = map(private, (inputs, run, output))
    require(not output.exists() and not output.is_relative_to(inputs) and
            not output.is_relative_to(run), 'Use a fresh, separate audit directory')
    evidence = {}

    def retain(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence')
        digest = sha(path); name = str(path.relative_to(ROOT))
        require(expected is None or expected == digest, 'Evidence checksum differs: '+name)
        require(name not in evidence or evidence[name] == digest, 'Conflicting evidence hash')
        evidence[name] = digest
        return path

    def manifest(folder):
        data = read(retain(folder/'manifest.json'))
        require(data.get('complete') is True and not (folder/'failure.json').exists(), 'Incomplete stage')
        for key, anchor in (('evidence_sha256', ROOT), ('artifact_sha256', folder)):
            require(isinstance(data[key], dict) and data[key], 'Missing evidence map')
            for name, digest in data[key].items():
                path = (anchor/name).resolve()
                require(path.is_relative_to(anchor.resolve()), 'Evidence escapes root')
                retain(path, digest)
        require(set(data['artifact_sha256']) == {p.name for p in folder.glob('*.json') if p.name != 'manifest.json'},
                'Unbound or missing JSON artifact')
        return data

    def bound(path):
        require(str(path.resolve().relative_to(ROOT)) in evidence, 'Unbound consumed evidence')
        return read(path)

    im, rm = manifest(inputs), manifest(run)
    require(all(rm['evidence_sha256'].get(k) == v for k, v in im['evidence_sha256'].items()),
            'Input closure differs in run')
    for p in (inputs/'manifest.json', inputs/'inputs.json'):
        require(rm['evidence_sha256'].get(str(p.relative_to(ROOT))) == sha(p), 'Run is not bound to inputs')
    payload = bound(inputs/'inputs.json')
    same({k:v for k,v in payload.items() if k != 'models'}, dict(cases=13, pilot_pairs=32,
        pilot_paths=4096, family_alpha='1/20', one_sided_tail_probability='1/1040', precision_bits=128,
        variance_summary_error='0', evaluation_summary_error='0',
        summary_error_status='Optimistic diagnostic only; no complete pilot summary-error certificate'), 'design')
    delta = q(payload['one_sided_tail_probability'])
    require(4*K*delta == q(payload['family_alpha']) == Fraction(1,20), 'Union allocation differs')
    require([m['case'] for m in payload['models']] == list(range(K)), 'Case coverage differs')
    old = bound(PREVIOUS/'inputs_v1/inputs.json')
    factors = bound(run/'radius_factors.json')
    require(set(factors) == {'precision_bits','pairs','paths','tail_probability','variance_factor','evaluation_factor','upper_endpoints_used','python','python_flint','source_sha256'},
            'Radius fields differ')
    same({k:factors[k] for k in ('precision_bits','tail_probability','upper_endpoints_used')},
         dict(precision_bits=128,tail_probability=delta,upper_endpoints_used=True), 'radius design')
    same(factors['pairs'], M, 'radius pairs'); same(factors['paths'], N0, 'radius paths')
    same(factors['source_sha256'], sha(ROOT/'scripts/certify_component_radii.py'), 'radius source')
    require(all(isinstance(factors[k],str) and factors[k] for k in ('python','python_flint')), 'Missing radius runtime')
    higher = higher_precision_factors(delta)
    for key in ('variance_factor','evaluation_factor'):
        require(set(factors[key]) == {'lower','upper'}, 'Enclosure fields differ')
        lo, hi = q(factors[key]['lower']), q(factors[key]['upper'])
        require(0 < lo <= higher[key]['lower'] <= higher[key]['upper'] <= hi,
                'Higher-precision radius is not contained in retained enclosure')
    records = []; moment_pairs = 0; polygons = 0; decision_count = 0; regret_nodes = 0; midpoint_count = 0
    for model in payload['models']:
        case = model['case']; label = 'case_'+str(case)
        require(set(model) == {'case','gain_bound','approximation_fingerprint','observed_variance','evaluation_mean_variance','regimes'},
                'Model fields differ')
        cert = bound(CERTIFIED/f'certified_residual_run_v1/case_{case:03d}.json')
        require(cert['case'] == case and cert['coverage_complete'] is True and
                cert['ideal_model_bound_certified'] is True and cert['recorded_floating_point_gains_certified'] is False,
                'Gain certificate scope differs')
        L = q(cert['gain_bound']); require(L > 0, 'Nonpositive gain bound')
        same(model['gain_bound'], L, label+'.gain_bound')
        same(model['approximation_fingerprint'], cert['approximation_fingerprint'], label+'.fingerprint')
        pilot = bound(PILOT/f'blocks/case_{case:03d}.json')
        require(pilot['case'] == case and [r['repetition'] for r in pilot['statistical']] == list(range(M)),
                'Pilot row coverage differs')
        means, errors = [], []
        for row in pilot['statistical']:
            ev = row['evaluation']; est = ev['gain']['estimate']
            require(ev['approximation_fingerprint'] == cert['approximation_fingerprint'] and
                    ev['evaluation_paths'] == est['observations'] == N0, 'Pilot and support differ')
            means.append(scalar(est['value'])); errors.append(scalar(est['standard_error']))
            require(errors[-1] >= 0, 'Negative standard error')
        # S^2 = sum_{r<s}(W_r-W_s)^2/[M(M-1)].
        s = sum(((x-y)**2 for x,y in combinations(means,2)),Fraction(0))/(M*(M-1))
        u = sum((x*x for x in errors),Fraction(0))/M
        moment_pairs += M*(M-1)//2
        same(model['observed_variance'], s, label+'.observed_variance')
        same(model['evaluation_mean_variance'], u, label+'.evaluation_mean_variance')
        same(model['regimes'], old['models'][case]['regimes'], label+'.retained_regimes')
        require([(r['accounting'],r['scope']) for r in model['regimes']] ==
                [(a,b) for a in ('marginal','screening_charged') for b in ('pilot','validation','joint')], 'Regime coverage differs')
        cap = L*L; hv = cap*q(factors['variance_factor']['upper']); hu = cap*q(factors['evaluation_factor']['upper'])
        polygon = rectangle_polygon(s,u,hv,hu,cap,delta)
        require(polygon['status'] == 'nondegenerate', 'Unexpected region status')
        polygons += len(polygon['vertices'])
        low, high = polygon['share_bounds']['lower'], polygon['share_bounds']['upper']
        regimes = []
        for regime in model['regimes']:
            analysis, nodes, midpoints = comparison(regime['candidates'],low,high,regime['retained_choice']['evaluation_paths'])
            regimes.append(dict(**regime,**analysis)); decision_count += 1; regret_nodes += nodes; midpoint_count += midpoints
        expected = dict(case=case,gain_bound=L,bound_squared=cap,observed_variance=s,evaluation_mean_variance=u,
            variance_radius=hv,evaluation_radius=hu,variance_radius_to_observed=hv/s if s>0 else None,
            evaluation_radius_to_observed=hu/u if u>0 else None,region=polygon,regimes=regimes,
            radii_rounded_outwards=True,complete_statistical_coverage_certified=False,optimistic_zero_summary_error=True)
        same(bound(run/f'case_{case:02d}.json'),expected,label)
        records.append(dict(case=case,share_bounds=polygon['share_bounds'],
            variance_radius_to_observed=expected['variance_radius_to_observed'],evaluation_radius_to_observed=expected['evaluation_radius_to_observed'],
            uniquely_selected_regimes=sum(r['uniformly_selected_size'] is not None for r in regimes),
            regimes=[dict(accounting=r['accounting'],scope=r['scope'],selected_sizes=r['selected_sizes'],
                maximum_retained_choice_regret=r['retained_choice_regret']['maximum']) for r in regimes]))
    summary = bound(run/'summary.json')
    same({k:v for k,v in summary.items() if k not in ('python','certification_python','python_flint')}, dict(complete=True,cases=13,
        fixed_cost_regimes=78,nominal_family_confidence='19/20',
        scope='Ideal bounded independent-cluster procedure; optimistic zero-summary-error evaluation',
        certified_radius_enclosures=True,complete_pilot_summary_errors_certified=False,
        complete_statistical_coverage_certified=False,new_simulations=0,policy_refits=0,
        unique_allocation_regimes=sum(r['uniquely_selected_regimes'] for r in records),models=records,publication_clearance=False), 'summary')
    require(all(isinstance(summary.get(k),str) and summary[k] for k in ('python','certification_python','python_flint')), 'Missing runtime provenance')
    same(summary['certification_python'],factors['python'],'certification runtime')
    same(summary['python_flint'],factors['python_flint'],'certification library')
    retain(Path(__file__))
    tamper_checks = tamper_test(inputs,run) if tamper else []
    # Recheck evidence after the independent calculations and negative controls.
    for name, digest in evidence.items():
        require(sha(ROOT/name) == digest, 'Evidence changed during audit: '+name)
    result = dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),cases=K,
        pilot_rows=K*M,pairwise_mean_differences=moment_pairs,component_vertices=polygons,
        decision_regimes=decision_count,regret_crossing_and_endpoint_nodes=regret_nodes,
        open_cell_midpoints=midpoint_count,independent_component_coordinate_geometry=True,
        higher_precision_bits=192,higher_precision_radius_enclosures=higher,
        retained_precision_bits=128,higher_precision_contained_in_retained_enclosures=True,
        arithmetic_library='Arb through python-flint, shared with the analysis',independent_arithmetic_library=False,
        ideal_sampling_law_certified=False,complete_pilot_summary_errors_certified=False,
        complete_statistical_coverage_certified=False,optimistic_zero_summary_error=True,
        new_simulations=0,policy_refits=0,publication_clearance=False,tamper_checks=tamper_checks,
        python=platform.python_version(),python_flint=flint.__version__,evidence_sha256=evidence)
    output.mkdir(parents=True)
    save(output/'checks.json',result)
    return result


def tamper_test(inputs, run):
    """Both negative controls must fail before an audit directory is created."""
    results = []
    with TemporaryDirectory(prefix='component_audit_control_',dir=BASE) as name:
        base = Path(name)
        for mode in ('checksum','rehashed_component'):
            copied = base/mode
            shutil.copytree(run,copied)
            altered = copied/'case_00.json'; data = read(altered)
            data['observed_variance'] = str(q(data['observed_variance'])+1)
            save(altered,data)
            if mode == 'rehashed_component':
                path = copied/'manifest.json'; manifest = read(path)
                manifest['artifact_sha256'][altered.name] = sha(altered); save(path,manifest)
            target = base/(mode+'_audit')
            try:
                audit(inputs,copied,target,tamper=False)
            except ValueError as error:
                text = str(error)
                expected = 'Evidence checksum differs' if mode == 'checksum' else 'Independent value differs: case_0.observed_variance'
                require(expected in text, 'Negative control failed for an unexpected reason: '+text)
                require(not target.exists(), 'Rejected evidence created an audit output')
                results.append(dict(control=mode,rejected=True,output_created=False,reason=text))
            else:
                raise ValueError('Altered evidence passed the independent audit')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,default=BASE/'inputs_v1')
    parser.add_argument('--run',type=Path,default=BASE/'run_v1')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.inputs,args.run,args.output)
    print(json.dumps({k:result[k] for k in ('complete','cases','decision_regimes','pairwise_mean_differences','component_vertices','regret_crossing_and_endpoint_nodes','open_cell_midpoints')},indent=2))


if __name__ == '__main__':
    main()
