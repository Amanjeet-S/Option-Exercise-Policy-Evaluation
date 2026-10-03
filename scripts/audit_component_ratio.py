"""Independently audit retained component-ratio inference and allocation maps.

Raw polynomial moments replace the analysis's centred influence calculations.
All pairwise allocation crossings partition the comparison independently of
the production weak-optimality-region algorithm. Arb display checks share the
analysis's arithmetic library and do not establish statistical coverage.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'component_ratio/2026-09-28'
OLD = PRIVATE / 'allocation_variance_sensitivity/2026-09-28'
TRAINING = PRIVATE / 'training_allocation/2026-09-27'
N0 = 4096
Z = Fraction(13, 4)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(items):
    result = {}
    for name, value in items:
        require(name not in result, 'Duplicate JSON key')
        result[name] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q(value):
    require(type(value) is str, 'Expected rational string')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational string')
    return result


def scalar(value):
    require(type(value) in (int, float) and math.isfinite(value),
            'Invalid saved scalar')
    return Fraction(value)


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(name): encode(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def save(path, value):
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False) + '\n')


def same(actual, expected, label):
    expected = encode(expected)
    require(type(actual) is type(expected), 'Record type differs: ' + label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Record fields differ: ' + label)
        for name, value in expected.items():
            same(actual[name], value, label + '.' + name)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Record length differs: ' + label)
        for i, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label + '[' + str(i) + ']')
    else:
        require(actual == expected, 'Independent value differs: ' + label)


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Require private storage')
    return path


def raw_moments(means, standard_errors, paths):
    """Use raw moments through degree four to reconstruct the covariance."""
    count = len(means)
    require(count >= 3 and len(standard_errors) == count, 'Invalid row count')
    require(type(paths) is int and paths >= 2, 'Invalid inner path count')
    require(all(x >= 0 for x in standard_errors), 'Negative standard error')
    x, y = list(means), [s * s for s in standard_errors]
    avg = lambda values: sum(values, Fraction(0)) / count
    x1, x2, x3, x4 = [avg(v ** order for v in x) for order in range(1, 5)]
    y1, y2 = avg(y), avg(v * v for v in y)
    xy = avg(a * b for a, b in zip(x, y))
    xxy = avg(a * a * b for a, b in zip(x, y))
    h1 = x2 - x1 * x1
    h2 = x4 - 4 * x1 * x3 + 6 * x1 * x1 * x2 - 3 * x1 ** 4
    hy = xxy - 2 * x1 * xy + x1 * x1 * y1
    vh, vy, chy = h2 - h1 * h1, y2 - y1 * y1, hy - h1 * y1
    k = Fraction(paths, N0)
    dd = (vh + 2 * (k - 1) * chy + (k - 1) ** 2 * vy) / (count - 1)
    de = k * (chy + (k - 1) * vy) / (count - 1)
    ee = k * k * vy / (count - 1)
    observed = count * h1 / (count - 1)
    denominator, numerator = observed + (k - 1) * y1, k * y1
    point = numerator / denominator if denominator > 0 else None
    determinant = dd * ee - de * de
    require(dd >= 0 and ee >= 0 and determinant >= 0, 'Invalid raw-moment covariance')
    d = [(v - x1) ** 2 - h1 + (k - 1) * (u - y1) for v, u in zip(x, y)]
    e = [k * (u - y1) for u in y]

    def concentration(values):
        squares = [v * v for v in values]
        total = sum(squares)
        return max(squares) / total if total > 0 else None

    return dict(pairs=count, paths=paths, reference_paths=N0,
        observed_variance=observed, evaluation_mean_variance=y1,
        A=observed-y1, B=paths*y1, denominator=denominator, numerator=numerator,
        covariance=dict(dd=dd, de=de, ee=ee),
        covariance_rank=2 if determinant > 0 else 1 if dd+ee > 0 else 0,
        point_share=point, point_in_parameter_domain=point is not None and 0 <= point <= 1,
        largest_squared_influence_share=dict(denominator=concentration(d),
            numerator=concentration(e),
            ratio_contrast=concentration(v-point*u for u,v in zip(d,e)) if point is not None else None))


def coefficients(moment):
    d, e = moment['denominator'], moment['numerator']
    covariance = moment['covariance']
    return dict(a=d*d-Z*Z*covariance['dd'],
                b=2*(Z*Z*covariance['de']-d*e),
                c=e*e-Z*Z*covariance['ee'])


def rational(value):
    return dict(kind='rational', value=Fraction(value))


def root(centre, radius_squared, sign):
    n, d = math.isqrt(radius_squared.numerator), math.isqrt(radius_squared.denominator)
    if n*n == radius_squared.numerator and d*d == radius_squared.denominator:
        return rational(centre + sign * Fraction(n, d))
    return dict(kind='root', centre=centre, radius_squared=radius_squared, sign=sign)


def compare(endpoint, value):
    """Order an algebraic endpoint and a rational by exact square comparison."""
    if endpoint['kind'] == 'rational':
        difference = endpoint['value'] - value
    else:
        distance = value - endpoint['centre']
        radius = endpoint['radius_squared']
        if endpoint['sign'] > 0:
            if distance <= 0:
                return 1
            difference = radius - distance*distance
        else:
            if distance >= 0:
                return -1
            difference = distance*distance - radius
    return (difference > 0) - (difference < 0)


def topology(a, b, c):
    """Solve the polynomial sign inequality and then restrict its components."""
    discriminant = b*b - 4*a*c
    entire = [(None, None)]
    if a:
        if discriminant > 0:
            r = discriminant / (4*a*a)
            left, right = root(-b/(2*a), r, -1), root(-b/(2*a), r, 1)
            kind, intervals = ('bounded_interval', [(left, right)]) if a > 0 else ('two_rays', [(None, left), (right, None)])
        elif discriminant == 0 and a > 0:
            point = rational(-b/(2*a))
            kind, intervals = 'singleton', [(point, point)]
        else:
            kind, intervals = ('all_real', entire) if a < 0 else ('empty', [])
    elif b:
        point = rational(-c/b)
        kind, intervals = 'half_line', [(None, point)] if b > 0 else [(point, None)]
    else:
        kind, intervals = ('all_real', entire) if c <= 0 else ('empty', [])
    result = []
    for left, right in intervals:
        if (left is not None and compare(left, 1) > 0) or (right is not None and compare(right, 0) < 0):
            continue
        if left is None or compare(left, 0) < 0:
            left = rational(0)
        if right is None or compare(right, 1) > 0:
            right = rational(1)
        result.append(dict(lower=left, upper=right))
    if not result:
        physical = 'empty'
    elif len(result) > 1:
        physical = 'two_intervals'
    elif result == [dict(lower=rational(0), upper=rational(1))]:
        physical = 'full_domain'
    elif result[0]['lower'] == result[0]['upper']:
        physical = 'singleton'
    else:
        physical = 'interval'
    return dict(unrestricted_kind=kind, physical_kind=physical,
                discriminant=discriminant, components=result)


def open_cell_has_point(a, b, c, lower, upper):
    """Minimise a quadratic on an open cell, retaining unattained boundaries."""
    require(lower < upper, 'Nonpositive open cell length')
    values = lambda w: a*w*w+b*w+c
    # If a boundary value is strictly negative, continuity supplies a point.
    if min(values(lower), values(upper)) < 0:
        return True
    if a > 0 and lower < -b/(2*a) < upper:
        return c - b*b/(4*a) <= 0
    # This remaining equality case is attained throughout a constant cell.
    return a == b == 0 and c <= 0


def comparison(a, b, c, candidates):
    """Partition all affine crossings, keeping ties separate from open cells."""
    rows = sorted(candidates, key=lambda r: (r['evaluation_paths'], r['repetitions']))
    require(rows and len({r['evaluation_paths'] for r in rows}) == len(rows), 'Invalid allocation grid')
    for row in rows:
        require(set(row) == {'evaluation_paths', 'repetitions'} and
                all(type(v) is int and v > 0 for v in row.values()), 'Invalid allocation candidate')
    lines = [(Fraction(1, r['repetitions']),
              Fraction(N0-r['evaluation_paths'], r['evaluation_paths']*r['repetitions'])) for r in rows]
    nodes = {Fraction(0), Fraction(1)}
    for (x, u), (y, v) in combinations(lines, 2):
        if u != v and 0 <= (y-x)/(u-v) <= 1:
            nodes.add((y-x)/(u-v))
    nodes = sorted(nodes)
    weak, selected = set(), set()

    def add(weight):
        values = [intercept+slope*weight for intercept, slope in lines]
        best = min(values)
        sizes = [row['evaluation_paths'] for row, value in zip(rows, values) if value == best]
        weak.update(sizes)
        selected.add(sizes[0])

    for w in nodes:
        if a*w*w+b*w+c <= 0:
            add(w)
    for lower, upper in zip(nodes, nodes[1:]):
        if open_cell_has_point(a,b,c,lower,upper):
            add((lower+upper)/2)
    return dict(weakly_compatible_sizes=sorted(weak), selected_sizes=sorted(selected),
        uniformly_selected_size=next(iter(selected)) if len(selected) == 1 else None), len(nodes), len(nodes)-1


def contains(a, b, c, point):
    return point is not None and 0 <= point <= 1 and a*point*point+b*point+c <= 0


def higher_precision(endpoint_input):
    """Enclose displays in a separate interpreter without importing the runner."""
    script = r'''
from fractions import Fraction
import json, platform, sys
import flint
from flint import arb, fmpq, ctx
payload = json.load(sys.stdin)
def ball(value):
    x = Fraction(value)
    return arb(fmpq(x.numerator,x.denominator))
def bounds(value):
    if not value.is_finite():
        raise ValueError('Nonfinite high-precision enclosure')
    return dict(lower=str(Fraction(str(value.lower().fmpq()))),
                upper=str(Fraction(str(value.upper().fmpq()))))
def endpoint(spec):
    if spec['kind'] == 'rational':
        return bounds(ball(spec['value']))
    return bounds(ball(spec['centre']) + spec['sign']*ball(spec['radius_squared']).sqrt())
with ctx.workprec(192):
    # Phi(-z) = erfc(z / sqrt(2)) / 2; this order differs from production.
    argument = (ball(payload['critical_value'])**2/2).sqrt()
    tail = bounds(payload['family_size']*argument.erfc())
    sets = [dict(case=row['case'],source=row['source'],components=[
        {side:endpoint(part[side]) for side in ('lower','upper')}
        for part in row['components']]) for row in payload['sets']]
print(json.dumps(dict(precision_bits=192,normal_family_tail=tail,sets=sets,
    python=platform.python_version(),python_flint=flint.__version__,
    flint_version=flint.__FLINT_VERSION__)))
'''
    result = subprocess.run([str(Path.home()/'.venvs/qf-certification/bin/python'), '-c', script],
                            input=json.dumps(endpoint_input), capture_output=True, text=True, check=True)
    return json.loads(result.stdout, object_pairs_hook=unique)


def audit(inputs, run, output, tamper=True):
    inputs, run, output = map(private, (inputs, run, output))
    require(not output.exists() and not output.is_relative_to(inputs) and
            not output.is_relative_to(run), 'Use a fresh, separate audit directory')
    evidence = {}

    def retain(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence')
        digest = sha(path)
        name = str(path.relative_to(ROOT))
        require(expected is None or expected == digest, 'Evidence checksum differs: '+name)
        require(name not in evidence or evidence[name] == digest, 'Conflicting evidence hash')
        evidence[name] = digest
        return path

    def manifest(folder):
        document = read(retain(folder/'manifest.json'))
        require(document.get('complete') is True and not (folder/'failure.json').exists(), 'Incomplete stage')
        for key, anchor in (('evidence_sha256', ROOT), ('artifact_sha256', folder)):
            mapping = document[key]
            require(isinstance(mapping,dict) and mapping, 'Missing evidence map')
            for name, digest in mapping.items():
                path = (anchor/name).resolve()
                require(path.is_relative_to(anchor.resolve()), 'Evidence escapes root')
                retain(path,digest)
        require(set(document['artifact_sha256']) == {p.name for p in folder.glob('*.json') if p.name != 'manifest.json'},
                'Unbound or missing JSON artifact')
        return document

    def bound(path):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) in evidence, 'Unbound consumed evidence')
        return read(path)

    im, rm = manifest(inputs), manifest(run)
    require(all(rm['evidence_sha256'].get(k) == v for k,v in im['evidence_sha256'].items()),
            'Input closure differs in run')
    for p in (inputs/'manifest.json',inputs/'inputs.json'):
        require(rm['evidence_sha256'].get(str(p.relative_to(ROOT))) == sha(p), 'Run is not bound to inputs')
    previous_audit = bound(PRIVATE/'component_confidence/2026-09-28/audit_v1/checks.json')
    require(previous_audit['complete'] is True, 'Prior component audit is incomplete')
    for name, digest in previous_audit['evidence_sha256'].items():
        retain(ROOT/name,digest)
    literature = bound(PRIVATE/'literature/component_ratio_2026-09-28/source_record.json')
    require(isinstance(literature['file_sha256'],dict) and literature['file_sha256'], 'Missing consulted-source file map')
    for name, digest in literature['file_sha256'].items():
        retain(ROOT/name,digest)
    payload = bound(inputs/'inputs.json')
    same({k:v for k,v in payload.items() if k != 'models'}, dict(cases=13,family_size=39,
        reference_paths=N0,critical_value=Z,family_alpha=Fraction(1,20),precision_bits=128,
        retrospective=True,new_holdout=False,rows=5446,unique_stream_ids=16338,
        publication_clearance=False), 'input design')
    require([m['case'] for m in payload['models']] == list(range(13)), 'Incomplete case coverage')
    retained_geometry = bound(OLD/'inputs_v1/inputs.json')
    registration = bound(TRAINING/'main_registration_v1/allocations.json')
    require(registration['main_outcomes_inspected'] is False, 'Original registration is not outcome-blind')
    selected = {r['case']:r['allocation']['selected'] for r in registration['models']}
    forbidden = set(registration['forbidden_streams'])
    sources = ('pilot','reference','selected')
    records, endpoint_sets, concordance = [], [], []
    streams, seen_rows = set(), set()
    row_count = crossing_count = cell_count = decision_count = 0
    for model in payload['models']:
        case, label = model['case'], 'case_'+str(model['case'])
        require(set(model) == {'case','approximation_fingerprint','training_specification_sha256','regimes','samples'},
                'Unexpected input model fields')
        same(model['regimes'],retained_geometry['models'][case]['regimes'],label+'.retained_regimes')
        require([(r['accounting'],r['scope']) for r in model['regimes']] ==
                [(a,b) for a in ('marginal','screening_charged') for b in ('pilot','validation','joint')],
                'Cost-regime coverage differs')
        require([sample['source'] for sample in model['samples']] == list(sources), 'Sample-source coverage differs')
        pilot = bound(TRAINING/f'pilot_v1/blocks/case_{case:03d}.json')
        require(pilot['case'] == case and len(pilot['statistical']) == 32, 'Pilot raw coverage differs')
        expected_signature = None
        case_records = []
        for sample in model['samples']:
            source = sample['source']
            sample_label = label+'.'+source
            require(set(sample) == {'source','pairs','paths','rows'}, 'Sample fields differ')
            count = 32 if source == 'pilot' else 64 if source == 'reference' else selected[case]['repetitions']
            paths = N0 if source != 'selected' else selected[case]['evaluation_paths']
            require(sample['pairs'] == count and sample['paths'] == paths and len(sample['rows']) == count,
                    'Registered sample size differs')
            means, errors = [], []
            for repetition, row in enumerate(sample['rows']):
                require(set(row) == {'raw_file','repetition','mean','standard_error','stream_ids'}, 'Input row fields differ')
                require(row['repetition'] == repetition, 'Input repetition differs')
                raw_path = TRAINING/(f'pilot_v1/blocks/case_{case:03d}.json' if source == 'pilot' else
                                     f'main_run_v1/blocks/case_{case:03d}_{source}_rep_{repetition:04d}.json')
                same(row['raw_file'],str(raw_path.relative_to(ROOT)),sample_label+'.raw_file')
                raw = pilot['statistical'][repetition] if source == 'pilot' else bound(raw_path)
                require(raw['repetition'] == repetition, 'Raw repetition differs')
                if source != 'pilot':
                    require(raw['case'] == case and raw['arm'] == source, 'Raw case or arm differs')
                identity = (row['raw_file'],repetition)
                require(identity not in seen_rows, 'Repeated raw observation')
                seen_rows.add(identity)
                training, ev = raw['training'], raw['evaluation']
                estimate = ev['gain']['estimate']
                require(ev['evaluation_paths'] == ev['stream']['paths'] == estimate['observations'] == paths,
                        'Raw evaluation count differs')
                require(training['training_paths'] == [5000,20000] and len(training['streams']) == 2,
                        'Raw training counts differ')
                signature = [{k:p[k] for k in ('model','option','decisions','requested_degree','training_paths',
                    'minimum_itm_paths','condition_limit')} for p in training['policies']]
                if expected_signature is None:
                    expected_signature = signature
                require(signature == expected_signature and ev['approximation_fingerprint'] == model['approximation_fingerprint'],
                        'Financial, training or approximation specification differs')
                raw_streams = [s['stream_id'] for s in training['streams']] + [ev['stream']['stream_id']]
                same(row['stream_ids'],raw_streams,sample_label+'.stream_ids')
                require(len(set(raw_streams)) == 3 and not streams.intersection(raw_streams), 'Repeated stream identifier')
                require(source == 'pilot' or not forbidden.intersection(raw_streams), 'Main stream occurs in excluded evidence')
                streams.update(raw_streams)
                w, error = scalar(estimate['value']), scalar(estimate['standard_error'])
                same(row['mean'],w,sample_label+'.row_mean')
                same(row['standard_error'],error,sample_label+'.row_standard_error')
                require(q(row['mean']) == w and q(row['standard_error']) == error, 'Noncanonical input summary')
                means.append(w); errors.append(error); row_count += 1
            moments = raw_moments(means,errors,paths)
            coefs = coefficients(moments)
            region = topology(**coefs)
            decisions = []
            for regime in model['regimes']:
                choices, nodes, cells = comparison(**coefs,candidates=regime['candidates'])
                decisions.append(dict(**regime,**choices))
                crossing_count += nodes; cell_count += cells; decision_count += 1
            expected = dict(case=case,source=source,moments=moments,coefficients=coefs,region=region,regimes=decisions)
            same(bound(run/f'case_{case:02d}_{source}.json'),expected,sample_label)
            records.append(expected); case_records.append(expected)
            endpoint_sets.append(dict(case=case,source=source,components=region['components']))
        signature_digest = hashlib.sha256(json.dumps(expected_signature,sort_keys=True).encode()).hexdigest()
        same(model['training_specification_sha256'],signature_digest,label+'.training_signature')
        comparisons = []
        for target in case_records:
            for other in case_records:
                if other['source'] == target['source']:
                    continue
                valid = other['moments']['point_in_parameter_domain']
                comparisons.append(dict(set_source=target['source'],point_source=other['source'],
                    point_is_admissible=valid,contains_point=contains(**target['coefficients'],point=other['moments']['point_share']) if valid else None))
        concordance.append(dict(case=case,comparisons=comparisons))
    require((row_count,len(streams),len(seen_rows),decision_count) == (5446,16338,5446,234), 'Total evidence coverage differs')
    endpoint_input = dict(family_size=39,critical_value=Z,family_alpha=Fraction(1,20),sets=endpoint_sets)
    same(bound(run/'endpoint_input.json'),endpoint_input,'endpoint input')
    same(bound(run/'concordance.json'),dict(scope='Descriptive comparison of retrospective sources; truth unknown',
        coverage_validation=False,cases=concordance),'concordance')
    aggregate = []
    for source in sources:
        rows = [r for r in records if r['source'] == source]
        regimes = [d for r in rows for d in r['regimes']]
        aggregate.append(dict(source=source,sets=len(rows),
            unrestricted_kind_counts=dict(Counter(r['region']['unrestricted_kind'] for r in rows)),
            physical_kind_counts=dict(Counter(r['region']['physical_kind'] for r in rows)),
            negative_A=sum(r['moments']['A'] < 0 for r in rows),
            nonpositive_denominators=sum(r['moments']['denominator'] <= 0 for r in rows),
            covariance_rank_counts=dict(Counter(r['moments']['covariance_rank'] for r in rows)),
            admissible_point_estimates=sum(r['moments']['point_in_parameter_domain'] for r in rows),
            unique_allocation_regimes=sum(r['uniformly_selected_size'] is not None for r in regimes),
            selected_size_count_distribution=dict(Counter(len(r['selected_sizes']) for r in regimes))))
    summary = bound(run/'summary.json')
    same({k:v for k,v in summary.items() if k != 'python'},dict(complete=True,cases=13,component_sets=39,
        fixed_cost_comparisons=234,rows=5446,reference_paths=N0,critical_value=Z,
        nominal_asymptotic_family_confidence_at_least='19/20',source_summaries=aggregate,
        retrospective=True,new_holdout=False,conditional_selected_counts=True,
        finite_sample_coverage_certified=False,simulated_summary_errors_certified=False,
        ideal_sampling_law_certified=False,exact_saved_summary_algebra=True,
        exact_rational_allocation_mapping=True,source_point_inclusion_is_coverage=False,
        new_simulations=0,policy_refits=0,new_market_observations=0,publication_clearance=False),'summary')
    require(isinstance(summary['python'],str) and summary['python'], 'Missing analysis runtime')
    numerical = bound(run/'enclosures.json')
    require(set(numerical) == {'schema_version','family_size','critical_value','family_alpha','normal_family_tail','sets',
        'input_path','input_sha256','source_sha256','precision_bits','python','python_flint','flint_version',
        'statistical_coverage_certified','allocation_decisions_certified','scope','tail_formula'}, 'Unexpected enclosure fields')
    same({k:v for k,v in numerical.items() if k not in ('normal_family_tail','sets','python','python_flint','flint_version')},
        dict(schema_version=1,family_size=39,critical_value=Z,family_alpha=Fraction(1,20),
            input_path=str((run/'endpoint_input.json').relative_to(ROOT)),input_sha256=sha(run/'endpoint_input.json'),
            source_sha256=sha(ROOT/'scripts/certify_component_ratio.py'),precision_bits=128,
            statistical_coverage_certified=False,allocation_decisions_certified=False,
            scope='Numerical endpoint enclosures and normal-tail calibration only',
            tail_formula='family_size * erfc(critical_value / sqrt(2))'),'enclosure design')
    require(all(isinstance(numerical[k],str) and numerical[k] for k in ('python','python_flint','flint_version')),
            'Missing certification runtime')
    higher = higher_precision(encode(endpoint_input))
    def nested(low, high, label):
        require(q(low['lower']) <= q(high['lower']) <= q(high['upper']) <= q(low['upper']),
                'Higher-precision enclosure escapes retained interval: '+label)
    require(set(numerical['normal_family_tail']) == {'lower','upper','display','upper_below_family_alpha'}, 'Tail fields differ')
    require(numerical['normal_family_tail']['upper_below_family_alpha'] is True and
            q(numerical['normal_family_tail']['upper']) < Fraction(1,20), 'Normal family-tail margin fails')
    nested(numerical['normal_family_tail'],higher['normal_family_tail'],'family tail')
    require(len(numerical['sets']) == len(higher['sets']) == 39, 'Enclosure set count differs')
    endpoint_count = 0
    for exact, low, high in zip(endpoint_sets,numerical['sets'],higher['sets']):
        require(set(low) == {'case','source','components'}, 'Enclosure set fields differ')
        require((exact['case'],exact['source']) == (low['case'],low['source']) == (high['case'],high['source']),
                'Enclosure identity differs')
        require(len(low['components']) == len(high['components']) == len(exact['components']), 'Enclosure component count differs')
        for part, low_part, high_part in zip(exact['components'],low['components'],high['components']):
            require(set(low_part) == {'lower','upper'}, 'Enclosure component fields differ')
            for side in ('lower','upper'):
                require(set(low_part[side]) == {'spec','lower','upper','display'}, 'Enclosure endpoint fields differ')
                same(low_part[side]['spec'],part[side],'exact endpoint')
                nested(low_part[side],high_part[side],'endpoint')
                endpoint_count += 1
    retain(Path(__file__))
    retain(ROOT/'tests/test_component_ratio_audit.py')
    controls = tamper_test(inputs,run) if tamper else []
    for name,digest in evidence.items():
        require(sha(ROOT/name) == digest,'Evidence changed during audit: '+name)
    result = dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),cases=13,component_sets=39,
        retained_rows=row_count,unique_stream_identifiers=len(streams),decision_comparisons=decision_count,
        crossing_and_endpoint_nodes=crossing_count,open_cells=cell_count,
        independent_raw_polynomial_moments=True,independent_crossing_partition=True,
        exact_topology_and_endpoint_ownership=True,higher_precision_bits=192,
        retained_precision_bits=128,higher_precision_contained=True,endpoint_enclosures=endpoint_count,
        higher_precision_result=higher,shared_arithmetic_library=True,
        independent_arithmetic_library=False,retrospective=True,new_holdout=False,
        finite_sample_coverage_certified=False,simulated_summary_errors_certified=False,
        ideal_sampling_law_certified=False,source_concordance_is_coverage=False,
        new_simulations=0,policy_refits=0,publication_clearance=False,tamper_checks=controls,
        python=platform.python_version(),evidence_sha256=evidence)
    output.mkdir(parents=True)
    save(output/'checks.json',result)
    return result


def tamper_test(inputs, run):
    """Reject stale and rehashed altered findings before audit output creation."""
    results = []
    with TemporaryDirectory(prefix='component_ratio_audit_control_',dir=BASE) as name:
        base = Path(name)
        for mode in ('checksum','rehashed_moment','rehashed_allocation'):
            copied = base/mode
            shutil.copytree(run,copied)
            altered = copied/'case_00_pilot.json'
            data = read(altered)
            if mode == 'rehashed_allocation':
                data['regimes'][0]['selected_sizes'] = [] if data['regimes'][0]['selected_sizes'] else [999]
            else:
                data['moments']['observed_variance'] = str(q(data['moments']['observed_variance'])+1)
            save(altered,data)
            if mode != 'checksum':
                manifest = read(copied/'manifest.json')
                manifest['artifact_sha256'][altered.name] = sha(altered)
                save(copied/'manifest.json',manifest)
            target = base/(mode+'_audit')
            try:
                audit(inputs,copied,target,tamper=False)
            except ValueError as error:
                reason = str(error)
                expected = 'Evidence checksum differs' if mode == 'checksum' else (
                    'Independent value differs: case_0.pilot.moments.observed_variance' if mode == 'rehashed_moment'
                    else 'Record length differs: case_0.pilot.regimes[0].selected_sizes')
                require(expected in reason,'Negative control failed for an unexpected reason: '+reason)
                require(not target.exists(),'Rejected evidence created audit output')
                results.append(dict(control=mode,rejected=True,output_created=False,reason=reason))
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
    print(json.dumps({k:result[k] for k in ('complete','component_sets','retained_rows','decision_comparisons',
                                         'crossing_and_endpoint_nodes','open_cells','endpoint_enclosures')},indent=2))


if __name__ == '__main__':
    main()
