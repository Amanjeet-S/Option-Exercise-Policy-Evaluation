"""Independently reconstruct a fixed prospective allocation experiment.

Raw polynomial moments provide a separate route to the estimated variance and
its fourth-moment uncertainty. Exact rational comparisons retain signed
contrasts and do not promote asymptotic intervals to finite-sample guarantees.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'prospective_allocation/2026-09-28'
Z = Fraction(13, 4)
REFERENCE_PATHS = 4096
CASES = [5,6,10]
GRID = [128,256,4096]
COST = PRIVATE/'allocation_cost_sensitivity/2026-09-27'
RATIO = PRIVATE/'component_ratio/2026-09-28'
TRAINING = PRIVATE/'training_allocation/2026-09-27'


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
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=invalid)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    Path(path).write_text(json.dumps(encode(value), indent=2, allow_nan=False) + '\n')


def same(actual, expected, label):
    expected = encode(expected)
    require(type(actual) is type(expected), 'Record type differs: ' + label)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Record fields differ: ' + label)
        for name, value in expected.items():
            same(actual[name], value, label + '.' + name)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Record length differs: ' + label)
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, label + '[' + str(index) + ']')
    else:
        require(actual == expected, 'Independent value differs: ' + label)


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Require private storage')
    return path


class Evidence:
    """Verify every consumed object against a retained immutable hash map."""

    def __init__(self):
        self.hashes = {}

    def retain(self,path,expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence')
        digest = sha(path)
        name = str(path.relative_to(ROOT))
        require(expected is None or digest == expected, 'Evidence checksum differs: '+name)
        require(name not in self.hashes or digest == self.hashes[name], 'Conflicting evidence hash')
        self.hashes[name] = digest
        return path

    def mapping(self,mapping,anchor):
        require(isinstance(mapping,dict) and mapping, 'Missing evidence mapping')
        anchor = Path(anchor).resolve()
        for name,digest in mapping.items():
            require(isinstance(name,str) and isinstance(digest,str), 'Malformed evidence mapping')
            path = (anchor/name).resolve()
            require(path.is_relative_to(anchor), 'Evidence escapes declared root')
            self.retain(path,digest)

    def manifest(self,folder):
        folder = Path(folder).resolve()
        document = read(self.retain(folder/'manifest.json'))
        require(not (folder/'failure.json').exists(), 'Execution has a failure record')
        mapped = False
        for key in ('evidence_sha256','input_sha256','source_sha256'):
            if key in document:
                self.mapping(document[key],ROOT)
                mapped = True
        require(mapped, 'No registered provenance mapping')
        self.mapping(document['artifact_sha256'],folder)
        actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
                  if path.is_file() and path.name != 'manifest.json'}
        require(set(document['artifact_sha256']) == actual, 'Unbound or missing stage artifact')
        return document

    def bound(self,path):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT) and str(path.relative_to(ROOT)) in self.hashes,
                'Unbound consumed evidence')
        self.retain(path,self.hashes[str(path.relative_to(ROOT))])
        return read(path)

    def unchanged(self):
        for name,digest in self.hashes.items():
            require(sha(ROOT/name) == digest, 'Evidence changed during audit: '+name)


def raw_moments(values):
    """Reconstruct empirical central moments through raw polynomial sums."""
    values = list(values)
    require(len(values) >= 3 and all(isinstance(x, Fraction) for x in values),
            'Require at least three exact rational observations')
    count = len(values)
    m1, m2, m3, m4 = [sum((v**order for v in values), Fraction()) / count
                      for order in range(1, 5)]
    central2 = m2 - m1*m1
    central4 = m4 - 4*m1*m3 + 6*m1*m1*m2 - 3*m1**4
    require(central2 >= 0 and central4 >= central2**2,
            'Raw polynomial moments are inconsistent')
    mean_variance = central2 / (count-1)
    variance_uncertainty = (central4-central2**2) / (count*count*(count-1))
    central3 = m3 - 3*m1*m2 + 2*m1**3
    covariance = central3/(count*(count-1))
    require(covariance*covariance <= mean_variance*variance_uncertainty,
            'Independent moment covariance is not positive semidefinite')
    second_contributions = [(v-m1)**2 for v in values]
    fourth_contributions = [(d-central2)**2 for d in second_contributions]
    # Recalculate the remaining raw moments, keeping the originally fixed R.
    deleted = []
    for value in values:
        first = (count*m1-value)/(count-1)
        second = (count*m2-value*value)/(count-1)
        deleted.append((count-1)*(second-first*first)/((count-2)*count))
    return dict(pairs=count, mean=m1, row_variance=count*central2/(count-1),
        variance_of_mean=mean_variance, variance_of_variance=variance_uncertainty,
        covariance_mean_variance=covariance,
        mean_variance_degenerate=mean_variance == 0,
        variance_influence_degenerate=variance_uncertainty == 0,
        largest_squared_mean_influence_share=max(second_contributions)/(count*central2)
            if central2 else None,
        largest_squared_variance_influence_share=max(fourth_contributions)/(count*(central4-central2**2))
            if central4 > central2**2 else None,
        leave_one_variance_at_fixed_count=dict(lower=min(deleted),upper=max(deleted),
            lower_row=deleted.index(min(deleted)),upper_row=deleted.index(max(deleted))))


def signed_interval(estimate, variance, critical_value=Z):
    """Classify a symmetric normal interval by an exact square comparison."""
    require(isinstance(estimate, Fraction) and isinstance(variance, Fraction)
            and variance >= 0 and critical_value > 0, 'Invalid contrast moments')
    radius_squared = critical_value*critical_value*variance
    excludes_zero = estimate*estimate > radius_squared
    sign = 'degenerate' if variance == 0 else (
        'positive' if excludes_zero and estimate > 0 else (
        'negative' if excludes_zero and estimate < 0 else 'includes_zero'))
    return dict(difference=estimate,estimated_sampling_variance=variance,
        radius_squared=radius_squared,normal_interval_status=sign,asymptotic_only=True)


def independent_contrast(left, right, component):
    if component == 'mean':
        return signed_interval(left['mean']-right['mean'],
                               left['variance_of_mean']+right['variance_of_mean'])
    require(component == 'mean_variance', 'Unknown comparison component')
    return signed_interval(left['variance_of_mean']-right['variance_of_mean'],
        left['variance_of_variance']+right['variance_of_variance'])


def comparison(left,right):
    denominator = right['variance_of_mean']
    return dict(mean_contrast=independent_contrast(left,right,'mean'),
        variance_contrast=independent_contrast(left,right,'mean_variance'),
        estimated_variance_ratio=left['variance_of_mean']/denominator if denominator > 0 else None,
        ratio_has_confidence_interval=False,critical_value=Z)


def minimax(candidates, intervals):
    """Evaluate every pairwise affine ratio at all retained outer endpoints."""
    candidates = sorted(candidates, key=lambda row: row['evaluation_paths'])
    require(candidates and len({r['evaluation_paths'] for r in candidates}) == len(candidates),
            'Require distinct allocation sizes')
    for row in candidates:
        require(all(type(row[k]) is int and row[k] > 0
                    for k in ('evaluation_paths', 'repetitions')), 'Invalid candidate count')
    require(intervals and all(0 <= lo <= hi <= 1 for lo, hi in intervals),
            'Invalid outer component intervals')
    nodes = sorted({w for interval in intervals for w in interval})

    def value(row, w):
        n, r = row['evaluation_paths'], row['repetitions']
        return ((1-w) + REFERENCE_PATHS*w/n) / r

    records = []
    for row in candidates:
        comparisons = [(value(row,w)/value(other,w), w, other['evaluation_paths'])
                       for w in nodes for other in candidates]
        maximum = max(item[0] for item in comparisons)
        records.append(dict(evaluation_paths=row['evaluation_paths'],
            repetitions=row['repetitions'], maximum_relative_variance=maximum))
    best = min(row['maximum_relative_variance'] for row in records)
    minimisers = [row['evaluation_paths'] for row in records if row['maximum_relative_variance'] == best]
    return dict(selected=next(row for row in records if row['evaluation_paths'] == minimisers[0]),
        minimisers=minimisers,scores=records)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def expected_stream(root_seed,key,paths):
    import numpy as np
    sequence = np.random.SeedSequence(root_seed,spawn_key=tuple(key))
    generator = np.random.Generator(np.random.PCG64(sequence))
    return dict(entropy=sequence.entropy,spawn_key=list(sequence.spawn_key),
        pool_size=sequence.pool_size,bit_generator='PCG64',paths=paths,
        stream_id=digest(generator.bit_generator.state))


def timing(record,training=False):
    costs = {}
    for name in ('cpu_seconds','elapsed_seconds'):
        costs[name] = scalar(record[name])
        require(costs[name] > 0, 'Nonpositive retained timing')
    if training:
        for suffix,total in (('_seconds','elapsed_seconds'),('_cpu_seconds','cpu_seconds')):
            values = [scalar(record['phases'][phase+suffix]) for phase in ('simulation','fitting')]
            require(min(values) >= 0 and sum(values) <= costs[total]+Fraction(1,10**9),
                    'Training phase timing exceeds total')
    return costs


def paired_summary(record,paths):
    """Check the observed pairing identities, allowing prior floating rounding."""
    estimate = record['estimate']
    require(estimate['observations'] == paths and estimate['confidence'] == .95,
            'Paired inference count or confidence differs')
    values = {key:scalar(estimate[key]) for key in ('value','standard_error','lower','upper')}
    require(values['standard_error'] >= 0, 'Negative standard error')
    lv,rv,cov = [scalar(record[key]) for key in ('left_variance','right_variance','covariance')]
    require(lv >= 0 and rv >= 0, 'Negative marginal variance')
    tol = Fraction(1,10**9)*max(Fraction(1),lv,rv)
    require(cov*cov <= lv*rv+tol, 'Invalid retained paired covariance')
    variance = lv+rv-2*cov
    require(variance >= -tol, 'Negative paired variance')
    def close(a,b):
        require(math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=2e-11),
                'Floating paired moment identity differs')
    close(values['value'],scalar(record['right_mean'])-scalar(record['left_mean']))
    close(values['standard_error']**2,max(Fraction(),variance)/paths)
    from scipy.stats import t
    radius = float(t.ppf(.975,paths-1))*float(values['standard_error'])
    close(values['lower'],float(values['value'])-radius)
    close(values['upper'],float(values['value'])+radius)
    return values['value']


def check_pair(block,cfg,base,case,paths,repetition,approximation_fingerprint,
               training_keys,evaluation_key,seen,training_paths=None):
    """Bind one pair to its registered problem, stream roles and inner count."""
    require((block['case'],block['evaluation_paths'],block['repetition']) ==
            (case,paths,repetition), 'Row identity differs')
    counts = cfg['training_paths'] if training_paths is None else training_paths
    training,evaluation = block['training'],block['evaluation']
    require(training['training_paths'] == counts == evaluation['training_paths'],
            'Training path count differs')
    require(evaluation['evaluation_paths'] == paths, 'Evaluation path count differs')
    for name in ('policies','policy_fingerprints','streams'):
        require(len(training[name]) == 2, 'Require exactly two fitted training policies')
    ids = []
    for i,count in enumerate(counts):
        expected = expected_stream(cfg['root_seed'],training_keys[i],count)
        same(training['streams'][i],expected,'training stream')
        policy = training['policies'][i]
        for name in ('model','option','decisions'):
            same(policy[name],base[name],'financial problem '+name)
        for name,value in dict(training_paths=count,requested_degree=cfg['degree'],
                minimum_itm_paths=cfg['minimum_itm_paths'],condition_limit=cfg['condition_limit'],
                training_stream_id=expected['stream_id']).items():
            same(policy[name],value,'training specification '+name)
        require(digest(policy) == training['policy_fingerprints'][i], 'Fitted policy fingerprint differs')
        require(len(policy['rules']) == len(base['decisions']) and policy['rules'][-1]['status'] == 'terminal',
                'Fitted exercise rule coverage differs')
        ids.append(expected['stream_id'])
    expected = expected_stream(cfg['root_seed'],evaluation_key,paths)
    same(evaluation['stream'],expected,'evaluation stream')
    ids.append(expected['stream_id'])
    require(len(set(ids)) == 3 and not seen.intersection(ids), 'Training or evaluation stream reused')
    seen.update(ids)
    same(evaluation['policy_fingerprints'],training['policy_fingerprints'],'evaluated policy fingerprints')
    require(evaluation['approximation_fingerprint'] == approximation_fingerprint,
            'Upper approximation fingerprint differs')
    mean = paired_summary(evaluation['gain'],paths)
    nonzero = evaluation['nonzero_gain_count']
    require(type(nonzero) is int and 0 <= nonzero <= paths, 'Invalid nonzero path count')
    require(nonzero != 0 or (mean == 0 and evaluation['gain']['estimate']['standard_error'] == 0),
            'All-zero sample has a nonzero retained moment')
    gaps = evaluation['minimum_gap_quote']
    require(len(gaps) == 2 and all(scalar(v) >= 0 for v in gaps), 'Negative or malformed minimum gaps')
    path_hash = evaluation['paths_sha256']
    require(isinstance(path_hash,str) and len(path_hash) == 64 and
            all(ch in '0123456789abcdef' for ch in path_hash), 'Invalid retained path digest')
    a,b = timing(training,True),timing(evaluation)
    return mean,{key:a[key]+b[key] for key in a}


def prior_streams(document):
    found,pending = set(),[document]
    while pending:
        item = pending.pop()
        if isinstance(item,dict):
            for name,value in item.items():
                if name in ('stream_id','training_stream_id') and isinstance(value,str):
                    found.add(value)
                elif name in ('prior_streams','forbidden_streams','stream_ids') and isinstance(value,list):
                    found.update(v for v in value if isinstance(v,str))
                else:
                    pending.append(value)
        elif isinstance(item,list):
            pending.extend(item)
    return found


def registered(registration):
    """Check the prospective allocation before inspecting any new outcome."""
    registration = private(registration)
    evidence = Evidence()
    manifest = evidence.manifest(registration)
    require(manifest['complete'] is True and manifest['stage'] == 'prospective_allocation_registration',
            'Incomplete or incorrect registration stage')
    require(manifest['public_registration'] is False and manifest['publication_clearance'] is False,
            'Registration publication status differs')
    cfg = evidence.bound(registration/'configuration.json')
    same(cfg,dict(cases=CASES,evaluation_grid=GRID,training_paths=[5000,20000],degree=4,
        minimum_itm_paths=32,condition_limit=1e8,
        approximation=dict(core_steps=800,wing_steps=40,tolerance=.001),
        root_seed=2026092856101282564096,warmup_pairs=4,cpu_limit=1800,elapsed_limit=5400,
        critical_value=Z,contrast_family_size=12,accounting='marginal',scope='joint',
        reference_paths=4096,budget_ratio_limits=[Fraction(9,10),Fraction(11,10)],
        expected_primary_pairs=3978,prior_outcomes_used_for_design=True,new_outcomes_inspected=False,
        public_registration=False,publication_clearance=False),'configuration')
    order = [dict(position=i,case=CASES[i%3],evaluation_paths=GRID[(i//3+i%3)%3]) for i in range(9)]
    same(evidence.bound(registration/'schedule.json'),order,'execution schedule')
    require({(r['case'],r['evaluation_paths']) for r in order} == {(c,n) for c in CASES for n in GRID},
            'Schedule does not cover each case and allocation exactly once')
    old_tasks = evidence.bound(TRAINING/'main_registration_v1/tasks.json')
    tasks = evidence.bound(registration/'tasks.json')
    same(tasks,[t for t in old_tasks if t['case'] in CASES],'retained financial tasks')
    source_costs = evidence.bound(COST/'inputs_v1/inputs.json')
    old_results = evidence.bound(COST/'run_v1/summary.json')
    enclosures = evidence.bound(RATIO/'run_v1/enclosures.json')
    require(enclosures['family_size'] == 39 and q(enclosures['critical_value']) == Z
            and q(enclosures['normal_family_tail']['upper']) < Fraction(1,20),
            'Retained normal calibration differs')
    ratio_inputs = evidence.bound(RATIO/'inputs_v1/inputs.json')
    allocations = evidence.bound(registration/'allocations.json')
    require([m['case'] for m in allocations['models']] == CASES, 'Registered case coverage differs')
    pair_count,charge = 0,Fraction()
    for model in allocations['models']:
        case = model['case']
        source = next(r for r in source_costs['models'] if r['case'] == case)
        old = next(r for r in old_results['models'] if r['case'] == case)['accounting']['marginal']['choices']['joint']
        ratio = evidence.bound(RATIO/f'run_v1/case_{case:02d}_selected.json')
        old_regime = next(r for r in ratio['regimes'] if (r['scope'],r['accounting']) == ('joint','marginal'))
        enclosure = next(r for r in enclosures['sets'] if (r['case'],r['source']) == (case,'selected'))
        intervals = [(max(Fraction(),q(r['lower']['lower'])),min(Fraction(1),q(r['upper']['upper'])))
                     for r in enclosure['components']]
        budget = q(source['budget'])
        candidates = []
        for n in GRID:
            limits,costs = {},{}
            for name,values in source['scenarios'].items():
                setup,pair = q(values[str(n)]['setup']),q(values[str(n)]['pair'])
                require(setup >= 0 and pair > 0 and budget > setup, 'Invalid retained CPU cost')
                limits[name] = (budget-setup)//pair
            count = min(1024,min(limits.values()))
            require(count >= 64, 'Candidate below the registered minimum count')
            for name,values in source['scenarios'].items():
                costs[name] = q(values[str(n)]['setup'])+count*q(values[str(n)]['pair'])
            retained = next(c for c in old['candidates'] if c['evaluation_paths'] == n)
            require(retained['repetitions'] == count and retained['feasible'] is True,
                    'Retained count is not the maximal feasible integer')
            same(retained['scenario_costs'],costs,'retained planning costs')
            candidate = dict(evaluation_paths=n,repetitions=count,predicted_scenario_costs=costs,
                             planning_charge=max(costs.values()))
            require(candidate['planning_charge'] <= budget, 'Registered plan exceeds its budget')
            candidates.append(candidate)
            pair_count += count
            charge += candidate['planning_charge']
        basic = [{k:r[k] for k in ('evaluation_paths','repetitions')} for r in candidates]
        same(old_regime['candidates'],basic,'ratio-analysis candidate family')
        point = q(ratio['moments']['point_share'])
        require(0 <= point <= 1, 'Nominal component estimate outside the parameter domain')
        values = [(1-point+REFERENCE_PATHS*point/r['evaluation_paths'])/r['repetitions'] for r in basic]
        best = min(values)
        minimisers = [r for r,v in zip(basic,values) if v == best]
        nominal = dict(**minimisers[0],criterion=best,minimisers=minimisers)
        expected = dict(case=case,budget=budget,components=[dict(lower=lo,upper=hi) for lo,hi in intervals],
            source='selected',point_share=point,candidates=candidates,minimax=minimax(basic,intervals),nominal=nominal,
            uniform_size=old_regime['uniformly_selected_size'],
            approximation_fingerprint=next(m for m in ratio_inputs['models'] if m['case'] == case)['approximation_fingerprint'])
        same(model,expected,'registered case '+str(case))
    require(pair_count == allocations['primary_pairs'] == 3978, 'Registered total pair count differs')
    same(allocations['planned_numerical_charge'],charge,'total planning charge')
    require(allocations['new_outcomes_inspected'] is False, 'New outcomes were inspected before registration')
    streams = set()
    for name in manifest['evidence_sha256']:
        if name.endswith('.json'):
            streams.update(prior_streams(evidence.bound(ROOT/name)))
    same(allocations['forbidden_stream_ids'],sorted(streams),'excluded prior streams')
    evidence.unchanged()
    return evidence,manifest,cfg,allocations,tasks,order


def reconstruct_arm(case,candidate,budget,setup,means,costs,zero_pairs):
    count,n = candidate['repetitions'],candidate['evaluation_paths']
    require(len(means) == len(costs) == count, 'Fixed arm row count differs')
    cpu = scalar(setup['cpu_seconds'])+sum((c['cpu_seconds'] for c in costs),Fraction())
    elapsed = scalar(setup['elapsed_seconds'])+sum((c['elapsed_seconds'] for c in costs),Fraction())
    forecast = q(candidate['planning_charge'])
    width = max(1,count//4)
    ratio = cpu/forecast
    return dict(case=case,evaluation_paths=n,repetitions=count,statistics=raw_moments(means),setup=setup,
        exact_numerical_cpu=cpu,exact_numerical_elapsed=elapsed,budget=budget,planning_charge=forecast,
        actual_to_budget=cpu/budget,over_budget=cpu > budget,actual_to_forecast=ratio,
        within_forecast_tolerance=Fraction(9,10) <= ratio <= Fraction(11,10),
        cpu_quartiles=dict(first_mean=sum(c['cpu_seconds'] for c in costs[:width])/width,
            last_mean=sum(c['cpu_seconds'] for c in costs[-width:])/width,rows_per_quartile=width),
        zero_gain_pairs=zero_pairs,complete=True)


def audit(registration,run,output,tamper=True):
    registration,run,output = map(private,(registration,run,output))
    require(not output.exists() and not output.is_relative_to(run) and not output.is_relative_to(registration),
            'Use a fresh separate audit directory')
    evidence,reg_manifest,cfg,allocations,tasks,order = registered(registration)
    manifest = evidence.manifest(run)
    require(manifest['complete'] is True and manifest['publication_clearance'] is False,
            'Require a complete private execution')
    require(manifest['registration_sha256'] == sha(registration/'manifest.json'),
            'Execution registration hash differs')
    for name,digest_value in reg_manifest['evidence_sha256'].items():
        require(manifest['evidence_sha256'].get(name) == digest_value, 'Registration closure differs in execution')
    for name in reg_manifest['artifact_sha256']:
        path = registration/name
        require(manifest['evidence_sha256'].get(str(path.relative_to(ROOT))) == sha(path),
                'Execution is not bound to every registration artifact')
    for path in (Path(__file__),ROOT/'tests/test_prospective_allocation_audit.py'):
        require(reg_manifest['evidence_sha256'].get(str(path.relative_to(ROOT))) == sha(path),
                'Audit source was not frozen before execution')
    launch = evidence.bound(run/'launch.json')
    require(launch['registration'] == str(registration.relative_to(ROOT)) and
            launch['registration_sha256'] == manifest['registration_sha256'], 'Launch registration differs')
    dates = [datetime.fromisoformat(value) for value in (reg_manifest['recorded_at_utc'],
             launch['started_at_utc'],manifest['completed_at_utc'])]
    require(dates[0] <= dates[1] <= dates[2], 'Registration does not precede execution')
    env = launch['environment']
    require(env['python_workers'] == 1 and env['thread_environment'] == {
        name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'Execution thread configuration differs')
    pools = env['reported_native_pools']
    require(pools is None or all(p['num_threads'] == 1 for p in pools), 'A reported pool used multiple threads')
    require(0 < scalar(manifest['process_cpu_seconds']) <= cfg['cpu_limit'] and
            0 < scalar(manifest['elapsed_seconds']) <= cfg['elapsed_limit'], 'Completed run exceeded the resource limit')
    bycase = {r['case']:r for r in allocations['models']}
    bases = {t['case']:t['source_policy']['policy'] for t in tasks}
    excluded = set(allocations['forbidden_stream_ids'])
    seen = set(excluded)
    warm = evidence.bound(run/'warmup.json')
    require(len(warm['rows']) == 4, 'Warm-up count differs')

    def setup_checked(setup,case):
        require(set(setup) == {'settings','approximation_fingerprint','elapsed_seconds','cpu_seconds'},
                'Unexpected setup fields')
        same(setup['settings'],cfg['approximation'],'approximation settings')
        require(setup['approximation_fingerprint'] == bycase[case]['approximation_fingerprint'],
                'Setup fingerprint differs from the retained construction')
        return timing(setup)

    warm_cost = setup_checked(warm['setup'],5)['cpu_seconds']
    for rep,row in enumerate(warm['rows']):
        require(row['phase'] == 0, 'Warm-up seed phase differs')
        _,cost = check_pair(row,cfg,bases[5],5,4096,rep,bycase[5]['approximation_fingerprint'],
            [(0,5,4096,rep,k) for k in (0,1)],(0,5,4096,rep,2),seen)
        require(scalar(row['outer_cpu_seconds']) >= cost['cpu_seconds'] and
                scalar(row['outer_elapsed_seconds']) >= cost['elapsed_seconds'], 'Warm-up outer timer undercounts')
        warm_cost += cost['cpu_seconds']
    expected_rows = {f'case_{m["case"]:02d}_n_{c["evaluation_paths"]:04d}_rep_{rep:04d}.json'
                     for m in bycase.values() for c in m['candidates'] for rep in range(c['repetitions'])}
    require({p.name for p in (run/'rows').glob('*.json')} == expected_rows, 'Missing or additional prospective rows')
    expected_files = {'launch.json','warmup.json','summary.json','comparisons.json'}
    expected_files.update('rows/'+name for name in expected_rows)
    expected_files.update(f'{kind}_{case:02d}_{n}.json' for case in CASES for n in GRID for kind in ('arm','setup'))
    require(set(manifest['artifact_sha256']) == expected_files, 'Run artifact coverage differs')
    arms = []
    for item in order:
        case,n = item['case'],item['evaluation_paths']
        model = bycase[case]
        candidate = next(r for r in model['candidates'] if r['evaluation_paths'] == n)
        setup = evidence.bound(run/f'setup_{case:02d}_{n}.json')
        setup_checked(setup,case)
        means,costs,zero = [],[],0
        for rep in range(candidate['repetitions']):
            row = evidence.bound(run/'rows'/f'case_{case:02d}_n_{n:04d}_rep_{rep:04d}.json')
            require(set(row) == {'case','evaluation_paths','repetition','phase','training','evaluation',
                                'outer_cpu_seconds','outer_elapsed_seconds'} and row['phase'] == 1,
                    'Unexpected primary row fields or seed phase')
            mean,cost = check_pair(row,cfg,bases[case],case,n,rep,model['approximation_fingerprint'],
                [(1,case,n,rep,k) for k in (0,1)],(1,case,n,rep,2),seen)
            require(scalar(row['outer_cpu_seconds']) >= cost['cpu_seconds'] and
                    scalar(row['outer_elapsed_seconds']) >= cost['elapsed_seconds'], 'Primary row outer timer undercounts')
            means.append(mean);costs.append(cost)
            zero += row['evaluation']['nonzero_gain_count'] == 0
        expected = reconstruct_arm(case,candidate,q(model['budget']),setup,means,costs,zero)
        arm = evidence.bound(run/f'arm_{case:02d}_{n}.json')
        extras = {'position','outer_cpu_seconds','outer_elapsed_seconds','telemetry_before','telemetry_after'}
        require(set(arm) == set(expected)|extras, 'Unexpected arm fields')
        same({k:v for k,v in arm.items() if k not in extras},expected,'prospective arm')
        require(arm['position'] == item['position'] and
                scalar(arm['outer_cpu_seconds']) >= expected['exact_numerical_cpu'] and
                scalar(arm['outer_elapsed_seconds']) >= expected['exact_numerical_elapsed'], 'Arm schedule or outer timer differs')
        arms.append(expected)
    comparisons = []
    for case in CASES:
        reference = next(a for a in arms if (a['case'],a['evaluation_paths']) == (case,4096))
        for n in (128,256):
            target = next(a for a in arms if (a['case'],a['evaluation_paths']) == (case,n))
            contrast = comparison(target['statistics'],reference['statistics'])
            ratio = contrast['estimated_variance_ratio']
            cost_ratio = target['exact_numerical_cpu']/reference['exact_numerical_cpu']
            comparisons.append(dict(case=case,evaluation_paths=n,reference_paths=4096,**contrast,
                actual_cpu_ratio=cost_ratio,estimated_variance_cpu_ratio=ratio*cost_ratio if ratio is not None else None))
    same(evidence.bound(run/'comparisons.json'),dict(comparisons=comparisons,family_size=12,critical_value=Z,
        finite_sample_coverage_certified=False,measured_mean_squared_error=False,
        numerical_summary_errors_certified=False),'prospective contrasts')
    cpu = sum(a['exact_numerical_cpu'] for a in arms)
    summary = dict(all_registered_counts_observed=True,completion_status_source='manifest.json',
        cases=CASES,arms=9,primary_pairs=3978,warmup_pairs=4,
        numerical_cpu=cpu,warmup_numerical_cpu=warm_cost,total_numerical_cpu=cpu+warm_cost,
        planning_charge=q(allocations['planned_numerical_charge']),
        over_budget_arms=sum(a['over_budget'] for a in arms),
        within_forecast_tolerance=sum(a['within_forecast_tolerance'] for a in arms),
        mean_contrast_status_counts=dict(Counter(r['mean_contrast']['normal_interval_status'] for r in comparisons)),
        variance_contrast_status_counts=dict(Counter(r['variance_contrast']['normal_interval_status'] for r in comparisons)),
        prospective_conditional_on_prior=True,original_financial_models_unchanged=True,
        selected_cases_only=True,new_market_observations=0,publication_clearance=False)
    same(evidence.bound(run/'summary.json'),summary,'prospective summary')
    require(len(seen)-len(excluded) == 3*(3978+4), 'New independent stream count differs')
    require(cpu+warm_cost <= scalar(manifest['process_cpu_seconds']), 'Numerical CPU exceeds complete process CPU')
    controls = tamper_check(registration,run) if tamper else []
    evidence.unchanged()
    result = dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_precedes_execution=True,cases=CASES,arms=9,primary_pairs=3978,warmup_pairs=4,
        new_stream_identifiers=len(seen)-len(excluded),excluded_prior_stream_identifiers=len(excluded),
        maximal_jointly_feasible_counts_verified=True,exact_minimax_comparisons_verified=True,
        independent_raw_polynomial_moments=True,exact_saved_summary_algebra=True,
        exact_retained_timing_sums=True,individual_clock_errors_certified=False,
        contrasts=12,critical_value=Z,normal_family_size_bounded_by_retained_calibration=39,
        finite_sample_coverage_certified=False,numerical_summary_errors_certified=False,
        independent_numerical_replay_performed=False,new_market_observations=0,
        publication_clearance=False,tamper_controls=controls,python=platform.python_version(),
        evidence_sha256=evidence.hashes)
    output.mkdir(parents=True)
    save(output/'checks.json',result)
    return result


def tamper_check(registration,run):
    with TemporaryDirectory(prefix='prospective-allocation-audit-',dir=PRIVATE) as directory:
        temporary = Path(directory)
        copied = temporary/'registration'
        shutil.copytree(registration,copied)
        target = copied/'allocations.json'
        target.write_text(target.read_text()+' ')
        output = temporary/'rejected_audit'
        rejected = False
        try:
            audit(copied,run,output,tamper=False)
        except ValueError as exc:
            require('checksum differs' in str(exc), 'Tamper rejected for an unintended reason')
            rejected = True
        require(rejected and not output.exists(), 'Stale evidence was accepted or created output')
        return [dict(alteration='Registered allocations changed without matching their frozen digest',
                     rejected=True,output_created=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,default=BASE/'registration_v1')
    parser.add_argument('--run',type=Path,default=BASE/'run_v1')
    parser.add_argument('--output',type=Path,default=BASE/'audit_v1')
    parser.add_argument('--registration-only',action='store_true')
    args = parser.parse_args()
    if args.registration_only:
        evidence,_,_,allocations,_,_ = registered(args.registration)
        print(json.dumps(dict(complete=True,registered_pairs=allocations['primary_pairs'],
                              verified_evidence_files=len(evidence.hashes))))
    else:
        result = audit(args.registration,args.run,args.output)
        print(json.dumps({k:v for k,v in result.items() if k != 'evidence_sha256'}))


if __name__ == '__main__':
    main()
