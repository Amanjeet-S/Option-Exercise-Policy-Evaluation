"""Stipulated audit fixtures, containing no market observations or provider data."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('admission_summary_tests', ROOT / 'scripts/summarise_market_admission.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def quote(identity, cutoff, age):
    instant = M.timestamp(cutoff) - age * 10**9
    return dict(ric=identity, cutoff_utc=cutoff, price_screen_pass=True, quote_state_resolved=True,
        issues=[], latest_candidates=[dict(BID=11, ASK=12, BIDSIZE=1, ASKSIZE=1,
            SOURCE_DATETIME='stipulated', QUALIFIERS='stipulated')],
        latest_event_time_ns=instant, source_time_ns_range=[instant, instant],
        event_age_seconds=age, source_age_seconds_max=age)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(M, 'ROOT', root)
    folder = root / 'data/private/stipulated_inputs'
    registration = root / 'data/private/stipulated_registration'
    dates = ['2030-01-02', '2030-01-03']
    config = dict(primary_quote_age_seconds=1, quote_sensitivity_ages_seconds=[5, 15, 60],
        primary_funding_age_seconds=300, funding_sensitivity_ages_seconds=[60, 1800],
        observation_dates=dates, strikes=[100, 200], underlying='PRIVATE_UNDERLYING_SENTINEL',
        option_rics=['PRIVATE_CALL_A', 'PRIVATE_PUT_A', 'PRIVATE_CALL_B', 'PRIVATE_PUT_B'],
        term_rics=['PRIVATE_TERM_SENTINEL'], front_rics=['PRIVATE_FRONT_SENTINEL'])
    audits, cases, plan = [], [], []
    for day_index, day in enumerate(dates):
        cutoff = day + 'T12:00:00+00:00'
        plan.append(dict(date=day, cutoff_utc=cutoff))
        age, funding_age = ((2, 100), (8, 1000))[day_index]
        funds = {str(a): funding_age <= a for a in M.FUNDING_AGES}
        audit = dict(groups=[], term={}, front={}, reference_issues=[], source_coverage={})
        if day_index == 0: audit['funding_construction'] = {'stipulated_success': True}
        else: audit['funding_failure'] = 'PRIVATE_FAILURE_TEXT'
        for kind in ('term', 'front'):
            identity = config[kind + '_rics'][0]
            row = dict(ric=identity, cutoff=cutoff, issues=[], event_age_seconds=funding_age,
                       mechanical_age_checks=funds)
            if kind == 'term': row['source_age_seconds'] = funding_age
            audit[kind][identity] = row
            audit['source_coverage'][identity] = {'stipulated': True}
        for strike_index, strike in enumerate(config['strikes']):
            identities = config['option_rics'][2 * strike_index:2 * strike_index + 2]
            quotes = {side: quote(identity, cutoff, age) for side, identity in
                      zip(M.SIDES, [*identities, config['underlying']])}
            for identity in [*identities, config['underlying']]: audit['source_coverage'][identity] = {}
            checks = {str(a): age <= a for a in M.QUOTE_AGES}
            pairings = [dict(option_ric=identity, underlying_ric=config['underlying'], cutoff_utc=cutoff,
                event_skew_seconds=0.0, source_skew_seconds_max=0.0,
                price_and_age_checks=checks, unique_state_and_age_checks=checks) for identity in identities]
            audit['groups'].append(dict(date=day, cutoff=cutoff, strike=strike, quotes=quotes,
                pairings=pairings, quote_age_checks=checks, funding_age_checks=funds,
                primary_quotes_pass=False, primary_funding_pass=funds['300']))
            cases.append(dict(date=day, cutoff=cutoff, strike=strike, date_index=day_index,
                              strike_index=strike_index, status='primary_quote_screen_failed'))
        audits.append(audit)
    save(registration / 'configuration.json', config)
    save(registration / 'sample_plan.json', plan)
    registered_hash = save(registration / 'manifest.json', dict(artifact_sha256={name:
        hashlib.sha256((registration / name).read_bytes()).hexdigest()
        for name in ('configuration.json', 'sample_plan.json')}))
    for name, value in [('configuration', config), ('audit', audits), ('cases', cases)]: save(folder / (name + '.json'), value)
    state = dict(root=root, folder=folder, registration=registration, registered_hash=registered_hash,
                 output=root / 'data/private/stipulated_output')
    seal(state)
    return state


def seal(b):
    save(b['folder'] / 'manifest.json', dict(role='prospective_call_only_inputs',
        registration_path=str(b['registration'].relative_to(b['root'])), registration_sha256=b['registered_hash'],
        input_sha256={str((b['registration'] / 'manifest.json').relative_to(b['root'])): b['registered_hash']},
        artifact_sha256={name: hashlib.sha256((b['folder'] / name).read_bytes()).hexdigest()
                        for name in ('audit.json', 'configuration.json', 'cases.json')}))


def mutate(b, name, operation, reseal=True):
    path = b['folder'] / name
    value = load(path)
    operation(value)
    save(path, value)
    if reseal: seal(b)


def execute(b, output=None):
    return M.run(b['root'], str(b['folder'].relative_to(b['root'])), output or b['output'])


def test_known_counts_use_joint_predicates_and_distinct_snapshots(bundle):
    result = execute(bundle)
    totals = result['summary']['totals']
    assert totals['quote_age_groups'] == {'1': 0, '5': 2, '15': 4, '60': 4}
    assert totals['joint_primary_funding_construction_reference_groups'] == {'1': 0, '5': 2, '15': 2, '60': 2}
    assert totals['joint_primary_funding_construction_reference_dates'] == {'1': 0, '5': 1, '15': 1, '60': 1}
    assert totals['quote_funding_age']['15']['300'] == {'groups': 2, 'dates': 1}
    assert totals['quote_funding_age']['15']['1800'] == {'groups': 4, 'dates': 2}
    assert totals['quotes']['underlying']['price_screen_groups'] == 4
    assert totals['underlying_daily_snapshots']['price_screen'] == 2
    assert result['summary']['empirical_validation_established'] is False
    assert M.verify_summary(result['summary'])


def test_summary_contains_no_observations_identifiers_or_diagnostic_text(bundle):
    result = execute(bundle)
    text = json.dumps(result)
    assert 'PRIVATE_' not in text
    assert '2030-01-02T12' not in text
    assert str(bundle['root']) not in text
    def check(value):
        if isinstance(value, dict):
            assert not set(value) & {'strike', 'strikes', 'ric', 'BID', 'ASK', 'latest_candidates', 'funding_failure'}
            for item in value.values(): check(item)
        elif isinstance(value, list):
            for item in value: check(item)
    check(result)


@pytest.mark.parametrize('name,change', [
    ('audit.json', lambda x: x.pop()),
    ('audit.json', lambda x: x[0]['groups'].pop()),
    ('audit.json', lambda x: x[0]['groups'].__setitem__(1, deepcopy(x[0]['groups'][0]))),
    ('cases.json', lambda x: x.pop()),
    ('cases.json', lambda x: x.__setitem__(1, deepcopy(x[0]))),
    ('cases.json', lambda x: x[0].update(strike_index=1)),
    ('audit.json', lambda x: x[0].pop('source_coverage')),
    ('audit.json', lambda x: x[0]['groups'][0]['pairings'].pop()),
    ('audit.json', lambda x: x[0]['groups'][0]['pairings'][0].update(option_ric='WRONG')),
    ('audit.json', lambda x: x[0]['groups'][0].update(primary_quotes_pass=1)),
    ('audit.json', lambda x: x[0]['groups'][0]['quote_age_checks'].update({'1': True})),
    ('audit.json', lambda x: x[0]['groups'][0]['funding_age_checks'].update({'300': False})),
    ('audit.json', lambda x: x[0]['groups'][0]['quotes']['call'].update(price_screen_pass=False)),
    ('audit.json', lambda x: x[0]['groups'][0]['quotes']['call'].update(quote_state_resolved=False)),
    ('audit.json', lambda x: x[0]['groups'][0]['quotes']['call'].update(event_age_seconds=0)),
    ('audit.json', lambda x: x[0]['groups'][0]['quotes']['underlying'].update(policy_version='different')),
    ('audit.json', lambda x: x[0].pop('funding_construction')),
    ('cases.json', lambda x: x[0].update(status='call_calibrated')),
])
def test_inconsistent_or_missing_evidence_is_rejected(bundle, name, change):
    mutate(bundle, name, change)
    with pytest.raises((ValueError, KeyError, TypeError)): execute(bundle)
    assert not bundle['output'].exists()


def test_changed_primary_threshold_rejected_even_if_both_configurations_resealed(bundle):
    path = bundle['registration'] / 'configuration.json'
    value = load(path); value['primary_quote_age_seconds'] = 5
    save(path, value); save(bundle['folder'] / 'configuration.json', value)
    manifest = load(bundle['registration'] / 'manifest.json')
    manifest['artifact_sha256']['configuration.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle['registered_hash'] = save(bundle['registration'] / 'manifest.json', manifest)
    seal(bundle)
    with pytest.raises(ValueError, match='thresholds'): execute(bundle)


def test_corrupted_source_hash_is_rejected(bundle):
    mutate(bundle, 'cases.json', lambda x: x[0].update(date_index=4), reseal=False)
    with pytest.raises(ValueError, match='hash'): execute(bundle)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_strict_json_rejects_duplicate_and_nonfinite_values(raw):
    with pytest.raises(ValueError): M.strict_json(raw)


@pytest.mark.parametrize('destination', ['outside', 'input', 'registration', 'existing'])
def test_output_confinement_and_nonoverwrite(bundle, destination):
    paths = {'outside': bundle['root'] / 'public', 'input': bundle['folder'] / 'new',
             'registration': bundle['registration'] / 'new', 'existing': bundle['output']}
    if destination == 'existing': bundle['output'].mkdir()
    with pytest.raises(ValueError): execute(bundle, paths[destination])


def test_symlinked_evidence_rejected(bundle):
    path = bundle['folder'] / 'audit.json'; saved = path.with_name('real_audit.json')
    path.rename(saved); path.symlink_to(saved)
    with pytest.raises(ValueError, match='symlink'): execute(bundle)


def test_input_traversal_is_rejected(bundle):
    with pytest.raises(ValueError): M.run(bundle['root'], 'data/private/../stipulated_inputs', bundle['output'])


def test_source_change_during_calculation_is_rejected(bundle, monkeypatch):
    original = M.summarise
    def changed(*args):
        result = original(*args)
        (bundle['folder'] / 'cases.json').write_text('[]')
        return result
    monkeypatch.setattr(M, 'summarise', changed)
    with pytest.raises(ValueError, match='changed'): execute(bundle)
    assert not bundle['output'].exists()


@pytest.mark.parametrize('field', ['joint_primary_funding_construction_reference_groups', 'quote_funding_age_groups'])
def test_public_check_rejects_false_intersections_even_with_recomputed_totals(bundle, field):
    summary = execute(bundle)['summary']
    if field == 'quote_funding_age_groups': summary['per_day'][0][field]['5']['300'] = 0
    else: summary['per_day'][0][field]['5'] = 0
    summary['totals'] = M.aggregate(summary['per_day'])
    with pytest.raises(ValueError): M.verify_summary(summary)


def test_public_check_rejects_empirical_validation_claim(bundle):
    summary = execute(bundle)['summary']; summary['empirical_validation_established'] = True
    with pytest.raises(ValueError): M.verify_summary(summary)


def test_public_check_rejects_duplicated_daily_snapshot_count(bundle):
    summary = execute(bundle)['summary']; summary['per_day'][0]['underlying_daily_snapshots']['present'] = 2
    summary['totals'] = M.aggregate(summary['per_day'])
    with pytest.raises(ValueError): M.verify_summary(summary)


def test_public_check_rejects_quote_passes_with_unresolved_underlying(bundle):
    summary = execute(bundle)['summary']
    day = summary['per_day'][0]
    for key in ('resolved_state_groups', 'price_and_state_groups'): day['quotes']['underlying'][key] = 0
    for key in ('resolved_state', 'price_and_state'): day['underlying_daily_snapshots'][key] = 0
    summary['totals'] = M.aggregate(summary['per_day'])
    with pytest.raises(ValueError): M.verify_summary(summary)


def test_public_check_rejects_impossible_price_state_intersection(bundle):
    summary = execute(bundle)['summary']
    day = summary['per_day'][0]
    day['quotes']['call'].update(present_groups=2, price_screen_groups=2,
                                resolved_state_groups=2, price_and_state_groups=0)
    for age in map(str, M.QUOTE_AGES):
        day['quote_age_groups'][age] = 0
        day['joint_primary_funding_construction_reference_groups'][age] = 0
        day['quote_funding_age_groups'][age] = dict.fromkeys(map(str, M.FUNDING_AGES), 0)
    summary['totals'] = M.aggregate(summary['per_day'])
    with pytest.raises(ValueError): M.verify_summary(summary)
