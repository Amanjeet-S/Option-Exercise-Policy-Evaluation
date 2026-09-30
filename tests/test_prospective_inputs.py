"""Invented market-format records and mathematical models, not empirical data."""
from copy import deepcopy
from dataclasses import replace
from datetime import date
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import pytest

from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.prospective_inputs import (
    contract_grid, review_day, dated_funding, calibrate_call, interval_residual)
from qf_research.market_data.prospective_sample import request_plan
from qf_research.market_data.short_ois import OisSchedule
from qf_research.optimal_stopping.prospective_evaluation import evaluate_case, price_case, stream


def settings():
    meshes = {name: dict(space_steps=n, time_steps=t, domain_multiple=d) for name, n, t, d in (
        ('base', 150, 50, 4), ('space_only', 300, 50, 4), ('time_only', 150, 100, 4),
        ('fine', 300, 100, 4), ('wide', 450, 100, 6))}
    return dict(underlying='SPY', expiry='2026-09-25', strikes=[100],
        option_rics=['SPY_FIXTURE_CALL', 'SPY_FIXTURE_PUT'],
        term_rics=['TERM_FIXTURE'], front_rics=['ON_FIXTURE', 'TN_FIXTURE'],
        quote_fields=['BID', 'ASK', 'BIDSIZE', 'ASKSIZE', 'SOURCE_DATETIME', 'RTL', 'QUALIFIERS'],
        term_fields=['BID', 'ASK', 'SOURCE_DATETIME', 'QUALIFIERS'],
        front_fields=['DISC_FACT', 'ZERO_YLD1', 'QUALIFIERS'],
        quote_lookback_seconds=60, funding_lookback_seconds=600, funding_chunk_seconds=300,
        event_count_cap=100, primary_quote_age_seconds=1, quote_sensitivity_ages_seconds=[5, 15, 60],
        primary_funding_age_seconds=300, funding_sensitivity_ages_seconds=[60, 1800],
        calibration=dict(volatility_bracket=[.000001, 3.], cash=0.),
        computation=dict(call_root_tolerance_quote=1e-8, pde_meshes=meshes,
            pde_numerical_allowance_quote=.001, python_native_tolerance=dict(absolute=2e-10, relative=2e-11),
            training_paths=[64, 128], degrees=[2, 4], training_repetitions=2, evaluation_paths=128,
            root_seed=9017, bit_generator='PCG64', extensions_seconds=[0, 900, 5400],
            value_approximations=dict(coarse=dict(core_steps=32, wing_steps=8, tolerance=.01),
                                      fine=dict(core_steps=64, wing_steps=8, tolerance=.005)),
            regression_safeguards=dict(minimum_itm_paths=8, condition_limit=1e8),
            confidence=.95, family_error_probability=.05))


def fixture():
    cfg = settings()
    slot = dict(date='2026-09-21', cutoff_utc='2026-09-21T19:45:00+00:00',
        quote_start_utc='2026-09-21T19:44:00+00:00', funding_start_utc='2026-09-21T19:35:00+00:00')
    contracts = [dict(RIC=ric, StrikePrice=100, CallPutOption=side, ExpiryDate=cfg['expiry'],
        UnderlyingQuoteRIC=['SPY'], ExerciseStyle='A', ExchangeCode='OPQ', Currency='USD', StrikeMultiplier=1)
        for ric, side in zip(cfg['option_rics'], ('Call', 'Put'))]
    metadata = {ric: dict(MarketRulesData=dict(ruleName='US_OPQ_EQ_115_1_1')) for ric in cfg['option_rics']}
    metadata['SPY'] = dict(MarketRulesData=dict(ruleName='US_PCQ_EQ_112_1_1'), Qualifiers=dict(
        regular=dict(code='R', fid='118', exchcode='60', desc='Regular - Resume'),
        odd=dict(code='BOLO', desc='Best Odd Lot Quote', fact='REU_QUAL')))
    payloads = {}
    for req in request_plan(cfg, slot, cfg['option_rics']):
        fields = ['DATE_TIME', 'EVENT_TYPE'] + req['arguments']['fields']
        blocks = []
        for ric in req['arguments']['universe']:
            # First funding chunk is empty. Latest records lie strictly inside
            # the last chunk, so shared boundaries reconcile without invention.
            latest = not req['name'].endswith('_00')
            if ric == 'SPY': bid, ask = 99.9, 100.1
            elif ric in cfg['option_rics']: bid, ask = .9, 1.1
            else: bid, ask = 3.9, 4.1
            row = dict(DATE_TIME='2026-09-21T19:44:59.5Z', EVENT_TYPE='quote', BID=bid, ASK=ask,
                BIDSIZE=10, ASKSIZE=10, SOURCE_DATETIME='2026-09-21T19:44:59.4Z', RTL=1, QUALIFIERS=None,
                DISC_FACT=.9999 if ric == 'ON_FIXTURE' else .9998, ZERO_YLD1=4.)
            blocks.append(dict(universe={'ric': ric}, headers=[{'name': f} for f in fields],
                adjustments=['unadjusted'], data=[[row[f] for f in fields]] if latest else []))
        payloads[req['name']] = blocks
    calendar = EventCalendar.from_config(dict(timezone='America/New_York', core_open='09:30:00', core_close='16:00:00',
        coverage_start='2026-09-01', coverage_end='2026-12-31', trading_holidays=[], early_closes={},
        settlement_holidays=[], dtc_schedule_through='2026-12-31'))
    schedules = (OisSchedule('TERM_FIXTURE', date(2026, 9, 21), date(2026, 9, 23),
        date(2026, 10, 23), date(2026, 10, 27), date(2026, 10, 23)),)
    return cfg, slot, payloads, metadata, contracts, calendar, schedules


def fitted_fixture():
    cfg, slot, payloads, metadata, contracts, calendar, schedules = fixture()
    audit = review_day(cfg, slot, payloads, metadata, contracts)
    funding, _ = dated_funding(cfg, slot, audit, schedules, calendar)
    row = calibrate_call(cfg, calendar, audit['groups'][0], funding)
    return cfg, dict(row, date_index=0, strike_index=0), audit, funding, calendar


def test_dated_call_fit_and_put_mutation_cannot_change_volatility():
    cfg, case, audit, funding, calendar = fitted_fixture()
    assert case['status'] == 'call_calibrated'
    assert abs(case['root']['residual_quote']) < 1e-8
    group = deepcopy(audit['groups'][0])
    group['quotes']['put']['price_consensus'] = {'BID': 20., 'ASK': 21.}
    changed = calibrate_call(cfg, calendar, group, funding)
    assert changed['model'] == case['model'] and changed['root'] == case['root']
    assert changed['observed']['put'] != case['observed']['put']


@pytest.mark.parametrize('defect', ['missing', 'cap', 'tie', 'invalid_latest', 'source_future'])
def test_quote_defects_remain_failed_groups(defect):
    cfg, slot, payloads, metadata, contracts, _, _ = fixture()
    block = payloads['options'][0]
    if defect == 'missing': payloads.pop('options')
    if defect == 'cap': block['data'] *= cfg['event_count_cap']
    if defect == 'tie':
        block['data'].append(deepcopy(block['data'][0])); block['data'][-1][4] = 20
    if defect == 'invalid_latest':
        block['data'][0][4] = 0
        older = deepcopy(block['data'][0]); older[0] = '2026-09-21T19:44:59Z'; older[4] = 10
        block['data'].append(older)
    if defect == 'source_future': block['data'][0][6] = '2026-09-21T19:45:00.1Z'
    result = review_day(cfg, slot, payloads, metadata, contracts)
    assert len(result['groups']) == 1 and not result['groups'][0]['primary_quotes_pass']


def test_missing_earlier_funding_chunk_holds_latest_quote():
    cfg, slot, payloads, metadata, contracts, calendar, schedules = fixture()
    payloads.pop('term_00')
    audit = review_day(cfg, slot, payloads, metadata, contracts)
    assert audit['groups'][0]['primary_quotes_pass']
    assert not audit['groups'][0]['primary_funding_pass']
    with pytest.raises(ValueError, match='funding screen'):
        dated_funding(cfg, slot, audit, schedules, calendar)


def test_misdated_schedule_is_not_shifted_or_extrapolated():
    cfg, slot, payloads, metadata, contracts, calendar, schedules = fixture()
    audit = review_day(cfg, slot, payloads, metadata, contracts)
    with pytest.raises(ValueError, match='misdated'):
        dated_funding(cfg, slot, audit, [replace(schedules[0], trade_date=date(2026, 9, 18))], calendar)
    with pytest.raises(ValueError, match='accrual start'):
        dated_funding(cfg, slot, audit, [replace(schedules[0], start=date(2026, 9, 24))], calendar)


def test_unbracketed_call_and_failed_dominance_are_retained():
    cfg, case, audit, funding, calendar = fitted_fixture()
    group = deepcopy(audit['groups'][0])
    group['quotes']['call']['price_consensus'] = {'BID': 101., 'ASK': 102.}
    assert calibrate_call(cfg, calendar, group, funding)['status'] == 'call_root_unbracketed'
    curve = replace(funding.spot_curve, factors=(1., 1.1))
    negative_rates = replace(funding, spot_curve=curve, settlement_to_spot=1.01)
    assert calibrate_call(cfg, calendar, audit['groups'][0], negative_rates)['status'] == 'terminal_call_dominance_failed'


def test_contract_side_or_expiry_cannot_silently_repair_identity():
    cfg, _, _, _, contracts, _, _ = fixture()
    contracts[0]['ExpiryDate'] = '2026-09-24'
    with pytest.raises(ValueError, match='terms'):
        contract_grid(contracts, cfg)


def test_put_residual_is_signed_and_allowance_does_not_move_price():
    assert [interval_residual(p, 1., 2.) for p in (.8, 1., 1.5, 2., 2.3)] == pytest.approx([-.2, 0, 0, 0, .3])
    cfg, case, _, _, _ = fitted_fixture()
    prices = price_case(case, cfg['computation'])
    assert len(prices) == 3
    assert abs(prices[0]['call_midpoint_residual']) < 1e-8
    assert prices[2]['call_midpoint_residual'] > prices[0]['call_midpoint_residual']
    assert all(not r['volatility_recalibrated'] and not r['allowance_is_certified_error_bound'] for r in prices)


def test_matched_policy_pipeline_streams_and_zero_extension_identities():
    cfg, case, _, _, _ = fitted_fixture()
    result = evaluate_case(case, cfg['computation'])
    repeat = evaluate_case(case, cfg['computation'])
    assert result == repeat
    assert len(result['training']) == 8
    assert result['evaluation_stream']['spawn_key'] == [0, 0, 1, 0]
    assert {tuple(r['stream']['spawn_key']) for r in result['training']} == {(0, 0, 0, 0), (0, 0, 0, 1)}
    for p in result['endpoints'][0]['policies']:
        for variant in p['comparisons'].values():
            assert variant['terminal_only'] == variant['retrained']
            assert variant['retraining_gain']['value'] == 0
            assert variant['terminal_adjustment_gain']['value'] == 0
    assert len(result['endpoints'][0]['larger_training_degree_comparison']) == 2
    assert len(result['endpoint_families']) == 8
    with pytest.raises(ValueError): stream(cfg['computation'], case, 1, 1)


def test_request_mutation_is_rejected_before_payload_read(monkeypatch):
    scripts = str(Path(__file__).resolve().parents[1]/'scripts')
    sys.path.insert(0, scripts)
    try:
        import adapt_prospective_sample as adapter
    finally:
        sys.path.remove(scripts)
    cfg, slot, *_ = fixture()
    cfg.update(contract_fields=['RIC'], reference_limit=30, term_reference_fields=[], front_reference_fields=[])
    slot['release_at_utc'] = '2026-09-21T21:00:00+00:00'
    class Inputs:
        def read(self, *args): return cfg
        def response(self, *args): raise AssertionError('Must not read altered request payload')
    request = adapter.expected_requests(cfg, slot)[0]
    request['arguments']['top'] = 999
    record = dict(registration_sha256='frozen', date=slot['date'], role='prospective_observation',
        source_sha256=adapter.sha(adapter.ROOT/'scripts/acquire_prospective_sample.py'), configuration_sha256='fixture',
        status='session_failed_evidence_retained', requests=[request])
    with pytest.raises(ValueError, match='Request order'):
        adapter.verified_day(Path('/fixture'), record, cfg, slot, 'frozen', Inputs())


@pytest.mark.parametrize('mode', ['complete', 'session_failure', 'tampered_payload', 'premature'])
def test_adapter_io_retains_failures_and_rejects_tampering(tmp_path, monkeypatch, mode):
    scripts = str(Path(__file__).resolve().parents[1]/'scripts')
    sys.path.insert(0, scripts)
    try:
        import adapt_prospective_sample as adapter
    finally:
        sys.path.remove(scripts)
    cfg, slot, payloads, metadata, contracts, calendar, schedules = fixture()
    cfg.update(contract_fields=['RIC'], reference_limit=30, term_reference_fields=[], front_reference_fields=[],
               calendar_path='calendar.json')
    slot.update(release_at_utc='2026-09-21T21:00:00+00:00')
    root = adapter.ROOT
    for name in ('scripts/acquire_prospective_sample.py', 'scripts/adapt_prospective_sample.py',
                 'scripts/acquire_prospective_schedules.py', 'src/qf_research/market_data/prospective_inputs.py',
                 'scripts/acquire_short_ois_conventions.py', 'scripts/review_lseg_pilot.py'):
        dest = tmp_path/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((root/name).read_bytes())
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    monkeypatch.setattr(adapter, '__file__', str(tmp_path/'scripts/adapt_prospective_sample.py'))
    monkeypatch.setattr(adapter, 'private', lambda p: p.resolve())
    class Clock:
        @staticmethod
        def now(zone):
            return datetime(2026, 9, 20 if mode == 'premature' else 26, tzinfo=timezone.utc)
    monkeypatch.setattr(adapter, 'datetime', Clock)
    registered = tmp_path/'registration'
    registered.mkdir()
    adapter.save(registered/'manifest.json', {'label': 'invented test fixture'})
    reference = tmp_path/'reference_v1'
    reference.mkdir()
    adapter.save(reference/'contract_reference.json', contracts)
    cfg['evidence_sha256'] = {'reference_v1/contract_reference.json': adapter.sha(reference/'contract_reference.json')}
    # The production lookup accepts any private parent ending in reference_v1.
    cfg['evidence_sha256'] = {'fixtures/reference_v1/contract_reference.json': cfg['evidence_sha256'].pop('reference_v1/contract_reference.json')}
    moved = tmp_path/'fixtures'
    moved.mkdir()
    reference.rename(moved/'reference_v1')
    reference = moved/'reference_v1'
    references = []
    all_rics = cfg['option_rics'] + ['SPY'] + cfg['term_rics'] + cfg['front_rics']
    for i, ric in enumerate(all_rics):
        value = metadata.get(ric, {'MarketRulesData': {'ruleName': 'funding fixture'}})
        payloads[f'metadata_{i:02d}'] = value
        path = reference/f'metadata_{i:02d}.json'
        adapter.save(path, value)
        references.append(dict(method='metadata', arguments={'query_parameters': {'ric': ric}},
                               file=path.name, sha256=adapter.sha(path)))
    adapter.save(reference/'manifest.json', {'requests': references})
    adapter.save(tmp_path/'calendar.json', dict(timezone='America/New_York', core_open='09:30:00', core_close='16:00:00',
        coverage_start='2026-09-01', coverage_end='2026-12-31', trading_holidays=[], early_closes={},
        settlement_holidays=[], dtc_schedule_through='2026-12-31'))
    monkeypatch.setattr(adapter, 'registration', lambda p: (cfg, [slot], {}))
    monkeypatch.setattr(adapter, 'load_schedules', lambda *args: schedules)
    acquired = tmp_path/'acquisition'
    acquired.mkdir()
    adapter.save(acquired/'configuration.json', cfg)
    payloads.update(contract_reference=contracts, term_reference={}, front_reference={})
    entries = []
    if mode != 'session_failure':
        for req in adapter.expected_requests(cfg, slot):
            path = acquired/(req['name']+'.json')
            adapter.save(path, payloads[req['name']])
            entries.append(dict(req, file=path.name, sha256=adapter.sha(path), status='retained',
                requested_at='2026-09-21T21:01:00+00:00', retrieved_at='2026-09-21T21:01:01+00:00'))
    manifest = dict(registration_sha256=adapter.sha(registered/'manifest.json'), date=slot['date'],
        role='prospective_observation', source_sha256=adapter.sha(tmp_path/'scripts/acquire_prospective_sample.py'),
        configuration_sha256=adapter.sha(acquired/'configuration.json'), requests=entries,
        status='session_failed_evidence_retained' if mode == 'session_failure' else 'retained_review_pending')
    adapter.save(acquired/'manifest.json', manifest)
    if mode == 'tampered_payload':
        (acquired/'options.json').write_text('[]')
    output = tmp_path/'result'
    args = (registered, {slot['date']: str(acquired)}, {}, output)
    if mode in ('tampered_payload', 'premature'):
        with pytest.raises((ValueError, RuntimeError), match='checksum|complete prescribed'):
            adapter.adapt(*args)
        assert not output.exists()
    else:
        adapter.adapt(*args)
        cases = json.loads((output/'cases.json').read_text())
        assert len(cases) == 1
        assert cases[0]['status'] == ('call_calibrated' if mode == 'complete' else 'reference_evidence_hold')
        assert cases[0]['date_index'] == cases[0]['strike_index'] == 0
