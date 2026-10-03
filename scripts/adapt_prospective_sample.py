"""Verify the complete prospective sample and retain every calibration case."""
import argparse
from datetime import date, datetime, timezone
from pathlib import Path

from acquire_prospective_sample import registration
from acquire_short_ois_conventions import arguments as schedule_arguments
from review_lseg_pilot import metadata_arguments
from run_conditional_calibration import ROOT, private, read, save, sha
from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.prospective_sample import instant, require_complete_sample, request_plan
from qf_research.market_data.prospective_inputs import contract_grid, review_day, dated_funding, calibrate_call
from qf_research.market_data.short_ois import inspect_contracts


class Evidence:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = private(Path(path))
        actual = sha(path)
        if expected is not None and actual != expected:
            raise RuntimeError('Evidence checksum differs: ' + str(path.relative_to(ROOT)))
        self.hashes[str(path.relative_to(ROOT))] = actual
        return read(path)

    def response(self, folder, record):
        path = (folder/record['file']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise RuntimeError('Response path leaves its acquisition directory')
        return self.read(path, record['sha256'])


def expected_requests(config, slot):
    """Mirror the frozen collector's reference and historical request contract."""
    result = [dict(name='contract_reference', method='discovery.search', arguments=dict(
        view='EQUITY_DERIVATIVE_QUOTES',
        filter=' or '.join("RIC eq '" + r.replace("'", "''") + "'" for r in config['option_rics']),
        select=','.join(config['contract_fields']), top=config['reference_limit']))]
    rics = config['option_rics'] + [config['underlying']] + config['term_rics'] + config['front_rics']
    result.extend(dict(name=f'metadata_{i:02d}', method='metadata', arguments=metadata_arguments(ric))
                  for i, ric in enumerate(rics))
    for label in ('term', 'front'):
        result.append(dict(name=label+'_reference', method='desktop_get_data', arguments=dict(
            universe=config[label+'_rics'], fields=config[label+'_reference_fields'])))
    return result + request_plan(config, slot, config['option_rics'])


def verified_day(folder, manifest, config, slot, registration_hash, evidence):
    """Reject altered sampling; allow documented missing transport responses."""
    if (manifest['registration_sha256'] != registration_hash or manifest['date'] != slot['date'] or
            manifest['role'] != 'prospective_observation' or
            manifest['source_sha256'] != sha(ROOT/'scripts/acquire_prospective_sample.py')):
        raise ValueError('Acquisition belongs to another protocol, date or collector')
    if evidence.read(folder/'configuration.json', manifest['configuration_sha256']) != config:
        raise ValueError('Acquisition configuration differs from registration')
    expected = expected_requests(config, slot)
    requests = manifest['requests']
    if len(requests) > len(expected):
        raise ValueError('Unexpected additional requests')
    if manifest['status'] != 'session_failed_evidence_retained' and len(requests) != len(expected):
        raise ValueError('Final acquisition lacks its prescribed request records')
    payloads = {}
    for actual, intended in zip(requests, expected):
        if any(actual.get(k) != intended[k] for k in ('name', 'method', 'arguments')):
            raise ValueError('Request order, universe, fields or window differs')
        if instant(actual['requested_at']) < instant(slot['release_at_utc']):
            raise ValueError('Request precedes its prescribed release time')
        if actual['status'] == 'retained':
            if instant(actual['retrieved_at']) < instant(actual['requested_at']):
                raise ValueError('Retrieval precedes request')
            payloads[actual['name']] = evidence.response(folder, actual)
        elif actual['status'] != 'failed':
            raise ValueError('Unfinished request record')
    return payloads


def load_schedules(folder, config, slot, registration_hash, evidence):
    if folder is None:
        raise ValueError('Dated schedule acquisition missing')
    folder = private(Path(folder))
    record = evidence.read(folder/'manifest.json')
    if (record.get('date') != slot['date'] or record.get('registration_sha256') != registration_hash or
            record.get('role') != 'dated_schedule_evidence' or
            record.get('arguments') != schedule_arguments({'funding_rics': config['term_rics']}, slot['date']) or
            record.get('source_sha256') != sha(ROOT/'scripts/acquire_prospective_schedules.py')):
        raise ValueError('Schedule acquisition differs from its dated specification')
    if instant(record['requested_at']) < instant(slot['release_at_utc']):
        raise ValueError('Schedule request precedes its release time')
    if instant(record['completed_at']) < instant(record['requested_at']):
        raise ValueError('Schedule completion precedes request')
    if record['status'] != 'retained':
        raise ValueError('Dated schedule request failed')
    return inspect_contracts(evidence.response(folder, record), date.fromisoformat(slot['date']), config['term_rics'])[0]


def adapt(registered, acquisitions, schedules, output):
    registered, output = private(registered), private(output)
    config, plan, _ = registration(registered)
    evidence = Evidence()
    registration_hash = sha(registered/'manifest.json')
    evidence.read(registered/'manifest.json')
    folders = {day: private(Path(path)) for day, path in acquisitions.items()}
    manifests = {day: evidence.read(path/'manifest.json') for day, path in folders.items()}
    # Clock/coverage gate precedes response reads, valuation and output creation.
    require_complete_sample(plan, manifests, datetime.now(timezone.utc).isoformat())
    if set(schedules) - set(acquisitions):
        raise ValueError('Schedule date lies outside the registered sample')
    if output.exists():
        raise FileExistsError('Use a fresh adapter output directory')
    calendar = EventCalendar.from_config(read(ROOT/config['calendar_path']))
    ref_name = next(n for n in config['evidence_sha256'] if n.endswith('/reference_v1/contract_reference.json'))
    frozen_contracts = evidence.read(ROOT/ref_name, config['evidence_sha256'][ref_name])
    frozen_grid = contract_grid(frozen_contracts, config)
    reference_dir = (ROOT/ref_name).parent
    reference_manifest = evidence.read(reference_dir/'manifest.json')
    frozen_metadata = {r['arguments']['query_parameters']['ric']: evidence.response(reference_dir, r)
                       for r in reference_manifest['requests'] if r['method'] == 'metadata'}
    rows, audits = [], []
    for date_index, slot in enumerate(plan):
        payloads = verified_day(folders[slot['date']], manifests[slot['date']], config, slot, registration_hash, evidence)
        issues = []
        # Contradictory reference evidence is held. Absence cannot be filled by an
        # old current snapshot, despite the prior identity corroboration.
        try:
            current = payloads.get('contract_reference')
            if contract_grid(current, config) != frozen_grid:
                raise ValueError('Current reference mapping differs')
            fields = ('LotSize', 'StrikeMultiplier', 'LastTradeDate', 'ExerciseEndDate',
                      'ESMAOptionExerciseStyle', 'ESMADeliveryType', 'ESMAPriceMultiplier')
            old = {r['RIC']: r for r in frozen_contracts}
            if any(r.get(k) != old[r['RIC']].get(k) for r in current for k in fields):
                raise ValueError('Contract terms changed or newly require review')
        except (ValueError, KeyError, TypeError) as exc:
            issues.append(str(exc))
        metadata = {}
        rics = config['option_rics'] + [config['underlying']] + config['term_rics'] + config['front_rics']
        for i, ric in enumerate(rics):
            supplied = payloads.get(f'metadata_{i:02d}')
            if supplied is None or any(supplied.get(k) != frozen_metadata[ric].get(k)
                                       for k in ('MarketRulesData', 'Qualifiers')):
                issues.append('Missing or changed metadata: ' + ric)
            else:
                metadata[ric] = supplied
        audit = review_day(config, slot, payloads, metadata, frozen_contracts)
        funding = None
        try:
            dated = load_schedules(schedules.get(slot['date']), config, slot, registration_hash, evidence)
            funding, details = dated_funding(config, slot, audit, dated, calendar)
            audit['funding_construction'] = details
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            audit['funding_failure'] = str(exc)
        audit['reference_issues'] = issues
        for strike_index, group in enumerate(audit['groups']):
            base = dict(date=slot['date'], cutoff=slot['cutoff_utc'], strike=group['strike'],
                        date_index=date_index, strike_index=strike_index)
            if issues:
                row = dict(base, status='reference_evidence_hold', reasons=issues)
            elif not group['primary_quotes_pass']:
                row = dict(base, status='primary_quote_screen_failed')
            elif funding is None:
                row = dict(base, status='funding_construction_failed', reason=audit['funding_failure'])
            else:
                try:
                    row = dict(calibrate_call(config, calendar, group, funding), **{
                        'date_index': date_index, 'strike_index': strike_index})
                except (ValueError, ArithmeticError) as exc:
                    row = dict(base, status='calibration_failed', reason=str(exc))
            rows.append(row)
        audits.append(audit)
    output.mkdir(parents=True)
    save(output/'configuration.json', config)
    save(output/'audit.json', audits)
    save(output/'cases.json', rows)
    sources = [Path(__file__), ROOT/'scripts/acquire_prospective_schedules.py',
        ROOT/'src/qf_research/market_data/prospective_inputs.py', ROOT/'scripts/acquire_short_ois_conventions.py',
        ROOT/'scripts/review_lseg_pilot.py']
    save(output/'manifest.json', dict(role='prospective_call_only_inputs',
        registration_sha256=registration_hash, created_at=datetime.now(timezone.utc).isoformat(),
        registration_path=str(registered.relative_to(ROOT)), input_sha256=evidence.hashes,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        artifact_sha256={p.name: sha(p) for p in output.glob('*.json')},
        empirical_calibration_accepted=False, publication_clearance=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--acquisitions', type=Path, required=True, help='Private JSON mapping of dates to acquisition directories')
    p.add_argument('--schedules', type=Path, required=True, help='Private JSON mapping of dates to dated schedule directories')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    adapt(a.registration, read(private(a.acquisitions)), read(private(a.schedules)), a.output)
