"""Reconstruct descriptive admission counts without repricing or acquisition.

Retained audit predicates are checked against the registered thresholds and
date-contract grid. Counts do not establish documentary clearance, independent
pricing validation or an alternative funding-curve construction.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = 'data/private/prospective_validation/2026-09-26/inputs_v1'
QUOTE_AGES = (1, 5, 15, 60)
FUNDING_AGES = (60, 300, 1800)
SIDES = ('call', 'put', 'underlying')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path).absolute()
    require('..' not in path.parts and path.resolve() == path, 'Noncanonical or symlink path')
    return path


def confined(root, relative):
    relative = str(relative)
    part = PurePosixPath(relative)
    require(not part.is_absolute() and '..' not in part.parts and str(part) == relative
            and '\\' not in relative, 'Unsafe relative input path')
    path = canonical(root / relative)
    require(path.is_relative_to(root / 'data/private'), 'Input must remain in archive data/private')
    return path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def finite(value):
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, list):
            for item in value: finite(item)
        elif isinstance(value, float):
            require(math.isfinite(value), 'Non-finite JSON number')
    result = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON number')))
    finite(result)
    return result


def boolean(value):
    require(type(value) is bool, 'Expected a recorded boolean')
    return value


def boolean_map(value, keys):
    require(isinstance(value, dict) and set(value) == {str(k) for k in keys}, 'Age coverage differs')
    return {key: boolean(item) for key, item in value.items()}


def in_age(value, limit):
    return value is not None and type(value) in (int, float) and 0 <= value <= limit


def strings(value):
    require(isinstance(value, list) and all(isinstance(x, str) for x in value), 'Invalid issue list')
    return value


def timestamp(value):
    require(isinstance(value, str), 'Missing timestamp')
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timestamp must specify timezone')
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def quote(record, cutoff, coverage):
    if record is None:
        return
    require(isinstance(record, dict) and record['ric'] in coverage, 'Quote source coverage absent')
    require(record['cutoff_utc'] == cutoff, 'Quote cutoff differs')
    price, state = boolean(record['price_screen_pass']), boolean(record['quote_state_resolved'])
    issues = strings(record['issues'])
    require(price == (not issues), 'Price-screen flag contradicts recorded issues')
    candidates = record['latest_candidates']
    require(isinstance(candidates, list), 'Candidate coverage absent')
    # Only exact identities are compared; no price differences or valuations.
    keys = ('BID', 'ASK', 'BIDSIZE', 'ASKSIZE', 'SOURCE_DATETIME', 'QUALIFIERS')
    identities = {json.dumps({k: row.get(k) for k in keys}, sort_keys=True) for row in candidates}
    require(state == (len(identities) == 1), 'Resolved-state flag contradicts candidate identities')
    event = record['latest_event_time_ns']
    source = record['source_time_ns_range']
    require(event is None or type(event) is int, 'Invalid event timestamp')
    if event is not None:
        require(record['event_age_seconds'] == (timestamp(cutoff) - event) / 1e9,
                'Recorded event age differs')
    else:
        require(record['event_age_seconds'] is None, 'Missing timestamp with retained age')
    if source is not None:
        require(isinstance(source, list) and len(source) == 2 and all(type(x) is int for x in source)
                and source[0] <= source[1], 'Invalid source timestamp range')
        require(record['source_age_seconds_max'] == (timestamp(cutoff) - source[0]) / 1e9,
                'Recorded source age differs')
    else:
        require(record['source_age_seconds_max'] is None, 'Missing source range with retained age')


def pair(left, right, stored):
    require(stored['option_ric'] == left['ric'] and stored['underlying_ric'] == right['ric']
            and stored['cutoff_utc'] == left['cutoff_utc'], 'Pairing identities differ')
    lt, rt = left['latest_event_time_ns'], right['latest_event_time_ns']
    ls, rs = left['source_time_ns_range'], right['source_time_ns_range']
    event = None if lt is None or rt is None else abs(lt - rt) / 1e9
    source = None if ls is None or rs is None else max(abs(a - b) for a in ls for b in rs) / 1e9
    require(stored['event_skew_seconds'] == event and stored['source_skew_seconds_max'] == source,
            'Pairing skews differ')
    measures = (event, source, left['event_age_seconds'], right['event_age_seconds'],
                left['source_age_seconds_max'], right['source_age_seconds_max'])
    prices = {str(age): left['price_screen_pass'] and right['price_screen_pass']
              and all(in_age(value, age) for value in measures) for age in QUOTE_AGES}
    states = {key: passed and left['quote_state_resolved'] and right['quote_state_resolved']
              for key, passed in prices.items()}
    require(boolean_map(stored['price_and_age_checks'], QUOTE_AGES) == prices
            and boolean_map(stored['unique_state_and_age_checks'], QUOTE_AGES) == states,
            'Pairing age or state predicates differ')
    return states


def funding(audit, config, cutoff):
    passed = {str(age): True for age in FUNDING_AGES}
    for kind in ('term', 'front'):
        records, expected = audit[kind], config[kind + '_rics']
        require(isinstance(records, dict) and not set(records) - set(expected), 'Funding identities differ')
        for identity in expected:
            record = records.get(identity)
            if record is None:
                passed = dict.fromkeys(passed, False)
                continue
            require(record['ric'] == identity and record['cutoff'] == cutoff
                    and identity in audit['source_coverage'], 'Funding source coverage differs')
            issues = strings(record['issues'])
            ages = [record['event_age_seconds']]
            if kind == 'term': ages.append(record['source_age_seconds'])
            checks = {str(age): not issues and all(in_age(value, age) for value in ages)
                      for age in FUNDING_AGES}
            require(boolean_map(record['mechanical_age_checks'], FUNDING_AGES) == checks,
                    'Funding-age predicate differs')
            passed = {key: value and checks[key] for key, value in passed.items()}
    return passed


def summarise(config, plan, audits, cases):
    require(config['primary_quote_age_seconds'] == 1 and type(config['primary_quote_age_seconds']) is int
            and config['quote_sensitivity_ages_seconds'] == [5, 15, 60]
            and all(type(x) is int for x in config['quote_sensitivity_ages_seconds'])
            and config['primary_funding_age_seconds'] == 300 and type(config['primary_funding_age_seconds']) is int
            and config['funding_sensitivity_ages_seconds'] == [60, 1800]
            and all(type(x) is int for x in config['funding_sensitivity_ages_seconds']), 'Registered thresholds changed')
    dates, strikes = config['observation_dates'], config['strikes']
    require(dates and strikes and len(set(dates)) == len(dates) and len(set(strikes)) == len(strikes),
            'Empty or duplicate registered grid')
    require(all(isinstance(day, str) and date.fromisoformat(day).isoformat() == day for day in dates),
            'Invalid registered date')
    require(all(type(x) in (int, float) for x in strikes), 'Invalid strike identity')
    require(isinstance(plan, list) and len(plan) == len(dates) and [p['date'] for p in plan] == dates,
            'Registered date coverage differs')
    require(isinstance(audits, list) and len(audits) == len(dates), 'Audit date coverage differs')
    expected = {(day, strike) for day in dates for strike in strikes}
    require(isinstance(cases, list) and len(cases) == len(expected)
            and {(c['date'], c['strike']) for c in cases} == expected, 'Case date-strike coverage differs')
    case_map = {(c['date'], c['strike']): c for c in cases}
    require(len(config['option_rics']) == 2 * len(strikes)
            and len(set(config['option_rics'])) == len(config['option_rics']), 'Option identity coverage differs')
    for kind in ('term', 'front'):
        require(config[kind + '_rics'] and len(set(config[kind + '_rics'])) == len(config[kind + '_rics']),
                'Missing or duplicate funding identities')
    days, stable_options = [], None
    for index, (slot, audit) in enumerate(zip(plan, audits)):
        day, cutoff = slot['date'], slot['cutoff_utc']
        groups = audit['groups']
        require(isinstance(groups, list) and len(groups) == len(strikes)
                and [(g['date'], g['strike']) for g in groups] == [(day, s) for s in strikes],
                'Audit date-strike coverage or order differs')
        require(isinstance(audit['source_coverage'], dict) and audit['source_coverage'], 'Source coverage absent')
        funds = funding(audit, config, cutoff)
        reference_clear = not strings(audit['reference_issues'])
        built = 'funding_construction' in audit
        require(built != ('funding_failure' in audit), 'Construction outcome missing or contradictory')
        if built:
            require(isinstance(audit['funding_construction'], dict) and audit['funding_construction']
                    and funds['300'], 'Construction contradicts primary funding checks')
        else:
            require(isinstance(audit['funding_failure'], str) and audit['funding_failure'], 'Construction failure absent')
        flags, option_ids, underlying = [], [], []
        price_counts = {side: Counter() for side in SIDES}
        statuses = Counter()
        for strike_index, group in enumerate(groups):
            require(group['cutoff'] == cutoff and set(group['quotes']) == set(SIDES), 'Group quote coverage differs')
            quotes = group['quotes']
            for side in SIDES:
                q = quotes[side]
                quote(q, cutoff, audit['source_coverage'])
                price_counts[side]['present_groups'] += q is not None
                price_counts[side]['price_screen_groups'] += q is not None and q['price_screen_pass']
                price_counts[side]['resolved_state_groups'] += q is not None and q['quote_state_resolved']
                price_counts[side]['price_and_state_groups'] += q is not None and q['price_screen_pass'] and q['quote_state_resolved']
                if q is not None:
                    if side == 'underlying':
                        require(q['ric'] == config['underlying'], 'Underlying identity differs')
                        underlying.append(q)
                    else:
                        require(q['ric'] in config['option_rics'], 'Option identity differs')
                        option_ids.append(q['ric'])
            complete = all(quotes[side] is not None for side in SIDES)
            require(len(group['pairings']) == (2 if complete else 0), 'Pairing coverage differs')
            checks = dict.fromkeys(map(str, QUOTE_AGES), False)
            if complete:
                pairs = [pair(quotes[side], quotes['underlying'], p)
                         for side, p in zip(('call', 'put'), group['pairings'])]
                checks = {str(age): all(p[str(age)] for p in pairs) for age in QUOTE_AGES}
            require(boolean_map(group['quote_age_checks'], QUOTE_AGES) == checks
                    and boolean(group['primary_quotes_pass']) == checks['1'], 'Group quote predicates differ')
            require(boolean_map(group['funding_age_checks'], FUNDING_AGES) == funds
                    and boolean(group['primary_funding_pass']) == funds['300'], 'Group funding predicates differ')
            case = case_map[day, group['strike']]
            require(case['cutoff'] == cutoff and type(case['date_index']) is int and case['date_index'] == index
                    and type(case['strike_index']) is int and case['strike_index'] == strike_index, 'Case pairing differs')
            status = case['status']
            required_status = ('reference_evidence_hold' if not reference_clear else
                               'primary_quote_screen_failed' if not checks['1'] else
                               'funding_construction_failed' if not built else None)
            if required_status:
                require(status == required_status, 'Case status contradicts screening order')
            else:
                require(status in {'calibration_failed', 'terminal_call_dominance_failed', 'call_root_unbracketed',
                                   'call_root_residual_failed', 'call_calibrated'}, 'Unknown downstream case status')
            statuses[status] += 1
            flags.append(checks)
        require(len(option_ids) == len(set(option_ids)), 'Duplicate option quote identity')
        if len(option_ids) == len(config['option_rics']):
            if stable_options is not None: require(option_ids == stable_options, 'Option grid changed between dates')
            stable_options = option_ids
        require(len(underlying) in (0, len(groups)) and
                (not underlying or all(q == underlying[0] for q in underlying)), 'Repeated daily underlying snapshot differs')
        u = underlying[0] if underlying else None
        quote_counts = {str(age): sum(f[str(age)] for f in flags) for age in QUOTE_AGES}
        joint = {age: count if funds['300'] and built and reference_clear else 0 for age, count in quote_counts.items()}
        days.append(dict(date=day, intended_groups=len(groups), quote_age_groups=quote_counts,
            joint_primary_funding_construction_reference_groups=joint,
            quote_funding_age_groups={age: {str(f): count if funds[str(f)] else 0 for f in FUNDING_AGES}
                                      for age, count in quote_counts.items()},
            primary_funding_groups=len(groups) if funds['300'] else 0,
            retained_construction_groups=len(groups) if built else 0,
            recorded_reference_clear_groups=len(groups) if reference_clear else 0,
            recorded_reference_issue_count=len(audit['reference_issues']), statuses=dict(statuses),
            quotes={side: dict(counts) for side, counts in price_counts.items()},
            underlying_daily_snapshots=dict(present=int(u is not None),
                price_screen=int(u is not None and u['price_screen_pass']),
                resolved_state=int(u is not None and u['quote_state_resolved']),
                price_and_state=int(u is not None and u['price_screen_pass'] and u['quote_state_resolved']))))
    return dict(per_day=days, totals=aggregate(days))


def aggregate(days):
    total = dict(dates=len(days), intended_groups=sum(d['intended_groups'] for d in days))
    for key in ('primary_funding', 'retained_construction', 'recorded_reference_clear'):
        total[key + '_groups'] = sum(d[key + '_groups'] for d in days)
        total[key + '_dates'] = sum(d[key + '_groups'] > 0 for d in days)
    total['recorded_reference_issue_count'] = sum(d['recorded_reference_issue_count'] for d in days)
    total['recorded_reference_issue_dates'] = sum(d['recorded_reference_issue_count'] > 0 for d in days)
    for key in ('quote_age', 'joint_primary_funding_construction_reference'):
        total[key + '_groups'] = {str(a): sum(d[key + '_groups'][str(a)] for d in days) for a in QUOTE_AGES}
        total[key + '_dates'] = {str(a): sum(d[key + '_groups'][str(a)] > 0 for d in days) for a in QUOTE_AGES}
    total['quote_funding_age'] = {str(a): {str(f): dict(
        groups=sum(d['quote_funding_age_groups'][str(a)][str(f)] for d in days),
        dates=sum(d['quote_funding_age_groups'][str(a)][str(f)] > 0 for d in days))
        for f in FUNDING_AGES} for a in QUOTE_AGES}
    total['quotes'] = {s: {k: sum(d['quotes'][s][k] for d in days) for k in days[0]['quotes'][s]} for s in SIDES}
    total['underlying_daily_snapshots'] = {k: sum(d['underlying_daily_snapshots'][k] for d in days)
                                         for k in days[0]['underlying_daily_snapshots']}
    total['statuses'] = dict(sum((Counter(d['statuses']) for d in days), Counter()))
    return total


def verify_summary(summary):
    """Check released count arithmetic only, without the withheld source audit."""
    require(summary['primary_quote_age_seconds'] == 1 and summary['quote_ages_seconds'] == list(QUOTE_AGES)
            and summary['primary_funding_age_seconds'] == 300
            and summary['funding_ages_seconds'] == list(FUNDING_AGES), 'Summary thresholds differ')
    require(type(summary['schema_version']) is int and summary['schema_version'] == 1 and summary['complete'] is True
            and summary['empirical_validation_established'] is False
            and summary['original_primary_screen_unchanged'] is True
            and summary['relaxed_thresholds_descriptive_only'] is True, 'Summary scope differs')
    days = summary['per_day']
    require(isinstance(days, list) and days and len({d['date'] for d in days}) == len(days),
            'Summary daily coverage differs')
    def counts(value):
        if isinstance(value, dict):
            for item in value.values(): counts(item)
        else:
            require(type(value) is int and value >= 0, 'Summary contains invalid counts')
    counts(summary['totals'])
    for day in days:
        counts({k: v for k, v in day.items() if k != 'date'})
        size = day['intended_groups']
        require(size > 0 and sum(day['statuses'].values()) == size, 'Daily status counts differ')
        funding_count, built_count, reference_count = (
            day[k + '_groups'] for k in ('primary_funding', 'retained_construction', 'recorded_reference_clear'))
        require(all(value in (0, size) for value in (funding_count, built_count, reference_count))
                and (not built_count or funding_count == size)
                and (reference_count == size) == (day['recorded_reference_issue_count'] == 0),
                'Daily funding, construction or reference flags differ')
        require(set(day['quote_age_groups']) == {str(a) for a in QUOTE_AGES}
                and set(day['joint_primary_funding_construction_reference_groups']) == {str(a) for a in QUOTE_AGES}
                and set(day['quote_funding_age_groups']) == {str(a) for a in QUOTE_AGES}, 'Daily threshold coverage differs')
        previous = 0
        for age in QUOTE_AGES:
            quoted = day['quote_age_groups'][str(age)]
            require(previous <= quoted <= size, 'Quote counts are not nested')
            require(quoted <= min(day['quotes'][side]['price_and_state_groups'] for side in SIDES),
                    'Joint quote counts exceed valid component quotes')
            previous = quoted
            joint = day['joint_primary_funding_construction_reference_groups'][str(age)]
            require(joint == (quoted if all((funding_count, built_count, reference_count)) else 0),
                    'Joint intersection differs from daily predicates')
            matrix = day['quote_funding_age_groups'][str(age)]
            require(set(matrix) == {str(f) for f in FUNDING_AGES}, 'Funding threshold coverage differs')
            require(0 <= matrix['60'] <= matrix['300'] <= matrix['1800'] <= quoted,
                    'Funding intersections are not nested')
            require(matrix['300'] == (quoted if funding_count else 0)
                    and all(value in (0, quoted) for value in matrix.values()), 'Primary funding intersection differs')
        require(set(day['quotes']) == set(SIDES), 'Quote-side coverage differs')
        for item in day['quotes'].values():
            require(item['price_and_state_groups'] <= min(item['price_screen_groups'], item['resolved_state_groups'])
                    and item['price_and_state_groups'] >= max(0, item['price_screen_groups'] +
                                                              item['resolved_state_groups'] - item['present_groups'])
                    and max(item['price_screen_groups'], item['resolved_state_groups']) <= item['present_groups'] <= size,
                    'Quote counts contradict presence or state')
        u = day['underlying_daily_snapshots']
        require(u['price_and_state'] <= min(u['price_screen'], u['resolved_state'])
                and u['price_and_state'] >= max(0, u['price_screen'] + u['resolved_state'] - u['present'])
                and max(u['price_screen'], u['resolved_state']) <= u['present'] <= 1,
                'Underlying snapshot counts differ')
        require(all(day['quotes']['underlying'][group_key] == size * u[snapshot_key]
                    for group_key, snapshot_key in (
                        ('present_groups', 'present'), ('price_screen_groups', 'price_screen'),
                        ('resolved_state_groups', 'resolved_state'), ('price_and_state_groups', 'price_and_state'))),
                'Underlying group counts differ from shared daily snapshots')
    require(summary['totals'] == aggregate(days), 'Aggregate counts differ from daily counts')
    return True


def run(archive_root, input_name, output):
    archive_root, output = canonical(archive_root), canonical(output)
    require(output.is_relative_to(ROOT / 'data/private'), 'Output must remain in live repository data/private')
    require(not output.exists(), 'Use a fresh output directory')
    folder = confined(archive_root, input_name)
    require(not output.is_relative_to(folder), 'Output cannot be inside the input bundle')
    retained = {}
    def read(path, expected=None):
        path = canonical(path)
        raw = path.read_bytes()
        actual = sha(raw)
        require(expected is None or actual == expected, 'Source hash mismatch')
        retained[path] = actual
        return strict_json(raw)
    manifest = read(folder / 'manifest.json')
    require(manifest['role'] == 'prospective_call_only_inputs', 'Unexpected input role')
    require(set(manifest['artifact_sha256']) == {'audit.json', 'cases.json', 'configuration.json'}, 'Input artifact coverage differs')
    records = {name: read(folder / name, value) for name, value in manifest['artifact_sha256'].items()}
    registered = confined(archive_root, manifest['registration_path'])
    require(not output.is_relative_to(registered), 'Output cannot be inside the registration bundle')
    registration_hash = manifest['registration_sha256']
    registration = read(registered / 'manifest.json', registration_hash)
    registration_name = str((registered / 'manifest.json').relative_to(archive_root))
    require(manifest['input_sha256'][registration_name] == registration_hash, 'Registration linkage differs')
    config = read(registered / 'configuration.json', registration['artifact_sha256']['configuration.json'])
    require(config == records['configuration.json'], 'Configuration differs from original registration')
    plan = read(registered / 'sample_plan.json', registration['artifact_sha256']['sample_plan.json'])
    result = summarise(config, plan, records['audit.json'], records['cases.json'])
    for path, expected in retained.items():
        require(sha(canonical(path).read_bytes()) == expected, 'Source changed during summarisation')
    result.update(schema_version=1, complete=True, empirical_validation_established=False,
        original_primary_screen_unchanged=True, relaxed_thresholds_descriptive_only=True,
        primary_quote_age_seconds=1, quote_ages_seconds=list(QUOTE_AGES),
        primary_funding_age_seconds=300, funding_ages_seconds=list(FUNDING_AGES),
        scope=[
            'Counts reconstruct retained audit predicates; no new acquisition, calibration or repricing is performed.',
            'Repeated underlying group records represent one daily snapshot and are counted separately from distinct daily snapshots.',
            'Empty recorded reference issues do not establish complete documentary clearance, synchronisation or equal-time ordering.',
            'Construction counts concern the retained original dated funding construction; alternative thresholds do not create alternative curves.',
            'Relaxed-threshold intersections are descriptive candidates, not admitted primary observations or independent pricing validation.',
        ])
    verify_summary(result)
    result = dict(summary=result, provenance=dict(
        source_sha256={name: manifest['artifact_sha256'][name] for name in ('audit.json', 'configuration.json', 'cases.json')},
        input_manifest_sha256=retained[folder / 'manifest.json'], registration_manifest_sha256=registration_hash,
        registered_sample_plan_sha256=retained[registered / 'sample_plan.json'],
        source_bytes_rechecked=True, summariser_sha256=sha(Path(__file__).read_bytes())))
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, default=ROOT)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-summary', type=Path, help='Check released arithmetic only; no private audit is read')
    args = parser.parse_args()
    if args.check_summary is not None:
        if args.output is not None: parser.error('--check-summary cannot write an output directory')
        record = strict_json(canonical(args.check_summary).read_bytes())
        verify_summary(record.get('summary', record))
        print('Released count arithmetic verified; withheld source evidence was not checked.')
        return
    if args.output is None: parser.error('--output is required for private-source reconstruction')
    run(args.archive_root, args.input, args.output)
    print('Admission-sensitivity counts verified; empirical validation remains unestablished.')


if __name__ == '__main__':
    main()
