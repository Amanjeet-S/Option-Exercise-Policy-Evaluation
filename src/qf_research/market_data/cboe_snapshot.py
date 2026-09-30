"""Conservative structural inspection of Cboe EOD 15:45 CSV snapshots.

The suffix identifies a product column, not an independently observed quote
update time. A matched pair is a format-level result. It does not establish
simultaneity, executable prices, contract equivalence or market validation.
"""

from collections import Counter, defaultdict
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import re


REQUIRED_FIELDS = (
    'underlying_symbol', 'quote_date', 'root', 'expiration', 'strike', 'option_type',
    'bid_1545', 'ask_1545', 'underlying_bid_1545', 'underlying_ask_1545',
)
KNOWN_FIELDS = frozenset(REQUIRED_FIELDS) | frozenset((
    'open', 'high', 'low', 'close', 'trade_volume', 'bid_size_1545', 'ask_size_1545',
    'implied_underlying_price_1545', 'active_underlying_price_1545',
    'implied_volatility_1545', 'delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545',
    'rho_1545', 'bid_size_eod', 'bid_eod', 'ask_size_eod', 'ask_eod',
    'underlying_bid_eod', 'underlying_ask_eod', 'vwap', 'open_interest', 'delivery_code',
))
SNAPSHOT_LABEL = '1545_product_columns'
EXTERNAL_GATES = (
    'provider_snapshot_time_timezone_and_early_close_conventions',
    'quote_update_age_and_correction_or_consolidation_semantics',
    'contract_exercise_style_deliverable_multiplier_and_settlement',
    'dated_funding_curve_and_front_interval_evidence',
    'dividend_information_available_at_the_valuation_cutoff',
    'permitted_research_use_and_redistribution_scope',
    'independent_sampling_and_predeclared_empirical_acceptance',
)
_DECIMAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z')
_DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')


def decimal_value(text):
    """Parse finite ordinary decimal syntax without binary floating conversion."""
    if not isinstance(text, str) or len(text) > 64 or not _DECIMAL.fullmatch(text):
        raise ValueError('Invalid decimal field')
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError('Invalid decimal field') from error
    if not value.is_finite() or abs(value.as_tuple().exponent) > 100:
        raise ValueError('Non-finite or unsupported decimal exponent')
    return value


def iso_date(text):
    if not isinstance(text, str) or not _DATE.fullmatch(text):
        raise ValueError('An explicit ISO calendar date is required')
    return date.fromisoformat(text)


def audit_csv(stream, *, max_rows=100_000, underlying=None):
    """Read one CSV variant and return aggregate diagnostics without prices.

    Duplicate identities, even identical duplicates, cannot be resolved by row
    order. Every affected pair is excluded. Conflicting positive underlying
    states anywhere within a symbol/date/snapshot group exclude that group.
    Zero option bids remain visible and can pass the non-crossed-price screen;
    positive-bid pair coverage is reported separately.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000:
        raise ValueError('max_rows must be an integer from one to one million')
    if underlying is not None and (not isinstance(underlying, str) or not underlying or underlying != underlying.strip()):
        raise ValueError('The underlying filter must be a nonempty unpadded symbol')
    result = dict(schema_version=1, format='cboe_eod_1545',
        scope='Structural format audit only; no empirical price or calibration acceptance',
        snapshot_label=SNAPSHOT_LABEL, snapshot_timezone_established=False,
        quote_age_established=False, contract_terms_established=False,
        empirical_acceptance=False, source_is_demonstration_or_user_supplied=True,
        missing_external_evidence=list(EXTERNAL_GATES), complete=False,
        file_structure_pass=False, aggregate_scope='parsed_prefix_only',
        row_limit=max_rows, rows=0, rows_in_scope=0, rows_excluded_by_filter=0,
        underlying_filter=underlying, scope_issues=[],
        row_count_scope='rows covers the whole parsed file; identity and pair counts use the selected underlying',
        row_issue_scope='selected rows, except malformed row widths counted across the whole file',
        root_mapping_scope='whole parsed file, including symbols outside the filter',
        header_fields=[], header_issues=[],
        row_issues={}, rows_with_issues=0, valid_identity_rows=0,
        underlying_counts={}, quote_date_counts={}, nonempty_delivery_code_rows=0,
        unique_identity_count=0, duplicate_identity_groups=0, extra_duplicate_rows=0,
        pair_key_count=0, unmatched_pair_keys=0, duplicated_pair_keys=0,
        matched_unique_call_put_keys=0, matched_non_crossed_option_pairs=0,
        matched_pairs_with_positive_option_bids=0,
        matched_pairs_with_consistent_positive_underlying=0,
        matched_pairs_with_positive_option_bids_and_underlying=0,
        pair_underlying_disagreements=0, conflicting_underlying_groups=0,
        ambiguous_root_underlying_groups=0,
        row_limit_exceeded=False, csv_parse_error=False)
    reader = csv.reader(stream, strict=True)
    try:
        header = next(reader)
    except StopIteration:
        result['header_issues'].append('empty_file')
        return result
    except csv.Error:
        result['header_issues'].append('invalid_csv_header')
        result['csv_parse_error'] = True
        return result
    # A headerless file must not echo its first observation as schema metadata.
    result['header_fields'] = [name for name in header if name in KNOWN_FIELDS]
    result['header_field_count'] = len(header)
    result['unknown_header_field_count'] = sum(name not in KNOWN_FIELDS for name in header)
    if not header or any(not name or name != name.strip() for name in header):
        result['header_issues'].append('empty_or_padded_header_name')
    if len(set(header)) != len(header):
        result['header_issues'].append('duplicate_header_names')
    missing = sorted(set(REQUIRED_FIELDS)-set(header))
    if missing:
        result['header_issues'].append('missing_required_fields')
        result['missing_fields'] = missing
    if result['header_issues']:
        return result

    issues, underlyings, dates = Counter(), Counter(), Counter()
    groups = defaultdict(lambda: {'C': [], 'P': []})
    identities = Counter()
    underlying_states = defaultdict(set)
    root_underlyings = defaultdict(set)
    try:
        for number, cells in enumerate(reader, 1):
            if number > max_rows:
                result['row_limit_exceeded'] = True
                break
            result['rows'] += 1
            row_issues = set()
            if len(cells) != len(header):
                issues['malformed_row_width'] += 1
                result['rows_with_issues'] += 1
                continue
            row = dict(zip(header, cells))
            # Root-to-underlying identity is checked before filtering so an
            # excluded symbol cannot conceal an ambiguous assignment.
            if all(row[name] and row[name] == row[name].strip() for name in ('root', 'underlying_symbol')):
                try:
                    mapping_date = iso_date(row['quote_date'])
                except ValueError:
                    pass
                else:
                    root_underlyings[(row['root'], mapping_date, SNAPSHOT_LABEL)].add(row['underlying_symbol'])
            if underlying is not None and row['underlying_symbol'] != underlying:
                result['rows_excluded_by_filter'] += 1
                continue
            result['rows_in_scope'] += 1
            valid_identity = True
            for name in ('underlying_symbol', 'root'):
                if not row[name] or row[name] != row[name].strip():
                    row_issues.add('invalid_'+name)
                    valid_identity = False
            quote_date = expiration = None
            for name in ('quote_date', 'expiration'):
                try:
                    parsed = iso_date(row[name])
                    if name == 'quote_date':
                        quote_date = parsed
                    else:
                        expiration = parsed
                except ValueError:
                    row_issues.add('invalid_'+name)
                    valid_identity = False
            if quote_date is not None and expiration is not None and expiration < quote_date:
                row_issues.add('expired_before_quote_date')
                valid_identity = False
            try:
                strike = decimal_value(row['strike'])
                if strike <= 0:
                    raise ValueError('Positive strike required')
            except ValueError:
                row_issues.add('invalid_strike')
                valid_identity = False
                strike = None
            if row['option_type'] not in ('C', 'P'):
                row_issues.add('invalid_option_type')
                valid_identity = False

            prices = {}
            for name in REQUIRED_FIELDS[6:]:
                if row[name] == '':
                    row_issues.add('missing_'+name)
                    continue
                try:
                    prices[name] = decimal_value(row[name])
                except ValueError:
                    row_issues.add('invalid_'+name)
            bid, ask = prices.get('bid_1545'), prices.get('ask_1545')
            ubid, uask = prices.get('underlying_bid_1545'), prices.get('underlying_ask_1545')
            if bid is not None and bid < 0:
                row_issues.add('negative_option_bid')
            if bid == 0:
                row_issues.add('zero_option_bid')
            if ask is not None and ask <= 0:
                row_issues.add('non_positive_option_ask')
            if bid is not None and ask is not None and bid > ask:
                row_issues.add('crossed_option_quote')
            for name, value in (('bid', ubid), ('ask', uask)):
                if value == 0:
                    row_issues.add('zero_underlying_'+name)
                elif value is not None and value < 0:
                    row_issues.add('negative_underlying_'+name)
            if ubid is not None and uask is not None and ubid > uask:
                row_issues.add('crossed_underlying_quote')
            option_screen = bid is not None and ask is not None and bid >= 0 and ask > 0 and bid <= ask
            underlying_screen = ubid is not None and uask is not None and 0 < ubid <= uask
            if row.get('delivery_code', ''):
                result['nonempty_delivery_code_rows'] += 1
            state_key = None
            root_key = None
            if quote_date is not None and row['underlying_symbol'] and row['underlying_symbol'] == row['underlying_symbol'].strip():
                state_key = (row['underlying_symbol'], quote_date, SNAPSHOT_LABEL)
                if underlying_screen:
                    underlying_states[state_key].add((ubid, uask))
                if row['root'] and row['root'] == row['root'].strip():
                    root_key = (row['root'], quote_date, SNAPSHOT_LABEL)
                    root_underlyings[root_key].add(row['underlying_symbol'])
            if valid_identity:
                result['valid_identity_rows'] += 1
                underlyings[row['underlying_symbol']] += 1
                dates[row['quote_date']] += 1
                pair_key = (*state_key, row['root'], expiration, strike)
                identity = (*pair_key, row['option_type'])
                identities[identity] += 1
                groups[pair_key][row['option_type']].append(dict(
                    option_screen=option_screen, positive_bid=bid is not None and bid > 0,
                    underlying_screen=underlying_screen, underlying=(ubid, uask),
                    state_key=state_key, root_key=root_key))
            if row_issues:
                result['rows_with_issues'] += 1
                issues.update(row_issues)
        else:
            result['complete'] = True
    except csv.Error:
        result['csv_parse_error'] = True
    if result['rows'] == 0:
        result['header_issues'].append('no_data_rows')
    if underlying is not None and result['rows_in_scope'] == 0:
        result['scope_issues'].append('no_rows_for_requested_underlying')
    result.update(row_issues=dict(sorted(issues.items())), underlying_counts=dict(sorted(underlyings.items())),
        quote_date_counts=dict(sorted(dates.items())), unique_identity_count=len(identities),
        duplicate_identity_groups=sum(count > 1 for count in identities.values()),
        extra_duplicate_rows=sum(count-1 for count in identities.values()), pair_key_count=len(groups),
        conflicting_underlying_groups=sum(len(states) > 1 for states in underlying_states.values()),
        ambiguous_root_underlying_groups=sum(len(symbols) > 1 for symbols in root_underlyings.values()))
    for sides in groups.values():
        if not sides['C'] or not sides['P']:
            result['unmatched_pair_keys'] += 1
        if len(sides['C']) > 1 or len(sides['P']) > 1:
            result['duplicated_pair_keys'] += 1
        if len(sides['C']) != 1 or len(sides['P']) != 1:
            continue
        result['matched_unique_call_put_keys'] += 1
        call, put = sides['C'][0], sides['P'][0]
        if call['underlying'] != put['underlying']:
            result['pair_underlying_disagreements'] += 1
        if not call['option_screen'] or not put['option_screen']:
            continue
        result['matched_non_crossed_option_pairs'] += 1
        positive = call['positive_bid'] and put['positive_bid']
        result['matched_pairs_with_positive_option_bids'] += int(positive)
        consistent = (call['underlying_screen'] and put['underlying_screen']
                      and call['underlying'] == put['underlying']
                      and len(underlying_states[call['state_key']]) == 1
                      and len(root_underlyings[call['root_key']]) == 1)
        result['matched_pairs_with_consistent_positive_underlying'] += int(consistent)
        result['matched_pairs_with_positive_option_bids_and_underlying'] += int(consistent and positive)
    result['file_structure_pass'] = (result['complete'] and result['rows'] > 0
        and not result['header_issues'] and not issues['malformed_row_width']
        and not issues['expired_before_quote_date']
        and not any(count for name, count in issues.items() if name.startswith(('invalid_', 'missing_'))))
    if result['complete']:
        result['aggregate_scope'] = 'complete_csv'
    return result
