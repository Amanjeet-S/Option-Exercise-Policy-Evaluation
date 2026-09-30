"""Declared CSV fixtures for format checks, not observed financial data."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import pytest

from qf_research.market_data.cboe_snapshot import audit_csv, decimal_value, REQUIRED_FIELDS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cboe_snapshot_cli', ROOT/'scripts/audit_cboe_snapshot.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def row(side='C', **changes):
    result = dict(underlying_symbol='MATHEMATICAL_FIXTURE', quote_date='2020-01-02',
        root='FIXTURE', expiration='2020-02-03', strike='100.0', option_type=side,
        bid_1545='1.0', ask_1545='2.0', underlying_bid_1545='99.0', underlying_ask_1545='101.0')
    result.update(changes)
    return result


def text(rows, header=REQUIRED_FIELDS):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    for item in rows:
        writer.writerow([item.get(key, '') for key in header] if isinstance(item, dict) else item)
    return stream.getvalue()


def audit(rows, **kwargs):
    return audit_csv(io.StringIO(text(rows)), **kwargs)


def test_genuine_unique_pair_without_any_market_acceptance():
    result = audit([row(), row('P', strike='100.00')])
    assert result['complete'] and result['file_structure_pass']
    assert result['matched_pairs_with_consistent_positive_underlying'] == 1
    assert result['matched_pairs_with_positive_option_bids_and_underlying'] == 1
    assert result['empirical_acceptance'] is False
    assert result['quote_age_established'] is False
    assert result['contract_terms_established'] is False
    assert result['missing_external_evidence']
    encoded = json.dumps(result)
    assert '99.0' not in encoded and '101.0' not in encoded
    assert '100.00' not in encoded  # No contract rows or numerical quote values.


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '1_000', '', ' 1', '1 ', '1/2', '1e999', '9'*65])
def test_strict_finite_decimal_parser(value):
    with pytest.raises(ValueError):
        decimal_value(value)


def test_decimal_parser_preserves_exact_text_value():
    assert str(decimal_value('0.100000000000000001')) == '0.100000000000000001'
    assert decimal_value('1e-2') == decimal_value('.01')


def test_empty_header_only_duplicate_and_missing_headers():
    assert audit_csv(io.StringIO(''))['header_issues'] == ['empty_file']
    assert 'no_data_rows' in audit([])['header_issues']
    duplicate = audit_csv(io.StringIO(text([row()], (*REQUIRED_FIELDS, 'root'))))
    assert 'duplicate_header_names' in duplicate['header_issues']
    assert not duplicate['complete']
    missing = audit_csv(io.StringIO(text([row()], REQUIRED_FIELDS[:-1])))
    assert missing['missing_fields'] == ['underlying_ask_1545']
    assert not missing['file_structure_pass']


def test_bad_width_and_csv_quoting_are_reported_without_filling():
    result = audit([row(), ['short'], row('P')])
    assert result['rows'] == 3 and result['row_issues']['malformed_row_width'] == 1
    assert not result['file_structure_pass']
    malformed = text([row()])+'"unterminated'
    result = audit_csv(io.StringIO(malformed))
    assert result['csv_parse_error'] and not result['complete']
    assert result['aggregate_scope'] == 'parsed_prefix_only'


@pytest.mark.parametrize('changes,issue', [
    ({'strike':'0'}, 'invalid_strike'),
    ({'strike':'NaN'}, 'invalid_strike'),
    ({'option_type':'call'}, 'invalid_option_type'),
    ({'quote_date':'20200102'}, 'invalid_quote_date'),
    ({'quote_date':'2020-02-30'}, 'invalid_quote_date'),
    ({'expiration':'2019-12-31'}, 'expired_before_quote_date'),
    ({'root':''}, 'invalid_root'),
    ({'underlying_symbol':' padded'}, 'invalid_underlying_symbol'),
])
def test_invalid_identity_cannot_form_a_pair(changes, issue):
    result = audit([row(**changes), row('P')])
    assert result['row_issues'][issue] == 1
    assert result['matched_unique_call_put_keys'] == 0


@pytest.mark.parametrize('changes,issue', [
    ({'bid_1545':'-1'}, 'negative_option_bid'),
    ({'ask_1545':'0'}, 'non_positive_option_ask'),
    ({'bid_1545':'3'}, 'crossed_option_quote'),
    ({'ask_1545':'NaN'}, 'invalid_ask_1545'),
    ({'bid_1545':''}, 'missing_bid_1545'),
])
def test_option_price_screen_rejects_bad_quotes(changes, issue):
    result = audit([row(**changes), row('P')])
    assert result['row_issues'][issue] == 1
    assert result['matched_unique_call_put_keys'] == 1
    assert result['matched_non_crossed_option_pairs'] == 0


def test_zero_bid_and_locked_option_quotes_are_not_silently_discarded():
    result = audit([row(bid_1545='0'), row('P', bid_1545='2', ask_1545='2')])
    assert result['row_issues']['zero_option_bid'] == 1
    assert result['matched_non_crossed_option_pairs'] == 1
    assert result['matched_pairs_with_positive_option_bids'] == 0
    assert result['matched_pairs_with_consistent_positive_underlying'] == 1


@pytest.mark.parametrize('changes,issue', [
    ({'underlying_bid_1545':''}, 'missing_underlying_bid_1545'),
    ({'underlying_bid_1545':'0'}, 'zero_underlying_bid'),
    ({'underlying_ask_1545':'0'}, 'zero_underlying_ask'),
    ({'underlying_bid_1545':'-1'}, 'negative_underlying_bid'),
    ({'underlying_ask_1545':'98'}, 'crossed_underlying_quote'),
    ({'underlying_bid_1545':'inf'}, 'invalid_underlying_bid_1545'),
])
def test_underlying_observation_is_not_imputed(changes, issue):
    result = audit([row(**changes), row('P', **changes)])
    assert result['row_issues'][issue] == 2
    assert result['matched_non_crossed_option_pairs'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0


def test_duplicate_identity_even_if_identical_invalidates_pair():
    result = audit([row(), row(strike='100.00'), row('P')])
    assert result['duplicate_identity_groups'] == 1
    assert result['extra_duplicate_rows'] == 1
    assert result['duplicated_pair_keys'] == 1
    assert result['matched_unique_call_put_keys'] == 0


@pytest.mark.parametrize('changes', [
    {'root':'OTHER'}, {'strike':'101'}, {'expiration':'2020-02-04'},
    {'quote_date':'2020-01-03'}, {'underlying_symbol':'OTHER'},
])
def test_call_and_put_must_share_the_complete_identity(changes):
    result = audit([row(), row('P', **changes)])
    assert result['matched_unique_call_put_keys'] == 0
    assert result['unmatched_pair_keys'] == 2


def test_conflicting_underlying_states_reject_every_pair_in_the_state_group():
    result = audit([row(), row('P'), row(strike='110', underlying_bid_1545='98'),
                    row('P', strike='110', underlying_bid_1545='98')])
    assert result['matched_unique_call_put_keys'] == 2
    assert result['conflicting_underlying_groups'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0


def test_call_put_underlying_disagreement_is_explicit():
    result = audit([row(), row('P', underlying_bid_1545='98')])
    assert result['pair_underlying_disagreements'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0


def test_ambiguous_root_underlying_mapping_is_not_accepted():
    result = audit([row(), row('P'), row(underlying_symbol='OTHER'), row('P', underlying_symbol='OTHER')])
    assert result['ambiguous_root_underlying_groups'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0


def test_row_bound_prevents_a_complete_file_claim():
    result = audit([row(), row('P')], max_rows=1)
    assert result['row_limit_exceeded'] and not result['complete']
    assert result['rows'] == 1 and result['aggregate_scope'] == 'parsed_prefix_only'


def test_wrapper_requires_explicit_archive_variant_and_retains_source(tmp_path):
    path = tmp_path/'sample.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('a.csv', text([row(), row('P')]))
        archive.writestr('b.csv', text([row()]))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='explicit CSV'):
        cli.inspect(path)
    result = cli.inspect(path, member='a.csv')
    assert result['audit']['rows'] == 2
    assert path.read_bytes() == original
    assert result['source_sha256'] == cli.digest(original)
    assert '100.0' not in json.dumps(result)


def test_cli_output_confinement_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    source = tmp_path/'input.csv'
    source.write_text(text([row(), row('P')]))
    with pytest.raises(ValueError, match='data/private'):
        cli.run(source, tmp_path/'public.json')
    output = tmp_path/'data/private/output.json'
    output.parent.mkdir(parents=True)
    output.write_text('preserve')
    with pytest.raises(FileExistsError):
        cli.run(source, output)
    assert output.read_text() == 'preserve'


def test_bounded_archive_member_cannot_expand_past_limit(tmp_path, monkeypatch):
    path = tmp_path/'sample.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('a.csv', text([row(), row('P')]))
    monkeypatch.setattr(cli, 'MAX_MEMBER_BYTES', 20)
    with pytest.raises(ValueError, match='member exceeds'):
        cli.inspect(path, member='a.csv')


def test_filter_precedes_identity_and_pair_counts_but_not_width_checks():
    result = audit([row(), row('P'), row(underlying_symbol='OTHER'),
                    row('P', underlying_symbol='OTHER'), ['bad', 'width']],
                    underlying='MATHEMATICAL_FIXTURE')
    assert result['rows'] == 5
    assert result['rows_in_scope'] == 2
    assert result['rows_excluded_by_filter'] == 2
    assert result['row_issues']['malformed_row_width'] == 1
    assert result['underlying_counts'] == {'MATHEMATICAL_FIXTURE': 2}
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0
    assert result['ambiguous_root_underlying_groups'] == 1
    assert not result['file_structure_pass']


def test_empty_filtered_scope_cannot_imply_coverage():
    result = audit([row(), row('P')], underlying='OTHER')
    assert result['rows'] == 2 and result['rows_in_scope'] == 0
    assert result['scope_issues'] == ['no_rows_for_requested_underlying']
    assert result['matched_unique_call_put_keys'] == 0


def test_expired_contract_prevents_structure_pass():
    result = audit([row(expiration='2019-12-31')])
    assert not result['file_structure_pass']


@pytest.mark.parametrize('changes', [{'strike':'bad'}, {'option_type':'invalid'}])
def test_bad_contract_identity_does_not_hide_underlying_conflict(changes):
    result = audit([row(), row('P'), row(underlying_bid_1545='98', **changes)])
    assert result['conflicting_underlying_groups'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0


def test_headerless_data_does_not_echo_observation_values():
    stream = io.StringIO()
    csv.writer(stream).writerow([row()[name] for name in REQUIRED_FIELDS])
    result = audit_csv(io.StringIO(stream.getvalue()))
    encoded = json.dumps(result)
    for value in ['100.0', '99.0', '101.0', 'MATHEMATICAL_FIXTURE', '2020-01-02']:
        assert value not in encoded
    assert result['header_fields'] == []
    assert result['unknown_header_field_count'] == len(REQUIRED_FIELDS)


def test_filter_does_not_hide_a_conflicting_root_assignment():
    result = audit([row(), row('P'), row(underlying_symbol='OTHER'),
                    row('P', underlying_symbol='OTHER')], underlying='MATHEMATICAL_FIXTURE')
    assert result['rows_in_scope'] == 2
    assert result['matched_unique_call_put_keys'] == 1
    assert result['ambiguous_root_underlying_groups'] == 1
    assert result['matched_pairs_with_consistent_positive_underlying'] == 0
