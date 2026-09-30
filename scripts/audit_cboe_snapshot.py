"""Audit one Cboe EOD demonstration CSV without publishing observations."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

from qf_research.market_data.cboe_snapshot import audit_csv

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 64*1024*1024
MAX_MEMBER_BYTES = 64*1024*1024


def digest(value):
    return hashlib.sha256(value).hexdigest()


def inspect(source, *, member=None, max_rows=100_000, underlying=None):
    source = Path(source).resolve()
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds the bounded format-audit size')
    with source.open('rb') as stream:
        raw = stream.read(MAX_SOURCE_BYTES+1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds the bounded format-audit size')
    payload = raw
    archive_member = None
    if zipfile.is_zipfile(io.BytesIO(raw)):
        if not member:
            raise ValueError('Select one explicit CSV archive member; variants are never combined')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            matches = [item for item in archive.infolist() if item.filename == member]
            if len(matches) != 1 or matches[0].is_dir() or not member.lower().endswith('.csv'):
                raise ValueError('Require one unambiguous CSV archive member')
            if matches[0].file_size > MAX_MEMBER_BYTES:
                raise ValueError('Archive member exceeds the bounded format-audit size')
            with archive.open(matches[0]) as stream:
                payload = stream.read(MAX_MEMBER_BYTES+1)
            if len(payload) > MAX_MEMBER_BYTES:
                raise ValueError('Expanded archive member exceeds the audit bound')
            archive_member = member
    elif member is not None:
        raise ValueError('A member can only be specified for a ZIP archive')
    text = payload.decode('utf-8-sig')
    result = audit_csv(io.StringIO(text, newline=''), max_rows=max_rows, underlying=underlying)
    with source.open('rb') as stream:
        if stream.read(MAX_SOURCE_BYTES+1) != raw:
            raise ValueError('Source changed during inspection')
    return dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(raw), source_bytes=len(raw),
        archive_member=archive_member, csv_sha256=digest(payload), csv_bytes=len(payload),
        audit=result, original_source_modified=False,
        source_code_sha256={name: digest((ROOT/name).read_bytes()) for name in (
            'scripts/audit_cboe_snapshot.py', 'src/qf_research/market_data/cboe_snapshot.py')})


def run(source, output, *, member=None, max_rows=100_000, underlying=None):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Aggregate diagnostics must remain in data/private pending review')
    if output.exists():
        raise FileExistsError('Use a new output file')
    result = inspect(source, member=member, max_rows=max_rows, underlying=underlying)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--member')
    parser.add_argument('--underlying', help='Filter identity and pair diagnostics after whole-file schema checks')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-rows', type=int, default=100_000)
    args = parser.parse_args()
    result = run(args.input, args.output, member=args.member, max_rows=args.max_rows, underlying=args.underlying)
    print(json.dumps({name: result['audit'][name] for name in (
        'complete', 'rows', 'rows_in_scope', 'matched_pairs_with_consistent_positive_underlying', 'empirical_acceptance')}))


if __name__ == '__main__':
    main()
