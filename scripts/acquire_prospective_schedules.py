"""Retain dated swap cash-flow schedules alongside the prescribed quote sample.

The schedule response supplies dates and conventions, never a substitute for
historical par-rate quotes. Each request follows that date's collection gate.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from acquire_prospective_sample import registration
from acquire_short_ois_conventions import arguments
from run_conditional_calibration import private, save, sha
from qf_research.market_data.prospective_sample import require_day


def acquire(registered, day, output):
    registered, output = private(registered), private(output)
    cfg, plan, _ = registration(registered)
    require_day(plan, day, datetime.now(timezone.utc).isoformat())
    if output.exists():
        raise FileExistsError('Use a fresh schedule acquisition directory')
    args = arguments({'funding_rics': cfg['term_rics']}, day)
    import lseg.data as ld
    output.mkdir(parents=True)
    record = dict(date=day, role='dated_schedule_evidence', status='in_progress', arguments=args,
        registration_sha256=sha(registered/'manifest.json'), source_sha256=sha(Path(__file__)),
        requested_at=datetime.now(timezone.utc).isoformat(),
        historical_convention_vintage_verified=False, publication_clearance=False)
    save(output/'manifest.json', record)
    try:
        ld.open_session(name='desktop.workspace')
        payload = ld.delivery.endpoint_request.Definition(**args).get_data().data.raw
        save(output/'response.json', payload)
        record.update(status='retained', file='response.json', sha256=sha(output/'response.json'))
    except Exception as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record['completed_at'] = datetime.now(timezone.utc).isoformat()
        save(output/'manifest.json', record)
        ld.close_session()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--day', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    acquire(a.registration, a.day, a.output)
