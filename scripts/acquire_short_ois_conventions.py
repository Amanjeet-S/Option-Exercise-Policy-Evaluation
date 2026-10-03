"""Retain provider cash-flow definitions for the fixed short-OIS development panel."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/short_ois_2026-09-20.json'
FIELDS = ['InstrumentCode', 'InstrumentTag', 'LegIndex', 'TradeDate', 'StartDate',
          'EndDate', 'Tenor', 'InterestType', 'InterestCalculationMethod',
          'InterestPaymentFrequency', 'InterestPaymentDelay',
          'PaymentBusinessDayConvention', 'PaymentBusinessDays', 'PaymentRollConvention',
          'IndexName', 'IndexTenor', 'IndexFixingRic', 'IndexFixingLag',
          'IndexLockoutPeriod', 'IndexObservationMethod', 'FixedRatePercent',
          'ParRatePercent', 'CashFlows', 'NotionalAmount', 'ErrorCode', 'ErrorMessage']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def arguments(cfg, day):
    return dict(url='/data/quantitative-analytics/v1/financial-contracts', method='POST',
        body_parameters=dict(fields=FIELDS, outputs=['Headers', 'Data', 'Statuses'],
        universe=[dict(instrumentType='Swap',
            instrumentDefinition=dict(instrumentCode=ric, instrumentTag=ric),
            pricingParameters=dict(valuationDate=day, marketDataDate=day, priceSide='Mid'))
            for ric in cfg['funding_rics']]))


def acquire(output):
    import lseg.data as ld
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'data/private/short_ois'):
        raise ValueError('Conventions and numerical outputs must remain private')
    cfg = json.loads(CONFIG.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / 'configuration.json').write_bytes(CONFIG.read_bytes())
    (output / 'acquisition_source.py').write_bytes(Path(__file__).read_bytes())
    manifest = dict(status='in_progress', requests=[], library_version=ld.__version__,
        configuration_sha256=digest(CONFIG), acquisition_source_sha256=digest(Path(__file__)),
        representation='SDK-decoded responses; current provider conventions applied to dated contracts',
        historical_convention_vintage_verified=False, publication_clearance=False)
    try:
        ld.open_session(name='desktop.workspace')
        for day in cfg['observation_dates']:
            args = arguments(cfg, day)
            item = dict(date=day, arguments=args, requested_at=datetime.now(timezone.utc).isoformat())
            manifest['requests'].append(item)
            save(output / 'manifest.json', manifest)
            try:
                r = ld.delivery.endpoint_request.Definition(**args).get_data()
                path = output / (day + '.json')
                save(path, r.data.raw)
                item.update(status='retained', file=path.name, sha256=digest(path),
                            retrieved_at=datetime.now(timezone.utc).isoformat())
                print('Retained dated contract definitions:', day, flush=True)
            except Exception as exc:
                item.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            save(output / 'manifest.json', manifest)
        manifest['status'] = 'retained_review_pending' if all(
            r['status'] == 'retained' for r in manifest['requests']) else 'partial'
    finally:
        ld.close_session()
        save(output / 'manifest.json', manifest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    acquire(p.parse_args().output)
