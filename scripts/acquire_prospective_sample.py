"""Archive reference evidence or acquire one completed, prospectively fixed date."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from review_lseg_pilot import metadata_arguments
from qf_research.market_data.prospective_sample import sample_plan, require_day, request_plan


def configuration(path):
    path = private(path)
    cfg = read(path)
    calendar_path = ROOT/cfg['calendar_path']
    if sha(calendar_path) != cfg['calendar_sha256']:
        raise ValueError('Calendar changed after protocol construction')
    return cfg, sample_plan(cfg, read(calendar_path))


def registration(path):
    path = private(path)
    record = read(path/'manifest.json')
    verify_mapping(record['input_sha256'])
    verify_mapping(record['source_sha256'])
    verify_mapping(record['artifact_sha256'], path)
    cfg, plan = configuration(path/'configuration.json')
    if plan != read(path/'sample_plan.json'):
        raise ValueError('Registered sample plan changed')
    return cfg, plan, record


def acquire(config_path, output, *, day=None, registration_path=None):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh acquisition directory')
    if day is None:
        cfg, _ = configuration(config_path)
        role = 'pre_observation_reference'
        registered = None
    else:
        cfg, plan, registered = registration(registration_path)
        slot = require_day(plan, day, datetime.now(timezone.utc).isoformat())
        role = 'prospective_observation'
    # In particular, no service import, session or output precedes the clock gate.
    import lseg.data as ld
    output.mkdir(parents=True)
    save(output/'configuration.json',cfg)
    manifest = dict(status='in_progress',role=role,date=day,requests=[],
        configuration_sha256=sha(output/'configuration.json'),source_sha256=sha(Path(__file__)),
        library_version=ld.__version__,representation='SDK-decoded payloads, not original HTTP bytes',
        registration_sha256=sha(private(registration_path)/'manifest.json') if registered else None,
        full_empirical_acceptance=False,publication_clearance=False)

    def request(name, method, args, function):
        entry = dict(name=name,method=method,arguments=args,requested_at=datetime.now(timezone.utc).isoformat())
        manifest['requests'].append(entry);save(output/'manifest.json',manifest)
        try:
            payload=function()
            save(output/(name+'.json'),payload)
            entry.update(file=name+'.json',sha256=sha(output/(name+'.json')),status='retained',
                         retrieved_at=datetime.now(timezone.utc).isoformat())
        except Exception as exc:
            entry.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                         completed_at=datetime.now(timezone.utc).isoformat())
        save(output/'manifest.json',manifest)

    try:
        ld.open_session(name='desktop.workspace')
        args=dict(view=ld.discovery.Views.EQUITY_DERIVATIVE_QUOTES,
            filter=' or '.join("RIC eq '"+r.replace("'","''")+"'" for r in cfg['option_rics']),
            select=','.join(cfg['contract_fields']),top=cfg['reference_limit'])
        request('contract_reference','discovery.search',dict(args,view='EQUITY_DERIVATIVE_QUOTES'),
                lambda:json.loads(ld.discovery.search(**args).to_json(orient='records',date_format='iso')))
        instruments=cfg['option_rics']+[cfg['underlying']]+cfg['term_rics']+cfg['front_rics']
        for i,ric in enumerate(instruments):
            args=metadata_arguments(ric)
            request(f'metadata_{i:02d}','metadata',args,
                lambda:ld.delivery.endpoint_request.Definition(**args).get_data().data.raw)
        for group in ('term','front'):
            args=dict(universe=cfg[group+'_rics'],fields=cfg[group+'_reference_fields'])
            request(group+'_reference','desktop_get_data',args,
                lambda:json.loads(ld.get_data(**args).to_json(orient='split',date_format='iso')))
        if day is not None:
            for entry in request_plan(cfg,slot,cfg['option_rics']):
                args=entry['arguments']
                request(entry['name'],entry['method'],args,
                    lambda:ld.content.historical_pricing.events.Definition(**args).get_data().data.raw)
        manifest['status']=('retained_review_pending' if all(r['status']=='retained' for r in manifest['requests'])
                            else 'partial_evidence_retained')
    except Exception as exc:
        manifest.update(status='session_failed_evidence_retained',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        ld.close_session();save(output/'manifest.json',manifest)
    print('Reference acquisition retained.' if day is None else 'Prescribed date acquisition retained for later review.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    pre=sub.add_parser('reference');pre.add_argument('--config',type=Path,required=True)
    collect=sub.add_parser('collect');collect.add_argument('--registration',type=Path,required=True);collect.add_argument('--day',required=True)
    for parser in (pre,collect):parser.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='reference':acquire(a.config,a.output)
    else:acquire(None,a.output,day=a.day,registration_path=a.registration)
