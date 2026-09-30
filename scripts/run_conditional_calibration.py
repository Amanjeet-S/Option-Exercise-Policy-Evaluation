"""Prepare and fit a private October SPY diagnostic under explicit assumptions."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

import review_lseg_pilot
import build_valuation_inputs
from qf_research.market_data.conditional_calibration import (
    CalibrationQuote, select_october_pairs, fit_shared_volatility, price_quote, quote_diagnostics,
)
from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.valuation_inputs import inspect_sofr_curve
from qf_research.market_data.occ_series import parse_series_search, single_underlying
from decimal import Decimal
from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'projects/01_optimal_stopping'
LSEG = ROOT/'data/private/lseg/2026-09-16'
ANNOUNCEMENTS = ROOT/'data/private/dividend_announcements/2026-09-17/audit_v1'
CONFIG = PROJECT/'configs/october_calibration_2026-09-17.json'
CODE = [Path(__file__).resolve(), ROOT/'src/qf_research/market_data/conditional_calibration.py',
        ROOT/'src/qf_research/market_data/funded_inputs.py',ROOT/'src/qf_research/market_data/event_calendar.py',
        ROOT/'src/qf_research/market_data/valuation_inputs.py',ROOT/'src/qf_research/optimal_stopping/funded_cash.py',
        ROOT/'src/qf_research/optimal_stopping/contracts.py']


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def save(p,obj): p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def private(p):
    p=p.resolve()
    if not p.is_relative_to((ROOT/'data/private').resolve()):
        raise ValueError('Keep all empirical input and output files private')
    return p


def verify_mapping(mapping, base=ROOT):
    for name,expected in mapping.items():
        p=(base/name).resolve()
        if not p.is_relative_to(base.resolve()) or not p.is_file() or sha(p)!=expected:
            raise ValueError(f'Evidence checksum or path failed: {name}')


def prepare(output):
    output=private(output)
    if output.exists(): raise FileExistsError('Use a fresh output directory')
    cfg=read(CONFIG)
    if not cfg['conditional_private_diagnostic_allowed'] or cfg['full_empirical_acceptance']:
        raise ValueError('This runner requires the declared conditional scope')
    protocol=ROOT/'data/private/conditional_calibration/2026-09-17/source_review/protocol_record.json'
    if read(protocol)['config_sha256']!=sha(CONFIG):raise ValueError('Pre-fit protocol changed')
    # Recompute original source audits; do not trust a favourable selection flag alone.
    scratch_base=ROOT/'data/private/lseg'
    with tempfile.TemporaryDirectory(prefix='verification_',dir=scratch_base) as temp:
        temp=Path(temp)
        with redirect_stdout(io.StringIO()):
            review_lseg_pilot.review(LSEG/'expanded_pilot_v2',LSEG/'field_metadata_v1',temp/'quotes')
            build_valuation_inputs.audit(PROJECT/'configs/valuation_inputs_2026-09-16.json',
                                         LSEG/'sofr_curves_v1',LSEG/'valuation_sources_v1',temp/'curves')
        for generated,retained in [(temp/'quotes/quote_review.json',LSEG/'quote_review_v1/quote_review.json'),
                                  (temp/'curves/valuation_inputs.json',LSEG/'valuation_inputs_v1/valuation_inputs.json')]:
            if generated.read_bytes()!=retained.read_bytes():raise ValueError('Original audit reproduction differs')
    ann=read(ANNOUNCEMENTS/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(ann[k])
    verify_mapping(ann['artifact_sha256'],ANNOUNCEMENTS)
    forecasts=read(ANNOUNCEMENTS/'reconstructed_forecasts.json')
    lookup={(f['as_of'],f['model'],f['ex_date']):f for f in forecasts}
    if len(lookup)!=len(forecasts):raise ValueError('Duplicate forecast record')
    review=read(LSEG/'quote_review_v1/quote_review.json')
    references=[r for p in sorted((LSEG/'expanded_pilot_v2').glob('reference_*.json')) for r in read(p)]
    reference_folder=LSEG/'empirical_acceptance_probe_v2'
    for r in read(reference_folder/'manifest.json')['requests']:
        if 'file' in r:verify_mapping({r['file']:r['sha256']},reference_folder)
    listed={(r.symbol,r.expiry.isoformat(),r.strike,r.kind) for r in parse_series_search((reference_folder/'occ_series_response.bin').read_text())}
    for r in references:
        identity=(single_underlying(r['UnderlyingQuoteRIC']),r['ExpiryDate'][:10],Decimal(str(r['StrikePrice'])),r['CallPutOption'].lower())
        if identity not in listed or identity[0]!='SPY' or r['ExerciseStyle']!='A' or r['Currency']!='USD' or r['LotSize']!=100 or r['StrikeMultiplier']!=1:
            raise ValueError('Current reference support differs from the declared standard-contract model')
    selected,exclusions=select_october_pairs(review,references,cfg)
    vcfg=read(PROJECT/'configs/valuation_inputs_2026-09-16.json')
    curves={}
    for request in read(LSEG/'sofr_curves_v1/manifest.json')['requests']:
        day=request['curve_date']
        curves[day],_=inspect_sofr_curve(read(LSEG/'sofr_curves_v1'/request['file']),day,vcfg['curve_id'])
    cal=EventCalendar.from_config(read(PROJECT/'configs/us_event_calendar_2026-09-16.json'))
    problems=[]
    for row in selected:
        observation=datetime.fromisoformat(row['cutoff'])
        day=date.fromisoformat(row['observation_date'])
        curve=curves[vcfg['curve_dates_by_observation'][row['observation_date']]]
        for name in cfg['dividend_models']:
            forecast=lookup[(row['cutoff'],name,cfg['ex_date'])]
            if (forecast['pay_date']!=cfg['pay_date'] or forecast['currency']!='USD'
                or datetime.fromisoformat(forecast['latest_input_publication_bound'])>=observation):
                raise ValueError('Forecast identity or information bound invalid')
            amount=float(forecast['regular_dividend_estimate'])
            problem=build_dated_problem(cal,curve,observation,date.fromisoformat(row['expiry']),
                row['quoted_spot'],row['strike'],0.,row['kind'],
                (DatedDividend(date.fromisoformat(cfg['ex_date']),date.fromisoformat(cfg['pay_date']),amount,
                               'dated announcement benchmark; explicit no-extra-payment model assumption'),),
                premium_payment_date=cal.settlement_after(day).day)
            if problem.provisional_delivery:raise ValueError('Provisional delivery is outside this pilot')
            problems.append({'observation_id':row['observation_id'],'dividend_model':name,'split':row['split'],
                'bid':row['bid'],'ask':row['ask'],'model':asdict(problem.model),'option':asdict(problem.option),
                'decisions':[asdict(x) for x in problem.decisions],
                'forecast':forecast,'curve_date':curve.anchor.isoformat(),
                'premium_payment_date':problem.premium_payment_date.isoformat(),
                'original_empirical_calibration_accepted':problem.empirical_calibration_accepted})
    # Inventory the retained source closure, not temporary reproductions.
    folders=['expanded_pilot_v2','field_metadata_v1','quote_review_v1','sofr_curves_v1','valuation_sources_v1',
             'valuation_inputs_v1','empirical_acceptance_probe_v2','contract_dividend_inputs_v1']
    files={p for folder in folders for p in (LSEG/folder).rglob('*') if p.is_file()}
    files.update(ROOT/n for n in ann['input_sha256'])
    files.update(ANNOUNCEMENTS/n for n in ann['artifact_sha256'])
    files.update([ANNOUNCEMENTS/'manifest.json',CONFIG,protocol,PROJECT/'configs/us_event_calendar_2026-09-16.json'])
    summary={'scope':'retrospective conditional private diagnostic','original_pairs':len(review['pairs']),
             'selected_rows':len(selected),'split_counts':dict(Counter(r['split'] for r in selected)),
             'selected_dates':sorted({r['observation_date'] for r in selected}),
             'exclusion_reasons':dict(Counter(reason for r in exclusions for reason in r['reasons'])),
             'scenario_rows':len(problems),'conditional_private_diagnostic_ready':True,
             'full_empirical_acceptance':False,'assumptions':cfg['assumptions']}
    output.mkdir(parents=True)
    for n,d in [('selected.json',selected),('exclusions.json',exclusions),('problems.json',problems),('audit.json',summary),('config.json',cfg)]:save(output/n,d)
    save(output/'manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
         'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in CODE},
         'artifact_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ('selected_rows','split_counts','exclusion_reasons','scenario_rows')}),flush=True)


def restore(p):
    model=dict(p['model']);model['dividends']=tuple(DividendCap(**d) for d in model['dividends'])
    return CalibrationQuote(p['observation_id'],p['split'],p['bid'],p['ask'],FundedCashModel(**model),
                            Option(**p['option']),tuple(FundedExercise(**d) for d in p['decisions']))


def fit(sample,output):
    sample,output=private(sample),private(output)
    if output.exists():raise FileExistsError('Use a fresh fit directory')
    manifest=read(sample/'manifest.json')
    for k in ('input_sha256','source_sha256'):verify_mapping(manifest[k])
    verify_mapping(manifest['artifact_sha256'],sample)
    cfg=read(sample/'config.json');raw=read(sample/'problems.json');selected={r['observation_id']:r for r in read(sample/'selected.json')}
    output.mkdir(parents=True)
    fits=[]; rows=[]; start=time.monotonic()
    for name in cfg['dividend_models']:
        records=[restore(p) for p in raw if p['dividend_model']==name]
        training=[r for r in records if r.split=='train']
        stages=[]
        for index,grid in enumerate(cfg['fit_grids']):
            result=fit_shared_volatility(training,cfg,grid)
            result['dividend_model']=name;stages.append(result);fits.append(result)
            save(output/'fits_partial.json',fits)
            print(f'Completed {name}, training grid {index+1}/{len(cfg["fit_grids"])}',flush=True)
        sigma=stages[-1]['best']['volatility']
        for record in records:
            levels=[price_quote(record,sigma,g,cfg['domain_multiple']) for g in [cfg['fit_grids'][-1],*cfg['validation_grids']]]
            domain_grid={'space_steps':cfg['validation_domain_space_steps'],'time_steps':cfg['validation_grids'][-1]['time_steps']}
            wide=price_quote(record,sigma,domain_grid,cfg['validation_domain_multiple'])
            tol=cfg['numerical_tolerance_fraction_of_full_spread']*(record.ask-record.bid)
            row={'observation_id':record.observation_id,'dividend_model':name,'split':record.split,
                 'date':selected[record.observation_id]['observation_date'],'kind':record.option.kind,
                 'volatility':sigma,'bid':record.bid,'ask':record.ask,
                 'diagnostics':quote_diagnostics(record,levels[-1]),'grid_prices':levels,'wider_domain_price':wide,
                 'last_space_time_change':levels[-1]-levels[-2], 'fit_grid_to_fine_change':levels[-1]-levels[0],
                 'domain_change':wide-levels[-1], 'numerical_tolerance':tol,
                 'refinement_screen_pass':max(abs(levels[-1]-levels[-2]),abs(wide-levels[-1]))<=tol,
                 'refit_grid_accuracy_screen_pass':abs(levels[-1]-levels[0])<=tol}
            rows.append(row)
        save(output/'rows_partial.json',rows)
        print(f'Completed {name}: training and held-out diagnostics with grid/domain checks',flush=True)
    summaries=[]
    for name in cfg['dividend_models']:
        for split in ('train','heldout'):
            subset=[r for r in rows if r['dividend_model']==name and r['split']==split]
            errors=[r['diagnostics']['midpoint_residual'] for r in subset]
            summaries.append({'dividend_model':name,'split':split,'rows':len(subset),'dates':len({r['date'] for r in subset}),
                'mean_error':float(np.mean(errors)),'mae':float(np.mean(np.abs(errors))),'rmse':float(np.sqrt(np.mean(np.square(errors)))),
                'inside_quote_intervals':sum(r['diagnostics']['inside_quote_interval'] for r in subset),
                'maximum_outside_full_spreads':max(r['diagnostics']['outside_quote_distance_in_full_spreads'] for r in subset),
                'refinement_failures':sum(not r['refinement_screen_pass'] for r in subset),
                'fit_grid_accuracy_failures':sum(not r['refit_grid_accuracy_screen_pass'] for r in subset)})
    save(output/'fits.json',fits);save(output/'price_diagnostics.json',rows);save(output/'summaries.json',summaries)
    report=['# Conditional October calibration diagnostic','',
            'Private source-valued results. Exploratory retrospective conditional fit; full empirical acceptance remains false.','',
            'All three dividend assumptions retained. Common volatility fitted to training dates only. Midpoint/width loss is not a statistical likelihood.','',
            '| Dividend model | Split | Rows | Dates | MAE | RMSE | Inside intervals | Refinement failures | Fit-grid failures |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:report.append(f'| {s["dividend_model"]} | {s["split"]} | {s["rows"]} | {s["dates"]} | {s["mae"]:.8f} | {s["rmse"]:.8f} | {s["inside_quote_intervals"]} | {s["refinement_failures"]} | {s["fit_grid_accuracy_failures"]} |')
    (output/'report.md').write_text('\n'.join(report)+'\n')
    save(output/'manifest.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{str((sample/p).relative_to(ROOT)):sha(sample/p) for p in ['manifest.json','config.json','problems.json','selected.json']},
         'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in CODE},
         'artifact_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
         'elapsed_seconds':time.monotonic()-start,'python':platform.python_version(),
         'numpy':np.__version__,'full_empirical_acceptance':False,'model_selection_on_heldout':False})
    print('Saved conditional fit and all diagnostics; no model selected using held-out data.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('fit');a.add_argument('--sample',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    else:fit(args.sample,args.output)
