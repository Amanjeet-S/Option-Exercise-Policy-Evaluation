"""Joint financial and expiry sensitivity with a prespecified maximal endpoint."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping
from study_expiry_extension import planned_inputs as expiry_plan, verified_results as expiry_results, YEAR
from study_policy_input_sensitivity import restore_scenario
from validate_funded_dual import load_frozen
from study_value_martingale import serialise_dual
from qf_research.computational.funded_kernel import NativeKernel, prepare
from qf_research.optimal_stopping.expiry_extension import extended_calendar, price_terminal_extension, european_increment
from qf_research.optimal_stopping.exercise_dominance import terminal_call_dominance
from qf_research.optimal_stopping.funded_policy import simulate_paths
from qf_research.optimal_stopping.value_martingale import build_values, observation_rights, value_martingale
from qf_research.optimal_stopping.joint_input_expiry import evaluate_joint_policy
from qf_research.optimal_stopping.policy_transfer import family_upper_limit
from qf_research.optimal_stopping.simulation import estimate_samples
from qf_research.optimal_stopping.cost_budget import independent_difference

OWN_SOURCES = ('scripts/study_joint_input_expiry.py',
               'src/qf_research/optimal_stopping/joint_input_expiry.py',
               'src/qf_research/optimal_stopping/cost_budget.py')


def planned_inputs(cfg):
    inherited, inputs, sources = expiry_plan()
    prior = ROOT/inherited[0]['configuration']['prior_run']
    old_tasks = {t['case']: t for t in read(prior/'tasks.json')}
    old_rows = {r['case']: r for r in read(prior/'rows.json')}
    expiry = ROOT/cfg['expiry_run']
    old_expiry, _ = expiry_results(expiry)
    verification = read(ROOT/cfg['expiry_verification'])
    if verification['experiment_manifest_sha256'] != sha(expiry/'manifest.json'):
        raise ValueError('Prior expiry verification differs')
    verify_mapping(verification['source_sha256'])
    sources.update(verification['source_sha256'])
    em = read(expiry/'manifest.json')
    inputs.update(em['input_sha256']); sources.update(em['source_sha256'])
    inputs.update({str((expiry/n).relative_to(ROOT)): v for n, v in em['artifact_sha256'].items()})
    for p in (expiry/'manifest.json', ROOT/cfg['expiry_verification']):
        inputs[str(p.relative_to(ROOT))] = sha(p)
    tasks = []
    for task, previous in zip(inherited, old_expiry):
        if task['case'] != previous['case']:
            raise ValueError('Model order changed')
        old, result = old_tasks[task['case']], old_rows[task['case']]
        if len(old['scenarios']) != cfg['family_size'] or old['configuration']['family_size'] != cfg['family_size']:
            raise ValueError('Financial family changed')
        if task['configuration']['extensions_seconds'] != cfg['covered_extensions_seconds']:
            raise ValueError('Expiry scenario family changed')
        tasks.append(dict(case=task['case'], scenarios=old['scenarios'], policy_rows=old['policy_rows'],
            old_financial_results=result['scenarios'], old_families=result['families'],
            library=task['library'], native_tolerance=task['native_tolerance'],
            previous_streams=sorted(set(task['previous_streams'])|{previous['stream']['stream_id']}),
            repeat_python=task['repeat_python'], configuration=cfg))
    for n in OWN_SOURCES:
        sources[n] = sha(ROOT/n)
    return tasks, inputs, sources


def quote_estimate(record, factor):
    return {k: v/factor if k in ('value', 'standard_error', 'lower', 'upper') else v for k, v in record.items()}


def evaluate(task):
    cfg = task['configuration']
    seconds = max(cfg['covered_extensions_seconds'])
    extra = tuple(s/YEAR for s in cfg['covered_extensions_seconds'] if s)
    entries = [(r['repetition'], p) for r in task['policy_rows'] for p in r['policies']]
    policies = [load_frozen(p)[0] for _, p in entries]
    native = NativeKernel(Path(task['library']))
    seed = np.random.SeedSequence(cfg['root_seed'], spawn_key=(task['case'],))
    rows, baseline_gaps, clock = [], [], None
    for index, (record, previous) in enumerate(zip(task['scenarios'], task['old_financial_results'])):
        if record['scenario'] != previous['scenario']:
            raise ValueError('Financial scenario order changed')
        common = dict(scenario=record['scenario'], fraction=record['fraction'], funding_path=record['funding_path'],
                      extension_seconds=seconds, original_calibration_status=record['fit']['status'])
        if record['fit']['status'] != 'conditional_numerical_witness':
            rows.append(dict(common, status='no_original_screened_construction', policies=[]))
            continue
        model, option, rights = restore_scenario(record)
        if previous['status'] != 'evaluated' or previous['premium_factor'] != model.premium_factor:
            raise ValueError('Prior scenario units or evaluation status changed')
        contract, calendar = extended_calendar(model, option, rights, extra)
        if option != policies[0].option or model.volatility <= 0:
            raise ValueError('Changed original contract or unsupported common-shock clock')
        if index == 0 and any(p.model != model or p.decisions != rights for p in policies):
            raise ValueError('Original baseline changed')
        fitted = record['fit']
        threshold = max(model.initial_state, *(option.strike*d.settlement_factor for d in rights))+sum(d.discounted_cash for d in model.dividends)
        prices, repeats, base_check = {}, {}, None
        for name, grid in cfg['meshes'].items():
            prepared = prepare(model, option, grid['space_steps'], grid['time_steps'],
                               grid['domain_multiple']*threshold, rights)
            prices[name] = price_terminal_extension(prepared, seconds/YEAR, native).premium_quote
            if name == 'fine':
                base_check = price_terminal_extension(prepared, 0., native).premium_quote
                tolerance = task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(base_check)
                if abs(base_check-fitted['put_prices'][1]) > tolerance:
                    raise ValueError('Original fine-grid price does not reproduce')
                # Prescribed baseline-financial scenarios at first and last local cases.
                if task['repeat_python'] and index == 0:
                    reference = price_terminal_extension(prepared, seconds/YEAR).premium_quote
                    tolerance = task['native_tolerance']['absolute']+task['native_tolerance']['relative']*abs(reference)
                    if abs(reference-prices[name]) > tolerance:
                        raise ValueError('Independent Python propagation differs')
                    repeats = dict(python_quote=reference, native_quote=prices[name], tolerance=tolerance)
        euro = european_increment(model, option, rights[-1], seconds/YEAR)
        call = european_increment(model, replace(option, kind='call'), rights[-1], seconds/YEAR)
        dominance = terminal_call_dominance(model, replace(contract, kind='call'), calendar)
        eta, tol = fitted['numerical_allowance'], fitted['call_tolerance']
        increment = prices['fine']-base_check
        screens = dict(joint_space_time=abs(prices['fine']-prices['base']) <= eta,
                       domain=abs(prices['wide']-prices['fine']) <= eta,
                       european_increment=-2*eta <= increment <= euro['increment_quote']+2*eta,
                       call_reference=(call['expanded'].quadrature_error_estimate+call['expanded'].omitted_tail_bound)/model.premium_factor <= tol)
        call_fits = bool(dominance.sufficient_condition_satisfied and
            fitted['call_bid'] <= call['expanded'].premium_quote-tol and call['expanded'].premium_quote+tol <= fitted['call_ask'])
        put_fits = bool(fitted['put_bid'] <= prices['fine']-eta and prices['fine']+eta <= fitted['put_ask'])
        start = time.perf_counter()
        approximation = build_values(model, contract, calendar, **cfg['value_approximation'])
        observations = observation_rights(approximation)
        current_clock = [(d.time, d.phase) for d in observations]
        if clock is not None and current_clock != clock:
            raise ValueError('Financial scenarios do not share one union observation clock')
        clock = current_clock
        draws = simulate_paths(model, contract, observations, cfg['evaluation_paths'], np.random.Generator(np.random.PCG64(seed)))
        if draws.stream_id in task['previous_streams']:
            raise ValueError('Prior random-stream role reused')
        sample = value_martingale(approximation, draws)
        shared_seconds = time.perf_counter()-start
        evaluations = []
        for j, ((rep, frozen), policy, old) in enumerate(zip(entries, policies, previous['policies'])):
            if frozen['fingerprint'] != old['source_fingerprint']:
                raise ValueError('Old and new policy identities differ')
            result, realised = evaluate_joint_policy(policy, sample, rights)
            gap = result.gap_payoffs/model.premium_factor
            if index == 0:
                baseline_gaps.append(gap.copy())
            if len(baseline_gaps) != len(entries):
                # During the baseline loop only the current prefix exists.
                if index != 0:
                    raise ValueError('A complete baseline is required')
            financial_effect = asdict(estimate_samples(gap-baseline_gaps[j]))
            old_gap = quote_estimate(old['result']['gap_bound'], model.premium_factor)
            new_gap = quote_estimate(asdict(result.gap_bound), model.premium_factor)
            evaluations.append(dict(repetition=rep, training_paths=policy.training_paths, degree=policy.requested_degree,
                fingerprint=policy.fingerprint, result=serialise_dual(result, model.premium_factor),
                joint_family_upper_quote=family_upper_limit(result.gap_bound, cfg['family_size'], cfg['family_error_probability'])/model.premium_factor,
                paired_financial_effect_quote=financial_effect,
                independent_expiry_gap_change_quote=independent_difference(new_gap, old_gap),
                interaction_quote=independent_difference(financial_effect, old['paired_gap_change']),
                stopping_counts=np.bincount(realised.stopping_indices, minlength=len(rights)).tolist()))
        rows.append(dict(common, status='evaluated', premium_factor=model.premium_factor,
            spread=fitted['put_ask']-fitted['put_bid'], prices=prices, original_fine_price=base_check,
            pde_increment_quote=increment, numerical_screens=screens,
            call_quote_with_margin=call_fits, put_quote_with_margin=put_fits,
            joint_quote_compatible=bool(call_fits and put_fits and all(screens.values())),
            terminal_call_condition=asdict(dominance), python_repeat=repeats,
            european={k: asdict(v) if k in ('baseline', 'expanded') else v for k,v in euro.items()},
            call={k: asdict(v) if k in ('baseline', 'expanded') else v for k,v in call.items()},
            approximation=asdict(approximation), approximation_fingerprint=approximation.fingerprint,
            stream_id=draws.stream_id, shared_seconds=shared_seconds, policies=evaluations))
    accepted = [r for r in rows if r['status'] == 'evaluated']
    if not accepted or accepted[0]['scenario'] != task['scenarios'][0]['scenario']:
        raise ValueError('Missing original financial baseline')
    complete = len(accepted) == cfg['family_size']
    compatible = complete and all(r['joint_quote_compatible'] for r in accepted)
    families = []
    for j, (rep, frozen) in enumerate(entries):
        bound, scenario = max((r['policies'][j]['joint_family_upper_quote'], r['scenario']) for r in accepted)
        old = task['old_families'][j]
        if old['source_fingerprint'] != frozen['fingerprint']:
            raise ValueError('Original family identity differs')
        spread = accepted[0]['spread']
        if any(r['spread'] != spread for r in accepted):
            raise ValueError('Comparison spread differs across financial scenarios')
        families.append(dict(repetition=rep, training_paths=frozen['training_paths'], degree=frozen['degree'],
            fingerprint=frozen['fingerprint'], full_family=complete, all_joint_quotes_compatible=compatible,
            maximum_upper_quote=bound, maximising_scenario=scenario, spread=spread,
            below_spread=complete and bound < spread,
            compatible_family_below_spread=compatible and bound < spread,
            old_financial_family_upper_quote=old['maximum_family_adjusted_upper_quote']))
    return dict(case=task['case'], stream=dict(entropy=seed.entropy, spawn_key=list(seed.spawn_key),
        stream_id=accepted[0]['stream_id'], paths=cfg['evaluation_paths']), scenarios=rows, families=families)


def numerical_record(row):
    return dict(row, scenarios=[{k:v for k,v in s.items() if k != 'shared_seconds'} for s in row['scenarios']])


def summarise(rows):
    if not rows or len({r['case'] for r in rows}) != len(rows) or len({r['stream']['stream_id'] for r in rows}) != len(rows):
        raise ValueError('Distinct cases and independent model streams required')
    scenarios = [s for r in rows for s in r['scenarios'] if s['status'] == 'evaluated']
    families = [f for r in rows for f in r['families']]
    variants = []
    for n, d in sorted({(f['training_paths'], f['degree']) for f in families}):
        selected = [f for f in families if (f['training_paths'], f['degree']) == (n,d)]
        variants.append(dict(training_paths=n, degree=d, policies=len(selected),
            complete_families=sum(f['full_family'] for f in selected),
            loss_bounds_below_spread=sum(f['below_spread'] for f in selected),
            compatible_families=sum(f['all_joint_quotes_compatible'] for f in selected),
            compatible_family_bounds_below_spread=sum(f['compatible_family_below_spread'] for f in selected),
            maximum_bound_range=[min(f['maximum_upper_quote'] for f in selected), max(f['maximum_upper_quote'] for f in selected)]))
    return dict(models=len(rows), evaluated_scenarios=len(scenarios),
        policy_evaluations=sum(len(s['policies']) for s in scenarios),
        joint_quote_compatible_scenarios=sum(s['joint_quote_compatible'] for s in scenarios),
        numerical_failures=sum(not all(s['numerical_screens'].values()) for s in scenarios),
        pde_increment_range=[min(s['pde_increment_quote'] for s in scenarios), max(s['pde_increment_quote'] for s in scenarios)],
        families=variants, refit=False, empirical_calibration_accepted=False, publication_clearance=False)


def register(configuration, protocol, output):
    configuration, protocol, output = map(private, (configuration, protocol, output))
    if output.exists(): raise FileExistsError('Use a fresh registration directory')
    cfg = read(configuration)
    if (cfg['covered_extensions_seconds'] != [0,900,5400] or cfg['family_size'] != 9
            or cfg['evaluation_paths'] != 100000 or cfg['root_seed'] != 202609261429
            or cfg['family_error_probability'] != .05 or cfg['publication_clearance']
            or cfg['empirical_calibration_accepted']):
        raise ValueError('Unsupported registered design')
    tasks, inputs, sources = planned_inputs(cfg)
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in (configuration, protocol)})
    verify_mapping(inputs); verify_mapping(sources)
    output.mkdir(parents=True)
    save(output/'configuration.json', cfg); save(output/'tasks.json', tasks)
    (output/'protocol.md').write_text(protocol.read_text())
    save(output/'manifest.json', dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=inputs, source_sha256=sources,
        artifact_sha256={n:sha(output/n) for n in ('configuration.json','tasks.json','protocol.md')},
        public_registration=False, publication_clearance=False))


def run(registration, output, workers):
    registration, output = private(registration), private(output)
    if output.exists(): raise FileExistsError('Use a fresh experiment directory')
    if workers not in (1,2): raise ValueError('Use one or two memory-bounded workers')
    m = read(registration/'manifest.json')
    for k in ('input_sha256','source_sha256'): verify_mapping(m[k])
    verify_mapping(m['artifact_sha256'], registration)
    tasks = read(registration/'tasks.json')
    output.mkdir(parents=True)
    save(output/'launch.json', dict(started_at=datetime.now(timezone.utc).isoformat(), workers=workers,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        registration=str(registration.relative_to(ROOT)), registration_sha256=sha(registration/'manifest.json')))
    start = time.monotonic(); rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate, tasks, chunksize=1):
            rows.append(row); save(output/'partial_rows.json', rows)
            print(f'Completed {len(rows)}/{len(tasks)} joint financial/expiry families', flush=True)
    for k in ('input_sha256','source_sha256'): verify_mapping(m[k])
    save(output/'rows.json', rows); save(output/'summary.json', summarise(rows))
    save(output/'manifest.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start, registration=str(registration.relative_to(ROOT)),
        registration_sha256=sha(registration/'manifest.json'), input_sha256=m['input_sha256'], source_sha256=m['source_sha256'],
        artifact_sha256={n:sha(output/n) for n in ('launch.json','rows.json','summary.json')}, publication_clearance=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command', required=True)
    p=commands.add_parser('register')
    for name in ('configuration','protocol','output'): p.add_argument('--'+name,type=Path,required=True)
    p=commands.add_parser('run'); p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.command=='register': register(args.configuration,args.protocol,args.output)
    else: run(args.registration,args.output,args.workers)
