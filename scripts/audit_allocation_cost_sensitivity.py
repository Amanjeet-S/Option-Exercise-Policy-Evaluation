"""Independently audit retrospective allocation-cost sensitivity.

The audit reconstructs financial planning inputs and measured timer means,
then exhaustively enumerates the restricted integer allocation grid. It does
not import the sensitivity kernel or runner, fit a policy or simulate paths.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
from itertools import groupby
import hashlib
import json
import math
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
BASE = PRIVATE / 'allocation_cost_sensitivity/2026-09-27'
ORIGINAL = PRIVATE / 'training_allocation/2026-09-27'
TIMING = PRIVATE / 'cpu_budget_transfer/2026-09-27'
GRID = (128, 256, 4096)
MINIMUM, MAXIMUM = 64, 1024
SCOPES = ('pilot', 'validation', 'joint')
ACCOUNTING = ('marginal', 'screening_charged')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=invalid)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Keep evidence under data/private')
    return path


def binary64(value):
    require(type(value) in (float, int) and math.isfinite(value), 'A finite original scalar is required')
    return Fraction(value)


def rational(value):
    require(isinstance(value, str), 'A canonical rational string is required')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational string')
    return result


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def same(actual, expected, location):
    expected = encode(expected)
    require(type(actual) is type(expected), 'Independent record type differs: ' + location)
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Independent record keys differ: ' + location)
        for key, value in expected.items():
            same(actual[key], value, location + '.' + key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Independent record length differs: ' + location)
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, location + '[' + str(index) + ']')
    else:
        require(actual == expected, 'Independent reconstruction differs: ' + location)


def row_for(A, B, n, r, costs, overhead):
    scenario_costs = {name: overhead + values[n]['setup'] + r*values[n]['pair']
                      for name, values in costs.items()}
    return dict(evaluation_paths=n, repetitions=r, variance=(A + B/n)/r,
                scenario_costs=scenario_costs, required_budget=max(scenario_costs.values()))


def ordering(row):
    return row['variance'], row['evaluation_paths'], row['repetitions']


def exhaustive(A, B, budget, costs, overhead, retained_choice):
    """Use every positive count to check candidates and the whole permitted grid."""
    require(set(retained_choice) == {'status', 'budget', 'overhead', 'selected', 'candidates'},
            'Unexpected choice record fields')
    require([row['evaluation_paths'] for row in retained_choice['candidates']] == list(GRID),
            'Candidate evaluation-grid coverage differs')
    candidates, permitted = [], []
    for n, retained in zip(GRID, retained_choice['candidates']):
        all_counts = [row_for(A, B, n, r, costs, overhead) for r in range(1, MAXIMUM+1)]
        permitted.extend(row for row in all_counts if row['repetitions'] >= MINIMUM)
        feasible_counts = [row for row in all_counts if row['required_budget'] <= budget]
        best_count = feasible_counts[-1] if feasible_counts else None
        r = best_count['repetitions'] if best_count else 0
        limits = retained['unconstrained_integer_limits']
        require(set(limits) == set(costs), 'Integer-limit scenario coverage differs')
        for name, values in costs.items():
            limit = limits[name]
            require(type(limit) is int, 'An exact integer limit is required')
            # This identifies the unique integer limit without dividing and flooring.
            left = overhead + values[n]['setup'] + limit*values[n]['pair']
            right = left + values[n]['pair']
            require(left <= budget < right, 'The reported integer limit violates an adjacent-cost inequality')
        capped = row_for(A, B, n, MAXIMUM+1, costs, overhead)['required_budget'] <= budget
        expected = dict(evaluation_paths=n, repetitions=r, feasible=r >= MINIMUM,
                        unconstrained_integer_limits=limits, capped=capped)
        expected.update(best_count if best_count is not None else
                        dict(variance=None, scenario_costs=None, required_budget=None))
        same(retained, expected, 'per-size integer candidate')
        candidates.append(expected)
    feasible = [row for row in permitted if row['required_budget'] <= budget]
    optimum = min(feasible, key=ordering) if feasible else None
    selected = next((row for row in candidates if optimum is not None
                     and (row['evaluation_paths'], row['repetitions']) ==
                     (optimum['evaluation_paths'], optimum['repetitions'])), None)
    expected_choice = dict(status='selected' if optimum else 'no_feasible_allocation',
        budget=budget, overhead=overhead, selected=selected, candidates=candidates)
    same(retained_choice, expected_choice, 'exhaustive optimum')
    frontier, best = [], None
    # Group simultaneous thresholds before deciding whether the optimum changes.
    ordered = sorted(permitted, key=lambda row: row['required_budget'])
    for _, batch in groupby(ordered, key=lambda row: row['required_budget']):
        entrant = min(batch, key=ordering)
        if best is None or ordering(entrant) < ordering(best):
            frontier.append(dict(**entrant, strict_variance_improvement=
                                 best is None or entrant['variance'] < best['variance']))
            best = entrant
    accessible = [row for row in frontier if row['required_budget'] <= budget]
    require(bool(accessible) == (optimum is not None), 'Frontier and direct feasible-set coverage differ')
    if optimum is not None:
        require(ordering(accessible[-1]) == ordering(optimum), 'Frontier and exhaustive optimum differ')
    return expected_choice, frontier, len(permitted), len(feasible)


def cross_evaluation(A, B, n, r, budget, costs, overhead):
    result = row_for(A, B, n, r, costs, overhead)
    values = result['scenario_costs']
    result.update(scenario_feasible={name: cost <= budget for name, cost in values.items()},
        scenario_budget_excess={name: max(Fraction(0), cost-budget) for name, cost in values.items()},
        scenario_cost_variance={name: cost*result['variance'] for name, cost in values.items()},
        jointly_feasible=all(cost <= budget for cost in values.values()))
    return result


def audit(inputs, run, output):
    inputs, run, output = map(private, (inputs, run, output))
    require(not output.exists() and not output.is_relative_to(inputs) and not output.is_relative_to(run),
            'Use a fresh, separate audit directory')
    evidence = {}

    def retain(path, expected=None):
        path = path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), 'Unsafe or missing evidence path')
        digest = sha(path)
        require(expected is None or digest == expected, 'Evidence checksum differs: ' + str(path.relative_to(ROOT)))
        key = str(path.relative_to(ROOT))
        require(key not in evidence or evidence[key] == digest, 'Conflicting evidence checksum')
        evidence[key] = digest
        return path

    def load_manifest(folder):
        record = read(retain(folder / 'manifest.json'))
        require(record.get('complete') is True and not (folder / 'incomplete.json').exists(),
                'A complete sensitivity stage is required')
        for group, base in (('evidence_sha256', ROOT), ('artifact_sha256', folder)):
            require(isinstance(record[group], dict) and record[group], 'A non-empty evidence map is required')
            for name, digest in record[group].items():
                path = (base / name).resolve()
                require(path.is_relative_to(base.resolve()), 'An evidence path escapes its declared root')
                retain(path, digest)
        return record

    input_manifest, run_manifest = load_manifest(inputs), load_manifest(run)
    require(set(input_manifest['artifact_sha256']) == {'inputs.json'}
            and {p.name for p in inputs.glob('*.json')} == {'inputs.json', 'manifest.json'},
            'Input file coverage differs')
    expected_files = {'summary.json'}
    for case in range(13):
        expected_files.add(f'case_{case:02d}.json')
        expected_files.update(f'case_{case:02d}_{account}_frontiers.json' for account in ACCOUNTING)
    require(set(run_manifest['artifact_sha256']) == expected_files
            and {str(p.relative_to(run)) for p in run.rglob('*.json')} == expected_files | {'manifest.json'},
            'Complete run artifact coverage differs')
    expected_closure = dict(input_manifest['evidence_sha256'])
    expected_closure[str((inputs / 'manifest.json').relative_to(ROOT))] = sha(inputs / 'manifest.json')
    expected_closure[str((inputs / 'inputs.json').relative_to(ROOT))] = sha(inputs / 'inputs.json')
    require(run_manifest['evidence_sha256'] == expected_closure, 'The run evidence closure differs from its input stage')
    for path in (ROOT / 'scripts/analyse_allocation_cost_sensitivity.py',
                 ROOT / 'src/qf_research/optimal_stopping/allocation_sensitivity.py'):
        require(str(path.relative_to(ROOT)) in input_manifest['evidence_sha256'],
                'The original runner and kernel must be bound by the input manifest')

    def bound_read(path):
        path = path.resolve()
        require(str(path.relative_to(ROOT)) in evidence, 'A consumed source record is absent from the evidence closure')
        return read(path)

    payload = read(inputs / 'inputs.json')
    configuration = bound_read(BASE / 'design_v1/configuration.json')
    require(payload['schema'] == 1 and payload['retrospective'] is True
            and payload['configuration'] == configuration, 'Input schema, scope or design differs')
    require(configuration['cases'] == list(range(13)) and configuration['evaluation_grid'] == list(GRID)
            and configuration['minimum_repetitions'] == MINIMUM
            and configuration['maximum_repetitions'] == MAXIMUM, 'The restricted allocation grid differs')
    originals = bound_read(ORIGINAL / 'main_registration_v1/allocations.json')['models']
    pilot_originals = bound_read(ORIGINAL / 'pilot_v1/summary.json')['models']
    require(originals == pilot_originals and [r['case'] for r in originals] == list(range(13)),
            'Original financial estimates or registered allocations differ')
    old_tasks = bound_read(ORIGINAL / 'main_registration_v1/tasks.json')
    local_tasks = bound_read(TIMING / 'local_timing_registration_v1/tasks.json')
    require([r['case'] for r in old_tasks] == [r['case'] for r in local_tasks] == list(range(13))
            and [r['source_policy'] for r in old_tasks] == [r['source_policy'] for r in local_tasks],
            'The financial cases differ between cost and financial studies')
    local_schedule = bound_read(TIMING / 'local_timing_registration_v1/schedule.json')
    require(local_schedule == [dict(position=i, case=5*i % 13, evaluation_paths=GRID[i % 3]) for i in range(39)],
            'The local timing schedule differs')
    locations = {(r['case'], r['evaluation_paths']): r['position'] for r in local_schedule}
    require([row['case'] for row in payload['models']] == list(range(13)), 'All thirteen input cases are required')
    reconstructed_models = []
    raw_timing_records = 0
    for original, retained in zip(originals, payload['models']):
        case, allocation = original['case'], original['allocation']
        stat = original['pilot_inference']
        A = binary64(stat['training_variance_moment_estimate'])
        B = binary64(stat['evaluation_path_variance'])
        require(A == binary64(allocation['raw_variance_components']['training'])
                and B == binary64(allocation['raw_variance_components']['evaluation']),
                'Original financial variance fields disagree')
        require(A > 0 and B > 0, 'The retained thirteen-case audit requires resolved positive variance inputs')
        budget = binary64(allocation['budget']['cpu_seconds'])
        scenarios = {name: {} for name in ('pilot', 'validation')}
        screening = Fraction(0)
        for n in GRID:
            position = locations[case, n]
            folder = TIMING / 'local_timing_run_v1' / f'cell_{position:02d}'
            for phase, count in (('pilot', 8), ('validation', 64)):
                setup = binary64(bound_read(folder / f'{phase}_setup.json')['cpu_seconds'])
                charges = []
                for repetition in range(count):
                    row = bound_read(folder / f'{phase}_{repetition:03d}.json')
                    require((row['case'], row['position'], row['evaluation_paths'], row['phase'], row['repetition']) ==
                            (case, position, n, phase, repetition), 'Raw timing observation identity differs')
                    require(row['pair_cpu_seconds'] == row['training']['cpu_seconds'] + row['evaluation']['cpu_seconds'],
                            'Raw complete-pair timer sum differs')
                    charge = binary64(row['pair_cpu_seconds'])
                    require(setup >= 0 and charge > 0, 'Invalid measured setup or pair charge')
                    charges.append(charge)
                    raw_timing_records += 1
                total = sum(charges, Fraction(0))
                scenarios[phase][n] = dict(setup=setup, pair=total/count)
                if phase == 'pilot':
                    screening += setup + total
        fixed = {name: {key: allocation[name][key] for key in ('evaluation_paths', 'repetitions')}
                 for name in ('reference', 'selected')}
        require(all(row['evaluation_paths'] in GRID and type(row['repetitions']) is int
                    and MINIMUM <= row['repetitions'] <= MAXIMUM for row in fixed.values()),
                'A frozen original allocation is outside this sensitivity grid')
        expected = dict(case=case, A=A, B=B, budget=budget, original=fixed, scenarios=scenarios,
                        screening_charge=screening, variance_estimates_resolved=True)
        same(retained, expected, 'retained financial inputs, measured costs and full screening charge')
        reconstructed_models.append(expected)

    summary = read(run / 'summary.json')
    require(summary['complete'] is True and summary['cases'] == 13 and summary['retrospective'] is True
            and summary['new_financial_simulations'] == 0 and summary['new_financial_estimates'] == 0
            and summary['frozen_original_budgets'] is True and summary['frontier_translation_verified'] is True,
            'Summary interpretation or completion flags differ')
    require([row['case'] for row in summary['models']] == list(range(13)), 'Summary case coverage differs')
    checks, enumeration_count, frontier_count, tie_changes = [], 0, 0, 0
    for model, summary_case in zip(reconstructed_models, summary['models']):
        case, A, B, budget, costs = (model[key] for key in ('case', 'A', 'B', 'budget', 'scenarios'))
        retained = read(run / f'case_{case:02d}.json')
        require(retained == summary_case and retained['case'] == case and retained['status'] == 'resolved'
                and set(retained['accounting']) == set(ACCOUNTING), 'Case output identity differs')
        account_checks, account_frontiers = {}, {}
        for account in ACCOUNTING:
            H = Fraction(0) if account == 'marginal' else model['screening_charge']
            actual = retained['accounting'][account]
            choices, frontiers, scope_checks = {}, {}, {}
            artifact_curves = read(run / f'case_{case:02d}_{account}_frontiers.json')
            require(set(artifact_curves) == set(SCOPES) and set(actual['choices']) == set(SCOPES),
                    'Scenario-scope coverage differs')
            for scope in SCOPES:
                active = costs if scope == 'joint' else {scope: costs[scope]}
                choice, curve, total, feasible = exhaustive(A, B, budget, active, H, actual['choices'][scope])
                same(artifact_curves[scope], curve, 'complete frontier including simultaneous thresholds and ties')
                choices[scope], frontiers[scope] = choice, curve
                enumeration_count += total
                frontier_count += len(curve)
                tied = sum(not row['strict_variance_improvement'] for row in curve)
                tie_changes += tied
                scope_checks[scope] = dict(enumerated_allocations=total, feasible_allocations=feasible,
                    frontier_changes=len(curve), tie_only_frontier_changes=tied,
                    selected=None if choice['selected'] is None else
                    {key: choice['selected'][key] for key in ('evaluation_paths', 'repetitions', 'variance')})
            cross = {}
            for name, old in model['original'].items():
                cross['original_' + name] = cross_evaluation(A, B, old['evaluation_paths'], old['repetitions'], budget, costs, H)
            for scope, choice in choices.items():
                chosen = choice['selected']
                cross[scope + '_choice'] = None if chosen is None else cross_evaluation(
                    A, B, chosen['evaluation_paths'], chosen['repetitions'], budget, costs, H)
            joint = choices['joint']['selected']
            penalties = {scope: (joint['variance']/choices[scope]['selected']['variance']
                                if joint is not None and choices[scope]['selected'] is not None else None)
                         for scope in ('pilot', 'validation')}
            require(all(value is None or value >= 1 for value in penalties.values()),
                    'Intersection feasibility cannot improve the single-scenario optimum')
            matched_reference = next(row for row in choices['joint']['candidates']
                                     if row['evaluation_paths'] == 4096)
            matched_ratio = (joint['variance']/matched_reference['variance']
                             if joint is not None and matched_reference['feasible'] else None)
            require(matched_ratio is None or matched_ratio <= 1,
                    'The joint optimum cannot exceed a feasible budget-matched reference variance')
            expected_account = dict(overhead=H, choices=choices, cross_evaluation=cross,
                joint_to_single_scenario_variance_ratio=penalties,
                joint_to_budget_matched_reference_variance_ratio=matched_ratio,
                joint_to_original_reference_variance_ratio=
                    joint['variance']/cross['original_reference']['variance'] if joint is not None else None)
            same(actual, expected_account, 'cross-evaluations, costs, feasibility and variance penalties')
            account_checks[account], account_frontiers[account] = scope_checks, frontiers
        H = model['screening_charge']
        for scope in SCOPES:
            original_curve = account_frontiers['marginal'][scope]
            charged_curve = account_frontiers['screening_charged'][scope]
            require(len(original_curve) == len(charged_curve), 'Screening translation changes frontier length')
            for before, after in zip(original_curve, charged_curve):
                require(all(before[key] == after[key] for key in
                            ('evaluation_paths', 'repetitions', 'variance', 'strict_variance_improvement')),
                        'Screening translation changes the frontier allocation')
                require(after['required_budget'] - before['required_budget'] == H
                        and all(after['scenario_costs'][name] - before['scenario_costs'][name] == H
                                for name in before['scenario_costs']), 'Screening costs do not translate exactly')
        checks.append(dict(case=case, accounting=account_checks, exact_frontier_translation_verified=True))
        print(f'Case {case}: raw inputs and exhaustive allocation frontiers verified', flush=True)
    require(raw_timing_records == 39*(8+64) and enumeration_count == 13*2*3*3*(MAXIMUM-MINIMUM+1),
            'Independent reconstruction coverage differs')
    retain(Path(__file__))
    for name, digest in evidence.items():
        require(sha(ROOT/name) == digest, 'Evidence changed during allocation audit')
    result = dict(complete=True, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        method='Independent standard-library Fraction reconstruction and exhaustive integer enumeration; no sensitivity kernel or runner imported.',
        cases=13, accounting_conventions=2, scopes_per_convention=3,
        raw_timing_pair_records_reconstructed=raw_timing_records,
        permitted_allocations_enumerated=enumeration_count, frontier_changes_verified=frontier_count,
        tie_only_frontier_changes_verified=tie_changes, original_variance_estimates_and_budgets_unchanged=True,
        original_financial_inputs_reconstructed=True, observed_cost_scenarios_and_screening_charges_reconstructed=True,
        all_choices_candidates_caps_and_cross_evaluations_verified=True,
        all_frontiers_and_translations_verified=True, exact_artifact_coverage_verified=True,
        cases_details=checks, retrospective=True, future_runtime_guarantee=False,
        financial_inference_performed=False, policies_refitted=0, paths_simulated=0,
        original_evidence_unchanged=True, python=platform.python_version(), publication_clearance=False,
        evidence_sha256=evidence, artifact_sha256={})
    output.mkdir(parents=True, exist_ok=False)
    pending = output / 'checks.pending.json'
    pending.write_text(json.dumps(encode(result), indent=2, allow_nan=False) + '\n')
    for name, digest in evidence.items():
        require(sha(ROOT/name) == digest, 'Evidence changed before audit completion')
    pending.rename(output / 'checks.json')
    return {key: result[key] for key in ('complete', 'cases', 'permitted_allocations_enumerated',
        'frontier_changes_verified', 'tie_only_frontier_changes_verified')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.inputs, args.run, args.output), sort_keys=True))
