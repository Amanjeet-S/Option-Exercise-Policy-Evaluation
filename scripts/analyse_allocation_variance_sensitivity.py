"""Analyse component geometry and whole-pair influence from retained evidence."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from qf_research.optimal_stopping.allocation_variance_sensitivity import (
    envelope, moments, region, regret, select, share,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/private/allocation_variance_sensitivity/2026-09-28'
PREVIOUS = ROOT/'data/private/allocation_cost_sensitivity/2026-09-27'
PILOT = ROOT/'data/private/training_allocation/2026-09-27/pilot_v1'
SCOPES = ('pilot', 'validation', 'joint')
ACCOUNTS = ('marginal', 'screening_charged')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False)+'\n')


def private(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT/'data/private'):
        raise ValueError('Keep numerical evidence in private storage')
    return path


def verify(mapping, anchor=ROOT):
    for name, expected in mapping.items():
        target = (anchor/name).resolve()
        if not target.is_relative_to(anchor.resolve()) or sha(target) != expected:
            raise ValueError('Changed or unsafe evidence: '+name)


def component_record(means, errors, omitted=None):
    retained = [i for i in range(len(means)) if i != omitted]
    values = moments([means[i] for i in retained], [errors[i] for i in retained], 4096)
    a, b = values['A'], values['B']
    status = ('unresolved_negative' if min(a, b) < 0 else
              'degenerate_zero' if a+b == 0 else
              'boundary' if min(a, b) == 0 else 'positive')
    return dict(omitted_pair=omitted, pairs=len(retained), **values, status=status,
                share=share(a, b) if status in ('positive', 'boundary') else None)


def prepare(output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh input directory')
    previous_completion = PREVIOUS/'completion_v1.json'
    completion = read(previous_completion)
    if completion['complete'] is not True:
        raise ValueError('Prior stage is incomplete')
    evidence = dict(completion['evidence_sha256'])
    evidence[str(previous_completion.relative_to(ROOT))] = sha(previous_completion)
    prior_audit = read(PREVIOUS/'audit_v1/checks.json')
    if prior_audit['complete'] is not True:
        raise ValueError('Prior audit is incomplete')
    for name, digest in prior_audit['evidence_sha256'].items():
        if name in evidence and evidence[name] != digest:
            raise ValueError('Conflicting evidence')
        evidence[name] = digest
    verify(evidence)
    previous = read(PREVIOUS/'inputs_v1/inputs.json')
    models = []
    for old in previous['models']:
        case = old['case']
        block_path = PILOT/f'blocks/case_{case:03d}.json'
        if str(block_path.relative_to(ROOT)) not in evidence:
            raise ValueError('Pilot block lacks retained provenance')
        block = read(block_path)
        rows = block['statistical']
        if block['case'] != case or [r['repetition'] for r in rows] != list(range(32)):
            raise ValueError('Pilot pair coverage differs')
        estimates = [r['evaluation']['gain']['estimate'] for r in rows]
        if any(e['observations'] != 4096 for e in estimates):
            raise ValueError('Pilot path count differs')
        pairs = [dict(pair=i, mean=Fraction(e['value']), standard_error=Fraction(e['standard_error']))
                 for i, e in enumerate(estimates)]
        prior_result = read(PREVIOUS/f'run_v1/case_{case:02d}.json')
        regimes = []
        for account in ACCOUNTS:
            for scope in SCOPES:
                choice = prior_result['accounting'][account]['choices'][scope]
                candidates = [{k: row[k] for k in ('evaluation_paths', 'repetitions')}
                              for row in choice['candidates'] if row['feasible']]
                regimes.append(dict(accounting=account, scope=scope, candidates=candidates,
                                    retained_choice={k: choice['selected'][k]
                                                     for k in ('evaluation_paths', 'repetitions')}))
        models.append(dict(case=case, pairs=pairs, retained_A=old['A'], retained_B=old['B'],
                           budget=old['budget'], regimes=regimes))
    if [m['case'] for m in models] != list(range(13)):
        raise ValueError('Expected all thirteen models')
    for path in (Path(__file__).resolve(), ROOT/'src/qf_research/optimal_stopping/allocation_variance_sensitivity.py',
                 ROOT/'tests/test_allocation_variance_sensitivity.py', BASE/'design_v1/protocol.md',
                 BASE/'theory_v1/component_geometry.md'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    output.mkdir(parents=True)
    save(output/'inputs.json', dict(models=models, reference_paths=4096, pilot_pairs=32,
                                   retrospective=True, coverage_claim=False))
    save(output/'manifest.json', dict(complete=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
         evidence_sha256=evidence, artifact_sha256={'inputs.json': sha(output/'inputs.json')}))


def run(inputs, output):
    inputs, output = private(inputs), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh analysis directory')
    manifest = read(inputs/'manifest.json')
    if manifest['complete'] is not True:
        raise ValueError('Incomplete inputs')
    verify(manifest['evidence_sha256'])
    verify(manifest['artifact_sha256'], inputs)
    summary = []
    payload = read(inputs/'inputs.json')
    output.mkdir(parents=True)
    try:
        for model in payload['models']:
            case = model['case']
            means = [Fraction(p['mean']) for p in model['pairs']]
            errors = [Fraction(p['standard_error']) for p in model['pairs']]
            full = component_record(means, errors)
            deletions = [component_record(means, errors, i) for i in range(32)]
            a, b = Fraction(model['retained_A']), Fraction(model['retained_B'])
            discrepancies = {key: full[key]-original for key, original in (('A', a), ('B', b))}
            if any(not math.isclose(float(full[k]), float(v), rel_tol=2e-12, abs_tol=2e-14)
                   for k, v in (('A', a), ('B', b))):
                raise ArithmeticError('Full reconstruction differs beyond round-off tolerance')
            original_share = share(a, b)
            admissible = [d['share'] for d in deletions if d['share'] is not None]
            diagnostic = [original_share]+admissible+([full['share']] if full['share'] is not None else [])
            lower, upper = min(diagnostic), max(diagnostic)
            regimes = []
            for source in model['regimes']:
                candidates = source['candidates']
                baseline = select(candidates, original_share)
                if any(baseline[k] != source['retained_choice'][k] for k in ('evaluation_paths', 'repetitions')):
                    raise ArithmeticError('Retained allocation does not match frozen components')
                choices = [dict(omitted_pair=d['omitted_pair'], status=d['status'],
                                choice=select(candidates, d['share']) if d['share'] is not None else None)
                           for d in deletions]
                changes = [d['omitted_pair'] for d in choices if d['choice'] is not None and
                           d['choice']['evaluation_paths'] != baseline['evaluation_paths']]
                target = baseline['evaluation_paths']
                regimes.append(dict(**source, retained_share_choice=baseline,
                    reconstructed_full_choice=select(candidates, full['share']) if full['share'] is not None else None,
                    envelope=envelope(candidates), retained_choice_region=region(candidates, target),
                    deletion_choices=choices, changed_deletion_pairs=changes,
                    diagnostic_hull=dict(lower=lower, upper=upper,
                        coverage_claim=False, unresolved_deletions=32-len(admissible)),
                    retained_choice_regret_over_diagnostic_hull=regret(candidates, target, lower, upper),
                    retained_choice_regret_over_nonnegative_components=regret(candidates, target, 0, 1)))
            save(output/f'case_{case:02d}.json', dict(case=case, retained_A=a, retained_B=b,
                 retained_share=original_share, reconstructed_full=full, reconstruction_difference=discrepancies,
                 deletions=deletions, regimes=regimes))
            summary.append(dict(case=case, retained_share=original_share,
                 deletion_statuses={status: sum(d['status'] == status for d in deletions)
                                    for status in ('positive', 'boundary', 'unresolved_negative', 'degenerate_zero')},
                 diagnostic_share_range=[lower, upper],
                 regimes=[dict(accounting=r['accounting'], scope=r['scope'],
                               retained_choice=r['retained_choice'],
                               changed_deletions=len(r['changed_deletion_pairs']),
                               diagnostic_maximum_regret=r['retained_choice_regret_over_diagnostic_hull']['maximum'],
                               nonnegative_components_maximum_regret=r['retained_choice_regret_over_nonnegative_components']['maximum'])
                          for r in regimes]))
        save(output/'summary.json', dict(complete=True, cases=13, reconstructed_pairs=416,
             delete_one_diagnostics=416, fixed_cost_regimes=78, models=summary,
             new_simulations=0, fitted_policies=0, confidence_regions_constructed=0,
             retrospective=True, costs_and_original_allocations_unchanged=True, publication_clearance=False))
    except Exception:
        save(output/'failure.json', dict(complete=False))
        raise
    evidence = dict(manifest['evidence_sha256'])
    evidence[str((inputs/'manifest.json').relative_to(ROOT))] = sha(inputs/'manifest.json')
    evidence[str((inputs/'inputs.json').relative_to(ROOT))] = sha(inputs/'inputs.json')
    verify(evidence)
    artifacts = {p.name: sha(p) for p in sorted(output.glob('*.json'))}
    save(output/'manifest.json', dict(complete=True, evidence_sha256=evidence, artifact_sha256=artifacts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--inputs', type=Path, default=BASE/'inputs_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
    else:
        run(args.inputs, args.output)


if __name__ == '__main__':
    main()
