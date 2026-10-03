"""Evaluate a fixed-sample component confidence procedure with certified radii.

The separate certification interpreter encloses the radii. Numerical results
are an optimistic zero-summary-error diagnostic, not a complete coverage claim.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from qf_research.optimal_stopping.component_confidence import region, decision_summary

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/private/component_confidence/2026-09-28'
PREVIOUS = ROOT/'data/private/allocation_variance_sensitivity/2026-09-28'
CERTIFIED = ROOT/'data/private/cpu_budget_transfer/2026-09-27'
PILOT = ROOT/'data/private/training_allocation/2026-09-27/pilot_v1'
LITERATURE = ROOT/'data/private/literature/component_confidence_2026-09-28'


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
        return [encode(x) for x in value]
    return value


def save(path, value):
    path.write_text(json.dumps(encode(value), indent=2, allow_nan=False)+'\n')


def private(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT/'data/private'):
        raise ValueError('Require private numerical storage')
    return path


def verify(mapping, anchor=ROOT):
    for name, digest in mapping.items():
        path = (anchor/name).resolve()
        if not path.is_relative_to(anchor.resolve()) or sha(path) != digest:
            raise ValueError('Changed or unsafe evidence: '+name)


def prepare(output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh input directory')
    evidence = {}
    for path in (PREVIOUS/'completion_v1.json', PREVIOUS/'audit_v1/checks.json',
                 CERTIFIED/'certified_residual_completion_v1.json'):
        value = read(path)
        if value['complete'] is not True:
            raise ValueError('Completed evidence required')
        for name, digest in value['evidence_sha256'].items():
            if name in evidence and evidence[name] != digest:
                raise ValueError('Conflicting evidence')
            evidence[name] = digest
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    old_inputs = read(PREVIOUS/'inputs_v1/inputs.json')
    models = []
    for old in old_inputs['models']:
        case = old['case']
        bound_path = CERTIFIED/f'certified_residual_run_v1/case_{case:03d}.json'
        bound = read(bound_path)
        if not all(bound[key] is True for key in ('coverage_complete', 'ideal_model_bound_certified')):
            raise ValueError('Full-domain certified bound required')
        pilot_path = PILOT/f'blocks/case_{case:03d}.json'
        pilot = read(pilot_path)
        if pilot['case'] != case or len(pilot['statistical']) != 32:
            raise ValueError('Pilot coverage differs')
        if any(r['evaluation']['approximation_fingerprint'] != bound['approximation_fingerprint']
               for r in pilot['statistical']):
            raise ValueError('Gain bound and statistical pilot approximations differ')
        result_path = PREVIOUS/f'run_v1/case_{case:02d}.json'
        result = read(result_path)
        full = result['reconstructed_full']
        for path in (bound_path, pilot_path, result_path):
            name = str(path.relative_to(ROOT))
            if name not in evidence or sha(path) != evidence[name]:
                raise ValueError('Missing verified evidence: '+name)
        models.append(dict(case=case, gain_bound=bound['gain_bound'],
             approximation_fingerprint=bound['approximation_fingerprint'],
             observed_variance=full['observed_variance'],
             evaluation_mean_variance=full['evaluation_mean_variance'],
             regimes=old['regimes']))
    if [m['case'] for m in models] != list(range(13)):
        raise ValueError('Expected all thirteen cases')
    for path in (Path(__file__).resolve(), ROOT/'src/qf_research/optimal_stopping/component_confidence.py',
                 ROOT/'scripts/certify_component_radii.py',
                 ROOT/'tests/test_component_confidence.py', BASE/'design_v1/protocol.md',
                 BASE/'theory_v1/fixed_sample_region.md', LITERATURE/'source_record.json', LITERATURE/'sources.md'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    output.mkdir(parents=True)
    save(output/'inputs.json', dict(cases=13, pilot_pairs=32, pilot_paths=4096,
         family_alpha='1/20', one_sided_tail_probability='1/1040', precision_bits=128,
         variance_summary_error='0', evaluation_summary_error='0',
         summary_error_status='Optimistic diagnostic only; no complete pilot summary-error certificate',
         models=models))
    save(output/'manifest.json', dict(complete=True, evidence_sha256=evidence,
         artifact_sha256={'inputs.json': sha(output/'inputs.json')}))


def run(inputs, output):
    inputs, output = private(inputs), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh result directory')
    source = read(inputs/'manifest.json')
    if source['complete'] is not True:
        raise ValueError('Incomplete inputs')
    verify(source['evidence_sha256']); verify(source['artifact_sha256'], inputs)
    payload = read(inputs/'inputs.json')
    m, n, k = payload['pilot_pairs'], payload['pilot_paths'], payload['cases']
    alpha, delta = Fraction(payload['family_alpha']), Fraction(payload['one_sided_tail_probability'])
    if (m, n, k) != (32, 4096, 13) or 4*k*delta != alpha or alpha != Fraction(1, 20):
        raise ValueError('Confidence design differs')
    output.mkdir(parents=True)
    records = []
    try:
        command = [str(Path.home()/'.venvs/qf-certification/bin/python'),
                   str(ROOT/'scripts/certify_component_radii.py'), '--pairs', str(m),
                   '--paths', str(n), '--tail-probability', str(delta),
                   '--precision-bits', str(payload['precision_bits']),
                   '--output', str(output/'radius_factors.json')]
        subprocess.run(command, check=True, capture_output=True, text=True)
        factors = read(output/'radius_factors.json')
        common_v = {k: Fraction(v) for k, v in factors['variance_factor'].items()}
        common_u = {k: Fraction(v) for k, v in factors['evaluation_factor'].items()}
        for model in payload['models']:
            case = model['case']; L = Fraction(model['gain_bound']); cap = L*L
            s, u = Fraction(model['observed_variance']), Fraction(model['evaluation_mean_variance'])
            rv, ru = cap*common_v['upper'], cap*common_u['upper']
            polygon = region(s, u, rv, ru, cap, pilot_paths=n, tail_probability=delta)
            if polygon['status'] != 'nondegenerate':
                raise ArithmeticError('Unexpected empty or zero-only diagnostic region')
            low, high = (polygon['share_bounds'][key] for key in ('lower', 'upper'))
            regimes = []
            for r in model['regimes']:
                analysis = decision_summary(r['candidates'], low, high, r['retained_choice']['evaluation_paths'])
                regimes.append(dict(**r, **analysis))
            record = dict(case=case, gain_bound=L, bound_squared=cap,
                 observed_variance=s, evaluation_mean_variance=u,
                 variance_radius=rv, evaluation_radius=ru,
                 variance_radius_to_observed=rv/s if s > 0 else None,
                 evaluation_radius_to_observed=ru/u if u > 0 else None,
                 region=polygon, regimes=regimes,
                 radii_rounded_outwards=True, complete_statistical_coverage_certified=False,
                 optimistic_zero_summary_error=True)
            save(output/f'case_{case:02d}.json', record)
            records.append(dict(case=case, share_bounds=polygon['share_bounds'],
                 variance_radius_to_observed=record['variance_radius_to_observed'],
                 evaluation_radius_to_observed=record['evaluation_radius_to_observed'],
                 uniquely_selected_regimes=sum(r['uniformly_selected_size'] is not None for r in regimes),
                 regimes=[dict(accounting=r['accounting'], scope=r['scope'],
                     selected_sizes=r['selected_sizes'],
                     maximum_retained_choice_regret=r['retained_choice_regret']['maximum']) for r in regimes]))
        save(output/'summary.json', dict(complete=True, cases=13, fixed_cost_regimes=78,
             nominal_family_confidence='19/20', scope='Ideal bounded independent-cluster procedure; optimistic zero-summary-error evaluation',
             certified_radius_enclosures=True, complete_pilot_summary_errors_certified=False,
             complete_statistical_coverage_certified=False, new_simulations=0, policy_refits=0,
             unique_allocation_regimes=sum(r['uniquely_selected_regimes'] for r in records), models=records,
             python=platform.python_version(), certification_python=factors['python'],
             python_flint=factors['python_flint'], publication_clearance=False))
    except Exception:
        save(output/'failure.json', dict(complete=False))
        raise
    evidence = dict(source['evidence_sha256'])
    for path in (inputs/'manifest.json', inputs/'inputs.json'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    save(output/'manifest.json', dict(complete=True, evidence_sha256=evidence,
         artifact_sha256={p.name: sha(p) for p in sorted(output.glob('*.json'))}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--inputs', type=Path, default=BASE/'inputs_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.output)
    else: run(args.inputs, args.output)


if __name__ == '__main__':
    main()
