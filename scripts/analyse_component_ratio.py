"""Retrospective ratio inference from separate retained training-pair samples."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from qf_research.optimal_stopping.component_ratio import (
    allocation_choices, fieller_coefficients, moment_covariance, quadratic_set,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'data/private/component_ratio/2026-09-28'
OLD = ROOT/'data/private/training_allocation/2026-09-27'
GEOMETRY = ROOT/'data/private/allocation_variance_sensitivity/2026-09-28'
PREVIOUS = ROOT/'data/private/component_confidence/2026-09-28'
LITERATURE = ROOT/'data/private/literature/component_ratio_2026-09-28'
SOURCES = ('pilot', 'reference', 'selected')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
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


def merge(evidence, mapping, anchor=ROOT):
    verify(mapping, anchor)
    for name, digest in mapping.items():
        relative = str((anchor/name).resolve().relative_to(ROOT))
        if relative in evidence and evidence[relative] != digest:
            raise ValueError('Conflicting evidence: '+relative)
        evidence[relative] = digest


def prepare(output):
    output = private(output)
    if output.exists():
        raise FileExistsError('Use a fresh input directory')
    evidence = {}
    previous = PREVIOUS/'completion_v1.json'
    completed = read(previous)
    if completed['complete'] is not True:
        raise ValueError('Previous stage is incomplete')
    merge(evidence, completed['evidence_sha256'])
    evidence[str(previous.relative_to(ROOT))] = sha(previous)
    for name in ('completion_manifest_v1.json', 'pilot_completion_manifest_v1.json'):
        path = OLD/name
        merge(evidence, read(path)['artifact_sha256'], OLD)
        evidence[str(path.relative_to(ROOT))] = sha(path)
    for directory in ('pilot_v1', 'main_run_v1', 'main_registration_v1'):
        path = OLD/directory/'manifest.json'
        manifest = read(path)
        if directory != 'main_registration_v1' and manifest['complete'] is not True:
            raise ValueError('Incomplete retained experiment')
        for key in ('input_sha256', 'source_sha256'):
            merge(evidence, manifest[key])
        merge(evidence, manifest['artifact_sha256'], path.parent)
        evidence[str(path.relative_to(ROOT))] = sha(path)
        if 'registration' in manifest:
            merge(evidence, {manifest['registration']+'/manifest.json': manifest['registration_sha256']})
    allocations = read(OLD/'main_registration_v1/allocations.json')
    if allocations['main_outcomes_inspected'] is not False:
        raise ValueError('Original counts must have been fixed before main outcomes')
    selected = {r['case']: r['allocation']['selected'] for r in allocations['models']}
    forbidden = set(allocations['forbidden_streams'])
    geometry = read(GEOMETRY/'inputs_v1/inputs.json')
    if [m['case'] for m in geometry['models']] != list(range(13)):
        raise ValueError('Require thirteen fixed financial cases')
    all_streams, models, signatures, fingerprints = set(), [], {}, {}
    for old_model in geometry['models']:
        case = old_model['case']
        pilot_path = OLD/f'pilot_v1/blocks/case_{case:03d}.json'
        pilot = read(pilot_path)
        if pilot['case'] != case:
            raise ValueError('Pilot case differs')
        samples = []
        for source in SOURCES:
            if source == 'pilot':
                count, paths = 32, 4096
                blocks = [(pilot_path, row) for row in pilot['statistical']]
            else:
                count = 64 if source == 'reference' else selected[case]['repetitions']
                paths = 4096 if source == 'reference' else selected[case]['evaluation_paths']
                block_paths = sorted((OLD/'main_run_v1/blocks').glob(f'case_{case:03d}_{source}_rep_*.json'))
                blocks = [(p, read(p)) for p in block_paths]
            if len(blocks) != count:
                raise ValueError('Retained row count differs')
            rows = []
            for rep, (path, row) in enumerate(blocks):
                relative = str(path.relative_to(ROOT))
                if relative not in evidence or sha(path) != evidence[relative]:
                    raise ValueError('Raw row lacks verified provenance')
                if row['repetition'] != rep or (source != 'pilot' and (row['case'], row['arm']) != (case, source)):
                    raise ValueError('Row identity differs')
                training, evaluation = row['training'], row['evaluation']
                estimate = evaluation['gain']['estimate']
                if (evaluation['evaluation_paths'], estimate['observations'], evaluation['stream']['paths']) != (paths,)*3:
                    raise ValueError('Evaluation count differs')
                if training['training_paths'] != [5000, 20000] or len(training['streams']) != 2:
                    raise ValueError('Training procedures differ')
                streams = [s['stream_id'] for s in training['streams']]+[evaluation['stream']['stream_id']]
                if len(set(streams)) != 3 or all_streams.intersection(streams):
                    raise ValueError('Overlapping retained streams')
                if source != 'pilot' and forbidden.intersection(streams):
                    raise ValueError('Main stream overlaps excluded pilot evidence')
                all_streams.update(streams)
                signature = [{k: p[k] for k in ('model', 'option', 'decisions', 'requested_degree',
                             'training_paths', 'minimum_itm_paths', 'condition_limit')}
                             for p in training['policies']]
                fingerprint = evaluation['approximation_fingerprint']
                if case in signatures and (signature != signatures[case] or fingerprint != fingerprints[case]):
                    raise ValueError('Financial, training or approximation specification differs')
                signatures[case], fingerprints[case] = signature, fingerprint
                mean, error = Fraction(estimate['value']), Fraction(estimate['standard_error'])
                if error < 0:
                    raise ValueError('Negative standard error')
                if source == 'pilot':
                    prior = old_model['pairs'][rep]
                    if (mean, error) != (Fraction(prior['mean']), Fraction(prior['standard_error'])):
                        raise ValueError('Pilot differs from preceding component analysis')
                rows.append(dict(raw_file=relative, repetition=rep, mean=mean,
                                 standard_error=error, stream_ids=streams))
            samples.append(dict(source=source, pairs=count, paths=paths, rows=rows))
        models.append(dict(case=case, approximation_fingerprint=fingerprints[case],
                           training_specification_sha256=hashlib.sha256(json.dumps(signatures[case], sort_keys=True).encode()).hexdigest(),
                           regimes=old_model['regimes'], samples=samples))
    if sum(s['pairs'] for m in models for s in m['samples']) != 5446 or len(all_streams) != 16338:
        raise ValueError('Total row or stream coverage differs')
    sources = [Path(__file__).resolve(), ROOT/'src/qf_research/optimal_stopping/component_ratio.py',
               ROOT/'src/qf_research/optimal_stopping/allocation_variance_sensitivity.py',
               ROOT/'scripts/certify_component_ratio.py', ROOT/'tests/test_component_ratio.py',
               BASE/'design_v1/protocol.md', BASE/'theory_v1/asymptotic_ratio.md',
               LITERATURE/'source_record.json', LITERATURE/'sources.md']
    for path in sources:
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    output.mkdir(parents=True)
    save(output/'inputs.json', dict(cases=13, family_size=39, reference_paths=4096,
         critical_value='13/4', family_alpha='1/20', precision_bits=128,
         retrospective=True, new_holdout=False, models=models,
         rows=5446, unique_stream_ids=16338, publication_clearance=False))
    save(output/'manifest.json', dict(complete=True, evidence_sha256=evidence,
         artifact_sha256={'inputs.json': sha(output/'inputs.json')}))


def run(inputs, output):
    inputs, output = private(inputs), private(output)
    if output.exists():
        raise FileExistsError('Use a fresh result directory')
    manifest = read(inputs/'manifest.json')
    if manifest['complete'] is not True:
        raise ValueError('Incomplete inputs')
    verify(manifest['evidence_sha256']); verify(manifest['artifact_sha256'], inputs)
    payload = read(inputs/'inputs.json')
    if (payload['cases'], payload['family_size'], payload['reference_paths'], payload['critical_value'],
        payload['family_alpha'], payload['retrospective'], payload['new_holdout']) != (13, 39, 4096, '13/4', '1/20', True, False):
        raise ValueError('Declared inference design differs')
    output.mkdir(parents=True)
    records, endpoints, concordance = [], [], []
    for model in payload['models']:
        case, case_records = model['case'], []
        for sample in model['samples']:
            stats = moment_covariance([Fraction(r['mean']) for r in sample['rows']],
                [Fraction(r['standard_error']) for r in sample['rows']], sample['paths'])
            coefficients = fieller_coefficients(stats['denominator'], stats['numerator'], stats['covariance'])
            region = quadratic_set(**coefficients)
            regimes = [dict(**r, **allocation_choices(**coefficients, candidates=r['candidates']))
                       for r in model['regimes']]
            record = dict(case=case, source=sample['source'], moments=stats,
                          coefficients=coefficients, region=region, regimes=regimes)
            save(output/f"case_{case:02d}_{sample['source']}.json", record)
            records.append(record); case_records.append(record)
            endpoints.append(dict(case=case, source=sample['source'], components=region['components']))
        comparisons = []
        for target in case_records:
            a, b, c = (target['coefficients'][k] for k in ('a', 'b', 'c'))
            for other in case_records:
                if other['source'] == target['source']:
                    continue
                point = other['moments']['point_share']
                valid = other['moments']['point_in_parameter_domain']
                comparisons.append(dict(set_source=target['source'], point_source=other['source'],
                    point_is_admissible=valid,
                    contains_point=(a*point*point+b*point+c <= 0) if valid else None))
        concordance.append(dict(case=case, comparisons=comparisons))
    save(output/'endpoint_input.json', dict(family_size=39, critical_value='13/4', family_alpha='1/20', sets=endpoints))
    subprocess.run([str(Path.home()/'.venvs/qf-certification/bin/python'),
        str(ROOT/'scripts/certify_component_ratio.py'), '--input', str(output/'endpoint_input.json'),
        '--precision-bits', '128', '--output', str(output/'enclosures.json')], check=True, capture_output=True, text=True)
    aggregates = []
    for source in SOURCES:
        rows = [r for r in records if r['source'] == source]
        regimes = [a for r in rows for a in r['regimes']]
        aggregates.append(dict(source=source, sets=len(rows),
            unrestricted_kind_counts=dict(Counter(r['region']['unrestricted_kind'] for r in rows)),
            physical_kind_counts=dict(Counter(r['region']['physical_kind'] for r in rows)),
            negative_A=sum(r['moments']['A'] < 0 for r in rows),
            nonpositive_denominators=sum(r['moments']['denominator'] <= 0 for r in rows),
            covariance_rank_counts=dict(Counter(r['moments']['covariance_rank'] for r in rows)),
            admissible_point_estimates=sum(r['moments']['point_in_parameter_domain'] for r in rows),
            unique_allocation_regimes=sum(r['uniformly_selected_size'] is not None for r in regimes),
            selected_size_count_distribution=dict(Counter(len(r['selected_sizes']) for r in regimes))))
    save(output/'concordance.json', dict(scope='Descriptive comparison of retrospective sources; truth unknown',
         coverage_validation=False, cases=concordance))
    save(output/'summary.json', dict(complete=True, cases=13, component_sets=39, fixed_cost_comparisons=234,
         rows=5446, reference_paths=4096, critical_value='13/4', nominal_asymptotic_family_confidence_at_least='19/20',
         source_summaries=aggregates, retrospective=True, new_holdout=False,
         conditional_selected_counts=True, finite_sample_coverage_certified=False,
         simulated_summary_errors_certified=False, ideal_sampling_law_certified=False,
         exact_saved_summary_algebra=True, exact_rational_allocation_mapping=True,
         source_point_inclusion_is_coverage=False, new_simulations=0, policy_refits=0,
         new_market_observations=0, publication_clearance=False, python=platform.python_version()))
    evidence = dict(manifest['evidence_sha256'])
    for path in (inputs/'manifest.json', inputs/'inputs.json'):
        evidence[str(path.relative_to(ROOT))] = sha(path)
    verify(evidence)
    save(output/'manifest.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
         evidence_sha256=evidence, artifact_sha256={p.name: sha(p) for p in sorted(output.glob('*.json'))}))


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
