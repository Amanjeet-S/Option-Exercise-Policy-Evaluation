"""Delete-one-repetition sensitivity of a retained private training experiment.

This is a post hoc influence diagnostic, not a replacement estimator, a
coverage study, an outlier exclusion rule or a new validation sample.
"""

import argparse
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

from scipy.stats import t

from audit_training_evaluation import verify
from run_conditional_calibration import ROOT, private, read, save, sha, verify_mapping


def summary(records, confidence=.95):
    """Reconstruct repetition inference without the experiment's estimator."""
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('Require confidence strictly between zero and one')
    if len(records) < 2:
        raise ValueError('At least two repetitions are required')
    identifiers = [r['repetition'] for r in records]
    if (any(type(i) is not int or i < 0 for i in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Require distinct non-negative repetition identifiers')
    for r in records:
        if (not math.isfinite(r['value']) or not math.isfinite(r['standard_error'])
                or r['standard_error'] < 0):
            raise ValueError('Require finite means and non-negative standard errors')
    means = [r['value'] for r in records]
    observed = statistics.variance(means)
    evaluation = statistics.mean(r['standard_error']**2 for r in records)
    estimate = statistics.mean(means)
    se = math.sqrt(observed/len(records))
    radius = float(t.ppf((1+confidence)/2, len(records)-1))*se
    return dict(repetitions=len(records), mean=estimate, standard_error=se,
                lower=estimate-radius, upper=estimate+radius,
                repetition_variance=observed,
                mean_conditional_evaluation_variance=evaluation,
                training_variance_moment_estimate=observed-evaluation,
                evaluation_share_percent=100*evaluation/observed if observed > 0 else None)


def delete_one(records, confidence=.95):
    """Retain every possible deletion; do not select a favourable subset."""
    if len(records) < 3:
        raise ValueError('At least three repetitions are required for deletion')
    original = summary(records, confidence)
    ordered = sorted(records, key=lambda r: r['repetition'])
    deleted = [dict(omitted_repetition=r['repetition'],
                    **summary([x for x in ordered if x['repetition'] != r['repetition']], confidence))
               for r in ordered]
    return dict(primary=original, deletions=deleted,
                minimum_deleted_mean=min(r['mean'] for r in deleted),
                maximum_deleted_mean=max(r['mean'] for r in deleted),
                maximum_absolute_mean_change=max(abs(r['mean']-original['mean']) for r in deleted),
                positive_deleted_intervals=sum(r['lower'] > 0 for r in deleted),
                all_deleted_means_positive=all(r['mean'] > 0 for r in deleted))


def review(run, paper, output):
    run, paper, output = map(private, (run, paper, output))
    if output.exists():
        raise FileExistsError('Use a fresh private review directory')
    blocks, cfg, tasks, checks = verify(run)
    verify_mapping(read(paper/'completion_manifest.json')['artifact_sha256'], paper)
    paper_rows = {r['case']: r for r in read(paper/'tables.json')['training_evaluation']['rows']}
    if set(paper_rows) != {r['case'] for r in tasks}:
        raise ValueError('Paper and experiment cover different models')
    models = []
    for task in tasks:
        chosen = sorted((b for b in blocks if b['case'] == task['case']), key=lambda b: b['repetition'])
        if [b['repetition'] for b in chosen] != list(range(cfg['training_repetitions'])):
            raise ValueError('Incomplete repetition coverage')
        records = []
        for b in chosen:
            e = next(g['estimate'] for g in b['policy_gains'] if
                     g['method'] == 'fine' and g['evaluation_paths'] == cfg['evaluation_paths'][-1])
            records.append(dict(repetition=b['repetition'], value=e['value'], standard_error=e['standard_error']))
        result = delete_one(records, cfg['confidence'])
        for computed, retained in [('mean', 'gain'), ('standard_error', 'standard_error'),
                                   ('lower', 'lower'), ('upper', 'upper')]:
            if not math.isclose(result['primary'][computed], paper_rows[task['case']][retained],
                                rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError('The paper primary comparison differs from independent reconstruction')
        models.append(dict(case=task['case'], repetition_estimates=records, **result))
    deletions = [dict(case=m['case'], **r) for m in models for r in m['deletions']]
    finite_shares = [r['evaluation_share_percent'] for r in deletions if r['evaluation_share_percent'] is not None]
    outcome = dict(models=len(models), deletions=len(deletions),
                   full_positive_interval_cases=[m['case'] for m in models if m['primary']['lower'] > 0],
                   positive_after_every_deletion_cases=[m['case'] for m in models if
                       m['positive_deleted_intervals'] == cfg['training_repetitions']],
                   full_positive_but_not_every_deletion_cases=[m['case'] for m in models if
                       m['primary']['lower'] > 0 and m['positive_deleted_intervals'] != cfg['training_repetitions']],
                   all_deleted_means_positive=all(m['all_deleted_means_positive'] for m in models),
                   evaluation_share_percent_range=[min(finite_shares), max(finite_shares)] if finite_shares else None)
    output.mkdir(parents=True)
    save(output/'results.json', dict(models=models, summary=outcome, post_hoc_diagnostic=True,
                                    replacement_primary_analysis=False, coverage_validation=False,
                                    new_market_observations=False, new_simulation=False, publication_clearance=False))
    with (output/'deletions.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deletions[0]))
        writer.writeheader()
        writer.writerows(deletions)
    save(output/'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
         input_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                       (run/'manifest.json', paper/'completion_manifest.json')},
         source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__)),
                        'tests/test_training_inference_review.py': sha(ROOT/'tests/test_training_inference_review.py')},
         artifact_sha256={n: sha(output/n) for n in ('results.json', 'deletions.csv')},
         experiment_checks=checks, primary_table_rows_reconstructed=len(models),
         publication_clearance=False))
    print(outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'paper', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    review(args.run, args.paper, args.output)
