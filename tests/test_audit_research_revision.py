"""Integrity, interpretation and accounting checks for the revision evidence audit."""

import hashlib
import json
from pathlib import Path
import sys

import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_research_revision import (
    Archive, cost_accounting, family_calibration, financial_scale,
    interval_summary, retraining_evidence, write_report,
)


def estimate(value, error=.01):
    return dict(value=value, standard_error=error, lower=value-2*error, upper=value+2*error)


def test_bound_reader_rejects_changed_evidence_and_path_escape(tmp_path):
    folder = tmp_path/'run'; folder.mkdir()
    source = folder/'rows.json'; source.write_text('[1, 2]')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (folder/'manifest.json').write_text(json.dumps({'artifact_sha256': {'rows.json': digest}}))
    archive = Archive(tmp_path)
    assert archive.read('run', 'rows.json') == [1, 2]
    source.write_text('[1, 3]')
    with pytest.raises(ValueError, match='hash differs'):
        archive.read('run', 'rows.json')
    outside = tmp_path.parent/'escaped-revision-evidence.json'
    outside.write_text('{}')
    try:
        with pytest.raises(ValueError, match='leaves'):
            archive._path('../escaped-revision-evidence.json')
    finally:
        outside.unlink()


def test_cutoff_means_a_union_bound_and_needs_no_contrast_independence():
    result = family_calibration()
    assert result['asymptotic_union_bound_family_error'] == pytest.approx(.0138486010173783)
    assert result['asymptotic_union_bound_coverage_lower'] > .986
    assert result['five_per_cent_bonferroni_normal_cutoff'] == pytest.approx(norm.ppf(1-.05/24))
    assert result['finite_sample_coverage_established'] is False
    assert result['independence_between_contrasts_required'] is False


def test_retraining_sign_counts_do_not_replace_magnitudes_or_pool_standard_errors():
    pairs = [(5000, 2), (5000, 4), (20000, 2), (20000, 4)]
    policies = [dict(training_paths=n, degree=d, repetition=0, paired_gain=estimate(value),
                     changed_stops=2) for (n, d), value in zip(pairs, [-.03, .12, -.04, -.005])]
    rows = [dict(case=0, stream={'paths': 20},
                 scenarios=[dict(extension_seconds=5400, policies=policies)])]
    result = retraining_evidence(rows)
    assert result['pooled']['value']['negative'] == 3
    assert result['pooled']['negative_intervals'] == 2
    assert result['pooled']['intervals_including_zero'] == 1
    assert result['pooled']['value']['mean'] > 0
    assert 'Negative values mean' in result['sign_convention']
    assert 'standard_error' not in result['pooled'] or isinstance(result['pooled']['standard_error'], dict)


def test_waiting_comparison_exports_research_outputs_without_copying_provider_inputs():
    policies = [dict(training_paths=n, degree=d,
        gain={'controlled_quote': estimate(.04)}, pde_shortfall=.06,
        captured_fraction=.4, early_exercise_fraction=.2)
        for n, d in [(5000, 2), (5000, 4), (20000, 2), (20000, 4)]]
    rows = [dict(case=0, european_quote=2., pde_quote=2.1, model_exercise_premium=.1,
                 policies=policies, put_bid='PRIVATE_SENTINEL', raw_provider_response='PRIVATE_SENTINEL')]
    result = financial_scale(rows)
    assert result['numerical_exercise_premium']['mean'] == .1
    assert result['configurations'][0]['gain_over_terminal_waiting']['value']['mean'] == .04
    assert 'PRIVATE_SENTINEL' not in json.dumps(result)
    rows[0]['model_exercise_premium'] = .2
    with pytest.raises(ValueError, match='identity'):
        financial_scale(rows)


def test_pilot_is_charged_once_and_observed_cost_difference_is_preserved():
    arms = {}
    for name, cost, count in [('reference', 10., 64), ('selected', 12., 128)]:
        arms[name] = dict(standalone_cpu_seconds=cost,
                         allocation={'repetitions': count, 'evaluation_paths': 4096 if count == 64 else 256})
    main = {'models': [dict(case=0, arms=arms, comparison={'estimated_variance_ratio': .6})]}
    timing = {'totals': {a: {'actual_cpu_seconds': {'setup': 1., 'training': 1., 'evaluation': c-2.}}
                         for a, c in [('reference', 10.), ('selected', 12.)]}}
    result = cost_accounting({'process_cpu_seconds': 8., 'elapsed_seconds': 9.},
        {'process_cpu_seconds': 23.}, main, timing, {'primary_numerical_cpu_seconds': 5.},
        {'numerical_totals': {'cpu': '7/2'}, 'process_cpu_seconds': 4.})
    assert result['one_off_pilot_plus_selected_cpu'] == 20.
    assert result['pilot_plus_both_standalone_arms_cpu'] == 30.
    assert result['simple_reference_variance_counterfactual_ratio'] == .5
    assert result['per_case'][0]['selected_to_reference_cpu'] == 1.2
    assert result['equal_observed_cost_comparison'] is False


def test_invalid_interval_and_public_or_existing_destination_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Invalid'):
        interval_summary([estimate(1., -.1)])
    private = tmp_path/'private'
    with pytest.raises(ValueError, match='private output'):
        write_report({}, tmp_path/'public', private)
    output = private/'run'
    path = write_report({'research_output': .5}, output, private)
    assert json.loads(path.read_text()) == {'research_output': .5}
    with pytest.raises(FileExistsError):
        write_report({}, output, private)
