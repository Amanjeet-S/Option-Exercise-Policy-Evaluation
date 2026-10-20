"""Check the revised manuscript against retained public-model summaries.

These checks concern transcription, coverage arithmetic and certificate
records. Re-execution of simulations and independent mathematical review are
separate tasks. Historical tables retain their own original checker.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1]


def validate_decision_study(study):
    """Check the declared simultaneous decision family and its derived summaries.

    The retained repetition indices verify union arithmetic. They do not
    independently regenerate simulation outcomes or certify floating-point
    confidence-limit calculations.
    """
    import math
    from statistics import NormalDist

    if not isinstance(study, dict):
        raise ValueError('The decision study must be a result mapping')

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('A finite scalar result is required')
        return Fraction(value) if isinstance(value, int) else Fraction.from_float(value)

    def rational(record):
        try:
            numerator, denominator = int(record['numerator']), int(record['denominator'])
            if denominator <= 0:
                raise ValueError('Certificate denominator must be positive')
            result = Fraction(numerator, denominator)
            display = record['approximation']
            if number(display) != number(float(result)):
                raise ValueError('Certificate display differs from its exact rational')
            return result
        except (TypeError, KeyError, OverflowError, ZeroDivisionError) as error:
            raise ValueError('Invalid rational certificate record') from error

    def interval(record):
        lo, hi, width = (rational(record[key]) for key in ('lower', 'upper', 'width'))
        if lo < 0 or hi < lo or width != hi-lo:
            raise ValueError('Decision certificate endpoint/width identity fails')
        return lo, hi

    def equal(left, right, message, tolerance=1e-12):
        number(left); number(right)
        if not math.isclose(left, right, rel_tol=0., abs_tol=tolerance):
            raise ValueError(message)

    def nonnegative(value):
        if number(value) < 0:
            raise ValueError('A decision cost or bound is negative')

    def count(value, total=256):
        if type(value) is not int or not 0 <= value <= total:
            raise ValueError('Invalid integer decision count')
        return value

    def wilson(successes, total, recorded):
        z = NormalDist().inv_cdf(.975)
        proportion = successes/total
        scale = 1+z*z/total
        centre = (proportion+z*z/(2*total))/scale
        radius = z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/scale
        if not isinstance(recorded, list) or len(recorded) != 2:
            raise ValueError('Invalid decision Wilson interval')
        equal(recorded[0], max(0., centre-radius), 'Decision Wilson lower limit differs from its count')
        equal(recorded[1], min(1., centre+radius), 'Decision Wilson upper limit differs from its count')

    def indices(values):
        if (not isinstance(values, list) or any(type(value) is not int or not 0 <= value < 256 for value in values)
                or values != sorted(set(values))):
            raise ValueError('False-certification repetition indices must be sorted, distinct and in range')
        return set(values)

    policies = ('threshold_100', 'threshold_110', 'threshold_120', 'threshold_130', 'always_wait')
    methods = ('zero', 'stock', 'terminal_call')
    counts = (128, 512, 2048, 8192, 32768)
    tolerances = (.1, .25, .5)
    design = study['design']
    if (design['thresholds'] != [100., 110., 120., 130., None] or
            design['methods'] != list(methods) or design['outer_counts'] != list(counts) or
            design['loss_tolerances'] != list(tolerances) or
            type(design['repetitions']) is not int or design['repetitions'] != 256 or
            type(design['pilot_paths']) is not int or design['pilot_paths'] != 8192 or
            type(design['family_members']) is not int or design['family_members'] != 75):
        raise ValueError('The prescribed decision design differs')
    alpha, family_alpha = number(design['fixed_look_error_probability']), number(design['family_error_probability'])
    if family_alpha != number(.025) or not 0 < alpha or alpha*75 > family_alpha:
        raise ValueError('The exact allocated probability exceeds the simultaneous family budget')
    for flag in ('market_validation_claim', 'new_general_theorem_claim', 'public_preregistration_claim'):
        if design[flag] is not False:
            raise ValueError('Unsupported decision scope: '+flag)
    if design.get('numerical_scope') != 'Certified reference shortfalls and deterministic caps; Monte Carlo simulation, gain evaluations, tail allowances and confidence-limit arithmetic are floating-point computations.':
        raise ValueError('The decision summary misstates numerical certification scope')
    if design.get('cost_scope') != 'Measured function work includes own pilot, coefficient/cap fitting, tail construction, evaluation and decisions. Imports, installation and file serialisation are excluded. Later validation looks do not enter an earlier decision cost.':
        raise ValueError('The standalone assessment cost scope differs')
    if design.get('standalone_cost_observations') != 'Each observed assessment prefix adds one retained independent frozen setup measurement; setup is not independently rerun in each repetition.':
        raise ValueError('The decision timing scope omits the single frozen setup measurement')
    if study.get('actual_study_cost_scope') != 'Observed validation workflow includes expected-gap reference integrations and per-arm evidence writing. It is not the standalone assessment phase cost. Final summary serialisation and process startup are excluded.':
        raise ValueError('Whole-study validation costs are confused with standalone assessments')
    process = dict(spot=100., strike=100., rate=.05, dividend_yield=.1, volatility=.2,
                   first_right=1.5, maturity=3.)
    if design['process'] != process or study['process'] != process:
        raise ValueError('The decision study uses a different process')
    reference = study['reference']
    analytical = study['analytical_reference']
    for record in (reference, analytical):
        if record['ideal_model_population_loss_enclosed'] is not True or any(record[flag] is not False
                for flag in ('realised_simulation_law_certified', 'statistical_coverage_certified')):
            raise ValueError('Unsupported decision-reference certification scope')
        if set(record['model']) != set(process) or any(rational(record['model'][key]) != number(value)
                for key, value in process.items()):
            raise ValueError('The decision certificate uses different exact model inputs')
    if analytical['population_loss_upper_only'] is not True or any(analytical[flag] is not False for flag in
            ('boundary_search_executed', 'numerical_quadrature_executed', 'simulation_executed')):
        raise ValueError('The analytical comparator is not a root-free upper-bound calculation')

    def reference_policy(row):
        theta = row['threshold']
        if theta is None:
            if row['policy'] != 'always_wait':
                raise ValueError('Invalid always-wait reference')
            return 'always_wait'
        threshold = rational(theta)
        if row['policy'] != 'first_right_threshold' or threshold not in (100, 110, 120, 130):
            raise ValueError('Invalid threshold reference')
        return 'threshold_'+str(int(threshold))

    reference_rows = {reference_policy(row): row for row in reference['policies']}
    analytical_rows = {reference_policy(row): row for row in analytical['policies']}
    if (len(reference['policies']) != 5 or set(reference_rows) != set(policies) or
            len(analytical['policies']) != 4 or set(analytical_rows) != set(policies[:-1])):
        raise ValueError('Missing or duplicated decision-reference policies')
    losses = {policy: interval(row['loss']) for policy, row in reference_rows.items()}
    bounds = {}
    for policy in policies[:-1]:
        reference_bound = rational(reference_rows[policy]['deterministic_gap_upper'])
        analytical_bound = rational(analytical_rows[policy]['deterministic_gap_upper'])
        if reference_bound < 0 or analytical_bound != reference_bound:
            raise ValueError('Independent analytical gap bounds differ')
        bounds[policy] = reference_bound
        if interval(analytical_rows[policy]['loss']) != (0, reference_bound):
            raise ValueError('The analytical reference is not an upper-only loss enclosure')
        if losses[policy][1] > reference_bound:
            raise ValueError('The population reference exceeds its deterministic gap bound')
    bounds['always_wait'] = None

    def grid(rows, fields, expected, message):
        found = [tuple(row[field] for field in fields) for row in rows]
        if len(found) != len(expected) or set(found) != expected:
            raise ValueError(message)
        return {key: row for key, row in zip(found, rows)}

    pair_grid = {(policy, method) for policy in policies for method in methods}
    setups = grid(study['setup'], ('policy', 'method'), pair_grid,
                  'Missing or duplicated frozen decision setups')
    for (policy, method), row in setups.items():
        direct = method == 'terminal_call' and policy != 'always_wait'
        expected_pilot = 0 if direct else 8192
        if type(row['pilot_paths']) is not int or row['pilot_paths'] != expected_pilot:
            raise ValueError('The independent pilot count differs from the decision method')
        if row['deterministic_bound_reused'] is not direct:
            raise ValueError('The setup misstates reuse of its deterministic bound')
        for value in row['setup_cost'].values():
            nonnegative(value)
        if set(row['caps']) != {str(value) for value in counts}:
            raise ValueError('The frozen cap family differs from the fixed looks')
        for cap in row['caps'].values():
            nonnegative(cap['tail_allowance'])
            if number(cap['cap']) <= 0:
                raise ValueError('A clipping cap must be positive')
            if direct and (number(cap['cap']) < bounds[policy] or cap['tail_allowance'] != 0.):
                raise ValueError('A finite deterministic cap was rounded down or assigned a spurious tail')

    expected_refs = grid(study['expected_gap_references'], ('policy', 'method'), pair_grid,
                         'Missing or duplicated expected-gap diagnostics')
    for (policy, method), row in expected_refs.items():
        midpoint = float(sum(losses[policy])/2)
        equal(row['reference_policy_loss'], midpoint, 'Diagnostic policy loss differs from its certificate')
        gap = row['expected_gap']
        equal(row['dual_slack_estimate'], gap['value']-midpoint,
              'Diagnostic dual slack differs from the expectation/loss subtraction')
        if method == 'terminal_call':
            if gap['rigorous_enclosure'] is not True or row['dual_slack_estimate'] != 0.:
                raise ValueError('The exact-centred expectation must match the population loss')
            try:
                supplied = tuple(Fraction(gap['rational_interval'][key]) for key in ('lower', 'upper'))
            except (TypeError, ValueError, ZeroDivisionError) as error:
                raise ValueError('Invalid exact-centred expectation interval') from error
            if supplied != losses[policy]:
                raise ValueError('The exact-centred expectation interval differs from the certified loss')
        else:
            if gap['rigorous_enclosure'] is not False:
                raise ValueError('A floating-point expected-gap diagnostic cannot claim a rigorous enclosure')
            for field in ('quadrature_error_estimate', 'omitted_tail_upper'):
                nonnegative(gap[field])
            if row.get('dual_slack_certified', False) is not False:
                raise ValueError('Floating-point dual-slack diagnostics are not certificates')

    deterministic = grid(study['deterministic_decisions'], ('policy', 'tolerance'),
                         {(policy, tolerance) for policy in policies for tolerance in tolerances},
                         'Missing or duplicated deterministic decisions')
    for (policy, tolerance), row in deterministic.items():
        lo, hi = losses[policy]
        tolerance_exact = number(tolerance)
        analytic_pass = bounds[policy] is not None and bounds[policy] <= tolerance_exact
        if (row['analytic_bound_certifies'] is not analytic_pass or
                row['certified_reference_certifies'] is not (hi <= tolerance_exact) or
                row['certified_reference_excludes'] is not (lo > tolerance_exact)):
            raise ValueError('A deterministic decision differs from its rational endpoints')
        equal(row['loss_lower'], float(lo), 'Deterministic loss display differs from its lower endpoint')
        equal(row['loss_upper'], float(hi), 'Deterministic loss display differs from its upper endpoint')
        if bounds[policy] is None:
            if row['analytic_gap_upper'] is not None or row['standalone_analytic_cost'] is not None:
                raise ValueError('Always-wait was assigned a finite analytical cap')
        else:
            equal(row['analytic_gap_upper'], float(bounds[policy]), 'Analytical gap display differs from its endpoint')
            for key in ('cpu_seconds', 'wall_seconds'):
                equal(row['standalone_analytic_cost'][key], analytical['work'][key],
                      'A standalone analytical decision omits family construction cost')
        for key in ('cpu_seconds', 'wall_seconds'):
            equal(row['standalone_certified_quadrature_cost'][key], reference['work'][key],
                  'A standalone reference decision omits full certificate computation')

    decisions = grid(study['decisions'], ('policy', 'method', 'loss_tolerance'),
                     {(policy, method, tolerance) for policy in policies for method in methods for tolerance in tolerances},
                     'Missing or duplicated decision cells')
    union = set()
    for (policy, method, tolerance), row in decisions.items():
        if type(row['repetitions']) is not int or row['repetitions'] != 256:
            raise ValueError('The decision repetition count differs')
        successes, false_count = count(row['certifications']), count(row['false_certifications'])
        if row['certification_rate'] != successes/256:
            raise ValueError('Decision rate differs from its certification count')
        wilson(successes, 256, row['certification_rate_wilson_95'])
        lo, hi = losses[policy]
        tolerance_exact = number(tolerance)
        classification = ('above_tolerance' if lo > tolerance_exact else
                          'within_tolerance' if hi <= tolerance_exact else 'reference_unresolved')
        if row['reference_classification'] != classification:
            raise ValueError('Decision reference classification differs from its rational interval')
        false_indices = indices(row['false_certification_repetition_indices'])
        if false_count != (successes if classification == 'above_tolerance' else 0) or len(false_indices) != false_count:
            raise ValueError('False-certification count or indices differ from the certified reference')
        union.update(false_indices)
        paths = number(row['mean_paths'])
        if not 0 <= paths <= 32768:
            raise ValueError('The decision path mean lies outside the declared budget')
        shortcut = method == 'terminal_call' and bounds[policy] is not None and bounds[policy] <= tolerance_exact
        if paths == 0 and (not shortcut or successes != 256 or row['mean_assessment_cpu_seconds'] != 0.):
            raise ValueError('Zero evaluation paths require a successful deterministic analytical bound')
        for field in ('mean_one_off_cpu_seconds', 'p95_one_off_cpu_seconds', 'mean_assessment_cpu_seconds', 'setup_cpu_seconds'):
            nonnegative(row[field])
        equal(row['setup_cpu_seconds'], setups[policy, method]['setup_cost']['cpu_seconds'],
              'A decision changes its frozen setup charge')
        equal(row['mean_one_off_cpu_seconds'], row['mean_assessment_cpu_seconds']+row['setup_cpu_seconds'],
              'Charged decision CPU omits or duplicates the frozen setup')
        successful_cost = row['mean_one_off_cpu_seconds_given_success']
        if successes == 0:
            if successful_cost is not None:
                raise ValueError('A success-conditioned cost requires successful decisions')
        else:
            nonnegative(successful_cost)
            if successful_cost+1e-12 < row['setup_cpu_seconds']:
                raise ValueError('Successful decision costs omit their frozen setup')
            if successes == 256:
                equal(successful_cost, row['mean_one_off_cpu_seconds'],
                      'Universal-success costs differ from unconditional charged costs')
    recorded_union = indices(study['family_false_certification_repetition_indices'])
    family_count = count(study['family_false_certification_repetitions'])
    if recorded_union != union or family_count != len(union):
        raise ValueError('Family false certifications are not the union of repetition events')
    if study['family_false_certification_rate'] != len(union)/256:
        raise ValueError('Family false-certification rate differs from its union count')
    wilson(len(union), 256, study['family_false_certification_rate_wilson_95'])
    for value in study['actual_study_cost'].values():
        nonnegative(value)
    return True


def _replicated_scalar(value):
    import math
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('A finite replicated-study scalar is required')
    return value


def _replicated_equal(actual, expected, message):
    import math
    _replicated_scalar(actual); _replicated_scalar(expected)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(message)


def _replicated_rational(record):
    try:
        values = []
        for key in ('numerator', 'denominator'):
            value = record[key]
            if type(value) not in (str, int) or not re.fullmatch(r'[+-]?\d+', str(value)):
                raise ValueError('An integer rational numerator and denominator are required')
            values.append(int(value))
        if values[1] <= 0:
            raise ValueError('A certificate denominator must be positive')
        answer = Fraction(*values)
        if _replicated_scalar(record['approximation']) != float(answer):
            raise ValueError('A certificate display differs from its exact rational')
        return answer
    except (KeyError, TypeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError('Invalid replicated-study rational certificate') from error


def _replicated_interval(record, *, signed=False):
    if set(record) != {'lower', 'upper', 'width'}:
        raise ValueError('A certificate requires lower, upper and width records')
    lo, hi, width = (_replicated_rational(record[key]) for key in ('lower', 'upper', 'width'))
    if hi < lo or width != hi-lo or (not signed and lo < 0):
        raise ValueError('Replicated certificate endpoint/width identity fails')
    return lo, hi


def _replicated_interval_record(lo, hi):
    def rational(value):
        return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))
    return dict(lower=rational(lo), upper=rational(hi), width=rational(hi-lo))


def _replicated_ratio_summary(values):
    import statistics
    if len(values) != 8 or any(_replicated_scalar(value) < 0 for value in values):
        raise ValueError('Eight non-negative fitting-unit ratios are required')
    return dict(fitting_units=8, minimum=min(values), median=statistics.median(values), maximum=max(values))


def _replicated_distribution(values):
    import math
    import statistics
    if len(values) != 8:
        raise ValueError('Eight independent fitting-unit summaries are required')
    for value in values:
        _replicated_scalar(value)
    return dict(fitting_units=8, mean=statistics.mean(values),
                standard_error=statistics.stdev(values)/math.sqrt(8), minimum=min(values), maximum=max(values))


def _replicated_check_mapping(actual, expected, message):
    if set(actual) != set(expected):
        raise ValueError(message)
    for key, value in expected.items():
        if key == 'fitting_units':
            if type(actual[key]) is not int or actual[key] != value:
                raise ValueError(message)
        else:
            _replicated_equal(actual[key], value, message)


def _replicated_path_summary(rows, *, signed=False, expected_count=256):
    import math
    import statistics
    if type(expected_count) is not int or expected_count < 2 or len(rows) != expected_count:
        raise ValueError('Every fitting unit requires its complete declared path count')
    endpoints = []
    for index, row in enumerate(rows):
        if (type(row['path_index']) is not int or row['path_index'] != index or
                type(row['stopping_index']) is not int or not 0 <= row['stopping_index'] <= 9):
            raise ValueError('Path order or frozen stopping index is invalid')
        bounds = _replicated_interval(row['signed_gap_certificate'], signed=signed)
        if bounds[1]-bounds[0] > Fraction(1,50 if signed else 100):
            raise ValueError('An exact path certificate exceeds its required width')
        endpoints.append(bounds)
    lo = sum(value[0] for value in endpoints)/expected_count
    hi = sum(value[1] for value in endpoints)/expected_count
    mids = [float((lower+upper)/2) for lower, upper in endpoints]
    return dict(mean_signed_gap_certificate=_replicated_interval_record(lo, hi),
                approximate_midpoint_mean=statistics.mean(mids),
                approximate_midpoint_mean_standard_error=statistics.stdev(mids)/math.sqrt(expected_count),
                mean_signed_width=float(hi-lo),
                maximum_signed_width=float(max(upper-lower for lower, upper in endpoints)),
                midpoint_mean_numerical_error_bound=float((hi-lo)/2)), endpoints


def _replicated_check_summary(row, expected, *, signed=False):
    if _replicated_interval(row['mean_signed_gap_certificate'], signed=signed) != _replicated_interval(
            expected['mean_signed_gap_certificate'], signed=signed):
        raise ValueError('The mean certificate differs from the exact path aggregation')
    for key in expected:
        if key != 'mean_signed_gap_certificate':
            _replicated_equal(row[key], expected[key], 'The path-derived statistic differs: '+key)


def _replicated_aggregates(cases):
    """Use fitting units for sampling summaries, and exact intervals for arithmetic."""
    methods = [('hinge', name) for name in ('cold_uniform', 'adaptive')]
    methods += [(degree, name) for degree in ('2', '3')
                for name in ('polynomial_all_dates', 'hybrid_exact_terminal')]
    values, paired, costs = {}, {}, {}
    for degree, name in methods:
        rows = [case['strategies'][name] if degree == 'hinge' else case['comparators'][degree][name] for case in cases]
        key = degree+'/'+name
        values[key] = _replicated_distribution([row['approximate_midpoint_mean'] for row in rows])
        bounds = [_replicated_interval(row['mean_signed_gap_certificate']) for row in rows]
        values[key]['arithmetic_mean_certificate'] = _replicated_interval_record(
            sum(lo for lo, hi in bounds)/8, sum(hi for lo, hi in bounds)/8)
        field = ('combined_preparation_and_assessment_cpu_seconds' if degree == 'hinge'
                 else 'standalone_verified_workflow_cpu_seconds')
        costs[key] = _replicated_ratio_summary([row[field] for row in rows])
        if degree != 'hinge':
            differences = [case['paired_gap_differences'][degree][name] for case in cases]
            paired[key] = _replicated_distribution([row['approximate_midpoint_mean'] for row in differences])
            bounds = [_replicated_interval(row['mean_signed_gap_certificate'], signed=True) for row in differences]
            paired[key]['arithmetic_mean_certificate'] = _replicated_interval_record(
                sum(lo for lo, hi in bounds)/8, sum(hi for lo, hi in bounds)/8)
            paired[key]['mean_within_fit_standard_error'] = sum(
                row['approximate_midpoint_mean_standard_error'] for row in differences)/8
            paired[key]['mean_numerical_error_allowance'] = float(sum(hi-lo for lo, hi in bounds)/16)
    return dict(gap_midpoint_by_method=values,
                paired_comparator_minus_adaptive=paired,
                standalone_verified_budget_cpu_seconds=costs)


def build_replicated_refinement(primary_summary, *, fit_artifacts, design):
    """Extract reviewed aggregates from eight complete private fitting-unit ledgers.

    The caller verifies file hashes and the original execution completion records.
    This check verifies matching retained identities, path arithmetic and budgets.
    No simulated states, fitted coefficients or path-level arrays are returned.
    """
    import copy
    result = copy.deepcopy(primary_summary)
    result['design'] = {key: copy.deepcopy(design[key]) for key in (
        'dimension', 'initial_spot', 'training_paths', 'value_paths', 'outer_paths_per_fit',
        'precision_bits', 'coarse_conditional_width', 'fits', 'source_sha256',
        'public_preregistration_claim', 'new_general_theorem_claim')}
    if set(fit_artifacts) != set(range(8)):
        raise ValueError('All eight complete fitting-unit ledgers are required')
    if len(result['cases']) != 8:
        raise ValueError('The primary aggregate must contain eight fitting units')
    for index, case in enumerate(result['cases']):
        artifacts = fit_artifacts[index]
        primary, polynomial, verified = (artifacts[key] for key in ('refinement', 'polynomial', 'verified_polynomial'))
        if {key: value for key, value in primary.items() if key != 'observations'} != case:
            raise ValueError('The primary summary differs from its complete retained result')
        for record in (polynomial, verified):
            for key in ('input_sha256', 'design_sha256', 'source_sha256', 'fitting_unit_index', 'root_seed'):
                if record.get(key) != case[key]:
                    raise ValueError('A comparator changes the registered input, source or fitting unit')
        if (polynomial.get('sample_count') != 256 or polynomial.get('value_fitting_paths') != 50000
                or polynomial.get('value_spawn_key') != [1]):
            raise ValueError('The comparator fitting or evaluation stream differs')
        for flag in ('frozen_policy_preserved', 'supplied_outer_stream_reproduced', 'no_clipping_of_fitted_values'):
            if polynomial.get(flag) is not True:
                raise ValueError('The ordinary comparator does not preserve its frozen policy or stream')
        if polynomial.get('numerical_certificate') is not False or polynomial.get('statistical_coverage_certificate') is not False:
            raise ValueError('An ordinary floating-point comparator cannot claim certification')
        for flag in ('observed_policy_decisions_checked', 'independent_arm_costs', 'exact_terminal_model_discount'):
            if verified.get(flag) is not True:
                raise ValueError('The verified comparator omits an assessment requirement')
        if verified.get('new_general_theorem_claim') is not False:
            raise ValueError('Established comparator mathematics cannot be presented as a new theorem')
        if (_replicated_interval(verified['path_gap_width_target']) != (Fraction(1,100), Fraction(1,100))
                or verified['outer_paths'] != 256 or verified['precision_bits'] != 96):
            raise ValueError('The verified comparator changes its numerical target or path count')
        case['polynomial_degree_order'] = [int(value) for value in polynomial['degrees']]
        if list(verified['degrees']) != list(polynomial['degrees']):
            raise ValueError('Verified and ordinary polynomial execution orders differ')
        case['verified_method_order'] = verified['verified_method_order']
        case['polynomial_costs'] = dict(
            shared_value_stream_and_target_cpu_seconds=polynomial['shared_value_stream_and_target_cpu_seconds'],
            stream_and_policy_reconstruction_cpu_seconds=polynomial['stream_and_policy_reconstruction_cpu_seconds'],
            degrees={})
        case['verified_input_and_policy_validation_cpu_seconds'] = verified['input_and_policy_validation_cpu_seconds']
        case['comparators'], case['paired_gap_differences'] = {}, {}
        primary_bounds = {}
        for name in ('cold_uniform', 'adaptive'):
            expected, endpoints = _replicated_path_summary(primary['observations'][name])
            _replicated_check_summary(case['strategies'][name], expected)
            primary_bounds[name] = endpoints
            for count_key in ('conditional_integrals', 'integrand_evaluations'):
                counts = [row['total_work'][count_key] for row in primary['observations'][name]]
                if any(type(value) is not int or value < 0 for value in counts):
                    raise ValueError('Primary numerical-work counts must be non-negative integers')
                total = sum(counts)
                if case['strategies'][name][count_key] != total:
                    raise ValueError('Primary work counts differ from their complete path ledger')
        if any(max(a[0], b[0]) > min(a[1], b[1]) for a, b in zip(
                primary_bounds['cold_uniform'], primary_bounds['adaptive'])):
            raise ValueError('The paired hinge path certificates do not overlap')
        for degree, exported in polynomial['degrees'].items():
            case['comparators'][degree], case['paired_gap_differences'][degree] = {}, {}
            cost_keys = ('polynomial_fit_cpu_seconds', 'polynomial_interior_fit_cpu_seconds',
                'terminal_polynomial_fit_cpu_seconds', 'dual_assessment_cpu_seconds',
                'common_policy_and_outer_preparation_cpu_seconds', 'standalone_polynomial_pricing_cpu_seconds',
                'standalone_polynomial_pricing_plus_validation_cpu_seconds', 'hybrid_setup_pricing_cpu_seconds')
            case['polynomial_costs']['degrees'][degree] = {key: exported[key] for key in cost_keys}
            methods = verified['degrees'][degree]['methods']
            if list(methods) != case['verified_method_order']:
                raise ValueError('Verified method execution order differs across degrees')
            for name, row in methods.items():
                rows = row['observations']
                expected, endpoints = _replicated_path_summary(rows)
                _replicated_check_summary(row, expected)
                for count_key in ('conditional_integrals', 'integrand_evaluations', 'polynomial_terms_evaluated',
                        'moment_factor_exponentials', 'moment_factor_cache_builds', 'moment_factor_cache_hits', 'moment_factors_reused'):
                    counts = [value[count_key] for value in rows]
                    if any(type(value) is not int or value < 0 for value in counts):
                        raise ValueError('Verified numerical-work counts must be non-negative integers')
                    if row[count_key] != sum(counts):
                        raise ValueError('Verified work counts differ from their complete path ledger')
                paired_rows = []
                for path_index, (current, adaptive) in enumerate(zip(rows, primary['observations']['adaptive'])):
                    if current['stopping_index'] != adaptive['stopping_index']:
                        raise ValueError('A paired comparison changes its frozen stopping decision')
                    if _replicated_interval(current['selected_date_gap_certificate']) != (0, 0):
                        raise ValueError('The selected-date gap must cancel exactly')
                    clo, chi = endpoints[path_index]
                    alo, ahi = primary_bounds['adaptive'][path_index]
                    paired_rows.append(dict(path_index=path_index, stopping_index=current['stopping_index'],
                        signed_gap_certificate=_replicated_interval_record(clo-ahi, chi-alo)))
                paired_summary, _ = _replicated_path_summary(paired_rows, signed=True)
                case['paired_gap_differences'][degree][name] = paired_summary
                case['comparators'][degree][name] = {key: copy.deepcopy(value) for key, value in row.items()
                                                   if key != 'observations'}
    result.update(established_comparators_executed=True,
        comparison_scope='Comparator gap minus adaptive hinge gap for the same frozen policy identifies relative expected dual slack; neither absolute policy shortfall nor an improved exercise policy is identified.',
        mean_certificate_scope='Arithmetic enclosures of retained sample means at exact binary input states. They do not enclose population means, simulator law or statistical coverage.',
        sampling_scope='Across-fit standard errors use eight independently fitted policy/value-function pairs. Within-fit standard errors of paired interval midpoints are approximate conditional diagnostics; 2,048 paths are not 2,048 independent fitted models.',
        cost_scope='Hinge budgets charge their complete preparation and assessment including policy checks. Polynomial verified budgets charge common policy and outer preparation, their own value simulation, targets and appropriate regression, their independently executed verified assessment, and policy validation. Hybrid interior setup is reconstructed from measured phases; file I/O and process start-up are excluded.')
    result.update(_replicated_aggregates(result['cases']))
    validate_replicated_refinement(result)
    return result


def validate_replicated_refinement(study):
    """Reject altered fit identities, arithmetic, orders and standalone budgets."""
    if not isinstance(study, dict):
        raise ValueError('A replicated-study mapping is required')
    for key, expected in (('completed_fits',8), ('fitting_units',8), ('outer_paths_per_fit',256)):
        if type(study[key]) is not int or study[key] != expected:
            raise ValueError('The replicated study does not contain eight complete fitting units')
    design = study['design']
    for key, expected in (('dimension',5), ('training_paths',50000), ('value_paths',50000),
                          ('outer_paths_per_fit',256), ('precision_bits',96)):
        if type(design[key]) is not int or design[key] != expected:
            raise ValueError('The registered counts or model dimension differ')
    if (design['initial_spot'] != 100. or design['coarse_conditional_width'] != '1'
            or design['public_preregistration_claim'] is not False or design['new_general_theorem_claim'] is not False
            or study['established_comparators_executed'] is not True):
        raise ValueError('The replicated study changes its declared scope or model')
    for key, words in (('comparison_scope', ('relative expected dual slack', 'absolute policy shortfall')),
                       ('mean_certificate_scope', ('retained sample means', 'population means')),
                       ('sampling_scope', ('eight independently fitted', '2,048 paths')),
                       ('cost_scope', ('own value simulation', 'policy validation', 'reconstructed'))):
        if not isinstance(study.get(key), str) or any(word not in study[key] for word in words):
            raise ValueError('The replicated summary omits its interpretation or scope: '+key)
    if len(study['cases']) != 8 or len(design['fits']) != 8:
        raise ValueError('Missing or duplicated fitting units')
    source = study['source_sha256']
    if source != design['source_sha256'] or not source:
        raise ValueError('The declared registered source identities differ')
    def sha(value):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('A complete SHA-256 identity is required')
    sha(study['design_sha256'])
    for value in source.values():
        sha(value)
    seeds, snapshots, inputs = [], [], []
    ratio_values = {key: [] for key in ('integrals','certificate_phase_cpu','complete_assessment_cpu','charged_pricing_budget_cpu')}
    def cost(value):
        if _replicated_scalar(value) < -1e-12:
            raise ValueError('A standalone cost must be non-negative')
        return value
    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError('Numerical-work counts must be non-negative integers')
    for index, (case, fit) in enumerate(zip(study['cases'], design['fits'])):
        order = ['cold_uniform','adaptive'] if index%2 == 0 else ['adaptive','cold_uniform']
        degrees = [2,3] if index%2 == 0 else [3,2]
        methods = ['polynomial_all_dates','hybrid_exact_terminal'] if index%2 == 0 else ['hybrid_exact_terminal','polynomial_all_dates']
        if (type(case['fitting_unit_index']) is not int or case['fitting_unit_index'] != index
                or type(case['root_seed']) is not int or case['root_seed'] != 2026100410+index
                or fit != dict(index=index,seed=case['root_seed'],directory=f'fit_{index:02d}',
                    case=dict(name=f'fit_{index:02d}',target='1/100',order=order),
                    polynomial_degree_order=degrees,verified_polynomial_method_order=methods)
                or case['name'] != f'fit_{index:02d}' or case['strategy_order'] != order
                or case['polynomial_degree_order'] != degrees or case['verified_method_order'] != methods
                or case['outer_paths'] != 256 or case['precision_bits'] != 96
                or case['required_maximum_path_gap_width'] != '1/100'):
            raise ValueError('Fitting-unit seed, count, accuracy target or balanced order differs')
        if case['source_sha256'] != source or case['design_sha256'] != study['design_sha256']:
            raise ValueError('A fitting unit changes the common registered sources or design')
        for key in ('input_sha256','frozen_model_policy_values_sha256'):
            sha(case[key])
        seeds.append(case['root_seed']); inputs.append(case['input_sha256']); snapshots.append(case['frozen_model_policy_values_sha256'])
        if case['all_path_certificates_overlap'] is not True:
            raise ValueError('The paired hinge path overlap was not established')
        phases = case['preparation_phases']
        if set(phases) != {'policy_training','hinge_value_stream_and_targets','hinge_regression',
                           'outer_evaluation_and_record_construction','preparation_bookkeeping'}:
            raise ValueError('Complete exclusive preparation phases are required')
        for name in ('cpu_seconds','wall_seconds'):
            cost(case['preparation_cost'][name])
            for row in phases.values():
                cost(row[name])
            _replicated_equal(case['preparation_cost'][name],sum(row[name] for row in phases.values()),
                              'Preparation phases do not sum to the complete preparation')
        if set(case['strategies']) != {'cold_uniform','adaptive'}:
            raise ValueError('Both hinge strategies are required')
        bounds = []
        def certificate(row):
            lo, hi = _replicated_interval(row['mean_signed_gap_certificate'])
            _replicated_equal(row['mean_signed_width'],float(hi-lo),'Mean certificate width differs')
            _replicated_equal(row['midpoint_mean_numerical_error_bound'],float((hi-lo)/2),'Numerical error allowance differs')
            _replicated_equal(row['approximate_midpoint_mean'],float((lo+hi)/2),'Midpoint mean differs from its certificate')
            if (row['all_targets_met'] is not True or cost(row['maximum_signed_width']) > .01+1e-12
                    or row['mean_signed_width'] > row['maximum_signed_width']+1e-12):
                raise ValueError('The actual required path-gap width was not attained')
            cost(row['approximate_midpoint_mean_standard_error'])
            for field in ('conditional_integrals','integrand_evaluations'):
                count(row[field])
            return lo, hi
        for row in case['strategies'].values():
            bounds.append(certificate(row))
            if row['all_signed_inside_absolute'] is not True or row['mean_signed_width'] > row['mean_absolute_width']+1e-12:
                raise ValueError('Signed cancellation leaves the absolute certificate')
            for field in ('assessment_cpu_seconds','input_and_policy_validation_cpu_seconds',
                          'certificate_phase_cpu_seconds','combined_preparation_and_assessment_cpu_seconds'):
                cost(row[field])
            _replicated_equal(row['certificate_phase_cpu_seconds'],row['assessment_cpu_seconds']-row['input_and_policy_validation_cpu_seconds'],
                              'Certificate phase omits or duplicates validation')
            _replicated_equal(row['combined_preparation_and_assessment_cpu_seconds'],case['preparation_cost']['cpu_seconds']+row['assessment_cpu_seconds'],
                              'Hinge budget omits or duplicates preparation')
        if max(lo for lo,hi in bounds) > min(hi for lo,hi in bounds):
            raise ValueError('Same-gain mean certificates do not overlap')
        cold, adaptive = (case['strategies'][name] for name in ('cold_uniform','adaptive'))
        for key, field in (('integrals','conditional_integrals'),('certificate_phase_cpu','certificate_phase_cpu_seconds'),
                           ('complete_assessment_cpu','assessment_cpu_seconds'),('charged_pricing_budget_cpu','combined_preparation_and_assessment_cpu_seconds')):
            if cold[field] <= 0:
                raise ValueError('A paired ratio denominator must be positive')
            ratio_values[key].append(adaptive[field]/cold[field])
        for key, value in (('adaptive_over_cold_uniform_integral_ratio',ratio_values['integrals'][-1]),
                           ('adaptive_over_cold_uniform_certificate_phase_cpu_ratio',ratio_values['certificate_phase_cpu'][-1])):
            _replicated_equal(case[key],value,'A per-fit adaptive ratio differs')
        polynomial_costs = case['polynomial_costs']
        shared = cost(polynomial_costs['shared_value_stream_and_target_cpu_seconds'])
        reconstruction = cost(polynomial_costs['stream_and_policy_reconstruction_cpu_seconds'])
        validation = cost(case['verified_input_and_policy_validation_cpu_seconds'])
        if (set(case['comparators']) != {'2','3'} or set(case['paired_gap_differences']) != {'2','3'}
                or set(polynomial_costs['degrees']) != {'2','3'}):
            raise ValueError('Both complete polynomial degrees are required')
        common = phases['policy_training']['cpu_seconds']+phases['outer_evaluation_and_record_construction']['cpu_seconds']
        for degree in ('2','3'):
            costs = polynomial_costs['degrees'][degree]
            for value in costs.values():
                cost(value)
            _replicated_equal(costs['polynomial_fit_cpu_seconds'],costs['polynomial_interior_fit_cpu_seconds']+costs['terminal_polynomial_fit_cpu_seconds'],
                              'Interior and terminal polynomial fitting do not sum')
            _replicated_equal(costs['common_policy_and_outer_preparation_cpu_seconds'],common,'Polynomial common preparation includes redundant hinge fitting')
            ordinary = common+shared+costs['polynomial_fit_cpu_seconds']+costs['dual_assessment_cpu_seconds']
            hybrid = common+shared+costs['polynomial_interior_fit_cpu_seconds']
            _replicated_equal(costs['standalone_polynomial_pricing_cpu_seconds'],ordinary,'Ordinary polynomial standalone budget differs')
            _replicated_equal(costs['standalone_polynomial_pricing_plus_validation_cpu_seconds'],ordinary+reconstruction,'Ordinary polynomial reconstruction is charged incorrectly')
            _replicated_equal(costs['hybrid_setup_pricing_cpu_seconds'],hybrid,'Hybrid setup includes a redundant terminal regression')
            if list(case['comparators'][degree]) != methods or set(case['paired_gap_differences'][degree]) != set(methods):
                raise ValueError('Verified comparator methods or balanced execution order differ')
            for name,row in case['comparators'][degree].items():
                clo, chi = certificate(row)
                if row['method'] != name or row['degree'] != int(degree) or row['outer_paths'] != 256:
                    raise ValueError('A verified comparator changes its degree, method or path count')
                setup = hybrid if name == 'hybrid_exact_terminal' else ordinary-costs['dual_assessment_cpu_seconds']
                for field,value in (('retained_setup_pricing_cpu_seconds',setup),
                        ('standalone_assessment_cpu_seconds_including_validation',validation+row['assessment_cpu_seconds']),
                        ('standalone_pricing_cpu_seconds',setup+row['assessment_cpu_seconds']),
                        ('standalone_verified_workflow_cpu_seconds',setup+row['assessment_cpu_seconds']+validation)):
                    _replicated_equal(row[field],value,'Verified standalone budget omits or duplicates fitting or validation')
                for field in ('assessment_cpu_seconds','standalone_verified_workflow_cpu_seconds'):
                    cost(row[field])
                for field in ('polynomial_terms_evaluated','moment_factor_exponentials','moment_factor_cache_builds','moment_factor_cache_hits','moment_factors_reused','precision_refinements'):
                    count(row[field])
                if name == 'polynomial_all_dates' and row['conditional_integrals'] != 0:
                    raise ValueError('Exact-moment all-polynomial assessment cannot charge terminal integrals')
                if name == 'hybrid_exact_terminal' and row['conditional_integrals'] < 256:
                    raise ValueError('Every hybrid path requires its own terminal query')
                paired = case['paired_gap_differences'][degree][name]
                plo, phi = _replicated_interval(paired['mean_signed_gap_certificate'],signed=True)
                alo, ahi = _replicated_interval(adaptive['mean_signed_gap_certificate'])
                if (plo,phi) != (clo-ahi,chi-alo):
                    raise ValueError('The paired mean interval differs from comparator minus adaptive')
                _replicated_equal(paired['approximate_midpoint_mean'],float((plo+phi)/2),'Paired midpoint mean differs')
                _replicated_equal(paired['mean_signed_width'],float(phi-plo),'Paired mean width differs')
                _replicated_equal(paired['midpoint_mean_numerical_error_bound'],float((phi-plo)/2),'Paired numerical allowance differs')
                cost(paired['approximate_midpoint_mean_standard_error'])
                if (paired['maximum_signed_width'] < paired['mean_signed_width']-1e-12
                        or paired['maximum_signed_width'] > .02+1e-12):
                    raise ValueError('Paired arithmetic width exceeds its two certified inputs')
    if len(set(seeds)) != 8 or len(set(inputs)) != 8 or len(set(snapshots)) != 8 or study['distinct_frozen_model_policy_value_hashes'] != 8:
        raise ValueError('Eight distinct fitted units, inputs and frozen identities are required')
    for key, values in ratio_values.items():
        _replicated_check_mapping(study['paired_adaptive_over_cold_uniform_ratios'][key],_replicated_ratio_summary(values),'A paired fitting-unit ratio range differs')
    expected = _replicated_aggregates(study['cases'])
    for field in expected:
        if set(study[field]) != set(expected[field]):
            raise ValueError('Missing or duplicated method aggregates')
        for name, row in expected[field].items():
            found = study[field][name]
            if set(found) != set(row):
                raise ValueError('An aggregate has unexpected or omitted summary fields')
            for key,value in row.items():
                if key == 'arithmetic_mean_certificate':
                    signed = field == 'paired_comparator_minus_adaptive'
                    if _replicated_interval(found[key],signed=signed) != _replicated_interval(value,signed=signed):
                        raise ValueError('An across-fit arithmetic mean certificate differs')
                elif key == 'fitting_units':
                    if type(found[key]) is not int or found[key] != 8:
                        raise ValueError('Sampling summaries must use eight independent fitting units')
                else:
                    _replicated_equal(found[key],value,'An across-fit mean, standard error or range differs')
    return True


def render_replicated_refinement(study):
    validate_replicated_refinement(study)
    rows = []
    labels = [('hinge/cold_uniform','Hinge, uniform'),('hinge/adaptive','Hinge, adaptive'),
              ('2/polynomial_all_dates','Degree two, polynomial'),('2/hybrid_exact_terminal','Degree two, hybrid'),
              ('3/polynomial_all_dates','Degree three, polynomial'),('3/hybrid_exact_terminal','Degree three, hybrid')]
    for name,label in labels:
        gap = study['gap_midpoint_by_method'][name]
        budget = study['standalone_verified_budget_cpu_seconds'][name]
        difference = study['paired_comparator_minus_adaptive'].get(name)
        paired = '--' if difference is None else f"{difference['mean']:.4f} ({difference['standard_error']:.4f})"
        rows.append([label,f"{gap['mean']:.4f}",f"{gap['standard_error']:.4f}",paired,f"{budget['median']:.3f}"])
    return table('Repeated-fit verified dual assessment', 'revision-replicated-refinement',
        ['Assessment','Mean gap','Fit SE',r'$\Delta$ (fit SE)','CPU (s)'],rows,
        'Eight independent fits of the five-asset benchmark use 50,000 policy-training paths, 50,000 value-fitting paths and 256 independent outer paths each. Every assessment attains maximum path-gap width 0.01. Fit SE uses eight fitting-unit means. The paired difference is comparator minus adaptive hinge for the same frozen policy and measures relative expected dual slack, not policy improvement. CPU is the median complete standalone budget including preparation and policy validation; hybrid setup uses measured component costs. Arithmetic sample-mean enclosures and within-fit diagnostics are retained separately.')


def render_replicated_tradeoff(study):
    """Transcribe six measured cost/quality points and their distinct dispersions."""
    validate_replicated_refinement(study)
    methods = (
        ('hinge/cold_uniform', 'Uniform hinge', 'square*', 'qfblue', ''),
        ('hinge/adaptive', 'Adaptive hinge', '*', 'qfgreen', ''),
        ('2/polynomial_all_dates', 'Polynomial, degree 2', 'triangle*', 'qforange', ''),
        ('2/hybrid_exact_terminal', 'Hybrid, degree 2', 'diamond*', 'qfamber', ''),
        ('3/polynomial_all_dates', 'Polynomial, degree 3', 'triangle*', 'qfpurple',
         ',mark options={rotate=180}'),
        ('3/hybrid_exact_terminal', 'Hybrid, degree 3', 'pentagon*', 'qfviolet', ''),
    )
    axis = (r'width=.97\textwidth,height=6.4cm,'
        r'xlabel={Preparation and assessment CPU (seconds)},ylabel={Mean stopped gap (price units)},'
        r'xmin=0,ymin=0,grid=major,grid style={gray!20},tick label style={font=\small},'
        r'label style={font=\small},legend columns=2,legend style={at={(0.5,-0.30)},anchor=north,'
        r'draw=none,font=\footnotesize,cells={anchor=west}},'
        r'enlarge x limits={upper,value=.05},enlarge y limits={upper,value=.07}')
    source = [r'\begin{figure}[!htb]', r'\centering', r'\begin{tikzpicture}',
              r'\begin{axis}['+axis+']']
    for name, label, mark, colour, marker_options in methods:
        gap = study['gap_midpoint_by_method'][name]
        budget = study['standalone_verified_budget_cpu_seconds'][name]
        values = (budget['median'], gap['mean'], budget['median']-budget['minimum'],
                  budget['maximum']-budget['median'], gap['standard_error'])
        options = ('only marks,mark='+mark+',mark size=2.8pt,color='+colour+marker_options
            +',error bars/.cd,x dir=both,x explicit,y dir=both,y explicit,'
            'error bar style={line width=0.55pt}')
        source.extend([
            r'\addplot+['+options+r'] table[x=cpu,y=gap,x error minus=cpu_minus,'
            r'x error plus=cpu_plus,y error=se] {',
            'cpu gap cpu_minus cpu_plus se',
            ' '.join(format(value, '.17g') for value in values), r'};',
            r'\addlegendentry{'+label+'}'])
    source.extend([r'\end{axis}', r'\end{tikzpicture}',
        r'\caption{Cost and quality of verified dual assessments across eight independent fits. '
        r'Points show the median preparation and assessment budget and mean stopped gap. Horizontal bars '
        r'span observed minimum and maximum CPU; vertical bars give one across-fit standard '
        r'error of the gap mean. These ranges and standard errors are not confidence intervals. '
        r'Budgets include policy validation. Process start-up and result serialisation are excluded. '
        r'Every method attains maximum '
        r'path-gap width 0.01.}',
        r'\label{fig:revision-replicated-tradeoff}', r'\end{figure}'])
    return '\n'.join(source)


FROZEN_EXTENSION_ARMS = ('cold_uniform', 'adaptive', 'hybrid_exact_terminal')


def _frozen_extension_aggregates(cases):
    """Separate retained-fit dispersion from conditional fresh-sample precision."""
    import math
    gap, assessment, fresh, standalone, prior_precision = {}, {}, {}, {}, {}
    for name in FROZEN_EXTENSION_ARMS:
        rows = [case['arms'][name] for case in cases]
        gap[name] = _replicated_distribution([row['approximate_midpoint_mean'] for row in rows])
        bounds = [_replicated_interval(row['mean_signed_gap_certificate']) for row in rows]
        gap[name]['arithmetic_mean_certificate'] = _replicated_interval_record(
            sum(lo for lo, hi in bounds)/8, sum(hi for lo, hi in bounds)/8)
        gap[name]['conditional_average_standard_error'] = math.sqrt(sum(
            row['approximate_midpoint_mean_standard_error']**2 for row in rows))/8
        assessment[name] = _replicated_ratio_summary([
            row['incremental_assessment_cpu_seconds_including_validation'] for row in rows])
        fresh[name] = _replicated_ratio_summary([
            case['incremental_evaluation_cpu_seconds_by_arm'][name] for case in cases])
        standalone[name] = _replicated_ratio_summary([
            case['reconstructed_standalone_cpu_seconds_by_arm'][name] for case in cases])
        previous = [case['prior_sample_diagnostics'][name]['approximate_midpoint_mean_standard_error'] for case in cases]
        prior_precision[name] = dict(
            prior_conditional_average_standard_error=math.sqrt(sum(value*value for value in previous))/8,
            fresh_conditional_average_standard_error=gap[name]['conditional_average_standard_error'])
    differences = [case['paired_hybrid_minus_adaptive'] for case in cases]
    paired = _replicated_distribution([row['approximate_midpoint_mean'] for row in differences])
    bounds = [_replicated_interval(row['mean_difference_certificate'], signed=True) for row in differences]
    paired.update(arithmetic_mean_certificate=_replicated_interval_record(
        sum(lo for lo, hi in bounds)/8, sum(hi for lo, hi in bounds)/8),
        conditional_average_standard_error=math.sqrt(sum(
            row['approximate_midpoint_mean_standard_error']**2 for row in differences))/8)
    ratios = {}
    for key in ('assessment_cpu', 'reconstructed_standalone_cpu', 'conditional_integrals'):
        values = []
        for case in cases:
            if key == 'assessment_cpu':
                values.append(case['arms']['adaptive']['incremental_assessment_cpu_seconds_including_validation']/
                              case['arms']['cold_uniform']['incremental_assessment_cpu_seconds_including_validation'])
            elif key == 'reconstructed_standalone_cpu':
                values.append(case['reconstructed_standalone_cpu_seconds_by_arm']['adaptive']/
                              case['reconstructed_standalone_cpu_seconds_by_arm']['cold_uniform'])
            else:
                values.append(case['arms']['adaptive']['conditional_integrals']/
                              case['arms']['cold_uniform']['conditional_integrals'])
        ratios[key] = _replicated_ratio_summary(values)
    return dict(gap_midpoint_by_arm=gap, paired_hybrid_minus_adaptive=paired,
        incremental_assessment_cpu_seconds=assessment, incremental_evaluation_cpu_seconds=fresh,
        reconstructed_standalone_cpu_seconds=standalone,
        paired_adaptive_over_uniform_ratios=ratios, conditional_sample_precision=prior_precision)


def _frozen_extension_check_aggregates(actual, expected):
    """Compare numerical aggregate mappings, including authoritative endpoints."""
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError('A frozen-extension aggregate omits or adds a field')
    for key, value in expected.items():
        found = actual[key]
        if isinstance(value, dict):
            if set(value) == {'lower', 'upper', 'width'}:
                if _replicated_interval(found, signed=True) != _replicated_interval(value, signed=True):
                    raise ValueError('The exact frozen-extension mean interval differs')
            elif set(value) == {'numerator', 'denominator', 'approximation'}:
                if not isinstance(found, dict) or set(found) != set(value) or _replicated_rational(found) != _replicated_rational(value):
                    raise ValueError('An exact aggregate rational differs')
            else:
                _frozen_extension_check_aggregates(found, value)
        elif type(value) is int:
            if type(found) is not int or found != value:
                raise ValueError('A frozen-extension fitting-unit count differs')
        else:
            _replicated_equal(found, value, 'A frozen-extension mean, dispersion or cost differs')


def validate_frozen_extension(study, original_replicated_refinement=None):
    """Check a compact fresh-stream record without treating retained fits as new fits."""
    import itertools
    if not isinstance(study, dict):
        raise ValueError('A frozen-extension mapping is required')
    for field, expected in (('fitting_units',8),('completed_fits',8),('outer_paths_per_fit',1024),
                            ('original_outer_paths_per_fit',256),('precision_bits',96)):
        if type(study.get(field)) is not int or study[field] != expected:
            raise ValueError('The frozen extension changes its complete fit or sample counts')
    if (study.get('new_outer_spawn_key') != [3] or study.get('original_outer_spawn_key') != [2]
            or study.get('required_maximum_path_gap_width') != '1/100'
            or study.get('newly_trained_fits') != 0 or type(study.get('newly_trained_fits')) is not int
            or study.get('frozen_policy_and_functions_preserved') is not True
            or study.get('fresh_sample_is_prefix_of_original') is not False
            or study.get('statistical_coverage_certificate') is not False
            or study.get('new_general_theorem_claim') is not False):
        raise ValueError('The fresh-stream or retained-fit scope differs')
    for field, words in (
            ('sampling_scope', ('held fixed','selected','conditional','descriptive','not confidence intervals')),
            ('cost_scope', ('reconstructed','not rerun','new outer','validation')),
            ('mean_certificate_scope', ('retained sample means','population')),
            ('comparison_scope', ('same frozen policy','relative dual slack'))):
        if not isinstance(study.get(field), str) or any(word not in study[field].lower() for word in words):
            raise ValueError('The frozen extension omits its interpretation: '+field)
    def sha(value):
        if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
            raise ValueError('A complete frozen-extension SHA-256 identity is required')
    sha(study['design_sha256']); sha(study['original_design_sha256'])
    if not study['source_sha256']:
        raise ValueError('The frozen extension must retain its source identities')
    for value in study['source_sha256'].values():
        sha(value)
    if len(study['cases']) != 8:
        raise ValueError('All eight retained fits are required')
    original_cases = None
    if original_replicated_refinement is not None:
        validate_replicated_refinement(original_replicated_refinement)
        if study['original_design_sha256'] != original_replicated_refinement['design_sha256']:
            raise ValueError('The frozen extension changes its original fit design')
        original_cases = original_replicated_refinement['cases']
    orders, identities, seeds, new_inputs = [], [], [], []
    def cost(value):
        if _replicated_scalar(value) < 0:
            raise ValueError('A frozen-extension cost or standard error is negative')
        return value
    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError('A frozen-extension count must be a non-negative integer')
    for index, case in enumerate(study['cases']):
        if (type(case['fitting_unit_index']) is not int or case['fitting_unit_index'] != index
                or type(case['root_seed']) is not int or case['root_seed'] != 2026100410+index
                or case['outer_paths'] != 1024 or type(case['outer_paths']) is not int
                or case['original_outer_spawn_key'] != [2] or case['new_outer_spawn_key'] != [3]
                or case['hinge_path_certificates_overlap'] is not True
                or case['design_sha256'] != study['design_sha256'] or case['source_sha256'] != study['source_sha256']):
            raise ValueError('A retained fitting unit changes its identity or fresh stream')
        order = case['arm_order']
        if len(order) != 3 or set(order) != set(FROZEN_EXTENSION_ARMS):
            raise ValueError('Every retained fit requires three distinct executed arms')
        orders.append(tuple(order)); seeds.append(case['root_seed'])
        for field in ('input_sha256','original_input_sha256','frozen_model_policy_values_sha256',
                      'frozen_degree_three_polynomial_sha256','polynomial_view_sha256'):
            sha(case[field])
        if case['input_sha256'] == case['original_input_sha256']:
            raise ValueError('The fresh sample cannot reuse its original input record')
        identities.append(case['frozen_model_policy_values_sha256']); new_inputs.append(case['input_sha256'])
        if original_cases is not None:
            old = original_cases[index]
            if (case['root_seed'] != old['root_seed'] or case['original_input_sha256'] != old['input_sha256']
                    or case['frozen_model_policy_values_sha256'] != old['frozen_model_policy_values_sha256']):
                raise ValueError('The fresh record does not match its original retained fit')
        retained = case['original_setup_measurements']
        for field in ('original_preparation_cpu_seconds','original_outer_preparation_cpu_seconds',
                      'original_hybrid_setup_cpu_seconds'):
            cost(retained[field])
        if original_cases is not None:
            expected_retained = dict(original_preparation_cpu_seconds=old['preparation_cost']['cpu_seconds'],
                original_outer_preparation_cpu_seconds=old['preparation_phases']['outer_evaluation_and_record_construction']['cpu_seconds'],
                original_hybrid_setup_cpu_seconds=old['polynomial_costs']['degrees']['3']['hybrid_setup_pricing_cpu_seconds'])
            _frozen_extension_check_aggregates(retained, expected_retained)
        outer = case['new_outer_preparation_cost']
        for value in outer.values():
            cost(value)
        new_outer = outer['cpu_seconds']
        hinge_setup = retained['original_preparation_cpu_seconds']-retained['original_outer_preparation_cpu_seconds']+new_outer
        hybrid_setup = retained['original_hybrid_setup_cpu_seconds']-retained['original_outer_preparation_cpu_seconds']+new_outer
        expected_setup = dict(cold_uniform=hinge_setup,adaptive=hinge_setup,hybrid_exact_terminal=hybrid_setup)
        _frozen_extension_check_aggregates(case['charged_setup_cpu_seconds_by_arm'], expected_setup)
        if set(case['arms']) != set(FROZEN_EXTENSION_ARMS) or set(case['prior_sample_diagnostics']) != set(FROZEN_EXTENSION_ARMS):
            raise ValueError('A frozen-extension assessment or prior diagnostic is omitted')
        arm_bounds = {}
        for name,row in case['arms'].items():
            lo, hi = _replicated_interval(row['mean_signed_gap_certificate'])
            arm_bounds[name] = (lo,hi)
            for field, value in (('mean_signed_width',float(hi-lo)),
                    ('approximate_midpoint_mean',float((lo+hi)/2)),
                    ('midpoint_mean_numerical_error_bound',float((hi-lo)/2))):
                _replicated_equal(row[field],value,'A fresh-sample certificate display differs')
            if (row['all_targets_met'] is not True or cost(row['maximum_signed_width']) > .01+1e-12
                    or row['mean_signed_width'] > row['maximum_signed_width']+1e-12):
                raise ValueError('The fresh path certificates do not attain their required width')
            cost(row['approximate_midpoint_mean_standard_error'])
            for field in ('conditional_integrals','integrand_evaluations'):
                count(row[field])
            for field in ('incremental_assessment_cpu_seconds_including_validation',
                          'input_and_policy_validation_cpu_seconds','certificate_phase_cpu_seconds'):
                cost(row[field])
            _replicated_equal(row['incremental_assessment_cpu_seconds_including_validation'],
                row['input_and_policy_validation_cpu_seconds']+row['certificate_phase_cpu_seconds'],
                'Fresh assessment omits or duplicates policy validation')
            _replicated_equal(case['incremental_evaluation_cpu_seconds_by_arm'][name],
                new_outer+row['incremental_assessment_cpu_seconds_including_validation'],
                'Fresh evaluation omits or duplicates newly executed outer preparation')
            _replicated_equal(case['reconstructed_standalone_cpu_seconds_by_arm'][name],
                expected_setup[name]+row['incremental_assessment_cpu_seconds_including_validation'],
                'Reconstructed standalone budget omits or duplicates setup or validation')
            prior = case['prior_sample_diagnostics'][name]
            cost(prior['approximate_midpoint_mean_standard_error'])
            _replicated_scalar(prior['approximate_midpoint_mean'])
            if original_cases is not None:
                expected_prior = old['comparators']['3']['hybrid_exact_terminal'] if name=='hybrid_exact_terminal' else old['strategies'][name]
                for field in ('approximate_midpoint_mean','approximate_midpoint_mean_standard_error'):
                    _replicated_equal(prior[field],expected_prior[field],'A prior sample diagnostic changes its retained fit')
        if max(lo for lo,hi in (arm_bounds['cold_uniform'],arm_bounds['adaptive'])) > min(hi for lo,hi in (arm_bounds['cold_uniform'],arm_bounds['adaptive'])):
            raise ValueError('The two same-function hinge sample-mean certificates do not overlap')
        paired = case['paired_hybrid_minus_adaptive']
        plo, phi = _replicated_interval(paired['mean_difference_certificate'], signed=True)
        hlo,hhi = arm_bounds['hybrid_exact_terminal']; alo,ahi = arm_bounds['adaptive']
        if (plo,phi) != (hlo-ahi,hhi-alo):
            raise ValueError('The exact paired mean difference changes its subtraction order')
        for field,value in (('approximate_midpoint_mean',float((plo+phi)/2)),
                            ('mean_signed_width',float(phi-plo)),
                            ('midpoint_mean_numerical_error_bound',float((phi-plo)/2))):
            _replicated_equal(paired[field],value,'The exact paired difference display differs')
        cost(paired['approximate_midpoint_mean_standard_error'])
        if (cost(paired['maximum_signed_width']) > .02+1e-12
                or paired['mean_signed_width'] > paired['maximum_signed_width']+1e-12):
            raise ValueError('A paired difference exceeds its two numerical input allowances')
        for name in ('cold_uniform',):
            if (case['arms'][name]['conditional_integrals'] <= 0 or
                    case['arms'][name]['incremental_assessment_cpu_seconds_including_validation'] <= 0 or
                    case['reconstructed_standalone_cpu_seconds_by_arm'][name] <= 0):
                raise ValueError('A fresh comparison ratio requires a positive denominator')
    if len(set(identities)) != 8 or len(set(seeds)) != 8 or len(set(new_inputs)) != 8:
        raise ValueError('The retained fitting units and fresh inputs must be distinct')
    if (set(orders[:6]) != set(itertools.permutations(FROZEN_EXTENSION_ARMS))
            or any(sorted(sum(order[position]==name for order in orders) for position in range(3)) != [2,3,3]
                   for name in FROZEN_EXTENSION_ARMS)
            or sum(order.index('cold_uniform') < order.index('adaptive') for order in orders) != 4):
        raise ValueError('The executed three-arm positions or hinge order are unbalanced')
    expected = _frozen_extension_aggregates(study['cases'])
    for field, value in expected.items():
        _frozen_extension_check_aggregates(study[field],value)
    return True


def render_frozen_extension(study):
    validate_frozen_extension(study)
    rows = []
    for name, label in (('cold_uniform','Hinge, uniform'),('adaptive','Hinge, adaptive'),
                        ('hybrid_exact_terminal','Degree three, hybrid')):
        gap = study['gap_midpoint_by_arm'][name]
        assessment = study['incremental_evaluation_cpu_seconds'][name]
        budget = study['reconstructed_standalone_cpu_seconds'][name]
        rows.append([label,f"{gap['mean']:.4f}",f"{gap['conditional_average_standard_error']:.4f}",
                     f"{assessment['median']:.3f}",f"{budget['median']:.3f}"])
    paired = study['paired_hybrid_minus_adaptive']
    return table('Fresh-sample assessment of retained five-asset fits', 'revision-frozen-extension',
        ['Assessment','Mean gap','Conditional SE','Fresh CPU (s)','Standalone CPU (s)'],rows,
        'Each of eight previously trained fits is held fixed and assessed on 1,024 fresh outer paths. '
        'The degree-three hybrid was selected from the earlier comparison; its new evaluation stream is independent of that sample. '
        'Across-fit dispersion is retained separately as a descriptive measure, not a confidence interval. '
        'Conditional SE is the square root of the sum of squared within-fit standard errors, divided by eight. '
        f"The paired hybrid-minus-adaptive mean is {paired['mean']:.4f}, with conditional SE {paired['conditional_average_standard_error']:.4f} "
        f"and descriptive across-fit SE {paired['standard_error']:.4f}. "
        'Both CPU columns report medians over eight fits. Fresh CPU includes new outer preparation and the complete assessment with policy validation. Standalone CPU reconstructs a budget by adding retained fitting and setup components to that fresh work; the original outer preparation is replaced and fitting is not rerun. '
        'All arms attain maximum path-gap width 0.01. Paired differences identify relative dual slack for the same frozen policy.')


def build_frozen_extension(primary_summary, *, fit_artifacts, design, original_replicated_refinement):
    """Review full fresh-stream ledgers, then retain only their compact summaries.

    The caller verifies file and execution hashes. This check independently
    recomputes every mean, paired difference and work total from path records,
    and checks the supplied frozen functions against their original inputs.
    """
    import copy
    import hashlib
    validate_replicated_refinement(original_replicated_refinement)
    if set(fit_artifacts) != set(range(8)) or len(primary_summary['cases']) != 8:
        raise ValueError('All eight complete fresh-stream ledgers are required')
    if (primary_summary['completed_fits'] != 8 or primary_summary['fitting_units'] != 8
            or primary_summary['outer_paths_per_fit'] != 1024 or primary_summary['total_new_outer_paths'] != 8192
            or primary_summary['fitting_and_value_streams_rerun'] is not False
            or primary_summary['original_setup_measurements_retained'] is not True):
        raise ValueError('The fresh summary changes its sample or retained-fit scope')
    if (design['source_sha256'] != primary_summary['source_sha256']
            or design['outer_paths_per_fit'] != 1024 or design['precision_bits'] != 96
            or design['maximum_path_gap_width'] != '1/100' or design['adaptive_coarse_width'] != '1'
            or design['new_outer_spawn_key'] != [3] or design['old_outer_spawn_key'] != [2]
            or design['fitting_spawn_keys'] != [[0],[1]] or design['training_or_value_streams_rerun'] is not False
            or design['recorded_before_new_outer_sampling'] is not True
            or design['public_preregistration_claim'] is not False or design['no_new_priority_claim'] is not True
            or len(design['fits']) != 8):
        raise ValueError('The fresh evaluation differs from its prior design')
    prior_source = design['source_study']
    if (prior_source['original_design_sha256'] != original_replicated_refinement['design_sha256']
            or prior_source['original_source_sha256'] != original_replicated_refinement['source_sha256']):
        raise ValueError('The fresh design changes the source of its retained fits')
    def canonical(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    def frozen(record):
        return {key:record[key] for key in ('model','knots','coefficients','frozen_policy')}
    result = dict(schema_version=1, fitting_units=8, completed_fits=8, newly_trained_fits=0,
        outer_paths_per_fit=1024, original_outer_paths_per_fit=256, precision_bits=96,
        required_maximum_path_gap_width='1/100', original_outer_spawn_key=[2],new_outer_spawn_key=[3],
        original_design_sha256=original_replicated_refinement['design_sha256'],
        design_sha256=primary_summary['design_sha256'],source_sha256=copy.deepcopy(primary_summary['source_sha256']),
        frozen_policy_and_functions_preserved=True, fresh_sample_is_prefix_of_original=False,
        statistical_coverage_certificate=False, new_general_theorem_claim=False,
        sampling_scope='Eight previously trained policy/value-function pairs are held fixed. The degree-three hybrid was selected from the preceding comparison before this fresh evaluation. Conditional standard errors use independent new streams given those fits and that selection. Across-fit standard errors are descriptive dispersion combining retained-fit heterogeneity and evaluation noise, not confidence intervals or new training replications.',
        cost_scope='New assessment and policy validation are measured anew. Full standalone budgets are reconstructed from retained setup phases, which are not rerun; the previous outer preparation is replaced by new outer preparation. Each arm charges its own validation. File serialisation and process start-up are excluded.',
        mean_certificate_scope='Exact arithmetic enclosures concern retained sample means at supplied binary states. They do not enclose population means, simulator law or statistical coverage.',
        comparison_scope='Hybrid gap minus adaptive hinge gap for the same frozen policy measures relative dual slack. A smaller assessor gap does not identify an improved exercise policy or absolute policy shortfall.',
        cases=[])
    for index in range(8):
        artifacts = fit_artifacts[index]
        ledger, record, view, old_record, old_poly = (artifacts[key] for key in
            ('result','input','polynomial_view','original_input','original_polynomial'))
        old = original_replicated_refinement['cases'][index]
        retained_functions = frozen(old_record)
        if (frozen(record) != retained_functions or canonical(retained_functions) != old['frozen_model_policy_values_sha256']
                or ledger['frozen_model_policy_values_sha256'] != old['frozen_model_policy_values_sha256']):
            raise ValueError('A fresh assessment changes its original model, policy or hinge functions')
        original_degree = {key:old_poly['degrees']['3'][key] for key in ('basis_exponents','coefficients')}
        view_degree = {key:view['degrees']['3'][key] for key in ('basis_exponents','coefficients')}
        degree_sha = canonical(original_degree)
        if (view_degree != original_degree or ledger['frozen_degree_three_polynomial_sha256'] != degree_sha
                or view['frozen_degree_three_polynomial_sha256'] != degree_sha):
            raise ValueError('A fresh hybrid changes its original polynomial basis or coefficients')
        if (record['original_input_sha256'] != old['input_sha256']
                or ledger['original_input_sha256'] != old['input_sha256']
                or prior_source['files'][f'fit_{index:02d}/inputs.json'] != old['input_sha256']):
            raise ValueError('A fresh fitting unit changes its original input identity')
        stream = record['design']
        if (stream['root_seed'] != old['root_seed'] or stream['outer_paths'] != 1024
                or stream['spawn_keys'] != [[0],[1],[3]] or stream['original_outer_spawn_key'] != [2]
                or stream['new_outer_spawn_key'] != [3] or stream['evaluation_independent_of_original_outer_stream'] is not True
                or stream['fitting_streams_not_rerun'] is not True
                or stream['training_paths'] != 50000 or stream['value_paths'] != 50000
                or ledger['precision_bits'] != 96 or ledger['target'] != '1/100'
                or view['sample_count'] != 1024 or view['value_spawn_key'] != [1]
                or view['value_fitting_paths'] != 50000 or len(record['states']) != 1024
                or len(record['stopping_indices']) != 1024):
            raise ValueError('The fresh stream is not a separate 1,024-path child-three evaluation')
        fit = design['fits'][index]
        if (ledger['fit_index'] != index or ledger['root_seed'] != old['root_seed']
                or ledger['arm_order'] != fit['arm_order'] or fit['index'] != index
                or fit['root_seed'] != old['root_seed'] or list(ledger['arms']) != ledger['arm_order']):
            raise ValueError('The actual fitting-unit identity or executed arm order differs')
        compact_source = copy.deepcopy(ledger)
        for row in compact_source['arms'].values():
            row.pop('observations')
        compact_source['paired_hybrid_minus_adaptive'].pop('observations')
        if primary_summary['cases'][index] != compact_source:
            raise ValueError('The completed primary summary differs from its full path ledger')
        case = {key:copy.deepcopy(ledger[key]) for key in
            ('root_seed','arm_order','input_sha256','original_input_sha256','polynomial_view_sha256',
             'design_sha256','source_sha256','frozen_model_policy_values_sha256',
             'frozen_degree_three_polynomial_sha256','original_outer_spawn_key','new_outer_spawn_key',
             'hinge_path_certificates_overlap','new_outer_preparation_cost','charged_setup_cpu_seconds_by_arm')}
        case.update(fitting_unit_index=index,outer_paths=1024,arms={},prior_sample_diagnostics={},
            reconstructed_standalone_cpu_seconds_by_arm=copy.deepcopy(ledger['standalone_pricing_cpu_seconds_by_arm']),
            incremental_evaluation_cpu_seconds_by_arm=copy.deepcopy(ledger['incremental_evaluation_cpu_seconds_by_arm']),
            original_setup_measurements=dict(
                original_preparation_cpu_seconds=old['preparation_cost']['cpu_seconds'],
                original_outer_preparation_cpu_seconds=old['preparation_phases']['outer_evaluation_and_record_construction']['cpu_seconds'],
                original_hybrid_setup_cpu_seconds=old['polynomial_costs']['degrees']['3']['hybrid_setup_pricing_cpu_seconds']))
        endpoints = {}
        for name in FROZEN_EXTENSION_ARMS:
            arm = ledger['arms'][name]
            rows = arm['observations']
            expected, bounds = _replicated_path_summary(rows,expected_count=1024)
            _replicated_check_summary(arm,expected)
            endpoints[name] = bounds
            for path_index,row in enumerate(rows):
                if row['stopping_index'] != record['stopping_indices'][path_index]:
                    raise ValueError('A fresh path changes its retained policy stopping decision')
                if name=='hybrid_exact_terminal':
                    if _replicated_interval(row['selected_date_gap_certificate']) != (0,0):
                        raise ValueError('The selected-date hybrid gap must cancel exactly')
                else:
                    lo,hi = bounds[path_index]
                    alo,ahi = _replicated_interval(row['absolute_gap_certificate'])
                    if not alo <= lo <= hi <= ahi:
                        raise ValueError('The signed hinge certificate leaves its absolute enclosure')
            counter_fields = ['conditional_integrals','integrand_evaluations']
            if name=='hybrid_exact_terminal':
                counter_fields += ['polynomial_terms_evaluated','moment_factor_exponentials','moment_factor_cache_builds',
                                   'moment_factor_cache_hits','moment_factors_reused']
            for field in counter_fields:
                work = [row[field] if name=='hybrid_exact_terminal' else row['total_work'][field] for row in rows]
                if any(type(value) is not int or value<0 for value in work) or arm[field] != sum(work):
                    raise ValueError('A fresh numerical-work count differs from its complete path ledger')
            case['arms'][name] = dict(expected,all_targets_met=arm['all_targets_met'],
                conditional_integrals=arm['conditional_integrals'],integrand_evaluations=arm['integrand_evaluations'],
                incremental_assessment_cpu_seconds_including_validation=arm['assessment_cpu_seconds'],
                input_and_policy_validation_cpu_seconds=arm['input_and_policy_validation_cpu_seconds'],
                certificate_phase_cpu_seconds=arm['certificate_phase_cpu_seconds'])
            prior_arm = old['comparators']['3']['hybrid_exact_terminal'] if name=='hybrid_exact_terminal' else old['strategies'][name]
            case['prior_sample_diagnostics'][name] = {key:prior_arm[key] for key in
                ('approximate_midpoint_mean','approximate_midpoint_mean_standard_error')}
            _replicated_equal(ledger['incremental_evaluation_cpu_seconds_by_arm'][name],
                ledger['new_outer_preparation_cost']['cpu_seconds']+arm['assessment_cpu_seconds'],
                'The incremental evaluation cost includes old fitting or omits fresh preparation')
        if any(max(a[0],u[0]) > min(a[1],u[1]) for a,u in zip(endpoints['adaptive'],endpoints['cold_uniform'])):
            raise ValueError('A matched pair of same-function hinge path certificates does not overlap')
        paired_rows = []
        original_paired = ledger['paired_hybrid_minus_adaptive']
        if len(original_paired['observations']) != 1024:
            raise ValueError('Every paired fresh-path difference is required')
        for path_index, ((hlo,hhi),(alo,ahi)) in enumerate(zip(endpoints['hybrid_exact_terminal'],endpoints['adaptive'])):
            interval = _replicated_interval_record(hlo-ahi,hhi-alo)
            observed = original_paired['observations'][path_index]
            if (observed['path_index'] != path_index or type(observed['path_index']) is not int
                    or observed['stopping_index'] != record['stopping_indices'][path_index]
                    or _replicated_interval(observed['difference_certificate'],signed=True) != (hlo-ahi,hhi-alo)):
                raise ValueError('A paired difference changes its path, stopping decision or subtraction order')
            paired_rows.append(dict(path_index=path_index,stopping_index=observed['stopping_index'],signed_gap_certificate=interval))
        paired, _ = _replicated_path_summary(paired_rows,signed=True,expected_count=1024)
        paired['mean_difference_certificate'] = paired.pop('mean_signed_gap_certificate')
        if _replicated_interval(original_paired['mean_difference_certificate'],signed=True) != _replicated_interval(paired['mean_difference_certificate'],signed=True):
            raise ValueError('The supplied paired mean differs from the exact fresh-path subtraction')
        for field in ('approximate_midpoint_mean','approximate_midpoint_mean_standard_error','midpoint_mean_numerical_error_bound'):
            _replicated_equal(original_paired[field],paired[field],'A paired fresh-path diagnostic differs')
        _replicated_equal(original_paired['maximum_observation_width'],paired['maximum_signed_width'],
            'The supplied paired maximum width differs')
        case['paired_hybrid_minus_adaptive'] = paired
        result['cases'].append(case)
    result.update(_frozen_extension_aggregates(result['cases']))
    def controller_aggregate(public, bounds):
        lower, upper = _replicated_interval(public['arithmetic_mean_certificate'],signed=True)
        return dict(mean_certificate=public['arithmetic_mean_certificate'],
            approximate_midpoint_mean=public['mean'],approximate_midpoint_mean_standard_error=public['standard_error'],
            midpoint_mean_numerical_error_bound=float((upper-lower)/2),
            maximum_observation_width=float(max(hi-lo for lo,hi in bounds)))
    expected_gaps = {name:controller_aggregate(result['gap_midpoint_by_arm'][name],[
        _replicated_interval(case['arms'][name]['mean_signed_gap_certificate']) for case in result['cases']])
        for name in FROZEN_EXTENSION_ARMS}
    expected_paired = controller_aggregate(result['paired_hybrid_minus_adaptive'],[
        _replicated_interval(case['paired_hybrid_minus_adaptive']['mean_difference_certificate'],signed=True)
        for case in result['cases']])
    _frozen_extension_check_aggregates(primary_summary['exact_across_fit_gap_summaries'],expected_gaps)
    _frozen_extension_check_aggregates(primary_summary['exact_across_fit_paired_difference'],expected_paired)
    _frozen_extension_check_aggregates(primary_summary['conditional_mean_gap_standard_error_by_arm'],{
        name:result['gap_midpoint_by_arm'][name]['conditional_average_standard_error'] for name in FROZEN_EXTENSION_ARMS})
    _replicated_equal(primary_summary['conditional_mean_paired_difference_standard_error'],
        result['paired_hybrid_minus_adaptive']['conditional_average_standard_error'],
        'The supplied conditional paired average standard error differs')
    expected_ratios = dict(complete_assessment_cpu=result['paired_adaptive_over_uniform_ratios']['assessment_cpu'],
        standalone_pricing_cpu=result['paired_adaptive_over_uniform_ratios']['reconstructed_standalone_cpu'],
        conditional_integrals=result['paired_adaptive_over_uniform_ratios']['conditional_integrals'],
        incremental_evaluation_cpu=_replicated_ratio_summary([
            (case['new_outer_preparation_cost']['cpu_seconds']+case['arms']['adaptive']['incremental_assessment_cpu_seconds_including_validation'])/
            (case['new_outer_preparation_cost']['cpu_seconds']+case['arms']['cold_uniform']['incremental_assessment_cpu_seconds_including_validation'])
            for case in result['cases']]))
    _frozen_extension_check_aggregates(primary_summary['paired_adaptive_over_cold_uniform_ratios'],expected_ratios)
    if primary_summary['new_outer_preparation_cost_by_fit'] != [case['new_outer_preparation_cost'] for case in result['cases']]:
        raise ValueError('The supplied new outer preparation measurements differ')
    validate_frozen_extension(result,original_replicated_refinement)
    return result



UNIFORM_ALLOCATION_ARMS = ('cold_common', 'coarse_common', 'adaptive')
UNIFORM_ALLOCATION_TIMER_FIELDS = frozenset((
    'assessment_cpu_seconds','assessment_wall_seconds','certificate_phase_cpu_seconds',
    'coarse_cpu_seconds','cpu_seconds_including_shared_coarse','input_and_policy_validation_cpu_seconds',
    'kernel_recorded_assessment_cpu_seconds','kernel_recorded_assessment_wall_seconds',
    'refinement_cpu_seconds','reward_cpu_seconds'))


def _uniform_without_timers(value):
    """Exclude only declared execution timings from a scientific replay record."""
    if isinstance(value,dict):
        return {key:_uniform_without_timers(item) for key,item in value.items()
                if key not in UNIFORM_ALLOCATION_TIMER_FIELDS}
    if isinstance(value,list):
        return [_uniform_without_timers(item) for item in value]
    return value


def _uniform_allocation_aggregates(cases):
    import statistics
    methods = {}
    ratios = {key:{} for key in ('complete_assessment_cpu','reconstructed_standalone_cpu','conditional_integrals')}
    for name in UNIFORM_ALLOCATION_ARMS:
        rows = [case['arms'][name] for case in cases]
        bounds = [_replicated_interval(row['mean_signed_gap_certificate']) for row in rows]
        maximum = max(_replicated_rational(row['maximum_signed_width_exact']) for row in rows)
        methods[name] = dict(fitting_units=8,total_outer_paths=8192,
            total_conditional_integrals=sum(row['conditional_integrals'] for row in rows),
            total_integrand_evaluations=sum(row['integrand_evaluations'] for row in rows),
            maximum_signed_width=float(maximum),maximum_signed_width_exact=dict(
                numerator=str(maximum.numerator),denominator=str(maximum.denominator),approximation=float(maximum)),
            mean_signed_gap_certificate=_replicated_interval_record(sum(lo for lo,hi in bounds)/8,sum(hi for lo,hi in bounds)/8),
            median_complete_assessment_cpu_seconds=statistics.median(row['assessment_cpu_seconds'] for row in rows),
            median_reconstructed_standalone_cpu_seconds=statistics.median(
                case['reconstructed_standalone_cpu_seconds_by_arm'][name] for case in cases))
    for name in ('cold_common','coarse_common'):
        for key in ratios:
            if key=='complete_assessment_cpu':
                values=[case['arms']['adaptive']['assessment_cpu_seconds']/case['arms'][name]['assessment_cpu_seconds'] for case in cases]
            elif key=='reconstructed_standalone_cpu':
                values=[case['reconstructed_standalone_cpu_seconds_by_arm']['adaptive']/case['reconstructed_standalone_cpu_seconds_by_arm'][name] for case in cases]
            else:
                values=[case['arms']['adaptive']['conditional_integrals']/case['arms'][name]['conditional_integrals'] for case in cases]
            ratios[key][name]=_replicated_ratio_summary(values)
    return dict(assessment_by_arm=methods,paired_adaptive_over_control_ratios=ratios)


def validate_uniform_allocation(study,frozen_extension=None):
    """Verify complete fixed-input comparisons and distinguish newly incurred work."""
    import itertools
    if not isinstance(study,dict):
        raise ValueError('A uniform-allocation summary mapping is required')
    for field,expected in (('fitting_units',8),('completed_fits',8),('outer_paths_per_fit',1024),
                           ('new_outer_paths',0),('newly_trained_fits',0),('precision_bits',96)):
        if type(study.get(field)) is not int or study[field]!=expected:
            raise ValueError('The uniform allocation changes its fixed-input counts')
    if (study['required_maximum_path_gap_width']!='1/100' or study['new_sampling_executed'] is not False
            or study['fitting_executed'] is not False or study['all_adaptive_scientific_replays_match'] is not True
            or study['all_path_certificates_overlap'] is not True or study['all_coarse_centres_match_adaptive'] is not True
            or study['new_general_theorem_claim'] is not False or study['statistical_coverage_certificate'] is not False):
        raise ValueError('The uniform-allocation execution or interpretation scope differs')
    for field,words in (
        ('comparison_scope',('same','required width','unchanged','stronger')),
        ('sampling_scope',('retained','no new','not independent')),
        ('cost_scope',('assessment','retained','not rerun','reconstructed'))):
        if not isinstance(study.get(field),str) or any(word not in study[field].lower() for word in words):
            raise ValueError('The uniform allocation omits its research scope: '+field)
    if sorted(study['scientific_replay_excluded_timer_fields'])!=sorted(UNIFORM_ALLOCATION_TIMER_FIELDS):
        raise ValueError('The scientific replay may exclude only its declared timer fields')
    def sha(value):
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
            raise ValueError('A complete uniform-allocation SHA-256 identity is required')
    for field in ('design_sha256','retained_extension_design_sha256'):
        sha(study[field])
    if not study['source_sha256']:
        raise ValueError('The uniform allocation must retain its source identities')
    for value in study['source_sha256'].values():
        sha(value)
    if len(study['cases'])!=8:
        raise ValueError('Every retained fitting unit must be included')
    prior_cases=None
    if frozen_extension is not None:
        validate_frozen_extension(frozen_extension)
        if study['retained_extension_design_sha256']!=frozen_extension['design_sha256']:
            raise ValueError('The uniform allocation changes its retained evaluation source')
        prior_cases=frozen_extension['cases']
    orders=[]; seeds=[]; inputs=[]; snapshots=[]
    def cost(value):
        if _replicated_scalar(value)<0:
            raise ValueError('A uniform-allocation cost must be non-negative')
        return value
    def count(value):
        if type(value) is not int or value<0:
            raise ValueError('A uniform-allocation work count must be a non-negative integer')
    for index,case in enumerate(study['cases']):
        if (type(case['fitting_unit_index']) is not int or case['fitting_unit_index']!=index
                or type(case['root_seed']) is not int or case['root_seed']!=2026100410+index
                or type(case['outer_paths']) is not int or case['outer_paths']!=1024
                or case['design_sha256']!=study['design_sha256'] or case['source_sha256']!=study['source_sha256']
                or case['adaptive_scientific_replay_matches'] is not True
                or case['coarse_centres_match_adaptive'] is not True
                or case['all_path_certificates_overlap'] is not True):
            raise ValueError('A uniform-allocation fitting unit changes its context or replay')
        for field in ('input_sha256','retained_result_sha256','frozen_model_policy_values_sha256','adaptive_scientific_sha256'):
            sha(case[field])
        seeds.append(case['root_seed']);inputs.append(case['input_sha256']);snapshots.append(case['frozen_model_policy_values_sha256'])
        if len(case['arm_order'])!=3 or set(case['arm_order'])!=set(UNIFORM_ALLOCATION_ARMS):
            raise ValueError('All three uniform-allocation arms require an executed order')
        orders.append(tuple(case['arm_order']))
        setup=cost(case['retained_preparation_cpu_seconds'])
        if prior_cases is not None:
            prior=prior_cases[index]
            if (case['root_seed']!=prior['root_seed'] or case['input_sha256']!=prior['input_sha256']
                    or case['frozen_model_policy_values_sha256']!=prior['frozen_model_policy_values_sha256']):
                raise ValueError('A uniform allocation changes its retained input or frozen functions')
            _replicated_equal(setup,prior['charged_setup_cpu_seconds_by_arm']['adaptive'],
                'The retained preparation budget changes its preceding phase measurements')
        if set(case['arms'])!=set(UNIFORM_ALLOCATION_ARMS) or set(case['reconstructed_standalone_cpu_seconds_by_arm'])!=set(UNIFORM_ALLOCATION_ARMS):
            raise ValueError('A uniform-allocation arm or standalone budget is missing')
        bounds=[]
        for name,row in case['arms'].items():
            lo,hi=_replicated_interval(row['mean_signed_gap_certificate'])
            bounds.append((lo,hi))
            maximum=_replicated_rational(row['maximum_signed_width_exact'])
            if (not 0<=maximum<=Fraction(1,100) or row['all_targets_met'] is not True
                    or row['all_signed_inside_absolute'] is not True):
                raise ValueError('A uniform allocation fails its actual required path-gap width')
            for field,expected in (('maximum_signed_width',float(maximum)),('mean_signed_width',float(hi-lo)),
                    ('approximate_midpoint_mean',float((lo+hi)/2)),('midpoint_mean_numerical_error_bound',float((hi-lo)/2))):
                _replicated_equal(row[field],expected,'An exact uniform-allocation summary differs')
            if hi-lo>maximum:
                raise ValueError('The mean width exceeds the recorded exact maximum path width')
            cost(row['approximate_midpoint_mean_standard_error'])
            for field in ('conditional_integrals','integrand_evaluations'):
                count(row[field])
            for field in ('assessment_cpu_seconds','input_and_policy_validation_cpu_seconds','certificate_phase_cpu_seconds'):
                cost(row[field])
            _replicated_equal(row['assessment_cpu_seconds'],row['input_and_policy_validation_cpu_seconds']+row['certificate_phase_cpu_seconds'],
                'The newly incurred assessment budget omits or duplicates validation')
            _replicated_equal(case['reconstructed_standalone_cpu_seconds_by_arm'][name],setup+row['assessment_cpu_seconds'],
                'The reconstructed standalone budget omits or duplicates retained preparation')
            if name in ('cold_common','coarse_common') and (row['conditional_integrals']<=0 or row['assessment_cpu_seconds']<=0 or
                    case['reconstructed_standalone_cpu_seconds_by_arm'][name]<=0):
                raise ValueError('A paired allocation ratio requires a positive control denominator')
        if max(lo for lo,hi in bounds)>min(hi for lo,hi in bounds):
            raise ValueError('The three unchanged-gain mean certificates do not intersect')
    if len(set(seeds))!=8 or len(set(inputs))!=8 or len(set(snapshots))!=8:
        raise ValueError('The uniform allocation requires eight distinct retained inputs and fits')
    if (set(orders[:6])!=set(itertools.permutations(UNIFORM_ALLOCATION_ARMS))
            or any(sorted(sum(order[position]==name for order in orders) for position in range(3))!=[2,3,3]
                   for name in UNIFORM_ALLOCATION_ARMS)
            or sum(order.index('cold_common')<order.index('adaptive') for order in orders)!=4):
        raise ValueError('The uniform-allocation execution positions or cold/adaptive order are unbalanced')
    for field,value in _uniform_allocation_aggregates(study['cases']).items():
        _frozen_extension_check_aggregates(study[field],value)
    return True


def render_uniform_allocation(study):
    validate_uniform_allocation(study)
    rows=[]
    for name,label in (('cold_common','Cold common budget'),('coarse_common','Coarse-centre common budget'),
                       ('adaptive','Gain-directed refinement')):
        row=study['assessment_by_arm'][name]
        rows.append([label,f"{row['total_conditional_integrals']:,}",f"{row['maximum_signed_width']:.5g}",
                     f"{row['median_complete_assessment_cpu_seconds']:.3f}",
                     f"{row['median_reconstructed_standalone_cpu_seconds']:.3f}"])
    return table('Uniform accuracy budgets and gain-directed refinement','revision-uniform-allocation',
        ['Assessment','Integral calls','Maximum width','New CPU (s)','Standalone CPU (s)'],rows,
        'All three arms assess the same 1,024 retained paths per fit across eight fixed fits and attain required maximum path-gap width 0.01. '
        'Integral calls total all eight fits; maximum width is the largest actual width across 8,192 paths. '
        'CPU columns give medians over eight fits. New CPU includes the complete assessment, policy validation and result construction. '
        'Standalone CPU adds retained preparation once, without rerunning simulation or fitting. '
        'Common controls retain fuller query traces; measured CPU compares verified implementations and does not isolate allocation-rule runtime.')


def _uniform_realised_values(record,path_index):
    """Evaluate the frozen hinge basis exactly, without a conditional oracle."""
    knots=[Fraction(_replicated_scalar(value)) for value in record['knots']]
    values=[]
    for state,coefficients in zip(record['states'][path_index],record['coefficients']):
        coordinates=[Fraction(_replicated_scalar(value)) for value in state]
        if not coordinates or any(value<=0 for value in coordinates):
            raise ValueError('A retained state must have finite positive coordinates')
        dimension=len(coordinates)
        features=[Fraction(1),sum(coordinates)/(dimension*100)]
        features.extend(max(max(coordinates)-knot,Fraction(0))/100 for knot in knots)
        features.extend(sum(max(value-knot,Fraction(0)) for value in coordinates)/(dimension*100) for knot in knots)
        if len(coefficients)!=len(features):
            raise ValueError('A retained value function omits a hinge coefficient')
        values.append(sum(Fraction(_replicated_scalar(coefficient))*feature for coefficient,feature in zip(coefficients,features)))
    return values


def _uniform_direct_gain(rewards,values,brackets,stopping):
    """Enclose gain candidates through shared-date cancellation in span sums."""
    if not len(rewards)==len(values)==len(brackets) or not 0<=stopping<len(rewards):
        raise ValueError('Direct span algebra requires matching complete calendars')
    candidates=[(Fraction(0),Fraction(0))]
    for date in range(len(rewards)):
        if date==stopping:
            continue
        lo=rewards[date][0]-rewards[stopping][1]
        hi=rewards[date][1]-rewards[stopping][0]
        span=range(min(date,stopping)+1,max(date,stopping)+1)
        realised=sum(values[index] for index in span)
        conditional_lower=sum(brackets[index][0] for index in span)
        conditional_upper=sum(brackets[index][1] for index in span)
        if date>stopping:
            candidates.append((lo-realised+conditional_lower,hi-realised+conditional_upper))
        else:
            candidates.append((lo+realised-conditional_upper,hi+realised-conditional_lower))
    return max(lo for lo,hi in candidates),max(hi for lo,hi in candidates)


def _uniform_check_trace(observation,record,arm,values):
    """Reconstruct global rounds through an independent direct-span calculation."""
    import hashlib
    initial_width=Fraction(1,100) if arm=='cold_common' else Fraction(1)
    if _replicated_interval(observation['initial_common_width_certificate'])!=(initial_width,initial_width):
        raise ValueError('A uniform arm changes its prescribed initial width')
    rights=len(record['model']['exercise_times'])
    if len(values)!=rights or len(observation['reward_certificates'])!=rights:
        raise ValueError('A uniform trace omits reward or value dates')
    rewards=[_replicated_interval(value) for value in observation['reward_certificates']]
    stopping=observation['stopping_index']
    initial=observation['initial_queries']
    if [query['increment'] for query in initial]!=list(range(1,rights)):
        raise ValueError('Uniform initialisation omits, repeats or reorders a conditional date')
    brackets=[(Fraction(0),Fraction(0))]
    centres=[Fraction(0)];penalty=[Fraction(0)]
    for query in initial:
        if _replicated_interval(query['requested_width_certificate'])!=(initial_width,initial_width):
            raise ValueError('Uniform initialisation uses unequal width requests')
        bounds=_replicated_interval(query['returned_certificate'],signed=True)
        if bounds[1]-bounds[0]>initial_width:
            raise ValueError('An initial conditional bracket exceeds its requested width')
        brackets.append(bounds)
        centres.append(sum(bounds)/2)
        penalty.append(penalty[-1]+values[query['increment']]-centres[-1])
    representation=[[str(value.numerator),str(value.denominator)] for value in (*centres,*penalty)]
    fingerprint=hashlib.sha256(json.dumps(representation,separators=(',',':')).encode()).hexdigest()
    if observation['frozen_centres_and_penalty_sha256']!=fingerprint:
        raise ValueError('A uniform trace changes its initial centres or realised penalty')
    rho=max(rewards[index][1]-rewards[index][0]+rewards[stopping][1]-rewards[stopping][0]
            for index in range(rights) if index!=stopping)
    if (_replicated_interval(observation['reward_difference_width_allowance_certificate'])!=(rho,rho)
            or type(observation['maximum_stopping_span_length']) is not int
            or observation['maximum_stopping_span_length']!=max(stopping,rights-1-stopping)):
        raise ValueError('The sufficient uniform-width diagnostic changes its reward error or span')
    theta=initial_width
    final=_uniform_direct_gain(rewards,values,brackets,stopping)
    rounds=observation['global_rounds']
    if len(rounds)>16:
        raise ValueError('A common-width trace exceeds its registered sixteen rounds')
    for round_index,step in enumerate(rounds):
        if final[1]-final[0]<=Fraction(1,100):
            raise ValueError('Uniform refinement continues after its actual gap target is met')
        theta/=4
        eligible=[index for index in range(1,rights) if brackets[index][1]-brackets[index][0]>theta]
        if (type(step['round_index']) is not int or step['round_index']!=round_index+1
                or step['eligible_increments']!=eligible or step['completed'] is not True
                or step['all_retained_conditional_widths_comply'] is not True
                or _replicated_interval(step['common_requested_width_certificate'])!=(theta,theta)
                or [query['increment'] for query in step['queries']]!=eligible):
            raise ValueError('A common round omits eligible dates or changes its common schedule')
        for query in step['queries']:
            date=query['increment']
            if (_replicated_interval(query['requested_width_certificate'])!=(theta,theta)
                    or _replicated_interval(query['previous_certificate'],signed=True)!=brackets[date]):
                raise ValueError('A uniform query changes its requested width or preceding bracket')
            returned=_replicated_interval(query['returned_certificate'],signed=True)
            if returned[1]-returned[0]>theta:
                raise ValueError('A returned uniform bracket exceeds its actual common target')
            intersection=max(brackets[date][0],returned[0]),min(brackets[date][1],returned[1])
            if intersection[1]<intersection[0] or _replicated_interval(query['retained_certificate'],signed=True)!=intersection:
                raise ValueError('A retained uniform bracket is not its exact nonempty intersection')
            brackets[date]=intersection
        if any(hi-lo>theta for lo,hi in brackets):
            raise ValueError('A completed common round leaves a non-compliant bracket')
        final=_uniform_direct_gain(rewards,values,brackets,stopping)
        if _replicated_interval(step['gap_certificate_after_round'])!=final:
            raise ValueError('A uniform round gap differs from independent direct-span algebra')
    if final[1]-final[0]>Fraction(1,100) or _replicated_interval(observation['signed_gap_certificate'])!=final:
        raise ValueError('The final uniform gain differs from reconstructed algebra or exceeds its width target')
    plain=[(Fraction(0),Fraction(0))]+[(
        rewards[date][0]-rewards[stopping][1]-penalty[date]+penalty[stopping],
        rewards[date][1]-rewards[stopping][0]-penalty[date]+penalty[stopping])
        for date in range(rights) if date!=stopping]
    allowance=sum(max(abs(lo-centres[date]),abs(hi-centres[date]))
                  for date,(lo,hi) in enumerate(brackets) if date>0)
    absolute=(max(Fraction(0),max(lo for lo,hi in plain)-allowance),max(hi for lo,hi in plain)+allowance)
    if _replicated_interval(observation['absolute_gap_certificate'])!=absolute:
        raise ValueError('The final absolute allocation bound differs from its independently reconstructed allowance')
    def work(queries):
        for query in queries:
            for field in ('integrals','integrand_evaluations'):
                if type(query['work'][field]) is not int or query['work'][field]<0:
                    raise ValueError('Uniform query work must be a non-negative integer')
            if set(query['complete_query_cost'])!={'cpu_seconds','wall_seconds'}:
                raise ValueError('A complete uniform query cost requires both clocks')
            for value in query['complete_query_cost'].values():
                if _replicated_scalar(value)<0:
                    raise ValueError('A uniform query cost is negative')
        return dict(conditional_integrals=sum(query['work']['integrals'] for query in queries),
                    integrand_evaluations=sum(query['work']['integrand_evaluations'] for query in queries))
    refinement=[query for step in rounds for query in step['queries']]
    if (observation['coarse_work']!=work(initial) or observation['refinement_work']!=work(refinement)
            or observation['total_work']!=work(initial+refinement)):
        raise ValueError('Uniform work excludes initialisation, returned retries or complete global rounds')
    for value in observation['complete_path_cost'].values():
        if _replicated_scalar(value)<0:
            raise ValueError('A complete uniform path cost is invalid')
    return True


def build_uniform_allocation(primary_summary,*,fit_artifacts,design,frozen_extension):
    """Review three complete fixed-input arms before dropping private path ledgers.

    File hashes and completed execution markers are checked by the caller.
    This check independently rebuilds exact means, direct-span round algebra,
    work totals, centre fingerprints and timing-free adaptive replay identities.
    Transcendental reward endpoints and conditional integral enclosures retain
    their numerical source checks; this aggregate checker does not rerun them.
    """
    import copy
    import hashlib
    import statistics
    validate_frozen_extension(frozen_extension)
    if (set(fit_artifacts)!=set(range(8)) or primary_summary['completed_fits']!=8
            or primary_summary['retained_fits']!=8 or primary_summary['outer_paths_per_fit']!=1024
            or primary_summary['total_retained_paths']!=8192 or len(primary_summary['cases'])!=8
            or primary_summary['adaptive_exact_numerical_replay'] is not True
            or primary_summary['all_path_certificates_intersect'] is not True):
        raise ValueError('The uniform comparison requires every retained fit and complete path ledger')
    if (design['source_sha256']!=primary_summary['source_sha256'] or design['fitting_units']!=8
            or design['outer_paths_per_fit']!=1024 or design['precision_bits']!=96 or design['target']!='1/100'
            or design['initial_common_widths']!={'cold_common':'1/100','coarse_common':'1'}
            or design['common_tolerance_factor']!='1/4' or design['maximum_global_rounds']!=16
            or design['new_simulation_or_fitting'] is not False or design['recorded_before_new_assessment'] is not True
            or design['priority_claim'] is not False or design['public_preregistration_claim'] is not False
            or design['exact_replay_scope']['adaptive_numerical_replay'] is not True
            or sorted(design['exact_replay_scope']['excluded_timing_fields'])!=sorted(UNIFORM_ALLOCATION_TIMER_FIELDS)
            or len(design['fits'])!=8):
        raise ValueError('The uniform allocation changes its recorded mathematical or execution design')
    evidence=design['source_evidence']
    if (evidence['frozen_extension_design_sha256']!=frozen_extension['design_sha256']
            or evidence['frozen_extension_source_sha256']!=frozen_extension['source_sha256']):
        raise ValueError('The uniform allocation changes the source of its retained states and results')
    def canonical(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    result=dict(schema_version=1,fitting_units=8,completed_fits=8,outer_paths_per_fit=1024,new_outer_paths=0,
        newly_trained_fits=0,precision_bits=96,required_maximum_path_gap_width='1/100',
        design_sha256=primary_summary['design_sha256'],source_sha256=copy.deepcopy(primary_summary['source_sha256']),
        retained_extension_design_sha256=frozen_extension['design_sha256'],new_sampling_executed=False,fitting_executed=False,
        all_adaptive_scientific_replays_match=True,all_path_certificates_overlap=True,all_coarse_centres_match_adaptive=True,
        scientific_replay_excluded_timer_fields=sorted(UNIFORM_ALLOCATION_TIMER_FIELDS),
        new_general_theorem_claim=False,statistical_coverage_certificate=False,
        comparison_scope='The stronger common-width controls and adaptive refinement enclose the same unchanged ideal hinge gain at a common required width. Actual achieved widths and measured costs differ.',
        sampling_scope='Eight retained fits and their existing 1,024-path samples are reused. No new draws or fits are generated; this is not independent confirmation from a new sampling experiment.',
        cost_scope='New work is complete assessment including input and policy validation and result construction. Standalone budgets are reconstructed by adding retained preparation once; simulation and fitting are not rerun. Common controls retain fuller query traces; measured CPU does not isolate allocation-rule runtime. Process start-up, imports and artifact I/O are excluded.',
        cases=[])
    for index in range(8):
        artifacts=fit_artifacts[index]
        ledger,record,retained=(artifacts[key] for key in ('result','input','retained_result'))
        prior=frozen_extension['cases'][index]
        fit=design['fits'][index]
        prefix=f'fit_{index:02d}/'
        if (ledger['fit_index']!=index or ledger['root_seed']!=prior['root_seed']
                or ledger['retained_input_sha256']!=prior['input_sha256']
                or ledger['retained_input_sha256']!=evidence['frozen_extension_files'][prefix+'inputs.json']
                or ledger['retained_result_sha256']!=evidence['frozen_extension_files'][prefix+'result.json']
                or retained['input_sha256']!=prior['input_sha256']
                or retained['frozen_model_policy_values_sha256']!=prior['frozen_model_policy_values_sha256']
                or ledger['arm_order']!=fit['arm_order'] or list(ledger['arms'])!=ledger['arm_order']
                or fit['index']!=index or fit['root_seed']!=prior['root_seed']
                or ledger['source_sha256']!=primary_summary['source_sha256']
                or ledger['design_sha256']!=primary_summary['design_sha256']):
            raise ValueError('A uniform-allocation unit changes its retained identity, source or executed order')
        if (record['design']['root_seed']!=prior['root_seed'] or record['design']['spawn_keys']!=[[0],[1],[3]]
                or record['design']['outer_paths']!=1024 or len(record['states'])!=1024
                or len(record['stopping_indices'])!=1024 or ledger['outer_paths']!=1024
                or ledger['retained_preparation_cost']!=record['preparation_cost']):
            raise ValueError('A uniform allocation changes its existing evaluation inputs or preparation')
        frozen={key:record[key] for key in ('model','knots','coefficients','frozen_policy')}
        if canonical(frozen)!=prior['frozen_model_policy_values_sha256']:
            raise ValueError('The uniform comparison alters its frozen model, policy or hinge coefficients')
        compact={key:copy.deepcopy(value) for key,value in ledger.items() if key!='arms'}
        compact['arms']={name:{key:copy.deepcopy(value) for key,value in arm.items() if key!='observations'}
                         for name,arm in ledger['arms'].items()}
        if compact!=primary_summary['cases'][index]:
            raise ValueError('The primary uniform summary differs from its full retained path ledger')
        original_adaptive=_uniform_without_timers(retained['arms']['adaptive'])
        replayed_adaptive=_uniform_without_timers(ledger['arms']['adaptive'])
        if original_adaptive!=replayed_adaptive:
            raise ValueError('Adaptive replay changes a certificate, work count, history or frozen fingerprint')
        if (ledger['adaptive_exact_numerical_replay'] is not True or ledger['all_path_certificates_intersect'] is not True
                or ledger['coarse_centres_match_adaptive'] is not True):
            raise ValueError('The uniform comparison omits a required numerical replay or centre check')
        setup=record['preparation_cost']['cpu_seconds']
        _replicated_equal(setup,prior['charged_setup_cpu_seconds_by_arm']['adaptive'],
            'The retained complete preparation differs from its preceding phase budget')
        case=dict(fitting_unit_index=index,root_seed=prior['root_seed'],outer_paths=1024,
            arm_order=copy.deepcopy(ledger['arm_order']),input_sha256=ledger['retained_input_sha256'],
            retained_result_sha256=ledger['retained_result_sha256'],design_sha256=ledger['design_sha256'],
            source_sha256=copy.deepcopy(ledger['source_sha256']),frozen_model_policy_values_sha256=prior['frozen_model_policy_values_sha256'],
            adaptive_scientific_sha256=canonical(replayed_adaptive),adaptive_scientific_replay_matches=True,
            coarse_centres_match_adaptive=True,all_path_certificates_overlap=True,retained_preparation_cpu_seconds=setup,
            reconstructed_standalone_cpu_seconds_by_arm=copy.deepcopy(ledger['standalone_pricing_cpu_seconds_by_arm']),arms={})
        by_arm={};value_cache={}
        for name in UNIFORM_ALLOCATION_ARMS:
            arm=ledger['arms'][name]
            observations=arm['observations']
            expected,bounds=_replicated_path_summary(observations,expected_count=1024)
            _replicated_check_summary(arm,expected)
            by_arm[name]=bounds
            for path_index,observation in enumerate(observations):
                if observation['stopping_index']!=record['stopping_indices'][path_index]:
                    raise ValueError('A uniform allocation changes its retained stopping decision')
                lo,hi=bounds[path_index]
                alo,ahi=_replicated_interval(observation['absolute_gap_certificate'])
                if not alo<=lo<=hi<=ahi:
                    raise ValueError('A signed allocation certificate leaves its absolute enclosure')
                if name in ('cold_common','coarse_common'):
                    if path_index not in value_cache:
                        value_cache[path_index]=_uniform_realised_values(record,path_index)
                    _uniform_check_trace(observation,record,name,value_cache[path_index])
            for field in ('conditional_integrals','integrand_evaluations'):
                work=[row['total_work'][field] for row in observations]
                if any(type(value) is not int or value<0 for value in work) or arm[field]!=sum(work):
                    raise ValueError('An allocation work total omits initial or refined path queries')
            if name in ('cold_common','coarse_common'):
                for field,value in (
                    ('initialisation_integrals',sum(row['coarse_work']['conditional_integrals'] for row in observations)),
                    ('global_refinement_integrals',sum(row['refinement_work']['conditional_integrals'] for row in observations)),
                    ('global_refinement_rounds',sum(len(row['global_rounds']) for row in observations))):
                    if arm[field]!=value or type(arm[field]) is not int:
                        raise ValueError('A uniform phase work count differs from its complete trace')
            maximum=max(hi-lo for lo,hi in bounds)
            case['arms'][name]=dict(expected,maximum_signed_width_exact=dict(
                numerator=str(maximum.numerator),denominator=str(maximum.denominator),approximation=float(maximum)),
                all_targets_met=arm['all_targets_met'],all_signed_inside_absolute=arm['all_signed_inside_absolute'],
                conditional_integrals=arm['conditional_integrals'],integrand_evaluations=arm['integrand_evaluations'],
                assessment_cpu_seconds=arm['assessment_cpu_seconds'],
                input_and_policy_validation_cpu_seconds=arm['input_and_policy_validation_cpu_seconds'],
                certificate_phase_cpu_seconds=arm['certificate_phase_cpu_seconds'])
            _replicated_equal(ledger['assessment_only_cpu_seconds_by_arm'][name],arm['assessment_cpu_seconds'],
                'The newly incurred assessment excludes or adds previous work')
        old_uniform=retained['arms']['cold_uniform']['observations']
        if len(old_uniform)!=1024:
            raise ValueError('Every preceding uniform path is required for the shared-gain intersection check')
        for path_index in range(1024):
            intervals=[by_arm[name][path_index] for name in UNIFORM_ALLOCATION_ARMS]
            if (old_uniform[path_index]['path_index']!=path_index
                    or old_uniform[path_index]['stopping_index']!=record['stopping_indices'][path_index]):
                raise ValueError('A preceding uniform observation changes its paired path')
            intervals.append(_replicated_interval(old_uniform[path_index]['signed_gap_certificate']))
            if max(lo for lo,hi in intervals)>min(hi for lo,hi in intervals):
                raise ValueError('Exact enclosures of the unchanged path gain do not intersect')
            if (ledger['arms']['coarse_common']['observations'][path_index]['frozen_centres_and_penalty_sha256']!=
                    ledger['arms']['adaptive']['observations'][path_index]['frozen_centres_and_penalty_sha256']):
                raise ValueError('Coarse-common initial centres differ from the adaptive construction')
        result['cases'].append(case)
    result.update(_uniform_allocation_aggregates(result['cases']))
    expected_summary={}
    for name,arm in result['assessment_by_arm'].items():
        costs=[case['arms'][name]['assessment_cpu_seconds'] for case in result['cases']]
        lo,hi=_replicated_interval(arm['mean_signed_gap_certificate'])
        expected_summary[name]=dict(mean_gap_certificate=arm['mean_signed_gap_certificate'],
            arithmetic_mean_gap=float((lo+hi)/2),conditional_integrals=arm['total_conditional_integrals'],
            integrand_evaluations=arm['total_integrand_evaluations'],maximum_signed_width=arm['maximum_signed_width'],
            assessment_cpu_seconds=dict(minimum=min(costs),arithmetic_mean=statistics.mean(costs),
                median=statistics.median(costs),maximum=max(costs)))
    _frozen_extension_check_aggregates(primary_summary['arm_summaries'],expected_summary)
    expected_ratios={name:dict(assessment_only_cpu=result['paired_adaptive_over_control_ratios']['complete_assessment_cpu'][name],
        standalone_pricing_cpu=result['paired_adaptive_over_control_ratios']['reconstructed_standalone_cpu'][name],
        conditional_integrals=result['paired_adaptive_over_control_ratios']['conditional_integrals'][name])
        for name in ('cold_common','coarse_common')}
    _frozen_extension_check_aggregates(primary_summary['adaptive_over_common_ratios'],expected_ratios)
    validate_uniform_allocation(result,frozen_extension)
    return result


def validate_restricted_initial_right(study, uniform_allocation):
    """Check the separately reconstructed restricted-calendar aggregates."""
    import re
    if uniform_allocation is None:
        raise ValueError('Restricted reconstruction requires its complete original allocation comparison')
    expected=dict(schema_version=1,completed_fits=8,retained_paths=8192,
        retained_paths_per_fit=1024,reconstruction_arm='coarse_common',
        retained_candidate_indices=list(range(1,10)))
    if any(study.get(key)!=value for key,value in expected.items()):
        raise ValueError('The restricted reconstruction changes its fixed calendar or sample')
    for key in ('all_coarse_traces_validated','all_full_gain_reconstructions_match','all_endpoint_dominance_checks_pass'):
        if study.get(key) is not True:
            raise ValueError('The restricted reconstruction lacks complete trace or algebra checks')
    for key in ('conditional_integrals_rerun','new_paths_generated','new_fitting_executed',
                'sampling_coverage_certified','policy_improvement_claim','cpu_saving_claim'):
        if study.get(key) is not False:
            raise ValueError('The restricted reconstruction adds an unsupported scope claim')
    if (study['design_sha256']!=uniform_allocation['design_sha256']
            or study['source_sha256']!=uniform_allocation['source_sha256']):
        raise ValueError('The restricted reconstruction changes its registered source context')
    current=Path(__file__).resolve().parents[2]/'scripts/certify_restricted_gain.py'
    import hashlib
    if study['audit_source_sha256']!=hashlib.sha256(current.read_bytes()).hexdigest():
        raise ValueError('The restricted audit source differs from the reviewed implementation')
    cases=study['cases']
    if len(cases)!=8 or [row['fit_index'] for row in cases]!=list(range(8)):
        raise ValueError('The restricted reconstruction omits or reorders a fitting unit')
    count_keys=('unchanged_full_interval_count','unchanged_lower_endpoint_count','unchanged_upper_endpoint_count',
        'strictly_ruled_out_initial_candidate_count','retained_adaptive_full_interval_intersections',
        'restricted_old_adaptive_interval_intersections','restricted_paths_above_original_width_target')
    mean_keys=('mean_full_gain_certificate','mean_restricted_gain_certificate','mean_gain_reduction_certificate')
    width_keys=('maximum_full_width','maximum_restricted_width','maximum_reduction_width')
    for row,prior in zip(cases,uniform_allocation['cases']):
        if (type(row['fit_index']) is not int or type(row['root_seed']) is not int
                or row['root_seed']!=prior['root_seed'] or type(row['retained_paths']) is not int
                or row['retained_paths']!=prior['outer_paths'] or row['retained_paths']!=1024
                or row['retained_input_sha256']!=prior['input_sha256']
                or row['retained_result_sha256']!=prior['retained_result_sha256']):
            raise ValueError('The restricted reconstruction changes its retained path or policy context')
        for key in ('ordered_proof_sha256','allocation_result_sha256'):
            if not isinstance(row[key],str) or re.fullmatch(r'[0-9a-f]{64}',row[key]) is None:
                raise ValueError('The restricted reconstruction omits a valid evidence identity')
        n=row['retained_paths']
        for key in count_keys:
            if type(row[key]) is not int or not 0<=row[key]<=n:
                raise ValueError('A restricted reconstruction count is invalid')
        if (any(row[key]!=n for key in count_keys[:-1])
                or row['restricted_paths_above_original_width_target']!=0):
            raise ValueError('The published unchanged-interval finding requires every retained path')
        full=_replicated_interval(row['mean_full_gain_certificate'])
        restricted=_replicated_interval(row['mean_restricted_gain_certificate'])
        reduction=_replicated_interval(row['mean_gain_reduction_certificate'])
        if full!=_replicated_interval(prior['arms']['coarse_common']['mean_signed_gap_certificate']):
            raise ValueError('The reconstructed full mean differs from its original coarse-common mean')
        if restricted!=full or reduction!=(Fraction(0),Fraction(0)):
            raise ValueError('A strictly dominated initial candidate must leave the reconstructed sample intervals unchanged')
        full_width=_replicated_rational(row['maximum_full_width'])
        restricted_width=_replicated_rational(row['maximum_restricted_width'])
        if (full_width!=_replicated_rational(prior['arms']['coarse_common']['maximum_signed_width_exact'])
                or restricted_width!=full_width or not 0<=restricted_width<=Fraction(1,100)
                or _replicated_rational(row['maximum_reduction_width'])!=0):
            raise ValueError('The restricted reconstruction changes its exact achieved width')
    aggregate=study['aggregate']
    for key in count_keys:
        if type(aggregate[key]) is not int or aggregate[key]!=sum(row[key] for row in cases):
            raise ValueError('The restricted aggregate count differs from its complete fitting units')
    for key in mean_keys:
        pairs=[_replicated_interval(row[key]) for row in cases]
        expected=(sum(lo for lo,hi in pairs)/8,sum(hi for lo,hi in pairs)/8)
        if _replicated_interval(aggregate[key])!=expected:
            raise ValueError('The restricted aggregate mean is not the exact equally weighted mean')
    for key in width_keys:
        if _replicated_rational(aggregate[key])!=max(_replicated_rational(row[key]) for row in cases):
            raise ValueError('The restricted aggregate width is not its exact maximum')
    return True


def _monetary_log_bounds():
    """Enclose log(1280) by exact atanh series and a geometric remainder."""
    def part(z, terms):
        value=2*sum((z**(2*k+1)/Fraction(2*k+1) for k in range(terms)),Fraction(0))
        remainder=2*z**(2*terms+1)/(Fraction(2*terms+1)*(1-z*z))
        return value,value+remainder
    # 1280 = 2**10 * 5/4. The positive-term logarithm series follows
    # by integrating the geometric series for 2/(1-z**2).
    a,b=part(Fraction(1,3),48);c,d=part(Fraction(1,9),24)
    return 10*a+c,10*b+d


def validate_multivariate_monetary(study):
    """Check aggregate transcription and eligibility, without private paths."""
    import hashlib
    import math
    def require(value, message):
        if not value: raise ValueError(message)
    def cost(value):
        require(set(value)=={'cpu_seconds','wall_seconds'} and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in value.values()), 'Invalid monetary cost')
        return value
    permitted={'schema_version','author','date','design_sha256','amendment_sha256','family','costed_looks','tolerances','costed_paths_across_fits','eligible_paths_across_fits','observations_assessed','excluded_prefix','caps','tail_allowances','source_sha256','verification_sha256','verification_counts','correction_workflow_cost','actual_workflow_costs','cases','sampling_law_certified','unrounded_ideal_path_enclosure_established','market_validation_established','policy_improved','substantive_original_research_contribution_established','original_numerical_records_preserved','all_generated_and_assessed_paths_charged','sampling_scope','cost_scope','source_scope'}
    require(set(study)==permitted, 'Unexpected public monetary field')
    def certificate_fields(value):
        if isinstance(value,dict):
            if 'numerator' in value or 'denominator' in value:
                require(set(value)=={'numerator','denominator','approximation'}, 'Unexpected public rational field')
            if 'lower' in value or 'upper' in value or 'width' in value:
                require(set(value)=={'lower','upper','width'}, 'Unexpected public interval field')
            for item in value.values(): certificate_fields(item)
        elif isinstance(value,list):
            for item in value: certificate_fields(item)
    certificate_fields(study)
    require(set(study['tail_allowances'])=={'zero','european_common'}, 'Unexpected public tail field')
    log_expected=_monetary_log_bounds()
    require(study['family']==dict(events=32,delta='1/640',alpha='1/20'), 'Changed monetary event allocation')
    require(study['costed_looks']==[4096,16384] and study['tolerances']==['1/2','1'], 'Changed monetary design')
    require(study['costed_paths_across_fits']==131072 and study['eligible_paths_across_fits']==131056 and study['observations_assessed']==262144, 'Changed monetary observation counts')
    require(study['excluded_prefix']==dict(fit_index=0,rows=16) and study['caps']==dict(zero=150,european_common=50), 'Changed exposure or caps')
    for key in ('sampling_law_certified','unrounded_ideal_path_enclosure_established','market_validation_established','policy_improved','substantive_original_research_contribution_established'):
        require(study[key] is False, 'Unsupported monetary scope: '+key)
    for key in ('original_numerical_records_preserved','all_generated_and_assessed_paths_charged'):
        require(study[key] is True, 'Missing original record or cost preservation')
    repo=PAPER.parent
    require(len(study['source_sha256'])==27, 'Incomplete monetary source inventory')
    require(set(study['verification_sha256'])=={'design','original_assessment','eligibility_amendment','corrected_inference','numerical_audit','path_replay','tail_rebuild','corrected_audit'}, 'Incomplete monetary verification identities')
    for name,identity in study['source_sha256'].items():
        require(hashlib.sha256((repo/name).read_bytes()).hexdigest()==identity, 'Changed registered monetary source: '+name)
    for key,identity in study['verification_sha256'].items():
        require(isinstance(identity,str) and re.fullmatch(r'[0-9a-f]{64}',identity) is not None, 'Invalid monetary evidence identity: '+key)
    checks=study['verification_counts']
    require(checks==dict(path_reconstructions=131072,retained_stop_comparisons=8192,numerical_observations=262144,full_european_traces=131072,fixed_upper_looks=32,corrected_looks=4,unchanged_looks=28,tail_reconstructions=2), 'Incomplete monetary verification counts')
    correction=cost(study['correction_workflow_cost'])
    for value in study['actual_workflow_costs'].values(): cost(value)
    require(len(study['cases'])==8 and [row['fit_index'] for row in study['cases']]==list(range(8)), 'Missing or reordered monetary fit')
    for row in study['cases']:
        require(set(row)=={'fit_index','assessors'}, 'Unexpected public monetary case field')
        require(set(row['assessors'])=={'zero','european_common'}, 'A monetary fit needs both assessors')
        excluded=16 if row['fit_index']==0 else 0
        for arm,result in row['assessors'].items():
            require(set(result)=={'looks','decisions'}, 'Unexpected public monetary assessor field')
            looks=result['looks']
            require([x['costed_paths'] for x in looks]==[4096,16384], 'Changed costed look schedule')
            for look in looks:
                require(set(look)=={'costed_paths','eligible_paths','excluded_paths','upper_limit','acceptance','clipped_gain_mean_certificate','raw_gain_mean_certificate','original_prefix_cost'}, 'Unexpected public monetary look field')
                n=look['costed_paths']-excluded
                require(type(look['eligible_paths']) is int and look['eligible_paths']==n and look['excluded_paths']==excluded, 'Changed eligible look count')
                limit=look['upper_limit']; stats=limit['statistics']; cap=Fraction(study['caps'][arm])
                require(set(limit)=={'author','confidence_inequality_source','construction','deterministic_zero_cap','empirical_bernstein_applied','failure_probability','financial_decision_certified','log_two_over_delta_certificate','mean_upper','observation_input_premise','population_tail_upper','precision_bits','python_flint_version','sample_midpoint_variance','sampling_input_premise','sampling_law_certified','sampling_radius_certificate','schema_version','squared_radius_budget','statistics','tail_input_premise','upper_limit','upper_limit_certificate','variance_bound_certificate','variance_upper'}, 'Unexpected public upper-limit field')
                require(set(stats)=={'schema_version','author','count','cap','sum_lower','sum_upper','sum_midpoint','sum_midpoint_squared','sum_radius_squared','maximum_raw_width','maximum_clipped_width'}, 'Unexpected public sufficient-statistic field')
                require(stats['count']==n and _replicated_rational(stats['cap'])==cap and _replicated_rational(limit['failure_probability'])==Fraction(1,640), 'Changed confidence count, cap or probability')
                sums={key:_replicated_rational(stats[key]) for key in ('sum_lower','sum_upper','sum_midpoint','sum_midpoint_squared','sum_radius_squared')}
                mean=_replicated_rational(limit['mean_upper'])
                require(mean==sums['sum_upper']/n and 0<=sums['sum_lower']<=sums['sum_upper']<=n*cap, 'Changed clipped mean')
                require(_replicated_interval(look['clipped_gain_mean_certificate'])==(sums['sum_lower']/n,mean), 'Changed clipped mean certificate')
                raw=_replicated_interval(look['raw_gain_mean_certificate'])
                require(raw[0]>=sums['sum_lower']/n and raw[1]>=mean, 'A raw mean cannot be below its clipped endpoint')
                require(2*sums['sum_midpoint']==sums['sum_lower']+sums['sum_upper'], 'Changed midpoint sum')
                raw_width=_replicated_rational(stats['maximum_raw_width']); clipped_width=_replicated_rational(stats['maximum_clipped_width'])
                require(0<=clipped_width<=raw_width<=Fraction(1,100) and 0<=sums['sum_radius_squared']<=n*(clipped_width/2)**2, 'Inconsistent numerical widths or radius budget')
                require(4*n*sums['sum_radius_squared']>=(sums['sum_upper']-sums['sum_lower'])**2, 'Radius budget loses the observed interval widths')
                variance=(sums['sum_midpoint_squared']-sums['sum_midpoint']**2/n)/(n-1)
                budget=sums['sum_radius_squared']/(n-1)
                require(variance>=0 and budget>=0 and _replicated_rational(limit['sample_midpoint_variance'])==variance and _replicated_rational(limit['squared_radius_budget'])==budget, 'Changed exact variance ingredients')
                inflated=_replicated_rational(limit['variance_upper'])
                require(inflated>=variance+budget and (inflated-variance-budget)**2>=4*variance*budget, 'Variance upper endpoint fails midpoint/radius inflation')
                variance_lower,variance_upper=_replicated_interval(limit['variance_bound_certificate'])
                require(inflated==variance_upper and (variance_lower<=variance+budget or (variance_lower-variance-budget)**2<=4*variance*budget), 'Changed variance enclosure endpoint')
                log_lower,log_upper=_replicated_interval(limit['log_two_over_delta_certificate'])
                require(log_lower<=log_expected[0] and log_expected[1]<=log_upper, 'Confidence logarithm does not enclose its exact-series bounds')
                radius_lower,radius_upper=_replicated_interval(limit['sampling_radius_certificate'])
                coefficient=7*cap/Fraction(3*(n-1))
                root_lower=2*inflated*log_lower/n; root_upper=2*inflated*log_upper/n
                require(radius_upper>=coefficient*log_upper and (radius_upper-coefficient*log_upper)**2>=root_upper, 'Sampling radius upper endpoint omits the Bernstein construction')
                require(radius_lower<=coefficient*log_lower or (radius_lower-coefficient*log_lower)**2<=root_lower, 'Sampling radius lower endpoint exceeds its expression')
                tail=_replicated_rational(limit['population_tail_upper'])
                require(tail==_replicated_interval(study['tail_allowances'][arm])[1], 'Changed evaluator tail allowance')
                upper=_replicated_rational(limit['upper_limit'])
                limit_lower,limit_upper=_replicated_interval(limit['upper_limit_certificate'])
                require(upper==limit_upper and limit_lower<=mean+tail+radius_lower and upper>=mean+tail+radius_upper, 'Authoritative upper endpoint loses its sampling radius')
                require(_replicated_rational(stats['maximum_raw_width'])<=Fraction(1,100), 'Changed gain-width requirement')
                require(limit['sampling_law_certified'] is False and limit['financial_decision_certified'] is False and limit['empirical_bernstein_applied'] is True, 'Changed numerical certification scope')
                require(look['acceptance']=={target:upper<=Fraction(target) for target in ('1/2','1')}, 'Changed monetary acceptance')
                cost(look['original_prefix_cost'])
            require(set(result['decisions'])=={'1/2','1'}, 'Missing monetary tolerance decision')
            for target,decision in result['decisions'].items():
                require(set(decision)=={'accepted','required_costed_paths','required_eligible_paths','upper_limit','original_required_prefix_cost','full_budget_charged_if_unresolved','conservative_after_run_cost'}, 'Unexpected public decision field')
                accepted=[look for look in looks if look['acceptance'][target]]
                selected=accepted[0] if accepted else looks[-1]
                require(decision['accepted'] is bool(accepted) and decision['required_costed_paths']==selected['costed_paths'] and decision['required_eligible_paths']==selected['eligible_paths'] and decision['upper_limit']==selected['upper_limit']['upper_limit'] and decision['full_budget_charged_if_unresolved'] is (not bool(accepted)), 'Changed earliest decision or unresolved budget')
                require(decision['original_required_prefix_cost']==selected['original_prefix_cost'], 'Lost original prefix charge')
                augmented=cost(decision['conservative_after_run_cost']); original=cost(decision['original_required_prefix_cost'])
                require(all(augmented[clock]==original[clock]+correction[clock] for clock in original), 'Lost complete correction charge')
    return True


def render_multivariate_monetary(study):
    rows=[]
    for row in study['cases']:
        eu=row['assessors']['european_common']; zero=row['assessors']['zero']
        mean=_replicated_interval(eu['looks'][-1]['raw_gain_mean_certificate'])
        rows.append([str(row['fit_index']+1), f'{float(sum(mean)/2):.4f}',
            *[f"{float(_replicated_rational(x['upper_limit']['upper_limit'])):.4f}" for x in eu['looks']],
            f"{float(_replicated_rational(zero['looks'][-1]['upper_limit']['upper_limit'])):.4f}",
            f"{eu['decisions']['1']['original_required_prefix_cost']['cpu_seconds']:.2f}",
            f"{zero['decisions']['1']['original_required_prefix_cost']['cpu_seconds']:.2f}"])
    correction=study['correction_workflow_cost']['cpu_seconds']
    return table('Monetary assessment of the frozen five-asset policies','revision-multivariate-monetary',
        ['Fit',r'$\overline G^{\mathrm E}$',r'$U^{\mathrm E}_{4k}$',r'$U^{\mathrm E}_{16k}$',r'$U^0_{16k}$','EU cost (s)','Zero cost (s)'],rows,
        'The first two upper-limit columns use the costed looks of 4,096 and 16,384 paths. Fit 1 excludes the sixteen exposed rows from inference, giving eligible counts 4,080 and 16,368. Other fits retain all rows. Mean gains are interval midpoints. CPU columns retain reconstructed original standalone charges, including all generated and assessed paths and policy training once. The complete after-run correction adds '+f'{correction:.2f}'+' CPU seconds to every reported decision charge; independent verification costs are separate. Acceptance uses exact upper endpoints rather than their rounded table displays. Limits share a simultaneous 95 per cent conditional guarantee under the stated ideal sampling and observation-enclosure premises. Both assessors evaluate unchanged policies; unresolved outcomes do not establish excessive actual shortfall.')


def validate(data):
    import math
    from statistics import NormalDist

    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite result value')
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    def equal(left, right, message, tolerance=1e-10):
        if not math.isclose(left, right, rel_tol=0., abs_tol=tolerance):
            raise ValueError(message)

    def coverage_count(row, prefix):
        total, count = row['repetitions'], row[prefix+'_covered']
        if type(total) is not int or total != 5000 or type(count) is not int or not 0 <= count <= total:
            raise ValueError('Invalid coverage count')
        proportion = count/total
        if row[prefix+'_coverage'] != proportion:
            raise ValueError('Coverage proportion differs from its count')
        z = NormalDist().inv_cdf(.975)
        scale = 1+z*z/total
        centre = (proportion+z*z/(2*total))/scale
        radius = z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/scale
        lo, hi = row[prefix+'_coverage_wilson_95']
        equal(lo, max(0., centre-radius), 'Wilson interval differs from its count', 1e-12)
        equal(hi, min(1., centre+radius), 'Wilson interval differs from its count', 1e-12)

    def certificate(row):
        values = {}
        for name, record in row['mean_signed_gap_certificate'].items():
            denominator = int(record['denominator'])
            if denominator <= 0:
                raise ValueError('Certificate denominator must be positive')
            values[name] = Fraction(int(record['numerator']), denominator)
            equal(record['approximation'], float(values[name]), 'Certificate display differs from its rational value', 1e-12)
        if (set(values) != {'lower', 'upper', 'width'} or values['lower'] < 0 or
                values['upper'] < values['lower'] or values['width'] != values['upper']-values['lower']):
            raise ValueError('Signed certificate endpoint/width identity fails')
        equal(row['mean_signed_width'], float(values['width']), 'Signed mean width differs from its certificate', 1e-12)
        for value, name in zip(row['mean_signed_gap_interval'], ('lower', 'upper')):
            equal(value, float(values[name]), 'Signed interval differs from its certificate', 1e-12)
        if (row['mean_signed_width'] > row['mean_absolute_width']+1e-12 or
                row['all_signed_inside_absolute'] is not True):
            raise ValueError('Signed cancellation does not improve the absolute allowance')
        return values

    finite(data)
    if 'multivariate_monetary' in data:
        validate_multivariate_monetary(data['multivariate_monetary'])
    if 'decision_study' in data:
        validate_decision_study(data['decision_study'])
    if 'replicated_refinement' in data:
        validate_replicated_refinement(data['replicated_refinement'])
    if 'frozen_extension' in data:
        validate_frozen_extension(data['frozen_extension'],data.get('replicated_refinement'))
    if 'uniform_allocation' in data:
        validate_uniform_allocation(data['uniform_allocation'],data.get('frozen_extension'))
    if 'restricted_initial_right' in data:
        validate_restricted_initial_right(data['restricted_initial_right'],data.get('uniform_allocation'))
    if data['source_scope'] != 'Published mathematical models and newly simulated aggregate results. No subscription observations or model inputs.':
        raise ValueError('The declared input scope differs from the revision')
    coverage = data['coverage']
    for flag in ('floating_point_bounds_are_certified', 'numerical_oracle_is_rigorous_enclosure',
                 'cap_selection_uses_evaluation', 'simultaneous_claim_across_counts_or_methods',
                 'new_general_theorem_claim', 'new_market_data_acquired'):
        if coverage[flag] is not False:
            raise ValueError('Unsupported coverage or input scope: '+flag)
    if (coverage['mathematical_guarantees_conditional_on_model_and_pilot'] is not True or
            coverage['mathematical_one_sided_coverage'] != .975 or
            set(coverage['finite_sample_methods']) != {'cantelli', 'hoeffding', 'empirical_bernstein'}):
        raise ValueError('The conditional finite-sample guarantee differs')
    expected = {(martingale, method, count) for martingale in ('zero', 'terminal_call')
                for method in ('student', 'empirical_bernstein', 'hoeffding', 'cantelli')
                for count in (8, 32, 128, 512, 2048)}
    found = [(row['martingale'], row['method'], row['outer_paths']) for row in coverage['records']]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError('Missing or duplicated coverage records')
    oracle = coverage['oracle']
    if oracle['numerical_oracle_is_rigorous_enclosure'] is not False:
        raise ValueError('Unsupported rigorous numerical oracle claim')
    for row in coverage['records']:
        coverage_count(row, 'expected_gap')
        coverage_count(row, 'policy_loss')
        target = oracle['expected_gaps'][row['martingale']]['value']
        equal(row['expected_gap_target'], target, 'Incorrect martingale expectation target', 1e-12)
        equal(row['policy_loss_target'], oracle['true_two_right_policy_loss'], 'Incorrect policy-loss target', 1e-12)
        equal(row['mean_excess_over_expected_gap'], row['average_upper_limit']-target,
              'Upper excess differs from the expectation target')
    equal(oracle['true_two_right_policy_loss']+oracle['expiry_only_value'], oracle['two_right_optimal_value'],
          'The oracle value/loss identity fails', 1e-12)
    equal(oracle['expected_gaps']['terminal_call']['value'], oracle['true_two_right_policy_loss'],
          'The centred gap has nonzero dual slack', 1e-12)
    equal(oracle['expected_gaps']['zero']['value']-oracle['true_two_right_policy_loss'],
          oracle['zero_martingale_dual_slack'], 'The zero-martingale slack identity fails', 1e-12)
    expected_cases = {(1, 100.)} | {(d, s) for d in (2, 3, 5) for s in (90., 100., 110.)}
    found_cases = [(case['dimension'], case['spot']) for case in data['benchmark']['cases']]
    if len(found_cases) != len(expected_cases) or set(found_cases) != expected_cases:
        raise ValueError('Missing or duplicated benchmark cases')
    for case in data['benchmark']['cases']:
        e = case['estimates']
        equal(e['policy_value']['mean']-e['terminal_waiting']['mean'], e['matched_gain']['mean'],
              'The paired-gain identity fails')
        if any(r['observations'] != 200000 for r in e.values()):
            raise ValueError('The evaluation count differs')
        for estimate in e.values():
            lo, hi = estimate['pointwise_student_interval']
            if estimate['standard_error'] < 0 or not lo <= estimate['mean'] <= hi:
                raise ValueError('Invalid benchmark sampling interval')
    for row in data.get('conditional_certification', {}).get('runs', []):
        certificate(row)
    refinement = data.get('gain_refinement')
    for case in (refinement or {}).get('cases', []):
        if (set(case['strategies']) != {'cold_uniform', 'adaptive'} or
                case['outer_paths'] != refinement['outer_paths'] or
                case['frozen_model_policy_values_sha256'] != refinement['frozen_model_policy_values_sha256']):
            raise ValueError('Refinement comparison changes its strategies, paths or frozen fit')
        if case['all_path_certificates_overlap'] is not True:
            raise ValueError('Refinement strategies have non-overlapping path certificates')
        target = float(Fraction(case['required_maximum_path_gap_width']))
        if target <= 0:
            raise ValueError('Invalid refinement width target')
        intervals = []
        for row in case['strategies'].values():
            intervals.append(certificate(row))
            if (row['all_targets_met'] is not True or row['maximum_signed_width'] > target+1e-12 or
                    row['mean_signed_width'] > row['maximum_signed_width']+1e-12):
                raise ValueError('The required refinement width was not achieved')
            equal(row['combined_preparation_and_assessment_cpu_seconds'],
                  refinement['preparation_cost']['cpu_seconds']+row['assessment_cpu_seconds'],
                  'Standalone refinement cost omits or duplicates preparation', 1e-9)
        if max(value['lower'] for value in intervals) > min(value['upper'] for value in intervals):
            raise ValueError('Mean certificates for the same gain do not overlap')
    if refinement:
        for strategy, recorded in refinement['strategy_whole_study_budget_cpu_seconds_including_preparation_once'].items():
            total = refinement['preparation_cost']['cpu_seconds']+sum(
                case['strategies'][strategy]['assessment_cpu_seconds'] for case in refinement['cases'])
            equal(recorded, total, 'Whole-study cost omits or duplicates preparation', 1e-9)
    return True


def table(caption, label, headers, rows, note):
    return '\n'.join([r'\begin{table}[!htbp]', r'\centering\small',
        r'\caption{'+caption+r'}\label{tab:'+label+'}',
        r'\begin{tabular}{@{}'+'l'+'r'*(len(headers)-1)+r'@{}}',
        r'\toprule', ' & '.join(headers)+r' \\', r'\midrule',
        *[' & '.join(row)+r' \\' for row in rows], r'\bottomrule', r'\end{tabular}',
        r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}', r'\end{table}'])


def render(data):
    validate(data)
    records = {(r['martingale'], r['method'], r['outer_paths']): r for r in data['coverage']['records']}
    methods = ('student', 'empirical_bernstein', 'hoeffding', 'cantelli')
    # The implementation uses these exact recorded method identifiers.
    if ('terminal_call', 'empirical_bernstein', 2048) not in records:
        methods = ('student', 'clipped_empirical_bernstein', 'clipped_hoeffding', 'cantelli')
    result = {'coverage': table('Observed coverage of the centred-gap upper limits (per cent)', 'revision-coverage',
        ['Paths', 'Student', 'Clipped EB', 'Clipped Hoeffding', 'Cantelli'],
        [[f'{n:,}', *[f"{100*records['terminal_call', m, n]['expected_gap_coverage']:.2f}" for m in methods]] for n in (8,32,128,512,2048)],
        'Each entry uses 5,000 independent repetitions and targets the two-right shortfall 1.156991. Nominal one-sided coverage is 97.5 per cent. The clipping cap is frozen using an independent pilot.')}
    rows = []
    for case in data['benchmark']['cases']:
        e = case['estimates']; p = e['policy_value']; g = e['matched_gain']
        rows.append([str(case['dimension']), f"{case['spot']:.0f}", f"{p['mean']:.5f}",
                     f"{p['standard_error']:.5f}", f"{g['mean']:.5f}", f"{g['standard_error']:.5f}"])
    result['benchmarks'] = table('Independent max-call policy evaluation', 'revision-benchmarks',
        ['Assets', 'Spot', 'Policy value', 'SE', 'Expiry gain', 'Gain SE'], rows,
        'Present-value units. Each fixed degree-two policy uses 50,000 training paths and 200,000 independent evaluation paths. Expiry gain is a matched-path comparison with always waiting. Cases share a seed layout.')
    comparison = data.get('gain_refinement')
    if comparison:
        rows = []
        for case in comparison['cases']:
            cold, adaptive = case['strategies']['cold_uniform'], case['strategies']['adaptive']
            target = float(Fraction(case['required_maximum_path_gap_width']))
            rows.append([f"{target:.3g}", str(cold['conditional_integrals']),
                str(adaptive['conditional_integrals']), f"{cold['certificate_phase_cpu_seconds']:.3f}",
                f"{adaptive['certificate_phase_cpu_seconds']:.3f}", f"{adaptive['maximum_signed_width']:.3g}"])
        result['certification'] = table('Conditional-integration cost at a required maximum gain width', 'revision-certification',
            [r'$\varepsilon$', 'Uniform calls', 'Refined calls', 'Uniform CPU', 'Refined CPU', 'Refined width'], rows,
            'Both strategies meet the stated maximum path-gain width. Achieved widths can differ. Uniform costs exclude unnecessary coarse initialisation; adaptive costs include every coarse and refined query. Preparation and decision checks are reported separately. Calls count finite maximum-hinge integrals; CPU is measured in seconds.')
    else:
        result['certification'] = 'The larger, repeated refinement comparison remains in progress. The completed component and pathwise prototype establish executable enclosures, rather than a population pricing or cost advantage.'
    decision = data.get('decision_study')
    if decision:
        records = {(row['policy'], row['method'], row['loss_tolerance']): row for row in decision['decisions']}
        losses = {}
        for row in decision['reference']['policies']:
            policy = 'always_wait' if row['threshold'] is None else 'threshold_'+str(int(row['threshold']['approximation']))
            lower, upper = (Fraction(int(row['loss'][key]['numerator']), int(row['loss'][key]['denominator']))
                            for key in ('lower', 'upper'))
            losses[policy] = float((lower+upper)/2)
        rows = []
        for policy in ('threshold_100', 'threshold_110', 'threshold_120', 'threshold_130', 'always_wait'):
            label = 'Always wait' if policy == 'always_wait' else r'$\theta='+policy.split('_')[1]+'$'
            final = records[policy, 'terminal_call', .5]
            rows.append([label, f"{losses[policy]:.6f}",
                         *[f"{100*records[policy, 'terminal_call', tolerance]['certification_rate']:.2f}" for tolerance in (.1, .25, .5)],
                         f"{final['mean_paths']:,.0f}", f"{1000*final['mean_one_off_cpu_seconds']:.3f}"])
        result['decision'] = table('Certification of monetary shortfall tolerances', 'revision-decision',
            ['Policy', 'Shortfall', r'$0.10$ (\%)', r'$0.25$ (\%)', r'$0.50$ (\%)', 'Mean paths', 'Charged CPU (ms)'], rows,
            'Certification frequencies for the terminal-call assessor, across 256 independent repetitions. The simultaneous family contains five policies, three methods and five fixed looks, hence 75 allocated bounds; one bound serves all three tolerances. Paths and charged CPU at tolerance 0.50 include unresolved assessments. Costs add the assessment prefix to one frozen independent setup measurement, which is not rerun in each repetition. Shortfalls are display midpoints of certified intervals. Simulation and confidence-limit arithmetic remain floating-point calculations.')
    if 'replicated_refinement' in data:
        result['replicated_refinement'] = render_replicated_refinement(data['replicated_refinement'])
        result['replicated_tradeoff'] = render_replicated_tradeoff(data['replicated_refinement'])
    if 'frozen_extension' in data:
        result['frozen_extension'] = render_frozen_extension(data['frozen_extension'])
    if 'uniform_allocation' in data:
        result['uniform_allocation'] = render_uniform_allocation(data['uniform_allocation'])
    if 'multivariate_monetary' in data:
        result['multivariate_monetary'] = render_multivariate_monetary(data['multivariate_monetary'])
    return result


def synchronise(source, tables, write=False):
    for name, expected in tables.items():
        start, end = f'% Begin revision-table:{name}', f'% End revision-table:{name}'
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError(f'Missing or repeated revision table: {name}')
        first, rest = source.split(start); found, last = rest.split(end)
        if not write and found.strip() != expected:
            raise ValueError(f'Revision table differs: {name}')
        if write:
            source = first+start+'\n'+expected+'\n'+end+last
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    data = json.loads((PAPER/'revision_results.json').read_text())
    source = synchronise((PAPER/'main.tex').read_text(), render(data), args.write)
    if args.write:
        (PAPER/'main.tex').write_text(source)
    print('Checked the revised tables against public-model aggregate results.')


if __name__ == '__main__':
    main()
