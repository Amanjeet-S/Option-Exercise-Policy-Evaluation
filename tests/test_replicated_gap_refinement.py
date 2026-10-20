"""Small mocked studies check registration, paired units and safe resumption.

The fixture kernels do not train or assess research models. Their declared
path counts exercise the orchestrator's control and exact aggregation logic.
"""

from copy import deepcopy
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('replicated_gap', SCRIPTS/'study_replicated_gap_refinement.py')
component = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(component)


def interval(lower, upper):
    def number(value):
        return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))
    return dict(lower=number(lower), upper=number(upper), width=number(upper-lower))


def fake_prepare(**kwargs):
    if kwargs != dict(dimension=5, training_count=50000, value_count=50000,
                      evaluation_count=256, seed=kwargs['seed']):
        raise AssertionError('Preparation must use the registered counts')
    model = dict(spots=[100.]*5, volatilities=[.2]*5, dividends=[.1]*5,
        correlation=[[float(i == j) for j in range(5)] for i in range(5)],
        rate=.05, strike=100., exercise_times=[3.*i/9 for i in range(10)])
    coefficients = [[0.]*10 for _ in range(10)]
    coefficients[1][0] = float(kwargs['seed']-2026100410)/1000
    phases = {name: dict(cpu_seconds=value, wall_seconds=value) for name, value in (
        ('policy_training', .002), ('hinge_value_stream_and_targets', .003), ('hinge_regression', .003),
        ('outer_evaluation_and_record_construction', .002), ('preparation_bookkeeping', 0.))}
    return dict(model=model, knots=[80., 100., 120., 140.], coefficients=coefficients,
        frozen_policy=[[] for _ in range(10)], states=[[[100.]*5 for _ in range(10)] for _ in range(256)],
        stopping_indices=[9]*256, design=dict(root_seed=kwargs['seed'], training_paths=50000,
            value_paths=50000, outer_paths=256, spawn_keys=[[0], [1], [2]], policy_degree=2,
            evaluation_independent_of_both_fitting_streams=True),
        preparation_cost=dict(cpu_seconds=.01, wall_seconds=.01), preparation_phases=phases)


def fake_case(record, case, *, precision, coarse_tolerance):
    if precision != 96 or coarse_tolerance != '1':
        raise AssertionError('Use the prior numerical design')
    strategies, observations = {}, {}
    for name in ('cold_uniform', 'adaptive'):
        lower, upper = (F(1), F(1001, 1000)) if name == 'cold_uniform' else (F(999, 1000), F(1002, 1000))
        calls = 2 if name == 'cold_uniform' else 1
        cpu = 1. if name == 'cold_uniform' else .5
        observations[name] = [dict(path_index=index, stopping_index=9,
            signed_gap_certificate=interval(lower, upper), absolute_gap_certificate=interval(F(9, 10), F(11, 10)),
            signed_width=float(upper-lower), total_work=dict(conditional_integrals=calls, integrand_evaluations=5*calls))
            for index in range(256)]
        strategies[name] = dict(mean_signed_gap_certificate=interval(lower, upper),
            mean_signed_gap_interval=[float(lower), float(upper)], mean_signed_width=float(upper-lower),
            maximum_signed_width=float(upper-lower), mean_absolute_width=.2,
            approximate_midpoint_mean=float((lower+upper)/2), approximate_midpoint_mean_standard_error=0.,
            midpoint_mean_numerical_error_bound=float((upper-lower)/2),
            conditional_integrals=256*calls, integrand_evaluations=1280*calls,
            all_targets_met=True, all_signed_inside_absolute=True, assessment_cpu_seconds=cpu,
            input_and_policy_validation_cpu_seconds=.1, certificate_phase_cpu_seconds=cpu-.1,
            combined_preparation_and_assessment_cpu_seconds=record['preparation_cost']['cpu_seconds']+cpu)
    return dict(name=case['name'], required_maximum_path_gap_width=case['target'], strategy_order=case['order'],
        frozen_model_policy_values_sha256=component.frozen_model_hash(record), outer_paths=256, precision_bits=96,
        strategies=strategies, observations=observations, all_path_certificates_overlap=True,
        adaptive_over_cold_uniform_integral_ratio=.5,
        adaptive_over_cold_uniform_certificate_phase_cpu_ratio=.4/.9,
        actual_case_cpu_seconds=1.5, actual_case_wall_seconds=1.5)


class StudyFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root/'data/private/study'
        self.sources = {'fixture_kernel': 'a'*64}
        self.patchers = [patch.object(component, 'ROOT', self.root),
                         patch.object(component, 'source_hashes', return_value=self.sources),
                         patch.object(component, 'prepare', side_effect=fake_prepare),
                         patch.object(component, 'case_result', side_effect=fake_case),
                         patch('builtins.print')]
        self.mocks = [p.start() for p in self.patchers]
        self.addCleanup(self.temporary.cleanup)
        for p in self.patchers:
            self.addCleanup(p.stop)

    def prepared(self):
        component.preparation(self.directory)

    def assessed(self):
        self.prepared()
        return component.assessment(self.directory)


class ReplicatedStudyChecks(StudyFixture, unittest.TestCase):
    def test_prior_design_precedes_every_training_call_and_fixes_eight_seeds(self):
        seeds = []
        def inspect(**kwargs):
            design = json.loads((self.directory/'design.json').read_bytes())
            self.assertTrue(design['recorded_before_any_training'])
            self.assertEqual(len(design['fits']), 8)
            self.assertFalse((self.directory/'prepared.json').exists())
            seeds.append(kwargs['seed'])
            return fake_prepare(**kwargs)
        self.mocks[2].side_effect = inspect
        self.prepared()
        self.assertEqual(seeds, list(range(2026100410, 2026100418)))
        design = json.loads((self.directory/'design.json').read_bytes())
        for index, fit in enumerate(design['fits']):
            self.assertEqual(fit['case']['order'], ['cold_uniform', 'adaptive'] if index % 2 == 0 else ['adaptive', 'cold_uniform'])
            self.assertEqual(fit['polynomial_degree_order'], [2, 3] if index % 2 == 0 else [3, 2])
            self.assertEqual(fit['verified_polynomial_method_order'], ['polynomial_all_dates', 'hybrid_exact_terminal']
                             if index % 2 == 0 else ['hybrid_exact_terminal', 'polynomial_all_dates'])
        self.assertEqual(len(design['planned_established_comparators']), 2)
        with self.assertRaises(ValueError):
            component.preparation(self.directory)

    def test_private_outputs_are_required_before_any_training(self):
        with self.assertRaises(ValueError):
            component.preparation(self.root/'public/study')
        self.assertFalse(self.mocks[2].called)

    def test_assessment_has_eight_statistical_units_and_complete_pricing_budgets(self):
        summary = self.assessed()
        self.assertEqual(self.mocks[3].call_count, 8)
        self.assertEqual(summary['completed_fits'], 8)
        self.assertEqual(summary['fitting_units'], 8)
        self.assertEqual(summary['outer_paths_per_fit'], 256)
        self.assertIn('not 2,048 independent', summary['statistical_unit'])
        self.assertEqual(summary['distinct_frozen_model_policy_value_hashes'], 8)
        ratios = summary['paired_adaptive_over_cold_uniform_ratios']
        self.assertEqual(ratios['integrals'], dict(fitting_units=8, minimum=.5, median=.5, maximum=.5))
        self.assertAlmostEqual(ratios['charged_pricing_budget_cpu']['median'], .51/1.01)
        for case in summary['cases']:
            self.assertEqual(case['preparation_cost']['cpu_seconds'], .01)
            self.assertEqual(case['strategies']['adaptive']['combined_preparation_and_assessment_cpu_seconds'], .51)
        completion = json.loads((self.directory/'assessment_completion.json').read_bytes())
        self.assertIn('result and summary writing', completion['timing_scope'])
        self.assertGreaterEqual(completion['complete_assessment_validation_workflow_cost']['cpu_seconds'], 0)

    def test_complete_resume_validates_results_without_new_kernel_calls_or_writes(self):
        expected = self.assessed()
        before = {str(p.relative_to(self.directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.directory.rglob('*.json')}
        self.mocks[3].side_effect = AssertionError('Completed fits must not rerun')
        self.assertEqual(component.assessment(self.directory), expected)
        after = {str(p.relative_to(self.directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.directory.rglob('*.json')}
        self.assertEqual(before, after)

    def test_modified_design_and_sources_fail_before_assessment(self):
        self.prepared()
        design_path = self.directory/'design.json'
        original = design_path.read_bytes()
        design = json.loads(original); design['fits'][0]['seed'] += 1
        design_path.write_text(json.dumps(design))
        with self.assertRaises(ValueError):
            component.assessment(self.directory)
        design_path.write_bytes(original)
        self.mocks[1].return_value = {'fixture_kernel': 'b'*64}
        with self.assertRaises(ValueError):
            component.assessment(self.directory)
        self.assertFalse(self.mocks[3].called)

    def test_altered_input_is_rejected_even_before_its_kernel_runs(self):
        self.prepared()
        path = self.directory/'fit_07/inputs.json'
        record = json.loads(path.read_bytes()); record['stopping_indices'][0] = 8
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            component.assessment(self.directory)
        self.assertFalse(self.mocks[3].called)

    def test_partial_failure_preserves_attempt_and_never_silently_restarts(self):
        self.prepared()
        calls = []
        def failure(record, case, **kwargs):
            calls.append(case['name'])
            if case['name'] == 'fit_02':
                raise ArithmeticError('Fixture width could not be certified')
            return fake_case(record, case, **kwargs)
        self.mocks[3].side_effect = failure
        with self.assertRaises(ArithmeticError):
            component.assessment(self.directory)
        self.assertTrue((self.directory/'fit_01/assessment_completion.json').exists())
        self.assertTrue((self.directory/'fit_02/failure.json').exists())
        with self.assertRaises(ValueError):
            component.assessment(self.directory)
        self.assertEqual(calls, ['fit_00', 'fit_01', 'fit_02'])
        self.assertFalse((self.directory/'summary.json').exists())

    def test_tampered_completion_file_is_rejected(self):
        self.assessed()
        path = self.directory/'fit_00/result.json'
        result = json.loads(path.read_bytes()); result['strategies']['adaptive']['mean_signed_width'] += .1
        path.write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            component.assessment(self.directory)

    def test_derived_results_are_checked_even_if_completion_digest_is_rewritten(self):
        self.assessed()
        path = self.directory/'fit_00/result.json'
        result = json.loads(path.read_bytes()); result['strategies']['adaptive']['conditional_integrals'] += 1
        path.write_text(json.dumps(result))
        completion_path = self.directory/'fit_00/assessment_completion.json'
        completion = json.loads(completion_path.read_bytes())
        completion['result_sha256'] = component.digest(path.read_bytes())
        completion_path.write_text(json.dumps(completion))
        with self.assertRaises(ValueError):
            component.assessment(self.directory)

    def test_nonoverlapping_actual_path_certificates_are_rejected(self):
        self.prepared()
        def separated(record, case, **kwargs):
            result = fake_case(record, case, **kwargs)
            result['observations']['adaptive'][0]['signed_gap_certificate'] = interval(F(2), F(2001, 1000))
            return result
        self.mocks[3].side_effect = separated
        with self.assertRaises(ValueError):
            component.assessment(self.directory)
        self.assertTrue((self.directory/'fit_00/failure.json').exists())

    def test_summary_is_reconstructed_from_fit_results_when_resuming(self):
        self.assessed()
        path = self.directory/'summary.json'
        summary = json.loads(path.read_bytes()); summary['completed_fits'] = 7
        path.write_text(json.dumps(summary))
        completion_path = self.directory/'assessment_completion.json'
        completion = json.loads(completion_path.read_bytes()); completion['summary_sha256'] = component.digest(path.read_bytes())
        completion_path.write_text(json.dumps(completion))
        with self.assertRaises(ValueError):
            component.assessment(self.directory)

    def test_polynomial_requires_completed_primary_assessments(self):
        self.prepared()
        with patch.object(component, 'polynomial_api', side_effect=AssertionError('No lazy import before readiness')):
            with self.assertRaises(ValueError):
                component.polynomial(self.directory)


    def test_preparation_failure_retains_prior_design_and_completed_inputs(self):
        def fail_second(**kwargs):
            if kwargs['seed'] == 2026100411:
                raise ArithmeticError('Fixture preparation failure')
            return fake_prepare(**kwargs)
        self.mocks[2].side_effect = fail_second
        with self.assertRaises(ArithmeticError):
            self.prepared()
        ledger = json.loads((self.directory/'preparation_failure.json').read_bytes())
        self.assertEqual(ledger['failed_fitting_unit_index'], 1)
        self.assertEqual(ledger['root_seed'], 2026100411)
        self.assertEqual(ledger['completed_fits'], 1)
        self.assertEqual(ledger['failed_phase'], 'scientific_preparation')
        self.assertFalse(ledger['failed_fit_input_saved'])
        self.assertEqual(ledger['message'], 'Fixture preparation failure')
        self.assertTrue((self.directory/'fit_00/inputs.json').exists())
        self.assertFalse((self.directory/'prepared.json').exists())
        self.assertEqual(ledger['design_sha256'], component.digest((self.directory/'design.json').read_bytes()))
        before = {str(path):path.read_bytes() for path in self.directory.rglob('*.json')}
        with self.assertRaises(ValueError):
            self.prepared()
        self.assertEqual(before, {str(path):path.read_bytes() for path in self.directory.rglob('*.json')})

try:
    import numpy
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@unittest.skipUnless(NUMPY_AVAILABLE, 'The polynomial numerical checks use the research environment')
class PolynomialModesChecks(StudyFixture, unittest.TestCase):
    def polynomial_api(self):
        import validate_polynomial_dual as library
        api = SimpleNamespace(**{name: getattr(library, name) for name in
            ('np', 'MaxCallModel', 'exponent_vectors', 'dual_paths', 'sample_summary', 'paired_comparison')})
        def run(record, hinge, *, degrees):
            model = api.MaxCallModel(**record['model'])
            states, stops = api.np.asarray(record['states']), api.np.asarray(record['stopping_indices'])
            result = dict(frozen_policy_preserved=True, supplied_outer_stream_reproduced=True,
                no_clipping_of_fitted_values=True, numerical_certificate=False, statistical_coverage_certificate=False,
                sample_count=256, value_fitting_paths=50000, value_spawn_key=[1], root_seed=record['design']['root_seed'],
                shared_value_stream_and_target_cpu_seconds=.1, stream_and_policy_reconstruction_cpu_seconds=.01, degrees={})
            for degree in degrees:
                powers = api.exponent_vectors(5, degree)
                coefficients = [api.np.zeros(len(powers)) for _ in range(10)]
                observed = api.dual_paths(states, model, stops, coefficients, degree)
                result['degrees'][str(degree)] = dict(degree=degree, basis_dimension=len(powers),
                    basis_exponents=[list(value) for value in powers], coefficients=[value.tolist() for value in coefficients],
                    all_gap_observations_nonnegative=True,
                    **{field: api.sample_summary(observed[field]) for field in ('gap', 'dual', 'policy_reward', 'stopped_martingale')},
                    paired_against_finest_hinge=api.paired_comparison(observed['gap'], hinge['observations']['adaptive']),
                    polynomial_fit_cpu_seconds=.2, polynomial_interior_fit_cpu_seconds=.15,
                    terminal_polynomial_fit_cpu_seconds=.05, dual_assessment_cpu_seconds=.01,
                    common_policy_and_outer_preparation_cpu_seconds=.004,
                    standalone_polynomial_pricing_cpu_seconds=.314,
                    standalone_polynomial_pricing_plus_validation_cpu_seconds=.324,
                    hybrid_setup_pricing_cpu_seconds=.254)
            return result
        api.run = run
        return api

    def test_polynomial_degrees_follow_registration_and_resume_verified_coefficients(self):
        self.assessed()
        api = self.polynomial_api()
        with patch.object(component, 'polynomial_api', return_value=api):
            outputs = component.polynomial(self.directory)
            self.assertEqual(len(outputs), 8)
            for index, output in enumerate(outputs):
                self.assertEqual(list(output['degrees']), ['2', '3'] if index % 2 == 0 else ['3', '2'])
            api.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('No polynomial refit on resume'))
            self.assertEqual(component.polynomial(self.directory), outputs)

    def test_polynomial_mean_tamper_is_detected_after_digest_replacement(self):
        self.assessed()
        api = self.polynomial_api()
        with patch.object(component, 'polynomial_api', return_value=api):
            component.polynomial(self.directory)
            path = self.directory/'fit_00/polynomial.json'
            result = json.loads(path.read_bytes()); result['degrees']['2']['gap']['mean'] += .01
            path.write_text(json.dumps(result))
            completion_path = self.directory/'fit_00/polynomial_completion.json'
            completion = json.loads(completion_path.read_bytes()); completion['polynomial_sha256'] = component.digest(path.read_bytes())
            completion_path.write_text(json.dumps(completion))
            with self.assertRaises(ValueError):
                component.polynomial(self.directory)

    def test_failed_polynomial_fit_is_retained_without_a_refit(self):
        self.assessed()
        api = self.polynomial_api()
        api.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(ArithmeticError('Fixture fit failure'))
        with patch.object(component, 'polynomial_api', return_value=api):
            with self.assertRaises(ArithmeticError):
                component.polynomial(self.directory)
            self.assertTrue((self.directory/'fit_00/polynomial_failure.json').exists())
            api.run = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('A failed fit cannot resume'))
            with self.assertRaises(ValueError):
                component.polynomial(self.directory)
            self.assertFalse((self.directory/'fit_00/polynomial.json').exists())

    def verified_api(self):
        def validate_export(record, polynomial, degrees):
            self.assertEqual(list(polynomial['degrees']), [str(value) for value in degrees])
        def assess(record, polynomial, *, target, precision, degrees, methods):
            self.assertEqual(target, F(1, 100)); self.assertEqual(precision, 96)
            result = dict(root_seed=record['design']['root_seed'], outer_paths=256, precision_bits=96,
                observed_policy_decisions_checked=True, independent_arm_costs=True, exact_terminal_model_discount=True,
                new_general_theorem_claim=False, path_gap_width_target=interval(target, target),
                input_and_policy_validation_cpu_seconds=.01, degrees={})
            for degree in degrees:
                result['degrees'][str(degree)] = dict(methods={})
                for method in methods:
                    observations = [dict(path_index=index, stopping_index=9,
                        signed_gap_certificate=interval(F(1), F(1001, 1000)),
                        selected_date_gap_certificate=interval(F(0), F(0)), conditional_integrals=int(method == 'hybrid_exact_terminal'),
                        integrand_evaluations=5*int(method == 'hybrid_exact_terminal'), polynomial_terms_evaluated=2, moment_factor_exponentials=2*int(index == 0),
                        moment_factor_cache_builds=int(index == 0), moment_factor_cache_hits=int(index > 0),
                        moment_factors_reused=2*int(index > 0))
                        for index in range(256)]
                    setup = .254 if method == 'hybrid_exact_terminal' else .304
                    row = dict(degree=degree, method=method, outer_paths=256, observations=observations,
                        mean_signed_gap_certificate=interval(F(1), F(1001, 1000)), mean_signed_width=.001,
                        maximum_signed_width=.001, approximate_midpoint_mean=1.0005,
                        approximate_midpoint_mean_standard_error=0., midpoint_mean_numerical_error_bound=.0005,
                        all_targets_met=True, conditional_integrals=256*int(method == 'hybrid_exact_terminal'),
                        integrand_evaluations=1280*int(method == 'hybrid_exact_terminal'), polynomial_terms_evaluated=512, moment_factor_exponentials=2,
                        moment_factor_cache_builds=1, moment_factor_cache_hits=255, moment_factors_reused=510,
                        assessment_cpu_seconds=.2, standalone_assessment_cpu_seconds_including_validation=.21,
                        retained_setup_pricing_cpu_seconds=setup, standalone_pricing_cpu_seconds=setup+.2,
                        standalone_verified_workflow_cpu_seconds=setup+.21)
                    result['degrees'][str(degree)]['methods'][method] = row
            return result
        return SimpleNamespace(validate_export=validate_export, assess=assess)

    def test_verified_modes_bind_exports_and_resume_without_reassessment(self):
        self.assessed()
        with patch.object(component, 'polynomial_api', return_value=self.polynomial_api()):
            component.polynomial(self.directory)
        api = self.verified_api()
        with patch.object(component, 'verified_polynomial_api', return_value=api):
            outputs = component.certify_polynomial(self.directory)
            self.assertEqual(len(outputs), 8)
            for index, output in enumerate(outputs):
                expected = ['polynomial_all_dates', 'hybrid_exact_terminal'] if index % 2 == 0 else ['hybrid_exact_terminal', 'polynomial_all_dates']
                self.assertEqual(output['verified_method_order'], expected)
                for row in output['degrees'].values():
                    self.assertEqual(list(row['methods']), expected)
            api.assess = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('No reassessment on resume'))
            self.assertEqual(component.certify_polynomial(self.directory), outputs)
            path = self.directory/'fit_00/verified_polynomial.json'
            result = json.loads(path.read_bytes()); result['degrees']['2']['methods']['hybrid_exact_terminal']['mean_signed_width'] = .2
            path.write_text(json.dumps(result))
            marker = self.directory/'fit_00/verified_polynomial_completion.json'
            completion = json.loads(marker.read_bytes()); completion['verified_polynomial_sha256'] = component.digest(path.read_bytes())
            marker.write_text(json.dumps(completion))
            with self.assertRaises(ValueError):
                component.certify_polynomial(self.directory)

    def test_verified_cache_work_tamper_is_rejected_even_with_a_new_digest(self):
        self.assessed()
        with patch.object(component, 'polynomial_api', return_value=self.polynomial_api()):
            component.polynomial(self.directory)
        with patch.object(component, 'verified_polynomial_api', return_value=self.verified_api()):
            component.certify_polynomial(self.directory)
            path = self.directory/'fit_00/verified_polynomial.json'
            result = json.loads(path.read_bytes())
            result['degrees']['2']['methods']['polynomial_all_dates']['moment_factor_cache_builds'] += 1
            path.write_text(json.dumps(result))
            marker = self.directory/'fit_00/verified_polynomial_completion.json'
            completion = json.loads(marker.read_bytes())
            completion['verified_polynomial_sha256'] = component.digest(path.read_bytes())
            marker.write_text(json.dumps(completion))
            with self.assertRaises(ValueError):
                component.certify_polynomial(self.directory)

    def test_failed_verified_comparison_is_not_restarted(self):
        self.assessed()
        with patch.object(component, 'polynomial_api', return_value=self.polynomial_api()):
            component.polynomial(self.directory)
        api = self.verified_api()
        api.assess = lambda *_args, **_kwargs: (_ for _ in ()).throw(ArithmeticError('Fixture enclosure failure'))
        with patch.object(component, 'verified_polynomial_api', return_value=api):
            with self.assertRaises(ArithmeticError):
                component.certify_polynomial(self.directory)
            self.assertTrue((self.directory/'fit_00/verified_polynomial_failure.json').exists())
            api.assess = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('No retry in this version'))
            with self.assertRaises(ValueError):
                component.certify_polynomial(self.directory)
            self.assertFalse((self.directory/'fit_00/verified_polynomial.json').exists())


@unittest.skipUnless(NUMPY_AVAILABLE, 'Actual small preparation uses the research environment')
class PreparationBoundaryIntegrationChecks(unittest.TestCase):
    def test_actual_small_prepare_normalises_before_exact_input_validation(self):
        from certify_multivariate_gap import prepare as scientific_prepare
        from qf_research.optimal_stopping.multivariate_benchmark import published_max_call
        raw = scientific_prepare(dimension=5, training_count=64, value_count=64,
                                 evaluation_count=2, seed=2026100410)
        self.assertIsInstance(raw['model']['correlation'][0], tuple)
        with self.assertRaisesRegex(ValueError, 'independent homogeneous positive-volatility assets'):
            component.check_inputs(raw)
        normalised = component.normalise_prepared_record(raw)
        model, states, stops, coefficients, times = component.check_inputs(normalised)
        self.assertEqual(model, component.expected_benchmark_model())
        self.assertEqual(model['exercise_times'], list(published_max_call(5, 100.).exercise_times))
        self.assertEqual(times, [F(value) for value in model['exercise_times']])
        self.assertEqual(states, raw['states'])
        self.assertEqual(stops, raw['stopping_indices'])
        self.assertEqual(coefficients, raw['coefficients'])
        self.assertEqual(component.frozen_model_hash(normalised), component.frozen_model_hash(raw))
        self.assertEqual(normalised, component.normalise_prepared_record(normalised))

    def test_nonfinite_values_are_rejected_before_saved_input_creation(self):
        with self.assertRaises(ValueError):
            component.normalise_prepared_record({'states':float('nan')})


if __name__ == '__main__':
    unittest.main()
