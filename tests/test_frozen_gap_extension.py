"""Verify retained-fit stream separation, numerical accounting and safe resumption."""

from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent/'src'))
SPEC = importlib.util.spec_from_file_location('frozen_extension', SCRIPTS/'study_frozen_gap_extension.py')
component = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(component)


def old_record(seed=2026100410):
    model = component.original.expected_benchmark_model()
    phases = {name:dict(cpu_seconds=cost, wall_seconds=cost) for name, cost in (
        ('policy_training', .001), ('hinge_value_stream_and_targets', .002),
        ('hinge_regression', .003), ('outer_evaluation_and_record_construction', .0005),
        ('preparation_bookkeeping', .0001))}
    return dict(model=model, knots=[80., 100., 120., 140.],
        coefficients=[[0.]*10 for _ in range(10)], frozen_policy=[[] for _ in range(10)],
        states=[[[100.]*5 for _ in range(10)] for _ in range(256)], stopping_indices=[9]*256,
        design=dict(root_seed=seed, training_paths=50000, value_paths=50000, outer_paths=256,
            spawn_keys=[[0], [1], [2]], policy_degree=2,
            evaluation_independent_of_both_fitting_streams=True),
        preparation_phases=phases,
        preparation_cost={key:sum(row[key] for row in phases.values()) for key in ('cpu_seconds', 'wall_seconds')})


def polynomial():
    from certify_polynomial_dual import exponents
    basis = [list(powers) for powers in exponents(5, 3)]
    return dict(root_seed=2026100410, sample_count=256, value_fitting_paths=50000, value_spawn_key=[1],
        shared_value_stream_and_target_cpu_seconds=.004,
        degrees={'3':dict(basis_exponents=basis, coefficients=[[0.]*len(basis) for _ in range(10)],
            polynomial_interior_fit_cpu_seconds=.005, terminal_polynomial_fit_cpu_seconds=.002,
            hybrid_setup_pricing_cpu_seconds=.0105)})


def fake_arm(record, view, name):
    count = len(record['states'])
    intervals, observations = [], []
    offset, width = {'cold_uniform':(F(2,5), F(1,2000)),
                     'adaptive':(F(1999,5000), F(1,1000)),
                     'hybrid_exact_terminal':(F(6,5), F(1,1250))}[name]
    calls = 2 if name == 'cold_uniform' else 1
    cpu = {'cold_uniform':.02, 'adaptive':.01, 'hybrid_exact_terminal':.005}[name]
    for index, stop in enumerate(record['stopping_indices']):
        lo, hi = offset+F(index, 1000), offset+F(index, 1000)+width
        intervals.append((lo, hi))
        row = dict(path_index=index, stopping_index=stop,
                   signed_gap_certificate=component.interval_record(lo, hi))
        if name == 'hybrid_exact_terminal':
            row.update(selected_date_gap_certificate=component.interval_record(F(0), F(0)),
                       conditional_integrals=calls, integrand_evaluations=3*calls)
        else:
            row.update(absolute_gap_certificate=component.interval_record(lo-F(1,100), hi+F(1,100)),
                       total_work=dict(conditional_integrals=calls, integrand_evaluations=3*calls))
        observations.append(row)
    aggregate = component.aggregate_intervals(intervals)
    return dict(observations=observations, all_targets_met=True,
        mean_signed_gap_certificate=aggregate['mean_certificate'],
        mean_signed_width=float(width), maximum_signed_width=float(width),
        approximate_midpoint_mean=aggregate['approximate_midpoint_mean'],
        approximate_midpoint_mean_standard_error=aggregate['approximate_midpoint_mean_standard_error'],
        midpoint_mean_numerical_error_bound=aggregate['midpoint_mean_numerical_error_bound'],
        conditional_integrals=calls*count, integrand_evaluations=3*calls*count,
        assessment_cpu_seconds=cpu, assessment_wall_seconds=cpu,
        input_and_policy_validation_cpu_seconds=.001, certificate_phase_cpu_seconds=cpu-.001)


class MathematicalChecks(unittest.TestCase):
    def test_near_balanced_positions_and_relative_hinge_order(self):
        fits = component.fit_designs()
        self.assertEqual([row['root_seed'] for row in fits], list(range(2026100410,2026100418)))
        for arm in component.ARMS:
            counts = [sum(row['arm_order'][position] == arm for row in fits) for position in range(3)]
            self.assertEqual(sorted(counts), [2,3,3])
        self.assertEqual(sum(row['arm_order'].index('cold_uniform') < row['arm_order'].index('adaptive')
                             for row in fits), 4)

    def test_actual_small_model_replays_only_new_stream_and_preserves_coefficients(self):
        import numpy as np
        from qf_research.optimal_stopping.multivariate_benchmark import MaxCallModel, simulate_max_call
        old = old_record()
        retained = deepcopy(old)
        states, stops, key = component.fresh_outer(old, count=8)
        self.assertEqual(old, retained)
        self.assertEqual(key, [3])
        self.assertEqual(component.fresh_outer(old, count=8), (states, stops, key))
        self.assertEqual(stops, [9]*8)
        model = MaxCallModel(**old['model'])
        streams = np.random.SeedSequence(old['design']['root_seed']).spawn(4)
        expected = simulate_max_call(model, 8, np.random.default_rng(streams[3])).states.tolist()
        earlier = simulate_max_call(model, 8, np.random.default_rng(streams[2])).states.tolist()
        self.assertEqual(states, expected)
        self.assertNotEqual(states, earlier)
        self.assertEqual(states[0][0], model.spots if isinstance(model.spots,list) else list(model.spots))
        with self.assertRaises(ValueError):
            component.fresh_outer(old, count=1)

    def test_setup_charges_exclude_old_outer_and_unused_fits(self):
        old, poly = old_record(), polynomial()
        outer = dict(cpu_seconds=.007, wall_seconds=.008)
        charged = component.setup_charges(old, poly, outer)
        self.assertAlmostEqual(charged['adaptive'], old['preparation_cost']['cpu_seconds']-.0005+.007)
        self.assertAlmostEqual(charged['hybrid_exact_terminal'], .001+.004+.005+.007)
        self.assertEqual(charged['cold_uniform'], charged['adaptive'])
        bad = deepcopy(poly); bad['degrees']['3']['hybrid_setup_pricing_cpu_seconds'] += .002
        with self.assertRaises(ValueError):
            component.setup_charges(old, bad, outer)

    def test_polynomial_view_preserves_fits_without_previous_path_summaries(self):
        old, poly = old_record(), polynomial()
        original = deepcopy(poly)
        old['states'] = old['states'][:8]
        view = component.adapted_polynomial(poly, old, dict(hybrid_exact_terminal=.017), 'a'*64)
        self.assertEqual(poly, original)
        self.assertEqual(view['sample_count'],8)
        self.assertEqual(view['original_outer_sample_count'],256)
        self.assertEqual(view['degrees']['3']['coefficients'], poly['degrees']['3']['coefficients'])
        self.assertEqual(view['frozen_degree_three_polynomial_sha256'], component.frozen_polynomial_identity(poly))
        self.assertNotIn('gap', view['degrees']['3'])
        changed = deepcopy(poly); changed['degrees']['3']['coefficients'][1][0] = 1.
        self.assertNotEqual(component.frozen_polynomial_identity(changed), component.frozen_polynomial_identity(poly))

    def test_paired_difference_subtracts_intervals_and_handles_negative_bounds(self):
        record = old_record(); record['states'] = record['states'][:8]; record['stopping_indices']=[9]*8
        arms = {name:fake_arm(record, {}, name) for name in component.ARMS}
        paired = component.paired_hybrid_difference(arms)
        h = component.decoded(arms['hybrid_exact_terminal']['observations'][0]['signed_gap_certificate'])
        a = component.decoded(arms['adaptive']['observations'][0]['signed_gap_certificate'])
        difference = component.decoded(paired['observations'][0]['difference_certificate'], nonnegative=False)
        self.assertEqual(difference, (h[0]-a[1], h[1]-a[0]))
        negative = component.interval_record(F(-2), F(-1))
        self.assertEqual(component.decoded(negative, nonnegative=False), (F(-2), F(-1)))
        with self.assertRaises(ValueError): component.decoded(negative)

    def test_real_certified_arms_at_a_two_date_boundary(self):
        import certify_max_hinge
        if certify_max_hinge.ctx is None:
            self.skipTest('Run the real enclosure boundary in the separate Flint environment')
        old=old_record()
        old['model']['exercise_times']=[0.,1./3.]
        old['coefficients']=old['coefficients'][:2]
        old['frozen_policy']=[[],[]]
        old['states']=[[[100.]*5,[105.]*5],[[100.]*5,[95.]*5]]
        old['stopping_indices']=[1,1]
        old['design']['outer_paths']=2
        poly=polynomial(); poly['degrees']['3']['coefficients']=poly['degrees']['3']['coefficients'][:2]
        view=component.adapted_polynomial(poly,old,dict(hybrid_exact_terminal=.017),'a'*64)
        for name in component.ARMS:
            row=component.arm_result(old,view,name)
            self.assertTrue(row['all_targets_met'])
            self.assertEqual(len(row['observations']),2)
            self.assertGreaterEqual(row['assessment_cpu_seconds'],row['input_and_policy_validation_cpu_seconds'])
            for observation in row['observations']:
                lo,hi=component.decoded(observation['signed_gap_certificate'])
                self.assertLessEqual(hi-lo,F(1,100))
                self.assertLessEqual(lo,0)
                self.assertGreaterEqual(hi,0)
            if name=='hybrid_exact_terminal':
                self.assertGreaterEqual(row['conditional_integrals'],2)
                self.assertNotIn('standalone_pricing_cpu_seconds',row)

    def test_original_incomplete_guard_never_enters_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'data/private/original'
            source.mkdir(parents=True)
            with patch.object(component, 'ROOT', Path(temporary)), patch.object(component.original, 'registration') as entered:
                with self.assertRaises(ValueError): component.verify_original(source)
                entered.assert_not_called()


class FrozenStudyFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'data/private/original'
        self.directory = self.root/'data/private/extension'
        self.records = [old_record(seed) for seed in component.original.SEEDS]
        self.polynomials = [polynomial() for _ in self.records]
        for p, seed in zip(self.polynomials, component.original.SEEDS): p['root_seed']=seed
        self.sources = {'frozen_kernel': 'a'*64}
        self.prior = dict(source_directory=str(self.source), original_source_sha256={'old':'b'*64},
            original_design_sha256='c'*64,
            files={f'fit_{i:02d}/{name}':str(i+1)*64 for i in range(8) for name in ('inputs.json','polynomial.json')})
        self.patchers = [patch.object(component,'ROOT',self.root),
            patch.object(component,'OUTER_PATHS',8),
            patch.object(component,'source_hashes',return_value=self.sources),
            patch.object(component,'verify_original',return_value=(self.prior,self.records,self.polynomials)),
            patch.object(component,'fresh_outer',side_effect=lambda old: (
                deepcopy(old['states'][:8]), old['stopping_indices'][:8], [3])),
            patch.object(component,'arm_result',side_effect=fake_arm), patch('builtins.print')]
        self.mocks = [p.start() for p in self.patchers]
        for p in self.patchers: self.addCleanup(p.stop)

    def prepared(self):
        return component.preparation(self.source,self.directory)

    def assessed(self):
        self.prepared()
        return component.assessment(self.source,self.directory)

    def test_design_written_before_every_new_stream_and_no_original_refit(self):
        def inspect(old):
            design=json.loads((self.directory/'design.json').read_bytes())
            self.assertTrue(design['recorded_before_new_outer_sampling'])
            self.assertFalse(design['training_or_value_streams_rerun'])
            self.assertEqual(design['source_study'],self.prior)
            return deepcopy(old['states'][:8]),old['stopping_indices'][:8],[3]
        self.mocks[4].side_effect=inspect
        self.prepared()
        self.assertEqual(self.mocks[4].call_count,8)
        with self.assertRaises(ValueError): self.prepared()
        self.assertEqual(self.mocks[4].call_count,8)

    def test_completed_fit_resume_does_not_rerun_any_arm(self):
        first=self.assessed()
        self.assertEqual(self.mocks[5].call_count,24)
        self.mocks[5].side_effect=AssertionError('No completed fit may rerun')
        second=component.assessment(self.source,self.directory)
        self.assertEqual(first,second)
        self.assertEqual(self.mocks[5].call_count,24)
        self.assertEqual(first['fitting_units'],8)
        self.assertEqual(first['total_new_outer_paths'],64)

    def test_conditional_paired_standard_error_uses_fresh_streams_and_not_fit_variation(self):
        summary=self.assessed()
        self.assertAlmostEqual(summary['conditional_mean_paired_difference_standard_error'],0.)
        self.assertAlmostEqual(summary['exact_across_fit_paired_difference']['approximate_midpoint_mean_standard_error'],0.)
        # Identical per-path increments cancel here; separate arm variances
        # are positive and would give an erroneous positive paired variance.
        self.assertGreater(summary['cases'][0]['arms']['adaptive']['approximate_midpoint_mean_standard_error'],0.)

    def test_preserved_inputs_and_polynomial_coefficients_are_hash_bound(self):
        self.prepared()
        folder=self.directory/'fit_00'
        for filename in ('inputs.json','polynomial_view.json'):
            path=folder/filename
            original=path.read_bytes()
            value=json.loads(original)
            if filename=='inputs.json': value['coefficients'][1][0]=1.
            else: value['degrees']['3']['coefficients'][1][0]=1.
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): component.registration(self.source,self.directory)
            path.write_bytes(original)

    def test_wrong_spawn_key_is_rejected_even_when_input_hash_is_refreshed(self):
        self.prepared()
        path=self.directory/'fit_00/inputs.json'
        value=json.loads(path.read_bytes()); value['design']['new_outer_spawn_key']=[2]
        path.write_text(json.dumps(value))
        prepared=self.directory/'prepared.json'
        manifest=json.loads(prepared.read_bytes())
        manifest['fits'][0]['input_sha256']=component.digest(path.read_bytes())
        prepared.write_text(json.dumps(manifest))
        completion=self.directory/'preparation_completion.json'
        done=json.loads(completion.read_bytes()); done['prepared_sha256']=component.digest(prepared.read_bytes())
        completion.write_text(json.dumps(done))
        with self.assertRaises(ValueError): component.registration(self.source,self.directory)

    def test_failed_assessment_is_retained_and_not_restarted(self):
        self.prepared()
        self.mocks[5].side_effect=ArithmeticError('A difficult path is unresolved')
        with self.assertRaises(ArithmeticError): component.assessment(self.source,self.directory)
        failure=json.loads((self.directory/'fit_00/assessment_failure.json').read_bytes())
        self.assertEqual(failure['failed_phase'],'cold_uniform')
        self.assertEqual(failure['completed_arms'],[])
        self.mocks[5].side_effect=fake_arm
        with self.assertRaises(ValueError): component.assessment(self.source,self.directory)
        self.assertEqual(self.mocks[5].call_count,1)

    def test_successful_earlier_arm_is_preserved_if_later_arm_fails(self):
        self.prepared()
        def fail_second(record,view,name):
            if name=='adaptive': raise ArithmeticError('The second arm failed')
            return fake_arm(record,view,name)
        self.mocks[5].side_effect=fail_second
        with self.assertRaises(ArithmeticError): component.assessment(self.source,self.directory)
        failure=json.loads((self.directory/'fit_00/assessment_failure.json').read_bytes())
        self.assertEqual(failure['completed_arms'],['cold_uniform'])
        retained=failure['completed_arm_results']['cold_uniform']
        self.assertEqual(len(retained['observations']),8)
        self.assertEqual(retained['conditional_integrals'],16)
        self.assertGreater(retained['assessment_cpu_seconds'],0.)

    def test_policy_degree_and_complete_phase_set_are_required(self):
        manifest=self.prepared()
        path=self.directory/'fit_00/inputs.json'
        record=json.loads(path.read_bytes())
        design_sha=component.digest((self.directory/'design.json').read_bytes())
        fit=component.fit_designs()[0]
        for kind in ('degree','phase'):
            altered=deepcopy(record)
            if kind=='degree': altered['design']['policy_degree']=3
            else: altered['preparation_phases'].pop('preparation_bookkeeping')
            with self.assertRaises(ValueError): component.validate_input(altered,self.records[0],fit,
                design_sha,self.sources,self.prior['files']['fit_00/inputs.json'])

    def test_stream_verification_completion_is_hash_bound(self):
        self.prepared()
        first=component.verify_streams(self.source,self.directory)
        self.assertEqual(component.verify_streams(self.source,self.directory),first)
        completion=self.directory/'stream_verification_completion.json'
        record=json.loads(completion.read_bytes()); record['verification_sha256']='f'*64
        completion.write_text(json.dumps(record))
        with self.assertRaises(ValueError): component.verify_streams(self.source,self.directory)

    def test_aggregate_or_completion_tampering_cannot_resume(self):
        self.assessed()
        result_path=self.directory/'fit_00/result.json'
        result=json.loads(result_path.read_bytes())
        result['arms']['adaptive']['conditional_integrals'] += 1
        result_path.write_text(json.dumps(result))
        done_path=self.directory/'fit_00/assessment_completion.json'
        done=json.loads(done_path.read_bytes()); done['result_sha256']=component.digest(result_path.read_bytes())
        done_path.write_text(json.dumps(done))
        with self.assertRaises(ValueError): component.assessment(self.source,self.directory)

    def test_mismatched_hinge_certificates_fail_without_requiring_hybrid_overlap(self):
        self.assessed()
        path=self.directory/'fit_00/result.json'
        result=json.loads(path.read_bytes())
        record=json.loads((self.directory/'fit_00/inputs.json').read_bytes())
        manifest=json.loads((self.directory/'prepared.json').read_bytes())
        fit=component.fit_designs()[0]
        design_sha=component.digest((self.directory/'design.json').read_bytes())
        context=component.fit_context(fit,manifest['fits'][0],design_sha,self.sources)
        self.assertTrue(component.validate_result(result,record,manifest['fits'][0],context))
        result['paired_hybrid_minus_adaptive']['approximate_midpoint_mean'] += .1
        with self.assertRaises(ValueError): component.validate_result(result,record,manifest['fits'][0],context)


if __name__=='__main__': unittest.main()
