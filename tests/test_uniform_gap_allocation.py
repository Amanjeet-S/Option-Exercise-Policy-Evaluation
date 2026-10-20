"""Common-tolerance allocation, trace reconstruction and immutable-run checks."""

from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS)); sys.path.insert(0,str(SCRIPTS.parent/'src'))
SPEC=importlib.util.spec_from_file_location('uniform_allocation',SCRIPTS/'study_uniform_gap_allocation.py')
component=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(component)


def record_fixture(seed=2026100410):
    model=component.extension.original.expected_benchmark_model()
    return dict(model=model,knots=[80.,100.,120.,140.],coefficients=[[0.]*10 for _ in range(10)],
        frozen_policy=[[] for _ in range(10)],states=[[[100.]*5 for _ in range(10)] for _ in range(2)],
        stopping_indices=[9,9],design=dict(root_seed=seed,training_paths=50000,value_paths=50000,
            outer_paths=2,spawn_keys=[[0],[1],[3]],policy_degree=2,
            evaluation_independent_of_both_fitting_streams=True),
        preparation_cost=dict(cpu_seconds=.1,wall_seconds=.1))


def fake_uniform(record,arm):
    theta=component.INITIAL_WIDTHS[arm]
    bracket=(-F(3,10000),F(3,10000))
    initial=[]
    for j in range(1,10):
        initial.append(dict(increment=j,requested_width_certificate=component.interval_record(theta,theta),
            returned_certificate=component.interval_record(*bracket),
            work=dict(integrals=1,integrand_evaluations=2),complete_query_cost=dict(cpu_seconds=.0001,wall_seconds=.0001)))
    rewards=[(F(0),F(0))]*10
    penalty=centres=(F(0),)*10
    brackets=((F(0),F(0)),)+(bracket,)*9
    certificate=component.signed_gap(rewards,penalty,component.drift_intervals(brackets,centres),9)
    observations=[]
    for index in range(2):
        observations.append(dict(path_index=index,stopping_index=9,
            signed_gap_certificate=component.interval_record(*certificate[:2]),
            absolute_gap_certificate=component.interval_record(*certificate[2:]),signed_width=float(certificate[1]-certificate[0]),
            reward_certificates=[component.interval_record(*reward) for reward in rewards],
            frozen_centres_and_penalty_sha256=component.frozen_hash(centres,penalty),
            initial_common_width_certificate=component.interval_record(theta,theta),
            reward_difference_width_allowance_certificate=component.interval_record(F(0),F(0)),
            maximum_stopping_span_length=9,initial_queries=deepcopy(initial),global_rounds=[],
            coarse_work=dict(conditional_integrals=9,integrand_evaluations=18),
            refinement_work=dict(conditional_integrals=0,integrand_evaluations=0),
            total_work=dict(conditional_integrals=9,integrand_evaluations=18),
            complete_path_cost=dict(cpu_seconds=.003,wall_seconds=.003)))
    aggregate=component.extension.aggregate_intervals([certificate[:2]]*2)
    return dict(arm=arm,initial_common_width=str(theta),outer_paths=2,observations=observations,all_targets_met=True,all_signed_inside_absolute=True,
        mean_signed_gap_certificate=aggregate['mean_certificate'],mean_signed_width=float(certificate[1]-certificate[0]),
        maximum_signed_width=aggregate['maximum_observation_width'],
        approximate_midpoint_mean=aggregate['approximate_midpoint_mean'],
        approximate_midpoint_mean_standard_error=aggregate['approximate_midpoint_mean_standard_error'],
        midpoint_mean_numerical_error_bound=aggregate['midpoint_mean_numerical_error_bound'],
        conditional_integrals=18,integrand_evaluations=36,initialisation_integrals=18,global_refinement_integrals=0,
        global_refinement_rounds=0,input_and_policy_validation_cpu_seconds=.001,assessment_cpu_seconds=.02,
        assessment_wall_seconds=.02,certificate_phase_cpu_seconds=.019)


class PureCommonChecks(unittest.TestCase):
    def setup_problem(self):
        return ([(F(100),F(100)),(F(0),F(0)),(F(0),F(0)),(F(0),F(0))],
            (F(0),)*4,(F(0),)*4,
            ((F(0),F(0)),(-F(1,2),F(1,2)),(-F(1,200),F(1,200)),(-F(1,4),F(1,4))),3)

    def test_common_rounds_refine_every_noncompliant_date_at_one_width(self):
        calls=[]; args=self.setup_problem(); before=deepcopy(args)
        def oracle(j,width):
            calls.append((j,width))
            return -width/4,width/4,dict(integrals=2,integrand_evaluations=11)
        brackets,certificate,history=component.uniform_refine(*args,oracle,initial_width=F(1))
        self.assertEqual(args,before)
        self.assertLessEqual(certificate[1]-certificate[0],F(1,100))
        self.assertEqual(history[0]['eligible_increments'],[1,3])
        self.assertEqual(history[-1]['eligible_increments'],[1,2,3])
        for step in history:
            widths={component.extension.decoded(q['requested_width_certificate'])[0] for q in step['queries']}
            self.assertEqual(len(widths),1)
            self.assertEqual([q['increment'] for q in step['queries']],step['eligible_increments'])
        self.assertEqual(sum(q['work']['integrals'] for step in history for q in step['queries']),2*len(calls))

    def test_actual_gap_success_despite_large_reward_bound_and_loose_absolute_width(self):
        calls=[]
        result=component.uniform_refine([(-F(100),-F(99)),(F(0),F(0))],(F(0),)*2,(F(0),)*2,
            ((F(0),F(0)),(-F(1,2),F(1,2))),1,lambda *a:calls.append(a),initial_width=F(1))
        self.assertEqual(result[1][:2],(0,0)); self.assertEqual(result[2],[]); self.assertEqual(calls,[])
        self.assertGreater(result[1][3]-result[1][2],F(1,100))

    def test_complete_empty_round_is_allowed_without_date_ranking(self):
        calls=[]
        args=([(F(100),F(100)),(F(0),F(0)),(F(0),F(0))],(F(0),)*3,(F(0),)*3,
              ((F(0),F(0)),(-F(1,20),F(1,20)),(-F(1,20),F(1,20))),2)
        def oracle(j,width):
            calls.append((j,width)); return -width/10,width/10,dict(integrals=1,integrand_evaluations=3)
        _,_,history=component.uniform_refine(*args,oracle,initial_width=F(1))
        self.assertEqual(history[0]['eligible_increments'],[])
        self.assertTrue(history[0]['completed']); self.assertGreater(len(calls),0)

    def test_round_limit_retains_partial_rounds_and_actual_gap(self):
        with self.assertRaises(component.UniformRefinementError) as caught:
            component.uniform_refine(*self.setup_problem(),lambda *_:None,initial_width=F(1),maximum_rounds=0)
        self.assertEqual(caught.exception.details['rounds'],[])
        self.assertGreater(component.extension.decoded(caught.exception.details['current_gap_certificate'])[1],0)

    def test_oracle_failure_retains_known_completed_query_and_unknown_failed_work(self):
        def oracle(j,width):
            if j==3: raise ArithmeticError('Unresolved integral')
            return -width/4,width/4,dict(integrals=3,integrand_evaluations=19)
        with self.assertRaises(component.UniformRefinementError) as caught:
            component.uniform_refine(*self.setup_problem(),oracle,initial_width=F(1))
        details=caught.exception.details; row=details['rounds'][0]
        self.assertFalse(details['internal_work_of_failed_oracle_known'])
        self.assertEqual(row['queries'][0]['work']['integrals'],3)
        self.assertEqual(row['failed_increment'],3); self.assertIsNone(row['failed_query_returned_work'])

    def test_invalid_returned_enclosure_preserves_returned_work_as_known(self):
        def oracle(j,width): return -F(1),F(1),dict(integrals=5,integrand_evaluations=23)
        with self.assertRaises(component.UniformRefinementError) as caught:
            component.uniform_refine(*self.setup_problem(),oracle,initial_width=F(1))
        details=caught.exception.details
        self.assertTrue(details['internal_work_of_failed_oracle_known'])
        self.assertEqual(details['rounds'][0]['failed_query_returned_work']['integrals'],5)

    def test_oversized_returned_bracket_cannot_pass_by_narrow_intersection(self):
        args=([(F(100),F(100)),(F(0),F(0))],(F(0),)*2,(F(0),)*2,
              ((F(0),F(0)),(F(0),F(3,10))),1)
        with self.assertRaises(component.UniformRefinementError) as caught:
            component.uniform_refine(*args,lambda j,width:(-F(1,2),F(1,10),dict(integrals=1,integrand_evaluations=2)),initial_width=F(1))
        self.assertTrue(caught.exception.details['internal_work_of_failed_oracle_known'])

    def test_inverted_returned_bracket_preserves_original_failure_and_work(self):
        with self.assertRaises(component.UniformRefinementError) as caught:
            component.uniform_refine(*self.setup_problem(),
                lambda j,width:(F(1),-F(1),dict(integrals=4,integrand_evaluations=9)),initial_width=F(1))
        detail=caught.exception.details['rounds'][0]
        self.assertEqual(detail['failed_query_returned_work']['integrals'],4)
        self.assertTrue(detail['failed_query_returned_certificate']['invalid_endpoint_order'])

    def test_fixed_centres_survive_all_common_rounds(self):
        args=self.setup_problem(); fingerprint=component.frozen_hash(args[2],args[1])
        component.uniform_refine(*args,lambda j,width:(-width/4,width/4,dict(integrals=1,integrand_evaluations=2)),initial_width=F(1))
        self.assertEqual(component.frozen_hash(args[2],args[1]),fingerprint)

    def test_only_enumerated_timer_fields_are_removed(self):
        scientific=dict(assessment_cpu_seconds=1.,observations=[dict(reward_cpu_seconds=.2,
            signed_width=.003,conditional_integrals=7)],timing_note='keep',elapsed_private_counter=9)
        value=component.without_timers(scientific)
        self.assertEqual(value,dict(observations=[dict(signed_width=.003,conditional_integrals=7)],
            timing_note='keep',elapsed_private_counter=9))

    def test_near_balanced_three_arm_orders(self):
        fits=component.fit_designs()
        for arm in component.ARMS:
            counts=[sum(f['arm_order'][j]==arm for f in fits) for j in range(3)]
            self.assertEqual(sorted(counts),[2,3,3])

    def test_reward_selected_date_cancellation_at_real_small_boundary(self):
        if component.ctx is None: self.skipTest('Run in the separate Flint environment')
        record=record_fixture(); record['model']['exercise_times']=[0.,1.]
        record['states']=[[[100.]*5,[110.]*5],[[100.]*5,[105.]*5]]
        record['stopping_indices']=[1,1]; record['coefficients']=[[0.]*10,[0.,0.,0.,100.,*[0.]*6]]
        record['frozen_policy']=[[],[]]
        rows={name:component.uniform_assess(record,name) for name in component.INITIAL_WIDTHS}
        for name,row in rows.items():
            self.assertTrue(row['all_targets_met']); self.assertGreater(row['conditional_integrals'],0)
            for observation in row['observations']:
                component.validate_trace(observation,record,name)
                self.assertLessEqual(component.extension.decoded(observation['signed_gap_certificate'])[1],F(1,100))
        for one,two in zip(rows['cold_common']['observations'],rows['coarse_common']['observations']):
            component.intersect(component.extension.decoded(one['signed_gap_certificate']),
                                component.extension.decoded(two['signed_gap_certificate']))


class AllocationFixture(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name); self.directory=self.root/'data/private/allocation'
        self.source=self.root/'data/private/original'; self.prior_extension=self.root/'data/private/extension'
        self.records=[record_fixture(seed) for seed in component.extension.original.SEEDS]
        self.retained=[]
        for record in self.records:
            adaptive=fake_uniform(record,'coarse_common'); adaptive['arm']='adaptive'
            oldcold=deepcopy(adaptive)
            self.retained.append(dict(arms=dict(adaptive=adaptive,cold_uniform=oldcold)))
        self.prior=dict(frozen_extension_files={f'fit_{i:02d}/{name}':str(i+1)*64
            for i in range(8) for name in ('inputs.json','result.json')})
        self.sources={'registered_kernel':'a'*64}
        self.patchers=[patch.object(component,'ROOT',self.root),patch.object(component.extension,'ROOT',self.root),
            patch.object(component,'source_hashes',return_value=self.sources),
            patch.object(component,'verify_sources',return_value=(self.prior,self.records,[{}]*8,self.retained)),
            patch.object(component,'uniform_assess',side_effect=fake_uniform),
            patch.object(component.extension,'arm_result',side_effect=lambda record,view,name:
                deepcopy(self.retained[component.extension.original.SEEDS.index(record['design']['root_seed'])]['arms']['adaptive'])),
            patch.object(component,'check_storage'),patch('builtins.print')]
        self.mocks=[p.start() for p in self.patchers]
        for p in self.patchers:self.addCleanup(p.stop)

    def registered(self): return component.register(self.source,self.prior_extension,self.directory)
    def assessed(self):
        self.registered(); return component.assessment(self.source,self.prior_extension,self.directory)

    def test_registration_precedes_every_assessment_and_contains_sources(self):
        design=self.registered()
        self.assertEqual(self.mocks[4].call_count,0); self.assertEqual(self.mocks[5].call_count,0)
        self.assertTrue(design['recorded_before_new_assessment'])
        self.assertFalse(design['new_simulation_or_fitting'])
        self.assertEqual(design['source_sha256'],self.sources)
        with self.assertRaises(ValueError):self.registered()

    def test_exact_replay_and_completed_resume_do_not_launch_new_arms(self):
        first=self.assessed()
        self.assertEqual(self.mocks[4].call_count,16); self.assertEqual(self.mocks[5].call_count,8)
        self.mocks[4].side_effect=AssertionError('Do not rerun'); self.mocks[5].side_effect=AssertionError('Do not rerun')
        second=component.assessment(self.source,self.prior_extension,self.directory)
        self.assertEqual(first,second)
        self.assertTrue(second['adaptive_exact_numerical_replay'])

    def test_fresh_budget_charges_only_assessment_and_standalone_adds_setup_once(self):
        summary=self.assessed()
        case=summary['cases'][0]
        for name in component.ARMS:
            self.assertEqual(case['assessment_only_cpu_seconds_by_arm'][name],.02)
            self.assertAlmostEqual(case['standalone_pricing_cpu_seconds_by_arm'][name],.12)

    def test_adaptive_numerical_change_is_rejected_but_timer_change_is_allowed(self):
        self.registered()
        self.mocks[5].side_effect=lambda record,view,name:dict(
            deepcopy(self.retained[0]['arms']['adaptive']),assessment_cpu_seconds=.04,
            assessment_wall_seconds=.04,certificate_phase_cpu_seconds=.039)
        # Identical fixture science across all fits, so changed observed CPU is legitimate.
        summary=component.assessment(self.source,self.prior_extension,self.directory)
        self.assertTrue(summary['adaptive_exact_numerical_replay'])
        result_path=self.directory/'fit_00/result.json'; result=json.loads(result_path.read_bytes())
        result['arms']['adaptive']['observations'][0]['total_work']['conditional_integrals']+=1
        result_path.write_text(json.dumps(result))
        completion_path=self.directory/'fit_00/assessment_completion.json'
        completion=json.loads(completion_path.read_bytes()); completion['result_sha256']=component.digest(result_path.read_bytes())
        completion_path.write_text(json.dumps(completion))
        with self.assertRaises(ValueError):component.assessment(self.source,self.prior_extension,self.directory)

    def test_failure_preserves_completed_earlier_arm_and_cannot_restart(self):
        self.registered()
        def fail_second(record,arm):
            if arm=='coarse_common':raise component.UniformAssessmentError('Unresolved query',dict(completed_path_observations=[]))
            return fake_uniform(record,arm)
        self.mocks[4].side_effect=fail_second
        with self.assertRaises(component.UniformAssessmentError):component.assessment(self.source,self.prior_extension,self.directory)
        failure=json.loads((self.directory/'fit_00/assessment_failure.json').read_bytes())
        self.assertEqual(list(failure['completed_arm_results']),['cold_common'])
        self.assertEqual(len(failure['completed_arm_results']['cold_common']['observations']),2)
        self.mocks[4].side_effect=fake_uniform
        with self.assertRaises(ValueError):component.assessment(self.source,self.prior_extension,self.directory)

    def test_trace_rejects_wrong_initial_width_and_unnecessary_round(self):
        record=self.records[0]; observation=fake_uniform(record,'cold_common')['observations'][0]
        component.validate_trace(observation,record,'cold_common')
        wrong=deepcopy(observation); wrong['initial_common_width_certificate']=component.interval_record(F(1),F(1))
        with self.assertRaises(ValueError):component.validate_trace(wrong,record,'cold_common')
        wrong=deepcopy(observation); wrong['global_rounds']=[{}]
        with self.assertRaises(ValueError):component.validate_trace(wrong,record,'cold_common')
        wrong=deepcopy(observation); wrong['global_rounds']=[{}]*17
        with self.assertRaises(ValueError):component.validate_trace(wrong,record,'cold_common')

    def test_trace_rejects_reward_or_frozen_hash_tampering(self):
        record=self.records[0]; observation=fake_uniform(record,'coarse_common')['observations'][0]
        for kind in ('reward','fingerprint','final'):
            altered=deepcopy(observation)
            if kind=='reward': altered['reward_certificates'][1]=component.interval_record(F(1),F(1))
            elif kind=='fingerprint':altered['frozen_centres_and_penalty_sha256']='f'*64
            else:altered['signed_gap_certificate']=component.interval_record(F(0),F(1,1000))
            with self.assertRaises(ValueError):component.validate_trace(altered,record,'coarse_common')

    def test_summary_hash_and_sources_are_checked_on_resume(self):
        self.assessed(); path=self.directory/'summary.json'
        value=json.loads(path.read_bytes()); value['retained_fits']=7
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):component.assessment(self.source,self.prior_extension,self.directory)

    def test_missing_original_completion_does_not_execute_prior_kernels(self):
        source=self.root/'data/private/missing_original'; extension=self.root/'data/private/missing_extension'
        extension.mkdir(parents=True)
        with patch.object(component.extension,'registration') as entered:
            # Bypass this fixture's verify_sources mock for the guarded real implementation.
            self.patchers[3].stop()
            with self.assertRaises(ValueError):component.verify_sources(source,extension)
            entered.assert_not_called()


if __name__=='__main__':unittest.main()
