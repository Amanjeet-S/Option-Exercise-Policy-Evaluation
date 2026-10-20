"""Driver checks using arithmetic fixtures, never research observations.

No sampling, fitting or numerical expectation evaluation is executed here.
"""
import gzip
import importlib.util
import json
import hashlib
import os
from pathlib import Path
from fractions import Fraction
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

DRIVER_PATH=Path(os.environ.get('QF_GAIN_ALLOCATION_DRIVER_PATH',str(Path(__file__).resolve().parents[1]/'scripts/study_european_gain_allocation.py')))
spec=importlib.util.spec_from_file_location('driver',DRIVER_PATH)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def cert(a,b=None):
    a=Fraction(a);b=a if b is None else Fraction(b)
    return driver.pair_record((a,b))


def native(arm='coarse_common',target=driver.TARGET,initial=driver.INITIAL_WIDTH,precision=96):
    exact_states=[[[str(Fraction(x).numerator),str(Fraction(x).denominator)] for x in state] for state in [[100.],[100.]]]
    return dict(schema_version=1,arm=arm,stopping_index=1,rights=2,dimension=1,
        state_sha256=hashlib.sha256(json.dumps(exact_states,separators=(',',':')).encode()).hexdigest(),
        completed=True,error=None,gain_certificate=cert(0),reward_certificates=[cert(0),cert(0)],
        value_certificates=[None,cert(0)],F_certificates=[None,cert(0)],initial_queries=[],refinement_rounds=[],
        excluded_dates=[],initially_queried_nonterminal_dates=[],refinement_counts_by_date={},requested_quarter_depths_by_date={},
        maximum_refinements_per_date=16,additional_query_budget=0,maximum_refinement_decisions=0 if arm=='gain_adaptive' else 16,
        additional_queries=0,request_width_floor_certificate=cert(initial/4**16),
        target_width_certificate=cert(target),initial_value_width_certificate=cert(initial),precision_bits=precision,
        maximum_oracle_attempts_per_requested_value=8,shared_structural_screening_and_dominance_pruning=True,
        policy_improvement_claim=False,paths_generated=False,policy_fitting_executed=False,
        total_work={key:0 for key in driver.WORK_KEYS},complete_path_cost=dict(cpu_seconds=0.,wall_seconds=0.))


def query_fixture(date,width,previous,phase):
    returned=(Fraction(1)-width,Fraction(1))
    work={key:0 for key in driver.WORK_KEYS};work['analytic_evaluations']=1;work['oracle_attempts']=1
    attempt=dict(attempt_index=1,returned_finite_certificate=True,requested_width_attained=True,
        value_certificate=driver.pair_record(returned),work=work,complete_attempt_cost=dict(cpu_seconds=0.,wall_seconds=0.))
    return dict(date=date,phase=phase,requested_width_certificate=cert(width),
        previous_value_certificate=None if previous is None else driver.pair_record(previous),
        value_certificate=driver.pair_record(returned),retained_value_certificate=driver.pair_record(returned),
        attempts=[attempt],work=work,work_complete=True,complete_query_cost=dict(cpu_seconds=0.,wall_seconds=0.))


def long_trace(arm):
    """Eighteen valid adaptive decisions, or nine two-query common rounds."""
    target=Fraction(1,4**9);row=native(arm,target=target)
    row.update(stopping_index=3,rights=4,reward_certificates=[cert(0),cert(1),cert(1),cert(0)],
        initially_queried_nonterminal_dates=[1,2],maximum_refinement_decisions=32 if arm=='gain_adaptive' else 16,
        additional_query_budget=32)
    values={1:(Fraction(0),Fraction(1)),2:(Fraction(0),Fraction(1))};counts={1:0,2:0};depths={1:0,2:0}
    row['initial_queries']=[query_fixture(j,Fraction(1),None,'initial') for j in (1,2)]
    rounds=[]
    for decision in range(18 if arm=='gain_adaptive' else 9):
        before=(Fraction(0),max(b-a for a,b in values.values()))
        selected=[1+decision%2] if arm=='gain_adaptive' else [1,2]
        step=dict(round_index=decision+1,selected_dates=selected,dominance_certificates=[],queries=[],gain_certificate_before=driver.pair_record(before))
        for j in selected:
            depth=depths[j]+1 if arm=='gain_adaptive' else decision+1;width=Fraction(1,4**depth)
            step['queries'].append(query_fixture(j,width,values[j],'refinement'))
            values[j]=(Fraction(1)-width,Fraction(1));counts[j]+=1;depths[j]=depth
        after=(Fraction(0),max(b-a for a,b in values.values()))
        step.update(gain_certificate_after=driver.pair_record(after),attained_width_reduction=driver.rational(before[1]-after[1]))
        if arm=='gain_adaptive':step.update(requested_quarter_depth=depth,skipped_already_attained_quarter_requests=0,optimistic_scores={})
        else:step['common_requested_width_certificate']=cert(width)
        rounds.append(step)
    row.update(refinement_rounds=rounds,additional_queries=sum(counts.values()),
        refinement_counts_by_date={str(j):counts[j] for j in counts},requested_quarter_depths_by_date={str(j):depths[j] for j in depths},
        value_certificates=[None]+[driver.pair_record(values[j]) for j in (1,2)]+[cert(0)],
        F_certificates=[None]+[cert(0,values[j][1]-values[j][0]) for j in (1,2)]+[cert(0)],gain_certificate=cert(0,target))
    row['total_work']['analytic_evaluations']=2+sum(counts.values());row['total_work']['oracle_attempts']=2+sum(counts.values())
    return row,target


def arm(cpu=1.):
    observations=[dict(path_index=i,stopping_index=1,assessment=native()) for i in range(driver.PATHS_PER_FIT)]
    return driver.arm_summary(observations,dict(cpu_seconds=cpu,wall_seconds=cpu))


class DriverChecks(unittest.TestCase):
    def test_shared_structural_settings_and_order(self):
        settings=driver.design_settings()
        self.assertIn('same safe interval',settings['shared_structural_rules'])
        self.assertEqual(settings['paths_per_fit'],32)
        self.assertEqual(settings['reference']['precision_bits'],128)
        orders=[list(driver.ARMS if i%2==0 else reversed(driver.ARMS)) for i in range(8)]
        self.assertEqual(sum(x[0]=='coarse_common' for x in orders),4)

    def test_exact_gain_reconstruction(self):
        row=dict(reward_certificates=[cert(0),cert(3),cert(1),cert(4)],
                 value_certificates=[None,cert(2),cert(Fraction(1,2)),cert(4)])
        self.assertEqual(driver.reconstruct_gain(row,4,2),(Fraction(1,2),Fraction(1,2)))

    def test_terminal_identity_overrides_interval_subtraction(self):
        row=dict(reward_certificates=[cert(0),cert(0),cert(4,5)],
                 value_certificates=[None,cert(5),cert(4,5)])
        self.assertEqual(driver.reconstruct_gain(row,3,2),(0,0))

    def test_disjoint_terminal_enclosures_rejected(self):
        row=dict(reward_certificates=[cert(0),cert(0),cert(4)],
                 value_certificates=[None,cert(5),cert(5)])
        with self.assertRaises(ValueError):driver.reconstruct_gain(row,3,2)

    def test_overlapping_nonidentical_terminal_enclosures_rejected(self):
        row=dict(reward_certificates=[cert(0),cert(0),cert(4,5)],value_certificates=[None,cert(5),cert(4,6)])
        with self.assertRaises(ValueError):driver.reconstruct_gain(row,3,2)

    def test_exact_interval_width_tamper_rejected(self):
        bad=cert(0,1);bad['width']=driver.rational(Fraction(2))
        with self.assertRaises(ValueError):driver.decode_pair(bad)

    def test_failed_native_target_rejected(self):
        row=native();row['gain_certificate']=cert(0,Fraction(1,10))
        with self.assertRaises(ValueError):driver.validate_native(row,driver.TARGET)

    def test_native_false_completion_rejected(self):
        row=native();row['completed']=False
        with self.assertRaises(ValueError):driver.validate_native(row,driver.TARGET)

    def test_native_registered_settings_and_floor_rejected_if_changed(self):
        for field,value in [('precision_bits',128),('arm','gain_adaptive'),('maximum_oracle_attempts_per_requested_value',16),
            ('request_width_floor_certificate',cert(Fraction(1,4**15))),('target_width_certificate',cert(Fraction(1,10))),
            ('initial_value_width_certificate',cert(2))]:
            row=native();row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):driver.validate_native(row,driver.TARGET)

    def test_more_than_sixteen_adaptive_decisions_are_valid_with_dated_caps(self):
        row,target=long_trace('gain_adaptive')
        self.assertEqual(len(row['refinement_rounds']),18)
        self.assertEqual(driver.validate_native(row,target,arm='gain_adaptive'),(0,target))

    def test_common_native_schema_and_full_rounds(self):
        row,target=long_trace('coarse_common')
        self.assertEqual(driver.validate_native(row,target),(0,target))
        row['refinement_rounds'][0]['requested_quarter_depth']=1
        with self.assertRaises(ValueError):driver.validate_native(row,target)

    def test_changed_budget_and_per_date_counts_rejected(self):
        for field,value in [('additional_query_budget',33),('additional_queries',19),('maximum_refinement_decisions',33),
            ('refinement_counts_by_date',{'1':17,'2':9}),('requested_quarter_depths_by_date',{'1':17,'2':9})]:
            row,target=long_trace('gain_adaptive');row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):driver.validate_native(row,target,arm='gain_adaptive')

    def test_query_attempt_budget_rejected(self):
        row,target=long_trace('coarse_common');query=row['initial_queries'][0]
        query['attempts']*=9
        with self.assertRaises(ValueError):driver.validate_native(row,target)

    def test_overwide_returned_price_not_hidden_by_retained_intersection(self):
        row,target=long_trace('coarse_common');query=row['refinement_rounds'][0]['queries'][0]
        query['value_certificate']=cert(0,1)
        with self.assertRaises(ValueError):driver.validate_native(row,target)

    def test_final_price_bracket_tamper_is_rejected(self):
        row,target=long_trace('coarse_common');row['value_certificates'][1]=cert(1)
        with self.assertRaises(ValueError):driver.validate_native(row,target)

    def test_unnecessary_late_round_is_rejected(self):
        row,target=long_trace('coarse_common');extra=dict(row['refinement_rounds'][-1]);extra['round_index']=10
        row['refinement_rounds'].append(extra)
        with self.assertRaises(ValueError):driver.validate_native(row,target)

    def test_unavailable_work_stays_distinct(self):
        row=native();row['total_work']['integrand_evaluations_unavailable_attempts']=1
        row['total_work']['integrand_evaluations_known']=9
        total=driver.aggregate_work([row]);self.assertEqual(total['integrand_evaluations_unavailable_attempts'],1)
        self.assertEqual(total['integrand_evaluations_known'],9)

    def test_boolean_work_not_an_integer_count(self):
        row=native();row['total_work']['oracle_attempts']=True
        with self.assertRaises(ValueError):driver.aggregate_work([row])

    def test_retries_derive_from_all_attempt_ledgers(self):
        row=native();row['initial_queries']=[dict(attempts=[{},{}])];row['refinement_rounds']=[dict(queries=[dict(attempts=[{},{},{}])])]
        self.assertEqual(driver.retry_count(row),3)

    def test_complete_summary_rejects_missing_or_duplicate_fits(self):
        rows=[dict(fit_index=i,arms=dict(coarse_common=arm(),gain_adaptive=arm(.5)),reference=arm()) for i in range(8)]
        out=driver.summary(rows,'design',{})
        self.assertEqual(out['adaptive_over_common_ratios']['complete_assessment_cpu']['median'],.5)
        with self.assertRaises(ValueError):driver.summary(rows[:-1],'design',{})
        rows[-1]['fit_index']=6
        with self.assertRaises(ValueError):driver.summary(rows,'design',{})

    def test_zero_denominator_remains_undefined(self):
        rows=[dict(fit_index=i,arms=dict(coarse_common=arm(0.),gain_adaptive=arm(.5)),reference=arm()) for i in range(8)]
        out=driver.summary(rows,'design',{})['adaptive_over_common_ratios']['complete_assessment_cpu']
        self.assertIsNone(out['median']);self.assertEqual(out['per_fit_ratios'],[None]*8)

    def test_lossless_gzip_exclusive_writer(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json.gz';value=dict(exact='1/3',observation=[1,2,3])
            artifact=driver.write_gzip(p,value)
            with gzip.open(p,'rt') as f:self.assertEqual(json.load(f),value)
            self.assertEqual(driver.digest(p),artifact['compressed_sha256'])
            with self.assertRaises(FileExistsError):driver.write_gzip(p,value)

    def test_immutable_attempt_cannot_restart(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'assessment_attempt.json').write_text('{}')
            args=SimpleNamespace(directory=folder)
            with self.assertRaises(FileExistsError):driver.assess(args)

    def test_second_path_failure_preserves_first_and_native_partial(self):
        class NativeFailure(RuntimeError):pass
        calls=[]
        def evaluate(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                error=NativeFailure('unresolved');error.record=dict(known_work='preserved');raise error
            return native()
        module=SimpleNamespace(evaluate_gain=evaluate)
        record=dict(model=dict(strike=100.,rate=.05,dividends=[.1],volatilities=[.2],exercise_times=[0.,1.]),states=[[[100.],[100.]] for _ in range(32)],stopping_indices=[1]*32)
        completed=[]
        with self.assertRaises(NativeFailure) as caught:
            driver.evaluate_arm(module,record,'coarse_common',target=driver.TARGET,initial=driver.INITIAL_WIDTH,precision=96,completed=completed,repository=Path('/private/tmp'))
        retained=caught.exception.study_failure
        self.assertEqual(len(retained['completed_or_returned_paths']),1)
        self.assertEqual(retained['failed_path_partial_record']['known_work'],'preserved')
        self.assertEqual(retained['path_index'],1)

    def snapshot_fixture(self,folder):
        root=folder/'root';root.mkdir();extension=folder/'extension';extension.mkdir();control=folder/'control';control.mkdir()
        sources={}
        for index in range(15):
            path=root/f'source_{index}.txt';path.write_text(f'frozen {index}');sources[path.name]=driver.digest(path)
        inputs={}
        for index in range(8):
            fit=extension/f'fit_{index:02d}';fit.mkdir();(fit/'inputs.json').write_text('{}');inputs[f'fit_{index:02d}/inputs.json']=driver.digest(fit/'inputs.json')
        (extension/'design.json').write_text('{}')
        design=dict(source_sha256=sources,source_evidence=dict(frozen_extension_files=inputs,frozen_extension_design_sha256=driver.digest(extension/'design.json')))
        (control/'design.json').write_text(json.dumps(design))
        allocator=folder/'allocator.py';allocator.write_text('pass');tail=folder/'certify_european_gain_tail.py';tail.write_text('pass')
        return SimpleNamespace(root=root,extension_study=extension,control_study=control,allocator_module=allocator,tail_module=tail,additional_source=[])

    def test_fifteen_source_guard_rejects_tampering(self):
        with TemporaryDirectory() as tmp:
            args=self.snapshot_fixture(Path(tmp));driver.snapshot(args)
            (args.root/'source_14.txt').write_text('changed')
            with self.assertRaises(ValueError):driver.snapshot(args)

    def test_retained_input_guard_rejects_tampering(self):
        with TemporaryDirectory() as tmp:
            args=self.snapshot_fixture(Path(tmp));driver.snapshot(args)
            (args.extension_study/'fit_03/inputs.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):driver.snapshot(args)

    def test_actual_sibling_helper_binding_rejects_other_helper(self):
        with TemporaryDirectory() as tmp:
            args=self.snapshot_fixture(Path(tmp));driver.snapshot(args)
            other=Path(tmp)/'different_tail.py';other.write_text('pass');args.tail_module=other
            with self.assertRaises(ValueError):driver.snapshot(args)

    def test_actual_sibling_helper_tamper_changes_registered_evidence(self):
        with TemporaryDirectory() as tmp:
            args=self.snapshot_fixture(Path(tmp));first=driver.snapshot(args)
            args.tail_module.write_text('changed')
            second=driver.snapshot(args)
            self.assertNotEqual(first['new_numerical_sources_sha256'],second['new_numerical_sources_sha256'])

    def test_runtime_preserves_venv_executable_identity(self):
        context=driver.runtime_identity(require_flint=False)
        self.assertEqual(context['python_executable'],driver.sys.executable)
        self.assertTrue(context['python_version']);self.assertTrue(context['platform'])

    def test_direct_audit_detects_native_gain_tamper(self):
        record=dict(model=dict(exercise_times=[0.,1.]))
        a=arm();b=arm();reference=arm();b['observations'][0]['assessment']['gain_certificate']=cert(Fraction(1,1000))
        with self.assertRaises(ValueError):driver.consistency_audit(record,dict(coarse_common=a,gain_adaptive=b),reference)


if __name__=='__main__':unittest.main()
