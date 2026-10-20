"""Bounded fixtures for fixed-look evidence, monetary scope and failure handling."""
from copy import deepcopy
from fractions import Fraction
import gzip
import importlib.util
import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(os.environ.get('QF_TEST_REPOSITORY',Path(__file__).resolve().parents[1]))

def module(name):
    source=ROOT/'scripts'/f'{name}.py'
    if not source.exists():source=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location(name+'_fixture',source)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

study=module('study_financial_decision');helper=module('certify_financial_decision');prep=module('prepare_financial_decision')


def make_binary(folder):
    states=[[100.]*5 for _ in range(10)]
    states[4]=[170.,110.,100.,100.,100.]
    vectors=[states]*4
    raw=b''.join(struct.pack('<'+'d'*50,*[x for row in v for x in row]) for v in vectors)
    stops=bytes([4,9,4,9]);(folder/'states.bin').write_bytes(raw);(folder/'stops.bin').write_bytes(stops)
    rows=[];sh=study.hashlib.sha256();th=study.hashlib.sha256()
    for i in range(2):
        block=raw[i*800:(i+1)*800];choices=stops[i*2:(i+1)*2];sh.update(block);th.update(choices)
        rows.append(dict(path_start=2*i,paths=2,state_bytes=len(block),stopping_bytes=2,
          state_sha256=study.hashlib.sha256(block).hexdigest(),stopping_sha256=study.hashlib.sha256(choices).hexdigest(),
          cumulative_state_sha256=sh.hexdigest(),cumulative_stopping_sha256=th.hexdigest()))
    return rows


class SettingsChecks(unittest.TestCase):
    def test_family_settings_and_tolerances(self):
        d=study.settings(prep.PRODUCTION_PREPARATION)
        d.update(recorded_before_sampling=True,fits=[dict(index=i,root_seed=2026100410+i) for i in range(8)],
          source_sha256={'scripts/study_financial_decision.py':'0'*64},
          retained_input_sha256={f'fit_{i:02d}/inputs.json':'1'*64 for i in range(8)})
        self.assertTrue(study.validate_design(d,prep))
        self.assertEqual(Fraction(d['family']['delta'])*d['family']['events'],Fraction(d['family']['alpha']))
        for key,value in [('looks',[4096,8192]),('tolerance_units',['1','2']),('assessors',['zero','european_common','adaptive'])]:
            altered=deepcopy(d);altered[key]=value
            with self.assertRaises(ValueError):study.validate_design(altered,prep)
        d['recorded_before_sampling']=False
        with self.assertRaises(ValueError):study.validate_design(d,prep)

    def test_negative_boolean_or_nonfinite_cost_rejected(self):
        for value in [-1.,True,float('nan')]:
            with self.assertRaises(ValueError):study.valid_cost(dict(cpu_seconds=value,wall_seconds=0.))

    def test_private_and_existing_attempt_guards(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);private=root/'data/private';private.mkdir(parents=True)
            self.assertEqual(study.private_path(root,private/'new',fresh=True),(private/'new').resolve())
            with self.assertRaises(ValueError):study.private_path(root,root/'public')
            with self.assertRaises(ValueError):study.private_path(root,private,fresh=True)

    def test_timing_free_digest_preserves_numerical_changes(self):
        row=dict(gain={'upper':1},complete_path_cost={'cpu_seconds':1},queries=[dict(complete_query_cost=3,attempts=[dict(complete_attempt_cost=4,work=5)])])
        clone=deepcopy(row);clone['complete_path_cost']['cpu_seconds']=7;clone['queries'][0]['attempts'][0]['complete_attempt_cost']=8
        self.assertNotEqual(study.hashed(row),study.hashed(clone))
        self.assertEqual(study.hashed(study.numerical_record(row)),study.hashed(study.numerical_record(clone)))
        clone['queries'][0]['attempts'][0]['work']=9
        self.assertNotEqual(study.hashed(study.numerical_record(row)),study.hashed(study.numerical_record(clone)))


class BinaryChecks(unittest.TestCase):
    def test_exact_binary_state_and_stopping_reconstruction(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);rows=make_binary(folder);observed=list(study.binary_batches(folder,rows))
            self.assertEqual([o[0] for row,batch in observed for o in batch],list(range(4)))
            self.assertEqual([o[2] for row,batch in observed for o in batch],[4,9,4,9])
            self.assertEqual(observed[0][1][0][1][4][0],170.)

    def test_byte_tamper_and_trailing_path_rejected(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary);rows=make_binary(folder)
            raw=(folder/'states.bin').read_bytes();(folder/'states.bin').write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaises(ValueError):list(study.binary_batches(folder,rows))
            (folder/'states.bin').write_bytes(raw+b'\0')
            with self.assertRaises(ValueError):list(study.binary_batches(folder,rows))

    @unittest.skipIf(helper.flint is None, "The separate python-flint environment is required")
    def test_zero_candidate_tamper_rejected(self):
        state=[[100.]*5 for _ in range(10)];state[4][0]=170.
        native=helper.evaluate_zero_gain(state,prep.expected_model()['exercise_times'],9)
        self.assertEqual(study.validate_zero_native(native,state,9,helper),helper.decode_interval(native['gain_certificate']))
        native['stopping_index']=4
        with self.assertRaises(ValueError):study.validate_zero_native(native,state,9,helper)


@unittest.skipIf(helper.flint is None, "The separate python-flint environment is required")
class ExecutionChecks(unittest.TestCase):
    def execute_fixture(self,folder,output,rows,assessor='zero',allocator=None):
        tail={'clipping_tail_allowance':helper.interval_record(Fraction(0),Fraction(0))}
        validator=SimpleNamespace(WORK_KEYS=('conditional_integral_calls','successful_integral_certificates','analytic_evaluations','integrand_evaluations_known','integrand_evaluations_unavailable_attempts','oracle_attempts'))
        fit=dict(index=0);design={'root':str(ROOT),'_sha256':'a'*64}
        return study.assess_one(folder,output,rows,fit,design,assessor,tail,study.zero_cost(),study.zero_cost(),study.zero_cost(),helper,allocator,validator,prep.expected_model(),study.zero_cost())

    def test_nested_looks_clipped_and_raw_means_and_full_budget(self):
        with TemporaryDirectory() as temporary,patch.object(study,'LOOKS',[2,4]),patch.object(study,'CAPS',{'zero':Fraction(10),'european_common':Fraction(50)}):
            folder=Path(temporary);rows=make_binary(folder)
            for n in [2,4]:study.write(folder/f'prepared_prefix_{n}.json',{'complete_preparation_prefix_cost':study.zero_cost()})
            result=self.execute_fixture(folder,folder/'assessment',rows)
            self.assertTrue(result['completed']);self.assertEqual(result['paths'],4)
            self.assertTrue(all(r['full_budget_charged_if_unresolved'] for r in result['decisions'].values()))
            self.assertTrue(all(r['required_paths']==4 for r in result['decisions'].values()))
            self.assertGreater(helper.decode_rational(result['look_records'][0]['raw_gain_mean_certificate']['lower']),helper.decode_rational(result['look_records'][0]['clipped_gain_mean_certificate']['upper']))
            records=[json.loads(line) for line in gzip.open(folder/'assessment/observations.jsonl.gz','rt')]
            self.assertEqual(len(records),4);self.assertTrue(all('assessment' in r for r in records))
            self.assertEqual(study.digest(folder/'assessment/observations.jsonl.gz'),result['compressed_observation_sha256'])
            self.assertTrue(all(value>=0 for value in result['complete_unallocated_bookkeeping_cost'].values()))
            with self.assertRaises(ValueError):self.execute_fixture(folder,folder/'assessment',rows)

    def test_first_acceptable_look_is_retained_for_each_tolerance(self):
        with TemporaryDirectory() as temporary,patch.object(study,'LOOKS',[2,4]),patch.object(study,'TOLERANCES',(Fraction(5000),Fraction(1))):
            folder=Path(temporary);rows=make_binary(folder)
            for n in [2,4]:study.write(folder/f'prepared_prefix_{n}.json',{'complete_preparation_prefix_cost':study.zero_cost()})
            result=self.execute_fixture(folder,folder/'assessment',rows)
            self.assertTrue(result['decisions']['5000']['accepted'])
            self.assertEqual(result['decisions']['5000']['required_paths'],2)
            self.assertFalse(result['decisions']['1']['accepted'])
            self.assertEqual(result['decisions']['1']['required_paths'],4)
            self.assertEqual([r['paths'] for r in result['look_records']],[2,4])

    def test_carried_failure_preserved_without_replacement(self):
        class Failure(ArithmeticError):
            def __init__(self):super().__init__('fixture raw oracle failure');self.record={'completed':False,'total_work':{'oracle_attempts':2}}
        def fail(*args,**kwargs):raise Failure()
        with TemporaryDirectory() as temporary,patch.object(study,'LOOKS',[2,4]):
            folder=Path(temporary);rows=make_binary(folder);output=folder/'assessment'
            with self.assertRaises(Failure):self.execute_fixture(folder,output,rows,'european_common',SimpleNamespace(evaluate_gain=fail))
            retained=study.read(output/'failure.json');self.assertEqual(retained['completed_paths'],0)
            self.assertEqual(retained['failed_native_work'],{'oracle_attempts':2})
            self.assertEqual(study.read(output/'failed_path_input.json')['path_index'],0)
            self.assertEqual(study.read(output/'exception_carried_native_ledger.json')['completed'],False)
            with self.assertRaises(ValueError):self.execute_fixture(folder,output,rows)

    def test_population_tail_failure_is_retained_before_paths(self):
        class TailFailure(ArithmeticError):
            def __init__(self):super().__init__('fixture tail integration failure');self.record={'work':{'known_calls':3}}
        def fail(*args,**kwargs):raise TailFailure()
        with TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();private=root/'data/private';private.mkdir(parents=True)
            sealed=private/'sealed';sealed.mkdir();prepared=private/'prepared';prepared.mkdir();retained=private/'retained';(retained/'fit_00').mkdir(parents=True)
            design={'retained_study':str(retained),'source_sha256':{'scripts/study_financial_decision.py':study.digest(Path(study.__file__))},'certification_runtime':{}}
            study.write(sealed/'design.json',design)
            study.write(retained/'fit_00/inputs.json',{'model':prep.expected_model()})
            study.write(prepared/'completion.json',{'completed':True,'design_sha256':study.digest(sealed/'design.json'),'completed_fits':8,'complete_preparation_workflow_cost':study.zero_cost(),'fits':[{'complete_preparation_cost':study.zero_cost()} for _ in range(8)]})
            modules={'prepare_financial_decision.py':prep,'study_european_gain_allocation.py':SimpleNamespace(runtime_identity=lambda:{}),
                     'certify_financial_decision.py':SimpleNamespace(build_zero_gain_tail_bound=lambda *a,**k:{'clipping_tail_allowance':helper.interval_record(0,0)}),
                     'refine_european_gain.py':SimpleNamespace(tail=SimpleNamespace(build_european_restricted_tail_bound=fail))}
            args=SimpleNamespace(root=root,study=sealed,preparation=prepared,output=private/'assessment')
            with patch.object(study,'load_module',side_effect=lambda path,name:modules[Path(path).name]),patch.object(study,'validate_design',return_value=True),patch.object(study,'guard_snapshot',return_value={}):
                with self.assertRaises(TailFailure):study.assess(args)
            self.assertTrue((args.output/'initial_attempt.json').exists())
            result=study.read(args.output/'failure.json');self.assertEqual(result['failed_phase'],'european_population_tail')
            self.assertEqual(study.read(args.output/'tail_exception_evidence.json')['work']['known_calls'],3)
            self.assertFalse(any(args.output.glob('fit_*')))

    def test_storage_guard_records_failure_before_first_gain(self):
        with TemporaryDirectory() as temporary,patch.object(study.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            folder=Path(temporary);rows=make_binary(folder);output=folder/'assessment'
            with self.assertRaises(OSError):self.execute_fixture(folder,output,rows)
            self.assertEqual(study.read(output/'failure.json')['completed_paths'],0)


if __name__=='__main__':unittest.main()
