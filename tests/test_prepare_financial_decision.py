"""Bounded arithmetic and stream fixtures for the new private preparation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(os.environ.get('QF_TEST_REPOSITORY',Path(__file__).resolve().parents[1])).resolve()
if not (ROOT/'src/qf_research/optimal_stopping/multivariate_benchmark.py').is_file():
    ROOT=Path.cwd().resolve()
if not (ROOT/'src/qf_research/optimal_stopping/multivariate_benchmark.py').is_file():
    raise RuntimeError('Run these fixtures from the repository root or set QF_TEST_REPOSITORY')
sys.path.insert(0,str(ROOT/'src'))
DRAFT=ROOT/'scripts/prepare_financial_decision.py'
if not DRAFT.is_file():
    DRAFT=Path(__file__).with_name('prepare_financial_decision.py')
if not DRAFT.is_file():
    raise RuntimeError('The installed preparation source or explicit adjacent draft is required')
spec=importlib.util.spec_from_file_location('financial_decision_preparation_fixture',DRAFT)
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
RESEARCH_RUNTIME_AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('numpy','scipy'))


def record():
    coefficients=[0.]*21
    return dict(model=prep.expected_model(),frozen_policy=[coefficients[:] for _ in range(9)]+[[]],
        design=dict(root_seed=2026100410,policy_degree=2,training_paths=50000,value_paths=50000,
                    spawn_keys=[[0],[1],[3]]))


def design():
    return dict(schema_version=1,author='Amanjeet Singh',preparation=deepcopy(prep.PRODUCTION_PREPARATION),
        looks=prep.PRODUCTION_LOOKS[:],fits=[dict(index=i,root_seed=seed) for i,seed in enumerate(prep.ROOT_SEEDS)],
        source_sha256={'scripts/prepare_financial_decision.py':'0'*64},
        retained_input_sha256={f'fit_{i:02d}/inputs.json':'1'*64 for i in range(8)})


def fixture_streams():
    streams,initial,prior=prep.construct_streams(2026100410,list(range(1,10)))
    model,policy=prep.policy_from_record(record(),prior['0']['legacy_spacing_state_sha256'])
    return model,policy,streams,initial,prior


@unittest.skipUnless(RESEARCH_RUNTIME_AVAILABLE,'The preparation fixtures require the research NumPy and SciPy environment')
class DesignChecks(unittest.TestCase):
    def test_production_design(self):
        self.assertTrue(prep.validate_design(design()))

    def test_look_or_layout_changes_rejected(self):
        for field,value in [('batch_paths',128),('new_parent_spawn_key',[3]),('state_dtype','>f8')]:
            d=design();d['preparation'][field]=value
            with self.assertRaises(ValueError):prep.validate_design(d)
        d=design();d['looks']=[4096,8192]
        with self.assertRaises(ValueError):prep.validate_design(d)

    def test_source_traversal_and_incomplete_inputs_rejected(self):
        d=design();d['source_sha256']={'../other.py':'0'*64}
        with self.assertRaises(ValueError):prep.validate_design(d)
        d=design();del d['retained_input_sha256']['fit_07/inputs.json']
        with self.assertRaises(ValueError):prep.validate_design(d)

    def test_policy_shape_and_initial_model_guard(self):
        r=record();fit=dict(index=0,root_seed=2026100410)
        self.assertEqual(prep.validate_retained_input(r,fit),prep.model_policy_identity(r))
        r['frozen_policy'][2].pop()
        with self.assertRaises(ValueError):prep.validate_retained_input(r,fit)
        r=record();r['model']['spots'][0]=101.
        with self.assertRaises(ValueError):prep.validate_retained_input(r,fit)

    def test_identity_binds_policy_parameters(self):
        r=record();identity=prep.model_policy_identity(r)
        r['frozen_policy'][3][0]=1.
        self.assertNotEqual(prep.model_policy_identity(r),identity)


@unittest.skipUnless(RESEARCH_RUNTIME_AVAILABLE,'The preparation fixtures require the research NumPy and SciPy environment')
class StreamChecks(unittest.TestCase):
    def test_complete_distinct_keys_and_unconsumed_prior_roles(self):
        model,policy,streams,initial,prior=fixture_streams()
        self.assertEqual([initial[str(j)]['spawn_key'] for j in range(1,10)],[[4,j] for j in range(1,10)])
        self.assertEqual(len({r['initial_state_sha256'] for r in initial.values()}),9)
        self.assertTrue(all(r['sampled'] is False for r in prior.values()))
        self.assertEqual(policy.training_stream_id,prior['0']['legacy_spacing_state_sha256'])
        self.assertFalse(set(r['initial_state_sha256'] for r in initial.values())&set(r['initial_state_sha256'] for r in prior.values()))

    def test_identical_fixed_batch_replay_and_checkpoint(self):
        m,p,r,initial,_=fixture_streams();first=prep.generate_batch(m,p,r,8,stream_binding='a')
        second=prep.generate_batch(m,p,r,8,stream_binding='b')
        m2,p2,r2,_,_=fixture_streams();first2=prep.generate_batch(m2,p2,r2,8,stream_binding='a')
        second2=prep.generate_batch(m2,p2,r2,8,stream_binding='b')
        self.assertEqual(first['state_bytes'],first2['state_bytes'])
        self.assertEqual(first['stop_bytes'],first2['stop_bytes'])
        self.assertEqual(second['state_bytes'],second2['state_bytes'])
        self.assertEqual(second['stop_bytes'],second2['stop_bytes'])
        self.assertEqual(first['after_states'],second['before_states'])
        self.assertEqual(first['before_states'],{k:v['initial_state'] for k,v in initial.items()})
        self.assertNotEqual(first['state_bytes'],second['state_bytes'])

    def test_binary64_bridge_reconstructs_without_numerical_dependencies(self):
        m,p,r,_,_=fixture_streams();block=prep.generate_batch(m,p,r,4,stream_binding='a')
        self.assertTrue(prep.validate_batch(block,m,4))
        values=struct.unpack('<'+('d'*(4*10*5)),block['state_bytes'])
        self.assertEqual(len(values),200)
        for offset in range(0,len(values),50):self.assertEqual(values[offset:offset+5],(100.,)*5)
        stops=struct.unpack('4B',block['stop_bytes'])
        self.assertTrue(all(1<=j<=9 for j in stops))

    def test_malformed_bytes_nonpositive_paths_and_zero_stopping_rejected(self):
        m,p,r,_,_=fixture_streams();block=prep.generate_batch(m,p,r,4,stream_binding='a')
        changed=deepcopy(block);changed['state_bytes']=changed['state_bytes'][:-1]
        with self.assertRaises(ValueError):prep.validate_batch(changed,m,4)
        changed=deepcopy(block);changed['state_bytes']=struct.pack('<d',float('nan'))+changed['state_bytes'][8:]
        with self.assertRaises(ValueError):prep.validate_batch(changed,m,4)
        changed=deepcopy(block);changed['stop_bytes']=bytes([0])+changed['stop_bytes'][1:]
        with self.assertRaises(ValueError):prep.validate_batch(changed,m,4)

    def test_legacy_date_major_count_does_not_preserve_prefix(self):
        import numpy as np
        from qf_research.optimal_stopping.multivariate_benchmark import published_max_call,simulate_max_call
        model=published_max_call()
        small=simulate_max_call(model,4,np.random.default_rng(2026100410))
        large=simulate_max_call(model,8,np.random.default_rng(2026100410))
        self.assertTrue(np.array_equal(small.states[:,1],large.states[:4,1]))
        self.assertFalse(np.array_equal(small.states,large.states[:4]))
        state=np.random.default_rng(2026100410).bit_generator.state
        self.assertEqual(small.stream_id,prep.legacy_stream_identity(state))


@unittest.skipUnless(RESEARCH_RUNTIME_AVAILABLE,'The preparation fixtures require the research NumPy and SciPy environment')
class PrivateBridgeChecks(unittest.TestCase):
    def preparation(self):
        setting=deepcopy(prep.PRODUCTION_PREPARATION)
        setting.update(batch_paths=4,total_paths_per_fit=12,state_shape=[12,10,5])
        return setting

    def run_fixture(self,path):
        return prep.prepare_fit(record(),dict(index=0,root_seed=2026100410),path,self.preparation(),[4,12],
            design_sha256='0'*64,sources_sha256={},input_sha256='1'*64,runtime={})

    def test_complete_private_fixture_and_prefix_digests(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            completed=self.run_fixture(path)
            self.assertTrue(completed['completed'])
            self.assertEqual(completed['paths'],12)
            states=(path/'states.bin').read_bytes();stops=(path/'stops.bin').read_bytes()
            self.assertEqual(len(states),12*10*5*8);self.assertEqual(len(stops),12)
            self.assertEqual(hashlib.sha256(states).hexdigest(),completed['state_sha256'])
            rows=[json.loads(line) for line in (path/'block_metadata.jsonl').read_text().splitlines()]
            self.assertEqual(prep.verify_chain(rows,completed['chain_genesis_sha256']),completed['last_block_sha256'])
            prefix=json.loads((path/'prepared_prefix_4.json').read_text())
            self.assertEqual(hashlib.sha256(states[:4*10*5*8]).hexdigest(),prefix['state_prefix_sha256'])
            self.assertEqual(hashlib.sha256(stops[:4]).hexdigest(),prefix['stopping_prefix_sha256'])
            self.assertEqual(rows[0]['after_generator_states'],rows[1]['before_generator_states'])
            for clock in ('cpu_seconds','wall_seconds'):
                self.assertGreaterEqual(completed['complete_preparation_cost'][clock],prefix['complete_preparation_prefix_cost'][clock])
                self.assertAlmostEqual(sum(c[clock] for c in completed['exclusive_phase_costs'].values()),completed['complete_preparation_cost'][clock],places=8)

    def test_chain_tampering_rejected(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit';completed=self.run_fixture(path)
            rows=[json.loads(line) for line in (path/'block_metadata.jsonl').read_text().splitlines()]
            rows[1]['path_stop']=11
            with self.assertRaises(ValueError):prep.verify_chain(rows,completed['chain_genesis_sha256'])

    def test_existing_attempt_refused(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit';path.mkdir()
            with self.assertRaises(ValueError):self.run_fixture(path)
            self.assertEqual(list(path.iterdir()),[])

    def test_failed_second_block_is_preserved_without_replacement(self):
        original=prep.generate_batch;calls=[]
        def fail(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise ArithmeticError('fixture interrupted generation')
            return original(*args,**kwargs)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            with patch.object(prep,'generate_batch',side_effect=fail):
                with self.assertRaises(ArithmeticError):self.run_fixture(path)
            failure=json.loads((path/'preparation_failure.json').read_text())
            self.assertEqual(failure['completed_blocks'],1);self.assertEqual(failure['completed_paths'],4)
            self.assertEqual(failure['failed_block_index'],1)
            self.assertIsNotNone(failure['failed_block_start_generator_states'])
            self.assertFalse((path/'completion.json').exists())
            self.assertEqual((path/'states.bin').stat().st_size,4*10*5*8)
            self.assertEqual((path/'stops.bin').stat().st_size,4)
            self.assertEqual(len((path/'block_metadata.jsonl').read_text().splitlines()),1)

    def test_storage_guard_preserves_attempt_before_new_draw(self):
        class Disk:
            free=1
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'fit'
            with patch.object(prep.shutil,'disk_usage',return_value=Disk()),patch.object(prep,'generate_batch') as sampler:
                with self.assertRaises(OSError):self.run_fixture(path)
                sampler.assert_not_called()
            failure=json.loads((path/'preparation_failure.json').read_text())
            self.assertEqual(failure['completed_paths'],0)
            self.assertEqual(failure['failed_phase'],'batch_storage_guard')


@unittest.skipUnless(RESEARCH_RUNTIME_AVAILABLE,'The preparation fixtures require the research NumPy and SciPy environment')
class PathwisePolicyChecks(unittest.TestCase):
    def test_partitions_permutations_and_other_path_values_do_not_change_decisions(self):
        import numpy as np
        m,p,r,_,_=fixture_streams()
        block=prep.generate_batch(m,p,r,16,stream_binding='a')
        states=np.frombuffer(block['state_bytes'],dtype='<f8').reshape(16,10,5)
        together=prep.pathwise_stopping(p,states)
        separate=np.concatenate([prep.pathwise_stopping(p,states[i:i+1]) for i in range(16)])
        partitioned=np.concatenate([prep.pathwise_stopping(p,states[:3]),prep.pathwise_stopping(p,states[3:11]),prep.pathwise_stopping(p,states[11:])])
        self.assertTrue(np.array_equal(together,separate))
        self.assertTrue(np.array_equal(together,partitioned))
        reverse=np.arange(15,-1,-1)
        self.assertTrue(np.array_equal(prep.pathwise_stopping(p,states[reverse])[reverse],together))
        changed=states.copy();changed[1:]=100.
        self.assertEqual(prep.pathwise_stopping(p,changed)[0],together[0])

    def test_basis_conventions_match_immutable_sorted_polynomial_basis(self):
        import numpy as np
        import math
        from qf_research.optimal_stopping.multivariate_benchmark import sorted_polynomial_basis,monomial_indices
        current=(93.,102.,112.,100.,88.)
        coordinates=tuple(value/100.-1 for value in sorted(current,reverse=True))
        features=(1.,)+tuple(math.prod(coordinates[index] for index in term) for term in monomial_indices(5,2)[1:])
        self.assertTrue(np.array_equal(np.asarray(features),sorted_polynomial_basis(np.asarray([current]),100.,2)[0]))

    def test_strict_ties_continue_and_neighbouring_constant_changes_exercise(self):
        import numpy as np
        states=np.full((1,10,5),105.);states[:,0]=100.
        r=record();r['frozen_policy']=[[] for _ in range(10)]
        immediate=float(np.exp(-.05*(3./9)))*5.
        for constant,expected in [(immediate,9),(float(np.nextafter(immediate,-np.inf)),1),(float(np.nextafter(immediate,np.inf)),9)]:
            r['frozen_policy'][1]=[constant]+[0.]*20
            m,p=prep.policy_from_record(r,'fixture-training')
            self.assertEqual(prep.pathwise_stopping(p,states)[0],expected)

    def test_overflowing_monomials_continue_and_are_counted(self):
        import numpy as np
        m,p,_,_,_=fixture_streams()
        states=np.full((2,10,5),105.);states[:,0]=100.;states[0,1:]=1e308
        stops,diagnostics=prep.pathwise_stopping(p,states,return_diagnostics=True)
        self.assertEqual(list(stops),[9,1])
        self.assertEqual(diagnostics['arithmetic_fallback_occurrences'],8)
        self.assertEqual(sum(diagnostics['arithmetic_fallback_counts_by_date'].values()),8)
        self.assertTrue(all(row['path_index']==0 and 1<=row['date']<=8 for row in diagnostics['arithmetic_fallbacks']))
        reversed_stops=prep.pathwise_stopping(p,states[::-1])
        self.assertEqual(list(reversed_stops),[1,9])

    def test_overflowing_weighted_sum_continues_without_changing_coefficients(self):
        import numpy as np
        r=record();r['frozen_policy']=[[] for _ in range(10)];r['frozen_policy'][1]=[1e308]*21
        frozen=deepcopy(r['frozen_policy'])
        m,p=prep.policy_from_record(r,'fixture-training')
        states=np.full((1,10,5),300.);states[:,0]=100.
        stops,diagnostics=prep.pathwise_stopping(p,states,return_diagnostics=True)
        self.assertEqual(stops[0],9);self.assertEqual(diagnostics['arithmetic_fallback_occurrences'],1)
        self.assertEqual(r['frozen_policy'],frozen)
        self.assertEqual(list(p.rules[1].coefficients),frozen[1])

    def test_nonfinite_raw_simulation_state_remains_invalid(self):
        import numpy as np
        m,p,_,_,_=fixture_streams()
        states=np.full((1,10,5),100.);states[0,2,1]=np.inf
        with self.assertRaises(ValueError):prep.pathwise_stopping(p,states)


if __name__=='__main__':
    unittest.main()
