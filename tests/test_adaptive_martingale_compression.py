"""Distinct literal teachers; separate nominal and outward environments.

No production random generator is called. Research prepares a durable fixture
of exact paths. The NumPy-free certification environment replays those paths
and independently recomputes full residual energy.
"""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get('QF_TEST_REPOSITORY', Path(__file__).resolve().parents[1])).resolve()
if not (ROOT/'src/qf_research/optimal_stopping/martingale_compression.py').is_file():
    ROOT = next(parent for parent in Path(__file__).resolve().parents
        if (parent/'src/qf_research/optimal_stopping/martingale_compression.py').is_file())
TOOLS = ROOT/'data/private/revision/2026-10-06/budget_compression_v1/tools'
SOURCE = ROOT/'src/qf_research/optimal_stopping/adaptive_martingale_compression.py'
if not SOURCE.is_file(): SOURCE = TOOLS/'adaptive_martingale_compression.py'
spec = importlib.util.spec_from_file_location('adaptive_compression_fixture_source', SOURCE)
compression = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compression
spec.loader.exec_module(compression)
NUMPY = importlib.util.find_spec('numpy') is not None
FLINT = importlib.util.find_spec('flint') is not None
FIXTURE = TOOLS/'literal_paths.json'


def parameters(d=3, **changes):
    result = dict(rate=F(1,20), dividends=(F(1,10),)*d,
        volatilities=(F(1,5),)*d, correlation=0)
    result.update(changes)
    return result


def literals():
    return dict(
        asymmetric=dict(c=(F(7,13),F(1),F(-2),F(3)), x=(1,F(6,5),F(4,5)), t0=0,t1=1, degree=1,
            law=parameters(dividends=(F(1,50),F(1,10),F(9,50)),volatilities=(F(1,10),F(1,5),F(3,10)))),
        symmetric=dict(c=(F(7,13),F(1,2),F(1,2),F(1,2)), x=(1,1,1),t0=0,t1=1,degree=1,law=parameters()),
        zero_time=dict(c=(F(3,7),F(-1,2),F(5,8)),x=(1,2),t0=1,t1=1,degree=1,law=parameters(2)),
        zero_sigma=dict(c=(F(9,11),F(2,3),F(-5,7)),x=(1,2),t0=0,t1=1,degree=1,law=parameters(2,volatilities=0)),
        absorbing=dict(c=(F(2,9),F(1,3),F(-1,5)),x=(0,0),t0=0,t1=1,degree=1,law=parameters(2)),
        resource=dict(c=(F(5,17),F(-7,13),F(2,5)),x=(1,1),t0=0,t1=1,degree=1,law=parameters(2), max_gram_entries=1),
        underflow=dict(c=(F(3,19),F(-11,23)),x=(1,),t0=0,t1=1,degree=1,law=parameters(1,dividends=1000)),
        huge_constant=dict(c=(F(10)**400,F(1,2),F(1,2)),x=(1,1),t0=0,t1=1,degree=1,law=parameters(2)),
        singular=dict(c=(F(1,7),F(1),F(-2),F(3),F(1,2),F(1,3),F(-1,4),F(1,5),F(1,6),F(-1,7)),
            x=(1,1,1),t0=0,t1=1,degree=2,law=parameters(correlation=1)),
        negative=dict(c=(F(8,13),F(1),F(-1)),x=(1,1),t0=0,t1=1,degree=1,law=parameters(2,correlation=-1)))


def prepared(name):
    settings = dict(literals()[name]); c=settings.pop('c'); x=settings.pop('x')
    t0=settings.pop('t0');t1=settings.pop('t1')
    if NUMPY:
        return compression.prepare_compression_path(c,x,t0,t1,**settings)
    if not FIXTURE.is_file():
        raise AssertionError('Run the persistent literal-path preparation in the research environment first')
    return json.loads(FIXTURE.read_text())[name]


def certify(path,name,epsilon=F(1,30),**changes):
    item=literals()[name]
    return compression.certify_compression_path(path,item['c'],item['x'],item['t0'],item['t1'],
        epsilon=epsilon,law=item['law'],expected_path_sha256=path['path_sha256'],**changes)


def rehash(path):
    path['path_sha256']=compression.compression_path_sha256(path)
    return path


class ExactInputChecks(unittest.TestCase):
    def test_invalid_laws_are_rejected_before_fallback(self):
        invalid=[parameters(correlation=F(-3,5)),parameters(volatilities=(.1,.2,.3),correlation=.1),
            parameters(volatilities=-1),dict(parameters(),unknown=1),dict(rate=0,dividends=0)]
        for law in invalid:
            with self.subTest(law=law), self.assertRaises(ValueError):
                compression.prepare_compression_path((0,1,2,3),(1,1,1),0,1,law=law,degree=1,max_gram_entries=1)
        with self.assertRaises(ValueError):
            compression.prepare_compression_path((0,1),(1,),1,0,law=parameters(1),degree=1)

    def test_arity_degree_configuration_and_unknown_parameters(self):
        for c,x,degree in [((0,), (1,),1),((0,1),(1,1),1),((0,1),(1,),4),((False,1),(1,),1)]:
            with self.subTest(c=c,x=x,degree=degree),self.assertRaises(ValueError):
                compression.prepare_compression_path(c,x,0,1,law=parameters(len(x)),degree=degree)
        for changes in [dict(rcond=0),dict(rcond=1),dict(max_gram_entries=0),dict(max_gram_entries=True)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                compression.prepare_compression_path((0,1),(1,),0,1,law=parameters(1),degree=1,**changes)
        with self.assertRaises(TypeError):
            compression.prepare_compression_path((0,1),(1,),0,1,law=parameters(1),degree=1,unknown=True)

    def test_resource_fallback_preserves_exact_original(self):
        path=prepared('resource'); final=path['candidates'][-1]
        self.assertEqual(path['preparation']['failure']['type'],'resource_limit')
        self.assertEqual(len(path['candidates']),1)
        self.assertIs(final['compression'],False)
        self.assertEqual(tuple(compression.core.decode_rational(x) for x in final['expanded_coefficients']),literals()['resource']['c'])
        self.assertTrue(all(x['monomial_index']>0 for x in final['labelled_coefficients']))

    def test_hash_binds_every_frozen_candidate_and_model(self):
        path=prepared('asymmetric')
        self.assertEqual(path['path_sha256'],compression.compression_path_sha256(path))
        changed=deepcopy(path);changed['candidates'][0]['orbit_coefficients'][1]=compression.core.rational_record(0)
        self.assertNotEqual(path['path_sha256'],compression.compression_path_sha256(changed))
        changed=deepcopy(path);changed['model']['rate']=compression.core.rational_record(0)
        self.assertNotEqual(path['path_sha256'],compression.compression_path_sha256(changed))

    def test_all_thirty_three_frozen_source_hashes_are_preserved(self):
        manifests=[ROOT/'data/private/revision/2026-10-05/symmetry_compression_v1/design.json',
            ROOT/'data/private/revision/2026-10-05/financial_decision_v1/study/design.json']
        counts=[]
        for manifest in manifests:
            hashes=json.loads(manifest.read_text())['source_sha256'];counts.append(len(hashes))
            for source,expected in hashes.items():
                with self.subTest(source=source):
                    self.assertEqual(hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),expected)
        self.assertEqual(counts,[6,27])

    def test_exact_hierarchy_expansions_and_constant(self):
        path=prepared('asymmetric');c=literals()['asymmetric']['c'];previous=[]
        for candidate in path['candidates'][:-1]:
            self.assertEqual(compression.core.decode_rational(candidate['orbit_coefficients'][0]),c[0])
            labels=[entry['monomial_index'] for entry in candidate['labelled_coefficients']]
            if previous:self.assertEqual(labels[:-1],previous)
            self.assertEqual(len(labels),len(set(labels)))
            previous=labels
        signatures=compression.core.complete_monomial_indices(3,1)
        decoded=compression._validate_candidates(path,c,signatures)
        self.assertEqual(decoded[-1],c)


@unittest.skipUnless(NUMPY,'Nominal fitting requires the research environment')
class NominalPathChecks(unittest.TestCase):
    def test_independent_gain_ranking_and_first_atom(self):
        import numpy as np
        item=literals()['asymmetric'];path=prepared('asymmetric')
        c,x,t0,t1=item['c'],item['x'],item['t0'],item['t1']
        initial,t0,t1,law,signatures=compression.core._setup(x,t0,t1,degree=1,**item['law'])
        K,_=compression.core._nominal_gram(initial,t0,law,signatures,100)
        b=np.asarray([0.,1.,1.,1.]);numeric=np.asarray([0.]+[float(z) for z in c[1:]])
        gram=float(b@K@b);beta=float(b@K@numeric)/gram;r=numeric-beta*b
        gains=[(float(K[i]@r)**2/(float(K[i,i])-(float(b@K[:,i]))**2/gram),i) for i in range(1,4)]
        chosen=max(gains,key=lambda pair:(pair[0],-pair[1]))
        actual=path['candidates'][1]['diagnostics']['selected_atom']
        self.assertEqual(actual['monomial_index'],chosen[1])
        self.assertAlmostEqual(actual['nominal_energy_reduction'],chosen[0],places=13)
        self.assertEqual(actual['monomial_index'],2)

    def test_multiple_orbits_and_two_atoms_match_independent_refits(self):
        import numpy as np
        c=(F(3,17),F(1,2),F(-2,3),F(3,4),F(-4,5),F(5,6),F(-6,7),F(7,8),F(-8,9),F(9,10))
        x=(F(9,10),F(6,5),F(4,5))
        law=parameters(dividends=(F(1,50),F(1,10),F(9,50)),volatilities=(F(1,10),F(1,5),F(3,10)))
        path=compression.prepare_compression_path(c,x,F(1,4),F(5,4),law=law,degree=2)
        initial,t0,t1,actual,signatures=compression.core._setup(x,F(1,4),F(5,4),degree=2,**law)
        K,_=compression.core._nominal_gram(initial,t0,actual,signatures,100)
        categories=[compression.core.monomial_orbit_index(signature) for signature in signatures]
        present=sorted(set(categories)-{0})
        numeric=np.asarray([0.]+[float(value) for value in c[1:]])
        self.assertGreaterEqual(len(path['candidates']),4)
        for stage in range(2):
            candidate=path['candidates'][stage]
            labels=[entry['monomial_index'] for entry in candidate['labelled_coefficients']]
            A=np.asarray([[float(category==orbit) for orbit in present] for category in categories])
            for index in labels:
                atom=np.zeros((len(c),1));atom[index,0]=1.;A=np.concatenate((A,atom),axis=1)
            G=A.T@K@A
            independent=np.linalg.solve(G,A.T@K@numeric)
            residual=numeric-A@independent
            np.testing.assert_allclose(A.T@K@residual,0.,atol=3e-13)
            frozen=np.asarray([float(compression.core.decode_rational(candidate['orbit_coefficients'][orbit])) for orbit in present]
                +[float(compression.core.decode_rational(entry['coefficient'])) for entry in candidate['labelled_coefficients']])
            np.testing.assert_allclose(frozen,independent,rtol=2e-10,atol=2e-10)
            gains=[]
            for index in range(1,len(c)):
                if index in labels:continue
                cross=A.T@K[:,index]
                s=float(K[index,index]-cross@np.linalg.solve(G,cross))
                if s<=1e-11*float(K[index,index]):continue
                q=float(K[index]@residual);gains.append((q*q/s,index))
            chosen=max(gains,key=lambda pair:(pair[0],-pair[1]))
            next_candidate=path['candidates'][stage+1]
            recorded=next_candidate['diagnostics']['selected_atom']
            self.assertEqual(recorded['monomial_index'],chosen[1])
            self.assertAlmostEqual(recorded['nominal_energy_reduction'],chosen[0],places=10)
            decrease=candidate['diagnostics']['nominal_residual_energy']-next_candidate['diagnostics']['nominal_residual_energy']
            self.assertAlmostEqual(decrease,chosen[0],places=10)

    def test_one_and_two_dimensional_absent_orbits(self):
        for d,degree,c in [(1,0,(F(5,9),)),(1,1,(F(4,7),F(-2,3))),
            (1,2,(F(3,5),F(-1,2),F(7,9))), (1,3,(F(2,11),F(1,3),F(-5,7),F(3,8))),
            (2,3,(F(1,13),F(1,2),F(-1,3),F(1,4),F(-1,5),F(1,6),F(-1,7),F(1,8),F(-1,9),F(1,10)))]:
            with self.subTest(d=d,degree=degree):
                path=compression.prepare_compression_path(c,(1,)*d,0,1,law=parameters(d),degree=degree)
                compression._validate_candidates(path,c,compression.core.complete_monomial_indices(d,degree))
                for candidate in path['candidates']:
                    self.assertEqual(compression.core.decode_rational(candidate['orbit_coefficients'][6]),0)
                    if d==1:
                        self.assertEqual(compression.core.decode_rational(candidate['orbit_coefficients'][3]),0)
                        self.assertEqual(compression.core.decode_rational(candidate['orbit_coefficients'][5]),0)

    def test_zero_and_singular_energy_have_typed_diagnostics(self):
        for name in ['zero_time','zero_sigma','absorbing']:
            path=prepared(name)
            self.assertEqual(path['candidates'][0]['diagnostics']['nominal_rank'],0)
            self.assertEqual(path['candidates'][0]['diagnostics']['nominal_residual_energy'],0.)
        path=prepared('singular')
        self.assertLessEqual(path['candidates'][0]['diagnostics']['nominal_rank'],2)
        self.assertFalse(path['candidates'][0]['diagnostics']['negative_eigenvalues_clipped'])

    def test_numerical_failure_keeps_exact_identity(self):
        path=prepared('underflow')
        self.assertIn('failure',path['preparation'])
        self.assertEqual(len(path['candidates']),1)
        self.assertEqual(compression.core.decode_rational(path['candidates'][-1]['expanded_coefficients'][1]),F(-11,23))
        path=prepared('huge_constant')
        self.assertNotIn('failure',path['preparation'])
        self.assertEqual(compression.core.decode_rational(path['candidates'][0]['orbit_coefficients'][0]),F(10)**400)

    def test_sparse_teacher_structural_count_is_not_dense_arity(self):
        c=(F(1,7),0,F(-2,3),0)
        path=compression.prepare_compression_path(c,(1,1,1),0,1,law=parameters(),degree=1)
        self.assertEqual(path['candidates'][-1]['structural_coefficient_count'],2)
        self.assertEqual(len(path['candidates'][-1]['labelled_coefficients']),1)


@unittest.skipUnless(FLINT,'Outward acceptance requires the NumPy-free certification environment')
class OutwardAcceptanceChecks(unittest.TestCase):
    def test_first_structurally_smaller_acceptable_candidate(self):
        path=prepared('asymmetric');result=certify(path,'asymmetric',epsilon=F(1,5))
        self.assertEqual(result['selected_candidate_index'],1)
        self.assertIs(result['compression'],True)
        self.assertEqual([x['status'] for x in result['attempts']],['over_budget','accepted'])
        residual=tuple(c-compression.core.decode_rational(g) for c,g in zip(literals()['asymmetric']['c'],result['selected_candidate']['expanded_coefficients']))
        direct=compression.core.certify_residual_energy(residual,(0,)*7,literals()['asymmetric']['x'],0,1,degree=1,**literals()['asymmetric']['law'])
        self.assertEqual(result['selected_certificate']['energy_certificate'],direct['energy_certificate'])
        self.assertLessEqual(compression.core.decode_interval(direct['square_root_energy_certificate'])[1],F(1,5))

    def test_zero_budget_selects_exact_original_when_needed(self):
        path=prepared('asymmetric');result=certify(path,'asymmetric',epsilon=0)
        self.assertIs(result['compression'],False)
        self.assertEqual(result['selected_candidate_index'],len(path['candidates'])-1)
        self.assertEqual(compression.core.decode_interval(result['selected_certificate']['energy_certificate']),(0,0))
        self.assertTrue(result['selected_certificate']['exact_zero_martingale_residual'])

    def test_valid_domain_resource_and_nominal_failure_fallback(self):
        for name in ['resource','underflow']:
            with self.subTest(name=name):
                result=certify(prepared(name),name,epsilon=0)
                self.assertIs(result['compression'],False)
                self.assertEqual(compression.core.decode_interval(result['selected_certificate']['square_root_energy_certificate']),(0,0))

    def test_certificate_failure_is_preserved_before_identity(self):
        path=prepared('asymmetric');result=certify(path,'asymmetric',epsilon=0,max_gram_entries=1)
        self.assertEqual(result['attempts'][0]['status'],'certificate_failed')
        self.assertIs(result['compression'],False)
        self.assertEqual(result['attempts'][-1]['status'],'accepted')

    def test_deterministic_absorbing_and_huge_constant(self):
        for name in ['zero_time','zero_sigma','absorbing','huge_constant','symmetric']:
            with self.subTest(name=name):
                result=certify(prepared(name),name)
                self.assertIs(result['compression'],True)
                upper=compression.core.decode_interval(result['selected_certificate']['square_root_energy_certificate'])[1]
                self.assertLessEqual(upper,F(1,30))
                if name in ['zero_time','zero_sigma','absorbing']:self.assertEqual(upper,0)

    def test_negative_correlations_keep_full_residual(self):
        result=certify(prepared('negative'),'negative',epsilon=10)
        certificate=result['selected_certificate']
        self.assertGreater(compression.core.decode_interval(certificate['energy_certificate'])[0],0)
        self.assertEqual(certificate['model']['correlation'],compression.core.rational_record(-1))

    def test_budget_and_invalid_independent_law_are_rejected(self):
        path=prepared('asymmetric')
        for epsilon in [-1,float('nan'),True]:
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):certify(path,'asymmetric',epsilon=epsilon)
        item=literals()['asymmetric']
        with self.assertRaises(ValueError):
            compression.certify_compression_path(path,item['c'],item['x'],0,1,epsilon=0,
                law=parameters(correlation=-1),expected_path_sha256=path['path_sha256'])
        with self.assertRaises(ValueError):certify(path,'asymmetric',precision=16)

    def test_hash_tampering_rejected_without_rehash(self):
        path=prepared('asymmetric');changed=deepcopy(path)
        changed['candidates'][0]['expanded_coefficients'][1]=compression.core.rational_record(100)
        with self.assertRaises(ValueError):certify(changed,'asymmetric')
        item=literals()['asymmetric']
        with self.assertRaises(ValueError):
            compression.certify_compression_path(path,item['c'],item['x'],0,1,epsilon=1,
                law=item['law'],expected_path_sha256='0'*64)

    def test_inconsistent_exact_hierarchy_rejected_after_rehash(self):
        original=prepared('asymmetric')
        cases=[]
        changed=deepcopy(original);changed['candidates'][0]['expanded_coefficients'][1]=compression.core.rational_record(100);cases.append(changed)
        changed=deepcopy(original);changed['candidates'][1]['labelled_coefficients'][0]['monomial_index']=0;cases.append(changed)
        changed=deepcopy(original);changed['candidates'][1]['labelled_coefficients'].append(deepcopy(changed['candidates'][1]['labelled_coefficients'][0]));cases.append(changed)
        changed=deepcopy(original);changed['candidates'][1]['structural_coefficient_count']=0;cases.append(changed)
        changed=deepcopy(original);changed['candidates'][-1]['compression']=True;cases.append(changed)
        changed=deepcopy(original);changed['candidates'][0]['orbit_coefficients'][0]=compression.core.rational_record(0);cases.append(changed)
        changed=deepcopy(original);changed['candidates'][0]['orbit_coefficients'][6]=compression.core.rational_record(1);cases.append(changed)
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):certify(rehash(case),'asymmetric')

    def test_external_teacher_model_time_basis_source_bindings(self):
        original=prepared('asymmetric');cases=[]
        for key,value in [('previous_time',compression.core.rational_record(F(1,2))),('monomial_count',5),('degree',2)]:
            changed=deepcopy(original);changed[key]=value;cases.append(changed)
        changed=deepcopy(original);changed['model']['rate']=compression.core.rational_record(0);cases.append(changed)
        changed=deepcopy(original);changed['teacher_coefficients'][1]=compression.core.rational_record(99);cases.append(changed)
        changed=deepcopy(original);changed['source_identity']['symmetry.py']='0'*64;cases.append(changed)
        changed=deepcopy(original);changed['monomial_signatures'][1]=[1];cases.append(changed)
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):certify(rehash(case),'asymmetric')
        item=literals()['asymmetric']
        with self.assertRaises(ValueError):
            compression.certify_compression_path(original,(0,1,-2,3),item['x'],0,1,
                epsilon=1,law=item['law'],expected_path_sha256=original['path_sha256'])

    def test_nominal_diagnostics_are_not_certificate_authority(self):
        path=prepared('asymmetric')
        path['candidates'][0]['diagnostics']['nominal_residual_energy']=0.
        result=certify(rehash(path),'asymmetric',epsilon=F(1,5))
        self.assertEqual(result['selected_candidate_index'],1)
        self.assertEqual(result['attempts'][0]['status'],'over_budget')

    def test_returned_certificate_is_a_detached_path_snapshot(self):
        path=prepared('asymmetric');result=certify(path,'asymmetric',epsilon=F(1,5))
        before=deepcopy(result)
        path['model']['rate']=compression.core.rational_record(999)
        path['source_identity']['symmetry.py']='0'*64
        path['candidates'][1]['expanded_coefficients'][1]=compression.core.rational_record(999)
        self.assertEqual(result,before)

    def test_noncanonical_exact_records_and_identity_flags_are_rejected(self):
        original=prepared('asymmetric');cases=[]
        changed=deepcopy(original);changed['candidates'][0]['orbit_coefficients'][0]['numerator']='007';cases.append(changed)
        changed=deepcopy(original);changed['candidates'][0]['orbit_coefficients'][1]['approximation']=0;cases.append(changed)
        changed=deepcopy(original);changed['candidates'][-1]['compression']=0;cases.append(changed)
        changed=deepcopy(original);changed['schema_version']=True;cases.append(changed)
        for case in cases:
            with self.subTest(case=case),self.assertRaises(ValueError):certify(rehash(case),'asymmetric')

    def test_eight_date_declared_prototype_budget(self):
        epsilon=F(1,30)
        self.assertLess(8*epsilon**2,F(1,10)**2)
        self.assertEqual([F(1,300),epsilon,F(1,3)],[F(1,300),F(1,30),F(1,3)])


if __name__=='__main__':unittest.main()
