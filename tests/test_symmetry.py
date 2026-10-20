"""Verify exact group projection and closed-form orbit moments and gradients."""
from collections import Counter
from fractions import Fraction as F
from itertools import permutations
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(os.environ.get('QF_TEST_REPOSITORY',Path(__file__).resolve().parents[1])).resolve()
SOURCE=ROOT/'src/qf_research/optimal_stopping/symmetry.py'
if not SOURCE.exists():SOURCE=Path(__file__).with_name('symmetry.py')
SPEC=importlib.util.spec_from_file_location('symmetry_verification_component',SOURCE)
symmetry=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=symmetry;SPEC.loader.exec_module(symmetry)
FIXTURE_SEED=2026999911


def polynomial(coefficients,state,indices):
    return sum((coefficient*math.prod(state[index] for index in signature)
                for coefficient,signature in zip(coefficients,indices)),F(0))


def direct_moment(signature,state,law):
    powers=Counter(signature)
    factor=math.prod(float(state[index])**power for index,power in powers.items())
    exponent=sum((power*(law.rate-law.dividends[index])+F(power*(power-1),2)*law.volatilities[index]**2
        for index,power in powers.items()),F(0))
    exponent+=sum((powers[i]*powers[j]*law.covariance[i][j] for i in powers for j in powers if i<j),F(0))
    return factor*math.exp(float(law.step*exponent))


def direct_gradient(signature,state,law,index):
    powers=Counter(signature);power=powers[index]
    if power==0:return 0.
    factor=power*math.prod(float(value)**(powers[i]-(1 if i==index else 0)) for i,value in enumerate(state))
    exponent=sum((count*(law.rate-law.dividends[i])+F(count*(count-1),2)*law.volatilities[i]**2
        for i,count in powers.items()),F(0))
    exponent+=sum((powers[i]*powers[j]*law.covariance[i][j] for i in powers for j in powers if i<j),F(0))
    return factor*math.exp(float(law.step*exponent))


class ExactProjectionTests(unittest.TestCase):
    def test_every_orbit_projection_matches_full_group_average(self):
        rng=random.Random(FIXTURE_SEED)
        for dimension in (1,2,3,5):
            indices=symmetry.complete_monomial_indices(dimension,3)
            coefficients=tuple(F(rng.randrange(-12,13),rng.randrange(1,8)) for _ in indices)
            state=tuple(F(index+1,3) for index in range(dimension))
            compressed=symmetry.project_coefficients(coefficients,dimension)
            projected=sum((coefficient*feature for coefficient,feature in zip(compressed,symmetry.orbit_basis_exact(state))),F(0))
            averaged=sum((polynomial(coefficients,tuple(state[index] for index in order),indices)
                for order in permutations(range(dimension))),F(0))/math.factorial(dimension)
            with self.subTest(dimension=dimension):
                self.assertEqual(projected,averaged)
                self.assertTrue(all(type(value) is F for value in compressed))

    def test_complete_degree_order_and_missing_groups(self):
        for dimension in (1,2,3,5):
            for degree in range(4):
                indices=symmetry.complete_monomial_indices(dimension,degree)
                self.assertEqual(len(indices),math.comb(dimension+degree,degree))
                self.assertEqual(indices[0],())
                self.assertEqual([len(signature) for signature in indices],sorted(len(signature) for signature in indices))
                projected=symmetry.project_coefficients([1]*len(indices),dimension,degree)
                present={symmetry.monomial_orbit_index(signature) for signature in indices}
                self.assertEqual(projected,tuple(F(1) if index in present else F(0) for index in range(7)))
        self.assertEqual(symmetry.project_coefficients([1,2,3,4],1),(F(1),F(2),F(3),F(0),F(4),F(0),F(0)))

    def test_mixed_orbit_counts_use_no_multinomial_factor(self):
        indices=symmetry.complete_monomial_indices(3,3);coefficients=[0]*len(indices)
        coefficients[indices.index((0,0,1))]=6
        projected=symmetry.project_coefficients(coefficients,3)
        self.assertEqual(projected[5],1)
        self.assertEqual(sum(projected),1)
        self.assertEqual(symmetry.orbit_basis_exact((1,2,3))[5],48)

    def test_exact_binary_floats_and_permutation_invariance(self):
        state=(0.1,0.2,0.3,0.,2.)
        exact=tuple(F.from_float(value) for value in state)
        expected=symmetry.orbit_basis_exact(exact)
        self.assertEqual(symmetry.orbit_basis_exact(state),expected)
        self.assertNotEqual(expected[1],F(13,5))
        for order in permutations(range(len(state))):
            self.assertEqual(symmetry.orbit_basis_exact(tuple(state[index] for index in order)),expected)

    def test_exact_recurrence_matches_every_monomial_and_zero_boundary(self):
        for state in [(F(0),),(F(0),F(2)),(F(1),F(0),F(3),F(2),F(4))]:
            expected=[F(0)]*7
            for signature in symmetry.complete_monomial_indices(len(state),3):
                expected[symmetry.monomial_orbit_index(signature)]+=math.prod(state[index] for index in signature)
            self.assertEqual(symmetry.orbit_basis_exact(iter(state)),tuple(expected))
        self.assertEqual(symmetry.orbit_basis_exact((0,0,0)),(1,0,0,0,0,0,0))

    def test_dimensions_coefficients_and_states_are_validated(self):
        for dimension,degree in [(0,3),(True,3),(2,4),(2,-1),(2,True)]:
            with self.subTest(dimension=dimension,degree=degree),self.assertRaises(ValueError):symmetry.complete_monomial_indices(dimension,degree)
        for coefficients in [[0]*9,[0]*11,[True]*10,[float('nan')]*10,[float('inf')]*10]:
            with self.subTest(coefficients=coefficients),self.assertRaises(ValueError):symmetry.project_coefficients(coefficients,2)
        for state in [(),(-1,),(True,),(float('nan'),),(float('inf'),),'100']:
            with self.subTest(state=state),self.assertRaises(ValueError):symmetry.orbit_basis_exact(state)
        for signature in [(1,0),(-1,), (True,), (0,0,0,0)]:
            with self.assertRaises(ValueError):symmetry.monomial_orbit_index(signature)


class ConditionalMomentTests(unittest.TestCase):
    def compare(self,state,**parameters):
        law=symmetry.validate_moment_law(len(state),**parameters)
        expected=[0.]*7
        for signature in symmetry.complete_monomial_indices(len(state),3):
            expected[symmetry.monomial_orbit_index(signature)]+=direct_moment(signature,state,law)
        result=symmetry.conditional_orbit_moments(state,**parameters)
        for orbit,(actual,reference) in enumerate(zip(result,expected)):
            self.assertTrue(math.isclose(actual,reference,rel_tol=3e-14,abs_tol=3e-14),(orbit,actual,reference))
        return result,law

    def test_common_sigma_exchangeable_singular_and_heterogeneous_q(self):
        for dimension in (1,2,3,5):
            state=tuple(F(index+1,5) for index in range(dimension))
            lower=-F(1,dimension-1) if dimension>1 else F(-1)
            for rho in (lower,F(0),F(1,3),F(1)):
                q=tuple(F(index-2,40) for index in range(dimension))
                with self.subTest(dimension=dimension,rho=rho):
                    self.compare(state,step=F(3,4),rate=F(-1,20),dividends=q,volatilities=F(1,5),correlation=rho)

    def test_independent_heterogeneous_sigma_and_zero_coordinates(self):
        state=(F(0),F(2,3),F(4,3),F(5,4),F(3))
        sigma=(F(0),F(1,10),F(1,5),F(3,10),F(2,5))
        result,law=self.compare(state,step=F(5,3),rate=F(3,100),dividends=(F(-1,10),F(0),F(1,10),F(1,5),F(-1,20)),volatilities=sigma,correlation=0)
        self.assertEqual(law.family,'independent_heterogeneous_volatility')
        self.assertTrue(all(value>=0 for value in result))

    def test_zero_time_zero_volatility_and_all_absorbing_states(self):
        state=(F(0),F(1,2),F(3,2))
        result,_=self.compare(state,step=0,rate=F(1,20),dividends=(F(0),F(1,10),F(1,5)),volatilities=(F(0),F(1,5),F(1,3)),correlation=0)
        self.assertEqual(result,tuple(float(value) for value in symmetry.orbit_basis_exact(state)))
        self.compare(state,step=F(2),rate=F(1,20),dividends=(F(0),F(1,10),F(1,5)),volatilities=0,correlation=-F(1,2))
        self.assertEqual(symmetry.conditional_orbit_moments((0,0,0),F(10),rate=0,dividends=0,volatilities=F(10),correlation=1),(1.,0.,0.,0.,0.,0.,0.))

    def test_joint_asset_permutation_preserves_heterogeneous_kernel(self):
        state=(F(1),F(2),F(3),F(4));q=(F(0),F(1,10),F(-1,5),F(3,10));sigma=(F(0),F(1,10),F(1,5),F(3,10))
        expected=symmetry.conditional_orbit_moments(state,F(2,3),rate=F(1,20),dividends=q,volatilities=sigma,correlation=0)
        for order in permutations(range(4)):
            actual=symmetry.conditional_orbit_moments(tuple(state[index] for index in order),F(2,3),rate=F(1,20),
                dividends=tuple(q[index] for index in order),volatilities=tuple(sigma[index] for index in order),correlation=0)
            for found,reference in zip(actual,expected):self.assertTrue(math.isclose(found,reference,rel_tol=3e-14,abs_tol=3e-14))

    def test_exchangeable_matrix_and_exact_covariance(self):
        matrix=((1,F(-1,2),F(-1,2)),(F(-1,2),1,F(-1,2)),(F(-1,2),F(-1,2),1))
        a=symmetry.validate_moment_law(3,F(1),rate=0,dividends=0,volatilities=F(1,5),correlation=matrix)
        b=symmetry.validate_moment_law(3,F(1),rate=0,dividends=0,volatilities=F(1,5),correlation=-F(1,2))
        self.assertEqual(a,b);self.assertEqual(a.covariance[0][0],F(1,25));self.assertEqual(a.covariance[0][1],-F(1,50))
        lower=symmetry.validate_moment_law(4,F(1),rate=0,dividends=0,volatilities=F(1,5),correlation=-F(1,3))
        self.assertEqual(lower.correlation,-F(1,3))

    def test_unsupported_laws_and_nonfinite_inputs_are_rejected(self):
        base=dict(step=F(1),rate=0,dividends=0,volatilities=F(1,5),correlation=0)
        changes=[dict(step=-1),dict(step=True),dict(rate=float('inf')),dict(dividends=True),dict(dividends=(0,0)),
            dict(volatilities=-1),dict(volatilities=float('nan')),dict(correlation=True),dict(correlation=float('nan')),
            dict(correlation=-F(51,100)),dict(correlation=F(101,100)),
            dict(volatilities=(F(1,10),F(1,5),F(1,3)),correlation=F(1,10)),
            dict(correlation=((1,F(1,10),F(1,5)),(F(1,10),1,0),(F(1,5),0,1))),
            dict(correlation=((1,0),(0,1))),dict(correlation=((2,0,0),(0,1,0),(0,0,1))),
            dict(correlation=((1,F(1,10),0),(0,1,0),(0,0,1)))]
        for changed in changes:
            parameters=dict(base,**changed)
            with self.subTest(changed=changed),self.assertRaises(ValueError):symmetry.conditional_orbit_moments((1,2,3),**parameters)
        for state in [(-1,2,3),(True,2,3),(float('nan'),2,3),()]:
            with self.subTest(state=state),self.assertRaises(ValueError):symmetry.conditional_orbit_moments(state,**base)

    def test_conditional_polynomial_value_matches_complete_monomial_formula(self):
        state=(F(2,3),F(1),F(4,3));indices=symmetry.complete_monomial_indices(3,3)
        raw=tuple(F((index%7)-3,5) for index in range(len(indices)))
        projected=symmetry.project_coefficients(raw,3)
        parameters=dict(step=F(3,2),rate=F(1,20),dividends=(0,F(1,10),F(-1,20)),volatilities=F(1,5),correlation=F(-1,2))
        law=symmetry.validate_moment_law(3,**parameters)
        expected=math.fsum(float(projected[symmetry.monomial_orbit_index(signature)])*direct_moment(signature,state,law) for signature in indices)
        self.assertTrue(math.isclose(symmetry.conditional_polynomial_value(projected,state,**parameters),expected,rel_tol=5e-14,abs_tol=5e-14))
        with self.assertRaises(ValueError):symmetry.conditional_polynomial_value([0]*6,state,**parameters)

    def test_floating_scope_and_unrepresentable_positive_intermediates(self):
        parameters=dict(step=1,rate=0,dividends=0,volatilities=F(1,5),correlation=0)
        for state in [(1e200,),(1e-200,)]:
            with self.subTest(state=state),self.assertRaises(ArithmeticError):symmetry.conditional_orbit_moments(state,**parameters)
        with self.assertRaises(ArithmeticError):symmetry.conditional_orbit_moments((1.,),1,rate=1000,dividends=0,volatilities=0)
        with self.assertRaisesRegex(ArithmeticError,'positive analytical product'):
            symmetry.conditional_orbit_moments((1e-100,)*3,1,rate=0,dividends=0,volatilities=15.,correlation=-F(1,2))
        self.assertIn('not certified exact numerical moments',symmetry.conditional_orbit_moments.__doc__)
        self.assertIn('no outward enclosure',symmetry.FLOATING_SCOPE)


class GradientTests(unittest.TestCase):
    def compare(self,state,parameters):
        coefficients=(F(1),F(-2,3),F(1,5),F(2,7),F(-1,9),F(1,11),F(1,13))
        law=symmetry.validate_moment_law(len(state),**parameters)
        expected=tuple(math.fsum(float(coefficients[symmetry.monomial_orbit_index(signature)])*direct_gradient(signature,state,law,index)
            for signature in symmetry.complete_monomial_indices(len(state),3)) for index in range(len(state)))
        actual=symmetry.conditional_polynomial_gradient(coefficients,state,**parameters)
        for found,reference in zip(actual,expected):self.assertTrue(math.isclose(found,reference,rel_tol=8e-14,abs_tol=8e-14),(found,reference))
        return coefficients,actual

    def test_gradient_matches_direct_monomial_derivatives_for_both_families(self):
        for dimension in (1,2,3,5):
            state=tuple(F(index,3) for index in range(dimension))
            lower=-F(1,dimension-1) if dimension>1 else F(-1)
            for rho in (lower,F(0),F(1)):
                with self.subTest(dimension=dimension,rho=rho):self.compare(state,dict(step=F(3,4),rate=F(1,20),dividends=tuple(F(index-2,30) for index in range(dimension)),volatilities=F(1,5),correlation=rho))
        self.compare((F(0),F(1),F(2)),dict(step=F(2),rate=F(-1,20),dividends=(F(0),F(1,10),F(-1,5)),volatilities=(F(0),F(1,10),F(3,10)),correlation=0))

    def test_zero_time_gradient_and_small_other_pair_are_retained(self):
        self.compare((F(0),F(2),F(3)),dict(step=0,rate=F(1,20),dividends=F(1,10),volatilities=F(1,5),correlation=1))
        coefficients=[0]*7;coefficients[-1]=1
        result=symmetry.conditional_polynomial_gradient(coefficients,(1e12,1e-12,1e-12),0,rate=0,dividends=0,volatilities=0)
        self.assertGreater(result[0],0);self.assertTrue(math.isclose(result[0],1e-24,rel_tol=2e-15))
        for parameters in [dict(rate=1e308,dividends=-1e308,volatilities=1e308),dict(rate=0,dividends=0,volatilities=(0,1e308,0))]:
            self.assertEqual(symmetry.conditional_polynomial_gradient([0,1,0,0,0,0,0],(1,2,3),0,**parameters),(1.,1.,1.))

    def test_finite_difference_is_secondary_sanity_and_scale_chain_rule(self):
        state=(.7,1.1,1.3);parameters=dict(step=.6,rate=.05,dividends=(.02,.1,-.01),volatilities=(.1,.2,.3),correlation=0)
        coefficients,actual=self.compare(state,parameters)
        for index in range(3):
            left=list(state);right=list(state);left[index]-=1e-5;right[index]+=1e-5
            difference=(symmetry.conditional_polynomial_value(coefficients,right,**parameters)-symmetry.conditional_polynomial_value(coefficients,left,**parameters))/2e-5
            self.assertTrue(math.isclose(difference,actual[index],rel_tol=2e-9,abs_tol=2e-9))
        scale=100.
        for index in range(3):
            left=[x*scale for x in state];right=left[:];left[index]-=.001;right[index]+=.001
            delta=(symmetry.conditional_polynomial_value(coefficients,[x/scale for x in right],**parameters)-symmetry.conditional_polynomial_value(coefficients,[x/scale for x in left],**parameters))/.002
            self.assertTrue(math.isclose(delta,actual[index]/scale,rel_tol=2e-9,abs_tol=2e-9))

    def test_gradient_validation_and_floating_scope(self):
        parameters=dict(step=1,rate=0,dividends=0,volatilities=F(1,5),correlation=0)
        for coefficients in [[0]*6,[True]*7,[float('inf')]*7]:
            with self.assertRaises(ValueError):symmetry.conditional_polynomial_gradient(coefficients,(1,2),**parameters)
        with self.assertRaises(ArithmeticError):symmetry.conditional_polynomial_gradient([1]*7,(1e200,2),**parameters)
        self.assertIn('divided by scale',symmetry.conditional_polynomial_gradient.__doc__)
        self.assertIn('No gradient rounding-error certificate',symmetry.conditional_polynomial_gradient.__doc__)


class OptionalNumpyBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import numpy as np
        except ImportError:raise unittest.SkipTest('The separate NumPy environment is needed only for the vectorised floating basis')
        cls.np=np

    def test_vectorised_basis_matches_exact_values_and_batch_partitions(self):
        np=self.np
        states=np.asarray([[0.,.25,.5,1.,2.],[.125,.375,.75,1.5,3.]])
        expected=np.asarray([[float(value) for value in symmetry.orbit_basis_exact(row)] for row in states])
        actual=symmetry.orbit_basis_float(states)
        self.assertTrue(np.array_equal(actual,expected))
        self.assertTrue(np.array_equal(actual,np.vstack([symmetry.orbit_basis_float(row) for row in states])))
        self.assertTrue(np.array_equal(symmetry.orbit_basis_float(np.zeros(3)),np.asarray([1.,0.,0.,0.,0.,0.,0.])))

    def test_vectorised_basis_rejects_boolean_complex_and_arithmetic_failures(self):
        np=self.np
        for values in [[],[True,1.],np.asarray([True]),np.asarray([1+1j]),[float('nan')],[-1.],np.empty((0,3))]:
            with self.subTest(values=values),self.assertRaises(ValueError):symmetry.orbit_basis_float(values)
        for values in [[1e200],[1e-200]]:
            with self.assertRaises(ArithmeticError):symmetry.orbit_basis_float(values)


class ArbConditionalCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import flint
        except ImportError:raise unittest.SkipTest('Outward analytical certificates use the separate python-flint environment')
        cls.flint=flint

    def ball(self,value):
        value=F(value);return self.flint.arb(self.flint.fmpq(value.numerator,value.denominator))

    def endpoints(self,value):
        self.assertTrue(value.is_finite())
        lower,upper=value.lower().fmpq(),value.upper().fmpq()
        return F(int(lower.p),int(lower.q)),F(int(upper.p),int(upper.q))

    def decoded(self,record):
        decode=lambda value:F(int(value['numerator']),int(value['denominator']))
        lo,hi=decode(record['lower']),decode(record['upper'])
        self.assertLessEqual(lo,hi);self.assertEqual(decode(record['width']),hi-lo)
        return lo,hi

    def independent_enumeration(self,coefficients,state,law):
        arb=self.flint.arb;value=arb(0);gradient=[arb(0) for _ in state]
        covariance=law.covariance
        for signature in symmetry.complete_monomial_indices(len(state),3):
            powers=Counter(signature)
            coefficient=self.ball(coefficients[symmetry.monomial_orbit_index(signature)])
            exponent=sum((power*(law.rate-law.dividends[i])+F(power*(power-1),2)*law.volatilities[i]**2
                for i,power in powers.items()),F(0))
            exponent+=sum((powers[i]*powers[j]*covariance[i][j] for i in powers for j in powers if i<j),F(0))
            multiplier=self.ball(law.step*exponent).exp()
            realised=math.prod(self.ball(state[i])**power for i,power in powers.items())
            value=value+coefficient*realised*multiplier
            for index in range(len(state)):
                power=powers[index]
                if power:
                    derivative=power*math.prod(self.ball(state[i])**(powers[i]-(1 if i==index else 0)) for i in range(len(state)))
                    gradient[index]=gradient[index]+coefficient*derivative*multiplier
        return self.endpoints(value),tuple(self.endpoints(item) for item in gradient)

    def assert_intersection(self,left,right):
        self.assertLessEqual(max(left[0],right[0]),min(left[1],right[1]))

    def test_certified_orbits_match_independent_full_monomial_value_and_gradient(self):
        coefficients=(F(1,2),F(-2,3),F(1,5),F(2,7),F(-1,9),F(1,11),F(1,13))
        cases=[]
        for dimension in (1,2,3,5):
            lower=-F(1,dimension-1) if dimension>1 else F(-1)
            for rho in (lower,F(0),F(1)):
                cases.append((tuple(F(index,3) for index in range(dimension)),dict(step=F(3,4),rate=F(-1,20),
                    dividends=tuple(F(index-2,30) for index in range(dimension)),volatilities=F(1,5),correlation=rho)))
        cases.append(((F(0),F(2,3),F(4,3)),dict(step=F(2),rate=F(1,20),dividends=(F(-1,10),F(0),F(1,10)),
            volatilities=(F(0),F(1,10),F(3,10)),correlation=0)))
        for state,parameters in cases:
            law=symmetry.validate_moment_law(len(state),**parameters)
            with self.subTest(dimension=len(state),family=law.family,rho=law.correlation):
                result=symmetry.certify_conditional_polynomial(coefficients,state,precision=128,**parameters)
                with self.flint.ctx.workprec(160):reference,gradient=self.independent_enumeration(coefficients,state,law)
                self.assert_intersection(self.decoded(result['value_certificate']),reference)
                for observed,expected in zip(result['gradient_certificates'],gradient):self.assert_intersection(self.decoded(observed),expected)

    def test_zero_time_has_exact_dyadic_value_and_gradient(self):
        state=(F(0),F(1,2),F(3,2));coefficients=(F(1,2),F(-1,4),F(1,8),F(3,4),F(-1,2),F(1,4),F(3,8))
        result=symmetry.certify_conditional_polynomial(coefficients,state,0,rate=F(1,20),dividends=F(1,10),volatilities=F(1,5),correlation=-F(1,2))
        expected=sum((coefficient*feature for coefficient,feature in zip(coefficients,symmetry.orbit_basis_exact(state))),F(0))
        self.assertEqual(self.decoded(result['value_certificate']),(expected,expected))
        law=symmetry.validate_moment_law(3,0,rate=0,dividends=0,volatilities=0,correlation=0)
        with self.flint.ctx.workprec(160):_,gradient=self.independent_enumeration(coefficients,state,law)
        for observed,expected in zip(result['gradient_certificates'],gradient):self.assertEqual(self.decoded(observed),expected)

    def test_multiple_precisions_intersect_and_context_is_restored(self):
        parameters=dict(step=F(5,3),rate=F(1,20),dividends=(0,F(1,10),F(-1,20)),volatilities=(0,F(1,5),F(1,3)),correlation=0)
        saved=self.flint.ctx.prec
        try:
            self.flint.ctx.prec=73
            low=symmetry.certify_conditional_polynomial([F(1,3)]*7,(F(1,3),F(2,3),F(4,3)),precision=128,**parameters)
            self.assertEqual(self.flint.ctx.prec,73)
            high=symmetry.certify_conditional_polynomial([F(1,3)]*7,(F(1,3),F(2,3),F(4,3)),precision=160,**parameters)
            self.assertEqual(self.flint.ctx.prec,73)
            self.assert_intersection(self.decoded(low['value_certificate']),self.decoded(high['value_certificate']))
            for left,right in zip(low['gradient_certificates'],high['gradient_certificates']):self.assert_intersection(self.decoded(left),self.decoded(right))
            from unittest.mock import patch
            with patch.object(symmetry,'_arb_interval_record',side_effect=ArithmeticError('fixture failed enclosure')):
                with self.assertRaises(ArithmeticError):symmetry.certify_conditional_polynomial([1]*7,(1,2,3),precision=128,**parameters)
            self.assertEqual(self.flint.ctx.prec,73)
        finally:self.flint.ctx.prec=saved

    def test_certification_retains_positive_quantities_below_float_underflow(self):
        coefficients=[0]*7;coefficients[-1]=1
        state=(1e-100,)*3;parameters=dict(step=1,rate=0,dividends=0,volatilities=15,correlation=-F(1,2))
        with self.assertRaises(ArithmeticError):symmetry.conditional_polynomial_value(coefficients,state,**parameters)
        result=symmetry.certify_conditional_polynomial(coefficients,state,precision=160,**parameters)
        self.assertGreater(self.decoded(result['value_certificate'])[0],0)
        self.assertEqual(result['value_certificate']['upper']['approximation'],0.)
        for gradient in result['gradient_certificates']:self.assertGreater(self.decoded(gradient)[0],0)

    def test_absorbing_zero_value_linear_gradient_and_scope_flags(self):
        coefficients=[F(3,2),F(1,7),1,2,3,4,5];state=(0,0,0)
        parameters=dict(step=F(2),rate=F(1,20),dividends=(F(0),F(1,10),F(-1,20)),volatilities=(0,F(1,5),F(1,3)),correlation=0)
        result=symmetry.certify_conditional_polynomial(coefficients,state,**parameters)
        self.assertEqual(self.decoded(result['value_certificate']),(F(3,2),F(3,2)))
        for index,q in enumerate(parameters['dividends']):
            with self.flint.ctx.workprec(160):expected=self.endpoints(self.ball(F(1,7))*self.ball((parameters['rate']-q)*parameters['step']).exp())
            self.assert_intersection(self.decoded(result['gradient_certificates'][index]),expected)
        for flag in ('analytical_value_enclosed','analytical_gradient_enclosed','supplied_numeric_inputs_treated_as_exact'):self.assertIs(result[flag],True)
        for flag in ('gaussian_sampling_law_certified','unrounded_ideal_path_enclosure_established','statistical_coverage_certified'):self.assertIs(result[flag],False)

    def test_certifier_validates_precision_and_unsupported_laws(self):
        parameters=dict(step=1,rate=0,dividends=0,volatilities=F(1,5),correlation=0)
        for precision in (True,31,128.0):
            with self.assertRaises(ValueError):symmetry.certify_conditional_polynomial([1]*7,(1,2,3),precision=precision,**parameters)
        with self.assertRaises(ValueError):symmetry.certify_conditional_polynomial([1]*6,(1,2,3),**parameters)
        with self.assertRaises(ValueError):symmetry.certify_conditional_polynomial([1]*7,(1,-2,3),**parameters)
        with self.assertRaises(ValueError):symmetry.certify_conditional_polynomial([1]*7,(1,2,3),1,rate=0,dividends=0,volatilities=(F(1,10),F(1,5),F(1,3)),correlation=F(1,10))


if __name__=='__main__':unittest.main()
