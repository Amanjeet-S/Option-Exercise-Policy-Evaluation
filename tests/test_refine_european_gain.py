"""Domain, shared-level algebra, allocation and failure-preservation checks."""
from fractions import Fraction as F
from itertools import product
import os
from pathlib import Path
import unittest

import refine_european_gain as module


def pair(record):
    return module._interval(record)


def fixture_oracle(centres, *, exact=False, failure_call=None, empty_call=None, too_wide=False):
    calls=[]
    def oracle(state,horizon,requested_width,**kwargs):
        calls.append((horizon,requested_width))
        if failure_call is not None and len(calls)==failure_call:
            raise ArithmeticError('Fixture oracle failure before work is returned')
        centre=centres[horizon]
        radius=F(0) if exact else requested_width/(1 if too_wide else 2)
        if empty_call is not None and len(calls)==empty_call:
            centre+=100
        bounds=(max(F(0),centre-radius),centre+radius)
        work=module._zero_work();work['analytic_evaluations']=1;work['oracle_attempts']=1
        attempt=dict(attempt_index=1,requested_width_attained=not too_wide,
            returned_finite_certificate=True,value_certificate=module._record(bounds),work=work,
            complete_attempt_cost={'cpu_seconds':0.,'wall_seconds':0.})
        return dict(value_certificate=module._record(bounds),attempts=[attempt],work=work,
            complete_query_cost={'cpu_seconds':0.,'wall_seconds':0.},work_complete=True)
    oracle.calls=calls
    return oracle


@unittest.skipIf(module.tail.flint is None,'The separate python-flint environment is required')
class EuropeanGainTests(unittest.TestCase):
    def run_gain(self,oracle,*,arm='gain_adaptive',tau=1,target=F(1,100),max_rounds=16,
                 states=None,times=None,**kwargs):
        options=dict(rate=0,dividend=0,volatility=0,arm=arm,target_width=target,oracle=oracle,max_rounds=max_rounds)
        options.update(kwargs)
        return module.evaluate_gain(states or [[100],[110],[112],[115]],times or [0,1,2,3],tau,**options)

    def test_j1_has_exact_zero_gain_and_no_queries_despite_positive_terminal_reward(self):
        oracle=fixture_oracle({})
        result=self.run_gain(oracle,tau=1,states=[[100],[150]],times=[0,1])
        self.assertEqual(pair(result['gain_certificate']),(F(0),F(0)))
        self.assertEqual(pair(result['reward_certificates'][-1]),(F(50),F(50)))
        self.assertEqual(result['value_certificates'][-1],result['reward_certificates'][-1])
        self.assertEqual(pair(result['F_certificates'][-1]),(F(0),F(0)))
        self.assertEqual(oracle.calls,[])
        self.assertEqual(result['total_work'],module._zero_work())

    def test_nonterminal_selected_level_is_shared_once(self):
        oracle=fixture_oracle({F(2):F(8),F(1):F(11)})
        result=self.run_gain(oracle,target=F(2))
        # F1=[1.5,2.5], F2=[.5,1.5], F3=0. Selected F1 can
        # attain zero gain even though its independently widened self differs.
        self.assertEqual(pair(result['gain_certificate']),(F(0),F(0)))
        self.assertTrue(result['completed'])
        self.assertEqual(result['refinement_rounds'],[])

    def test_rectangular_propagation_contains_all_enumerated_exact_levels(self):
        levels=[None,(F(-2),F(1)),(F(3),F(4)),(F(0),F(0))]
        for tau in (1,2,3):
            bounds=module._gain(levels,tau)
            truths=[max(values)-values[tau-1] for values in product(*levels[1:])]
            self.assertEqual(bounds,(min(truths),max(truths)))

    def test_zero_reward_nonselected_dates_screen_true_positive_premiums(self):
        oracle=fixture_oracle({})
        result=self.run_gain(oracle,tau=3,states=[[100],[90],[95],[150]])
        self.assertEqual(oracle.calls,[])
        self.assertEqual(pair(result['gain_certificate']),(F(0),F(0)))
        self.assertEqual(len(result['excluded_dates']),2)
        self.assertGreater(pair(result['value_certificates'][1])[1],0)
        self.assertLess(pair(result['F_certificates'][1])[0],0)
        self.assertEqual(pair(result['F_certificates'][1])[1],0)
        self.assertEqual(result['value_certificates'][-1],result['reward_certificates'][-1])

    def test_zero_reward_selected_date_still_queries_a_positive_premium(self):
        oracle=fixture_oracle({F(2):F(5)},exact=True)
        result=self.run_gain(oracle,tau=1,states=[[100],[90],[95],[150]])
        self.assertEqual(len(oracle.calls),1)
        self.assertEqual(pair(result['value_certificates'][1]),(F(5),F(5)))
        self.assertEqual(pair(result['gain_certificate']),(F(5),F(5)))
        self.assertEqual([row['date'] for row in result['excluded_dates']],[2])

    def test_tied_dominance_retains_a_witness_and_ambiguous_candidates(self):
        levels=[None,(F(1),F(2)),(F(3),F(3)),(F(3),F(3)),(F(0),F(0))]
        active,excluded=module._dominance(levels,1)
        self.assertEqual(active,[2])
        self.assertIn(3,[row['date'] for row in excluded])
        # A wider candidate whose upper endpoint exceeds the witness remains.
        levels[3]=(F(2),F(4))
        active,excluded=module._dominance(levels,1)
        self.assertEqual(active,[2,3])

    def test_constant_terminal_witness_dominates_nonpositive_candidates(self):
        levels=[None,(F(-2),F(-1)),(F(-4),F(0)),(F(0),F(0))]
        active,excluded=module._dominance(levels,3)
        self.assertEqual(active,[])
        self.assertEqual([row['date'] for row in excluded],[1,2])
        self.assertTrue(all(row['witness_date']==3 for row in excluded))
        levels[2]=(F(-4),F(1))
        active,excluded=module._dominance(levels,3)
        self.assertEqual(active,[2])

    def test_point_score_retains_reward_uncertainty(self):
        rewards=[(F(0),F(0)),(F(10),F(11)),(F(12),F(13)),(F(0),F(0))]
        values=[None,(F(8),F(9)),(F(10),F(11)),(F(0),F(0))]
        levels=module._level_boxes(rewards,values)
        score=module._point_score(1,rewards,values,levels,1)
        self.assertEqual(score,F(1))
        self.assertLess(score,module._width(module._gain(levels,1)))

    def test_all_score_breakpoint_roles_match_an_exhaustive_exact_domain(self):
        reward_boxes=[(F(0),F(0)),(F(0),F(1)),(F(1),F(1)),(F(1),F(3)),(F(3),F(4))]
        price_boxes=[(F(0),F(0)),(F(0),F(1)),(F(1),F(3)),(F(2),F(4))]
        checked=0
        for H1,H2 in product(reward_boxes,repeat=2):
            for P1,P2 in product(price_boxes,repeat=2):
                rewards=[(F(0),F(0)),H1,H2,(F(0),F(0))]
                values=[None,P1,P2,(F(0),F(0))]
                levels=module._level_boxes(rewards,values)
                for stopping,date in product((1,2,3),(1,2)):
                    widths=[]
                    for numerator in range(int(4*values[date][0]),int(4*values[date][1])+1):
                        point=F(numerator,4)
                        changed=list(values);changed[date]=(point,point)
                        widths.append(module._width(module._gain(module._level_boxes(rewards,changed),stopping)))
                    # All breakpoints for these integer boxes are integer, so
                    # the exact quarter grid includes every possible minimum.
                    expected=max(F(0),module._width(module._gain(levels,stopping))-min(widths))
                    self.assertEqual(module._point_score(date,rewards,values,levels,stopping),expected)
                    checked+=1
        self.assertEqual(checked,2400)

    def test_common_and_adaptive_reach_the_same_exact_gain_with_equal_scientific_ledgers(self):
        results=[]
        for arm in ('coarse_common','gain_adaptive','reference_common'):
            oracle=fixture_oracle({F(2):F(10),F(1):F(7)})
            result=self.run_gain(oracle,arm=arm)
            results.append(result)
            self.assertTrue(result['completed'])
            self.assertLessEqual(module._width(pair(result['gain_certificate'])),F(1,100))
            self.assertLessEqual(sum(result['refinement_counts_by_date'].values()),result['additional_query_budget'])
            self.assertEqual(set(result['total_work']),set(module.WORK_KEYS))
            self.assertTrue(result['shared_structural_screening_and_dominance_pruning'])
            self.assertEqual(set(result),set(results[0]))
        for result in results:
            lower,upper=pair(result['gain_certificate'])
            self.assertLessEqual(lower,5);self.assertGreaterEqual(upper,5)

    def test_strong_common_does_not_refine_a_safely_dominated_date(self):
        # F1≈0 is selected; F2≈11 supplies the gain. An extra F3≈1
        # is screened by the same exact endpoint pruning in both methods.
        for arm in ('coarse_common','gain_adaptive'):
            oracle=fixture_oracle({F(3):F(10),F(2):F(1),F(1):F(19)})
            result=self.run_gain(oracle,arm=arm,states=[[100],[110],[112],[120],[130]],times=[0,1,2,3,4])
            refined=[q['date'] for row in result['refinement_rounds'] for q in row['queries']]
            self.assertNotIn(3,refined)
            self.assertTrue(result['completed'])

    def test_selected_uncertainty_refines_under_both_methods(self):
        for arm in ('coarse_common','gain_adaptive'):
            oracle=fixture_oracle({F(2):F(10),F(1):F(7)})
            result=self.run_gain(oracle,arm=arm)
            refined=[q['date'] for row in result['refinement_rounds'] for q in row['queries']]
            self.assertIn(1,refined)

    def test_refinement_failure_preserves_the_failed_query_and_unknown_work(self):
        oracle=fixture_oracle({F(2):F(10),F(1):F(7)},failure_call=3)
        with self.assertRaises(module.GainAssessmentFailure) as caught:
            self.run_gain(oracle)
        record=caught.exception.record
        self.assertFalse(record['completed'])
        self.assertEqual(len(record['initial_queries']),2)
        self.assertEqual(len(record['refinement_rounds'][0]['queries']),1)
        failed=record['refinement_rounds'][0]['queries'][0]
        self.assertIsNone(failed['attempts'][0]['integrand_evaluations'])
        self.assertEqual(record['total_work']['integrand_evaluations_unavailable_attempts'],1)
        self.assertFalse(record['work_complete'])
        self.assertEqual(record['total_work']['oracle_attempts'],2)
        self.assertEqual(record['unknown_outer_oracle_invocations'],1)
        self.assertIsNone(failed['primitive_attempt_count'])
        self.assertIn('gain_certificate',record)

    def test_returned_width_and_empty_intersection_fail_closed(self):
        for oracle in (fixture_oracle({F(2):F(10),F(1):F(7)},too_wide=True),
                       fixture_oracle({F(2):F(10),F(1):F(7)},empty_call=3)):
            with self.subTest(oracle=oracle),self.assertRaises(module.GainAssessmentFailure) as caught:
                self.run_gain(oracle)
            self.assertFalse(caught.exception.record['completed'])
            queries=caught.exception.record['initial_queries']+[q for step in caught.exception.record['refinement_rounds'] for q in step['queries']]
            self.assertIn('error',queries[-1])

    def test_forged_query_work_and_attempt_limit_are_rejected(self):
        for corruption in ('work','attempt_limit','negative_clock'):
            genuine=fixture_oracle({F(2):F(10),F(1):F(7)})
            def forged(*args,**kwargs):
                payload=genuine(*args,**kwargs)
                if corruption=='work':
                    payload['work']=dict(payload['work'],analytic_evaluations=100)
                elif corruption=='attempt_limit':
                    payload['attempts']*=20
                else:
                    payload['complete_query_cost']['cpu_seconds']=-1
                return payload
            with self.subTest(corruption=corruption),self.assertRaises(module.GainAssessmentFailure) as caught:
                self.run_gain(forged)
            self.assertFalse(caught.exception.record['completed'])
            self.assertFalse(caught.exception.record['work_complete'])
            self.assertGreater(caught.exception.record['total_work']['integrand_evaluations_unavailable_attempts'],0)

    def test_contradictory_final_outcome_or_changed_certificate_is_rejected(self):
        for corruption in ('finite_flag','attained_flag','final_certificate','query_clock'):
            genuine=fixture_oracle({F(2):F(10),F(1):F(7)})
            def contradictory(*args,**kwargs):
                payload=genuine(*args,**kwargs)
                if corruption=='finite_flag':
                    payload['attempts'][-1]['returned_finite_certificate']=False
                elif corruption=='attained_flag':
                    payload['attempts'][-1]['requested_width_attained']=False
                elif corruption=='final_certificate':
                    payload['attempts'][-1]['value_certificate']=module._record((F(20),F(20)))
                else:
                    payload['attempts'][-1]['complete_attempt_cost']['cpu_seconds']=1.
                return payload
            with self.subTest(corruption=corruption),self.assertRaises(module.GainAssessmentFailure) as caught:
                self.run_gain(contradictory)
            self.assertFalse(caught.exception.record['completed'])

    def test_already_tight_initial_returns_skip_only_compatible_quarters_above_floor(self):
        genuine=fixture_oracle({F(2):F(10),F(1):F(7)})
        def narrow(*args,**kwargs):
            payload=genuine(*args,**kwargs)
            centre=(pair(payload['value_certificate'])[0]+pair(payload['value_certificate'])[1])/2
            radius=min(args[2]/2,F(1,20000))
            payload['value_certificate']=module._record((centre-radius,centre+radius))
            payload['attempts'][-1]['value_certificate']=payload['value_certificate']
            return payload
        result=self.run_gain(narrow,target=F(1,10**8))
        self.assertTrue(result['completed'])
        self.assertTrue(any(row['skipped_already_attained_quarter_requests']>0 for row in result['refinement_rounds']))
        self.assertTrue(all(depth<=16 for depth in result['requested_quarter_depths_by_date'].values()))
        floor=pair(result['request_width_floor_certificate'])[0]
        self.assertTrue(all(pair(query['requested_width_certificate'])[0]>=floor
            for step in result['refinement_rounds'] for query in step['queries']))

    def test_no_shared_cache_between_assessments(self):
        oracle=fixture_oracle({F(2):F(10),F(1):F(7)},exact=True)
        first=self.run_gain(oracle)
        count=len(oracle.calls)
        second=self.run_gain(oracle)
        self.assertEqual(len(oracle.calls),2*count)
        self.assertEqual(first['gain_certificate'],second['gain_certificate'])
        self.assertIsNot(first['initial_queries'],second['initial_queries'])

    def test_budget_exhaustion_retains_unresolved_actual_width(self):
        oracle=fixture_oracle({F(2):F(10),F(1):F(7)})
        with self.assertRaises(module.GainAssessmentFailure) as caught:
            self.run_gain(oracle,max_rounds=1,target=F(1,10**12))
        record=caught.exception.record
        self.assertFalse(record['completed'])
        self.assertGreater(module._width(pair(record['gain_certificate'])),F(1,10**12))
        self.assertLessEqual(record['additional_queries'],record['additional_query_budget'])

    def test_common_and_adaptive_share_a_frozen_quarter_request_floor(self):
        for arm in ('coarse_common','gain_adaptive'):
            oracle=fixture_oracle({F(2):F(10),F(1):F(7)})
            with self.subTest(arm=arm),self.assertRaises(module.GainAssessmentFailure) as caught:
                self.run_gain(oracle,arm=arm,target=F(1,10**13))
            record=caught.exception.record
            floor=F(1,4**16)
            self.assertEqual(pair(record['request_width_floor_certificate']),(floor,floor))
            self.assertTrue(all(depth<=16 for depth in record['requested_quarter_depths_by_date'].values()))
            all_queries=record['initial_queries']+[query for step in record['refinement_rounds'] for query in step['queries']]
            self.assertTrue(all(pair(query['requested_width_certificate'])[0]>=floor for query in all_queries))
            self.assertLessEqual(record['additional_queries'],32)
            self.assertLessEqual(len(record['refinement_rounds']),16 if arm=='coarse_common' else 32)

    def test_invalid_calendar_state_policy_and_domain_are_rejected(self):
        cases=({'tau':0},{'tau':True},{'tau':4},{'states':[[100],[110],[112],[0]]},
               {'states':[[101],[110],[112],[115]]},{'times':[0,1,1,3]},
               {'states':[[100],[110,111],[112],[115]]},{'precision':16},
               {'rate':-1},{'dividend':-1},{'volatility':-1},{'correlation':F(1,10)})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises((ValueError,TypeError)):
                self.run_gain(fixture_oracle({}),**kwargs)

    def test_default_oracle_retries_preserve_wide_and_failed_attempt_work(self):
        real_loader=module.tail._load_certifier
        calls=[]
        def certifier(state,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                raise ArithmeticError('A failed finite integral fixture')
            width=F(2) if len(calls)==2 else F(1,1000)
            interval=module._record((F(1),F(1)+width))
            return dict(expectation=interval,finite_integral=interval,tail_allowance=module._record((F(0),F(0))),
                work={'integrand_evaluations':7,'nonfinite_integrand_evaluations':1})
        module.tail._load_certifier=lambda repository:certifier
        try:
            oracle=module._default_oracle_factory(Path('/unused'),max_attempts=3,cutoff_multiplier=F(2))
            result=oracle((F(100),F(100)),F(1),F(1,100),terminal_time=F(2),strike=F(100),
                rate=F(0),dividend=F(0),volatility=F(1,5),precision=96)
        finally:
            module.tail._load_certifier=real_loader
        self.assertEqual(len(result['attempts']),3)
        self.assertEqual(result['work']['conditional_integral_calls'],3)
        self.assertEqual(result['work']['successful_integral_certificates'],2)
        self.assertEqual(result['work']['integrand_evaluations_known'],14)
        self.assertEqual(result['work']['integrand_evaluations_unavailable_attempts'],1)
        self.assertFalse(result['work_complete'])
        self.assertFalse(result['attempts'][1]['requested_width_attained'])
        self.assertTrue(result['attempts'][2]['requested_width_attained'])
        self.assertEqual([q['cutoff'] for q in result['attempts']],
            [module.tail.rational_record(F(value)) for value in (200,400,800)])

    def test_default_oracle_failure_retains_every_attempt(self):
        real_loader=module.tail._load_certifier
        module.tail._load_certifier=lambda repository:(lambda *args,**kwargs:(_ for _ in ()).throw(ArithmeticError('finite failure')))
        try:
            oracle=module._default_oracle_factory(Path('/unused'),max_attempts=2,cutoff_multiplier=F(2))
            with self.assertRaises(module.OracleCertificationFailure) as caught:
                oracle((F(100),F(100)),F(1),F(1,100),terminal_time=F(2),strike=F(100),
                    rate=F(0),dividend=F(0),volatility=F(1,5),precision=96)
        finally:
            module.tail._load_certifier=real_loader
        self.assertEqual(len(caught.exception.record['attempts']),2)
        self.assertEqual(caught.exception.record['work']['conditional_integral_calls'],2)
        self.assertEqual(caught.exception.record['work']['integrand_evaluations_unavailable_attempts'],2)

    def test_small_real_two_asset_oracle_and_analytic_routes(self):
        repository=Path(os.environ.get('QF_REPOSITORY',str(Path(__file__).resolve().parents[1])))
        result=module.evaluate_gain([[100,100],[110,100],[115,120]], [0,F(1,100),F(2,100)],1,
            rate=0,dividend=0,volatility=F(1,5),target_width=F(1,100),repository=repository,precision=64)
        self.assertTrue(result['completed'])
        self.assertGreater(result['total_work']['conditional_integral_calls'],0)
        self.assertLessEqual(module._width(pair(result['gain_certificate'])),F(1,100))
        for sigma,states in ((0,[[100,100],[110,100],[115,120]]),(F(1,5),[[100],[110],[115]])):
            result=module.evaluate_gain(states,[0,F(1,100),F(2,100)],1,rate=0,dividend=0,
                volatility=sigma,target_width=F(1,100),repository=repository)
            self.assertTrue(result['completed'])
            self.assertEqual(result['total_work']['conditional_integral_calls'],0)
            self.assertGreater(result['total_work']['analytic_evaluations'],0)


if __name__=='__main__':
    unittest.main()
