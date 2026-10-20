"""Numerical fixtures for clipping tails and uncertain bounded observations.

These small deterministic fixtures exercise mathematical boundary cases and
interval propagation. They are not simulated observations or research data.
"""
from fractions import Fraction
import importlib.util
import itertools
import os
from pathlib import Path
import unittest

try:
    import flint
    from flint import ctx
except ImportError:
    flint=ctx=None

SPEC = importlib.util.spec_from_file_location('decision_fixture_module',
    Path(__file__).with_name('certify_financial_decision.py')
    if Path(__file__).with_name('certify_financial_decision.py').is_file()
    else Path(__file__).resolve().parents[1]/'scripts/certify_financial_decision.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REPOSITORY = Path(os.environ.get('QF_TEST_REPOSITORY', Path(__file__).resolve().parents[1]))


def fraction(record):
    return MODULE.decode_rational(record)


def pair(record):
    return MODULE.decode_interval(record)


def exact_variance(values):
    n = len(values)
    return (sum(x*x for x in values)-sum(values)**2/n)/(n-1)


@unittest.skipIf(flint is None, "The separate python-flint environment is required")
class IntervalStatisticsTests(unittest.TestCase):
    def make(self, intervals, cap=10):
        accumulator = MODULE.IntervalClippedAccumulator(cap)
        for lower, upper in intervals:
            accumulator.add_interval(lower, upper)
        return accumulator

    def result(self, intervals, cap=10, delta=Fraction(1,640), tail=0):
        return self.make(intervals, cap).upper_limit(delta=delta, tail_upper=tail)

    def test_exact_point_statistics(self):
        result = self.result([(1,1),(3,3)], tail=Fraction(1,4))
        self.assertEqual(fraction(result['mean_upper']), 2)
        self.assertEqual(fraction(result['sample_midpoint_variance']), 2)
        self.assertEqual(fraction(result['squared_radius_budget']), 0)
        lower, upper = pair(result['variance_bound_certificate'])
        self.assertLessEqual(lower, 2)
        self.assertGreaterEqual(upper, 2)
        self.assertEqual(fraction(result['upper_limit']), pair(result['upper_limit_certificate'])[1])

    def test_midpoint_constant_does_not_remove_endpoint_variance(self):
        intervals = [(Fraction(2,5),Fraction(3,5))]*2
        result = self.result(intervals, cap=1)
        self.assertEqual(fraction(result['sample_midpoint_variance']), 0)
        self.assertEqual(fraction(result['squared_radius_budget']), Fraction(1,50))
        self.assertGreaterEqual(fraction(result['variance_upper']), Fraction(1,50))
        self.assertEqual(fraction(result['mean_upper']), Fraction(3,5))

    def test_all_interval_corners_have_enclosed_variance(self):
        intervals = [(Fraction(1,2),1),(2,Fraction(9,4)),(3,Fraction(7,2)),(5,6)]
        result = self.result(intervals)
        upper = fraction(result['variance_upper'])
        mean_upper = fraction(result['mean_upper'])
        for values in itertools.product(*intervals):
            self.assertGreaterEqual(upper, exact_variance(values))
            self.assertGreaterEqual(mean_upper, sum(values)/len(values))

    def test_bound_dominates_each_corner_exact_construction(self):
        intervals = [(0,Fraction(1,10)),(1,Fraction(6,5)),(4,Fraction(41,10))]
        result = self.result(intervals, cap=5, tail=Fraction(1,3))
        authoritative = fraction(result['upper_limit'])
        with ctx.workprec(192):
            for values in itertools.product(*intervals):
                variance = exact_variance(values)
                radius = (MODULE.ball(2*variance/len(values))*MODULE.ball(1280).log()).sqrt() + \
                    MODULE.ball(Fraction(7*5,3*(len(values)-1)))*MODULE.ball(1280).log()
                direct = MODULE.ball(sum(values)/len(values))+radius+MODULE.ball(Fraction(1,3))
                direct_lower, direct_upper = MODULE.endpoints(direct)
                self.assertGreaterEqual(authoritative, direct_upper)

    def test_clipping_is_monotone_and_retains_raw_width(self):
        accumulator = self.make([(-1,Fraction(1,4)),(9,12),(20,30)], cap=10)
        snapshot = accumulator.snapshot()
        self.assertEqual(fraction(snapshot['sum_lower']), 19)
        self.assertEqual(fraction(snapshot['sum_upper']), Fraction(81,4))
        self.assertEqual(fraction(snapshot['maximum_raw_width']), 10)
        self.assertEqual(fraction(snapshot['maximum_clipped_width']), 1)
        self.assertEqual(snapshot['count'], 3)

    def test_prefix_accumulation_matches_batch_sufficient_statistics(self):
        intervals = [(0,1),(3,4),(5,7),(0,0)]
        first = self.make(intervals[:2])
        for row in intervals[2:]:
            first.add_interval(*row)
        batch = self.make(intervals)
        self.assertEqual(first.snapshot(), batch.snapshot())

    def test_exact_all_zero_sample_retains_range_penalty(self):
        result = self.result([(0,0)]*8, cap=50, tail=Fraction(1,10))
        self.assertEqual(fraction(result['mean_upper']), 0)
        self.assertEqual(fraction(result['variance_upper']), 0)
        self.assertGreater(fraction(result['upper_limit']), Fraction(1,10))
        self.assertGreater(pair(result['sampling_radius_certificate'])[0], 0)

    def test_zero_cap_needs_no_assessment_observations(self):
        accumulator = MODULE.IntervalClippedAccumulator(0)
        result = accumulator.upper_limit(delta=Fraction(1,640), tail_upper=Fraction(3,7))
        self.assertFalse(result['empirical_bernstein_applied'])
        self.assertTrue(result['deterministic_zero_cap'])
        self.assertEqual(fraction(result['upper_limit']), Fraction(3,7))
        self.assertEqual(pair(result['upper_limit_certificate']), (Fraction(3,7),Fraction(3,7)))

    def test_zero_cap_clips_positive_uncertainty_exactly(self):
        accumulator = self.make([(1,10),(2,5)], cap=0)
        snapshot = accumulator.snapshot()
        self.assertEqual(fraction(snapshot['sum_midpoint_squared']), 0)
        self.assertEqual(fraction(snapshot['sum_radius_squared']), 0)
        self.assertEqual(fraction(snapshot['maximum_clipped_width']), 0)

    def test_positive_cap_needs_two_observations(self):
        for intervals in ([],[(1,2)]):
            with self.assertRaises(ValueError):
                self.result(intervals)

    def test_invalid_cap_is_rejected(self):
        for cap in (-1,True,float('nan'),float('inf'),'50'):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                MODULE.IntervalClippedAccumulator(cap)

    def test_reversed_and_negative_upper_gain_is_rejected(self):
        accumulator = MODULE.IntervalClippedAccumulator(10)
        with self.assertRaises(ValueError):
            accumulator.add_interval(2,1)
        with self.assertRaises(ArithmeticError):
            accumulator.add_interval(-2,-1)
        self.assertEqual(accumulator.count, 0)

    def test_invalid_endpoint_values_are_rejected_without_partial_update(self):
        accumulator = MODULE.IntervalClippedAccumulator(10)
        for lower, upper in ((True,1),(0,float('nan')),(0,float('inf')),(0,'1')):
            with self.subTest(lower=lower,upper=upper), self.assertRaises(ValueError):
                accumulator.add_interval(lower,upper)
        self.assertEqual(accumulator.count, 0)

    def test_invalid_probability_or_tail_is_rejected(self):
        accumulator = self.make([(0,1),(1,2)])
        for delta in (0,1,-1,True,float('nan')):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                accumulator.upper_limit(delta=delta, tail_upper=0)
        for tail in (-1,True,float('inf')):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                accumulator.upper_limit(delta=Fraction(1,640), tail_upper=tail)

    def test_insufficient_precision_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make([(0,1),(1,2)]).upper_limit(delta=Fraction(1,640), tail_upper=0,precision=31)

    def test_certificate_width_tampering_is_rejected(self):
        record = MODULE.interval_record(Fraction(1,3),Fraction(2,3))
        record['width'] = MODULE.rational_record(0)
        with self.assertRaises(ValueError):
            self.make([]).add_certificate(record)

    def test_zero_denominator_endpoint_is_rejected(self):
        record = MODULE.rational_record(1)
        record['denominator'] = '0'
        with self.assertRaises(ValueError):
            MODULE.decode_rational(record)

    def test_precision_context_is_restored(self):
        before = ctx.prec
        self.result([(1,2),(3,4)])
        self.assertEqual(ctx.prec,before)

    def test_binary_float_input_retains_exact_value(self):
        accumulator = self.make([(0.1,0.1),(0.2,0.2)])
        self.assertEqual(accumulator.sum_midpoint,Fraction(0.1)+Fraction(0.2))

    def test_claims_preserve_sampling_and_tail_premises(self):
        result = self.result([(1,1),(2,2)])
        self.assertFalse(result['sampling_law_certified'])
        self.assertFalse(result['financial_decision_certified'])
        self.assertIn('iid',result['sampling_input_premise'])
        self.assertIn('dominates',result['tail_input_premise'])


@unittest.skipIf(flint is None, "The separate python-flint environment is required")
class ZeroTailTests(unittest.TestCase):
    def build(self, spots=(100,), times=(0,1,2), **kwargs):
        return MODULE.build_zero_gain_tail_bound(spots,times,repository=REPOSITORY,**kwargs)

    def test_single_terminal_right_is_exact_zero(self):
        result = self.build(spots=(200,120),times=(0,3),cap=0)
        self.assertEqual(pair(result['clipping_tail_allowance']),(Fraction(0),Fraction(0)))
        self.assertTrue(result['sole_terminal_right_exact_zero'])
        self.assertEqual(result['work']['marginal_call_evaluations'],0)
        self.assertEqual(result['rows'],[])

    def test_deterministic_equal_rate_yield_fixture_has_exact_union_allowance(self):
        result = self.build(spots=(200,120),rate=0,dividend=0,volatility=0,cap=20)
        self.assertEqual(pair(result['clipping_tail_allowance']),(Fraction(0),Fraction(160)))
        self.assertEqual(result['work']['marginal_call_evaluations'],4)
        self.assertEqual(result['work']['conditional_integral_calls'],0)
        self.assertEqual(len(result['rows']),2)
        self.assertEqual(result['rows'][-1]['exercise_index'],2)

    def test_zero_cap_does_not_divide_by_cap(self):
        result = self.build(spots=(200,120),rate=0,dividend=0,volatility=0,cap=0)
        self.assertEqual(pair(result['clipping_tail_allowance']),(Fraction(0),Fraction(240)))
        for row in result['rows']:
            self.assertEqual(pair(row['threshold']),(Fraction(100),Fraction(100)))

    def test_non_positive_deterministic_payoffs_have_exact_zero_allowance(self):
        result = self.build(spots=(100,50),rate=0,dividend=0,volatility=0,cap=20)
        self.assertEqual(pair(result['clipping_tail_allowance']),(Fraction(0),Fraction(0)))

    def test_positive_rare_log_normal_tail_is_not_discarded(self):
        result = self.build(spots=(100,),times=(0,1,2),rate=0,dividend=0,volatility=.2,cap=10000)
        upper = pair(result['clipping_tail_allowance'])[1]
        self.assertGreater(upper,0)
        for row in result['rows']:
            self.assertGreater(pair(row['marginal_calls'][0])[1],0)

    def test_larger_cap_reduces_a_nontrivial_bound(self):
        small = self.build(spots=(100,100),cap=50)
        large = self.build(spots=(100,100),cap=150)
        self.assertLess(pair(large['clipping_tail_allowance'])[1],pair(small['clipping_tail_allowance'])[1])

    def test_discount_and_threshold_choices_are_outward(self):
        result = self.build(spots=(100,),cap=150)
        with ctx.workprec(192):
            for row in result['rows']:
                instant = fraction(row['time'])
                rate = Fraction(.05)
                d = (-MODULE.ball(rate)*MODULE.ball(instant)).exp()
                threshold = MODULE.ball(100)+MODULE.ball(150)*(MODULE.ball(rate)*MODULE.ball(instant)).exp()
                dl,du=MODULE.endpoints(d)
                tl,tu=MODULE.endpoints(threshold)
                self.assertGreaterEqual(pair(row['discount'])[1],du)
                self.assertLessEqual(fraction(row['threshold_lower_used']),tl)

    def test_tail_allowance_is_sum_of_recorded_terms(self):
        result = self.build(spots=(100,120),cap=150)
        total = sum(fraction(row['tail_upper']) for row in result['rows'])
        self.assertEqual(pair(result['clipping_tail_allowance'])[1],total)
        for row in result['rows']:
            discount_upper = pair(row['discount'])[1]
            self.assertEqual(fraction(row['tail_upper']),
                discount_upper*sum(pair(record)[1] for record in row['marginal_calls']))

    def test_invalid_spots_are_rejected(self):
        for spots in ([],[0],[-1],[float('nan')],['100'],'100'):
            with self.subTest(spots=spots), self.assertRaises(ValueError):
                self.build(spots=spots)

    def test_invalid_calendar_is_rejected(self):
        for times in ([0],[1,2],[0,0,1],[0,2,1],[0,True],[0,float('inf')],'0,1'):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.build(times=times)

    def test_invalid_model_or_precision_is_rejected(self):
        for kwargs in ({'strike':0},{'rate':-.01},{'dividend':-.01},{'volatility':-.1},
                       {'cap':-1},{'cap':True},{'correlation':.1},{'precision':31}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_tail_claim_does_not_certify_financial_sampling(self):
        result = self.build()
        self.assertTrue(result['ideal_model_tail_bound_certified'])
        self.assertFalse(result['paths_generated'])
        self.assertFalse(result['sampling_coverage_certified'])
        self.assertFalse(result['financial_decision_certified'])


@unittest.skipIf(flint is None, "The separate python-flint environment is required")
class ZeroGainTests(unittest.TestCase):
    def evaluate(self, states=((100,100),(140,80),(100,180)), times=(0,1,2), stopping=1, **kwargs):
        return MODULE.evaluate_zero_gain(states,times,stopping,**kwargs)

    def test_selected_candidate_is_exact_zero(self):
        result=self.evaluate(rate=0)
        self.assertEqual(pair(result['candidate_certificates']['1']),(Fraction(0),Fraction(0)))
        self.assertEqual(pair(result['gain_certificate']),(Fraction(40),Fraction(40)))
        self.assertEqual(result['work']['conditional_integral_calls'],0)
        self.assertTrue(result['completed'])

    def test_first_and_terminal_stops(self):
        self.assertEqual(pair(self.evaluate(stopping=1,rate=0)['gain_certificate']),
                         (Fraction(40),Fraction(40)))
        self.assertEqual(pair(self.evaluate(stopping=2,rate=0)['gain_certificate']),
                         (Fraction(0),Fraction(0)))

    def test_sole_terminal_right_cancels_reward_uncertainty(self):
        result=self.evaluate(states=((100,),(200,)),times=(0,3),stopping=1,rate=.05)
        self.assertGreater(pair(result['reward_certificates'][1])[1]-pair(result['reward_certificates'][1])[0],0)
        self.assertEqual(pair(result['gain_certificate']),(Fraction(0),Fraction(0)))

    def test_initial_date_is_excluded_from_gain(self):
        result=self.evaluate(states=((1000,1000),(140,80),(100,180)),rate=0)
        self.assertEqual(pair(result['gain_certificate']),(Fraction(40),Fraction(40)))

    def test_discounted_gain_encloses_independent_high_precision_expression(self):
        result=self.evaluate(rate=.05)
        lower,upper=pair(result['gain_certificate'])
        with ctx.workprec(192):
            truth=MODULE.ball(80)*(-MODULE.ball(Fraction(.05))*2).exp()-MODULE.ball(40)*(-MODULE.ball(Fraction(.05))).exp()
            tl,tu=MODULE.endpoints(truth)
        self.assertLessEqual(lower,tl)
        self.assertGreaterEqual(upper,tu)

    def test_exact_zero_rewards_produce_zero_gain(self):
        result=self.evaluate(states=((50,60),(80,90),(90,100)))
        self.assertEqual(pair(result['gain_certificate']),(Fraction(0),Fraction(0)))
        self.assertEqual(result['work']['reward_exponentials_evaluated'],0)

    def test_all_state_dimensions_are_checked(self):
        for states in (((100,),(120,130),(140,)),((100,),(0,),(140,)),
                       ((100,),('120',),(140,)),((100,),(float('inf'),),(140,))):
            with self.subTest(states=states),self.assertRaises(ValueError):
                self.evaluate(states=states)

    def test_stopping_index_validation(self):
        for stopping in (0,3,True,1.0):
            with self.subTest(stopping=stopping),self.assertRaises(ValueError):
                self.evaluate(stopping=stopping)

    def test_invalid_calendar_or_model(self):
        for kwargs in ({'times':(0,2,1)},{'times':(1,2,3)},{'strike':0},{'rate':-.1},{'precision':31}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.evaluate(**kwargs)

    def test_unsupported_scalar_state_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(states=(100,120,140))

    def test_gain_retains_probability_and_adapted_policy_premises(self):
        result=self.evaluate()
        self.assertFalse(result['sampling_law_certified'])
        self.assertIn('adapted',result['admissibility_input_premise'])


if __name__ == '__main__':
    unittest.main()
