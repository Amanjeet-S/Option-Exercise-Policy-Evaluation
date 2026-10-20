"""Exact-domain, envelope and small outward-arithmetic checks."""
from fractions import Fraction as F
import os
from pathlib import Path
import unittest

import certify_european_gain_tail as module


def decode(value):
    return F(int(value['numerator']), int(value['denominator']))


def pair(value):
    return decode(value['lower']), decode(value['upper'])


@unittest.skipIf(module.flint is None, 'The separate python-flint environment is required')
class EuropeanGainTailTests(unittest.TestCase):
    def builder(self, **overrides):
        args = dict(spots=[150,150], exercise_times=[0, 1, 2], strike=100,
            rate=0, dividend=0, volatility=0, b_A=1, b_D=10,
            atm_certificates={F(1): (0, 10)}, precision=96)
        args.update(overrides)
        return module.build_european_restricted_tail_bound(**args)

    def test_exact_zero_time_and_zero_threshold_branches(self):
        for spot, threshold, expected in ((100, 90, 10), (100, 100, 0), (100, 120, 0), (100, 0, 100)):
            self.assertEqual(module.lognormal_call_moment(spot, 0, threshold,
                rate=0, dividend=0, volatility=0), (F(expected), F(expected)))
        self.assertEqual(module.lognormal_call_moment(100, 1, 0,
            rate=0, dividend=0, volatility=0), (F(100), F(100)))

    def test_zero_volatility_calls_have_deterministic_payoffs(self):
        self.assertEqual(module.lognormal_call_moment(150, 1, 100,
            rate=0, dividend=0, volatility=0), (F(50), F(50)))
        self.assertEqual(module.lognormal_call_moment(90, 1, 100,
            rate=0, dividend=0, volatility=0), (F(0), F(0)))
        lo, hi = module.lognormal_call_moment(100, 1, 100,
            rate=F(1,10), dividend=0, volatility=0)
        with module.ctx.workprec(160):
            expected = module.endpoints(100*(module.ball(F(1,10)).exp()-1))
        self.assertLessEqual(lo, expected[0])
        self.assertGreaterEqual(hi, expected[1])

    def test_positive_volatility_moment_contains_independent_closed_form(self):
        lo, hi = module.lognormal_call_moment(100, F(1,10), 100,
            rate=0, dividend=0, volatility=F(1,5), precision=64)
        with module.ctx.workprec(160):
            sd = module.ball(F(1,5))*module.ball(F(1,10)).sqrt()
            expected = module.endpoints(100*(2*module._normal_cdf(sd/2)-1))
        self.assertLessEqual(lo, expected[0])
        self.assertGreaterEqual(hi, expected[1])
        self.assertGreater(lo, 0)

    def test_tiny_positive_tail_keeps_a_serialisable_outward_upper_bound(self):
        lo, hi = module.lognormal_call_moment(100, F(1,100), 10**8,
            rate=0, dividend=0, volatility=F(1,5), precision=96)
        self.assertEqual(lo, 0)
        self.assertEqual(hi, F(1,2**96))
        import json
        self.assertLess(len(json.dumps(module.interval_record(lo,hi))), 1000)

    def test_terminal_row_is_exact_zero_and_needs_no_certificate(self):
        result = self.builder()
        terminal = result['rows'][-1]
        for key in ('discounted_atm_call', 'a', 'c', 'd'):
            self.assertEqual(pair(terminal[key]), (F(0), F(0)))
        self.assertEqual(decode(terminal['positive_tail_upper']), 0)
        self.assertEqual(decode(terminal['negative_tail_upper']), 0)
        self.assertIsNone(terminal['positive_threshold'])
        self.assertIsNone(terminal['negative_threshold'])
        self.assertEqual(result['cap_split']['B'], module.rational_record(F(11)))

    def test_terminal_only_calendar_has_zero_tail_for_every_spot(self):
        result = self.builder(exercise_times=[0, 2], atm_certificates={}, spots=[80, 140, 200])
        self.assertEqual(pair(result['clipping_tail_allowance']), (F(0), F(0)))

    def test_equal_negative_cap_and_intercept_is_valid_and_keeps_branch(self):
        result = self.builder()
        first = result['rows'][0]
        self.assertEqual(decode(first['discounted_atm_intercept_upper']), 10)
        self.assertEqual(decode(first['negative_threshold']), 100)
        self.assertGreater(decode(first['negative_slope_upper']), 0)
        self.assertGreaterEqual(decode(first['negative_tail_upper']), 10)
        self.assertLess(decode(first['negative_tail_upper'])-10, F(1,10**20))
        self.assertEqual(decode(first['positive_tail_upper']), 0)

    def test_cap_below_certified_intercept_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'negative cap'):
            self.builder(b_D=F(999,100))
        with self.assertRaisesRegex(ValueError, 'negative cap'):
            self.builder(b_D=0)

    def test_exact_zero_slopes_are_skipped(self):
        result = self.builder(atm_certificates={F(1):(0,0)}, b_D=0)
        self.assertEqual(pair(result['clipping_tail_allowance']), (F(0), F(0)))
        self.assertEqual(pair(result['rows'][0]['a']), (F(0),F(0)))
        self.assertEqual(pair(result['rows'][0]['d']), (F(0),F(0)))
        self.assertEqual(result['rows'][0]['positive_marginal_calls'], [])
        self.assertEqual(result['rows'][0]['negative_marginal_calls'], [])

    def test_unresolved_d_slope_is_not_discarded(self):
        # Cbar=K/2 and exp(-r h)=1/2 are compared at finite precision.
        # The exact rational Cbar is positive and near the zero boundary.
        result = self.builder(rate=float(__import__('math').log(2)),
            atm_certificates={F(1):(0,50)}, b_D=50, precision=32)
        lo, hi = pair(result['rows'][0]['d'])
        self.assertEqual(lo, 0)
        self.assertGreater(hi, 0)
        self.assertIsNotNone(result['rows'][0]['negative_threshold'])
        self.assertEqual(len(result['rows'][0]['negative_marginal_calls']), 2)

    def test_positive_envelope_matches_a_deterministic_single_date_excess(self):
        result = self.builder(dividend=F(1,10), atm_certificates={F(1):(0,0)}, b_D=0,
            b_A=1, spots=[200])
        first = result['rows'][0]
        self.assertEqual(pair(first['c']), (F(0), F(0)))
        self.assertEqual(decode(first['negative_tail_upper']), 0)
        with module.ctx.workprec(160):
            expected = module.endpoints((1-(-module.ball(F(1,10))).exp())*200*
                                        (-module.ball(F(1,10))).exp()-1)
        upper = decode(first['positive_tail_upper'])
        self.assertGreaterEqual(upper, expected[1])
        self.assertLess(upper-expected[0], F(1,10**20))

    def test_complete_tail_encloses_every_deterministic_stopping_date(self):
        times=[F(0),F(1,2),F(1),F(3,2)]
        result=self.builder(spots=[300,200],exercise_times=times,rate=F(1,20),
            dividend=F(1,10),volatility=0,b_A=1,b_D=0,atm_certificates=None)
        bound=pair(result['clipping_tail_allowance'])[1]
        self.assertTrue(result['ideal_model_tail_bound_certified'])
        with module.ctx.workprec(160):
            r,q=module.ball(F(1,20)),module.ball(F(1,10))
            T=module.ball(times[-1])
            terminal_cash=(-r*T).exp()*(300*((r-q)*T).exp()-100)
            F_dates=[(-r*module.ball(t)).exp()*(300*((r-q)*module.ball(t)).exp()-100)-terminal_cash
                     for t in times[1:]]
            for stopping,F_tau in enumerate(F_dates):
                upper=max(module.endpoints(value-F_tau)[1] for value in F_dates)
                clipped_upper=max(F(0),upper-1)
                with self.subTest(stopping=stopping+1):
                    self.assertGreaterEqual(bound,clipped_upper)
        self.assertGreater(bound,0)
        for row in result['rows']:
            self.assertEqual(decode(row['negative_tail_upper']),0)

    def test_outward_slope_and_intercept_endpoints_determine_safe_thresholds(self):
        result = self.builder(rate=F(1,20), dividend=F(1,10), volatility=F(1,5), spots=[100]*3,
            atm_certificates={F(1):(0,15)}, b_A=20, b_D=30)
        row=result['rows'][0]
        a=pair(row['a']); c=pair(row['c']); d=pair(row['d']); discount=pair(row['discount'])
        Cbar=decode(row['atm_upper_used'])
        self.assertEqual(decode(row['positive_threshold']), (c[0]+20)/a[1])
        self.assertEqual(decode(row['negative_slope_upper']), discount[1]*d[1])
        self.assertEqual(decode(row['negative_threshold']),
            100+(30-discount[1]*Cbar)/(discount[1]*d[1]))
        self.assertEqual(decode(row['positive_tail_upper']), a[1]*sum(pair(call)[1] for call in row['positive_marginal_calls']))
        self.assertEqual(decode(row['negative_tail_upper']), discount[1]*d[1]*sum(pair(call)[1] for call in row['negative_marginal_calls']))
        self.assertEqual(pair(result['clipping_tail_allowance'])[1],
            sum(decode(value['positive_tail_upper'])+decode(value['negative_tail_upper']) for value in result['rows']))

    def test_missing_extra_or_false_terminal_atm_rows_are_rejected(self):
        for supplied in ({}, {F(2):(0,1)}, {F(1):(0,10),F(2):(0,10)}, {F(1):(0,10),F(0):(0,1)}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                self.builder(atm_certificates=supplied)

    def test_invalid_supplied_atm_bounds_and_width_are_rejected(self):
        for supplied in ((-1,10),(10,1),(0,float('inf')),None,('0','1')):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                self.builder(atm_certificates={F(1):supplied})
        changed=module.interval_record(F(0),F(10))
        changed['width']=module.rational_record(F(9))
        with self.assertRaisesRegex(ValueError,'exact width'):
            self.builder(atm_certificates={F(1):changed})

    def test_unsupported_model_and_invalid_cap_or_calendar_are_rejected(self):
        cases=({'rate':-1},{'dividend':-1},{'volatility':-1},{'correlation':F(1,10)},
               {'spots':[]},{'spots':[0]},{'spots':[float('nan')]},{'strike':0},
               {'b_A':0},{'b_A':-1},{'b_D':-1},{'precision':True},{'precision':16},
               {'exercise_times':[0]},{'exercise_times':[1,2]},{'exercise_times':[0,1,1]},
               {'exercise_times':[0,2,1]},{'exercise_times':[0,float('inf')]},
               {'cutoff_multiplier':1},{'evaluation_limit':0},{'absolute_tolerance':0})
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.builder(**overrides)

    def test_negative_or_invalid_marginal_inputs_are_rejected(self):
        for spot,time,threshold,sigma in ((0,1,100,.2),(100,-1,100,.2),(100,1,-1,.2),(100,1,100,-.2),
                                         (True,1,100,.2),(100,1,float('nan'),.2)):
            with self.subTest(inputs=(spot,time,threshold,sigma)), self.assertRaises(ValueError):
                module.lognormal_call_moment(spot,time,threshold,rate=0,dividend=0,volatility=sigma)

    def test_injected_certificate_scope_is_explicit(self):
        result=self.builder()
        self.assertTrue(result['supplied_atm_certificates'])
        self.assertFalse(result['ideal_model_tail_bound_certified'])
        for key in ('financial_decision_certified','sampling_coverage_certified','paths_generated','policy_fitting_executed'):
            self.assertFalse(result[key])

    def test_analytic_single_asset_atm_route_needs_no_integrator(self):
        result=self.builder(atm_certificates=None, spots=[100], volatility=F(1,5),
            exercise_times=[0,F(1,100),F(2,100)], b_A=20,b_D=30)
        self.assertTrue(result['ideal_model_tail_bound_certified'])
        self.assertIsNone(result['rows'][0]['atm_integration'])
        self.assertEqual(pair(result['rows'][0]['d']),(F(0),F(0)))
        self.assertEqual(result['rows'][0]['negative_marginal_calls'],[])
        self.assertEqual(result['rows'][0]['recession_coefficient_method'],'exact_single_asset_dividend_discount')
        self.assertGreater(pair(result['rows'][0]['discounted_atm_call'])[0],0)

    def test_one_dimensional_recession_slope_is_exactly_zero_even_with_loose_atm_bound(self):
        result=self.builder(spots=[150],atm_certificates={F(1):(0,50)},b_D=50)
        row=result['rows'][0]
        self.assertEqual(pair(row['d']),(F(0),F(0)))
        self.assertEqual(decode(row['negative_tail_upper']),0)
        self.assertEqual(row['negative_marginal_calls'],[])

    def test_actual_small_two_asset_atm_uses_existing_verified_api(self):
        repository=Path(os.environ.get('QF_REPOSITORY',str(Path(__file__).resolve().parents[1])))
        result=self.builder(atm_certificates=None, spots=[100,100], volatility=F(1,5),
            exercise_times=[0,F(1,100),F(2,100)], b_A=20,b_D=30,
            repository=repository, precision=64, cutoff_multiplier=2,
            absolute_tolerance=F(1,10**8),relative_tolerance=F(1,10**8),evaluation_limit=3000)
        first=result['rows'][0]
        self.assertTrue(result['ideal_model_tail_bound_certified'])
        self.assertIsNotNone(first['atm_integration'])
        self.assertGreater(first['atm_integration']['work']['integrand_evaluations'],0)
        single=module.lognormal_call_moment(100,F(1,100),100,rate=0,dividend=0,volatility=F(1,5),precision=96)
        atm=pair(first['discounted_atm_call'])
        self.assertGreaterEqual(atm[1],single[0])
        self.assertLessEqual(atm[0],2*single[1])

    def test_arithmetic_precision_context_is_restored(self):
        old=module.ctx.prec
        self.builder(precision=64)
        self.assertEqual(module.ctx.prec,old)


if __name__=='__main__':
    unittest.main()
