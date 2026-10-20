"""Reject altered public-model results and manuscript transcriptions.

These tests check arithmetic and the stated scope of the retained aggregates.
They do not rerun simulations or certify statistical coverage.
"""

import copy
import importlib.util
import json
import math
from statistics import NormalDist
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('revision_result_summary', HERE / 'revision_results.py')
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class RevisionResultsTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE.parent / 'revision_results.json').read_text())

    def certificate_row(self):
        if 'gain_refinement' in self.data:
            return self.data['gain_refinement']['cases'][0]['strategies']['adaptive']
        return self.data['conditional_certification']['runs'][0]

    def test_current_summary_and_manuscript_tables_agree(self):
        self.assertTrue(SUMMARY.validate(self.data))
        source = (HERE.parent / 'main.tex').read_text()
        self.assertEqual(SUMMARY.synchronise(source, SUMMARY.render(self.data)), source)

    def test_coverage_count_tamper_is_rejected(self):
        self.data['coverage']['records'][0]['expected_gap_covered'] += 1
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_coverage_grid_cannot_omit_or_duplicate_an_observation(self):
        records = self.data['coverage']['records']
        for changed in (records[:-1], records + [copy.deepcopy(records[0])]):
            data = copy.deepcopy(self.data)
            data['coverage']['records'] = changed
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                SUMMARY.validate(data)

    def test_wilson_interval_must_match_the_recorded_count(self):
        self.data['coverage']['records'][0]['expected_gap_coverage_wilson_95'] = [0., 1.]
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_centred_gap_cannot_target_the_zero_martingale_expectation(self):
        row = next(row for row in self.data['coverage']['records'] if row['martingale'] == 'terminal_call')
        row['expected_gap_target'] = self.data['coverage']['records'][0]['expected_gap_target']
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_upper_excess_cannot_be_transcribed_as_an_interval_width(self):
        row = self.data['coverage']['records'][0]
        row['mean_excess_over_expected_gap'] = row['upper_minus_sample_mean']['mean']
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_pairwise_policy_gain_identity_is_required(self):
        self.data['benchmark']['cases'][0]['estimates']['matched_gain']['mean'] += .01
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_non_finite_means_cannot_bypass_the_gain_identity(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            data = copy.deepcopy(self.data)
            data['benchmark']['cases'][0]['estimates']['policy_value']['mean'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                SUMMARY.validate(data)

    def test_unsupported_input_and_guarantee_scope_is_rejected(self):
        unsupported = (
            'floating_point_bounds_are_certified',
            'numerical_oracle_is_rigorous_enclosure',
            'cap_selection_uses_evaluation',
            'simultaneous_claim_across_counts_or_methods',
            'new_general_theorem_claim',
            'new_market_data_acquired',
        )
        for flag in unsupported:
            data = copy.deepcopy(self.data)
            data['coverage'][flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                SUMMARY.validate(data)
        data = copy.deepcopy(self.data)
        data['source_scope'] = 'Subscription observations and fitted market inputs.'
        with self.assertRaises(ValueError):
            SUMMARY.validate(data)

    def test_signed_certificate_width_is_exact_endpoint_difference(self):
        row = self.certificate_row()
        width = row['mean_signed_gap_certificate']['width']
        width['numerator'] = str(int(width['numerator']) + 1)
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_signed_cancellation_must_not_widen_the_absolute_allowance(self):
        row = self.certificate_row()
        row['mean_absolute_width'] = row['mean_signed_width'] / 2.
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_refinement_width_target_and_path_overlap_are_required(self):
        if 'gain_refinement' not in self.data:
            self.skipTest('Final refinement aggregate has not yet been inserted')
        for change in ('width', 'overlap'):
            data = copy.deepcopy(self.data)
            case = data['gain_refinement']['cases'][0]
            if change == 'width':
                case['strategies']['adaptive']['maximum_signed_width'] = 1.
            else:
                case['all_path_certificates_overlap'] = False
            with self.subTest(change=change), self.assertRaises(ValueError):
                SUMMARY.validate(data)

    def test_refinement_comparison_retains_the_frozen_fit(self):
        if 'gain_refinement' not in self.data:
            self.skipTest('Final refinement aggregate has not yet been inserted')
        self.data['gain_refinement']['cases'][0]['frozen_model_policy_values_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_whole_study_budget_charges_preparation_once(self):
        if 'gain_refinement' not in self.data:
            self.skipTest('Final refinement aggregate has not yet been inserted')
        refinement = self.data['gain_refinement']
        refinement['strategy_whole_study_budget_cpu_seconds_including_preparation_once']['adaptive'] += refinement['preparation_cost']['cpu_seconds']
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_numeric_table_tamper_is_rejected_and_repair_is_local(self):
        source = (HERE.parent / 'main.tex').read_text()
        tables = SUMMARY.render(self.data)
        expected = tables['coverage']
        altered = expected.replace('72.48', '72.49', 1)
        self.assertNotEqual(expected, altered)
        changed = source.replace(expected, altered, 1)
        self.assertNotEqual(source, changed)
        with self.assertRaises(ValueError):
            SUMMARY.synchronise(changed, tables)
        self.assertEqual(SUMMARY.synchronise(changed, tables, write=True), source)

    def test_missing_or_repeated_table_marker_is_rejected(self):
        source = (HERE.parent / 'main.tex').read_text()
        marker = '% Begin revision-table:coverage'
        for changed in (source.replace(marker, '', 1), source + marker):
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                SUMMARY.synchronise(changed, SUMMARY.render(self.data), write=True)


class DecisionStudySummaryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE.parent / 'revision_results.json').read_text())
        if 'decision_study' not in self.data:
            self.skipTest('The retained decision-study aggregate has not yet been inserted')
        self.study = self.data['decision_study']

    def reject(self):
        with self.assertRaises(ValueError):
            SUMMARY.validate(self.data)

    def test_decision_summary_validates_and_renders_terminal_call_rows(self):
        self.assertTrue(SUMMARY.validate(self.data))
        table = SUMMARY.render(self.data)['decision']
        self.assertIn('75 allocated bounds', table)
        self.assertIn('include unresolved assessments', table)
        self.assertIn('not rerun in each repetition', table)
        self.assertIn('Charged CPU (ms)', table)
        self.assertEqual(table.count(r'$\theta='), 4)
        self.assertEqual(table.count('Always wait &'), 1)

    def test_decision_grid_cannot_omit_or_duplicate_cells(self):
        original = copy.deepcopy(self.study['decisions'])
        for rows in (original[:-1], original+[copy.deepcopy(original[0])]):
            self.study['decisions'] = rows
            with self.subTest(length=len(rows)):
                self.reject()

    def test_exact_alpha_allocation_cannot_round_above_family_budget(self):
        self.study['design']['fixed_look_error_probability'] = .025/75
        self.reject()

    def test_decision_count_rate_and_wilson_are_linked(self):
        original = copy.deepcopy(self.study['decisions'][0])
        for field, value in (('certifications', 1), ('certification_rate', .5),
                             ('certification_rate_wilson_95', [0., 1.]), ('repetitions', 255)):
            self.study['decisions'][0] = copy.deepcopy(original)
            self.study['decisions'][0][field] = value
            with self.subTest(field=field):
                self.reject()

    def test_reference_classification_uses_certified_rational_endpoints(self):
        row = next(row for row in self.study['decisions'] if row['reference_classification'] == 'above_tolerance')
        row['reference_classification'] = 'within_tolerance'
        self.reject()

    def test_reference_model_and_interval_cannot_be_changed_independently(self):
        original = copy.deepcopy(self.study['reference'])
        self.study['reference']['model']['rate']['numerator'] = str(int(original['model']['rate']['numerator'])+1)
        self.reject()
        self.study['reference'] = original
        width = self.study['reference']['policies'][0]['loss']['width']
        width['numerator'] = str(int(width['numerator'])+1)
        self.reject()

    def test_false_certification_family_requires_actual_repetition_union(self):
        self.study['family_false_certification_repetition_indices'] = [0]
        self.reject()

    def test_false_certification_event_indices_must_be_valid_and_distinct(self):
        row = self.study['decisions'][0]
        for values in ([256], [-1], [True], [0, 0], [1, 0]):
            row['false_certification_repetition_indices'] = values
            with self.subTest(values=values):
                self.reject()

    def test_overlapping_false_events_are_counted_once_in_the_family(self):
        def limits(successes):
            total = 256
            z = NormalDist().inv_cdf(.975)
            rate = successes/total
            scale = 1+z*z/total
            centre = (rate+z*z/(2*total))/scale
            radius = z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/scale
            return [max(0., centre-radius), min(1., centre+radius)]
        rows = [row for row in self.study['decisions'] if row['reference_classification'] == 'above_tolerance'][:2]
        for row in rows:
            row['certifications'] = row['false_certifications'] = 1
            row['certification_rate'] = 1/256
            row['certification_rate_wilson_95'] = limits(1)
            row['false_certification_repetition_indices'] = [7]
            row['mean_one_off_cpu_seconds_given_success'] = row['mean_one_off_cpu_seconds']
        self.study['family_false_certification_repetition_indices'] = [7]
        self.study['family_false_certification_repetitions'] = 1
        self.study['family_false_certification_rate'] = 1/256
        self.study['family_false_certification_rate_wilson_95'] = limits(1)
        self.assertTrue(SUMMARY.validate(self.data))
        self.study['family_false_certification_repetitions'] = 2
        self.reject()

    def test_false_certification_counts_cannot_be_boolean(self):
        self.study['family_false_certification_repetitions'] = False
        self.reject()

    def test_zero_paths_require_a_genuine_deterministic_shortcut(self):
        row = next(row for row in self.study['decisions'] if row['method'] == 'zero')
        row['mean_paths'] = 0.
        self.reject()

    def test_path_means_must_respect_the_full_evaluation_budget(self):
        row = self.study['decisions'][0]
        for value in (-1., 32769., float('nan')):
            row['mean_paths'] = value
            with self.subTest(value=value):
                self.reject()

    def test_charged_cpu_must_include_the_frozen_setup_once(self):
        row = self.study['decisions'][0]
        row['mean_one_off_cpu_seconds'] -= row['setup_cpu_seconds']
        self.reject()

    def test_negative_costs_and_unearned_success_conditioning_are_rejected(self):
        row = next(row for row in self.study['decisions'] if row['certifications'] == 0)
        original = copy.deepcopy(row)
        row['mean_one_off_cpu_seconds_given_success'] = .001
        self.reject()
        row.update(original)
        row['mean_assessment_cpu_seconds'] = -.001
        self.reject()

    def test_finite_terminal_call_caps_need_no_independent_simulation_pilot(self):
        setup = next(row for row in self.study['setup'] if row['method'] == 'terminal_call' and row['policy'] != 'always_wait')
        setup['pilot_paths'] = 8192
        self.reject()

    def test_deterministic_caps_cannot_be_rounded_down(self):
        setup = next(row for row in self.study['setup'] if row['method'] == 'terminal_call' and row['policy'] != 'always_wait')
        setup['caps']['128']['cap'] = math.nextafter(setup['caps']['128']['cap'], -math.inf)
        self.reject()

    def test_floating_gap_diagnostics_cannot_claim_certified_dual_slack(self):
        row = next(row for row in self.study['expected_gap_references'] if row['method'] == 'stock')
        row['expected_gap']['rigorous_enclosure'] = True
        self.reject()
        row['expected_gap']['rigorous_enclosure'] = False
        row['dual_slack_certified'] = True
        self.reject()

    def test_root_free_comparator_cannot_claim_integration_or_omit_costs(self):
        self.study['analytical_reference']['numerical_quadrature_executed'] = True
        self.reject()
        self.study['analytical_reference']['numerical_quadrature_executed'] = False
        row = next(row for row in self.study['deterministic_decisions'] if row['standalone_analytic_cost'] is not None)
        row['standalone_analytic_cost']['cpu_seconds'] = 0.
        self.reject()

    def test_timing_and_certification_scope_cannot_be_overstated(self):
        original = copy.deepcopy(self.study['design'])
        self.study['design']['standalone_cost_observations'] = 'Every setup was independently repeated.'
        self.reject()
        self.study['design'] = original
        self.study['design']['numerical_scope'] = 'All simulation and confidence limits are numerically certified.'
        self.reject()


class ReplicatedRefinementTests(unittest.TestCase):
    """Synthetic arithmetic fixtures only; none are retained study outcomes."""

    def setUp(self):
        from fractions import Fraction
        import hashlib
        self.Fraction = Fraction
        cert = SUMMARY._replicated_interval_record
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        source = {'scripts/checker_fixture.py': sha('synthetic source')}
        design_sha = sha('synthetic design')
        self.design = dict(dimension=5,initial_spot=100.,training_paths=50000,value_paths=50000,
            outer_paths_per_fit=256,precision_bits=96,coarse_conditional_width='1',
            public_preregistration_claim=False,new_general_theorem_claim=False,source_sha256=source,fits=[])
        self.artifacts = {}
        cases = []
        def observations(index, shift=Fraction(0), *, verified=False):
            result = []
            for path in range(256):
                lower = Fraction(10)+Fraction(index,10)+Fraction(path,1000)+shift
                row = dict(path_index=path,stopping_index=path%10,
                    signed_gap_certificate=cert(lower,lower+Fraction(1,1000)))
                if verified:
                    row.update(selected_date_gap_certificate=cert(Fraction(0),Fraction(0)),
                        conditional_integrals=0,integrand_evaluations=0,polynomial_terms_evaluated=1,
                        moment_factor_exponentials=0,moment_factor_cache_builds=0,
                        moment_factor_cache_hits=1,moment_factors_reused=1)
                else:
                    row['total_work'] = dict(conditional_integrals=4,integrand_evaluations=16)
                result.append(row)
            return result
        for index in range(8):
            order = ['cold_uniform','adaptive'] if index%2==0 else ['adaptive','cold_uniform']
            degrees = [2,3] if index%2==0 else [3,2]
            methods = ['polynomial_all_dates','hybrid_exact_terminal'] if index%2==0 else ['hybrid_exact_terminal','polynomial_all_dates']
            self.design['fits'].append(dict(index=index,seed=2026100410+index,directory=f'fit_{index:02d}',
                case=dict(name=f'fit_{index:02d}',target='1/100',order=order),
                polynomial_degree_order=degrees,verified_polynomial_method_order=methods))
            context = dict(fitting_unit_index=index,root_seed=2026100410+index,
                input_sha256=sha('synthetic input '+str(index)),design_sha256=design_sha,source_sha256=source)
            phase_names = ('policy_training','hinge_value_stream_and_targets','hinge_regression',
                           'outer_evaluation_and_record_construction','preparation_bookkeeping')
            primary = dict(**context,name=f'fit_{index:02d}',outer_paths=256,precision_bits=96,
                strategy_order=order,required_maximum_path_gap_width='1/100',
                frozen_model_policy_values_sha256=sha('synthetic fit '+str(index)),
                preparation_cost=dict(cpu_seconds=5.,wall_seconds=5.),
                preparation_phases={name:dict(cpu_seconds=1.,wall_seconds=1.) for name in phase_names},
                all_path_certificates_overlap=True,strategies={},observations={})
            for name in order:
                rows = observations(index)
                aggregate,_ = SUMMARY._replicated_path_summary(rows)
                elapsed = 2. if name=='cold_uniform' else 1.
                aggregate.update(all_targets_met=True,all_signed_inside_absolute=True,mean_absolute_width=.002,
                    conditional_integrals=1024,integrand_evaluations=4096,assessment_cpu_seconds=elapsed,
                    input_and_policy_validation_cpu_seconds=.1,certificate_phase_cpu_seconds=elapsed-.1,
                    combined_preparation_and_assessment_cpu_seconds=5.+elapsed)
                primary['strategies'][name]=aggregate
                primary['observations'][name]=rows
            primary.update(adaptive_over_cold_uniform_integral_ratio=1.,
                           adaptive_over_cold_uniform_certificate_phase_cpu_ratio=.9/1.9)
            cases.append({key:copy.deepcopy(value) for key,value in primary.items() if key!='observations'})
            polynomial = dict(**context,sample_count=256,value_fitting_paths=50000,value_spawn_key=[1],
                frozen_policy_preserved=True,supplied_outer_stream_reproduced=True,no_clipping_of_fitted_values=True,
                numerical_certificate=False,statistical_coverage_certificate=False,
                shared_value_stream_and_target_cpu_seconds=.2,stream_and_policy_reconstruction_cpu_seconds=.1,
                degrees={})
            verified = dict(**context,outer_paths=256,precision_bits=96,
                observed_policy_decisions_checked=True,independent_arm_costs=True,exact_terminal_model_discount=True,
                new_general_theorem_claim=False,path_gap_width_target=cert(Fraction(1,100),Fraction(1,100)),
                input_and_policy_validation_cpu_seconds=.1,verified_method_order=methods,degrees={})
            for degree in degrees:
                exported = dict(polynomial_fit_cpu_seconds=.5,polynomial_interior_fit_cpu_seconds=.4,
                    terminal_polynomial_fit_cpu_seconds=.1,dual_assessment_cpu_seconds=.3,
                    common_policy_and_outer_preparation_cpu_seconds=2.,standalone_polynomial_pricing_cpu_seconds=3.,
                    standalone_polynomial_pricing_plus_validation_cpu_seconds=3.1,hybrid_setup_pricing_cpu_seconds=2.6)
                polynomial['degrees'][str(degree)] = exported
                verified['degrees'][str(degree)] = dict(methods={})
                for name in methods:
                    shift = Fraction(1,2) if degree==2 else -Fraction(1,5)
                    shift += Fraction(index,100)+(Fraction(1,10) if name=='polynomial_all_dates' else Fraction(0))
                    rows = observations(index,shift,verified=True)
                    if name=='hybrid_exact_terminal':
                        for row in rows:
                            row.update(conditional_integrals=1,integrand_evaluations=10)
                    aggregate,_ = SUMMARY._replicated_path_summary(rows)
                    elapsed = .3+index/100
                    setup = 2.6 if name=='hybrid_exact_terminal' else 2.7
                    aggregate.update(method=name,degree=degree,outer_paths=256,all_targets_met=True,
                        conditional_integrals=sum(row['conditional_integrals'] for row in rows),
                        integrand_evaluations=sum(row['integrand_evaluations'] for row in rows),
                        polynomial_terms_evaluated=256,moment_factor_exponentials=0,moment_factor_cache_builds=0,
                        moment_factor_cache_hits=256,moment_factors_reused=256,precision_refinements=0,
                        assessment_cpu_seconds=elapsed,retained_setup_pricing_cpu_seconds=setup,
                        standalone_assessment_cpu_seconds_including_validation=elapsed+.1,
                        standalone_pricing_cpu_seconds=setup+elapsed,
                        standalone_verified_workflow_cpu_seconds=setup+elapsed+.1,observations=rows)
                    verified['degrees'][str(degree)]['methods'][name]=aggregate
            self.artifacts[index]=dict(refinement=primary,polynomial=polynomial,verified_polynomial=verified)
        self.primary = dict(schema_version=1,completed_fits=8,fitting_units=8,outer_paths_per_fit=256,
            design_sha256=design_sha,source_sha256=source,cases=cases,distinct_frozen_model_policy_value_hashes=8,
            established_comparators_executed=False,paired_adaptive_over_cold_uniform_ratios={
                name:SUMMARY._replicated_ratio_summary([value]*8) for name,value in
                (('integrals',1.),('certificate_phase_cpu',.9/1.9),('complete_assessment_cpu',.5),('charged_pricing_budget_cpu',6/7))})
        self.study = self.build()

    def build(self):
        return SUMMARY.build_replicated_refinement(self.primary,fit_artifacts=self.artifacts,design=self.design)

    def reject(self):
        with self.assertRaises(ValueError):
            SUMMARY.validate_replicated_refinement(self.study)

    def test_public_subset_omits_paths_coefficients_and_uses_eight_units(self):
        self.assertTrue(SUMMARY.validate_replicated_refinement(self.study))
        encoded=json.dumps(self.study)
        for prohibited in ('observations','basis_exponents','coefficients','stopping_indices','states'):
            self.assertNotIn('"'+prohibited+'"',encoded)
        self.assertEqual(self.study['paired_comparator_minus_adaptive']['3/hybrid_exact_terminal']['fitting_units'],8)
        self.assertLess(self.study['paired_comparator_minus_adaptive']['3/hybrid_exact_terminal']['mean'],0.)
        self.assertIn('relative expected dual slack',SUMMARY.render_replicated_refinement(self.study))

    def test_tradeoff_transcribes_median_charged_cpu_ranges_and_fit_standard_errors(self):
        import re
        source = SUMMARY.render_replicated_tradeoff(self.study)
        rows = re.findall(r'cpu gap cpu_minus cpu_plus se\n([^\n]+)\n\};', source)
        names = ('hinge/cold_uniform','hinge/adaptive','2/polynomial_all_dates',
                 '2/hybrid_exact_terminal','3/polynomial_all_dates','3/hybrid_exact_terminal')
        self.assertEqual(len(rows),6)
        self.assertEqual(source.count(r'\addlegendentry{'),6)
        self.assertIn('not confidence intervals',source)
        self.assertIn('one across-fit',source)
        self.assertIn('Budgets include policy validation',source)
        self.assertIn('Process start-up and result serialisation are excluded',source)
        self.assertIn('Preparation and assessment CPU (seconds)',source)
        self.assertIn('Mean stopped gap (price units)',source)
        for name,row in zip(names,rows):
            cpu,gap,minus,plus,se = map(float,row.split())
            cost = self.study['standalone_verified_budget_cpu_seconds'][name]
            quality = self.study['gap_midpoint_by_method'][name]
            self.assertEqual(cpu,cost['median'])
            self.assertEqual(gap,quality['mean'])
            self.assertEqual(minus,cost['median']-cost['minimum'])
            self.assertEqual(plus,cost['maximum']-cost['median'])
            self.assertEqual(se,quality['standard_error'])
            self.assertAlmostEqual(cpu-minus,cost['minimum'],places=12)
            self.assertAlmostEqual(cpu+plus,cost['maximum'],places=12)
        # Six colours and distinct marker shapes preserve the method encodings.
        encodings = (('square*','qfblue'),('*','qfgreen'),('triangle*','qforange'),
                     ('diamond*','qfamber'),('triangle*','qfpurple'),('pentagon*','qfviolet'))
        for mark,colour in encodings:
            self.assertIn('mark='+mark+',mark size=2.8pt,color='+colour,source)
        self.assertIn('color=qfpurple,mark options={rotate=180}',source)

    def test_tradeoff_cannot_render_assessment_only_cost_as_a_complete_budget(self):
        self.study['standalone_verified_budget_cpu_seconds']['2/hybrid_exact_terminal']['median'] = .3
        with self.assertRaises(ValueError):
            SUMMARY.render_replicated_tradeoff(self.study)

    def test_tradeoff_marker_rejects_transcription_changes_and_repairs_only_figure(self):
        expected=SUMMARY.render_replicated_tradeoff(self.study)
        start,end='% Begin revision-table:replicated_tradeoff','% End revision-table:replicated_tradeoff'
        source='Before the figure.\n'+start+'\n'+expected+'\n'+end+'\nAfter the figure.\n'
        altered=source.replace('Preparation and assessment CPU (seconds)','Assessment CPU (seconds)',1)
        with self.assertRaises(ValueError):
            SUMMARY.synchronise(altered,{'replicated_tradeoff':expected})
        self.assertEqual(SUMMARY.synchronise(altered,{'replicated_tradeoff':expected},write=True),source)

    def test_missing_and_repeated_fit_ledgers_are_rejected(self):
        del self.artifacts[7]
        with self.assertRaises(ValueError):
            self.build()
        for field,value in (('completed_fits',7),('fitting_units',2048),('outer_paths_per_fit',True)):
            changed=copy.deepcopy(self.study);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_registered_counts_and_balanced_orders_cannot_change(self):
        for container,field,value in (
                ('design','training_paths',49999),('design','value_paths',50001),
                ('case','root_seed',2026100411),('case','strategy_order',['adaptive','cold_uniform']),
                ('case','polynomial_degree_order',[3,2]),
                ('case','verified_method_order',['hybrid_exact_terminal','polynomial_all_dates']),
                ('case','required_maximum_path_gap_width','1/10')):
            changed=copy.deepcopy(self.study)
            target=changed['design'] if container=='design' else changed['cases'][0]
            target[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_new_fits_require_distinct_input_and_frozen_hashes(self):
        for field in ('input_sha256','frozen_model_policy_values_sha256'):
            changed=copy.deepcopy(self.study)
            changed['cases'][1][field]=changed['cases'][0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_actual_accuracy_and_overlap_flags_are_required(self):
        for mutation in ('overlap','width','target'):
            changed=copy.deepcopy(self.study)
            if mutation=='overlap':
                changed['cases'][0]['all_path_certificates_overlap']=False
            elif mutation=='width':
                changed['cases'][0]['comparators']['2']['hybrid_exact_terminal']['maximum_signed_width']=.02
            else:
                changed['cases'][0]['strategies']['adaptive']['all_targets_met']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_ratios_and_ranges_must_reconstruct_from_all_fits(self):
        for field in ('minimum','median','maximum'):
            changed=copy.deepcopy(self.study)
            changed['paired_adaptive_over_cold_uniform_ratios']['integrals'][field]+=.1
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_standalone_cpu_omits_no_phase_and_duplicates_no_preparation(self):
        mutations=(('hinge','combined_preparation_and_assessment_cpu_seconds'),
                   ('hinge','certificate_phase_cpu_seconds'),
                   ('verified','standalone_verified_workflow_cpu_seconds'),
                   ('verified','retained_setup_pricing_cpu_seconds'),
                   ('polynomial','hybrid_setup_pricing_cpu_seconds'))
        for kind,field in mutations:
            changed=copy.deepcopy(self.study);case=changed['cases'][0]
            if kind=='hinge':row=case['strategies']['adaptive']
            elif kind=='verified':row=case['comparators']['2']['hybrid_exact_terminal']
            else:row=case['polynomial_costs']['degrees']['2']
            row[field]+=.1
            with self.subTest(kind=kind,field=field),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)

    def test_paired_difference_is_signed_and_reversing_the_sign_is_rejected(self):
        row=self.study['cases'][0]['paired_gap_differences']['3']['hybrid_exact_terminal']
        self.assertLess(row['approximate_midpoint_mean'],0.)
        row['approximate_midpoint_mean']*=-1
        self.reject()

    def test_between_fit_standard_error_cannot_treat_all_paths_as_fit_units(self):
        row=self.study['paired_comparator_minus_adaptive']['2/hybrid_exact_terminal']
        row['standard_error']/=16
        self.reject()
        self.study=self.build()
        self.study['paired_comparator_minus_adaptive']['2/hybrid_exact_terminal']['fitting_units']=2048
        self.reject()

    def test_exact_mean_certificate_and_display_tampering_are_rejected(self):
        interval=self.study['cases'][0]['paired_gap_differences']['2']['hybrid_exact_terminal']['mean_signed_gap_certificate']
        interval['width']['numerator']=str(int(interval['width']['numerator'])+1)
        self.reject()
        self.study=self.build()
        interval=self.study['gap_midpoint_by_method']['hinge/adaptive']['arithmetic_mean_certificate']
        interval['upper']['approximation']+=1e-10
        self.reject()

    def test_private_ledgers_require_matching_paths_stops_and_identifiers(self):
        original=copy.deepcopy(self.artifacts[0]['verified_polynomial'])
        for mutation in ('path','stop','input','selected_date','counter','midpoint'):
            verified=copy.deepcopy(original)
            row=verified['degrees']['2']['methods']['hybrid_exact_terminal']
            if mutation=='path':row['observations'][0]['path_index']=1
            elif mutation=='stop':row['observations'][0]['stopping_index']=1
            elif mutation=='input':verified['input_sha256']='0'*64
            elif mutation=='selected_date':row['observations'][0]['selected_date_gap_certificate']=SUMMARY._replicated_interval_record(self.Fraction(1),self.Fraction(1))
            elif mutation=='counter':row['conditional_integrals']-=1
            else:row['approximate_midpoint_mean']+=.01
            self.artifacts[0]['verified_polynomial']=verified
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.build()

    def test_private_comparisons_require_pathwise_overlap_not_only_mean_overlap(self):
        row=self.artifacts[0]['refinement']['observations']['cold_uniform'][0]
        cert=SUMMARY._replicated_interval_record
        row['signed_gap_certificate']=cert(self.Fraction(20),self.Fraction(20))
        aggregate,_=SUMMARY._replicated_path_summary(self.artifacts[0]['refinement']['observations']['cold_uniform'])
        self.artifacts[0]['refinement']['strategies']['cold_uniform'].update(aggregate)
        self.primary['cases'][0]['strategies']['cold_uniform'].update(aggregate)
        with self.assertRaises(ValueError):
            self.build()

    def test_non_finite_and_boolean_costs_and_counts_are_rejected(self):
        for value in (float('nan'),float('inf'),True,-1.):
            changed=copy.deepcopy(self.study)
            changed['cases'][0]['comparators']['2']['hybrid_exact_terminal']['assessment_cpu_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                SUMMARY.validate_replicated_refinement(changed)
        self.study['cases'][0]['comparators']['2']['hybrid_exact_terminal']['conditional_integrals']=True
        self.reject()

    def test_scopes_do_not_allow_policy_improvement_or_population_certificates(self):
        self.study['comparison_scope']='A certified improvement in exercise-policy value.'
        self.reject()


class FrozenExtensionSummaryTests(unittest.TestCase):
    """Synthetic arithmetic ledgers; no simulated or retained research outcomes."""

    @classmethod
    def setUpClass(cls):
        from fractions import Fraction
        import hashlib
        import itertools
        cls.Fraction = Fraction
        previous = ReplicatedRefinementTests()
        previous.setUp()
        cls.original = copy.deepcopy(previous.study)
        cert = SUMMARY._replicated_interval_record
        sha = lambda value: hashlib.sha256(value.encode()).hexdigest()
        canonical = lambda value: sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
        source = {'scripts/fixture_extension.py':sha('synthetic extension source')}
        design_sha = sha('synthetic extension design')
        orders = list(itertools.permutations(SUMMARY.FROZEN_EXTENSION_ARMS))+[
            SUMMARY.FROZEN_EXTENSION_ARMS,('adaptive','hybrid_exact_terminal','cold_uniform')]
        cls.design = dict(source_sha256=source,outer_paths_per_fit=1024,precision_bits=96,
            maximum_path_gap_width='1/100',adaptive_coarse_width='1',new_outer_spawn_key=[3],old_outer_spawn_key=[2],
            fitting_spawn_keys=[[0],[1]],training_or_value_streams_rerun=False,recorded_before_new_outer_sampling=True,
            public_preregistration_claim=False,no_new_priority_claim=True,
            source_study=dict(original_design_sha256=cls.original['design_sha256'],
                original_source_sha256=cls.original['source_sha256'],files={}),fits=[])
        cls.artifacts = {}
        cls.primary = dict(completed_fits=8,fitting_units=8,outer_paths_per_fit=1024,total_new_outer_paths=8192,
            fitting_and_value_streams_rerun=False,original_setup_measurements_retained=True,
            design_sha256=design_sha,source_sha256=source,cases=[])
        for index,old in enumerate(cls.original['cases']):
            root = old['root_seed']
            functions = dict(model=dict(fixture_index=index),knots=[[0.,1.]],coefficients=[[index+.1]],frozen_policy=[[1.]])
            frozen_sha = canonical(functions)
            old['frozen_model_policy_values_sha256'] = frozen_sha
            old_input = dict(functions,design=dict(root_seed=root))
            record = copy.deepcopy(old_input)
            record.update(original_input_sha256=old['input_sha256'],states=[[[1.]]]*1024,
                stopping_indices=[path%10 for path in range(1024)])
            record['design'].update(outer_paths=1024,training_paths=50000,value_paths=50000,spawn_keys=[[0],[1],[3]],original_outer_spawn_key=[2],
                new_outer_spawn_key=[3],evaluation_independent_of_original_outer_stream=True,fitting_streams_not_rerun=True)
            cls.design['source_study']['files'][f'fit_{index:02d}/inputs.json'] = old['input_sha256']
            degree = dict(basis_exponents=[[0,0,0,0,0]],coefficients=[[index+.2]])
            old_poly = dict(degrees={'3':copy.deepcopy(degree)})
            degree_sha = canonical(degree)
            view = dict(degrees={'3':copy.deepcopy(degree)},frozen_degree_three_polynomial_sha256=degree_sha,
                sample_count=1024,value_spawn_key=[1],value_fitting_paths=50000)
            cls.design['fits'].append(dict(index=index,root_seed=root,arm_order=list(orders[index])))
            outer_cost = dict(cpu_seconds=.4,wall_seconds=.45)
            hinge_setup = old['preparation_cost']['cpu_seconds']-old['preparation_phases']['outer_evaluation_and_record_construction']['cpu_seconds']+.4
            hybrid_setup = old['polynomial_costs']['degrees']['3']['hybrid_setup_pricing_cpu_seconds']-old['preparation_phases']['outer_evaluation_and_record_construction']['cpu_seconds']+.4
            charges = dict(cold_uniform=hinge_setup,adaptive=hinge_setup,hybrid_exact_terminal=hybrid_setup)
            ledger = dict(fit_index=index,root_seed=root,arm_order=list(orders[index]),input_sha256=sha('synthetic fresh input '+str(index)),
                original_input_sha256=old['input_sha256'],polynomial_view_sha256=sha('synthetic polynomial view '+str(index)),
                design_sha256=design_sha,source_sha256=source,frozen_model_policy_values_sha256=frozen_sha,
                frozen_degree_three_polynomial_sha256=degree_sha,original_outer_spawn_key=[2],new_outer_spawn_key=[3],
                outer_paths=1024,precision_bits=96,target='1/100',hinge_path_certificates_overlap=True,new_outer_preparation_cost=outer_cost,
                charged_setup_cpu_seconds_by_arm=charges,standalone_pricing_cpu_seconds_by_arm={},
                incremental_evaluation_cpu_seconds_by_arm={},arms={})
            for name in orders[index]:
                rows = []
                for path in range(1024):
                    base = Fraction(5)+Fraction(index,10)+Fraction(path,10000)
                    if name=='hybrid_exact_terminal':
                        base += Fraction(index-4,50)+Fraction(path%7,1000)
                    row = dict(path_index=path,stopping_index=path%10,signed_gap_certificate=cert(base,base+Fraction(1,1000)))
                    if name=='hybrid_exact_terminal':
                        row.update(selected_date_gap_certificate=cert(Fraction(0),Fraction(0)),
                            conditional_integrals=1,integrand_evaluations=8,polynomial_terms_evaluated=1,
                            moment_factor_exponentials=0,moment_factor_cache_builds=0,moment_factor_cache_hits=1,moment_factors_reused=1)
                    else:
                        row.update(absolute_gap_certificate=cert(base-Fraction(1,1000),base+Fraction(1,500)),
                            total_work=dict(conditional_integrals=4,integrand_evaluations=16))
                    rows.append(row)
                derived,_ = SUMMARY._replicated_path_summary(rows,expected_count=1024)
                elapsed = dict(cold_uniform=8.,adaptive=4.,hybrid_exact_terminal=1.)[name]+index/100
                derived.update(observations=rows,all_targets_met=True,
                    conditional_integrals=sum(row['conditional_integrals'] if name=='hybrid_exact_terminal' else row['total_work']['conditional_integrals'] for row in rows),
                    integrand_evaluations=sum(row['integrand_evaluations'] if name=='hybrid_exact_terminal' else row['total_work']['integrand_evaluations'] for row in rows),
                    assessment_cpu_seconds=elapsed,input_and_policy_validation_cpu_seconds=.1,
                    certificate_phase_cpu_seconds=elapsed-.1)
                if name=='hybrid_exact_terminal':
                    derived.update(polynomial_terms_evaluated=1024,moment_factor_exponentials=0,moment_factor_cache_builds=0,
                                   moment_factor_cache_hits=1024,moment_factors_reused=1024)
                ledger['arms'][name] = derived
                ledger['standalone_pricing_cpu_seconds_by_arm'][name] = charges[name]+elapsed
                ledger['incremental_evaluation_cpu_seconds_by_arm'][name] = .4+elapsed
            paired_rows = []
            for path,(hybrid,adaptive) in enumerate(zip(ledger['arms']['hybrid_exact_terminal']['observations'],ledger['arms']['adaptive']['observations'])):
                hlo,hhi = SUMMARY._replicated_interval(hybrid['signed_gap_certificate'])
                alo,ahi = SUMMARY._replicated_interval(adaptive['signed_gap_certificate'])
                paired_rows.append(dict(path_index=path,stopping_index=path%10,signed_gap_certificate=cert(hlo-ahi,hhi-alo)))
            paired,_ = SUMMARY._replicated_path_summary(paired_rows,signed=True,expected_count=1024)
            ledger['paired_hybrid_minus_adaptive'] = dict(
                mean_difference_certificate=paired['mean_signed_gap_certificate'],
                approximate_midpoint_mean=paired['approximate_midpoint_mean'],
                approximate_midpoint_mean_standard_error=paired['approximate_midpoint_mean_standard_error'],
                midpoint_mean_numerical_error_bound=paired['midpoint_mean_numerical_error_bound'],
                maximum_observation_width=paired['maximum_signed_width'],observations=[dict(
                    path_index=row['path_index'],stopping_index=row['stopping_index'],difference_certificate=row['signed_gap_certificate']) for row in paired_rows])
            compact = copy.deepcopy(ledger)
            for arm in compact['arms'].values():
                arm.pop('observations')
            compact['paired_hybrid_minus_adaptive'].pop('observations')
            cls.primary['cases'].append(compact)
            cls.artifacts[index] = dict(result=ledger,input=record,polynomial_view=view,
                original_input=old_input,original_polynomial=old_poly)
        import statistics
        def aggregate(bounds):
            lower=sum(lo for lo,hi in bounds)/8
            upper=sum(hi for lo,hi in bounds)/8
            mids=[float((lo+hi)/2) for lo,hi in bounds]
            return dict(mean_certificate=cert(lower,upper),approximate_midpoint_mean=statistics.mean(mids),
                approximate_midpoint_mean_standard_error=statistics.stdev(mids)/math.sqrt(8),
                midpoint_mean_numerical_error_bound=float((upper-lower)/2),
                maximum_observation_width=float(max(hi-lo for lo,hi in bounds)))
        cls.primary['exact_across_fit_gap_summaries']={name:aggregate([
            SUMMARY._replicated_interval(cls.artifacts[index]['result']['arms'][name]['mean_signed_gap_certificate'])
            for index in range(8)]) for name in SUMMARY.FROZEN_EXTENSION_ARMS}
        cls.primary['exact_across_fit_paired_difference']=aggregate([
            SUMMARY._replicated_interval(cls.artifacts[index]['result']['paired_hybrid_minus_adaptive']['mean_difference_certificate'],signed=True)
            for index in range(8)])
        cls.primary['conditional_mean_gap_standard_error_by_arm']={name:math.sqrt(sum(
            cls.artifacts[index]['result']['arms'][name]['approximate_midpoint_mean_standard_error']**2
            for index in range(8)))/8 for name in SUMMARY.FROZEN_EXTENSION_ARMS}
        cls.primary['conditional_mean_paired_difference_standard_error']=math.sqrt(sum(
            cls.artifacts[index]['result']['paired_hybrid_minus_adaptive']['approximate_midpoint_mean_standard_error']**2
            for index in range(8)))/8
        ratios={}
        for label,key in (('complete_assessment_cpu',None),('standalone_pricing_cpu','standalone_pricing_cpu_seconds_by_arm'),
                          ('incremental_evaluation_cpu','incremental_evaluation_cpu_seconds_by_arm')):
            values=[]
            for index in range(8):
                ledger=cls.artifacts[index]['result']
                values.append(ledger[key]['adaptive']/ledger[key]['cold_uniform'] if key else
                    ledger['arms']['adaptive']['assessment_cpu_seconds']/ledger['arms']['cold_uniform']['assessment_cpu_seconds'])
            ratios[label]=dict(fitting_units=8,minimum=min(values),median=statistics.median(values),maximum=max(values))
        ratios['conditional_integrals']=dict(fitting_units=8,minimum=1.,median=1.,maximum=1.)
        cls.primary['paired_adaptive_over_cold_uniform_ratios']=ratios
        cls.primary['new_outer_preparation_cost_by_fit']=[cls.artifacts[index]['result']['new_outer_preparation_cost'] for index in range(8)]
        cls.base = cls.build_static(cls.primary,cls.artifacts,cls.design,cls.original)

    @staticmethod
    def build_static(primary,artifacts,design,original):
        return SUMMARY.build_frozen_extension(primary,fit_artifacts=artifacts,design=design,original_replicated_refinement=original)

    def setUp(self):
        self.study = copy.deepcopy(self.base)

    def reject(self):
        with self.assertRaises(ValueError):
            SUMMARY.validate_frozen_extension(self.study,self.original)

    def test_eight_retained_fits_and_conditional_average_precision_are_distinct(self):
        self.assertTrue(SUMMARY.validate_frozen_extension(self.study,self.original))
        paired = self.study['paired_hybrid_minus_adaptive']
        expected = math.sqrt(sum(case['paired_hybrid_minus_adaptive']['approximate_midpoint_mean_standard_error']**2 for case in self.study['cases']))/8
        self.assertEqual(paired['conditional_average_standard_error'],expected)
        self.assertNotAlmostEqual(paired['conditional_average_standard_error'],paired['standard_error'])
        self.assertEqual(self.study['newly_trained_fits'],0)
        self.assertEqual(paired['fitting_units'],8)
        encoded = json.dumps(self.study)
        for excluded in ('observations','coefficients','basis_exponents','states','stopping_indices'):
            self.assertNotIn('"'+excluded+'"',encoded)

    def test_all_three_arms_render_new_and_reconstructed_costs(self):
        source = SUMMARY.render_frozen_extension(self.study)
        self.assertEqual(source.count('Hinge,'),2)
        self.assertIn('1,024 fresh',source)
        self.assertIn('not rerun',source)
        self.assertIn('descriptive',source)
        self.assertIn('Conditional SE',source)
        self.assertIn('new outer preparation and the complete assessment',source)
        self.assertIn('Standalone CPU reconstructs a budget',source)
        new_cpu=self.study['incremental_evaluation_cpu_seconds']['adaptive']['median']
        assessment_cpu=self.study['incremental_assessment_cpu_seconds']['adaptive']['median']
        self.assertGreater(new_cpu,assessment_cpu)
        self.assertIn(f"{new_cpu:.3f}",source)

    def test_fresh_stream_cannot_be_old_stream_or_prefix(self):
        for key,value in (('new_outer_spawn_key',[2]),('fresh_sample_is_prefix_of_original',True),
                          ('newly_trained_fits',8),('outer_paths_per_fit',256),('fitting_units',8192)):
            changed=copy.deepcopy(self.study);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                SUMMARY.validate_frozen_extension(changed,self.original)

    def test_retained_unit_matching_requires_original_input_and_frozen_identity(self):
        for key in ('root_seed','original_input_sha256','frozen_model_policy_values_sha256'):
            changed=copy.deepcopy(self.study)
            changed['cases'][0][key]=changed['cases'][1][key]
            with self.subTest(key=key),self.assertRaises(ValueError):
                SUMMARY.validate_frozen_extension(changed,self.original)

    def test_reconstructed_budget_excludes_old_outer_and_adds_new_outer_once(self):
        self.study['cases'][0]['charged_setup_cpu_seconds_by_arm']['adaptive'] += 1.
        self.reject()
        self.study=copy.deepcopy(self.base)
        self.study['cases'][0]['reconstructed_standalone_cpu_seconds_by_arm']['hybrid_exact_terminal'] += .1
        self.reject()

    def test_fresh_evaluation_must_charge_outer_preparation_once(self):
        self.study['cases'][0]['incremental_evaluation_cpu_seconds_by_arm']['adaptive'] -= .4
        self.reject()
        self.study=copy.deepcopy(self.base)
        self.study['incremental_evaluation_cpu_seconds']['hybrid_exact_terminal']['median'] += .4
        self.reject()

    def test_validation_cannot_be_excluded_or_double_counted(self):
        for field in ('incremental_assessment_cpu_seconds_including_validation','certificate_phase_cpu_seconds'):
            changed=copy.deepcopy(self.study)
            changed['cases'][0]['arms']['hybrid_exact_terminal'][field] += .1
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_frozen_extension(changed,self.original)

    def test_conditional_average_standard_error_is_not_fit_dispersion(self):
        self.study['paired_hybrid_minus_adaptive']['conditional_average_standard_error'] = self.study['paired_hybrid_minus_adaptive']['standard_error']
        self.reject()

    def test_descriptive_dispersion_cannot_pool_8192_training_units(self):
        self.study['gap_midpoint_by_arm']['adaptive']['fitting_units'] = 8192
        self.reject()

    def test_width_counts_and_numeric_identity_tampering_are_rejected(self):
        for field,value in (('maximum_signed_width',.011),('conditional_integrals',True),
                            ('incremental_assessment_cpu_seconds_including_validation',float('inf'))):
            changed=copy.deepcopy(self.study)
            changed['cases'][0]['arms']['adaptive'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_frozen_extension(changed,self.original)
        width=self.study['paired_hybrid_minus_adaptive']['arithmetic_mean_certificate']['width']
        width['numerator']=str(int(width['numerator'])+1)
        self.reject()

    def test_balanced_orders_and_each_executed_arm_are_required(self):
        self.study['cases'][0]['arm_order']=['adaptive','cold_uniform','hybrid_exact_terminal']
        self.reject()
        self.study=copy.deepcopy(self.base)
        del self.study['cases'][0]['arms']['hybrid_exact_terminal']
        self.reject()

    def test_different_martingales_need_not_have_overlapping_gap_means(self):
        first = self.study['cases'][0]
        a = SUMMARY._replicated_interval(first['arms']['adaptive']['mean_signed_gap_certificate'])
        h = SUMMARY._replicated_interval(first['arms']['hybrid_exact_terminal']['mean_signed_gap_certificate'])
        self.assertLess(h[1],a[0])
        self.assertTrue(SUMMARY.validate_frozen_extension(self.study,self.original))

    def test_full_ledgers_reject_refitted_functions_and_changed_polynomial_coefficients(self):
        for part,key in (('input','coefficients'),('polynomial_view','degrees')):
            artifacts=copy.deepcopy(self.artifacts)
            if part=='input':
                artifacts[0][part][key][0][0] += .01
            else:
                artifacts[0][part][key]['3']['coefficients'][0][0] += .01
            with self.subTest(part=part),self.assertRaises(ValueError):
                self.build_static(self.primary,artifacts,self.design,self.original)

    def test_full_ledger_rejects_missing_paths_and_changed_selected_date_gap(self):
        for mutation in ('path','selected','counter','difference','paired_standard_error'):
            artifacts=copy.deepcopy(self.artifacts)
            ledger=artifacts[0]['result'];arm=ledger['arms']['hybrid_exact_terminal']
            if mutation=='path':
                arm['observations'].pop()
            elif mutation=='selected':
                arm['observations'][0]['selected_date_gap_certificate'] = SUMMARY._replicated_interval_record(self.Fraction(1),self.Fraction(1))
            elif mutation=='counter':
                arm['moment_factors_reused'] -= 1
            elif mutation=='difference':
                ledger['paired_hybrid_minus_adaptive']['observations'][0]['difference_certificate'] = SUMMARY._replicated_interval_record(self.Fraction(1),self.Fraction(1))
            else:
                ledger['paired_hybrid_minus_adaptive']['approximate_midpoint_mean_standard_error'] += .01
            primary=copy.deepcopy(self.primary)
            compact=copy.deepcopy(ledger)
            for row in compact['arms'].values():
                row.pop('observations')
            compact['paired_hybrid_minus_adaptive'].pop('observations')
            primary['cases'][0] = compact
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                self.build_static(primary,artifacts,self.design,self.original)

    def test_full_ledger_requires_pathwise_overlap_not_only_aggregate_overlap(self):
        artifacts=copy.deepcopy(self.artifacts)
        row=artifacts[0]['result']['arms']['cold_uniform']['observations'][0]
        row['signed_gap_certificate'] = SUMMARY._replicated_interval_record(self.Fraction(20),self.Fraction(20))
        row['absolute_gap_certificate'] = row['signed_gap_certificate']
        arm=artifacts[0]['result']['arms']['cold_uniform']
        derived,_=SUMMARY._replicated_path_summary(arm['observations'],expected_count=1024)
        arm.update(derived)
        primary=copy.deepcopy(self.primary)
        primary['cases'][0]['arms']['cold_uniform'].update(derived)
        with self.assertRaises(ValueError):
            self.build_static(primary,artifacts,self.design,self.original)

    def test_coverage_or_new_training_claim_is_rejected(self):
        for field,value in (('statistical_coverage_certificate',True),('sampling_scope','Eight new independent training replications.')):
            changed=copy.deepcopy(self.study);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_frozen_extension(changed,self.original)


class UniformAllocationSummaryTests(unittest.TestCase):
    """Synthetic exact-arithmetic and cost fixtures, without research outcomes."""

    @classmethod
    def setUpClass(cls):
        from fractions import Fraction
        import hashlib
        import itertools
        import statistics
        if not hasattr(FrozenExtensionSummaryTests,'base'):
            FrozenExtensionSummaryTests.setUpClass()
        cls.Fraction=Fraction
        cls.prior=copy.deepcopy(FrozenExtensionSummaryTests.base)
        cert=SUMMARY._replicated_interval_record
        sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
        canonical=lambda value:sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
        fingerprint=sha(json.dumps([['0','1']]*20,separators=(',',':')))
        source={'scripts/fixture_common_budget.py':sha('synthetic common-budget source')}
        design_sha=sha('synthetic common-budget design')
        orders=list(itertools.permutations(SUMMARY.UNIFORM_ALLOCATION_ARMS))+[
            SUMMARY.UNIFORM_ALLOCATION_ARMS,('coarse_common','adaptive','cold_common')]
        cls.design=dict(source_sha256=source,fitting_units=8,outer_paths_per_fit=1024,precision_bits=96,target='1/100',
            initial_common_widths={'cold_common':'1/100','coarse_common':'1'},common_tolerance_factor='1/4',maximum_global_rounds=16,
            new_simulation_or_fitting=False,recorded_before_new_assessment=True,priority_claim=False,public_preregistration_claim=False,
            exact_replay_scope=dict(adaptive_numerical_replay=True,excluded_timing_fields=sorted(SUMMARY.UNIFORM_ALLOCATION_TIMER_FIELDS)),
            source_evidence=dict(frozen_extension_design_sha256=cls.prior['design_sha256'],
                frozen_extension_source_sha256=cls.prior['source_sha256'],frozen_extension_files={}),fits=[])
        cls.artifacts={};cls.primary=dict(completed_fits=8,retained_fits=8,outer_paths_per_fit=1024,total_retained_paths=8192,
            adaptive_exact_numerical_replay=True,all_path_certificates_intersect=True,design_sha256=design_sha,source_sha256=source,cases=[])
        zero=cert(Fraction(0),Fraction(0))
        for index,old in enumerate(cls.prior['cases']):
            root=old['root_seed']
            policy=[[float(index+1)]+[0.]*20 for _ in range(9)]
            record=dict(model=dict(exercise_times=list(map(float,range(10))),rate=0.,strike=100.,spots=[100.]*5),
                knots=[80.,100.,120.,140.],coefficients=[[0.]*10 for _ in range(10)],frozen_policy=policy,
                states=[[[100.]*5 for _ in range(9)]+[[105.]*5]]*1024,stopping_indices=[9]*1024,
                design=dict(root_seed=root,spawn_keys=[[0],[1],[3]],outer_paths=1024),
                preparation_cost=dict(cpu_seconds=old['charged_setup_cpu_seconds_by_arm']['adaptive'],wall_seconds=5.))
            snapshot=canonical({key:record[key] for key in ('model','knots','coefficients','frozen_policy')})
            old['frozen_model_policy_values_sha256']=snapshot
            adaptive_rows=[dict(path_index=path,stopping_index=9,signed_gap_certificate=copy.deepcopy(zero),
                absolute_gap_certificate=cert(Fraction(0),Fraction(9,10)),frozen_centres_and_penalty_sha256=fingerprint,
                total_work=dict(conditional_integrals=4,integrand_evaluations=16),coarse_cpu_seconds=.001,
                refinement_history=[]) for path in range(1024)]
            adaptive,_=SUMMARY._replicated_path_summary(adaptive_rows,expected_count=1024)
            adaptive.update(all_targets_met=True,all_signed_inside_absolute=True,conditional_integrals=4096,integrand_evaluations=16384,
                assessment_cpu_seconds=2.+index/100,input_and_policy_validation_cpu_seconds=.1,
                certificate_phase_cpu_seconds=1.9+index/100,observations=adaptive_rows)
            for name in ('adaptive','cold_uniform'):
                old['arms'][name].update({key:value for key,value in adaptive.items() if key in (
                    'mean_signed_gap_certificate','approximate_midpoint_mean','approximate_midpoint_mean_standard_error',
                    'mean_signed_width','maximum_signed_width','midpoint_mean_numerical_error_bound')})
            hybrid=old['arms']['hybrid_exact_terminal']
            old['paired_hybrid_minus_adaptive']=dict(
                mean_difference_certificate=copy.deepcopy(hybrid['mean_signed_gap_certificate']),
                approximate_midpoint_mean=hybrid['approximate_midpoint_mean'],
                approximate_midpoint_mean_standard_error=hybrid['approximate_midpoint_mean_standard_error'],
                mean_signed_width=hybrid['mean_signed_width'],maximum_signed_width=hybrid['maximum_signed_width'],
                midpoint_mean_numerical_error_bound=hybrid['midpoint_mean_numerical_error_bound'])
            retained=dict(input_sha256=old['input_sha256'],frozen_model_policy_values_sha256=snapshot,
                arms=dict(adaptive=copy.deepcopy(adaptive),cold_uniform=copy.deepcopy(adaptive)))
            result_hash=sha('synthetic retained result '+str(index))
            cls.design['source_evidence']['frozen_extension_files'][f'fit_{index:02d}/inputs.json']=old['input_sha256']
            cls.design['source_evidence']['frozen_extension_files'][f'fit_{index:02d}/result.json']=result_hash
            cls.design['fits'].append(dict(index=index,root_seed=root,arm_order=list(orders[index])))
            ledger=dict(fit_index=index,root_seed=root,arm_order=list(orders[index]),design_sha256=design_sha,source_sha256=source,
                retained_input_sha256=old['input_sha256'],retained_result_sha256=result_hash,outer_paths=1024,
                adaptive_exact_numerical_replay=True,all_path_certificates_intersect=True,coarse_centres_match_adaptive=True,
                retained_preparation_cost=copy.deepcopy(record['preparation_cost']),arms={},assessment_only_cpu_seconds_by_arm={},
                standalone_pricing_cpu_seconds_by_arm={})
            for name in orders[index]:
                if name=='adaptive':
                    arm=copy.deepcopy(adaptive)
                    arm['assessment_cpu_seconds']+=.5
                    arm['certificate_phase_cpu_seconds']+=.5
                else:
                    theta=Fraction(1,100) if name=='cold_common' else Fraction(1)
                    delta=Fraction(1,250) if name=='cold_common' else Fraction(1,10)
                    initial=[dict(increment=j,requested_width_certificate=cert(theta,theta),
                        returned_certificate=cert(-delta,delta),work=dict(integrals=1,integrand_evaluations=4),
                        complete_query_cost=dict(cpu_seconds=.001,wall_seconds=.001)) for j in range(1,10)]
                    rewards=[copy.deepcopy(zero) for _ in range(9)]+[cert(Fraction(5),Fraction(5))]
                    rows=[dict(path_index=path,stopping_index=9,signed_gap_certificate=copy.deepcopy(zero),
                        absolute_gap_certificate=cert(Fraction(0),9*delta),frozen_centres_and_penalty_sha256=fingerprint,
                        initial_common_width_certificate=cert(theta,theta),reward_certificates=copy.deepcopy(rewards),
                        reward_difference_width_allowance_certificate=copy.deepcopy(zero),maximum_stopping_span_length=9,
                        initial_queries=copy.deepcopy(initial),global_rounds=[],coarse_work=dict(conditional_integrals=9,integrand_evaluations=36),
                        refinement_work=dict(conditional_integrals=0,integrand_evaluations=0),
                        total_work=dict(conditional_integrals=9,integrand_evaluations=36),
                        complete_path_cost=dict(cpu_seconds=.009,wall_seconds=.009)) for path in range(1024)]
                    arm,_=SUMMARY._replicated_path_summary(rows,expected_count=1024)
                    elapsed=(4. if name=='cold_common' else 5.)+index/100
                    arm.update(observations=rows,all_targets_met=True,all_signed_inside_absolute=True,
                        conditional_integrals=9216,integrand_evaluations=36864,initialisation_integrals=9216,
                        global_refinement_integrals=0,global_refinement_rounds=0,assessment_cpu_seconds=elapsed,
                        input_and_policy_validation_cpu_seconds=.1,certificate_phase_cpu_seconds=elapsed-.1)
                ledger['arms'][name]=arm
                ledger['assessment_only_cpu_seconds_by_arm'][name]=arm['assessment_cpu_seconds']
                ledger['standalone_pricing_cpu_seconds_by_arm'][name]=record['preparation_cost']['cpu_seconds']+arm['assessment_cpu_seconds']
            cls.artifacts[index]=dict(result=ledger,input=record,retained_result=retained)
            compact=copy.deepcopy(ledger)
            for arm in compact['arms'].values():arm.pop('observations')
            cls.primary['cases'].append(compact)
        cls.prior.update(SUMMARY._frozen_extension_aggregates(cls.prior['cases']))
        cls.primary['arm_summaries']={}
        for name in SUMMARY.UNIFORM_ALLOCATION_ARMS:
            rows=[cls.artifacts[index]['result']['arms'][name] for index in range(8)]
            costs=[row['assessment_cpu_seconds'] for row in rows]
            cls.primary['arm_summaries'][name]=dict(mean_gap_certificate=copy.deepcopy(zero),arithmetic_mean_gap=0.,
                conditional_integrals=sum(row['conditional_integrals'] for row in rows),
                integrand_evaluations=sum(row['integrand_evaluations'] for row in rows),maximum_signed_width=0.,
                assessment_cpu_seconds=dict(minimum=min(costs),arithmetic_mean=statistics.mean(costs),median=statistics.median(costs),maximum=max(costs)))
        cls.primary['adaptive_over_common_ratios']={}
        for control in ('cold_common','coarse_common'):
            cls.primary['adaptive_over_common_ratios'][control]={}
            for label,field in (('assessment_only_cpu','assessment_only_cpu_seconds_by_arm'),('standalone_pricing_cpu','standalone_pricing_cpu_seconds_by_arm')):
                values=[cls.artifacts[index]['result'][field]['adaptive']/cls.artifacts[index]['result'][field][control] for index in range(8)]
                cls.primary['adaptive_over_common_ratios'][control][label]=dict(fitting_units=8,minimum=min(values),median=statistics.median(values),maximum=max(values))
            cls.primary['adaptive_over_common_ratios'][control]['conditional_integrals']=dict(fitting_units=8,minimum=4/9,median=4/9,maximum=4/9)
        cls.base=cls.build_static(cls.primary,cls.artifacts,cls.design,cls.prior)

    @staticmethod
    def build_static(primary,artifacts,design,prior):
        return SUMMARY.build_uniform_allocation(primary,fit_artifacts=artifacts,design=design,frozen_extension=prior)

    def setUp(self):self.study=copy.deepcopy(self.base)

    def reject(self):
        with self.assertRaises(ValueError):SUMMARY.validate_uniform_allocation(self.study,self.prior)

    def test_compact_record_keeps_fixed_inputs_and_exact_maximum_width(self):
        self.assertTrue(SUMMARY.validate_uniform_allocation(self.study,self.prior))
        self.assertEqual(self.study['new_outer_paths'],0)
        self.assertEqual(self.study['newly_trained_fits'],0)
        self.assertEqual(self.study['assessment_by_arm']['cold_common']['total_outer_paths'],8192)
        for forbidden in ('observations','coefficients','states','initial_queries','global_rounds'):
            self.assertNotIn('"'+forbidden+'"',json.dumps(self.study))

    def test_table_distinguishes_totals_maxima_and_median_costs(self):
        rendered=SUMMARY.render_uniform_allocation(self.study)
        self.assertIn('total all eight fits',rendered)
        self.assertIn('largest actual width across 8,192 paths',rendered)
        self.assertIn('medians over eight fits',rendered)
        self.assertIn('without rerunning simulation or fitting',rendered)
        self.assertIn('fuller query traces',rendered)
        self.assertIn('does not isolate allocation-rule runtime',rendered)
        for name in SUMMARY.UNIFORM_ALLOCATION_ARMS:
            row=self.study['assessment_by_arm'][name]
            self.assertIn(f"{row['total_conditional_integrals']:,}",rendered)
            self.assertIn(f"{row['median_complete_assessment_cpu_seconds']:.3f}",rendered)

    def test_no_new_sample_or_fitting_scope_can_be_claimed(self):
        for field,value in (('new_outer_paths',8192),('newly_trained_fits',8),('new_sampling_executed',True),
                            ('statistical_coverage_certificate',True),('outer_paths_per_fit',256)):
            changed=copy.deepcopy(self.study);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):SUMMARY.validate_uniform_allocation(changed,self.prior)

    def test_new_assessment_and_retained_setup_must_be_charged_separately(self):
        self.study['cases'][0]['arms']['adaptive']['assessment_cpu_seconds']+=.1
        self.reject()
        self.study=copy.deepcopy(self.base)
        self.study['cases'][0]['reconstructed_standalone_cpu_seconds_by_arm']['cold_common']+=.1
        self.reject()

    def test_exact_width_and_work_totals_cannot_be_changed(self):
        for field,value in (('maximum_signed_width_exact',dict(numerator='2',denominator='100',approximation=.02)),
                            ('conditional_integrals',True)):
            changed=copy.deepcopy(self.study);changed['cases'][0]['arms']['cold_common'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):SUMMARY.validate_uniform_allocation(changed,self.prior)
        self.study['assessment_by_arm']['cold_common']['total_conditional_integrals']-=1
        self.reject()
        for changed in (dict(numerator='1',denominator='1000',approximation=.001),
                        dict(numerator='0',denominator='1',approximation=.001),
                        dict(numerator='0',denominator='1',approximation=0.,extra='invalid')):
            self.study=copy.deepcopy(self.base)
            self.study['assessment_by_arm']['cold_common']['maximum_signed_width_exact']=changed
            with self.subTest(aggregate_rational=changed):self.reject()

    def test_timer_allowlist_cannot_hide_changed_scientific_fields(self):
        self.study['scientific_replay_excluded_timer_fields'].append('conditional_integrals')
        self.reject()
        original=dict(conditional_integrals=10,assessment_cpu_seconds=1.,observations=[dict(coarse_cpu_seconds=1.,stopping_index=3)])
        same=copy.deepcopy(original);same['assessment_cpu_seconds']=2.;same['observations'][0]['coarse_cpu_seconds']=4.
        self.assertEqual(SUMMARY._uniform_without_timers(original),SUMMARY._uniform_without_timers(same))
        same['observations'][0]['stopping_index']=4
        self.assertNotEqual(SUMMARY._uniform_without_timers(original),SUMMARY._uniform_without_timers(same))

    def test_order_and_retained_input_changes_are_rejected(self):
        self.study['cases'][0]['arm_order']=['adaptive','coarse_common','cold_common']
        self.reject()
        self.study=copy.deepcopy(self.base)
        self.study['cases'][0]['input_sha256']=self.study['cases'][1]['input_sha256']
        self.reject()

    def test_direct_span_gain_cancels_selected_date_and_shared_increments(self):
        F=self.Fraction
        rewards=[(F(0),F(0)),(F(6),F(6)),(F(7),F(7))]
        values=[F(0),F(4),F(3)]
        brackets=[(F(0),F(0)),(F(4),F(4)),(F(2),F(2))]
        self.assertEqual(SUMMARY._uniform_direct_gain(rewards,values,brackets,1),(F(0),F(0)))
        self.assertEqual(SUMMARY._uniform_direct_gain(rewards,values,brackets,2),(F(0),F(0)))
        self.assertEqual(SUMMARY._uniform_direct_gain(rewards,values,brackets,0),(F(6),F(6)))

    def common_round_fixture(self):
        import hashlib
        F=self.Fraction
        cert=SUMMARY._replicated_interval_record
        old=cert(-F(1,250),F(1,250))
        tight=cert(-F(1,10000),F(1,10000))
        initial=[dict(increment=date,requested_width_certificate=cert(F(1,100),F(1,100)),
            returned_certificate=copy.deepcopy(old),work=dict(integrals=1,integrand_evaluations=4),
            complete_query_cost=dict(cpu_seconds=.001,wall_seconds=.001)) for date in range(1,10)]
        queries=[dict(increment=date,requested_width_certificate=cert(F(1,400),F(1,400)),
            previous_certificate=copy.deepcopy(old),returned_certificate=copy.deepcopy(tight),
            retained_certificate=copy.deepcopy(tight),work=dict(integrals=1,integrand_evaluations=4),
            complete_query_cost=dict(cpu_seconds=.001,wall_seconds=.001)) for date in range(1,10)]
        gap=cert(F(49991,10000),F(50009,10000))
        fingerprint=hashlib.sha256(json.dumps([['0','1']]*20,separators=(',',':')).encode()).hexdigest()
        row=dict(path_index=0,stopping_index=0,initial_common_width_certificate=cert(F(1,100),F(1,100)),
            reward_certificates=[cert(F(0),F(0)) for _ in range(9)]+[cert(F(5),F(5))],
            initial_queries=initial,frozen_centres_and_penalty_sha256=fingerprint,
            reward_difference_width_allowance_certificate=cert(F(0),F(0)),maximum_stopping_span_length=9,
            global_rounds=[dict(round_index=1,eligible_increments=list(range(1,10)),queries=queries,completed=True,
                all_retained_conditional_widths_comply=True,common_requested_width_certificate=cert(F(1,400),F(1,400)),
                gap_certificate_after_round=copy.deepcopy(gap))],signed_gap_certificate=copy.deepcopy(gap),
            absolute_gap_certificate=copy.deepcopy(gap),coarse_work=dict(conditional_integrals=9,integrand_evaluations=36),
            refinement_work=dict(conditional_integrals=9,integrand_evaluations=36),
            total_work=dict(conditional_integrals=18,integrand_evaluations=72),complete_path_cost=dict(cpu_seconds=.02,wall_seconds=.02))
        return row,self.artifacts[0]['input'],[F(0)]*10

    def test_complete_common_round_is_reconstructed_from_direct_span_algebra(self):
        row,record,values=self.common_round_fixture()
        self.assertTrue(SUMMARY._uniform_check_trace(row,record,'cold_common',values))
        expected=(self.Fraction(49991,10000),self.Fraction(50009,10000))
        self.assertEqual(SUMMARY._replicated_interval(row['signed_gap_certificate']),expected)

    def test_common_round_must_cover_every_eligible_date_and_terminate_at_actual_width(self):
        for mutation in ('gap','missing_date','extra_round','intersection','absolute'):
            row,record,values=self.common_round_fixture()
            if mutation=='gap':
                row['global_rounds'][0]['gap_certificate_after_round']=SUMMARY._replicated_interval_record(self.Fraction(5),self.Fraction(5))
            elif mutation=='missing_date':row['global_rounds'][0]['queries'].pop()
            elif mutation=='extra_round':row['global_rounds'].append(copy.deepcopy(row['global_rounds'][0]))
            elif mutation=='intersection':row['global_rounds'][0]['queries'][0]['retained_certificate']=SUMMARY._replicated_interval_record(self.Fraction(0),self.Fraction(0))
            else:row['absolute_gap_certificate']=SUMMARY._replicated_interval_record(self.Fraction(5),self.Fraction(5))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):SUMMARY._uniform_check_trace(row,record,'cold_common',values)

    def test_full_adaptive_replay_rejects_changed_certificates_or_work(self):
        artifacts=dict(self.artifacts)
        artifacts[0]=copy.deepcopy(self.artifacts[0])
        arm=artifacts[0]['result']['arms']['adaptive']
        arm['observations'][0]['total_work']['conditional_integrals']+=1
        primary=self.refreshed_primary(artifacts)
        with self.assertRaises(ValueError):self.build_static(primary,artifacts,self.design,self.prior)

    def refreshed_primary(self,artifacts):
        primary=copy.deepcopy(self.primary)
        for index in range(8):
            ledger=artifacts[index]['result']
            compact={key:copy.deepcopy(value) for key,value in ledger.items() if key!='arms'}
            compact['arms']={name:{key:copy.deepcopy(value) for key,value in arm.items() if key!='observations'}
                             for name,arm in ledger['arms'].items()}
            primary['cases'][index]=compact
        return primary

    def test_full_trace_rejects_incomplete_initialisation_and_changed_round_gap(self):
        for mutation in ('date','width','reward','centre','work'):
            artifacts=dict(self.artifacts)
            artifacts[0]=copy.deepcopy(self.artifacts[0])
            row=artifacts[0]['result']['arms']['coarse_common']['observations'][0]
            if mutation=='date':row['initial_queries'].pop()
            elif mutation=='width':row['initial_common_width_certificate']=SUMMARY._replicated_interval_record(self.Fraction(2),self.Fraction(2))
            elif mutation=='reward':row['reward_certificates'][-1]=SUMMARY._replicated_interval_record(self.Fraction(0),self.Fraction(0))
            elif mutation=='centre':row['frozen_centres_and_penalty_sha256']='0'*64
            else:row['initial_queries'][0]['work']['integrals']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.build_static(self.refreshed_primary(artifacts),artifacts,self.design,self.prior)

    def test_full_record_rejects_retained_preparation_and_summary_tampering(self):
        artifacts=dict(self.artifacts)
        artifacts[0]=copy.deepcopy(self.artifacts[0])
        artifacts[0]['input']['preparation_cost']['cpu_seconds']+=.1
        with self.assertRaises(ValueError):self.build_static(self.primary,artifacts,self.design,self.prior)
        primary=copy.deepcopy(self.primary);primary['arm_summaries']['cold_common']['conditional_integrals']+=1
        with self.assertRaises(ValueError):self.build_static(primary,self.artifacts,self.design,self.prior)




class RestrictedInitialRightResultsTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((HERE.parent/'revision_results.json').read_text())
        self.study=self.data['restricted_initial_right']
        self.prior=self.data['uniform_allocation']

    def check(self):
        return SUMMARY.validate_restricted_initial_right(self.study,self.prior)

    def test_retained_reconstruction_matches_its_full_calendar_comparator(self):
        self.assertTrue(self.check())

    def test_reconstruction_cannot_claim_new_sampling_coverage_or_policy_improvement(self):
        for field in ('conditional_integrals_rerun','new_paths_generated','new_fitting_executed',
                      'sampling_coverage_certified','policy_improvement_claim','cpu_saving_claim'):
            changed=copy.deepcopy(self.study);changed[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_restricted_initial_right(changed,self.prior)

    def test_full_trace_and_algebra_checks_are_required(self):
        for field in ('all_coarse_traces_validated','all_full_gain_reconstructions_match',
                      'all_endpoint_dominance_checks_pass'):
            changed=copy.deepcopy(self.study);changed[field]=False
            with self.subTest(field=field),self.assertRaises(ValueError):
                SUMMARY.validate_restricted_initial_right(changed,self.prior)

    def test_every_retained_path_is_required_for_the_published_dominance_count(self):
        self.study['cases'][0]['strictly_ruled_out_initial_candidate_count']-=1
        with self.assertRaises(ValueError):self.check()

    def test_original_input_and_source_context_cannot_be_substituted(self):
        for kind in ('input','source','audit','design'):
            changed=copy.deepcopy(self.study)
            if kind=='input':changed['cases'][0]['retained_input_sha256']='0'*64
            elif kind=='source':changed['source_sha256']['scripts/refine_multivariate_gap.py']='0'*64
            elif kind=='audit':changed['audit_source_sha256']='0'*64
            else:changed['design_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                SUMMARY.validate_restricted_initial_right(changed,self.prior)

    def test_a_zero_observed_reduction_cannot_be_replaced_by_an_improvement(self):
        self.study['cases'][0]['mean_gain_reduction_certificate']=SUMMARY._replicated_interval_record(
            SUMMARY.Fraction(1,10),SUMMARY.Fraction(1,10))
        with self.assertRaises(ValueError):self.check()

    def test_unchanged_full_mean_must_match_the_original_coarse_control(self):
        changed=SUMMARY._replicated_interval_record(SUMMARY.Fraction(0),SUMMARY.Fraction(0))
        self.study['cases'][0]['mean_full_gain_certificate']=changed
        self.study['cases'][0]['mean_restricted_gain_certificate']=changed
        with self.assertRaises(ValueError):self.check()

    def test_restricted_width_cannot_be_inferred_only_from_endpoint_dominance(self):
        self.study['cases'][0]['maximum_restricted_width']=dict(
            numerator='1',denominator='50',approximation=.02)
        with self.assertRaises(ValueError):self.check()

    def test_reconstruction_requires_every_fit_in_original_order(self):
        self.study['cases'][0],self.study['cases'][1]=self.study['cases'][1],self.study['cases'][0]
        with self.assertRaises(ValueError):self.check()

    def test_equal_fit_mean_and_total_count_are_recomputed(self):
        for kind in ('mean','count'):
            changed=copy.deepcopy(self.study)
            if kind=='mean':changed['aggregate']['mean_full_gain_certificate']=SUMMARY._replicated_interval_record(
                SUMMARY.Fraction(0),SUMMARY.Fraction(0))
            else:changed['aggregate']['retained_adaptive_full_interval_intersections']-=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                SUMMARY.validate_restricted_initial_right(changed,self.prior)


class MultivariateMonetarySummaryTests(unittest.TestCase):
    def setUp(self):
        self.study=json.loads((HERE.parent/'revision_results.json').read_text())['multivariate_monetary']
    def test_current_monetary_summary_checks(self):
        self.assertTrue(SUMMARY.validate_multivariate_monetary(self.study))
    def test_exposure_and_eligible_counts_cannot_change(self):
        for kind in ('prefix','case','total'):
            x=copy.deepcopy(self.study)
            if kind=='prefix':x['excluded_prefix']['rows']=0
            elif kind=='case':x['cases'][0]['assessors']['zero']['looks'][0]['eligible_paths']=4096
            else:x['eligible_paths_across_fits']=131072
            with self.subTest(kind=kind),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_probability_cap_and_tail_cannot_change(self):
        for kind in ('family','cap','tail'):
            x=copy.deepcopy(self.study)
            if kind=='family':x['family']['events']=16
            elif kind=='cap':x['caps']['european_common']=150
            else:x['cases'][0]['assessors']['zero']['looks'][0]['upper_limit']['population_tail_upper']=dict(numerator='0',denominator='1',approximation=0.)
            with self.subTest(kind=kind),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_acceptable_policy_and_unresolved_budget_are_distinct(self):
        for kind in ('acceptance','budget','cost','correction'):
            x=copy.deepcopy(self.study); result=x['cases'][0]['assessors']['zero']
            if kind=='acceptance':result['looks'][0]['acceptance']['1']=True
            elif kind=='budget':result['decisions']['1']['full_budget_charged_if_unresolved']=False
            elif kind=='cost':result['decisions']['1']['original_required_prefix_cost']['cpu_seconds']=0.
            else:result['decisions']['1']['conservative_after_run_cost']['cpu_seconds']-=1.
            with self.subTest(kind=kind),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_mean_and_variance_cannot_be_replaced_by_display_values(self):
        for field in ('mean_upper','variance_upper','squared_radius_budget'):
            x=copy.deepcopy(self.study)
            x['cases'][0]['assessors']['european_common']['looks'][-1]['upper_limit'][field]=dict(numerator='0',denominator='1',approximation=0.)
            with self.subTest(field=field),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_missing_fit_assessor_and_look_are_rejected(self):
        for kind in ('fit','assessor','look'):
            x=copy.deepcopy(self.study)
            if kind=='fit':x['cases'].pop()
            elif kind=='assessor':x['cases'][0]['assessors'].pop('zero')
            else:x['cases'][0]['assessors']['zero']['looks'].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_statistical_and_market_scope_cannot_be_strengthened(self):
        for field in ('sampling_law_certified','unrounded_ideal_path_enclosure_established','market_validation_established','policy_improved','substantive_original_research_contribution_established'):
            x=copy.deepcopy(self.study);x[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_sampling_radius_cannot_be_deleted_from_the_upper_limit(self):
        from fractions import Fraction
        x=copy.deepcopy(self.study)
        look=x['cases'][0]['assessors']['european_common']['looks'][-1]
        limit=look['upper_limit']; value=SUMMARY._replicated_rational(limit['mean_upper'])+SUMMARY._replicated_rational(limit['population_tail_upper'])
        record=lambda q:dict(numerator=str(q.numerator),denominator=str(q.denominator),approximation=float(q))
        limit['upper_limit']=record(value)
        limit['upper_limit_certificate']=dict(lower=record(value),upper=record(value),width=record(Fraction(0)))
        look['acceptance']={target:value<=Fraction(target) for target in ('1/2','1')}
        with self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_log_and_radius_certificates_cannot_be_zeroed(self):
        for field in ('log_two_over_delta_certificate','sampling_radius_certificate'):
            x=copy.deepcopy(self.study)
            zero=dict(numerator='0',denominator='1',approximation=0.)
            x['cases'][0]['assessors']['zero']['looks'][0]['upper_limit'][field]=dict(lower=zero.copy(),upper=zero.copy(),width=zero.copy())
            with self.subTest(field=field),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_raw_observations_cannot_be_appended_to_public_aggregates(self):
        for location in ('study','case','assessor','look','limit','statistics','decision','rational','interval','tail'):
            x=copy.deepcopy(self.study);case=x['cases'][0];arm=case['assessors']['zero'];look=arm['looks'][0]
            target={'study':x,'case':case,'assessor':arm,'look':look,'limit':look['upper_limit'],'statistics':look['upper_limit']['statistics'],'decision':arm['decisions']['1'],'rational':look['upper_limit']['mean_upper'],'interval':look['raw_gain_mean_certificate'],'tail':x['tail_allowances']}[location]
            target['states']=[[100.]*5]
            with self.subTest(location=location),self.assertRaises(ValueError):SUMMARY.validate_multivariate_monetary(x)
    def test_monetary_table_has_one_row_for_each_fit(self):
        value=SUMMARY.render_multivariate_monetary(self.study)
        self.assertIn('revision-multivariate-monetary',value)
        self.assertIn('4,080 and 16,368',value)
        self.assertIn('after-run correction',value)
        import re
        self.assertEqual(len(re.findall(r'^[1-8] & ',value,re.MULTILINE)),8)


if __name__ == '__main__':
    unittest.main()
