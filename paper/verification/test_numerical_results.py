"""Reject inconsistencies in the released aggregate summaries.

These checks cover the public presentation contract. They do not rerun licensed
experiments, prove statistical coverage or establish publication permission.
"""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('released_numerical_results', HERE / 'numerical_results.py')
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class ReleasedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((HERE.parent / 'numerical_results.json').read_text())

    def test_released_summary_and_nine_rendered_tables(self):
        SUMMARY.check(self.data)
        tables = SUMMARY.render(self.data)
        self.assertEqual(len(tables), 9)
        prose = (HERE.parent / 'numerical_results.md').read_text()
        for name, table in tables.items():
            start, end = f'<!-- table:{name} -->', f'<!-- endtable:{name} -->'
            self.assertEqual(prose.count(start), 1)
            self.assertEqual(prose.count(end), 1)
            self.assertEqual(prose.split(start)[1].split(end)[0].strip(), table)

    def test_unreviewed_fields_are_rejected_at_each_summary_level(self):
        for location in ('root', 'table', 'design', 'influence'):
            data = copy.deepcopy(self.data)
            row = {'root': data, 'table': data['cpu_prediction'][0],
                   'design': data['prospective_design'],
                   'influence': data['prospective_design']['largest_squared_variance_influence_percent'][0]}[location]
            row['unexpected_input'] = 'not part of the released schema'
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, '[Uu]nexpected'):
                SUMMARY.check(data)

    def test_non_finite_summary_values_are_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            self.data['cpu_prediction'][0]['mean_absolute_relative_error_percent'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Non-finite'):
                SUMMARY.check(self.data)

    def test_reordered_or_omitted_contrasts_are_rejected(self):
        for rows in (self.data['prospective_contrasts'][::-1], self.data['prospective_contrasts'][:-1]):
            data = copy.deepcopy(self.data)
            data['prospective_contrasts'] = rows
            with self.subTest(length=len(rows)), self.assertRaisesRegex(ValueError, 'contrast coverage'):
                SUMMARY.check(data)

    def test_incorrect_interval_classification_is_rejected(self):
        self.data['prospective_contrasts'][0]['variance_upper'] = 1e-7
        with self.assertRaisesRegex(ValueError, 'Variance interval classification'):
            SUMMARY.check(self.data)

    def test_changed_fixed_counts_are_rejected(self):
        self.data['prospective_allocation'][0]['pairs_128'] -= 1
        self.data['prospective_allocation'][0]['pairs_256'] += 1
        with self.assertRaisesRegex(ValueError, 'per-arm counts'):
            SUMMARY.check(self.data)

    def test_changed_variance_cost_product_is_rejected(self):
        self.data['prospective_allocation'][0]['variance_cpu_ratio_256'] += .01
        with self.assertRaisesRegex(ValueError, 'variance-cost identity'):
            SUMMARY.check(self.data)

    def test_calibration_count_and_cost_categories_are_distinct(self):
        for field, value in (('calibration_pairs', 288), ('within_budget_with_immediate_calibration', 14),
                             ('within_budget_with_both_calibration_schemes', 17)):
            data = copy.deepcopy(self.data)
            data['cpu_prediction_design'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                SUMMARY.check(data)

    def test_relative_error_denominator_is_explicit(self):
        self.data['cpu_prediction_design']['relative_error_denominator'] = 'actual'
        with self.assertRaisesRegex(ValueError, 'denominator'):
            SUMMARY.check(self.data)

    def test_numerical_cpu_components_must_sum(self):
        self.data['cpu_prediction_design']['total_numerical_cpu_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'CPU total'):
            SUMMARY.check(self.data)

    def test_unsupported_coverage_claim_is_rejected(self):
        for section, flag in (('cpu_prediction_design', 'prediction_coverage_claim'),
                              ('prospective_design', 'finite_sample_coverage_verified')):
            data = copy.deepcopy(self.data)
            data[section][flag] = True
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, 'Unsupported'):
                SUMMARY.check(data)


if __name__ == '__main__':
    unittest.main()
