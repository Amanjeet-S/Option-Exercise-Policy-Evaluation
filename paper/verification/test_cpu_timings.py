"""Independent arithmetic and mutation checks for the public CPU ledger."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("released_cpu_timings", HERE / "cpu_timings.py")
CPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CPU)


class CpuTimingsTests(unittest.TestCase):
    def setUp(self):
        self.data = CPU.read(HERE.parent / "cpu_timings.json")
        self.summary = CPU.read(HERE.parent / "numerical_results.json")

    def test_complete_ledger_reconstructs_existing_findings_and_note(self):
        result = CPU.reconstruct(self.data)
        CPU.check_summary(result, self.summary)
        CPU.check_markdown((HERE.parent / "cpu_timings.md").read_text(), CPU.render(result))
        self.assertEqual(result["counts"], {"validation_arms": 18, "validation_pairs": 7956,
                         "calibration_pairs": 576, "warmup_pairs": 4})

    def test_overrun_is_retained_and_identified(self):
        failed = [r for r in CPU.reconstruct(self.data)["workloads"] if r["budget_excess_cpu"] > 0]
        self.assertEqual([(r["pass_id"], r["position"], r["case"], r["evaluation_paths"])
                          for r in failed], [(0, 6, 10, 4096)])
        self.assertEqual(CPU.number(failed[0]["budget_excess_cpu"], 3), "3.852")
        self.assertEqual(CPU.number(failed[0]["comparisons"]["immediate"]["ratio"], 4), "1.2261")

    def test_forecast_denominator_and_equal_weighting(self):
        result = CPU.reconstruct(self.data)
        rows = result["workloads"]
        actual_denominator = sum(abs(r["actual_cpu"] - r["comparisons"]["historical"]["forecast"])
                                 / r["actual_cpu"] for r in rows) * 100 / 18
        reported = result["predictors"][0]["mean_absolute_relative_error_percent"]
        self.assertNotEqual(CPU.number(actual_denominator, 3), CPU.number(reported, 3))
        self.assertEqual(CPU.number(reported, 3), "29.065")
        altered = copy.deepcopy(result)
        altered["predictors"][0]["mean_absolute_relative_error_percent"] = actual_denominator
        with self.assertRaisesRegex(ValueError, "forecast summary"):
            CPU.check_summary(altered, self.summary)

    def test_changed_validation_or_setup_timer_fails_summary_reconciliation(self):
        for block, key in (("validation", "pairs_cpu_seconds"),
                           ("pass_start_calibration", "setup_cpu_seconds"),
                           ("immediate_calibration", "pairs_cpu_seconds")):
            data = copy.deepcopy(self.data)
            data["workloads"][0][block][key] = str(Fraction(data["workloads"][0][block][key]) + 1)
            with self.subTest(block=block), self.assertRaises(ValueError):
                CPU.check_summary(CPU.reconstruct(data), self.summary)

    def test_missing_duplicate_and_reordered_workloads_fail(self):
        variations = [self.data["workloads"][:-1],
                      self.data["workloads"][:-1] + [self.data["workloads"][0]],
                      list(reversed(self.data["workloads"]))]
        for rows in variations:
            data = copy.deepcopy(self.data)
            data["workloads"] = rows
            with self.subTest(length=len(rows)), self.assertRaises(ValueError):
                CPU.reconstruct(data)

    def test_changed_fixed_counts_and_case_identity_fail(self):
        for key, value in (("position", 4), ("training_pairs", 999), ("case", 5),
                           ("pass_id", True)):
            data = copy.deepcopy(self.data)
            data["workloads"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CPU.reconstruct(data)

    def test_unexpected_fields_fail_at_every_level(self):
        for location in ("root", "design", "run", "workload", "block"):
            data = copy.deepcopy(self.data)
            node = {"root": data, "design": data["design"], "run": data["run_costs"],
                    "workload": data["workloads"][0],
                    "block": data["workloads"][0]["validation"]}[location]
            node["financial_input"] = "excluded from this timing schema"
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "fields"):
                CPU.reconstruct(data)

    def test_invalid_rational_encodings_fail(self):
        for value in ("-1", "NaN", "Infinity", "1/0", "2/2", "1.25", True, 1.0):
            data = copy.deepcopy(self.data)
            data["workloads"][0]["validation"]["pairs_cpu_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                CPU.reconstruct(data)

    def test_json_duplicate_keys_and_nonfinite_values_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for text in ('{"cost": 1, "cost": 2}', '{"cost": NaN}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    CPU.read(path)

    def test_calibration_is_additional_expenditure(self):
        result = CPU.reconstruct(self.data)
        self.assertEqual(result["affordability"],
                         {"validation": 17, "pass_start": 17, "immediate": 17, "both": 14})
        costs = result["costs"]
        self.assertEqual(costs["numerical"], sum(costs[k] for k in
            ("validation", "pass_start_calibration", "immediate_calibration", "warmup")))
        self.assertEqual(costs["process"], costs["numerical"] + costs["process_overhead"])
        changed = copy.deepcopy(result)
        changed["costs"]["numerical"] -= changed["costs"]["pass_start_calibration"]
        with self.assertRaisesRegex(ValueError, "expenditure"):
            CPU.check_summary(changed, self.summary)

    def test_smaller_process_clock_cannot_contain_numerical_work(self):
        self.data["run_costs"]["process_cpu_seconds"] = "1"
        with self.assertRaisesRegex(ValueError, "exceeds process"):
            CPU.reconstruct(self.data)

    def test_changed_table_or_missing_marker_fails(self):
        result = CPU.reconstruct(self.data)
        original = (HERE.parent / "cpu_timings.md").read_text()
        for text in (original.replace("29.065", "29.066"),
                     original.replace("<!-- cpu:prediction -->", "")):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                CPU.check_markdown(text, CPU.render(result))


if __name__ == "__main__":
    unittest.main()
