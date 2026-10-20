"""Reconstruct CPU forecast findings from the released workload measurements.

Only standard-library arithmetic is required. This calculation does not rerun
the workloads or certify the financial assumptions underlying them.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parent.parent
PREDICTORS = ("historical", "pass_start", "immediate")
LABELS = {"historical": "Historical", "pass_start": "Pass-start calibration",
          "immediate": "Immediate calibration"}
BLOCKS = ("validation", "pass_start_calibration", "immediate_calibration")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, context):
    require(isinstance(value, dict) and set(value) == set(names.split()),
            "Unexpected or missing fields: " + context)


def read(path):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "Repeated JSON key")
            result[name] = value
        return result

    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=reject)


def rational(value, positive=False):
    require(isinstance(value, str) and len(value) <= 160
            and re.fullmatch(r"(0|[1-9][0-9]*)(/[1-9][0-9]*)?", value) is not None,
            "A non-negative canonical rational string is required")
    result = Fraction(value)
    require(str(result) == value, "Non-canonical rational string")
    require(not positive or result > 0, "A positive duration is required")
    return result


def number(value, places):
    """Round an exact rational only for presentation."""
    value = Fraction(value)
    with localcontext() as context:
        context.prec = 100
        return f"{Decimal(value.numerator) / Decimal(value.denominator):.{places}f}"


def integer(value, minimum, maximum=None):
    require(type(value) is int and value >= minimum
            and (maximum is None or value <= maximum), "Invalid integer design value")


def validate(data):
    fields(data, "schema_version title measurement_date unit encoding design run_costs workloads",
           "timing supplement")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1,
            "Unsupported timing schema")
    require(data["title"] == "Complete-workload CPU forecast measurements"
            and data["measurement_date"] == "2026-09-28"
            and data["unit"] == "CPU seconds"
            and data["encoding"] == "Exact rational representations of retained binary timer measurements",
            "Timing metadata differs")
    design = data["design"]
    fields(design, "cases evaluation_paths training_paths degree passes calibration_pairs_per_workload_per_scheme warmup_pairs tolerance allowance_multiplier counts_fixed_before_validation", "timing design")
    expected = {"cases": [5, 6, 10], "evaluation_paths": [128, 256, 4096],
                "training_paths": [5000, 20000], "degree": 4, "passes": 2,
                "calibration_pairs_per_workload_per_scheme": 16, "warmup_pairs": 4}
    for key, value in expected.items():
        require(design[key] == value, "Registered design differs: " + key)
        for item in design[key] if isinstance(design[key], list) else [design[key]]:
            integer(item, 1)
    require(design["tolerance"] == ["9/10", "11/10"]
            and design["allowance_multiplier"] == "11/10"
            and design["counts_fixed_before_validation"] is True,
            "Registered tolerance or sampling rule differs")
    fields(data["run_costs"], "warmup_cpu_seconds process_cpu_seconds process_elapsed_seconds",
           "whole-run costs")
    for value in data["run_costs"].values():
        rational(value, positive=True)
    rows = data["workloads"]
    require(isinstance(rows, list) and len(rows) == 18, "Eighteen workloads are required")
    for row in rows:
        fields(row, "pass_id position case evaluation_paths training_pairs budget_cpu_seconds historical_forecast_cpu_seconds validation pass_start_calibration immediate_calibration", "workload")
        integer(row["pass_id"], 0, 1)
        integer(row["position"], 0, 8)
        integer(row["case"], 1)
        integer(row["evaluation_paths"], 1)
        integer(row["training_pairs"], 1)
        require(row["case"] in design["cases"] and row["evaluation_paths"] in design["evaluation_paths"],
                "Unexpected case or evaluation count")
        rational(row["budget_cpu_seconds"], positive=True)
        rational(row["historical_forecast_cpu_seconds"], positive=True)
        for block in BLOCKS:
            fields(row[block], "setup_cpu_seconds pairs_cpu_seconds", block)
            for value in row[block].values():
                rational(value, positive=True)
    require([(r["pass_id"], r["position"]) for r in rows]
            == [(p, i) for p in range(2) for i in range(9)],
            "Missing, duplicate or reordered workload position")
    identities = [(r["case"], r["evaluation_paths"]) for r in rows]
    require(set(identities[:9]) == {(c, n) for c in design["cases"] for n in design["evaluation_paths"]}
            and identities[9:] == identities[:9][::-1], "Workload identity or reverse order differs")
    first = {(r["case"], r["evaluation_paths"]): r for r in rows[:9]}
    for row in rows[9:]:
        previous = first[(row["case"], row["evaluation_paths"])]
        require(all(row[k] == previous[k] for k in
                    ("training_pairs", "budget_cpu_seconds", "historical_forecast_cpu_seconds")),
                "Fixed workload or historical plan differs between passes")


def reconstruct(data):
    validate(data)
    n_cal = data["design"]["calibration_pairs_per_workload_per_scheme"]
    low, high = map(Fraction, data["design"]["tolerance"])
    allowance = Fraction(data["design"]["allowance_multiplier"])
    diagnostics = []
    totals = {name: Fraction(0) for name in BLOCKS}
    affordable = dict(validation=0, pass_start=0, immediate=0, both=0)
    for row in data["workloads"]:
        charges = {name: sum((Fraction(v) for v in row[name].values()), Fraction(0))
                   for name in BLOCKS}
        for name, value in charges.items():
            totals[name] += value
        actual = charges["validation"]
        budget = Fraction(row["budget_cpu_seconds"])
        forecasts = {"historical": Fraction(row["historical_forecast_cpu_seconds"])}
        for name, block in (("pass_start", "pass_start_calibration"),
                            ("immediate", "immediate_calibration")):
            forecasts[name] = (Fraction(row[block]["setup_cpu_seconds"])
                               + row["training_pairs"] * Fraction(row[block]["pairs_cpu_seconds"]) / n_cal)
        comparisons = {}
        for name, forecast in forecasts.items():
            ratio = actual / forecast
            upper = allowance * forecast
            comparisons[name] = {
                "forecast": forecast, "ratio": ratio,
                "absolute_relative_error": abs(actual - forecast) / forecast,
                "within_tolerance": low <= ratio <= high,
                "within_allowance": actual <= upper,
                "forecast_affordable": forecast <= budget,
                "allowance_affordable": upper <= budget,
                "falsely_forecast_affordable": forecast <= budget < actual,
                "falsely_allowance_affordable": upper <= budget < actual,
            }
        affordable["validation"] += actual <= budget
        affordable["pass_start"] += actual + charges["pass_start_calibration"] <= budget
        affordable["immediate"] += actual + charges["immediate_calibration"] <= budget
        affordable["both"] += actual + charges["pass_start_calibration"] + charges["immediate_calibration"] <= budget
        diagnostics.append({k: row[k] for k in
                            ("pass_id", "position", "case", "evaluation_paths", "training_pairs")} |
                           {"actual_cpu": actual, "budget_cpu": budget,
                            "budget_excess_cpu": max(Fraction(0), actual - budget),
                            "comparisons": comparisons})
    summary = []
    count_fields = ("within_tolerance", "within_allowance", "forecast_affordable",
                    "allowance_affordable", "falsely_forecast_affordable",
                    "falsely_allowance_affordable")
    for name in PREDICTORS:
        entries = [r["comparisons"][name] for r in diagnostics]
        summary.append({"predictor": name,
            **{key: sum(e[key] for e in entries) for key in count_fields},
            "mean_absolute_relative_error_percent": 100 * sum(e["absolute_relative_error"] for e in entries) / len(entries),
            "actual_to_forecast_min": min(e["ratio"] for e in entries),
            "actual_to_forecast_max": max(e["ratio"] for e in entries)})
    run = {key: Fraction(value) for key, value in data["run_costs"].items()}
    totals["warmup"] = run["warmup_cpu_seconds"]
    totals["numerical"] = sum(totals.values(), Fraction(0))
    totals["process"] = run["process_cpu_seconds"]
    totals["process_elapsed"] = run["process_elapsed_seconds"]
    require(totals["process"] >= totals["numerical"], "Numerical CPU exceeds process CPU")
    totals["process_overhead"] = totals["process"] - totals["numerical"]
    counts = {"validation_arms": len(diagnostics),
              "validation_pairs": sum(r["training_pairs"] for r in diagnostics),
              "calibration_pairs": len(diagnostics) * n_cal * 2,
              "warmup_pairs": data["design"]["warmup_pairs"]}
    return {"predictors": summary, "workloads": diagnostics, "costs": totals,
            "affordability": affordable, "counts": counts}


def check_summary(result, released):
    """Compare reconstructed quantities with their existing published precision."""
    expected = released["cpu_prediction"]
    require(len(expected) == len(result["predictors"]), "Predictor coverage differs")
    precision = {"mean_absolute_relative_error_percent": 3,
                 "actual_to_forecast_min": 4, "actual_to_forecast_max": 4}
    for actual, saved in zip(result["predictors"], expected):
        require(set(actual) == set(saved), "Predictor summary fields differ")
        for key, value in actual.items():
            if key in precision:
                require(Decimal(number(value, precision[key])) == Decimal(str(saved[key])),
                        "Reconstructed forecast summary differs: " + key)
            else:
                require(value == saved[key], "Reconstructed forecast count differs: " + key)
    design = released["cpu_prediction_design"]
    for key, value in result["counts"].items():
        require(value == design[key], "Reconstructed workload count differs: " + key)
    for name, key in {"validation": "validation_within_original_budget",
                      "pass_start": "within_budget_with_pass_start_calibration",
                      "immediate": "within_budget_with_immediate_calibration",
                      "both": "within_budget_with_both_calibration_schemes"}.items():
        require(result["affordability"][name] == design[key],
                "Reconstructed affordability differs: " + name)
    cost_fields = {"validation": "validation_numerical_cpu_seconds",
                   "pass_start_calibration": "pass_start_calibration_cpu_seconds",
                   "immediate_calibration": "immediate_calibration_cpu_seconds",
                   "warmup": "warmup_numerical_cpu_seconds", "numerical": "total_numerical_cpu_seconds",
                   "process": "process_cpu_seconds", "process_elapsed": "process_elapsed_seconds"}
    for name, key in cost_fields.items():
        require(Decimal(number(result["costs"][name], 3)) == Decimal(str(design[key])),
                "Reconstructed expenditure differs: " + key)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                     "| " + " | ".join(["---"] * len(headers)) + " |"]
                    + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def render(result):
    tables = {}
    tables["prediction"] = table(
        ["Forecast", "Within ten per cent", "Within upper allowance", "Mean absolute relative error (per cent)", "Actual/forecast range"],
        [[LABELS[r["predictor"]], f'{r["within_tolerance"]}/18', f'{r["within_allowance"]}/18',
          number(r["mean_absolute_relative_error_percent"], 3),
          f'[{number(r["actual_to_forecast_min"], 4)}, {number(r["actual_to_forecast_max"], 4)}]']
         for r in result["predictors"]])
    tables["affordability"] = table(
        ["Forecast", "Point forecast affordable", "False point declarations", "Upper allowance affordable", "False allowance declarations"],
        [[LABELS[r["predictor"]]] + [f'{r[k]}/18' for k in
          ("forecast_affordable", "falsely_forecast_affordable", "allowance_affordable", "falsely_allowance_affordable")]
         for r in result["predictors"]])
    tables["workloads"] = table(
        ["Pass", "Position", "Case", "Evaluation paths", "Training pairs", "Actual CPU", "Budget", "Actual/immediate forecast", "Budget excess"],
        [[r["pass_id"] + 1, r["position"] + 1, r["case"], r["evaluation_paths"], r["training_pairs"],
          number(r["actual_cpu"], 3), number(r["budget_cpu"], 3),
          number(r["comparisons"]["immediate"]["ratio"], 4), number(r["budget_excess_cpu"], 3)]
         for r in result["workloads"]])
    tables["costs"] = table(
        ["Component", "Seconds"],
        [[label, number(result["costs"][key], 3)] for key, label in
         (("validation", "Validation numerical CPU"),
          ("pass_start_calibration", "Pass-start calibration CPU"),
          ("immediate_calibration", "Immediate calibration CPU"),
          ("warmup", "Warm-up CPU"), ("numerical", "Total numerical CPU"),
          ("process_overhead", "Other measured process CPU"),
          ("process", "Whole-process CPU"), ("process_elapsed", "Whole-process elapsed time"))])
    tables["charged"] = table(
        ["Charge compared with the unchanged workload budget", "Affordable workloads"],
        [[label, f'{result["affordability"][key]}/18'] for key, label in
         (("validation", "Validation only"), ("pass_start", "Validation and pass-start calibration"),
          ("immediate", "Validation and immediate calibration"),
          ("both", "Validation and both calibration schemes"))])
    return tables


def check_markdown(text, tables):
    for name, content in tables.items():
        start, end = f"<!-- cpu:{name} -->", f"<!-- endcpu:{name} -->"
        require(text.count(start) == text.count(end) == 1, "Missing or repeated table marker")
        require(text.split(start)[1].split(end)[0].strip() == content,
                "Reconstructed timing table differs: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PAPER / "cpu_timings.json")
    args = parser.parse_args()
    result = reconstruct(read(args.data))
    check_summary(result, read(PAPER / "numerical_results.json"))
    check_markdown((PAPER / "cpu_timings.md").read_text(), render(result))
    print("Reconstructed 18 workloads, three forecast comparisons and complete CPU expenditure.")
    print("The released findings agree at their published precision; no workloads were rerun.")


if __name__ == "__main__":
    main()
