"""Frozen funded policies: separate training, control pilot and paired evaluation."""

import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.funded_policy import (
    simulate_paths, prefix_paths, fit_policy, calibrate_control, evaluate_policy,
)
from qf_research.optimal_stopping.simulation import estimate_samples

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(directory):
    manifest = json.loads((directory/"manifest.json").read_text())
    for group in ("input_sha256", "source_sha256"):
        for path, expected in manifest[group].items():
            if sha(ROOT/path) != expected:
                raise ValueError(f"Reference provenance changed: {path}")
    for path, expected in manifest["artifact_sha256"].items():
        if sha(directory/path) != expected:
            raise ValueError(f"Reference artifact changed: {path}")
    with (directory/"refinement.csv").open() as f:
        return list(csv.DictReader(f)), manifest


def make_cases(base_path):
    case = json.loads(base_path.read_text())["dated_validation"]
    calendar_path, issuer_path = ROOT/case["calendar_config"], ROOT/case["issuer_config"]
    calendar = EventCalendar.from_config(json.loads(calendar_path.read_text()))
    issuer = json.loads(issuer_path.read_text())
    observation, end = datetime.fromisoformat(case["observation"]), date.fromisoformat(case["curve_end"])
    curve = DiscountCurve(observation.date(), (observation.date(), end),
                           (1., math.exp(-case["flat_rate"]*(end-observation.date()).days/365)))
    dividends = tuple(DatedDividend(date.fromisoformat(e["ex_date"]), date.fromisoformat(e["pay_date"]),
                                    None if e["conditional"] else case["regular_dividend_cash_fixture"],
                                    "Declared mathematical cash fixture, not an observation",
                                    None if e["conditional"] else True) for e in issuer["events"])
    problems = []
    for expiry in case["expiries"]:
        for kind in ("call", "put"):
            problem = build_dated_problem(calendar, curve, observation, date.fromisoformat(expiry),
                                           case["quoted_spot"], case["strike"], case["volatility"], kind, dividends,
                                           premium_payment_date=date.fromisoformat(case["premium_payment_date"]),
                                           allow_provisional=case["allow_provisional_settlement"])
            problems.append((expiry, kind, problem))
    return problems, (calendar_path, issuer_path)


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path, output):
    config = json.loads(config_path.read_text())
    if config["bit_generator"] != "PCG64" or config["confidence"] != .95:
        raise ValueError("This frozen protocol implements PCG64 and 95% pointwise intervals")
    base_path, reference_dir = ROOT/config["base_config"], ROOT/config["reference_directory"]
    references, reference_manifest = verify_reference(reference_dir)
    problems, extra_inputs = make_cases(base_path)
    output.mkdir(parents=True, exist_ok=False)
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    root_seed = np.random.SeedSequence(config["root_seed"])
    case_seeds = root_seed.spawn(len(problems))
    evaluations, pairs, diagnostics, fits, streams, summaries = [], [], [], [], [], []
    selection = config["reference_selection"]
    baseline = config["paired_baseline"]

    for (expiry, kind, problem), case_seed in zip(problems, case_seeds):
        selected = [r for r in references if r["expiry"] == expiry and r["kind"] == kind
                    and int(r["space_steps"]) == selection["space_steps"] and float(r["y_max"]) == selection["y_max"]
                    and (r["method"] == "integral" or int(r["requested_time_steps"]) == selection["pde_time_steps"])]
        ref = {r["method"]:float(r["present_value"]) for r in selected}
        if len(selected) != 2 or set(ref) != {"PDE", "integral"}:
            raise ValueError("Reference selection is missing or ambiguous")
        for repetition, repetition_seed in enumerate(case_seed.spawn(config["repetitions"])):
            path_sets = []
            for role, count, seed in zip(("training", "control_pilot", "evaluation"),
                                         (max(config["training_sizes"]), config["pilot_paths"], config["evaluation_paths"]),
                                         repetition_seed.spawn(3)):
                sample = simulate_paths(problem.model, problem.option, problem.decisions, count,
                                         np.random.Generator(np.random.PCG64(seed)))
                path_sets.append(sample)
                streams.append(dict(expiry=expiry, kind=kind, repetition=repetition, role=role,
                                    paths=count, entropy=seed.entropy, spawn_key=list(seed.spawn_key),
                                    stream_id=sample.stream_id))
            training, pilot, evaluation = path_sets
            outcomes = {}
            for n in config["training_sizes"]:
                for degree in config["degrees"]:
                    key = dict(expiry=expiry, kind=kind, repetition=repetition, training_paths=n, degree=degree)
                    policy = fit_policy(problem.option, prefix_paths(training, n), degree,
                                         minimum_itm_paths=config["minimum_itm_paths"], condition_limit=config["condition_limit"])
                    control = calibrate_control(policy, pilot)
                    result = evaluate_policy(policy, evaluation, control=control)
                    outcomes[n, degree] = result
                    row = dict(**key, policy_fingerprint=policy.fingerprint, control_coefficient=control.coefficient,
                               pde_reference=ref["PDE"], integral_reference=ref["integral"],
                               reference_method_difference=ref["integral"]-ref["PDE"],
                               signed_pde_minus_controlled_policy=ref["PDE"]-result.controlled_pv.value,
                               early_exercise_fraction=float(np.mean(result.stopping_indices < len(problem.decisions)-1)),
                               terminal_zero_fraction=float(np.mean(evaluation.states[:, -1] == 0)),
                               gain_control_mean=result.control_mean.value, gain_control_se=result.control_mean.standard_error)
                    for name in ("raw_pv", "controlled_pv", "raw_quote", "controlled_quote"):
                        estimate = getattr(result, name)
                        for field in ("value", "standard_error", "lower", "upper"):
                            row[f"{name}_{field}"] = getattr(estimate, field)
                    evaluations.append(row)
                    for d in result.diagnostics:
                        diagnostics.append(dict(**key, **d))
                    fits.append(dict(**key, fingerprint=policy.fingerprint, policy=asdict(policy), control=asdict(control)))
            base = outcomes[baseline["training_paths"], baseline["degree"]]
            for (n, degree), result in outcomes.items():
                if (n, degree) == (baseline["training_paths"], baseline["degree"]):
                    continue
                row = dict(expiry=expiry, kind=kind, repetition=repetition, training_paths=n, degree=degree,
                           baseline_training_paths=baseline["training_paths"], baseline_degree=baseline["degree"])
                for name, a, b in (("raw", result.discounted_payoffs, base.discounted_payoffs),
                                    ("controlled", result.controlled_payoffs, base.controlled_payoffs)):
                    estimate = estimate_samples(a-b)
                    for field in ("value", "standard_error", "lower", "upper"):
                        row[f"{name}_difference_{field}"] = getattr(estimate, field)
                pairs.append(row)
            print(f"Evaluated {expiry} {kind}, repetition {repetition+1}/{config['repetitions']}", flush=True)

    for expiry, kind, _ in problems:
        for n in config["training_sizes"]:
            for degree in config["degrees"]:
                rows = [r for r in evaluations if (r["expiry"], r["kind"], r["training_paths"], r["degree"]) == (expiry, kind, n, degree)]
                estimates = [r["controlled_pv_value"] for r in rows]
                summaries.append(dict(expiry=expiry, kind=kind, training_paths=n, degree=degree,
                                      repetitions=len(rows), mean_controlled_pv=float(np.mean(estimates)),
                                      repeat_mean_sd=float(np.std(estimates, ddof=1)),
                                      rms_conditional_se=float(np.sqrt(np.mean([r["controlled_pv_standard_error"]**2 for r in rows]))),
                                      mean_signed_pde_minus_policy=float(np.mean([r["signed_pde_minus_controlled_policy"] for r in rows])),
                                      mean_se_ratio=float(np.mean([r["controlled_pv_standard_error"]/r["raw_pv_standard_error"] for r in rows]))))
    for filename, rows in (("evaluation.csv", evaluations), ("paired_comparisons.csv", pairs),
                            ("decision_diagnostics.csv", diagnostics), ("repeat_summary.csv", summaries)):
        save_csv(output/filename, rows)
    (output/"policies.json").write_text(json.dumps(fits, indent=2, allow_nan=False)+"\n")
    (output/"random_streams.json").write_text(json.dumps(streams, indent=2)+"\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    labels = [f"{n:,} / d={degree}" for n in config["training_sizes"] for degree in config["degrees"]]
    for ax, (expiry, kind, _) in zip(axes.flat, problems):
        rows = [r for r in evaluations if r["expiry"] == expiry and r["kind"] == kind]
        for i, (n, degree) in enumerate((n, d) for n in config["training_sizes"] for d in config["degrees"]):
            group = [r for r in rows if r["training_paths"] == n and r["degree"] == degree]
            for rep, row in enumerate(group):
                # Reverse CI ends when plotting reference minus policy.
                centre = row["signed_pde_minus_controlled_policy"]
                radius = row["controlled_pv_upper"]-row["controlled_pv_value"]
                ax.errorbar(i+(rep-1.5)*.13, centre, yerr=radius, fmt="o", ms=3,
                            color=plt.cm.tab10(rep), capsize=2)
        ax.axhline(0., color="black", lw=.8)
        ax.set(title=expiry+" "+kind, xticks=range(len(labels)), xticklabels=labels,
               ylabel="PDE PV minus policy estimate", xlabel="Training paths / polynomial degree")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Funded policy fixtures | colours: four independent repetitions\n95% conditional evaluation intervals; numerical reference is approximate", fontsize=11)
    fig.savefig(output/"policy_comparison.png", dpi=170)
    plt.close(fig)
    sources = {**reference_manifest["source_sha256"]}
    for relative in ("scripts/validate_funded_policy.py", "src/qf_research/optimal_stopping/funded_policy.py",
                     "src/qf_research/optimal_stopping/simulation.py"):
        sources[relative] = sha(ROOT/relative)
    inputs = (config_path, base_path, *extra_inputs, reference_dir/"manifest.json", reference_dir/"refinement.csv")
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), classification=config["classification"],
                    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs}, source_sha256=sources,
                    artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                    row_counts=dict(evaluation=len(evaluations), pairs=len(pairs), diagnostics=len(diagnostics), summaries=len(summaries)),
                    python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    empirical_calibration_accepted=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved funded policy validation to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
