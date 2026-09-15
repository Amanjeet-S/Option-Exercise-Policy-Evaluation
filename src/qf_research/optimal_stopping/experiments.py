"""Reproduce Project 1's synthetic benchmark tables, figures and findings.

Run from the repository root. All design choices come from the saved JSON config.
Timing covers individual computations, excludes plotting and includes no claims
about peak memory. Results are validation evidence under the specified model.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analytical import black_scholes
from .contracts import Market, Option
from .finite_difference import finite_difference_price
from .least_squares import evaluate_policy, train_policy
from .simulation import european_monte_carlo, simulate_gbm
from .tree import binomial_price


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _cpu_description():
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    return platform.processor() or platform.machine()


def run(config_path: Path, output: Path):
    started = perf_counter()
    config_bytes = config_path.read_bytes()
    c = json.loads(config_bytes)
    if c.get("purpose") != "numerical_validation_only":
        raise ValueError("This runner is for numerical validation only, not an empirical study")
    output.mkdir(parents=True, exist_ok=True)
    market, option = Market(**c["market"]), Option(**c["option"])
    dates = np.arange(1, c["exercise_count"] + 1) * option.maturity / c["exercise_count"]
    exact = black_scholes(market, option)
    seed_groups = np.random.SeedSequence(c["seed"]).spawn(2)
    mc_seeds = iter(seed_groups[0].spawn(len(c["mc_paths"]) * c["mc_repetitions"] * 2))
    policy_seeds = seed_groups[1].spawn(c["policy_repetitions"])
    print("Computing analytical, tree and PDE references", flush=True)
    tree_ref = binomial_price(market, option, c["tree_reference_steps"], dates)
    pde_ref = finite_difference_price(market, option, *c["pde_reference_grid"], c["s_max"], dates)
    grids = []
    for steps in c["tree_steps"]:
        for contract, calendar, reference in [("European", None, exact), ("Bermudan", dates, pde_ref.value)]:
            tick = perf_counter()
            result = binomial_price(market, option, steps, calendar)
            grids.append(dict(method="CRR", contract=contract, space_steps=0, time_steps=steps,
                              value=result.value, reference=reference, error=result.value - reference,
                              seconds=perf_counter() - tick, upwind_nodes=0))
    for space, time in c["pde_grids"]:
        for contract, calendar, reference in [("European", None, exact), ("Bermudan", dates, pde_ref.value)]:
            tick = perf_counter()
            result = finite_difference_price(market, option, space, time, c["s_max"], calendar)
            grids.append(dict(method="PDE", contract=contract, space_steps=space, time_steps=time,
                              value=result.value, reference=reference, error=result.value - reference,
                              seconds=perf_counter() - tick, upwind_nodes=result.upwind_nodes))
    domains = []
    spacing = c["s_max"] / c["pde_reference_grid"][0]
    for maximum in [c["s_max"], *c["domain_checks"]]:
        intervals = maximum / spacing
        if abs(intervals - round(intervals)) > 1e-9:
            raise ValueError("Domain comparisons must preserve the same spot spacing")
        result = finite_difference_price(market, option, round(intervals), c["pde_reference_grid"][1], maximum, dates)
        domains.append(dict(s_max=maximum, space_steps=round(intervals), spacing=spacing,
                            value=result.value, difference_from_main=result.value - pde_ref.value))

    print("Measuring European Monte Carlo error over independent repetitions", flush=True)
    mc_rows = []
    for n in c["mc_paths"]:
        for antithetic in [False, True]:
            for rep in range(c["mc_repetitions"]):
                seed = next(mc_seeds)
                tick = perf_counter()
                estimate = european_monte_carlo(market, option, n, np.random.default_rng(seed), antithetic)
                mc_rows.append(dict(paths=n, antithetic=antithetic, repetition=rep,
                                    seed_key=".".join(map(str, seed.spawn_key)), **asdict(estimate),
                                    error=estimate.value - exact,
                                    interval_contains_analytical=estimate.lower <= exact <= estimate.upper,
                                    seconds=perf_counter() - tick))

    print("Training and independently evaluating exercise policies", flush=True)
    policy_rows = []
    snapshot = None
    for rep, rep_seed in enumerate(policy_seeds):
        train_seeds = rep_seed.spawn(1 + len(c["training_paths"]) * len(c["degrees"]))
        evaluation_seed = train_seeds[0]
        times = np.insert(dates, 0, 0)
        evaluation = simulate_gbm(market, times, c["evaluation_paths"], np.random.default_rng(evaluation_seed))
        training_iterator = iter(train_seeds[1:])
        for n in c["training_paths"]:
            for degree in c["degrees"]:
                seed = next(training_iterator)
                tick = perf_counter()
                policy = train_policy(market, option, dates, n, degree, np.random.default_rng(seed))
                training_seconds = perf_counter() - tick
                tick = perf_counter()
                result = evaluate_policy(policy, evaluation)
                evaluation_seconds = perf_counter() - tick
                fits = [fit for fit in policy.regressions[1:-1] if fit is not None]
                row = dict(repetition=rep, training_paths=n, degree=degree,
                           training_seed_key=".".join(map(str, seed.spawn_key)),
                           evaluation_seed_key=".".join(map(str, evaluation_seed.spawn_key)),
                           training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                           raw_value=result.raw.value, raw_se=result.raw.standard_error,
                           **asdict(result.european_control),
                           difference_from_pde=result.european_control.value - pde_ref.value,
                           early_exercise_fraction=float(np.mean(result.stopping_indices < len(times) - 1)),
                           largest_condition_number=max((fit.condition_number for fit in fits), default=0),
                           reduced_degree_fits=sum(len(fit.coefficients) < degree + 1 for fit in fits),
                           missing_fits=len(policy.regressions[1:-1]) - len(fits))
                policy_rows.append(row)
                if rep == 0 and n == max(c["training_paths"]) and degree == 3:
                    snapshot = policy
        print(f"  Completed policy repetition {rep + 1}/{c['policy_repetitions']}", flush=True)

    economics = []
    for rate in c["rates"]:
        for volatility in c["volatilities"]:
            scenario = replace(market, rate=rate, volatility=volatility)
            euro = black_scholes(scenario, option)
            berm = finite_difference_price(scenario, option, *c["pde_reference_grid"], c["s_max"], dates).value
            economics.append(dict(rate=rate, volatility=volatility, european=euro, bermudan=berm,
                                  premium=berm - euro))
    calendars = []
    for count in c["exercise_counts"]:
        calendar = np.arange(1, count + 1) * option.maturity / count
        result = finite_difference_price(market, option, *c["pde_reference_grid"], c["s_max"], calendar)
        calendars.append(dict(exercise_dates=count, value=result.value, premium=result.value - exact))

    for filename, rows in [("grid_convergence.csv", grids), ("domain_sensitivity.csv", domains),
                           ("european_mc.csv", mc_rows), ("policy_evaluation.csv", policy_rows),
                           ("economic_sensitivity.csv", economics), ("exercise_calendars.csv", calendars)]:
        _write_table(output / filename, rows)
    _write_json(output / "configuration.json", c)
    if snapshot is not None:
        _write_json(output / "policy_snapshot.json", asdict(snapshot))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for method, marker in [("CRR", "o"), ("PDE", "s")]:
        points = [r for r in grids if r["contract"] == "European" and r["method"] == method]
        ax[0].loglog([r["time_steps"] for r in points], [abs(r["error"]) for r in points], marker=marker, label=method)
    ax[0].set(title="European numerical error", xlabel="Time steps (PDE also refines space)", ylabel="Absolute price error")
    ax[0].legend()
    for offset, degree in enumerate(c["degrees"]):
        points = [r for r in policy_rows if r["degree"] == degree]
        x = [r["training_paths"] * (1 + 0.04 * (offset - 1.5)) for r in points]
        ax[1].scatter(x, [r["value"] for r in points], label=f"Degree {degree}", s=21, alpha=0.75)
    ax[1].axhline(pde_ref.value, color="black", ls="--", lw=1, label="PDE reference")
    ax[1].set(xscale="log", title="Independent policy values", xlabel="Training paths (offset for visibility)", ylabel="Value with European control")
    ax[1].legend(fontsize=8, ncol=2)
    for result, name in [(tree_ref, "CRR reference"), (pde_ref, "PDE reference")]:
        ax[2].plot(list(result.exercise_boundary), list(result.exercise_boundary.values()), "o-", label=name, markersize=4)
    ax[2].set(title="Monthly put exercise boundary", xlabel="Calendar time (years)", ylabel="Highest exercising grid spot")
    ax[2].legend()
    fig.savefig(output / "validation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.3), layout="constrained")
    summaries = []
    for antithetic in [False, True]:
        rmse = []
        for n in c["mc_paths"]:
            points = [r for r in mc_rows if r["paths"] == n and r["antithetic"] == antithetic]
            error = np.array([r["error"] for r in points])
            rmse.append(float(np.sqrt(np.mean(error**2))))
            summaries.append(dict(paths=n, antithetic=antithetic, rmse=rmse[-1],
                                  coverage=float(np.mean([r["interval_contains_analytical"] for r in points])),
                                  repetitions=len(points)))
        ax.loglog(c["mc_paths"], rmse, "o-", label="Antithetic" if antithetic else "Plain")
    reference_curve = summaries[0]["rmse"] * np.sqrt(c["mc_paths"][0] / np.array(c["mc_paths"]))
    ax.loglog(c["mc_paths"], reference_curve, "--", color="grey", label="N^(-1/2), anchored to first plain point")
    ax.set(title="European Monte Carlo error across repetitions", xlabel="Total terminal paths", ylabel="Root mean squared error")
    ax.legend(fontsize=9)
    fig.savefig(output / "monte_carlo.png")
    plt.close(fig)
    _write_table(output / "mc_summary.csv", summaries)

    manifest = dict(purpose=c["purpose"], input_origin=c["input_origin"],
                    generated_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                    platform=platform.platform(), cpu=_cpu_description(), logical_cpus=os.cpu_count(),
                    dependencies={name: version(name) for name in ["numpy", "scipy", "matplotlib", "pytest"]},
                    thread_environment={k: os.environ.get(k, "unset") for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
                    random_generator="NumPy PCG64 via default_rng and SeedSequence.spawn",
                    seed_group_roles={"0": "European MC", "1": "LSM repetitions"},
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src/qf_research").rglob("*.py"))},
                    total_seconds=perf_counter() - started,
                    references={"european_analytical": exact, "bermudan_tree": tree_ref.value, "bermudan_pde": pde_ref.value})
    _write_json(output / "manifest.json", manifest)
    _write_findings(output, c, grids, domains, policy_rows, summaries, calendars, economics, manifest)
    print(f"Wrote results to {output} in {manifest['total_seconds']:.1f}s", flush=True)


def _write_findings(output, c, grids, domains, policies, mc, calendars, economics, manifest):
    ref = manifest["references"]
    rows = ["# Preliminary numerical validation record", "",
            "**Stage 1 prototype. These are mathematical test fixtures and model simulations, not observed market data or empirical research findings.** The inputs are analytically specified to check implementation. The market-data study is still to be developed under the research design.", "",
            "Generated by the benchmark command from the saved configuration. Prices are in the same arbitrary currency units as spot and strike. Simulated paths follow risk-neutral geometric Brownian motion.", "",
            "## Contract and numerical references", "",
            f"The {c['option']['kind']} has spot {c['market']['spot']:g}, strike {c['option']['strike']:g}, maturity {c['option']['maturity']:g} year(s), rate {c['market']['rate']:.1%}, volatility {c['market']['volatility']:.1%}, dividend yield {c['market']['dividend']:.1%} and {c['exercise_count']} equally spaced positive exercise dates. Exercise at time zero is excluded.", "",
            "| Reference | Value |", "| --- | ---: |",
            f"| European analytical | {ref['european_analytical']:.8f} |",
            f"| Bermudan CRR, {c['tree_reference_steps']} steps | {ref['bermudan_tree']:.8f} |",
            f"| Bermudan PDE, {c['pde_reference_grid'][0]} space intervals and {c['pde_reference_grid'][1]} time steps | {ref['bermudan_pde']:.8f} |", "",
            f"The two Bermudan references differ by {abs(ref['bermudan_tree'] - ref['bermudan_pde']):.6f}. This is evidence of numerical agreement, not a certified error bound. The largest domain-check difference at fixed spot spacing is {max(abs(r['difference_from_main']) for r in domains):.3g}.", "",
            "## Refinement", "", "| Method | Space intervals | Time steps | European error | Bermudan difference from fine PDE |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for row in [r for r in grids if r["contract"] == "European"]:
        pair = next(r for r in grids if r["contract"] == "Bermudan" and r["method"] == row["method"] and r["time_steps"] == row["time_steps"])
        rows.append(f"| {row['method']} | {row['space_steps']} | {row['time_steps']} | {row['error']:+.6f} | {pair['error']:+.6f} |")
    rows += ["", "CRR space resolution is determined by its time step; zero in its space column means no separate spatial-grid input. The PDE refinement above changes both space and time, so it does not isolate their individual orders.", "",
             "![Numerical validation and exercise boundaries](validation.png)", "", "## Policy approximation", "",
             "Every row below averages independently trained policies across the configured repetitions. Within each repetition, all policies share one independent evaluation sample. Configurations were specified before this run; these results do not select a winning configuration for a further performance claim.", "",
             "| Training paths | Degree | Mean policy value | Mean difference from PDE | SD of repeated values | Mean training seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in c["training_paths"]:
        for degree in c["degrees"]:
            subset = [r for r in policies if r["training_paths"] == n and r["degree"] == degree]
            values = np.array([r["value"] for r in subset])
            rows.append(f"| {n} | {degree} | {values.mean():.6f} | {values.mean() - ref['bermudan_pde']:+.6f} | {values.std(ddof=1):.6f} | {np.mean([r['training_seconds'] for r in subset]):.3f} |")
    rows += ["", "Values use a fixed European control variate. Individual raw estimates, sampling standard errors, approximate 95% intervals, seeds, regression conditioning and exercise fractions are in `policy_evaluation.csv`. Those intervals concern evaluation noise conditional on a fitted policy. The repeated-value standard deviation includes both new training and new evaluation samples and does not isolate training variability.", "",
             "A policy's population value cannot exceed the optimal value for this model and calendar. A finite sample estimate can exceed a numerical reference. Consequently a negative estimated shortfall is not evidence of beating the optimum, and these computations do not certify policy loss.", "",
             "## Monte Carlo", "", "| Paths | Sampling | RMSE | Interval coverage | Repetitions |", "| ---: | --- | ---: | ---: | ---: |"]
    for row in mc:
        rows.append(f"| {row['paths']} | {'Antithetic' if row['antithetic'] else 'Plain'} | {row['rmse']:.6f} | {row['coverage']:.1%} | {row['repetitions']} |")
    rows += ["", "![Monte Carlo convergence](monte_carlo.png)", "",
             "Coverage is an empirical fraction across a modest number of repetitions, not a guarantee. Antithetic uncertainty uses independent pair averages. The reference slope follows the finite-variance Monte Carlo rate; three sample sizes do not establish an asymptotic convergence theorem.", "",
             "## Exercise opportunities and economic mechanisms", "", "| Exercise dates | Value | Difference from analytical European |", "| ---: | ---: | ---: |"]
    for row in calendars:
        rows.append(f"| {row['exercise_dates']} | {row['value']:.6f} | {row['premium']:.6f} |")
    rows += ["", "These are different contracts. Additional exercise dates expand the feasible set of stopping rules. Their value differences must not be described as Monte Carlo or regression error. The one-date PDE difference from the analytical value measures numerical error.", "",
             "| Rate | Volatility | European value | Monthly Bermudan value | Early-exercise premium |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in economics:
        rows.append(f"| {row['rate']:.1%} | {row['volatility']:.1%} | {row['european']:.6f} | {row['bermudan']:.6f} | {row['premium']:.6f} |")
    rows += ["", "For the non-dividend-paying put, exercise exchanges exposure to future stock movements for receipt of the strike less the current stock value. Positive interest rates make earlier receipt of the strike attractive, while preserving the option retains downside protection and convexity. Changing volatility affects the value of that protection. The premium therefore requires joint analysis of timing and uncertainty; it is not inferred from the European vega alone.", "",
             "At zero rates and zero dividends, Jensen's inequality for the convex payoff of a martingale implies no benefit from early exercise. Small negative premiums against the analytical European benchmark here are discretisation residuals, not economic losses from having more rights.", "",
             "## What this establishes and what remains", "",
             "This prototype provides a reproducible numerical check under one diffusion model, a test contract and controlled sensitivities. It provides preliminary evidence for implementation accuracy. It does not establish a universally best regression basis, continuous-time American value, martingale dual bounds, market calibration, hedging performance or a completed research contribution.", "",
             "Next experiments should isolate time and space refinement, add more independent training repetitions, evaluate policy differences using paired cash flows, measure loss with validated dual bounds, and test stress cases before adding model complexity.", "",
             f"Run environment: Python {manifest['python']}, {manifest['cpu']}, {manifest['platform']}. Total measured runtime: {manifest['total_seconds']:.1f} seconds. Per-method timings are single-process wall-clock observations on this machine; they include no peak-memory measurement. Dependency versions, source hashes and random-stream roles are recorded in `manifest.json`.", ""]
    (output / "findings.md").write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("projects/01_optimal_stopping/configs/benchmark.json"))
    parser.add_argument("--output", type=Path, default=Path("projects/01_optimal_stopping/results/baseline"))
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
