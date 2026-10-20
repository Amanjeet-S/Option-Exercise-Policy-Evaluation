"""Finite-sample clipped-gap upper limits with analytic lognormal tails.

Fixed caps are selected on an independent pilot and saved before evaluation.
The evaluation reuses the registered public-process streams of the original
coverage diagnostic. Hoeffding and Maurer-Pontil bounds concern clipped iid
observations; analytic tail allowances extend them to the unbounded gap mean.
The guarantees are mathematical statements under the stipulated process, not
floating-point certificates or rigorous enclosures of the numerical oracle.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from qf_research.optimal_stopping.contracts import positive_integer
from validate_gap_coverage import (
    GapProcess, analytic_second_moment_bounds, coverage_record, gap_samples,
    numerical_oracle, undiscounted_call, upper_limits,
)


THEOREM_REVIEW = dict(
    authors="Andreas Maurer and Massimiliano Pontil", year=2009,
    title="Empirical Bernstein Bounds and Sample Variance Penalization", venue="COLT 2009",
    primary_url="https://www.cs.mcgill.ca/~colt2009/papers/012.pdf", version="Final published COLT 2009 paper",
    checked_sections=["Theorem 1", "Theorem 4", "The sample-variance definition following Theorem 4"],
    physical_pdf_pages=dict(theorem_1=1, theorem_4=2, sample_variance_definition=2),
    conditions="Independent identically distributed observations in [0,1]; use n>=2 and 0<alpha<1",
    hoeffding_scaled_radius="b*sqrt(log(1/alpha)/(2*n))",
    empirical_bernstein_scaled_radius="sqrt(2*s_squared*log(2/alpha)/n)+7*b*log(2/alpha)/(3*(n-1))",
    variance_convention="s_squared=sum((Y_i-mean(Y))^2)/(n-1), equivalent to the published pairwise definition",
    scaling="Apply each theorem to clipped observations divided by their fixed cap b",
    cap_selection="Independent pilot, followed by conditioning on its frozen cap; no evaluation-based cap tuning",
    scope="Existing concentration inequalities. No new general theorem is claimed.")


def envelope_coefficients(process, martingale, *, shifted=True):
    """G <= (scale*S_1-shift)^+ by conditional Jensen.

    The zero gap is bounded by H_1. For the terminal-call gap, conditional
    Jensen gives C_1 >= exp(-r*t_1)*(S_1*exp(-q*h)-K*exp(-r*h))^+.
    Splitting according to the sign inside this latter positive part gives
    G <= (a*S_1-c)^+, with c=exp(-r*t_1)*K*(1-exp(-r*h)). Dropping c
    recovers the simpler requested envelope a*S_1.
    """
    if martingale == "zero":
        scale = math.exp(-process.rate*process.first_right)
        return scale, scale*process.strike
    if martingale != "terminal_call":
        raise ValueError("Unknown martingale")
    horizon = process.maturity-process.first_right
    scale = math.exp(-process.rate*process.first_right)*(-math.expm1(-process.dividend_yield*horizon))
    shift = math.exp(-process.rate*process.first_right)*process.strike*(-math.expm1(-process.rate*horizon))
    return scale, shift if shifted else 0.


def analytic_tail_allowance(process, martingale, cap, *, shifted=True):
    """E[(G-b)^+] <= E[(scale*S_1-shift-b)^+] in closed form.

    The right side is a scaled undiscounted lognormal call expectation. The
    cap is deterministic conditional on the independent pilot. No observed
    evaluation maximum or estimated tail frequency enters this allowance.
    """
    if isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
        raise ValueError("A finite non-negative cap is required")
    scale, shift = envelope_coefficients(process, martingale, shifted=shifted)
    if scale == 0.:
        return 0.
    strike = (cap+shift)/scale
    if strike == 0.:
        return scale*process.spot*math.exp((process.rate-process.dividend_yield)*process.first_right)
    return max(0., scale*float(undiscounted_call(process.spot, strike, process.first_right, process)))


def bounded_upper_limits(clipped, cap, tail_allowance, *, alpha=.025):
    """Finite-sample upper limits for an unbounded mean from bounded samples."""
    clipped = np.asarray(clipped, dtype=float)
    if (clipped.ndim != 2 or clipped.shape[1] < 2 or not np.all(np.isfinite(clipped)) or
            isinstance(cap, bool) or not math.isfinite(cap) or cap < 0 or
            np.any(clipped < 0.) or np.any(clipped > cap)):
        raise ValueError("At least two finite clipped observations in [0,cap] per repetition required")
    if (not math.isfinite(alpha) or not 0 < alpha < 1 or
            not math.isfinite(tail_allowance) or tail_allowance < 0):
        raise ValueError("Valid alpha and deterministic non-negative tail allowance required")
    count = clipped.shape[1]
    mean = clipped.mean(axis=1)
    variance = clipped.var(axis=1, ddof=1)
    hoeffding = cap*math.sqrt(math.log(1/alpha)/(2*count))
    log_probability = math.log(2/alpha)
    empirical = np.sqrt(2*variance*log_probability/count)+7*cap*log_probability/(3*(count-1))
    return mean, dict(hoeffding=mean+hoeffding+tail_allowance,
                     empirical_bernstein=mean+empirical+tail_allowance)


def select_caps(pilot, caps, outer_counts, process, *, alpha=.025):
    """Freeze one cap per method/count using only independent pilot payoffs.

    The declared objective is the pilot forecast of the upper limit, including
    its deterministic tail allowance. Inaccurate pilot forecasts affect width,
    not validity of the subsequent bound conditional on the chosen cap.
    """
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("A finite tail probability in (0,1) is required")
    if (not caps or list(caps) != sorted(set(caps)) or
            any(not math.isfinite(cap) or cap <= 0 for cap in caps)):
        raise ValueError("A fixed increasing positive cap grid is required")
    records = []
    for martingale, samples in pilot.items():
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1 or len(samples) < 2 or np.any(~np.isfinite(samples)) or np.any(samples < 0):
            raise ValueError("Independent finite non-negative pilot payoffs required")
        statistics = []
        for cap in caps:
            clipped = np.minimum(samples, cap)
            statistics.append(dict(cap=cap, clipped_mean=float(clipped.mean()),
                clipped_variance=float(clipped.var(ddof=1)),
                tail_allowance=analytic_tail_allowance(process, martingale, cap),
                unshifted_tail_allowance=analytic_tail_allowance(process, martingale, cap, shifted=False)))
        for count in outer_counts:
            positive_integer(count, "outer_counts", 2)
            for method in ("hoeffding", "empirical_bernstein"):
                def forecast(record):
                    if method == "hoeffding":
                        radius = record["cap"]*math.sqrt(math.log(1/alpha)/(2*count))
                    else:
                        logarithm = math.log(2/alpha)
                        radius = math.sqrt(2*record["clipped_variance"]*logarithm/count)
                        radius += 7*record["cap"]*logarithm/(3*(count-1))
                    return record["clipped_mean"]+radius+record["tail_allowance"]
                selected = min(statistics, key=lambda record: (forecast(record), record["cap"]))
                records.append(dict(martingale=martingale, method=method, outer_paths=count,
                    **selected, pilot_forecast_upper=forecast(selected)))
    return records


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clock_record(started, cpu):
    return dict(elapsed_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)


def baseline_replay_status(record, original):
    """Preserve exact counts/means and disclose stable-formula roundoff changes."""
    exact_fields = ("expected_gap_covered", "policy_loss_covered", "average_sample_mean")
    if any(record[field] != original[field] for field in exact_fields):
        raise ValueError("Original coverage baseline counts or sample means did not replay exactly")
    difference = record["average_upper_limit"]-original["average_upper_limit"]
    tolerance = 16*max(math.ulp(record["average_upper_limit"]), math.ulp(original["average_upper_limit"]))
    if not math.isfinite(difference) or abs(difference) > tolerance:
        raise ValueError("Original upper-limit summary changed beyond the declared roundoff tolerance")
    return dict(original_baseline_replayed_exactly=difference == 0.,
                original_baseline_replayed_to_roundoff=True,
                original_average_upper_difference=difference,
                original_average_upper_roundoff_tolerance=tolerance)


def simulate_pairs(rng, process, shape):
    z = rng.standard_normal((*shape, 2))
    drift = process.rate-process.dividend_yield-.5*process.volatility**2
    first = process.spot*np.exp(drift*process.first_right
                               +process.volatility*math.sqrt(process.first_right)*z[..., 0])
    horizon = process.maturity-process.first_right
    terminal = first*np.exp(drift*horizon+process.volatility*math.sqrt(horizon)*z[..., 1])
    return first, terminal


def run(args):
    output, original = args.output.resolve(), args.original.resolve()
    if not output.is_relative_to((ROOT/"data/private").resolve()):
        raise ValueError("Keep results in this checkout's data/private")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    positive_integer(args.pilot_paths, "pilot_paths", 2)
    original_manifest = read(original/"manifest.json")
    for name in ("design.json", "results.json", "oracle.json"):
        if sha(original/name) != original_manifest["artifact_sha256"][name]:
            raise ValueError("Original diagnostic artefact checksum mismatch")
    old_design, old_results = read(original/"design.json"), read(original/"results.json")
    process = GapProcess(**old_design["process"])
    counts, repetitions = old_design["outer_counts"], old_design["repetitions"]
    input_paths = [original/name for name in ("manifest.json", "design.json", "results.json", "oracle.json")]
    source_paths = [Path(__file__), ROOT/"scripts/validate_gap_coverage.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    input_hashes = {str(path): sha(path) for path in input_paths}
    output.mkdir(parents=True)
    started, cpu = time.perf_counter(), time.process_time()
    caps = args.caps
    if caps != sorted(set(caps)) or any(cap <= 0 or not math.isfinite(cap) for cap in caps):
        raise ValueError("A fixed increasing positive cap grid is required")
    design = dict(original_design_sha256=sha(original/"design.json"), process=old_design["process"],
        outer_counts=counts, repetitions=repetitions, root_seed=old_design["root_seed"],
        evaluation_streams=old_design["streams"], repetition_block=old_design["repetition_block"],
        pilot_paths=args.pilot_paths, pilot_spawn_key=[1000], bit_generator="PCG64", caps=caps,
        alpha=.025, methods=["student", "cantelli", "hoeffding", "empirical_bernstein"],
        cap_selection="Independent pilot minimises forecast upper value over the declared cap grid",
        shifted_jensen_envelope=True, cap_selection_uses_evaluation=False,
        baseline_replay_rule="Coverage counts and sample means exact; average upper limit within 16 ulps; actual difference recorded",
        centred_second_moment_formula="Cancellation-stable -expm1(-q*h), mathematically equivalent to 1-exp(-q*h)",
        cost_scope="Incremental comparison runtime; includes pilot, cap selection and evaluation; reuses the retained numerical oracle without charging its historical calculation",
        source_sha256=source_hashes, input_sha256=input_hashes,
        public_preregistration_claim=False, new_general_theorem_claim=False)
    save(output/"design.json", design)
    save(output/"registration.json", dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=sha(output/"design.json"), execution_started=False,
        public_preregistration_claim=False))
    save(output/"theorem_review.json", THEOREM_REVIEW)
    pilot_started, pilot_cpu = time.perf_counter(), time.process_time()
    pilot_seed = np.random.SeedSequence(old_design["root_seed"], spawn_key=(1000,))
    rng = np.random.Generator(np.random.PCG64(pilot_seed))
    first, terminal = simulate_pairs(rng, process, (args.pilot_paths,))
    pilot = gap_samples(first, terminal, process)
    pilot_cost = clock_record(pilot_started, pilot_cpu)
    selection_started, selection_cpu = time.perf_counter(), time.process_time()
    selected = select_caps(pilot, caps, counts, process)
    selection_cost = clock_record(selection_started, selection_cpu)
    save(output/"frozen_caps.json", dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_started=False, selected=selected, pilot_cost=pilot_cost,
        cap_selection_cost=selection_cost, pilot_spawn_key=[1000]))
    del first, terminal, pilot
    oracle = old_results["oracle"]
    moment_bounds = analytic_second_moment_bounds(process)
    records, evaluation_costs = [], []
    for index, count in enumerate(counts):
        count_started, count_cpu = time.perf_counter(), time.process_time()
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
            old_design["root_seed"], spawn_key=(index,))))
        arrays = {(martingale, method): {"raw_mean": [], "mean": [], "upper": []}
                  for martingale in ("zero", "terminal_call") for method in design["methods"]}
        completed = 0
        while completed < repetitions:
            size = min(old_design["repetition_block"], repetitions-completed)
            first, terminal = simulate_pairs(rng, process, (size, count))
            samples = gap_samples(first, terminal, process)
            for martingale, values in samples.items():
                mean, limits = upper_limits(values, moment_bounds[martingale])
                for method, upper in limits.items():
                    arrays[martingale, method]["raw_mean"].append(mean)
                    arrays[martingale, method]["mean"].append(mean)
                    arrays[martingale, method]["upper"].append(upper)
                for method in ("hoeffding", "empirical_bernstein"):
                    setting = next(record for record in selected if
                                   (record["martingale"], record["method"], record["outer_paths"])
                                   == (martingale, method, count))
                    clipped = np.minimum(values, setting["cap"])
                    clipped_mean, clipped_limits = bounded_upper_limits(
                        clipped, setting["cap"], setting["tail_allowance"])
                    arrays[martingale, method]["raw_mean"].append(mean)
                    arrays[martingale, method]["mean"].append(clipped_mean)
                    arrays[martingale, method]["upper"].append(clipped_limits[method])
            completed += size
        for (martingale, method), parts in arrays.items():
            raw_mean, mean, upper = (np.concatenate(parts[name]) for name in ("raw_mean", "mean", "upper"))
            target = oracle["expected_gaps"][martingale]["value"]
            record = dict(martingale=martingale, method=method, outer_paths=count,
                expected_gap_target=target, policy_loss_target=oracle["true_two_right_policy_loss"],
                **coverage_record(upper, mean, target, oracle["true_two_right_policy_loss"]),
                mean_excess_over_expected_gap=float(upper.mean()-target),
                mean_upper_minus_unclipped_sample_mean=float((upper-raw_mean).mean()))
            if method in ("student", "cantelli"):
                old = next(old for old in old_results["records"] if
                           (old["martingale"], old["method"], old["outer_paths"]) == (martingale, method, count))
                record.update(baseline_replay_status(record, old))
            else:
                setting = next(record for record in selected if
                               (record["martingale"], record["method"], record["outer_paths"])
                               == (martingale, method, count))
                record["cap"] = setting["cap"]
                record["tail_allowance"] = setting["tail_allowance"]
                record["unshifted_tail_allowance"] = setting["unshifted_tail_allowance"]
                record["upper_minus_sample_mean_base"] = "Clipped sample mean; radius includes deterministic tail allowance"
            records.append(record)
        evaluation_costs.append(dict(outer_paths=count, **clock_record(count_started, count_cpu)))
        save(output/"partial_results.json", records)
        print(f"Completed clipped-bound comparison: {repetitions} repetitions at n={count}", flush=True)
    result = dict(records=records, oracle=oracle, selected_caps=selected,
        mathematical_one_sided_coverage=.975, finite_sample_methods=["cantelli", "hoeffding", "empirical_bernstein"],
        mathematical_guarantees_conditional_on_model_and_pilot=True,
        floating_point_bounds_are_certified=False, numerical_oracle_is_rigorous_enclosure=False,
        cap_selection_uses_evaluation=False, simultaneous_claim_across_counts_or_methods=False,
        new_general_theorem_claim=False, theorem_review=THEOREM_REVIEW,
        cost_scope=design["cost_scope"], reused_numerical_oracle=True,
        costs=dict(pilot=pilot_cost, cap_selection=selection_cost, evaluation=evaluation_costs,
                   total=clock_record(started, cpu)),
        new_market_data_acquired=False, publication_clearance=False)
    save(output/"results.json", result)
    with (output/"figure_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["martingale", "method", "outer_paths", "coverage", "coverage_lower",
            "coverage_upper", "average_upper", "mean_radius", "mean_excess", "cap", "tail_allowance"])
        writer.writeheader()
        for record in records:
            interval = record["expected_gap_coverage_wilson_95"]
            writer.writerow(dict(martingale=record["martingale"], method=record["method"], outer_paths=record["outer_paths"],
                coverage=record["expected_gap_coverage"], coverage_lower=interval[0], coverage_upper=interval[1],
                average_upper=record["average_upper_limit"], mean_radius=record["upper_minus_sample_mean"]["mean"],
                mean_excess=record["mean_excess_over_expected_gap"], cap=record.get("cap"), tail_allowance=record.get("tail_allowance")))
    for path in input_paths+source_paths:
        expected = input_hashes[str(path)] if path in input_paths else source_hashes[str(path.relative_to(ROOT))]
        if sha(path) != expected:
            raise ValueError("Coverage inputs or source changed during execution")
    names = ["design.json", "registration.json", "theorem_review.json", "frozen_caps.json", "results.json", "figure_data.csv"]
    save(output/"manifest.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256=input_hashes, source_sha256=source_hashes,
        artifact_sha256={name: sha(output/name) for name in names}, costs=result["costs"],
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform()), publication_clearance=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--original", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pilot-paths", type=int, default=8192)
    result.add_argument("--caps", type=float, nargs="+", default=[1., 2., 4., 6., 8., 12., 16., 24., 32., 48., 64., 96., 128.])
    return result


if __name__ == "__main__":
    run(parser().parse_args())
