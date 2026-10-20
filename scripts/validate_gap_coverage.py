"""Repeated-sampling coverage for two-right call-policy gap estimators.

The process uses the published Andersen and Broadie (2004) parameters with a
deliberately modified two-right calendar. It is a diagnostic, not a replication
of their table. The stopping policy always waits. Exact conditional call values
give a one-dimensional quadrature target for its loss, allowing sampling
coverage to be measured without a PDE approximation to an unknown optimum.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
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
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from qf_research.optimal_stopping.contracts import positive_integer


@dataclass(frozen=True)
class GapProcess:
    spot: float = 100.
    strike: float = 100.
    rate: float = .05
    dividend_yield: float = .10
    volatility: float = .20
    first_right: float = 1.5
    maturity: float = 3.

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("Finite process parameters required")
        if (min(self.spot, self.strike, self.volatility, self.first_right) <= 0 or
                self.maturity <= self.first_right or self.rate < 0 or self.dividend_yield < 0):
            raise ValueError("Positive levels/volatility, ordered rights and non-negative yields required")


SOURCES = [
    dict(authors="Black and Scholes", year=1973,
         title="The Pricing of Options and Corporate Liabilities",
         url="https://www.journals.uchicago.edu/doi/10.1086/260062",
         role="Established lognormal conditional-call calculation"),
    dict(authors="Andersen and Broadie", year=2004,
         title="Primal-Dual Simulation Algorithm for Pricing Multidimensional American Options",
         url="https://doi.org/10.1287/mnsc.1040.0258",
         role="Published process parameters only; the two-right calendar is a new diagnostic design"),
]


def undiscounted_call(states, strike, horizon, process):
    """E[(S_(t+h)-K)^+ | S_t], including the continuous dividend yield."""
    states, strike = np.broadcast_arrays(np.asarray(states, dtype=float), np.asarray(strike, dtype=float))
    if np.any(states <= 0) or np.any(strike <= 0) or horizon <= 0:
        raise ValueError("Positive conditional states, strikes and horizon required")
    sd = process.volatility*math.sqrt(horizon)
    d2 = (np.log(states/strike)+(process.rate-process.dividend_yield-.5*process.volatility**2)*horizon)/sd
    return states*math.exp((process.rate-process.dividend_yield)*horizon)*ndtr(d2+sd)-strike*ndtr(d2)


def first_reward(states, process):
    return math.exp(-process.rate*process.first_right)*np.maximum(np.asarray(states)-process.strike, 0.)


def conditional_terminal_reward(states, process):
    return math.exp(-process.rate*process.maturity)*undiscounted_call(
        states, process.strike, process.maturity-process.first_right, process)


def centred_gap(states, process):
    states = np.asarray(states, dtype=float)
    if np.any(~np.isfinite(states)) or np.any(states <= 0):
        raise ValueError("Finite positive conditional states required")
    if process.dividend_yield == 0.:
        # Conditional Jensen and r>=0 imply C_1>=H_1 exactly. Avoid a
        # positive cancellation residue masquerading as early-exercise loss.
        return np.zeros_like(states)
    return np.maximum(first_reward(states, process)-conditional_terminal_reward(states, process), 0.)


def zero_gap_conditional_mean(states, process):
    """Conditional capped-call identity for E[(H_1-H_2)^+ | S_1]."""
    reward = first_reward(states, process)
    horizon = process.maturity-process.first_right
    cap_strike = process.strike+reward*math.exp(process.rate*process.maturity)
    conditional_spread = (undiscounted_call(states, process.strike, horizon, process)
                          -undiscounted_call(states, cap_strike, horizon, process))
    return np.maximum(reward-math.exp(-process.rate*process.maturity)*conditional_spread, 0.)


def gap_samples(first_states, terminal_states, process):
    """Both martingale gaps on the same paths for the fixed always-wait rule."""
    first_states, terminal_states = np.broadcast_arrays(first_states, terminal_states)
    reward = first_reward(first_states, process)
    terminal = math.exp(-process.rate*process.maturity)*np.maximum(terminal_states-process.strike, 0.)
    return {"zero": np.maximum(reward-terminal, 0.),
            "terminal_call": centred_gap(first_states, process)}


def analytic_second_moment_bounds(process):
    """Known raw second-moment upper bounds, not estimated variances.

    The zero-martingale gap is bounded by H_1 <= exp(-r*t_1)*S_1.
    Conditional Jensen and r,q >= 0 give the centred gap <= a*S_1,
    where a=exp(-r*t_1)*(1-exp(-q*(T-t_1))). The GBM second moment is
    E[S_1^2]=S_0^2*exp((2*(r-q)+sigma^2)*t_1).
    """
    second = process.spot**2*math.exp((2*(process.rate-process.dividend_yield)
                                     +process.volatility**2)*process.first_right)
    zero_a = math.exp(-process.rate*process.first_right)
    centred_a = zero_a*(-math.expm1(-process.dividend_yield*(process.maturity-process.first_right)))
    return dict(zero=zero_a**2*second, terminal_call=centred_a**2*second)


def numerical_oracle(process, *, normal_cutoff=12., epsabs=1e-10, epsrel=1e-11):
    """One-dimensional lognormal quadrature and an analytic omitted-tail bound.

    QUADPACK error estimates are disclosed numerical diagnostics. They are not
    certified integration enclosures. The omitted-tail expectation bound uses
    the common pathwise bound G <= exp(-r*t_1)*S_1 and the exact truncated
    lognormal first moment.
    """
    sd = process.volatility*math.sqrt(process.first_right)
    drift = (process.rate-process.dividend_yield-.5*process.volatility**2)*process.first_right
    to_state = lambda z: process.spot*math.exp(drift+sd*z)
    strike_z = (math.log(process.strike/process.spot)-drift)/sd
    boundary = None
    if process.dividend_yield > 0:
        objective = lambda state: float(first_reward(state, process)-conditional_terminal_reward(state, process))
        high = 2*process.strike
        while objective(high) <= 0:
            high *= 2
            if high > process.strike*1e8:
                raise ArithmeticError("Could not bracket the first-right exercise boundary")
        boundary = brentq(objective, process.strike, high, xtol=1e-11)
    boundary_z = None if boundary is None else (math.log(boundary/process.spot)-drift)/sd
    split = sorted({-normal_cutoff, normal_cutoff, *[z for z in (strike_z, boundary_z)
                    if z is not None and -normal_cutoff < z < normal_cutoff]})
    density = lambda z: math.exp(-.5*z*z)/math.sqrt(2*math.pi)
    targets = {}
    for name, function in (("zero", zero_gap_conditional_mean), ("terminal_call", centred_gap)):
        values = [quad(lambda z: float(function(to_state(z), process))*density(z), a, b,
                       epsabs=epsabs, epsrel=epsrel) for a, b in zip(split, split[1:])]
        targets[name] = dict(value=sum(value for value, _ in values),
                             quadrature_error_estimate=sum(error for _, error in values))
    expected_terminal = math.exp(-process.rate*process.maturity)*float(undiscounted_call(
        process.spot, process.strike, process.maturity, process))
    tail_bound = process.spot*math.exp(-process.dividend_yield*process.first_right)*(
        float(ndtr(-normal_cutoff-sd))+float(ndtr(sd-normal_cutoff)))
    for target in targets.values():
        target["omitted_tail_expectation_bound"] = tail_bound
    return dict(expected_gaps=targets, true_two_right_policy_loss=targets["terminal_call"]["value"],
        expiry_only_value=expected_terminal,
        two_right_optimal_value=expected_terminal+targets["terminal_call"]["value"],
        zero_martingale_dual_slack=targets["zero"]["value"]-targets["terminal_call"]["value"],
        terminal_call_martingale_dual_slack=0., exercise_boundary_at_first_right=boundary,
        integration=dict(variable="Standard normal innovation for S_1", range=[-normal_cutoff, normal_cutoff],
                         breakpoints=split, epsabs=epsabs, epsrel=epsrel),
        numerical_oracle_is_rigorous_enclosure=False)


def wilson_interval(successes, count, confidence=.95):
    if not isinstance(successes, (int, np.integer)) or not 0 <= successes <= count or count < 1:
        raise ValueError("Valid integer coverage counts required")
    z = float(norm.ppf((1+confidence)/2))
    rate = successes/count
    denominator = 1+z*z/count
    centre = (rate+z*z/(2*count))/denominator
    radius = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/denominator
    return [max(0., centre-radius), min(1., centre+radius)]


def upper_limits(samples, second_moment_bound, alpha=.025):
    """Student and Cantelli upper mean limits, with separate justifications.

    For Cantelli, Var(bar G) <= B/n and
    P(mu > bar G + sqrt(B*(1-alpha)/(alpha*n))) <= alpha.
    This is a finite-sample inequality under independent paths and the stated
    analytic B. Student limits retain their usual approximate justification.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2 or np.any(~np.isfinite(samples)):
        raise ValueError("Finite repetitions-by-paths samples required")
    if not 0 < alpha < 1 or not math.isfinite(second_moment_bound) or second_moment_bound < 0:
        raise ValueError("Valid tail probability and analytic second-moment bound required")
    n = samples.shape[1]
    mean = samples.mean(axis=1)
    student_radius = float(student_t.ppf(1-alpha, n-1))*samples.std(axis=1, ddof=1)/math.sqrt(n)
    cantelli_radius = math.sqrt(second_moment_bound*(1-alpha)/(alpha*n))
    return mean, dict(student=mean+student_radius, cantelli=mean+cantelli_radius)


def coverage_record(upper, mean, expected_gap, policy_loss):
    count = len(upper)
    successes = int(np.count_nonzero(upper >= expected_gap))
    policy_successes = int(np.count_nonzero(upper >= policy_loss))
    widths = upper-mean
    return dict(repetitions=count, expected_gap_covered=successes,
        expected_gap_coverage=successes/count, expected_gap_coverage_wilson_95=wilson_interval(successes, count),
        policy_loss_covered=policy_successes, policy_loss_coverage=policy_successes/count,
        policy_loss_coverage_wilson_95=wilson_interval(policy_successes, count),
        upper_minus_sample_mean=dict(mean=float(widths.mean()), median=float(np.median(widths)),
                                     p05=float(np.quantile(widths, .05)), p95=float(np.quantile(widths, .95))),
        average_upper_limit=float(upper.mean()), average_sample_mean=float(mean.mean()))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/"data/private").resolve()):
        raise ValueError("Keep the diagnostic results under this checkout's data/private")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    positive_integer(args.repetitions, "repetitions", 2)
    positive_integer(args.repetition_block, "repetition_block", 1)
    for count in args.outer_counts:
        positive_integer(count, "outer_counts", 2)
    if args.outer_counts != sorted(set(args.outer_counts)):
        raise ValueError("Outer path counts must be unique and increasing")
    process = GapProcess()
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_hash = sha(Path(__file__))
    design = dict(process=asdict(process), exercise_dates=[process.first_right, process.maturity],
        policy="Always wait until the terminal exercise right", martingales=["zero", "terminal_call"],
        martingale_formulas=dict(zero="M_1=M_2=0",
            terminal_call="M_1=C_1-E[H_2], M_2=H_2-E[H_2], C_1=E[H_2|S_1]"),
        gap_formulas=dict(zero="(H_1-H_2)^+", terminal_call="(H_1-C_1)^+"),
        zero_gap_conditional_mean="H_1-exp(-r*T)*(undiscounted_call(S_1,K,h)-undiscounted_call(S_1,K+H_1*exp(r*T),h))",
        root_seed=args.root_seed, bit_generator="PCG64", repetitions=args.repetitions,
        outer_counts=args.outer_counts, repetition_block=args.repetition_block,
        streams=[dict(outer_paths=count, entropy=args.root_seed, spawn_key=[index])
                 for index, count in enumerate(args.outer_counts)],
        stream_layout="Each count has an independent stream. Consecutive (n,2) arrays define independent complete repetitions; both martingales share each path.",
        one_sided_confidence=.975, coverage_interval_confidence=.95, sources=SOURCES,
        published_parameter_process=True, modified_calendar=True,
        replication_of_Andersen_Broadie_Table_2=False, new_general_theorem_claim=False)
    save(output/"design.json", design)
    # This persistent record is created before oracle calculation or simulation.
    save(output/"registration.json", dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        design_sha256=sha(output/"design.json"), source_sha256=source_hash,
        execution_started=False, public_preregistration_claim=False))
    oracle = numerical_oracle(process)
    bounds = analytic_second_moment_bounds(process)
    save(output/"oracle.json", dict(**oracle, analytic_second_moment_bounds=bounds))
    records = []
    for index, count in enumerate(args.outer_counts):
        seed = np.random.SeedSequence(args.root_seed, spawn_key=(index,))
        rng = np.random.Generator(np.random.PCG64(seed))
        accumulators = {martingale: {method: {"means": [], "upper": []}
                        for method in ("student", "cantelli")} for martingale in ("zero", "terminal_call")}
        zero_samples = {martingale: 0 for martingale in accumulators}
        completed = 0
        while completed < args.repetitions:
            size = min(args.repetition_block, args.repetitions-completed)
            z = rng.standard_normal((size, count, 2))
            first = process.spot*np.exp((process.rate-process.dividend_yield-.5*process.volatility**2)
                *process.first_right+process.volatility*math.sqrt(process.first_right)*z[:, :, 0])
            horizon = process.maturity-process.first_right
            terminal = first*np.exp((process.rate-process.dividend_yield-.5*process.volatility**2)
                *horizon+process.volatility*math.sqrt(horizon)*z[:, :, 1])
            samples = gap_samples(first, terminal, process)
            for martingale, values in samples.items():
                zero_samples[martingale] += int(np.count_nonzero(np.all(values == 0., axis=1)))
                means, limits = upper_limits(values, bounds[martingale])
                for method, upper in limits.items():
                    accumulators[martingale][method]["means"].append(means)
                    accumulators[martingale][method]["upper"].append(upper)
            completed += size
        for martingale, methods in accumulators.items():
            for method, arrays in methods.items():
                means, upper = np.concatenate(arrays["means"]), np.concatenate(arrays["upper"])
                records.append(dict(martingale=martingale, method=method, outer_paths=count,
                    expected_gap_target=oracle["expected_gaps"][martingale]["value"],
                    policy_loss_target=oracle["true_two_right_policy_loss"],
                    all_zero_sample_repetitions=zero_samples[martingale],
                    **coverage_record(upper, means, oracle["expected_gaps"][martingale]["value"],
                                      oracle["true_two_right_policy_loss"])))
        save(output/"partial_results.json", records)
        print(f"Completed {args.repetitions} independent coverage repetitions at n={count}", flush=True)
    result = dict(records=records, oracle=oracle, analytic_second_moment_bounds=bounds,
        nominal_one_sided_coverage=.975, expected_gap_and_policy_loss_are_distinct_for_zero_martingale=True,
        cantelli_scope="Finite-sample inequality for the model's unknown mean, using an analytic second-moment upper bound. Reported simulation coverage uses a numerical oracle rather than a rigorous integration enclosure.",
        student_scope="Approximate Student upper mean limit; its coverage is measured rather than assumed.",
        new_market_data_acquired=False, new_general_theorem_claim=False,
        numerical_oracle_is_rigorous_enclosure=False, public_preregistration_claim=False)
    save(output/"results.json", result)
    fields = ["martingale", "method", "outer_paths", "expected_gap_coverage", "coverage_lower", "coverage_upper",
              "policy_loss_coverage", "mean_radius", "median_radius", "radius_p05", "radius_p95"]
    with (output/"figure_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            interval, widths = record["expected_gap_coverage_wilson_95"], record["upper_minus_sample_mean"]
            writer.writerow({key: record[key] for key in fields[:4]} | dict(
                coverage_lower=interval[0], coverage_upper=interval[1], policy_loss_coverage=record["policy_loss_coverage"],
                mean_radius=widths["mean"], median_radius=widths["median"], radius_p05=widths["p05"], radius_p95=widths["p95"]))
    if sha(Path(__file__)) != source_hash:
        raise ValueError("Coverage source changed during execution")
    save(output/"manifest.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started, source_sha256=source_hash,
        artifact_sha256={name: sha(output/name) for name in
                        ("design.json", "registration.json", "oracle.json", "results.json", "figure_data.csv")},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform()), publication_clearance=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--repetitions", type=int, default=5000)
    result.add_argument("--outer-counts", type=int, nargs="+", default=[8, 32, 128, 512, 2048])
    result.add_argument("--root-seed", type=int, default=202610042)
    result.add_argument("--repetition-block", type=int, default=100)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
