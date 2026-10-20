"""Compare grid, frozen fitted and newly trained rules on licensed model inputs.

This is a new conditional numerical study, not a replay of the archived design.
All observations and derived outputs stay private. Grid/PDE agreement is a
convergence diagnostic and is never reported as a rigorous error enclosure.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from qf_research.optimal_stopping.contracts import positive_integer
from qf_research.optimal_stopping.funded_cash import (
    european_reference, finite_difference_price, integral_price,
)
from qf_research.optimal_stopping.funded_policy import (
    fit_policy, realise_policy, simulate_paths,
)
from qf_research.optimal_stopping.reference_policy import (
    ReferencePolicy, realise_reference_policy,
)
from qf_research.optimal_stopping.value_martingale import (
    build_values, observation_rights, value_martingale,
)
from validate_funded_dual import load_frozen


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


class OnlineEstimate:
    """Stable block merging of independent sample means and second moments."""

    def __init__(self):
        self.count = 0
        self.mean = 0.
        self.m2 = 0.

    def update(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("Finite one-dimensional payoff samples required")
        count = len(values)
        mean = float(values.mean())
        centred = values-mean
        m2 = float(centred @ centred)
        total = self.count+count
        delta = mean-self.mean
        self.m2 += m2+delta*delta*self.count*count/total
        self.mean += delta*count/total
        self.count = total

    def record(self):
        if self.count < 2:
            raise ValueError("At least two evaluation paths required")
        se = float(np.sqrt(max(self.m2, 0.)/(self.count-1)/self.count))
        radius = float(student_t.ppf(.975, self.count-1)*se)
        return dict(value=self.mean, standard_error=se, lower=self.mean-radius,
                    upper=self.mean+radius, observations=self.count, confidence=.95)


def check_rows(path):
    """Check the direct archived artefact rather than changing its provenance."""
    manifest_path = path.parent/"manifest.json"
    manifest = read(manifest_path)
    if manifest.get("artifact_sha256", {}).get(path.name) != sha(path):
        raise ValueError(f"Archived row artefact checksum mismatch: {path}")
    return read(path), {str(path): sha(path), str(manifest_path): sha(manifest_path)}


def load_case(args):
    policy_path = (args.policy_rows or args.archive_root/
                   "data/private/fitted_policy_gain/2026-09-20/run_v1/rows.json").resolve()
    value_path = (args.value_rows or args.archive_root/
                  "data/private/value_martingale/2026-09-20/run_v1/rows.json").resolve()
    old_rows, policy_hashes = check_rows(policy_path)
    value_rows, value_hashes = check_rows(value_path)
    matching = [row for row in value_rows if row["case"] == args.case]
    prior = [row for row in old_rows if row["case"] == args.case]
    if len(matching) != 1 or not prior:
        raise ValueError("Require one original value study case and its frozen policies")
    row = matching[0]
    if any((old["cutoff"], old["strike"]) != (row["cutoff"], row["strike"]) for old in prior):
        raise ValueError("Frozen model identifiers differ between the two original studies")
    frozen = []
    for old in sorted(prior, key=lambda record: record["repetition"]):
        for entry in old["policies"]:
            policy, _ = load_frozen(entry)
            frozen.append((f"lsm_n{policy.training_paths}_d{policy.requested_degree}_r{old['repetition']}", policy))
    first = frozen[0][1]
    if any((policy.model, policy.option, policy.decisions) !=
           (first.model, first.option, first.decisions) for _, policy in frozen):
        raise ValueError("The selected case contains different funded stopping problems")
    original_value = row["approximations"]["fine"]["value"]
    if (asdict(first.model) != original_value["model"] or
            asdict(first.option) != original_value["option"] or
            [asdict(right) for right in first.decisions] != original_value["decisions"]):
        # JSON lists and dataclass tuple schedules have intentionally different containers.
        canonical = json.loads(json.dumps(asdict(first.model)))
        if (canonical != original_value["model"] or
                asdict(first.option) != original_value["option"] or
                [asdict(right) for right in first.decisions] != original_value["decisions"]):
            raise ValueError("Archived approximation and frozen policy models differ")
    return row, frozen, {**policy_hashes, **value_hashes}


def numerical_references(model, option, rights, args):
    threshold = max(model.initial_state, *(option.strike*right.settlement_factor for right in rights))
    threshold += sum(event.discounted_cash for event in model.dividends
                     if event.ex_time <= option.maturity)
    space, steps = args.pde_space_steps, args.pde_time_steps
    meshes = [("base", space, steps, 4.), ("space_only", 2*space, steps, 4.),
              ("time_only", space, 2*steps, 4.), ("fine", 2*space, 2*steps, 4.),
              ("wide_same_spacing", 3*space, 2*steps, 6.)]
    pde = []
    for name, count, intervals, multiple in meshes:
        start = time.perf_counter()
        result = finite_difference_price(model, option, count, intervals,
                                         multiple*threshold, rights)
        pde.append(dict(name=name, requested_time_steps=intervals,
                        y_max=multiple*threshold, seconds=time.perf_counter()-start,
                        **asdict(result)))
    integration = []
    for count in args.integral_steps:
        start = time.perf_counter()
        result = integral_price(model, option, count, 4*threshold, rights)
        integration.append(dict(y_max=4*threshold, seconds=time.perf_counter()-start,
                                **asdict(result)))
    terminal = european_reference(model, option, rights[-1])
    return dict(pde=pde, integral=integration, expiry_only=asdict(terminal),
                additional_propagation_times=[], numerical_error_enclosure=False)


def run(args):
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT/"data/private").resolve()):
        raise ValueError("Keep the new study under this checkout's data/private directory")
    if args.output.exists():
        raise FileExistsError("Use a fresh private output directory")
    for name in ("evaluation_paths", "chunk_size", "large_training_paths", "pde_space_steps",
                 "pde_time_steps"):
        positive_integer(getattr(args, name), name, 2)
    for count in args.integral_steps:
        positive_integer(count, "integral_steps", 4)
    row, frozen, input_hashes = load_case(args)
    model, option, rights = frozen[0][1].model, frozen[0][1].option, frozen[0][1].decisions
    if args.large_training_paths <= max(policy.training_paths for _, policy in frozen):
        raise ValueError("Fresh training must exceed every retained training count")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()}
    save(args.output/"configuration.json", configuration)
    save(args.output/"inputs.json", dict(case=args.case, cutoff=row["cutoff"],
        model=asdict(model), option=asdict(option), decisions=[asdict(right) for right in rights],
        input_sha256=input_hashes, original_policy_fingerprints=[policy.fingerprint for _, policy in frozen],
        direct_archived_artefacts_checked=True, original_acquisition_reaudited=False))

    approximations = {}
    grid_records = []
    for name, count, tolerance in [("coarse", 400, .005), ("fine", 800, .001),
                                   ("refined", 1600, .00025), ("price_refinement", 3200, .0000625)]:
        start = time.perf_counter()
        approximation = build_values(model, option, rights, core_steps=count,
                                     wing_steps=40, tolerance=tolerance)
        approximations[name] = approximation
        grid_records.append(dict(name=name, core_steps=count, wing_steps=40, tolerance=tolerance,
            initial_grid_value_pv=approximation.initial_value,
            initial_grid_value_quote=approximation.initial_value/model.premium_factor,
            initial_grid_value_is_policy_reward=False, fingerprint=approximation.fingerprint,
            seconds=time.perf_counter()-start,
            maximum_knots=max(len((node.before or node.after).knots) for node in approximation.nodes)))
    reference = numerical_references(model, option, rights, args)
    print(f"Case {args.case}: independent PDE/integral and grid refinements completed", flush=True)

    training_seed = np.random.SeedSequence(args.root_seed, spawn_key=(args.case, 0))
    evaluation_seed = np.random.SeedSequence(args.root_seed, spawn_key=(args.case, 1))
    start = time.perf_counter()
    training = simulate_paths(model, option, rights, args.large_training_paths,
                              np.random.Generator(np.random.PCG64(training_seed)))
    large = fit_policy(option, training, args.large_degree)
    training_stream_id = training.stream_id
    del training
    training_seconds = time.perf_counter()-start
    policies = [(name, policy, realise_policy) for name, policy in frozen]
    policies.append((f"fresh_lsm_n{large.training_paths}_d{large.requested_degree}", large, realise_policy))
    policies += [(f"grid_{name}", ReferencePolicy(approximations[name]), realise_reference_policy)
                 for name in ("coarse", "fine", "refined")]
    fingerprints = {name: policy.fingerprint for name, policy, _ in policies}
    training_streams = {policy.training_stream_id for _, policy in frozen} | {training_stream_id}
    summary = {name: {metric: OnlineEstimate() for metric in
                     ("raw_reward", "centred_reward", "dual_gap", "gain_over_expiry",
                      "gain_over_grid_refined", "early_exercise")}
               for name in ["expiry_only", *(name for name, _, _ in policies)]}
    upper_estimate = OnlineEstimate()
    terminal_martingale = OnlineEstimate()
    rng = np.random.Generator(np.random.PCG64(evaluation_seed))
    assessment = approximations["refined"]
    stream_records = []
    completed = 0
    while completed < args.evaluation_paths:
        count = min(args.chunk_size, args.evaluation_paths-completed)
        if args.evaluation_paths-completed-count == 1:
            count += 1
        observations = simulate_paths(model, option, observation_rights(assessment), count, rng)
        if observations.stream_id in training_streams:
            raise ValueError("Evaluation stream cannot coincide with a training stream")
        sample = value_martingale(assessment, observations)
        paths, martingale = sample.paths, sample.values
        rewards = np.column_stack([model.reward(option, right, paths.states[:, index])
                                   for index, right in enumerate(rights)])
        upper = (rewards-martingale).max(axis=1)
        raw = {"expiry_only": rewards[:, -1]}
        lower = {"expiry_only": raw["expiry_only"]-martingale[:, -1]}
        stopping = {"expiry_only": np.full(count, len(rights)-1)}
        for name, policy, realise in policies:
            realised = realise(policy, paths)
            raw[name] = realised.discounted_payoffs
            stopping[name] = realised.stopping_indices
            stopped = martingale[np.arange(count), realised.stopping_indices]
            lower[name] = raw[name]-stopped
        factor = model.premium_factor
        upper_estimate.update(upper/factor)
        terminal_martingale.update(martingale[:, -1]/factor)
        for name, estimates in summary.items():
            gap = upper-lower[name]
            if np.any(gap < 0.):
                raise ArithmeticError("Pathwise dual inequality failed")
            estimates["raw_reward"].update(raw[name]/factor)
            estimates["centred_reward"].update(lower[name]/factor)
            estimates["dual_gap"].update(gap/factor)
            estimates["gain_over_expiry"].update((lower[name]-lower["expiry_only"])/factor)
            estimates["gain_over_grid_refined"].update((lower[name]-lower["grid_refined"])/factor)
            estimates["early_exercise"].update((stopping[name] < len(rights)-1).astype(float))
        completed += count
        stream_records.append(dict(paths=count, stream_id=observations.stream_id))
        print(f"Case {args.case}: {completed}/{args.evaluation_paths} matched evaluation paths", flush=True)
    fine_pde = next(record["premium_quote"] for record in reference["pde"] if record["name"] == "fine")
    results = {}
    for name, estimates in summary.items():
        record = {metric: estimator.record() for metric, estimator in estimates.items()}
        value = record["centred_reward"]
        record["reference_shortfall_quote"] = dict(value=fine_pde-value["value"],
            sampling_lower=fine_pde-value["upper"], sampling_upper=fine_pde-value["lower"],
            deterministic_error_included=False)
        results[name] = record
    upper = upper_estimate.record()
    result = dict(case=args.case, numerical_references=reference, value_grid_refinement=grid_records,
        evaluation_paths=args.evaluation_paths, policies=results, upper_value_quote=upper,
        upper_minus_fine_pde_quote=upper["value"]-fine_pde,
        terminal_martingale_quote=terminal_martingale.record(), assessment_fingerprint=assessment.fingerprint,
        policy_fingerprints=fingerprints, fresh_training_seconds=training_seconds,
        fresh_policy=asdict(large), evaluation_streams=stream_records,
        stream_design=dict(root_seed=args.root_seed, bit_generator="PCG64",
            training_spawn_key=[args.case, 0], evaluation_spawn_key=[args.case, 1],
            fresh_training_stream_id=training_stream_id, chunk_size=args.chunk_size),
        scope="Conditional on original licensed model inputs and each frozen stopping rule",
        intervals="Approximate pointwise 95% Student sampling intervals across independent evaluation paths; paired comparisons retain common-path covariance",
        finite_sample_coverage_verified=False, rigorous_optimal_value_enclosure=False,
        grid_refinement_does_not_change_other_policies=True,
        new_market_data_acquired=False, publication_clearance=False)
    save(args.output/"results.json", result)
    source_paths = [Path(__file__), ROOT/"scripts/validate_funded_dual.py"] + [
        ROOT/"src/qf_research/optimal_stopping"/f"{name}.py" for name in
        ("reference_policy", "funded_policy", "funded_dual", "funded_cash", "value_martingale",
         "cash_dividends", "analytical", "contracts", "simulation")]
    save(args.output/"manifest.json", dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started, input_sha256=input_hashes,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in source_paths},
        artifact_sha256={name: sha(args.output/name) for name in ("configuration.json", "inputs.json", "results.json")},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform()), publication_clearance=False))
    print(f"Completed private case {args.case}; {len(results)} policies assessed", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--archive-root", type=Path, required=True)
    result.add_argument("--policy-rows", type=Path)
    result.add_argument("--value-rows", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--case", type=int, default=0)
    result.add_argument("--evaluation-paths", type=int, default=20000)
    result.add_argument("--chunk-size", type=int, default=5000)
    result.add_argument("--large-training-paths", type=int, default=100000)
    result.add_argument("--large-degree", type=int, default=4)
    result.add_argument("--root-seed", type=int, default=202610041)
    result.add_argument("--pde-space-steps", type=int, default=9600)
    result.add_argument("--pde-time-steps", type=int, default=2048)
    result.add_argument("--integral-steps", type=int, nargs="+", default=[800, 1600, 3200])
    return result


if __name__ == "__main__":
    run(parser().parse_args())
