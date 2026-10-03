"""New-stream martingale bounds for every previously frozen funded policy."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from qf_research.optimal_stopping.contracts import Option
from qf_research.optimal_stopping.funded_cash import FundedCashModel, DividendCap, FundedExercise
from qf_research.optimal_stopping.funded_policy import FundedPolicy, ContinuationRule, GainControl, simulate_paths
from qf_research.optimal_stopping.funded_dual import evaluate_dual

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(entry):
    data = dict(entry["policy"])
    model = dict(data["model"])
    model["dividends"] = tuple(DividendCap(**d) for d in model["dividends"])
    data["model"], data["option"] = FundedCashModel(**model), Option(**data["option"])
    data["decisions"] = tuple(FundedExercise(**d) for d in data["decisions"])
    data["rules"] = tuple(ContinuationRule(**{**r, "coefficients":tuple(r["coefficients"])}) for r in data["rules"])
    policy = FundedPolicy(**data)
    if policy.fingerprint != entry["fingerprint"]:
        raise ValueError("Saved policy fingerprint mismatch")
    return policy, GainControl(**entry["control"])


def run(config_path, output):
    config = json.loads(config_path.read_text())
    if config["bit_generator"] != "PCG64" or config["confidence"] != .95:
        raise ValueError("Frozen protocol uses PCG64 and 95% pointwise intervals")
    previous = ROOT/config["policy_directory"]
    prior_manifest = json.loads((previous/"manifest.json").read_text())
    for group in ("input_sha256", "source_sha256"):
        for name, value in prior_manifest[group].items():
            if sha(ROOT/name) != value:
                raise ValueError(f"Previous study provenance changed: {name}")
    for name, value in prior_manifest["artifact_sha256"].items():
        if sha(previous/name) != value:
            raise ValueError(f"Previous artifact changed: {name}")
    entries = json.loads((previous/"policies.json").read_text())
    old_streams = {s["stream_id"] for s in json.loads((previous/"random_streams.json").read_text())}
    groups = sorted({(e["expiry"], e["kind"], e["repetition"]) for e in entries})
    seeds = np.random.SeedSequence(config["root_seed"]).spawn(len(groups))
    rows, streams, summary = [], [], []
    output.mkdir(parents=True, exist_ok=False)
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    for group, seed in zip(groups, seeds):
        selected = [e for e in entries if (e["expiry"], e["kind"], e["repetition"]) == group]
        p, _ = load_frozen(selected[0])
        paths = simulate_paths(p.model, p.option, p.decisions, config["evaluation_paths"], np.random.default_rng(seed))
        if paths.stream_id in old_streams:
            raise ValueError("Dual study must use a fresh evaluation stream")
        streams.append(dict(expiry=group[0], kind=group[1], repetition=group[2],
                            entropy=seed.entropy, spawn_key=list(seed.spawn_key), stream_id=paths.stream_id))
        for entry in selected:
            policy, control = load_frozen(entry)
            for hedge in config["hedges"]:
                result = evaluate_dual(policy, paths, control, hedge=hedge)
                row = {k:entry[k] for k in ("expiry", "kind", "repetition", "training_paths", "degree")}
                row.update(hedge=hedge, policy_fingerprint=policy.fingerprint,
                           pricing_lower=result.pricing_lower, pricing_upper=result.pricing_upper,
                           loss_upper_97_5=result.gap_bound.upper,
                           sample_min_gap=float(result.gap_payoffs.min()),
                           sample_zero_gap_fraction=float(np.mean(result.gap_payoffs == 0.)))
                for label in ("policy_value", "upper_value", "gap_bound", "stopped_martingale", "terminal_martingale"):
                    estimate = getattr(result, label)
                    for field in ("value", "standard_error", "lower", "upper"):
                        row[f"{label}_{field}"] = getattr(estimate, field)
                rows.append(row)
        print(f"Bounded policies: {group}", flush=True)
    keys = sorted({(r["expiry"], r["kind"], r["training_paths"], r["degree"], r["hedge"]) for r in rows})
    for key in keys:
        selected = [r for r in rows if tuple(r[k] for k in ("expiry", "kind", "training_paths", "degree", "hedge")) == key]
        row = dict(zip(("expiry", "kind", "training_paths", "degree", "hedge"), key))
        for label in ("policy_value_value", "upper_value_value", "gap_bound_value", "loss_upper_97_5"):
            row[f"mean_{label}"] = float(np.mean([r[label] for r in selected]))
        summary.append(row)
    for filename, data in (("bounds.csv", rows), ("summary.csv", summary)):
        with (output/filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (output/"random_streams.json").write_text(json.dumps(streams, indent=2)+"\n")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    cases = sorted({(r["expiry"], r["kind"]) for r in rows})
    variants = sorted({(r["training_paths"], r["degree"]) for r in rows})
    for ax, case in zip(axes.flat, cases):
        for hedge in config["hedges"]:
            points = [next(r["mean_gap_bound_value"] for r in summary if
                           (r["expiry"], r["kind"], r["training_paths"], r["degree"], r["hedge"]) == (*case, *v, hedge)) for v in variants]
            ax.plot(range(len(variants)), points, "o-", label=hedge)
        ax.set(title=" ".join(case), xticks=range(len(variants)),
               xticklabels=[f"{n:,} / d={d}" for n, d in variants], ylabel="Mean dual slack (PV)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Funded martingale bounds | four repetitions, all hedges retained\nSlack includes both policy loss and looseness of the dual construction", fontsize=11)
    fig.savefig(output/"dual_slack.png", dpi=170); plt.close(fig)
    sources = dict(prior_manifest["source_sha256"])
    for path in ("scripts/validate_funded_dual.py", "src/qf_research/optimal_stopping/funded_dual.py"):
        sources[path] = sha(ROOT/path)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), classification=config["classification"],
                    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                                  (config_path, previous/"manifest.json", previous/"policies.json", previous/"random_streams.json")},
                    source_sha256=sources, artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                    rows=len(rows), summaries=len(summary), python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    empirical_calibration_accepted=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
