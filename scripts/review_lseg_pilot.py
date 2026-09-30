"""Retrieve instrument field metadata, or review the retained pilot offline.

The original acquisition and conservative audit are left intact. Every revised
result is written separately and keeps empirical acceptance false.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys

import lseg_option_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qf_research.market_data.option_quotes import response_blocks  # noqa: E402
from qf_research.market_data.quote_review import (  # noqa: E402
    POLICY_VERSION, metadata_context, review_quote_block, reviewed_pair,
)

METADATA_URL = "/data/historical-pricing/v1/metadata/instrument-metadata"
METADATA_TYPES = "MarketRulesData,Qualifiers,RuleReferenceData,TimeZone,TradingSessions"


def checked_snapshot(snapshot):
    snapshot = pilot.private_path(snapshot)
    with redirect_stdout(io.StringIO()):
        pilot.audit(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    cfg = pilot.load_config(snapshot / "config.json")
    reference = []
    for request in manifest["requests"]:
        if request["name"].startswith("reference_"):
            reference.extend(json.loads((snapshot / request["file"]).read_text()))
    return snapshot, manifest, cfg, pilot.verify_contracts(reference, cfg)


def metadata_arguments(ric):
    if ric is None:
        return {"url": "/data/historical-pricing/v1/metadata/global-metadata", "method": "GET"}
    return {"url": METADATA_URL, "method": "GET", "query_parameters": {
        "version": "v2", "types": METADATA_TYPES, "ric": ric}}


def acquire_metadata(snapshot, output):
    snapshot, source_manifest, cfg, rics = checked_snapshot(snapshot)
    output = pilot.private_path(output)
    if output.exists():
        raise FileExistsError("Use a new metadata output directory")
    import lseg.data as ld
    output.mkdir(parents=True)
    manifest = {
        "status": "in_progress", "provider": "LSEG Workspace",
        "library_version": ld.__version__, "requests": [],
        "source_snapshot_manifest_sha256": pilot.digest(snapshot / "manifest.json"),
        "acquisition_script_sha256": pilot.digest(Path(__file__).resolve()),
        "limitation": "Current instrument metadata, not a historical archive of rule revisions.",
    }
    pilot.write_json(output / "manifest.json", manifest)
    try:
        ld.open_session(name="desktop.workspace")
        for n, ric in enumerate([None, cfg["underlying"]] + rics):
            args = metadata_arguments(ric)
            entry = {"ric": ric, "arguments": args,
                     "requested_at_utc": datetime.now(timezone.utc).isoformat()}
            manifest["requests"].append(entry)
            pilot.write_json(output / "manifest.json", manifest)
            response = ld.delivery.endpoint_request.Definition(**args).get_data()
            payload = response.data.raw
            path = output / f"metadata_{n:02d}.json"
            pilot.write_json(path, payload)
            entry.update(file=path.name, sha256=pilot.digest(path),
                         retrieved_at_utc=datetime.now(timezone.utc).isoformat())
            if ric is not None:
                metadata_context(ric, payload)
            elif not isinstance(payload, dict) or "MasterMapping" not in payload:
                raise ValueError("Unexpected global metadata response")
            entry["status"] = "retained_and_context_checked"
            pilot.write_json(output / "manifest.json", manifest)
            print(f"Retained metadata {n + 1}/{len(rics) + 2}", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed_evidence_retained", error=str(exc))
        raise
    finally:
        ld.close_session()
        pilot.write_json(output / "manifest.json", manifest)


def load_metadata(folder, snapshot, expected_rics):
    folder = pilot.private_path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Metadata acquisition is incomplete")
    if manifest["source_snapshot_manifest_sha256"] != pilot.digest(snapshot / "manifest.json"):
        raise ValueError("Metadata belongs to a different pilot snapshot")
    result, seen = {}, set()
    for entry in manifest["requests"]:
        ric = entry["ric"]
        if ric in seen or entry["arguments"] != metadata_arguments(ric):
            raise ValueError("Duplicate or unexpected metadata request")
        seen.add(ric)
        path = (folder / entry["file"]).resolve()
        if not path.is_relative_to(folder) or pilot.digest(path) != entry["sha256"]:
            raise ValueError("Metadata path or checksum mismatch")
        if entry["status"] != "retained_and_context_checked":
            raise ValueError("Unverified metadata response")
        payload = json.loads(path.read_text())
        if ric is not None:
            metadata_context(ric, payload)
            result[ric] = payload
    if seen != set(expected_rics) | {None}:
        raise ValueError("Metadata coverage does not match the pilot")
    return folder, result


def review(snapshot, metadata_folder, output):
    snapshot, manifest, cfg, rics = checked_snapshot(snapshot)
    folder, metadata = load_metadata(metadata_folder, snapshot, rics + [cfg["underlying"]])
    output = pilot.private_path(output)
    if output == snapshot or output == folder:
        raise ValueError("Review output must be separate from retained source data")
    events, pairs = [], []
    for entry in manifest["requests"]:
        if not entry["name"].startswith("events_"):
            continue
        blocks = response_blocks(json.loads((snapshot / entry["file"]).read_text()))
        revised = {
            b["universe"]["ric"]: review_quote_block(
                b, metadata[b["universe"]["ric"]], entry["arguments"]["end"],
                cfg["lookback_seconds"], cfg["event_count_cap"]
            ) for b in blocks
        }
        events.extend(revised.values())
        pairs.extend(reviewed_pair(revised[ric], revised[cfg["underlying"]],
                                   cfg["age_thresholds_seconds"]) for ric in rics)
    summary = {
        "policy_version": POLICY_VERSION, "empirical_acceptance": False,
        "option_date_pairs": len(pairs), "instrument_date_responses": len(events),
        "original_returned_events": sum(r["original_returned_events"] for r in events),
        "separate_odd_lot_events": sum(r["excluded_odd_lot_events"] for r in events),
        "instrument_dates_passing_price_screen": sum(r["price_screen_pass"] for r in events),
        "instrument_dates_with_unresolved_quote_state": sum(not r["quote_state_resolved"] for r in events),
        "paired_price_and_age_counts": {str(t): sum(p["price_and_age_checks"][str(t)] for p in pairs)
                                        for t in cfg["age_thresholds_seconds"]},
        "paired_unique_state_and_age_counts": {
            str(t): sum(p["unique_state_and_age_checks"][str(t)] for p in pairs)
            for t in cfg["age_thresholds_seconds"]},
        "issue_counts": {i: sum(i in r["issues"] for r in events)
                         for i in sorted({i for r in events for i in r["issues"]})},
        "maximum_event_age_seconds": max(r["event_age_seconds"] for r in events if r["event_age_seconds"] is not None),
        "maximum_source_age_seconds": max(r["source_age_seconds_max"] for r in events if r["source_age_seconds_max"] is not None),
        "maximum_source_skew_seconds": max(p["source_skew_seconds_max"] for p in pairs if p["source_skew_seconds_max"] is not None),
        "unresolved": ["Exact tied-event ordering and unique quoted size",
                       "Historical timestamp lineage, corrections and rule revisions",
                       "NBBO/protected-quote conventions and additional eligibility fields",
                       "Contract deliverables, dividends and discounting inputs"],
    }
    code_paths = [Path(__file__).resolve(), Path(pilot.__file__).resolve(),
                  ROOT / "src/qf_research/market_data/quote_review.py",
                  ROOT / "src/qf_research/market_data/option_quotes.py"]
    report = {
        "source_snapshot_manifest_sha256": pilot.digest(snapshot / "manifest.json"),
        "original_timing_audit_sha256": pilot.digest(snapshot / "timing_audit.json"),
        "metadata_manifest_sha256": pilot.digest(folder / "manifest.json"),
        "source_hashes": {str(p.relative_to(ROOT)): pilot.digest(p) for p in code_paths},
        "summary": summary, "events": events, "pairs": pairs,
    }
    content = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    path = output / "quote_review.json"
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError("Review changed; preserve the earlier version and choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    live = sub.add_parser("metadata")
    live.add_argument("--snapshot", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    offline = sub.add_parser("audit")
    offline.add_argument("--snapshot", type=Path, required=True)
    offline.add_argument("--metadata", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "metadata":
        acquire_metadata(args.snapshot, args.output)
    else:
        review(args.snapshot, args.metadata, args.output)


if __name__ == "__main__":
    main()
