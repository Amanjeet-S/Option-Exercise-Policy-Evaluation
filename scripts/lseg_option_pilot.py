"""Acquire or re-audit a bounded SPY quote pilot using the open Workspace app.

Network acquisition requires the optional, separate LSEG environment. Offline
audit requires only the standard library and repository source. Source responses
and derived quote values must remain inside the ignored private data directory.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qf_research.market_data.option_quotes import (  # noqa: E402
    inspect_quote_block, pair_diagnostics, response_blocks,
)

PRIVATE_ROOT = ROOT / "data/private/lseg"
REFERENCE_FIELDS = [
    "RIC", "DocumentTitle", "StrikePrice", "ExpiryDate", "UnderlyingQuoteRIC",
    "CallPutOption", "ExerciseStyle", "ExerciseStyleName", "ExchangeCode",
    "LotSize", "StrikeMultiplier", "Currency", "PeriodicityCode",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def private_path(path):
    path = path.resolve()
    if not path.is_relative_to(PRIVATE_ROOT.resolve()) or path == PRIVATE_ROOT.resolve():
        raise ValueError("Snapshot must be a subdirectory of data/private/lseg")
    return path


def load_config(path):
    cfg = json.loads(path.read_text())
    # This deliberately bounded acquisition is not a general chain downloader.
    if cfg["underlying"] != "SPY" or cfg["exchange_code"] != "OPQ":
        raise ValueError("This pilot supports only the inspected SPY / OPQ universe")
    for field in ("expiries", "strikes", "option_types", "observation_dates"):
        if not cfg[field] or len(cfg[field]) != len(set(cfg[field])):
            raise ValueError(f"Empty or duplicate {field}")
    if set(cfg["option_types"]) != {"Call", "Put"} or len(cfg["expiries"]) > 3 or len(cfg["strikes"]) > 3:
        raise ValueError("At most three expiries, three strikes and both option types are supported")
    if len(cfg["observation_dates"]) > 5 or cfg["observation_dates"] != sorted(cfg["observation_dates"]):
        raise ValueError("Use at most five chronological observation dates")
    today = datetime.now(ZoneInfo(cfg["timezone"])).date()
    for day in cfg["observation_dates"]:
        if date.fromisoformat(day) >= today:
            raise ValueError("Only completed local calendar dates may be requested")
        for expiry in cfg["expiries"]:
            maturity = (date.fromisoformat(expiry) - date.fromisoformat(day)).days
            if not 30 <= maturity <= 180:
                raise ValueError("All sampled maturities must be between 30 and 180 days")
    if not 1 <= cfg["lookback_seconds"] <= 60 or not 1 <= cfg["event_count_cap"] <= 10000:
        raise ValueError("Pilot query limits exceeded")
    if not 1 <= cfg["search_top"] <= 100:
        raise ValueError("Search cap must be between 1 and 100")
    if not all(0 < n <= cfg["lookback_seconds"] for n in cfg["age_thresholds_seconds"]):
        raise ValueError("Age thresholds must fit the acquisition window")
    if cfg["adjustments"] != ["unadjusted"]:
        raise ValueError("This event pilot requires explicit unadjusted requests")
    return cfg


def verify_contracts(records, cfg):
    expected = set(itertools.product(cfg["expiries"], cfg["strikes"], cfg["option_types"]))
    seen, rics = set(), set()
    for row in records:
        key = (row["ExpiryDate"][:10], row["StrikePrice"], row["CallPutOption"])
        if key not in expected or key in seen or not row.get("RIC") or row["RIC"] in rics:
            raise ValueError("Unexpected, missing or ambiguous contract identity")
        underlying = row.get("UnderlyingQuoteRIC")
        if isinstance(underlying, str):
            underlying = [underlying]
        if cfg["underlying"] not in (underlying or []):
            raise ValueError("Reference record has a different underlying")
        for field, setting in (("ExerciseStyle", "exercise_style"), ("ExchangeCode", "exchange_code"), ("Currency", "currency")):
            if row.get(field) != cfg[setting]:
                raise ValueError(f"Reference mismatch: {field}")
        seen.add(key)
        rics.add(row["RIC"])
    if seen != expected:
        raise ValueError(f"Incomplete contract grid: found {len(seen)} of {len(expected)}")
    return sorted(rics)


def event_arguments(cfg, rics, day):
    cutoff = datetime.fromisoformat(day + "T" + cfg["snapshot_local_time"]).replace(
        tzinfo=ZoneInfo(cfg["timezone"])).astimezone(timezone.utc)
    return dict(
        universe=rics + [cfg["underlying"]], eventTypes=["quote"],
        start=(cutoff - timedelta(seconds=cfg["lookback_seconds"])).isoformat(),
        end=cutoff.isoformat(), adjustments=cfg["adjustments"],
        fields=cfg["event_fields"], count=cfg["event_count_cap"],
    )


def acquire(config_path, output):
    cfg = load_config(config_path)
    output = private_path(output)
    if output.exists():
        raise FileExistsError("Choose a new snapshot directory; retained evidence cannot be overwritten")
    import lseg.data as ld
    output.mkdir(parents=True)
    (output / "config.json").write_bytes(config_path.read_bytes())
    manifest = {
        "provider": "LSEG Workspace", "library_version": ld.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(output / "config.json"),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__).resolve(), ROOT / "src/qf_research/market_data/option_quotes.py")},
        "status": "acquisition_in_progress", "requests": [],
        "limitations": "Feasibility sample using current contract discovery and retrospective history; not an as-published vintage or a certified NBBO panel.",
    }
    write_json(output / "manifest.json", manifest)

    def request(name, method, arguments, function):
        item = {"name": name, "method": method, "arguments": arguments,
                "requested_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest["requests"].append(item)
        write_json(output / "manifest.json", manifest)
        try:
            value = function()
            path = output / f"{name}.json"
            write_json(path, value)
            item.update(file=path.name, sha256=digest(path),
                        retrieved_at_utc=datetime.now(timezone.utc).isoformat(), status="retained")
            print(f"Retained {name}", flush=True)
            return value
        except Exception as exc:
            item.update(status="failed", error=str(exc))
            raise
        finally:
            write_json(output / "manifest.json", manifest)

    try:
        ld.open_session(name="desktop.workspace")
        contracts = []
        for expiry in cfg["expiries"]:
            strikes = " or ".join(f"StrikePrice eq {float(k)}" for k in cfg["strikes"])
            query = (f"UnderlyingQuoteRIC xeq 'SPY' and IsChain eq false and ExchangeCode eq 'OPQ' "
                     f"and ExpiryDate eq {expiry} and ({strikes})")
            args = dict(filter=query, select=",".join(REFERENCE_FIELDS), top=cfg["search_top"])
            records = request(
                "reference_" + expiry, "ld.discovery.search",
                dict(view="EQUITY_DERIVATIVE_QUOTES", **args),
                lambda: json.loads(ld.discovery.search(
                    view=ld.discovery.Views.EQUITY_DERIVATIVE_QUOTES, **args
                ).to_json(orient="records", date_format="iso")),
            )
            if len(records) >= cfg["search_top"]:
                raise ValueError("Search count cap reached; coverage is uncertain")
            contracts.extend(records)
        rics = verify_contracts(contracts, cfg)
        for day in cfg["observation_dates"]:
            args = event_arguments(cfg, rics, day)
            request("events_" + day, "historical_pricing.events.Definition.get_data", args,
                    lambda: ld.content.historical_pricing.events.Definition(**args).get_data().data.raw)
        manifest["status"] = "acquisition_complete_audit_pending"
    except Exception:
        manifest["status"] = "acquisition_failed_evidence_retained"
        raise
    finally:
        ld.close_session()
        write_json(output / "manifest.json", manifest)
    audit(output)


def audit(output):
    output = private_path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if digest(output / "config.json") != manifest["config_sha256"]:
        raise ValueError("Configuration checksum mismatch")
    cfg = load_config(output / "config.json")
    if manifest["status"] != "acquisition_complete_audit_pending":
        raise ValueError("Acquisition did not complete")
    records, requests = [], []
    names = set()
    for entry in manifest["requests"]:
        if entry["name"] in names:
            raise ValueError("Duplicate request name")
        names.add(entry["name"])
        path = (output / entry["file"]).resolve()
        if not path.is_relative_to(output) or digest(path) != entry["sha256"]:
            raise ValueError("Invalid snapshot path or checksum mismatch")
        payload = json.loads(path.read_text())
        if entry["name"].startswith("reference_"):
            records.extend(payload)
        else:
            requests.append((entry, payload))
    rics = verify_contracts(records, cfg)
    expected_names = {"reference_" + d for d in cfg["expiries"]} | {"events_" + d for d in cfg["observation_dates"]}
    if names != expected_names:
        raise ValueError("Snapshot request coverage differs from the configuration")
    events, pairs = [], []
    for entry, payload in requests:
        day = entry["name"].removeprefix("events_")
        if entry["arguments"] != event_arguments(cfg, rics, day):
            raise ValueError("Recorded event request differs from the configuration")
        blocks = response_blocks(payload)
        returned = [b.get("universe", {}).get("ric") for b in blocks]
        if len(set(returned)) != len(returned) or set(returned) != set(rics + [cfg["underlying"]]):
            raise ValueError("Missing, duplicate or unexpected response instrument")
        inspected = {
            b["universe"]["ric"]: inspect_quote_block(
                b, entry["arguments"]["end"], cfg["lookback_seconds"], cfg["event_count_cap"]
            ) for b in blocks
        }
        events.extend(inspected.values())
        pairs.extend(pair_diagnostics(inspected[ric], inspected[cfg["underlying"]],
                                      cfg["age_thresholds_seconds"]) for ric in rics)
    summary = {
        "status": "timing_and_schema_audit_only_not_approved_model_inputs",
        "contracts": len(rics), "observation_dates": cfg["observation_dates"],
        "expiries": cfg["expiries"], "option_date_pairs": len(pairs),
        "instrument_date_responses": len(events),
        "returned_events": sum(r["returned_events"] for r in events),
        "mechanically_passing_instrument_dates": sum(r["mechanical_pass"] for r in events),
        "paired_age_check_counts": {str(t): sum(p["passes_age_checks"][str(t)] for p in pairs)
                                    for t in cfg["age_thresholds_seconds"]},
        "issue_counts": {issue: sum(issue in r["issues"] for r in events)
                         for issue in sorted({i for r in events for i in r["issues"]})},
        "missing_source_timestamps_on_selected_events": sum(
            r["selected"] is not None and not r["selected"].get("SOURCE_DATETIME") for r in events),
        "reference_lot_sizes": sorted({str(r.get("LotSize")) for r in records}),
        "reference_strike_multipliers": sorted({str(r.get("StrikeMultiplier")) for r in records}),
        "empirical_acceptance": False,
        "unresolved": ["Contract-specific deliverables and adjustment history",
                       "DATE_TIME, SOURCE_DATETIME and NBBO/quote-condition definitions",
                       "Dividend information available at each valuation time",
                       "Discount factors and financing assumptions", "Open-interest date semantics"],
    }
    report = {
        "audit_source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__).resolve(), ROOT / "src/qf_research/market_data/option_quotes.py")},
        "summary": summary, "events": events, "pairs": pairs,
    }
    report_bytes = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    path = output / "timing_audit.json"
    if path.exists() and path.read_bytes() != report_bytes:
        raise FileExistsError("Audit changed; preserve the earlier report and use a new audit version")
    path.write_bytes(report_bytes)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("acquire")
    live.add_argument("--config", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    offline = commands.add_parser("audit")
    offline.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.config, args.output)
    else:
        audit(args.snapshot)


if __name__ == "__main__":
    main()
