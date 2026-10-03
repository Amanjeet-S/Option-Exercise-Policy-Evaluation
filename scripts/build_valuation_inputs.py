"""Acquire dated SOFR curves and build a reproducible conditional input study.

All curve-valued results stay private. Income forecasts use publicly filed
pre-sample accounting data; they are never labelled as declared dividends.
"""

import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import lseg_option_pilot as pilot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qf_research.market_data.valuation_inputs import (  # noqa: E402
    discounted_distribution, inspect_sofr_curve, released_report_forecast,
)

ENDPOINT = "/data/quantitative-analytics-curves-and-surfaces/v1/curves/"


def load_config(path):
    cfg = json.loads(path.read_text())
    if cfg["underlying"] != "SPY" or cfg["currency"] != "USD":
        raise ValueError("This study is scoped to USD SPY inputs")
    if not 1 <= len(cfg["curve_dates_by_observation"]) <= 5:
        raise ValueError("Use a bounded sample of at most five curves")
    for day, anchor in cfg["curve_dates_by_observation"].items():
        gap = (date.fromisoformat(day) - date.fromisoformat(anchor)).days
        if not 1 <= gap <= 4 or date.fromisoformat(day) >= datetime.now(ZoneInfo("America/New_York")).date():
            raise ValueError("Require a completed observation date and an earlier nearby curve date")
    return cfg


def curve_arguments(cfg, day):
    return {"url": ENDPOINT + "zc-curves", "method": "POST", "body_parameters": {
        "universe": [{"curveDefinition": {"id": cfg["curve_id"], "discountingTenor": "OIS", "indexTenors": ["OIS"]},
                      "curveParameters": {"valuationDate": day, "priceSide": "Mid",
                                          "useDelayedDataIfDenied": False, "ignoreInvalidInstrument": False,
                                          "marketDataAccessDeniedFallback": "ReturnError",
                                          "marketDataLookBack": {"value": 0, "unit": "CalendarDay"}}}],
        "outputs": ["Constituents", "DetailedCurvePoint"]}}


def acquire(config, output):
    cfg, output = load_config(config), pilot.private_path(output)
    if output.exists():
        raise FileExistsError("Use a new curve acquisition directory")
    import lseg.data as ld
    output.mkdir(parents=True)
    (output / "config.json").write_bytes(config.read_bytes())
    (output / "acquisition_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"status": "in_progress", "requests": [], "library_version": ld.__version__,
                "config_sha256": pilot.digest(output / "config.json"),
                "acquisition_source_sha256": pilot.digest(output / "acquisition_source.py"),
                "representation": "SDK-decoded response payloads, not original HTTP bytes"}
    try:
        ld.open_session(name="desktop.workspace")
        for day in sorted(set(cfg["curve_dates_by_observation"].values())):
            args = curve_arguments(cfg, day)
            entry = {"curve_date": day, "arguments": args, "requested_at_utc": datetime.now(timezone.utc).isoformat()}
            manifest["requests"].append(entry)
            pilot.write_json(output / "manifest.json", manifest)
            response = ld.delivery.endpoint_request.Definition(**args).get_data()
            file = output / f"curve_{day}.json"
            pilot.write_json(file, response.data.raw)
            entry.update(file=file.name, sha256=pilot.digest(file), retrieved_at_utc=datetime.now(timezone.utc).isoformat())
            _, diagnostics = inspect_sofr_curve(response.data.raw, day, cfg["curve_id"])
            entry["status"] = "structure_checked_constituent_limits_retained"
            pilot.write_json(output / "manifest.json", manifest)
            print(f"Retained {day}: {diagnostics['nodes']} nodes; {diagnostics['masked_constituents']} masked constituents", flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed_evidence_retained", error=str(exc))
        raise
    finally:
        ld.close_session()
        pilot.write_json(output / "manifest.json", manifest)


def verify_sources(folder, cfg):
    manifest = json.loads((folder / "manifest.json").read_text())
    by_name = {}
    for entry in manifest["sources"]:
        file = (folder / entry["file"]).resolve()
        if not file.is_relative_to(folder) or entry["file"] in by_name or pilot.digest(file) != entry["sha256"]:
            raise ValueError("Source identity or checksum mismatch")
        by_name[entry["file"]] = entry
    submissions = json.loads((folder / "sec_submissions.json").read_text())
    if int(submissions["cik"]) != 884394:
        raise ValueError("Incorrect SEC registrant")
    filings = submissions["filings"]["recent"]
    for prefix in ("income", "schedule"):
        indices = [i for i, value in enumerate(filings["accessionNumber"]) if value == cfg[prefix + "_accession"]]
        if len(indices) != 1:
            raise ValueError("Ambiguous SEC filing identity")
        i = indices[0]
        if filings["form"][i] != cfg[prefix + "_form"] or filings["filingDate"][i] != cfg[prefix + "_filing_date"]:
            raise ValueError("SEC filing metadata disagrees with the research configuration")
        available = datetime.combine(date.fromisoformat(filings["filingDate"][i]) + timedelta(days=1), datetime.min.time(), ZoneInfo("America/New_York"))
        if datetime.fromisoformat(cfg[prefix + "_available_at"]) != available:
            raise ValueError("Incorrect conservative SEC publication bound")
        expected_url = ("https://www.sec.gov/Archives/edgar/data/884394/" +
                        filings["accessionNumber"][i].replace("-", "") + "/" + filings["primaryDocument"][i])
        if cfg[prefix + "_source_url"] != expected_url or by_name[cfg[prefix + "_source_file"]]["url"] != expected_url:
            raise ValueError("SEC document URL does not match its accession")
    income_text = (folder / cfg["income_source_file"]).read_text()
    marker = "Financial Highlights\n\nSelected data for a Unit outstanding throughout each period"
    if income_text.count(marker) != 1:
        raise ValueError("Cannot locate the reviewed financial-highlights table")
    table = income_text.split(marker)[1].split("See accompanying notes")[0]
    if "3/31/26" not in table or "9/30/25" not in table:
        raise ValueError("Financial-highlights periods changed")
    line = next(line for line in table.splitlines() if line.startswith("| Net investment income (loss)"))
    values = re.findall(r"\|\s*(\d+\.\d+)\s*(?=\|)", line)
    if values[:2] != [r["net_investment_income_per_unit"] for r in cfg["forecast_reports"]]:
        raise ValueError("Transcribed income differs from the retained source row")
    text = (folder / cfg["schedule_source_file"]).read_text()
    if "third (3rd) Friday in each of March, June, September and" not in text or "last Business Day in the calendar month following" not in text:
        raise ValueError("Cannot verify the reviewed regular-distribution rule")
    return manifest


def audit(config, inputs, sources, output):
    cfg = load_config(config)
    inputs, sources, output = map(pilot.private_path, (inputs, sources, output))
    if output in (inputs, sources):
        raise ValueError("Keep analysis separate from source data")
    manifest = json.loads((inputs / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["config_sha256"] != pilot.digest(config) or pilot.digest(inputs / "config.json") != pilot.digest(config):
        raise ValueError("Incomplete inputs or configuration mismatch")
    if pilot.digest(inputs / "acquisition_source.py") != manifest["acquisition_source_sha256"]:
        raise ValueError("Acquisition-source checksum mismatch")
    verify_sources(sources, cfg)
    curves, diagnostics = {}, {}
    for entry in manifest["requests"]:
        day = entry["curve_date"]
        if day in curves or entry["arguments"] != curve_arguments(cfg, day):
            raise ValueError("Duplicate or unexpected curve request")
        file = (inputs / entry["file"]).resolve()
        if not file.is_relative_to(inputs) or pilot.digest(file) != entry["sha256"]:
            raise ValueError("Curve path or checksum mismatch")
        curves[day], diagnostics[day] = inspect_sofr_curve(json.loads(file.read_text()), day, cfg["curve_id"])
    if set(curves) != set(cfg["curve_dates_by_observation"].values()):
        raise ValueError("Curve-date coverage mismatch")
    horizons = []
    for day, anchor in cfg["curve_dates_by_observation"].items():
        cutoff = datetime.fromisoformat(day + "T15:45:00").replace(tzinfo=ZoneInfo("America/New_York")).isoformat()
        if datetime.fromisoformat(cfg["schedule_available_at"]) > datetime.fromisoformat(cutoff):
            raise ValueError("Prospectus rule was not available at the cutoff")
        forecasts = [released_report_forecast(r, cfg["income_available_at"], cutoff) for r in cfg["forecast_reports"]]
        curve, valuation = curves[anchor], date.fromisoformat(day)
        for expiry in ("2026-10-16", "2026-12-18", "2027-01-15"):
            events = [e for e in cfg["regular_events"] if day < e["ex_date"] <= expiry]
            scenarios = []
            for forecast in forecasts:
                cash = float(forecast["forecast_cash_per_regular_distribution"])
                flows = [discounted_distribution(curve, valuation, date.fromisoformat(e["ex_date"]), date.fromisoformat(e["pay_date"]), cash) for e in events]
                scenarios.append({"forecast": forecast, "regular_distributions": flows,
                                  "present_value_regular_distributions": sum(f["present_value"] for f in flows)})
            horizons.append({"observation_date": day, "curve_date": anchor, "expiry_date": expiry,
                             "discount_factor_to_expiry": curve.between(valuation, date.fromisoformat(expiry)),
                             "continuous_zero_act365f_decimal": curve.continuous_zero(valuation, date.fromisoformat(expiry)),
                             "scenarios": scenarios,
                             "unresolved_conditional_distribution": day < cfg["unresolved_conditional_event"]["ex_date"] <= expiry,
                             "calibration_ready": False})
    result = {"config_sha256": pilot.digest(config), "curve_manifest_sha256": pilot.digest(inputs / "manifest.json"),
              "sources_manifest_sha256": pilot.digest(sources / "manifest.json"),
              "source_hashes": {str(p.relative_to(ROOT)): pilot.digest(p) for p in
                                (Path(__file__).resolve(), ROOT / "src/qf_research/market_data/valuation_inputs.py")},
              "summary": {"curves": len(curves), "horizons": len(horizons), "forecast_models": len(cfg["forecast_reports"]),
                          "scenario_horizons": sum(len(h["scenarios"]) for h in horizons),
                          "independently_reconstructable_curves": sum(d["independent_bootstrap_ready"] for d in diagnostics.values()),
                          "horizons_with_unresolved_conditional_distribution": sum(h["unresolved_conditional_distribution"] for h in horizons),
                          "empirical_acceptance": False},
              "curve_diagnostics": diagnostics, "horizons": horizons}
    target = output / "valuation_inputs.json"
    content = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError("Analysis changed; retain the earlier report and choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "audit":
            p.add_argument("--inputs", type=Path, required=True)
            p.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire(args.config, args.output)
    else:
        audit(args.config, args.inputs, args.sources, args.output)


if __name__ == "__main__":
    main()
