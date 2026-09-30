"""Target daily-calendar projection error after the initial frozen experiment."""

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

from qf_research.market_data.event_calendar import EventCalendar
from qf_research.market_data.funded_inputs import DatedDividend, build_dated_problem
from qf_research.market_data.valuation_inputs import DiscountCurve
from qf_research.optimal_stopping.funded_cash import finite_difference_price, integral_price

ROOT = Path(__file__).resolve().parents[1]


def run(config_path, output):
    config = json.loads(config_path.read_text())
    base_path = ROOT/config["base_config"]
    case = json.loads(base_path.read_text())["dated_validation"]
    calendar_path, issuer_path = ROOT/case["calendar_config"], ROOT/case["issuer_config"]
    calendar = EventCalendar.from_config(json.loads(calendar_path.read_text()))
    issuer = json.loads(issuer_path.read_text())
    observation = datetime.fromisoformat(case["observation"])
    end = date.fromisoformat(case["curve_end"])
    curve = DiscountCurve(observation.date(), (observation.date(), end),
                           (1., math.exp(-case["flat_rate"]*(end-observation.date()).days/365)))
    dividends = tuple(DatedDividend(date.fromisoformat(e["ex_date"]), date.fromisoformat(e["pay_date"]),
                                    None if e["conditional"] else case["regular_dividend_cash_fixture"],
                                    "Explicit analytical cash fixture, not an observation",
                                    None if e["conditional"] else True) for e in issuer["events"])
    output.mkdir(parents=True, exist_ok=False)
    (output/"configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    rows = []
    for expiry in case["expiries"]:
        for kind in ("call", "put"):
            problem = build_dated_problem(calendar, curve, observation, date.fromisoformat(expiry),
                                           case["quoted_spot"], case["strike"], case["volatility"], kind, dividends,
                                           premium_payment_date=date.fromisoformat(case["premium_payment_date"]),
                                           allow_provisional=case["allow_provisional_settlement"])
            for setting in config["cases"]:
                m, upper, n = setting["space_steps"], setting["y_max"], setting["time_steps"]
                for method in setting["methods"]:
                    if method == "PDE":
                        result = finite_difference_price(problem.model, problem.option, m, n, upper, problem.decisions)
                    elif method == "integral":
                        result = integral_price(problem.model, problem.option, m, upper, problem.decisions)
                    else:
                        raise ValueError("Unknown method")
                    rows.append(dict(expiry=expiry, kind=kind, method=method, space_steps=m,
                                     y_max=upper, spacing=upper/m, requested_time_steps=n if method == "PDE" else None,
                                     actual_intervals=result.time_intervals, present_value=result.present_value,
                                     premium_quote=result.premium_quote, purpose=setting["purpose"]))
            print(f"Refined dated fixture: {expiry} {kind}", flush=True)
    with (output/"refinement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = (config_path, base_path, calendar_path, issuer_path)
    sources = [Path(__file__)] + [ROOT/p for p in (
        "src/qf_research/optimal_stopping/funded_cash.py", "src/qf_research/market_data/funded_inputs.py",
        "src/qf_research/optimal_stopping/cash_dividends.py", "src/qf_research/optimal_stopping/escrowed_dividends.py",
        "src/qf_research/optimal_stopping/analytical.py", "src/qf_research/optimal_stopping/contracts.py",
        "src/qf_research/market_data/event_calendar.py", "src/qf_research/market_data/valuation_inputs.py")]
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), classification=config["classification"],
                    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                    artifact_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},
                    rows=len(rows), python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__), empirical_calibration_accepted=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
