"""Funding, quotation and exact dated-grid validation using declared fixtures."""

import argparse
import csv
from dataclasses import replace
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
from qf_research.optimal_stopping.analytical import black_scholes
from qf_research.optimal_stopping.contracts import Market, Option
from qf_research.optimal_stopping.funded_cash import (
    DividendCap, FundedCashModel, FundedExercise, european_reference,
    finite_difference_price, integral_price,
)

ROOT = Path(__file__).resolve().parents[1]


def run(config_path, output):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    df = lambda t: math.exp(-config["flat_rate"]*t)
    lag, dividend = config["constant_settlement_lag"], config["dividend"]
    model = FundedCashModel(config["quoted_spot"], config["volatility"], df(lag), df(lag),
                            (DividendCap(dividend["ex_time"], dividend["amount"]*df(dividend["pay_time"])),))
    rights = tuple(FundedExercise(t, df(t+lag)) for t in config["exercise_times"])
    domain = config["y_max"]
    convergence, bridges, dated = [], [], []

    def save(name, rows):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for kind in ("call", "put"):
        option = Option(config["strike"], config["maturity"], kind)
        oracle = european_reference(model, option, rights[-1])
        for label, selected in (("European", (rights[-1],)), ("Bermudan", rights)):
            def record(method, sweep, m, n, upper, value):
                convergence.append(dict(kind=kind, exercise=label, method=method, sweep=sweep,
                                        space_steps=m, requested_time_steps=n, actual_intervals=value.time_intervals,
                                        y_max=upper, present_value=value.present_value, premium_quote=value.premium_quote,
                                        reference=oracle.present_value if label == "European" else None,
                                        signed_error=value.present_value-oracle.present_value if label == "European" else None))
            n = config["reference_time_steps"]
            for m in config["space_steps"]:
                record("PDE", "space", m, n, domain, finite_difference_price(model, option, m, n, domain, selected))
                record("integral", "space", m, None, domain, integral_price(model, option, m, domain, selected))
            m = config["reference_space_steps"]
            for n in config["time_steps"]:
                record("PDE", "time", m, n, domain, finite_difference_price(model, option, m, n, domain, selected))
            for upper in config["domain_maxima"]:
                location = upper/config["domain_spacing"]
                if not float(location).is_integer():
                    raise ValueError("Domain sweep must keep spatial spacing fixed")
                m, n = int(location), config["reference_time_steps"]
                record("PDE", "domain", m, n, upper, finite_difference_price(model, option, m, n, upper, selected))
                record("integral", "domain", m, None, upper, integral_price(model, option, m, upper, selected))
        print(f"Finished irregular-calendar convergence: {kind}", flush=True)

        for rate in (-.03, .04):
            for settlement_lag in (0., 1/365, 4/365):
                for premium_case, premium_lag in (("matched", settlement_lag), ("premium_paid_immediately", 0.)):
                    discount = lambda t: math.exp(-rate*t)
                    m = FundedCashModel(config["quoted_spot"], config["volatility"], discount(settlement_lag), discount(premium_lag))
                    terminal = FundedExercise(option.maturity, discount(option.maturity+settlement_lag))
                    reference = european_reference(m, option, terminal)
                    immediate = black_scholes(Market(config["quoted_spot"], rate, config["volatility"]), option)
                    expected = discount(settlement_lag)/discount(premium_lag)*immediate
                    if not math.isclose(reference.premium_quote, expected, abs_tol=1e-12):
                        raise AssertionError("Analytical quotation identity failed")
                    bridges.append(dict(kind=kind, rate=rate, share_and_exercise_lag=settlement_lag,
                                        premium_case=premium_case, premium_lag=premium_lag,
                                        present_value=reference.present_value, premium_quote=reference.premium_quote,
                                        immediate_black_scholes=immediate, independently_scaled_quote=expected))

    save("convergence.csv", convergence)
    save("quotation_identities.csv", bridges)

    case = config["dated_validation"]
    calendar_path, issuer_path = ROOT/case["calendar_config"], ROOT/case["issuer_config"]
    calendar = EventCalendar.from_config(json.loads(calendar_path.read_text()))
    issuer = json.loads(issuer_path.read_text())
    observation = datetime.fromisoformat(case["observation"])
    end = date.fromisoformat(case["curve_end"])
    curve = DiscountCurve(observation.date(), (observation.date(), end),
                           (1., math.exp(-case["flat_rate"]*(end-observation.date()).days/365)))
    dividends = tuple(DatedDividend(date.fromisoformat(e["ex_date"]), date.fromisoformat(e["pay_date"]),
                                    None if e["conditional"] else case["regular_dividend_cash_fixture"],
                                    "Explicit analytical cash fixture; not an issuer declaration or forecast",
                                    None if e["conditional"] else True) for e in issuer["events"])
    for expiry_text in case["expiries"]:
        for kind in ("call", "put"):
            problem = build_dated_problem(calendar, curve, observation, date.fromisoformat(expiry_text),
                                           case["quoted_spot"], case["strike"], case["volatility"], kind, dividends,
                                           premium_payment_date=date.fromisoformat(case["premium_payment_date"]),
                                           allow_provisional=case["allow_provisional_settlement"])
            for m in case["space_steps"]:
                results = (("PDE", finite_difference_price(problem.model, problem.option, m, case["time_steps"], case["y_max"], problem.decisions)),
                           ("integral", integral_price(problem.model, problem.option, m, case["y_max"], problem.decisions)))
                for method, result in results:
                    dated.append(dict(expiry=expiry_text, kind=kind, method=method, space_steps=m,
                                      actual_intervals=result.time_intervals, decision_count=result.decision_count,
                                      event_count=result.event_count, present_value=result.present_value,
                                      premium_quote=result.premium_quote, terminal_delivery=problem.deliveries[-1][1].isoformat(),
                                      provisional_delivery=problem.provisional_delivery,
                                      empirical_calibration_accepted=problem.empirical_calibration_accepted))
            print(f"Finished calendar-mapped fixture: {expiry_text} {kind}", flush=True)
    save("dated_validation.csv", dated)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, marker in (("PDE", "o"), ("integral", "x")):
        rows = [r for r in convergence if r["kind"] == "call" and r["exercise"] == "European" and r["sweep"] == "space" and r["method"] == method]
        axes[0].semilogx([r["space_steps"] for r in rows], [r["signed_error"] for r in rows], marker+"-", label=method)
    for expiry in case["expiries"]:
        for kind, marker in (("call", "o"), ("put", "x")):
            gaps = []
            for m in case["space_steps"]:
                pair = {r["method"]:r["present_value"] for r in dated if r["expiry"] == expiry and r["kind"] == kind and r["space_steps"] == m}
                gaps.append(pair["integral"]-pair["PDE"])
            axes[1].semilogx(case["space_steps"], gaps, marker+"-", label=expiry+" "+kind)
    axes[0].set(title="Funded European call: independent reference", xlabel="Spatial intervals", ylabel="Signed PV error")
    axes[1].set(title="Actual calendars with mathematical cash inputs", xlabel="Spatial intervals", ylabel="Integral PV minus PDE PV")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Funding validation | no observed option valuation or calibration")
    fig.savefig(output/"validation.png", dpi=170)
    plt.close(fig)

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    sources = [Path(__file__)] + [ROOT/p for p in (
        "src/qf_research/optimal_stopping/funded_cash.py", "src/qf_research/market_data/funded_inputs.py",
        "src/qf_research/optimal_stopping/cash_dividends.py", "src/qf_research/optimal_stopping/escrowed_dividends.py",
        "src/qf_research/optimal_stopping/analytical.py", "src/qf_research/optimal_stopping/contracts.py",
        "src/qf_research/market_data/event_calendar.py", "src/qf_research/market_data/valuation_inputs.py")]
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), classification=config["classification"],
                    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (config_path, calendar_path, issuer_path)},
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
                    artifact_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                    row_counts=dict(convergence=len(convergence), quotation=len(bridges), dated=len(dated)),
                    python=sys.version, platform=platform.platform(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    empirical_calibration_accepted=False)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Saved funded validation to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
