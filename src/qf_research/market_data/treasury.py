"""Retrieve observed US Treasury par yields, preserving raw source bytes.

These are percentage par yields, not continuously compounded zero rates. No
conversion into option-pricing discount factors is performed by this module.
"""

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


FEED = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
DOCUMENTATION = "https://home.treasury.gov/treasury-daily-interest-rate-xml-feed"
NAMESPACES = {
    "a": "http://www.w3.org/2005/Atom",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
}


def parse_par_yields(raw: bytes, cutoff: date) -> list[dict]:
    """Keep observations through cutoff, preserve missing values, reject duplicates.

    A cutoff filters observation dates; it does not reconstruct a historical
    publication vintage. The source may contain revisions made after that date.
    """
    root = ET.fromstring(raw)
    rows = []
    seen = set()
    for entry in root.findall("a:entry", NAMESPACES):
        properties = entry.find(".//m:properties", NAMESPACES)
        if properties is None:
            raise ValueError("An entry is missing its Treasury properties")
        fields = {node.tag.split("}")[-1]: node.text for node in properties}
        if not fields.get("NEW_DATE"):
            raise ValueError("An entry has no observation date")
        observed = datetime.fromisoformat(fields["NEW_DATE"]).date()
        if observed > cutoff:
            continue
        if observed in seen:
            raise ValueError(f"Duplicate observation date: {observed}")
        seen.add(observed)
        row = {"date": observed.isoformat()}
        for key, text in fields.items():
            if key.startswith("BC_"):
                value = None if text is None or text.strip() == "" else float(text)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"Non-finite yield in {key} on {observed}")
                row[key] = value
        if not any(key.startswith("BC_") for key in row):
            raise ValueError("No recognised yield fields in an entry")
        rows.append(row)
    if not rows:
        raise ValueError("No Treasury observations on or before the requested cutoff")
    return sorted(rows, key=lambda row: row["date"])


def fetch_snapshot(cutoff: date, output: Path):
    """Download the cutoff year's current feed and create a new dated snapshot.

    An existing directory is rejected to avoid silently revising saved evidence.
    Network errors propagate; there is no synthetic-data fallback.
    """
    if output.exists():
        raise FileExistsError(f"Snapshot already exists: {output}; choose a new output directory")
    retrieved = datetime.now(timezone.utc)
    if cutoff > retrieved.date():
        raise ValueError("The cutoff cannot be in the future")
    query = urllib.parse.urlencode({"data": "daily_treasury_yield_curve", "field_tdr_date_value": cutoff.year})
    url = f"{FEED}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/xml, text/xml"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        content_type = response.headers.get("Content-Type")
        final_url = response.url
    rows = parse_par_yields(raw, cutoff)
    if any(date.fromisoformat(row["date"]).year != cutoff.year for row in rows):
        raise ValueError("Source returned observations outside the requested year")
    output.mkdir(parents=True)
    (output / "source.xml").write_bytes(raw)
    fields = ["date"] + sorted(set().union(*(set(row) - {"date"} for row in rows)))
    with (output / "par_yields.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "provider": "US Department of the Treasury",
        "dataset": "Daily Treasury Par Yield Curve Rates",
        "data_kind": "observed_official_published_series",
        "source_url": url, "final_url": final_url, "documentation_url": DOCUMENTATION,
        "retrieved_utc": retrieved.isoformat(), "observation_cutoff": cutoff.isoformat(),
        "first_observation": rows[0]["date"], "latest_observation": rows[-1]["date"],
        "calendar_days_since_latest_observation": (cutoff - date.fromisoformat(rows[-1]["date"])).days,
        "observations": len(rows), "units": "annual percentage par yields, as published",
        "missing_value_policy": "preserved as empty CSV cells; no imputation",
        "transformation": "XML fields parsed numerically; sorted by observation date; cutoff applied",
        "vintage_limitation": "Current source retrieval, not an as-published historical vintage",
        "pricing_limitation": "Par yields are not zero rates or an OIS discount curve; no direct substitution into pricing",
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "content_type": content_type,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "observations": len(rows), "latest_observation": rows[-1]["date"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch_snapshot(args.as_of, args.output)


if __name__ == "__main__":
    main()
