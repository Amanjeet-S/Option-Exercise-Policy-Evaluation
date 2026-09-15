"""Check ingestion against a retained official source, with no network dependency."""

from datetime import date
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from qf_research.market_data.treasury import NAMESPACES, parse_par_yields


SNAPSHOT = Path(__file__).resolve().parents[1] / "data/public/treasury/2026-09-14"


def test_official_snapshot_integrity_and_observations():
    raw = (SNAPSHOT / "source.xml").read_bytes()
    meta = json.loads((SNAPSHOT / "metadata.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == meta["raw_sha256"]
    rows = parse_par_yields(raw, date.fromisoformat(meta["observation_cutoff"]))
    assert len(rows) == meta["observations"]
    assert rows[-1]["date"] == meta["latest_observation"]
    assert rows[0]["date"] < rows[-1]["date"]
    assert rows[-1]["BC_1YEAR"] == 4.35  # Official 11 September 2026 observation.
    assert rows[-1]["BC_10YEAR"] == 4.96


def test_cutoff_never_includes_a_later_observation():
    raw = (SNAPSHOT / "source.xml").read_bytes()
    rows = parse_par_yields(raw, date(2026, 1, 9))
    assert len(rows) == 6
    assert all(row["date"] <= "2026-01-09" for row in rows)
    assert rows[-1]["date"] == "2026-01-09"


def test_duplicate_source_observation_is_rejected():
    root = ET.fromstring((SNAPSHOT / "source.xml").read_bytes())
    root.append(root.find("a:entry", NAMESPACES))
    with pytest.raises(ValueError, match="Duplicate"):
        parse_par_yields(ET.tostring(root), date(2026, 9, 14))


def test_missing_value_is_preserved_without_imputation():
    root = ET.fromstring((SNAPSHOT / "source.xml").read_bytes())
    entry = root.find("a:entry", NAMESPACES)
    properties = entry.find(".//m:properties", NAMESPACES)
    field = next(node for node in properties if node.tag.endswith("}BC_1YEAR"))
    observed = next(node.text[:10] for node in properties if node.tag.endswith("}NEW_DATE"))
    field.text = None  # Inject missingness into the real source to test the policy.
    rows = parse_par_yields(ET.tostring(root), date(2026, 9, 14))
    assert next(row for row in rows if row["date"] == observed)["BC_1YEAR"] is None


def test_empty_or_wrong_document_fails_closed():
    with pytest.raises(ValueError, match="No Treasury"):
        parse_par_yields(b"<html>No observations</html>", date(2026, 9, 14))
