"""Strict parser for the inspected OCC series-search batch text layout.

Listing identity is distinct from deliverable terms and historical information.
Open-interest columns are not interpreted as same-time liquidity observations.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class ListedSeries:
    symbol: str
    expiry: date
    strike: Decimal
    kind: str
    source_line: int


def single_underlying(value):
    """Accept a scalar or singleton reference list, rejecting ambiguous baskets."""
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("Require exactly one underlying identifier")
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Require a non-empty underlying identifier")
    return value.strip()


def parse_series_search(text):
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.startswith("ProductSymbol\t")]
    if len(headers) != 1:
        raise ValueError("Require one OCC batch series header")
    result = []
    for index, line in enumerate(lines[headers[0]+1:], headers[0]+2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 11 or fields[1] != "":
            raise ValueError(f"Unexpected OCC row structure at line {index}")
        symbol, _, year, month, day, whole, fraction, cp, calls, puts, limit = [f.strip() for f in fields]
        if (not symbol or not whole.isdigit() or len(fraction) != 3 or not fraction.isdigit()
                or cp not in ("C", "P", "C P")):
            raise ValueError(f"Invalid series identity at line {index}")
        expiry = date(int(year), int(month), int(day))
        strike = Decimal(whole)+Decimal(fraction)/Decimal(1000)
        for flag, kind in (("C", "call"), ("P", "put")):
            if flag in cp:
                result.append(ListedSeries(symbol, expiry, strike, kind, index))
    keys = [(r.symbol, r.expiry, r.strike, r.kind) for r in result]
    if not result or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicated OCC series identities")
    return tuple(result)
