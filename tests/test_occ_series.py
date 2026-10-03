from datetime import date
from decimal import Decimal
import pytest

from qf_research.market_data.occ_series import parse_series_search, single_underlying


HEADER = "ProductSymbol\tyear\tMonth\tDay\tInteger\tDec\tC/P\tCall\tPut\tPosition Limit\t\n"
ROW = "SPY   \t\t2026\t10\t16\t750\t125\tC P \t0\t3\t360000000\n"


def test_decimal_strike_both_sides_and_source_line():
    rows = parse_series_search("Source heading\n"+HEADER+ROW)
    assert len(rows) == 2
    assert rows[0].strike == Decimal("750.125")
    assert rows[0].expiry == date(2026, 10, 16)
    assert rows[0].source_line == 3
    assert [r.kind for r in rows] == ["call", "put"]


@pytest.mark.parametrize("text", [ROW, HEADER, HEADER+ROW+ROW,
                                  HEADER+ROW.replace("\t125\t", "\t12\t"),
                                  HEADER+ROW.replace("C P", "unknown")])
def test_incomplete_ambiguous_or_changed_layout_rejected(text):
    with pytest.raises(ValueError):
        parse_series_search(text)


def test_scalar_and_singleton_underlying_reference_agree():
    assert single_underlying("SPY") == single_underlying(["SPY"]) == "SPY"


@pytest.mark.parametrize("value", [[], ["SPY", "QQQ"], [None], "", None])
def test_ambiguous_or_missing_underlying_rejected(value):
    with pytest.raises(ValueError):
        single_underlying(value)
