"""Dated SPY cash declarations, preserving the Singapore/US calendar distinction.

This is a reconstruction from public filings retrieved later. A broadcast label
is evidence of a publication date, not proof of an immutable historical archive.
Current issuer rows are corroboration and never supply the forecast amounts.
"""

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
from html.parser import HTMLParser
import re
from zoneinfo import ZoneInfo

from .distribution_history import cash, parse_quarter, quarter_label, quarter_number
from .dividends import zoned_time


class _Fields(HTMLParser):
    """Read the inspected SGX definition-list layout, failing on duplicate keys."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields, self.titles, self.meta_dates, self.links = {}, [], [], {}
        self.active, self.key, self.parts, self.hrefs = None, None, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name") == "Date":
            self.meta_dates.append(attrs.get("content"))
        if tag in ("dt", "dd", "h1"):
            if self.active:
                raise ValueError("Nested or unclosed announcement field")
            self.active, self.parts, self.hrefs = tag, [], []
        if tag == "a" and self.active == "dd" and attrs.get("href"):
            self.hrefs.append(attrs["href"])

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag != self.active:
            return
        value = " ".join("".join(self.parts).split())
        if tag == "dt":
            if self.key is not None:
                raise ValueError("Announcement label has no value")
            self.key = value
        elif tag == "dd":
            if self.key is None or self.key in self.fields:
                raise ValueError("Missing or repeated announcement label")
            self.fields[self.key], self.links[self.key] = value, self.hrefs
            self.key = None
        else:
            self.titles.append(value)
        self.active = None


def _sgx_date(text):
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", text):
        raise ValueError("Unexpected SGX event-date format")
    return datetime.strptime(text, "%d/%m/%Y").date().isoformat()


def parse_announcement(body, *, source_url, retrieved_at):
    """Parse one actual USD declaration; retain raw clock and local ex-date."""
    p = _Fields()
    p.feed(body.decode("utf-8"))
    p.close()
    if p.active or p.key is not None or p.titles != ["Cash Dividend/ Distribution::Mandatory"]:
        raise ValueError("Incomplete or unreviewed announcement layout")
    f = p.fields
    expected = {
        "Announcement Title": "Mandatory Cash Dividend/ Distribution",
        "Status": "New", "Dividend/ Distribution Type": "Interim",
        "Payment Type": "Payment Rate in Gross", "Gross Rate Status": "Actual Rate",
    }
    if any(f.get(k) != v for k, v in expected.items()):
        raise ValueError("Unreviewed declaration type, status or rate")
    if f.get("Issuer/ Manager") not in (
        "SPDR S&P 500 ETF TRUST", "STATE STREET SPDR S&P 500 ETF TRUST"
    ) or f.get("Security") not in (
        "SPDR S&P 500 ETF TRUST - US78462F1030 - S27",
        "SS SPDR S&P 500 ETF TRUST - US78462F1030 - S27",
    ):
        raise ValueError("Wrong security identity")
    if "Related Announcements" not in f or f["Related Announcements"] or p.links["Related Announcements"]:
        raise ValueError("Related announcement or missing relationship field requires review")
    reference = f["Corporate Action Reference"]
    if not re.fullmatch(r"SG\d{6}DVCA[A-Z0-9]{4}", reference):
        raise ValueError("Unexpected corporate action reference")
    rates = []
    for key in ("Declared Dividend/ Distribution Rate (Per Share/ Unit)",
                "Gross Rate (Per Share)", "Net Rate (Per Share)"):
        match = re.fullmatch(r"USD (\d+(?:\.\d+)?)", f[key])
        if not match:
            raise ValueError("Missing USD per-unit cash amount")
        rates.append(cash(match[1]))
    if len(set(map(Decimal, rates))) != 1:
        raise ValueError("Declared, gross and net rates differ; separate review required")
    raw = f["Date &Time of Broadcast"]
    broadcast = datetime.strptime(raw, "%d-%b-%Y %H:%M:%S")
    if len(p.meta_dates) != 1:
        raise ValueError("Missing or repeated metadata date")
    # SGX's Z-suffixed meta value numerically equals its displayed clock. Do
    # not infer a UTC conversion from that suffix without timezone evidence.
    if p.meta_dates[0] != broadcast.strftime("%Y%m%dT%H:%M:%SZ"):
        raise ValueError("Display and metadata date labels differ")
    if reference[2:8] != broadcast.strftime("%y%m%d"):
        raise ValueError("Reference and broadcast date differ")
    bound = datetime.combine(broadcast.date() + timedelta(days=2), time(), timezone.utc)
    retrieval = zoned_time(retrieved_at)
    if bound > retrieval:
        raise ValueError("Declaration is too recent for the conservative use bound")
    ex, record, pay = (_sgx_date(f[k]) for k in ("Ex Date", "Record Date", "Pay Date"))
    if not ex <= record <= pay:
        raise ValueError("Inconsistent declaration dates")
    return dict(
        reference=reference, isin="US78462F1030", sgx_ticker="S27", currency="USD",
        gross_per_unit=rates[1], declared_per_unit=rates[0], net_per_unit=rates[2],
        sgx_ex_date=ex, record_date=record, pay_date=pay,
        quarter=quarter_label(quarter_number(record)), broadcast_label=raw,
        metadata_date_label=p.meta_dates[0], broadcast_timezone_verified=False,
        publication_bound_at=bound.isoformat(), retrieved_at=retrieval.isoformat(),
        source_url=source_url, source_sha256=sha256(body).hexdigest(),
        status=f["Status"], rate_status=f["Gross Rate Status"],
        related_announcements_visible=False, immutable_archive_verified=False,
    )


def reconcile_announcements(announcements, issuer):
    """Match by US record date, never by replacing it with the SGX ex-date."""
    result = []
    references, quarters = set(), set()
    for a in announcements:
        if a["reference"] in references or a["quarter"] in quarters:
            raise ValueError("Duplicate declaration or reference quarter")
        references.add(a["reference"])
        quarters.add(a["quarter"])
        candidates = [r for r in issuer if r["record_date"] == a["record_date"]]
        if len(candidates) != 1:
            raise ValueError("Announcement has no unique issuer record-date match")
        r = candidates[0]
        checks = dict(
            identity=r["instrument"] == "SPY" and r["cusip"] == "78462F103"
                     and a["isin"] == "US78462F1030" and a["sgx_ticker"] == "S27",
            currency=r["currency"] == a["currency"] == "USD",
            amount=r["dividend"] is not None and Decimal(r["dividend"]) == Decimal(a["gross_per_unit"]),
            payment=r["pay_date"] == a["pay_date"],
            quarter=quarter_label(quarter_number(r["ex_date"])) == a["quarter"],
            us_dates_consistent=not r["date_issues"],
        )
        result.append(dict(reference=a["reference"], issuer_row=r["source_row"],
                           sgx_ex_date=a["sgx_ex_date"], us_ex_date=r["ex_date"],
                           checks=checks, all_checks_passed=all(checks.values()),
                           blank_capital_gains_imputed=False))
    return result


def reconstruct_benchmarks(announcements, as_of, targets, last_completed_quarter):
    """Reconstruct three unestimated benchmarks from filed declaration amounts.

    This is explicitly a calculation made later, under public-archive evidence.
    It does not turn current workbook retrievals into historical observations.
    """
    origin = zoned_time(as_of)
    local_day = origin.astimezone(ZoneInfo("America/New_York")).date()
    last = parse_quarter(last_completed_quarter)
    if last != quarter_number(local_day.isoformat()) - 1:
        raise ValueError("Reference end is not the last completed quarter")
    by_quarter, references = {}, set()
    for a in announcements:
        q = parse_quarter(a["quarter"])
        if q in by_quarter or a["reference"] in references:
            raise ValueError("Duplicate declaration")
        by_quarter[q] = a
        references.add(a["reference"])

    def required(q):
        if q not in by_quarter:
            raise ValueError("Missing declaration quarter")
        a = by_quarter[q]
        if zoned_time(a["publication_bound_at"]) > origin:
            raise ValueError("Declaration unavailable at cutoff")
        if datetime.fromisoformat(a["pay_date"]).date() >= local_day:
            raise ValueError("Reference payment is not strictly before cutoff date")
        if a["isin"] != "US78462F1030" or a["currency"] != "USD":
            raise ValueError("Wrong declaration identity or currency")
        if a["status"] != "New" or a["rate_status"] != "Actual Rate" or a["related_announcements_visible"]:
            raise ValueError("Unreviewed declaration status")
        amount = cash(a["gross_per_unit"])
        if amount is None:
            raise ValueError("Missing declaration amount")
        return a, Decimal(amount)

    if not targets:
        raise ValueError("No forecast targets")
    results, seen_targets = [], set()
    for target in targets:
        ex = date.fromisoformat(target["ex_date"])
        pay = date.fromisoformat(target["pay_date"])
        q = quarter_number(ex.isoformat())
        if q in seen_targets or q - last not in (1, 2) or not local_day < ex <= pay:
            raise ValueError("Duplicate, expired or unsupported target")
        if ex.month not in (3, 6, 9, 12):
            raise ValueError("Off-cycle target")
        seen_targets.add(q)
        for model, qs in (
            ("last_regular", [last]),
            ("same_quarter_previous_year", [q - 4]),
            ("last_four_regular_mean", list(range(last - 3, last + 1))),
        ):
            inputs = [required(i) for i in qs]
            estimate = sum((value for _, value in inputs), Decimal(0)) / len(inputs)
            results.append(dict(
                as_of=origin.isoformat(), model=model, **target,
                regular_dividend_estimate=str(estimate), currency="USD",
                input_references=[a["reference"] for a, _ in inputs],
                latest_input_publication_bound=max(a["publication_bound_at"] for a, _ in inputs),
                calculation_type="retrospective_public_announcement_reconstruction",
                first_target_announcement_absence_verified=False,
                complete_cash_schedule=False, risk_neutral_expectation_established=False,
            ))
    return results
