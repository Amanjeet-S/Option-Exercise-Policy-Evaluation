"""Bounded, source-versioned dates for a core-close research exercise schedule.

These dates do not certify a broker instruction deadline or a full American
exercise opportunity set. Settlement projections require explicit opt-in.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


def _date(value):
    if type(value) is not date:
        raise ValueError("A calendar date, not a timestamp, is required")


def _aware(value):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("An explicitly zoned timestamp is required")


def act365f(start: datetime, end: datetime) -> float:
    """Elapsed UTC seconds, including DST changes, divided by 365 days."""
    _aware(start)
    _aware(end)
    seconds = (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()
    if seconds < 0:
        raise ValueError("Negative time interval")
    return seconds / (365 * 86400)


@dataclass(frozen=True)
class Session:
    opens_at: datetime
    closes_at: datetime


@dataclass(frozen=True)
class SettlementDate:
    day: date
    evidence: str


@dataclass(frozen=True)
class EventCalendar:
    start: date
    end: date
    dtc_schedule_through: date
    trading_holidays: frozenset[date]
    settlement_holidays: frozenset[date]
    early_closes: tuple[tuple[date, time], ...]

    def __post_init__(self):
        for d in (self.start, self.end, self.dtc_schedule_through):
            _date(d)
        if not self.start <= self.dtc_schedule_through <= self.end:
            raise ValueError("Invalid bounded calendar coverage")
        early_dates = [d for d, _ in self.early_closes]
        if len(set(early_dates)) != len(early_dates):
            raise ValueError("Duplicate early-close dates")
        for d in self.trading_holidays | self.settlement_holidays | set(early_dates):
            self._covered(d)
        for d, close in self.early_closes:
            if not self.is_trading_day(d) or close.tzinfo is not None or not time(9, 30) < close < time(16):
                raise ValueError("Invalid underlying early close")

    @classmethod
    def from_config(cls, config):
        if config["timezone"] != "America/New_York" or config["core_open"] != "09:30:00" or config["core_close"] != "16:00:00":
            raise ValueError("Unsupported market-session convention")
        return cls(date.fromisoformat(config["coverage_start"]),
                   date.fromisoformat(config["coverage_end"]),
                   date.fromisoformat(config["dtc_schedule_through"]),
                   frozenset(date.fromisoformat(d) for d in config["trading_holidays"]),
                   frozenset(date.fromisoformat(d) for d in config["settlement_holidays"]),
                   tuple((date.fromisoformat(d), time.fromisoformat(t))
                         for d, t in config["early_closes"].items()))

    def _covered(self, d):
        _date(d)
        if not self.start <= d <= self.end:
            raise ValueError("Date outside the source calendar's coverage")

    def is_trading_day(self, d: date) -> bool:
        self._covered(d)
        return d.weekday() < 5 and d not in self.trading_holidays

    def session(self, d: date) -> Session:
        if not self.is_trading_day(d):
            raise ValueError("No core trading session on this date")
        close = dict(self.early_closes).get(d, time(16))
        return Session(datetime.combine(d, time(9, 30), NEW_YORK),
                       datetime.combine(d, close, NEW_YORK))

    def previous_session(self, d: date) -> Session:
        self._covered(d)
        previous = d - timedelta(days=1)
        while not self.is_trading_day(previous):
            previous -= timedelta(days=1)
        return self.session(previous)

    def is_settlement_day(self, d: date, *, allow_provisional=False) -> bool:
        self._covered(d)
        if d > self.dtc_schedule_through and not allow_provisional:
            raise ValueError("No DTC schedule evidence: provisional settlement requires opt-in")
        return d.weekday() < 5 and d not in self.settlement_holidays

    def settlement_after(self, d: date, *, lag=1, allow_provisional=False) -> SettlementDate:
        """Positive business-day lag from an ordinary trading-day instruction."""
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("Settlement lag must be a positive integer")
        self.session(d)
        candidate = d
        remaining = lag
        while remaining:
            candidate += timedelta(days=1)
            if self.is_settlement_day(candidate, allow_provisional=allow_provisional):
                remaining -= 1
        evidence = ("published_2026_dtc_schedule" if candidate <= self.dtc_schedule_through
                    else "provisional_2027_no_dtc_schedule")
        return SettlementDate(candidate, evidence)

    def research_decisions(self, observation: datetime, expiry: date) -> tuple[datetime, ...]:
        """Underlying core closes, including expiry; no after-hours decisions."""
        _aware(observation)
        local = observation.astimezone(NEW_YORK)
        session = self.session(local.date())
        if not session.opens_at <= local <= session.closes_at:
            raise ValueError("Observation is outside the underlying core session")
        self.session(expiry)
        if expiry < local.date():
            raise ValueError("Expiry precedes observation")
        result = []
        d = local.date()
        while d <= expiry:
            if self.is_trading_day(d):
                result.append(self.session(d).closes_at)
            d += timedelta(days=1)
        return tuple(result)

    def model_ex_event(self, ex_date: date) -> datetime:
        """09:30 New York is a numerical convention, not a legal event time."""
        return self.session(ex_date).opens_at


def prior_ex_date_test(instruction: datetime, ex_date: date) -> bool:
    """Date condition only; broker acceptance and contract terms remain separate."""
    _aware(instruction)
    _date(ex_date)
    return instruction.astimezone(NEW_YORK).date() < ex_date
