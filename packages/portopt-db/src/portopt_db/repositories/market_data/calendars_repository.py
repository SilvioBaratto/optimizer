"""Repository for market-wide calendars (SPEC B2) — idempotent upserts.

Rows arrive as loosely-typed record dicts (the yfinance DataFrame rows), so
each field is read defensively across the spellings Yahoo has used. Uses
``index_elements`` upserts (SQLite-testable).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any, cast

import pandas as pd
from sqlalchemy import Table, delete, select, tuple_
from sqlalchemy.orm import DeclarativeBase

from portopt_db.models.market_data.calendars import (
    EarningsCalendar,
    EconomicEventCalendar,
    IpoCalendar,
    SplitCalendar,
)
from portopt_db.repositories.market_data._calendar_timezones import region_timezone
from portopt_db.repository import RepositoryBase

# Keeps the IN list well under SQLite's bound-variable limit.
_DELETE_CHUNK = 500
# Three bound variables per key (event, country, event_date).
_KEY_DELETE_CHUNK = 150


def _col(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _timestamp(v: Any) -> pd.Timestamp | None:
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if ts is None or pd.isna(ts):
            return None
        return cast(pd.Timestamp, ts)
    except (TypeError, ValueError):
        return None


def _date(v: Any) -> date | None:
    """Return the UTC calendar date of ``v``.

    Right for date-only fields (IPO date, split payable date), which Yahoo
    stamps at midnight US Eastern, so their UTC date is the intended date.
    """
    ts = _timestamp(v)
    return None if ts is None else cast(date, ts.date())


def _release_date(v: Any, region: str | None) -> date | None:
    """Return the calendar date of ``v`` in the timezone of ``region``.

    A timestamp at exactly 00:00Z is Yahoo's placeholder for an event with no
    release time; converting it to a zone west of UTC would move the event to
    the previous day, so it keeps its UTC date. An unknown region also falls
    back to the UTC date.
    """
    ts = _timestamp(v)
    if ts is None:
        return None
    tz = region_timezone(region)
    if tz is None or ts == ts.normalize():
        return cast(date, ts.date())
    return cast(date, ts.tz_convert(tz).date())


def _num(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _ratio_str(old: float, new: float) -> str:
    """Render a split ratio ``old:new`` (e.g. ``5:1``), trimming whole floats."""

    def _fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:g}"

    return f"{_fmt(old)}:{_fmt(new)}"


def _str(v: Any, n: int) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)[:n]


class CalendarsRepository(RepositoryBase):
    """Persist market-wide calendar data via idempotent upserts.

    Each upsert method deduplicates on the natural key before writing,
    so callers may pass the raw yfinance payload repeatedly without
    accumulating duplicates. The ticker-keyed calendars (earnings, IPOs,
    splits) also drop a ticker's stale upcoming rows when the payload now
    lists that ticker on other dates, so a rescheduled event does not leave
    its old date behind.
    """

    def upsert_earnings(self, rows: list[dict[str, Any]]) -> int:
        """Upsert earnings-calendar records and return the count written.

        Args:
            rows: Raw yfinance DataFrame rows as dicts.  Ticker and event
                date are extracted defensively across all known Yahoo
                column-name variants; rows missing either field are skipped.

        Returns:
            Number of rows passed to the upsert (including no-ops where
            the natural key already existed).
        """
        out: dict[tuple[Any, ...], dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "Symbol", "ticker", "symbol", "index"), 30)
            # yfinance builds columns from humanized labels then renames; the
            # earnings date label is "Event Start Date".
            event_date = _date(
                _col(r, "Event Start Date", "startdatetime", "Earnings Date", "date")
            )
            if not ticker or event_date is None:
                continue
            out[(ticker, event_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "event_date": event_date,
                "company_name": _str(
                    _col(r, "Company", "Company Name", "companyshortname"), 255
                ),
                "eps_estimate": _num(_col(r, "EPS Estimate", "epsestimate")),
                "eps_actual": _num(_col(r, "Reported EPS", "epsactual")),
                "eps_surprise_pct": _num(_col(r, "Surprise(%)", "epssurprisepct")),
            }
        written = self._commit(
            EarningsCalendar,
            list(out.values()),
            ["ticker", "event_date"],
            ["company_name", "eps_estimate", "eps_actual", "eps_surprise_pct"],
        )
        self._drop_rescheduled(
            EarningsCalendar, list(out.values()), "event_date", "eps_actual"
        )
        return written

    def upsert_ipos(self, rows: list[dict[str, Any]]) -> int:
        """Upsert IPO-calendar records and return the count written.

        Args:
            rows: Raw yfinance DataFrame rows as dicts.  Ticker and IPO
                date are extracted defensively; rows missing either are
                skipped.

        Returns:
            Number of rows passed to the upsert.
        """
        out: dict[tuple[Any, ...], dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "Symbol", "ticker", "symbol", "index"), 30)
            # IPO datetime label is "Date"; identity fields use humanized labels.
            ipo_date = _date(_col(r, "Date", "startdatetime"))
            if not ticker or ipo_date is None:
                continue
            # yfinance's IPO calendar labels the company "Company" (not "Company
            # Name") and the currency "Currency" (not "Currency Name"). It carries
            # no price/share figures at all, so those columns were dropped.
            out[(ticker, ipo_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "ipo_date": ipo_date,
                "company_name": _str(
                    _col(r, "Company", "Company Name", "companyshortname"), 255
                ),
                "exchange": _str(_col(r, "Exchange", "exchange"), 50),
                "currency": _str(_col(r, "Currency", "Currency Name", "currency"), 10)
                or None,
            }
        written = self._commit(
            IpoCalendar,
            list(out.values()),
            ["ticker", "ipo_date"],
            ["company_name", "exchange", "currency"],
        )
        self._drop_rescheduled(IpoCalendar, list(out.values()), "ipo_date")
        return written

    def upsert_splits(self, rows: list[dict[str, Any]]) -> int:
        """Upsert split-calendar records and return the count written.

        Args:
            rows: Raw yfinance DataFrame rows as dicts.  The split ratio
                is derived from ``Old Share Worth`` and ``Share Worth``
                columns and serialised as ``old:new`` (e.g. ``5:1``).

        Returns:
            Number of rows passed to the upsert.
        """
        out: dict[tuple[Any, ...], dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "Symbol", "ticker", "symbol", "index"), 30)
            # Split datetime label is "Payable On".
            split_date = _date(_col(r, "Payable On", "startdatetime"))
            if not ticker or split_date is None:
                continue
            # The split ratio is encoded as Old Share Worth : Share Worth
            # (e.g. 5:1 = a 1-for-5 consolidation); the company label is "Company".
            old_worth = _num(_col(r, "Old Share Worth", "oldshareworth"))
            new_worth = _num(_col(r, "Share Worth", "shareworth"))
            ratio = (
                _ratio_str(old_worth, new_worth)
                if old_worth is not None and new_worth is not None
                else None
            )
            out[(ticker, split_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "split_date": split_date,
                "company_name": _str(
                    _col(r, "Company", "Company Name", "companyshortname"), 255
                ),
                "ratio": ratio,
            }
        written = self._commit(
            SplitCalendar,
            list(out.values()),
            ["ticker", "split_date"],
            ["company_name", "ratio"],
        )
        self._drop_rescheduled(SplitCalendar, list(out.values()), "split_date")
        return written

    def upsert_economic_events(self, rows: list[dict[str, Any]]) -> int:
        """Upsert economic-event calendar records and return the count written.

        Events are keyed on their local release date. Rows stored before that
        keying (under the UTC date) are removed when the payload re-sends the
        same event under its local date, so the move does not leave a permanent
        duplicate next to each early-morning or late-evening release.

        Args:
            rows: Raw yfinance DataFrame rows as dicts.  Country defaults
                to ``"?"`` when absent so the natural key (event, country,
                event_date) remains non-null.

        Returns:
            Number of rows passed to the upsert.
        """
        out: dict[tuple[Any, ...], dict[str, Any]] = {}
        utc_keyed: set[tuple[str, str, date]] = set()
        for r in rows:
            event = _str(_col(r, "Event", "event", "eventName"), 255)
            # country_code is renamed to "Region"; date label is "Event Time".
            country = _str(_col(r, "Region", "country", "Country"), 50) or "?"
            instant = _col(r, "Event Time", "startdatetime", "date")
            event_date = _release_date(instant, country)
            if not event or event_date is None:
                continue
            utc_date = _date(instant)
            if utc_date is not None and utc_date != event_date:
                utc_keyed.add((event, country, utc_date))
            out[(event, country, event_date)] = {
                "id": uuid.uuid4(),
                "event": event,
                "country": country,
                "event_date": event_date,
                "actual": _str(_col(r, "Actual", "actual"), 50),
                "forecast": _str(_col(r, "Expected", "forecast", "Forecast"), 50),
                "prior": _str(_col(r, "Last", "prior", "Prior"), 50),
            }
        written = self._commit(
            EconomicEventCalendar,
            list(out.values()),
            ["event", "country", "event_date"],
            ["actual", "forecast", "prior"],
        )
        # A key the payload itself dates there is a genuine occurrence (a daily
        # event), not the legacy copy of another one, so it is never removed.
        self._drop_utc_keyed_events(
            {key for key in utc_keyed if key not in out},
        )
        return written

    def _commit(
        self,
        model: type,
        rows: list[dict[str, Any]],
        index_elements: list[str],
        update_columns: list[str],
    ) -> int:
        if not rows:
            return 0
        self._upsert(
            model,
            rows,
            index_elements=index_elements,
            update_columns=[*update_columns, "updated_at"],
        )
        return len(rows)

    def _drop_utc_keyed_events(self, keys: set[tuple[str, str, date]]) -> None:
        """Delete economic-event rows stored under a superseded UTC-date key.

        Args:
            keys: ``(event, country, utc_date)`` keys of events the payload has
                just stored under their local release date instead.
        """
        if not keys:
            return
        table = cast(Table, EconomicEventCalendar.__table__)
        ordered = sorted(keys)
        for start in range(0, len(ordered), _KEY_DELETE_CHUNK):
            self.session.execute(
                delete(table).where(
                    tuple_(table.c.event, table.c.country, table.c.event_date).in_(
                        ordered[start : start + _KEY_DELETE_CHUNK]
                    )
                )
            )

    def _drop_rescheduled(
        self,
        model: type[DeclarativeBase],
        rows: Sequence[dict[str, Any]],
        date_col: str,
        realized_col: str | None = None,
    ) -> None:
        """Delete stale upcoming rows of tickers the payload now dates differently.

        The natural keys embed the event date, so a rescheduled event is
        inserted as a new row and its old row would stay behind as a phantom
        upcoming event. Only a ticker with upcoming rows in the payload is
        reconciled, and only its upcoming rows without a realized value are
        removed: a ticker missing from the payload (Yahoo caps each fetch, so
        absence proves nothing) and past or reported rows are history and stay.

        Args:
            model: Calendar model keyed on ``(ticker, <date_col>)``.
            rows: The rows just upserted.
            date_col: Name of the event-date column.
            realized_col: Column that is non-NULL once the event has a result;
                rows holding one are never removed.
        """
        today = datetime.now(timezone.utc).date()
        current: dict[str, set[date]] = {}
        for row in rows:
            if row[date_col] >= today:
                current.setdefault(row["ticker"], set()).add(row[date_col])
        if not current:
            return

        table = cast(Table, model.__table__)
        stmt = select(table.c.id, table.c.ticker, table.c[date_col]).where(
            table.c[date_col] >= today
        )
        if realized_col is not None:
            stmt = stmt.where(table.c[realized_col].is_(None))
        stale = [
            row_id
            for row_id, ticker, event_date in self.session.execute(stmt)
            if ticker in current and event_date not in current[ticker]
        ]
        for start in range(0, len(stale), _DELETE_CHUNK):
            self.session.execute(
                delete(table).where(
                    table.c.id.in_(stale[start : start + _DELETE_CHUNK])
                )
            )
