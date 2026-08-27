"""Repository for market-wide calendars (SPEC B2) — idempotent upserts.

Rows arrive as loosely-typed record dicts (the yfinance DataFrame rows), so
each field is read defensively across the spellings Yahoo has used. Uses
``index_elements`` upserts (SQLite-testable).
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from app.models.market_data.calendars import (
    EarningsCalendar,
    EconomicEventCalendar,
    IpoCalendar,
    SplitCalendar,
)
from app.repositories._shared.base import RepositoryBase


def _col(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _date(v: Any):
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, errors="coerce", utc=True)
        if ts is None or pd.isna(ts):
            return None
        return ts.date()
    except (TypeError, ValueError):
        return None


def _num(v: Any) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    f = _num(v)
    return int(f) if f is not None else None


def _str(v: Any, n: int) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)[:n]


class CalendarsRepository(RepositoryBase):
    def upsert_earnings(self, rows: list[dict[str, Any]]) -> int:
        out: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "ticker", "Symbol", "symbol", "index"), 30)
            event_date = _date(
                _col(r, "startdatetime", "Earnings Date", "earningsDate", "date")
            )
            if not ticker or event_date is None:
                continue
            out[(ticker, event_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "event_date": event_date,
                "company_name": _str(
                    _col(r, "companyshortname", "Company", "company"), 255
                ),
                "eps_estimate": _num(_col(r, "epsestimate", "EPS Estimate")),
                "eps_actual": _num(_col(r, "epsactual", "Reported EPS")),
                "eps_surprise_pct": _num(_col(r, "epssurprisepct", "Surprise(%)")),
            }
        return self._commit(
            EarningsCalendar,
            list(out.values()),
            ["ticker", "event_date"],
            ["company_name", "eps_estimate", "eps_actual", "eps_surprise_pct"],
        )

    def upsert_ipos(self, rows: list[dict[str, Any]]) -> int:
        out: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "ticker", "Symbol", "symbol", "index"), 30)
            ipo_date = _date(_col(r, "startdatetime", "date", "Date"))
            if not ticker or ipo_date is None:
                continue
            out[(ticker, ipo_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "ipo_date": ipo_date,
                "company_name": _str(_col(r, "companyshortname", "Company"), 255),
                "exchange": _str(_col(r, "exchange", "Exchange"), 50),
                "price_range": _str(_col(r, "pricefrom", "priceRange", "Price"), 100),
                "currency": _str(_col(r, "currency", "Currency"), 10),
                "shares": _int(_col(r, "offersize", "shares", "Shares")),
            }
        return self._commit(
            IpoCalendar,
            list(out.values()),
            ["ticker", "ipo_date"],
            ["company_name", "exchange", "price_range", "currency", "shares"],
        )

    def upsert_splits(self, rows: list[dict[str, Any]]) -> int:
        out: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            ticker = _str(_col(r, "ticker", "Symbol", "symbol", "index"), 30)
            split_date = _date(_col(r, "startdatetime", "date", "Date"))
            if not ticker or split_date is None:
                continue
            out[(ticker, split_date)] = {
                "id": uuid.uuid4(),
                "ticker": ticker,
                "split_date": split_date,
                "company_name": _str(_col(r, "companyshortname", "Company"), 255),
                "ratio": _str(_col(r, "optionable", "ratio", "Ratio"), 50),
            }
        return self._commit(
            SplitCalendar,
            list(out.values()),
            ["ticker", "split_date"],
            ["company_name", "ratio"],
        )

    def upsert_economic_events(self, rows: list[dict[str, Any]]) -> int:
        out: dict[tuple, dict[str, Any]] = {}
        for r in rows:
            event = _str(_col(r, "event", "Event", "eventName"), 255)
            country = _str(_col(r, "country", "Country", "region"), 50) or "?"
            event_date = _date(_col(r, "startdatetime", "date", "Date"))
            if not event or event_date is None:
                continue
            out[(event, country, event_date)] = {
                "id": uuid.uuid4(),
                "event": event,
                "country": country,
                "event_date": event_date,
                "actual": _str(_col(r, "actual", "Actual"), 50),
                "forecast": _str(_col(r, "forecast", "Forecast", "expected"), 50),
                "prior": _str(_col(r, "prior", "Prior", "previous"), 50),
            }
        return self._commit(
            EconomicEventCalendar,
            list(out.values()),
            ["event", "country", "event_date"],
            ["actual", "forecast", "prior"],
        )

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
