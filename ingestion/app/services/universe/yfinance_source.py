"""Screener-backed universe source (SPEC D1/D9/D14).

Implements the ``Trading212ApiClient`` seam (``get_exchanges`` / ``get_instruments``)
so it drops into the existing ``UniverseBuilder`` in place of the Trading212 client.
Instruments come from ``yf.screen`` — no seed lists, no ISIN (identity is
``(ticker, exchange)``; dedup by symbol; ISIN is fetched lazily only by the T212
annotation step). Venues without a Yahoo-code → config-name mapping are dropped
and logged, so nothing is silently truncated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf

from app.services.market_data.yfinance.screener.screener_client import ScreenerClient
from app.services.universe.trading212.config import UniverseBuilderConfig

logger = logging.getLogger(__name__)

_PAGE_SIZE = 250

# Stable sort field so offset pagination is deterministic — without it, paging over
# Yahoo's default order can duplicate or skip rows across pages. "ticker" is yf.screen's
# own default sort and is present for every query class.
_SORT_FIELD = "ticker"

# Yahoo exchange code -> config exchange name (must be a key of
# UniverseBuilderConfig.yahoo_suffix_map, since the builder filters on that set).
# Codes are Yahoo's exchange abbreviations; verify against live screener output
# and extend as coverage grows.
_CODE_TO_CONFIG_NAME: dict[str, str] = {
    # US
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "NCM": "NASDAQ",
    "NYQ": "NYSE",
    "PCX": "NYSE",
    "ASE": "NYSE",
    # UK
    "LSE": "London Stock Exchange",
    "IOB": "London Stock Exchange",
    # Euronext
    "PAR": "Euronext Paris",
    "AMS": "Euronext Amsterdam",
    "BRU": "Euronext Brussels",
    "LIS": "Euronext Lisbon",
    # Germany / Austria / Switzerland
    "GER": "Deutsche Börse Xetra",
    "MUN": "Gettex",
    "EBS": "SIX Swiss Exchange",
    "VIE": "Wiener Börse",
    # Italy / Spain
    "MIL": "Borsa Italiana",
    "MCE": "Bolsa de Madrid",
    "MAD": "Bolsa de Madrid",
    # Canada
    "TOR": "Toronto Stock Exchange",
}


@dataclass
class PassThroughTickerMapper:
    """Yahoo symbols are already resolved tickers — ``discover`` echoes the symbol."""

    def discover(self, symbol: str, exchange_name: str | None = None) -> str | None:
        return symbol or None


@dataclass
class YFinanceUniverseSource:
    """Builds the ``UniverseBuilder`` exchange/instrument shape from ``yf.screen``."""

    screener: ScreenerClient
    config: UniverseBuilderConfig = field(default_factory=UniverseBuilderConfig)
    max_pages: int = 4
    _loaded: bool = field(default=False, init=False)
    _exchanges: list[dict[str, Any]] = field(default_factory=list, init=False)
    _instruments: list[dict[str, Any]] = field(default_factory=list, init=False)

    def get_exchanges(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return self._exchanges

    def get_instruments(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return self._instruments

    def _build_queries(self) -> list[tuple[Any, str]]:
        """(query, instrument_type) pairs. Scoping is enforced downstream via the
        exchange-code allowlist, so a broad query is fine."""
        codes = list(_CODE_TO_CONFIG_NAME)
        return [
            (yf.EquityQuery("is-in", ["exchange", *codes]), "STOCK"),
            (yf.ETFQuery("is-in", ["exchange", *codes]), "ETF"),
        ]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        quotes: list[dict[str, Any]] = []
        for query, kind in self._build_queries():
            quotes.extend(self._paginate(query, kind))
        self._build_shape(quotes)
        self._loaded = True

    def _paginate(self, query: Any, kind: str) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for page in range(self.max_pages):
            offset = page * _PAGE_SIZE
            result = self.screener.screen(
                query,
                size=_PAGE_SIZE,
                offset=offset,
                sort_field=_SORT_FIELD,
                sort_asc=True,
            )
            page_quotes = (result or {}).get("quotes", [])
            if not page_quotes:
                break
            for quote in page_quotes:
                quote["_kind"] = kind
            collected.extend(page_quotes)
            if len(page_quotes) < _PAGE_SIZE:
                break
        return collected

    def _build_shape(self, quotes: list[dict[str, Any]]) -> None:
        schedule_ids: dict[str, int] = {}
        by_exchange: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str]] = set()
        dropped = 0

        for quote in quotes:
            symbol = quote.get("symbol")
            code = quote.get("exchange")
            name = _CODE_TO_CONFIG_NAME.get(code) if code else None
            if not symbol or name is None:
                dropped += 1
                continue
            key = (symbol, name)
            if key in seen:
                continue
            seen.add(key)
            sched_id = schedule_ids.setdefault(name, len(schedule_ids) + 1)
            by_exchange.setdefault(name, []).append(
                {
                    "ticker": symbol,
                    "type": quote.get("_kind", "STOCK"),
                    "isin": None,
                    "currencyCode": quote.get("currency"),
                    "name": quote.get("longName") or quote.get("shortName") or symbol,
                    "shortName": symbol,
                    "workingScheduleId": sched_id,
                }
            )

        if dropped:
            logger.info("Dropped %d out-of-scope/invalid screener quotes", dropped)

        self._exchanges = [
            {"name": name, "workingSchedules": [{"id": schedule_ids[name]}]}
            for name in by_exchange
        ]
        self._instruments = [inst for insts in by_exchange.values() for inst in insts]
