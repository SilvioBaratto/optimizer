"""Screener-backed universe source (SPEC D1/D9/D14).

Implements the ``Trading212ApiClient`` seam (``get_exchanges`` / ``get_instruments``)
so it drops into the existing ``UniverseBuilder`` in place of the Trading212 client.
Instruments come from ``yf.screen`` — no seed lists, no ISIN at build time (ISIN is
backfilled later from ticker profiles). Venues without a Yahoo-code → config-name
mapping are dropped and logged, so nothing is silently truncated.

Pipeline (why this matters): ``yf.screen`` exposes ~68k equities + ~49k ETFs, ~90%
of which are cross-listings of the same entity (``NVDA`` / ``NVD.DE`` / ``1NVDA.MI`` /
``NVDA.SW``) or microcap/OTC junk. This source (1) enumerates **per exchange code**
(stocks, market-cap desc) and **per region** (ETFs, net-assets desc), paging to the
screener's ~10k window with no artificial size cap; (2) **collapses cross-listings**
to one canonical listing per entity — group by normalized ``longName`` guarded by
``financialCurrency``, keep the max USD-ADDV (primary) line (see ``canonical.py``);
(3) applies a loose USD-normalized **anti-junk floor** (drops shells/dead/unpriced,
not small caps). Dedup + floor are the only reducers, so the result is the full
deduplicated investable cross-section rather than an alphabetical slice.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf

from app.services.market_data.yfinance.screener.screener_client import ScreenerClient
from app.services.universe.canonical import (
    IngestionFloorConfig,
    Listing,
    dedup_canonical,
    derive_metrics,
    passes_floor,
    split_currency,
)
from app.services.universe.trading212.config import UniverseBuilderConfig

logger = logging.getLogger(__name__)

_PAGE_SIZE = 250
# The screener returns at most a ~10k-row window per query; page to that ceiling
# and let dedup + the floor be the reducer (no artificial per-venue size cap).
_MAX_PER_QUERY = 10_000

# Rank by size, descending — pull the investable head of each venue, not the
# alphabetical (microcap) tail. Stocks rank on market cap; ETFs have no market cap
# so they rank on net assets (the field yf.screen accepts for ETFQuery).
_STOCK_SORT_FIELD = "intradaymarketcap"
_ETF_SORT_FIELD = "fundnetassets"

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

# ETFs are not filterable by exchange code on yf.screen (``eq exchange NMS`` returns
# nothing for funds); they are queried by region. Map each in-scope exchange code to
# its Yahoo region so ETF coverage tracks the same venue set. Returned ETF quotes are
# still bucketed by their own exchange code via _CODE_TO_CONFIG_NAME, so out-of-scope
# venues (e.g. a US ETF on BATS) drop exactly as stocks do.
_CODE_TO_REGION: dict[str, str] = {
    "NMS": "us",
    "NGM": "us",
    "NCM": "us",
    "NYQ": "us",
    "PCX": "us",
    "ASE": "us",
    "LSE": "gb",
    "IOB": "gb",
    "PAR": "fr",
    "AMS": "nl",
    "BRU": "be",
    "LIS": "pt",
    "GER": "de",
    "MUN": "de",
    "EBS": "ch",
    "VIE": "at",
    "MIL": "it",
    "MCE": "es",
    "MAD": "es",
    "TOR": "ca",
}


# Deterministic venue tiebreak for canonical selection (US primaries first, then
# the main European home venues). ADDV + home-currency dominate; this only breaks
# exact ties. Values are Yahoo exchange codes.
_EXCHANGE_PREF: tuple[str, ...] = (
    "NMS",
    "NYQ",
    "NGM",
    "NCM",
    "PCX",
    "ASE",  # US
    "LSE",
    "GER",
    "PAR",
    "AMS",
    "MIL",
    "MCE",
    "EBS",
    "TOR",
    "VIE",
    "BRU",
    "LIS",
    "IOB",
    "MUN",
    "MAD",
)


def _live_usd_per_major(majors: set[str]) -> dict[str, float]:
    """USD per 1 major unit via yfinance ``{MAJOR}USD=X`` spot (last close over a
    short window). ``USD`` is 1.0; a currency that fails to resolve is simply
    omitted (the floor then fail-opens on it). Network — injected so tests don't hit it.
    """
    rates: dict[str, float] = {"USD": 1.0}
    for major in majors:
        if not major or major == "USD":
            continue
        try:
            close = yf.Ticker(f"{major}USD=X").history(period="5d")["Close"].dropna()
            if not close.empty:
                rates[major] = float(close.iloc[-1])
            else:
                logger.warning("FX resolve empty for %s (fail-open)", major)
        except Exception:
            logger.warning("FX resolve failed for %s (fail-open)", major)
    return rates


def _quote_to_listing(quote: dict[str, Any]) -> Listing:
    """Map a screener quote onto the dedup/floor :class:`Listing`. ``exchange`` is
    kept as the Yahoo CODE (NMS/GER/…) for cross-venue grouping + bucketing."""
    return Listing(
        symbol=quote["symbol"],
        exchange=quote.get("exchange"),
        long_name=quote.get("longName"),
        short_name=quote.get("shortName"),
        currency=quote.get("currency"),
        financial_currency=quote.get("financialCurrency"),
        price=quote.get("regularMarketPrice"),
        avg_volume=(
            quote.get("averageDailyVolume3Month")
            or quote.get("averageDailyVolume10Day")
            or quote.get("regularMarketVolume")
        ),
        market_cap=quote.get("marketCap"),
        shares_outstanding=quote.get("sharesOutstanding"),
        net_assets=quote.get("netAssets"),
    )


@dataclass
class PassThroughTickerMapper:
    """Yahoo symbols are already resolved tickers — ``discover`` echoes the symbol."""

    def discover(self, symbol: str, exchange_name: str | None = None) -> str | None:
        return symbol or None


@dataclass
class YFinanceUniverseSource:
    """Builds the ``UniverseBuilder`` exchange/instrument shape from ``yf.screen``.

    There is no artificial per-venue size cap — each query pages to the screener's
    ~10k window (market-cap / net-asset ranked) and dedup + the coarse floor are the
    only reducers. Cross-listings collapse to one canonical listing per entity
    (``dedup=True``); the loose USD floor drops non-instruments only.
    """

    screener: ScreenerClient
    config: UniverseBuilderConfig = field(default_factory=UniverseBuilderConfig)
    max_stocks_per_exchange: int = _MAX_PER_QUERY
    max_etfs_per_region: int = _MAX_PER_QUERY
    dedup: bool = True
    floor_config: IngestionFloorConfig = field(default_factory=IngestionFloorConfig)
    # Injected so tests don't hit the network; production resolves live FX spot.
    fx_resolver: Callable[[set[str]], dict[str, float]] = _live_usd_per_major
    _loaded: bool = field(default=False, init=False)
    _exchanges: list[dict[str, Any]] = field(default_factory=list, init=False)
    _instruments: list[dict[str, Any]] = field(default_factory=list, init=False)

    def get_exchanges(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return self._exchanges

    def get_instruments(self) -> list[dict[str, Any]]:
        self._ensure_loaded()
        return self._instruments

    def _build_queries(self) -> list[tuple[Any, str, str, int]]:
        """``(query, instrument_type, sort_field, cap)`` tuples.

        Stocks: one query per exchange code, market-cap descending. ETFs: one query
        per unique region, net-assets descending. Scope is enforced downstream by the
        exchange-code allowlist in :meth:`_build_shape`, so a per-region ETF query is
        fine — its out-of-scope results drop there.
        """
        queries: list[tuple[Any, str, str, int]] = []
        for code in _CODE_TO_CONFIG_NAME:
            queries.append(
                (
                    yf.EquityQuery("eq", ["exchange", code]),
                    "STOCK",
                    _STOCK_SORT_FIELD,
                    self.max_stocks_per_exchange,
                )
            )
        for region in dict.fromkeys(_CODE_TO_REGION.values()):
            queries.append(
                (
                    yf.ETFQuery("eq", ["region", region]),
                    "ETF",
                    _ETF_SORT_FIELD,
                    self.max_etfs_per_region,
                )
            )
        return queries

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        quotes: list[dict[str, Any]] = []
        for query, kind, sort_field, cap in self._build_queries():
            # One venue/region failing must not abort the whole universe build.
            try:
                quotes.extend(self._paginate(query, kind, sort_field, cap))
            except Exception:
                logger.exception("Screener query failed (kind=%s); skipping", kind)
        self._build_shape(quotes)
        self._loaded = True

    def _paginate(
        self, query: Any, kind: str, sort_field: str, cap: int
    ) -> list[dict[str, Any]]:
        """Page one ranked query with ``offset`` until ``cap`` rows or exhaustion.

        A stable, meaningful ``sort_field`` (size descending) makes offset paging
        deterministic — Yahoo's default order can duplicate or skip rows across pages.
        """
        collected: list[dict[str, Any]] = []
        seen_syms: set[Any] = set()
        offset = 0
        while len(collected) < cap:
            result = self.screener.screen(
                query,
                size=_PAGE_SIZE,
                offset=offset,
                sort_field=sort_field,
                sort_asc=False,
            )
            page_quotes = (result or {}).get("quotes", [])
            if not page_quotes:
                break
            # Past the ~10k window the screener clamps offset and re-returns the last
            # page; a page with no new symbols means we've hit that wall — stop.
            fresh = [q for q in page_quotes if q.get("symbol") not in seen_syms]
            if not fresh:
                break
            for quote in fresh:
                quote["_kind"] = kind
                seen_syms.add(quote.get("symbol"))
            collected.extend(fresh)
            if len(page_quotes) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return collected[:cap]

    def _build_shape(self, quotes: list[dict[str, Any]]) -> None:
        # 1. Parse in-allowlist quotes -> Listing; drop cross-query dups by (symbol, code).
        parsed: list[Listing] = []
        kind_by_key: dict[tuple[str, str | None], str] = {}
        seen_raw: set[tuple[str, str | None]] = set()
        dropped = 0
        for quote in quotes:
            symbol = quote.get("symbol")
            code = quote.get("exchange")
            if not symbol or code not in _CODE_TO_CONFIG_NAME:
                dropped += 1
                continue
            raw_key = (symbol, code)
            if raw_key in seen_raw:
                continue
            seen_raw.add(raw_key)
            parsed.append(_quote_to_listing(quote))
            kind_by_key[raw_key] = quote.get("_kind", "STOCK")

        # 2. Resolve FX for the major currencies present (USD-numeraire floor).
        majors = {m for m in (split_currency(lst.currency)[0] for lst in parsed) if m}
        fx = self.fx_resolver(majors) if majors else {"USD": 1.0}
        fx.setdefault("USD", 1.0)

        # 3. Collapse cross-listings to one canonical listing per entity.
        survivors = (
            dedup_canonical(
                parsed, fx, config=self.floor_config, exchange_pref=_EXCHANGE_PREF
            )
            if self.dedup
            else parsed
        )

        # 4. Coarse anti-junk floor, then bucket survivors by config exchange name.
        schedule_ids: dict[str, int] = {}
        by_exchange: dict[str, list[dict[str, Any]]] = defaultdict(list)
        floored = 0
        for lst in survivors:
            if not passes_floor(derive_metrics(lst, fx), self.floor_config):
                floored += 1
                continue
            name = _CODE_TO_CONFIG_NAME.get(lst.exchange or "")
            if name is None:
                continue
            sched_id = schedule_ids.setdefault(name, len(schedule_ids) + 1)
            by_exchange[name].append(
                {
                    "ticker": lst.symbol,
                    "type": kind_by_key.get((lst.symbol, lst.exchange), "STOCK"),
                    "isin": None,
                    "currencyCode": lst.currency,
                    "name": lst.long_name or lst.short_name or lst.symbol,
                    "shortName": lst.symbol,
                    "workingScheduleId": sched_id,
                }
            )

        if dropped or floored:
            logger.info(
                "universe: dropped %d out-of-scope, %d below floor", dropped, floored
            )
        self._exchanges = [
            {"name": name, "workingSchedules": [{"id": schedule_ids[name]}]}
            for name in by_exchange
        ]
        self._instruments = [inst for insts in by_exchange.values() for inst in insts]
        logger.info(
            "yfinance universe source: %d canonical instruments across %d exchanges",
            len(self._instruments),
            len(self._exchanges),
        )
