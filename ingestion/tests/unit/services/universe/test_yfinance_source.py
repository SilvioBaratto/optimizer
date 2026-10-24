"""YFinanceUniverseSource contract.

The source implements the ``Trading212ApiClient`` seam (get_exchanges /
get_instruments) from ``yf.screen`` results: no seed lists, no ISIN (identity is
(ticker, exchange); dedup by symbol), out-of-scope venues dropped. Query
construction is patched so tests exercise the transform, not live Yahoo.

Ranking contract: stocks are screened per exchange code by market cap descending,
ETFs per region by net assets descending, each capped — so paging returns the
investable head of every venue, not the alphabetical microcap tail.
"""

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("yfinance")

from app.services.market_data.yfinance.screener.screener_client import (
    ScreenerClient,
    ScreenerError,
)
from app.services.universe import yfinance_source
from app.services.universe.trading212.builder import UniverseBuilder
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.yfinance_source import (
    _CODE_TO_CONFIG_NAME,
    _CODE_TO_REGION,
    _ETF_SORT_FIELD,
    _STOCK_SORT_FIELD,
    PassThroughTickerMapper,
    YFinanceUniverseSource,
)

_BIG_CAP = 10_000


# Injected FX so tests never hit the network: 1.0 USD per unit for every major.
def _fake_fx(majors: set[str]) -> dict[str, float]:
    return {**dict.fromkeys(majors, 1.0), "USD": 1.0}


def _quote(symbol: str, exchange: str, **kw: object) -> dict:
    # Defaults clear the anti-junk floor (priced, liquid, sized) and carry a
    # financialCurrency so cross-listing dedup groups correctly.
    return {
        "symbol": symbol,
        "exchange": exchange,
        "currency": kw.get("currency", "USD"),
        "financialCurrency": kw.get("financialCurrency", "USD"),
        "longName": kw.get("longName", symbol),
        "regularMarketPrice": kw.get("regularMarketPrice", 100.0),
        "averageDailyVolume3Month": kw.get("averageDailyVolume3Month", 1_000_000),
        "sharesOutstanding": kw.get("sharesOutstanding", 1_000_000_000),
        "marketCap": kw.get("marketCap"),
    }


def _source(
    pages: list, monkeypatch: pytest.MonkeyPatch, queries: list | None = None
) -> tuple[YFinanceUniverseSource, MagicMock]:
    screener = MagicMock()
    screener.screen.side_effect = pages
    src = YFinanceUniverseSource(screener=screener, fx_resolver=_fake_fx)
    monkeypatch.setattr(
        src,
        "_build_queries",
        lambda: queries or [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)],
    )
    return src, screener


def test_emits_exchanges_and_instruments(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source(
        [{"quotes": [_quote("AAPL", "NMS"), _quote("MSFT", "NMS")]}, {"quotes": []}],
        monkeypatch,
    )
    assert src.get_exchanges() == [{"name": "NASDAQ", "workingSchedules": [{"id": 1}]}]
    insts = src.get_instruments()
    assert {i["ticker"] for i in insts} == {"AAPL", "MSFT"}
    assert all(i["isin"] is None for i in insts)
    assert all(i["workingScheduleId"] == 1 for i in insts)
    assert all(i["shortName"] == i["ticker"] for i in insts)


def test_paginates_past_250(monkeypatch: pytest.MonkeyPatch) -> None:
    page1 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    page2 = {"quotes": [_quote("EXTRA", "NMS")]}
    src, screener = _source([page1, page2, {"quotes": []}], monkeypatch)
    assert len(src.get_instruments()) == 251
    assert screener.screen.call_count >= 2


def test_dedup_by_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source(
        [{"quotes": [_quote("AAPL", "NMS"), _quote("AAPL", "NMS")]}, {"quotes": []}],
        monkeypatch,
    )
    assert len(src.get_instruments()) == 1


def test_drops_out_of_scope_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source(
        [
            {"quotes": [_quote("AAPL", "NMS"), _quote("XXX", "NOT_A_REAL_CODE")]},
            {"quotes": []},
        ],
        monkeypatch,
    )
    assert {i["ticker"] for i in src.get_instruments()} == {"AAPL"}


def test_empty_universe_is_a_failed_build_not_a_completed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, _ = _source([{"quotes": []}], monkeypatch)
    with pytest.raises(RuntimeError, match="no quotes"):
        src.get_instruments()


def test_screen_none_is_a_failed_query_not_an_empty_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, _ = _source([None], monkeypatch)
    with pytest.raises(RuntimeError, match="page failed"):
        src.get_instruments()


def test_screener_error_is_a_failed_query_not_an_empty_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ScreenerClient.screen raises once its retries are spent; the source must not
    # read that as the end of the results.
    src, _ = _source([ScreenerError("503")], monkeypatch)
    with pytest.raises(RuntimeError, match="page failed"):
        src.get_instruments()


class _DelistingRepo:
    """Universe repository fake that records which tickers get marked delisted."""

    def __init__(self, active: set[str]) -> None:
        self._active = active
        self.delisted: list[str] = []

    def save_exchange(self, exchange_data: dict) -> MagicMock:
        return MagicMock(id="ex-1")

    def save_instruments_batch(
        self, instruments_data: list, exchange_id: object
    ) -> int:
        return len(instruments_data)

    def get_active_tickers(
        self, exchange_id: object, instrument_type: str | None = None
    ) -> set[str]:
        return set(self._active)

    def mark_delisted(self, *, ticker: str, **_: object) -> bool:
        self.delisted.append(ticker)
        return True


def _build_with_second_page(
    monkeypatch: pytest.MonkeyPatch, second_page: object, repo: _DelistingRepo
) -> None:
    # Real client + source + builder; only yf.screen is stubbed, so the test
    # covers how a failed Yahoo page propagates into delisting reconciliation.
    page1 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    monkeypatch.setattr(
        "app.services.market_data.yfinance.screener.screener_client.yf.screen",
        MagicMock(side_effect=[page1, second_page]),
    )
    client = ScreenerClient(
        rate_limiter=MagicMock(), circuit_breaker=MagicMock(), default_max_retries=1
    )
    src = YFinanceUniverseSource(screener=client, fx_resolver=_fake_fx)
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    UniverseBuilder(
        config=UniverseBuilderConfig(),
        api_client=src,
        ticker_mapper=PassThroughTickerMapper(),
        repository=repo,
        max_workers=1,
    ).build()


def test_failed_page_aborts_the_build_before_any_instrument_is_delisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _DelistingRepo(active={"S0", "S249", "LIVE_PAGE_2"})
    with pytest.raises(RuntimeError, match="page failed"):
        _build_with_second_page(
            monkeypatch, RuntimeError("503 Service Unavailable"), repo
        )
    assert repo.delisted == []


def test_genuinely_absent_instrument_is_still_delisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # LIVE_PAGE_2 sat on the page that is now empty, i.e. Yahoo no longer lists it.
    repo = _DelistingRepo(active={"S0", "S249", "LIVE_PAGE_2"})
    _build_with_second_page(monkeypatch, {"quotes": []}, repo)
    assert repo.delisted == ["LIVE_PAGE_2"]


def test_tags_instrument_type(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source(
        [{"quotes": [_quote("AAPL", "NMS")]}, {"quotes": [_quote("BND", "NMS")]}],
        monkeypatch,
        queries=[
            ("QS", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP),
            ("QE", "ETF", _ETF_SORT_FIELD, _BIG_CAP),
        ],
    )
    by_ticker = {i["ticker"]: i["type"] for i in src.get_instruments()}
    assert by_ticker == {"AAPL": "STOCK", "BND": "ETF"}


def test_requires_no_trading212(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.universe.trading212.protocols import (
        TickerMapper,
        Trading212ApiClient,
    )

    src, _ = _source([{"quotes": []}], monkeypatch)
    assert isinstance(src, Trading212ApiClient)
    assert isinstance(PassThroughTickerMapper(), TickerMapper)


def test_passthrough_mapper_echoes_symbol() -> None:
    mapper = PassThroughTickerMapper()
    assert mapper.discover("AAPL", "NASDAQ") == "AAPL"
    assert mapper.discover("") is None


def test_build_queries_constructs_stocks_per_exchange_and_etfs_per_region() -> None:
    # Exercises the real yf.EquityQuery/ETFQuery construction (offline, no network)
    # so an invalid exchange code or region would surface here.
    src = YFinanceUniverseSource(screener=MagicMock())
    built = src._build_queries()
    kinds = [kind for _, kind, _, _ in built]
    assert set(kinds) == {"STOCK", "ETF"}
    # one stock query per exchange code, one ETF query per unique region
    assert kinds.count("STOCK") == len(_CODE_TO_CONFIG_NAME)
    assert kinds.count("ETF") == len(set(_CODE_TO_REGION.values()))
    stock_sorts = {sf for _, kind, sf, _ in built if kind == "STOCK"}
    etf_sorts = {sf for _, kind, sf, _ in built if kind == "ETF"}
    assert stock_sorts == {_STOCK_SORT_FIELD}
    assert etf_sorts == {_ETF_SORT_FIELD}


def test_respects_per_query_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ranked queries are bounded: paging stops at the cap even if more rows exist.
    # Distinct symbols per page so the cap (not symbol dedup) is what bounds the result.
    page1 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    page2 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250, 500)]}
    src, screener = _source(
        [page1, page2, page2],
        monkeypatch,
        queries=[("Q", "STOCK", _STOCK_SORT_FIELD, 300)],
    )
    insts = src.get_instruments()
    assert len(insts) == 300
    assert screener.screen.call_count == 2


def test_pagination_requests_a_stable_descending_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Offset paging over Yahoo's default order can dup/skip rows across pages;
    # a size-descending sortField makes paging deterministic and investable-first.
    full = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    src, screener = _source([full, {"quotes": []}], monkeypatch)
    src.get_instruments()
    assert screener.screen.call_args_list, "screen was never called"
    for call in screener.screen.call_args_list:
        assert call.kwargs.get("sort_field")  # non-None deterministic sort field
        assert call.kwargs.get("sort_asc") is False  # largest first


def test_cross_listings_collapse_to_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same longName on NMS + Xetra -> one survivor, the higher-ADDV (US) line.
    nvda = _quote(
        "NVDA",
        "NMS",
        longName="NVIDIA Corporation",
        regularMarketPrice=220.0,
        averageDailyVolume3Month=138_000_000,
    )
    nvd_de = _quote(
        "NVD.DE",
        "GER",
        longName="NVIDIA Corporation",
        currency="EUR",
        regularMarketPrice=187.0,
        averageDailyVolume3Month=100_000,
    )
    src, _ = _source([{"quotes": [nvda, nvd_de]}, {"quotes": []}], monkeypatch)
    insts = src.get_instruments()
    assert [i["ticker"] for i in insts] == ["NVDA"]


def test_below_floor_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A nano-cap shell (mcap ~$10k, ADDV ~$10) is below the anti-junk floor.
    tiny = _quote(
        "TINY",
        "NMS",
        regularMarketPrice=0.1,
        averageDailyVolume3Month=100,
        sharesOutstanding=100_000,
    )
    src, _ = _source([{"quotes": [tiny]}, {"quotes": []}], monkeypatch)
    assert src.get_instruments() == []


def test_dedup_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    nvda = _quote(
        "NVDA",
        "NMS",
        longName="NVIDIA Corporation",
        regularMarketPrice=220.0,
        averageDailyVolume3Month=138_000_000,
    )
    nvd_de = _quote(
        "NVD.DE",
        "GER",
        longName="NVIDIA Corporation",
        currency="EUR",
        regularMarketPrice=187.0,
        averageDailyVolume3Month=100_000,
    )
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": [nvda, nvd_de]}, {"quotes": []}]
    src = YFinanceUniverseSource(screener=screener, fx_resolver=_fake_fx, dedup=False)
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    assert {i["ticker"] for i in src.get_instruments()} == {"NVDA", "NVD.DE"}


def test_certificate_without_fundamentals_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A structured product: no financialCurrency / marketCap / shares / netAssets.
    cert = {
        "symbol": "AT0000A3.VI",
        "exchange": "VIE",
        "longName": "RBI Expr.Z./Nvidia 25-30",
        "currency": "EUR",
        "regularMarketPrice": 100.0,
        "averageDailyVolume3Month": 500,
    }
    real = _quote(
        "OMV", "VIE", longName="OMV AG", currency="EUR", financialCurrency="EUR"
    )
    src, _ = _source([{"quotes": [cert, real]}, {"quotes": []}], monkeypatch)
    assert {i["ticker"] for i in src.get_instruments()} == {"OMV"}


def test_etf_with_net_assets_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    # ETFs carry netAssets (no marketCap/shares) -> must clear the existence gate.
    etf = {
        "symbol": "SPY",
        "exchange": "NMS",
        "longName": "SPDR S&P 500 ETF Trust",
        "currency": "USD",
        "regularMarketPrice": 760.0,
        "averageDailyVolume3Month": 50_000_000,
        "netAssets": 795_000_000_000,
    }
    src, _ = _source(
        [{"quotes": [etf]}, {"quotes": []}],
        monkeypatch,
        queries=[("Q", "ETF", _ETF_SORT_FIELD, _BIG_CAP)],
    )
    assert [i["ticker"] for i in src.get_instruments()] == ["SPY"]


class _RecordingRepo:
    """In-memory universe repo that records the writes the builder issues."""

    def __init__(self, active: dict[str, set[str]]) -> None:
        self._active = active  # exchange name -> active tickers
        self._names: dict[str, str] = {}
        self.saved: list[str] = []
        self.delisted: list[tuple[str, float]] = []
        self.delisting_returns: dict[str, float | None] = {}

    def save_exchange(self, exchange_data: dict[str, Any]) -> Any:
        exchange = MagicMock()
        exchange.id = exchange_data["name"]
        self._names[exchange.id] = exchange_data["name"]
        return exchange

    def save_instruments_batch(
        self, rows: list[dict[str, Any]], exchange_id: Any
    ) -> int:
        self.saved.extend(r["ticker"] for r in rows)
        return len(rows)

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        return set(self._active.get(exchange_id, set()))

    def mark_delisted(
        self,
        ticker: str,
        exchange_id: Any,
        delisted_at: Any,
        delisting_return: float | None = -0.30,
    ) -> bool:
        self.delisted.append((ticker, exchange_id))
        self.delisting_returns[ticker] = delisting_return
        return True


def _build(src: YFinanceUniverseSource, repo: _RecordingRepo) -> None:
    UniverseBuilder(
        config=UniverseBuilderConfig(),
        api_client=src,
        ticker_mapper=PassThroughTickerMapper(),
        repository=repo,
        max_workers=1,
    ).build()


def test_failed_page_aborts_instead_of_truncating_the_venue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Page 1 succeeds, page 2 exhausts its retries (screen -> None). That is a
    # failed fetch, not the end of the venue's results.
    page1 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    src, _ = _source([page1, None], monkeypatch)
    with pytest.raises(RuntimeError, match="page failed"):
        src.get_instruments()


def test_failed_page_never_delists_the_unseen_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = {f"S{i}" for i in range(500)}
    repo = _RecordingRepo({"NASDAQ": active})
    page1 = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    src, _ = _source([page1, None], monkeypatch)

    with pytest.raises(RuntimeError, match="page failed"):
        _build(src, repo)

    assert repo.delisted == []
    assert repo.saved == []


def test_failed_code_inside_a_merged_exchange_never_delists_its_listings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NMS and NGM both feed the NASDAQ exchange: when only the NGM query fails,
    # NASDAQ is still populated by NMS, so a tolerated failure would delist every
    # active NGM ticker.
    active = {"AAA", "BBB", "NGM1", "NGM2"}
    repo = _RecordingRepo({"NASDAQ": active})
    src, _ = _source(
        [{"quotes": [_quote("AAA", "NMS"), _quote("BBB", "NMS")]}, None],
        monkeypatch,
        queries=[
            ("NMS", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP),
            ("NGM", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP),
        ],
    )

    with pytest.raises(RuntimeError, match="page failed"):
        _build(src, repo)

    assert repo.delisted == []


def test_live_line_dropped_by_the_floor_gets_no_fabricated_delisting_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TINY still trades (the screener returned it) but sits below the floor, so it
    # leaves the universe. That is not a delisting: the CRSP -30% terminal return
    # would be fabricated. GONE is absent from the screener altogether.
    tiny = _quote(
        "TINY",
        "NMS",
        regularMarketPrice=0.1,
        averageDailyVolume3Month=100,
        sharesOutstanding=100_000,
    )
    repo = _RecordingRepo({"NASDAQ": {"AAA", "TINY", "GONE"}})
    src, _ = _source(
        [{"quotes": [_quote("AAA", "NMS"), tiny]}, {"quotes": []}], monkeypatch
    )

    _build(src, repo)

    assert repo.delisting_returns == {"TINY": None, "GONE": -0.30}


def test_live_line_dropped_by_dedup_gets_no_fabricated_delisting_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvda = _quote(
        "NVDA",
        "NMS",
        longName="NVIDIA Corporation",
        regularMarketPrice=220.0,
        averageDailyVolume3Month=138_000_000,
    )
    nvd_de = _quote(
        "NVD.DE",
        "GER",
        longName="NVIDIA Corporation",
        currency="EUR",
        regularMarketPrice=187.0,
        averageDailyVolume3Month=100_000,
    )
    sap = _quote("SAP.DE", "GER", longName="SAP SE", currency="EUR")
    repo = _RecordingRepo({"Deutsche Börse Xetra": {"NVD.DE", "SAP.DE"}})
    src, _ = _source([{"quotes": [nvda, nvd_de, sap]}, {"quotes": []}], monkeypatch)

    _build(src, repo)

    assert repo.delisting_returns == {"NVD.DE": None}


def test_live_line_moved_to_an_out_of_scope_venue_gets_no_fabricated_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NYSE Arca -> Cboe BZX ("BTS" is not a mapped code): the symbol is still
    # trading, it just left the covered venues.
    moved = _quote("MOVED", "BTS")
    repo = _RecordingRepo({"NYSE": {"MOVED", "KEEP"}})
    src, _ = _source(
        [{"quotes": [_quote("KEEP", "NYQ"), moved]}, {"quotes": []}], monkeypatch
    )

    _build(src, repo)

    assert repo.delisting_returns == {"MOVED": None}


class _FxTicker:
    """Stand-in for ``yf.Ticker('{MAJOR}USD=X')`` replaying scripted outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def history(self, period: str) -> pd.DataFrame:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return pd.DataFrame({"Close": outcome})


def _patch_fx_tickers(
    monkeypatch: pytest.MonkeyPatch, scripted: dict[str, list[object]]
) -> dict[str, _FxTicker]:
    tickers = {f"{m}USD=X": _FxTicker(o) for m, o in scripted.items()}
    monkeypatch.setattr(yfinance_source.yf, "Ticker", lambda symbol: tickers[symbol])
    monkeypatch.setattr("app.services.infrastructure.retry.time.sleep", lambda _s: None)
    return tickers


def test_fx_spot_is_retried_after_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = _patch_fx_tickers(
        monkeypatch,
        {"EUR": [RuntimeError("Too Many Requests. Rate limited."), [1.05, 1.1]]},
    )
    rates = yfinance_source._live_usd_per_major({"EUR", "USD"})
    assert rates == {"USD": 1.0, "EUR": 1.1}
    assert tickers["EURUSD=X"].calls == 2


def test_fx_spot_goes_through_the_shared_breaker_and_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fx_tickers(monkeypatch, {"CHF": [[1.2]]})
    nesn = _quote("NESN.SW", "EBS", currency="CHF", financialCurrency="CHF")
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": [nesn]}, {"quotes": []}]
    src = YFinanceUniverseSource(screener=screener)
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    assert [i["ticker"] for i in src.get_instruments()] == ["NESN.SW"]
    screener.circuit_breaker.check.assert_called()
    screener.rate_limiter.acquire.assert_called()


def test_unresolved_fx_aborts_instead_of_ranking_with_a_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CHF has no rate: NESN.SW would get ADDV=None, lose every comparison to the
    # thin EUR cross-listing and make it the canonical line.
    nesn = _quote(
        "NESN.SW",
        "EBS",
        longName="Nestle S.A.",
        currency="CHF",
        financialCurrency="CHF",
        averageDailyVolume3Month=3_000_000,
    )
    nsrgy = _quote(
        "NSRGY.DE",
        "GER",
        longName="Nestle S.A.",
        currency="EUR",
        financialCurrency="CHF",
        averageDailyVolume3Month=2_000,
    )
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": [nesn, nsrgy]}, {"quotes": []}]
    src = YFinanceUniverseSource(
        screener=screener, fx_resolver=lambda majors: {"USD": 1.0, "EUR": 1.1}
    )
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    with pytest.raises(RuntimeError, match="CHF"):
        src.get_instruments()


def _hysteresis_source(
    monkeypatch: pytest.MonkeyPatch,
    quote: dict,
    members: set[tuple[str, str]],
) -> YFinanceUniverseSource:
    screener = MagicMock()
    screener.screen.side_effect = [{"quotes": [quote]}, {"quotes": []}]
    src = YFinanceUniverseSource(
        screener=screener, fx_resolver=_fake_fx, member_lookup=lambda: members
    )
    monkeypatch.setattr(
        src, "_build_queries", lambda: [("Q", "STOCK", _STOCK_SORT_FIELD, _BIG_CAP)]
    )
    return src


# $15.2M market cap: above the $10M exit band, below the $25M entry band.
_MID_CAP = {"regularMarketPrice": 15.2, "sharesOutstanding": 1_000_000}


def test_existing_member_is_retained_down_to_the_exit_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _hysteresis_source(
        monkeypatch, _quote("MID", "NMS", **_MID_CAP), {("MID", "NASDAQ")}
    )
    assert [i["ticker"] for i in src.get_instruments()] == ["MID"]


def test_new_entrant_must_clear_the_entry_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _hysteresis_source(monkeypatch, _quote("MID", "NMS", **_MID_CAP), set())
    assert src.get_instruments() == []


def test_membership_is_per_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same symbol held on a different exchange does not buy retention here.
    src = _hysteresis_source(
        monkeypatch, _quote("MID", "NMS", **_MID_CAP), {("MID", "NYSE")}
    )
    assert src.get_instruments() == []


def test_member_below_the_exit_band_is_still_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tiny = _quote("TINY", "NMS", regularMarketPrice=5.0, sharesOutstanding=1_000_000)
    src = _hysteresis_source(monkeypatch, tiny, {("TINY", "NASDAQ")})
    assert src.get_instruments() == []


def test_cross_listed_currency_without_a_rate_fails_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A silently omitted CHF rate would rank the Swiss primary below a thin EUR
    # line and the builder would then mark the real primary delisted.
    nesn = _quote(
        "NESN.SW",
        "EBS",
        longName="Nestle S.A.",
        currency="CHF",
        financialCurrency="CHF",
        averageDailyVolume3Month=3_000_000,
    )
    thin = _quote(
        "NSRGF.DE",
        "GER",
        longName="Nestle S.A.",
        currency="EUR",
        financialCurrency="CHF",
        averageDailyVolume3Month=2_000,
    )
    src, _ = _source([{"quotes": [nesn, thin]}, {"quotes": []}], monkeypatch)
    src.fx_resolver = lambda majors: {"USD": 1.0, "EUR": 1.1}
    with pytest.raises(RuntimeError, match="CHF"):
        src.get_instruments()


def test_live_fx_retries_a_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FlakyTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str) -> pd.DataFrame:
            calls.append(self.symbol)
            if len(calls) == 1:
                raise RuntimeError("Too Many Requests")
            return pd.DataFrame({"Close": [1.05, 1.1]})

    monkeypatch.setattr("app.services.infrastructure.retry.time.sleep", lambda _s: None)
    monkeypatch.setattr("app.services.universe.yfinance_source.yf.Ticker", _FlakyTicker)
    assert yfinance_source._live_usd_per_major({"EUR"}) == {"USD": 1.0, "EUR": 1.1}
    assert calls == ["EURUSD=X", "EURUSD=X"]


def _azn_pair() -> list[dict]:
    # AZN.L is ~1.3x more liquid than AZN: a near-tie that flips without hysteresis.
    azn = _quote(
        "AZN",
        "NMS",
        longName="AstraZeneca PLC",
        regularMarketPrice=100.0,
        averageDailyVolume3Month=1_000_000,
    )
    azn_l = _quote(
        "AZN.L",
        "LSE",
        longName="AstraZeneca PLC",
        currency="GBp",
        regularMarketPrice=8_000.0,
        averageDailyVolume3Month=1_300_000,
    )
    return [azn, azn_l]


def test_incumbent_listing_survives_a_near_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source([{"quotes": _azn_pair()}, {"quotes": []}], monkeypatch)
    src.member_lookup = lambda: {("AZN", "NASDAQ")}
    assert [i["ticker"] for i in src.get_instruments()] == ["AZN"]


def _foo_pair(**incumbent_overrides: object) -> list[dict]:
    # FOO (the incumbent) trades 40k USD/day: under the 50k entry bar, over the 25k
    # member exit bar. FOO.DE trades 55k, inside the 1.5x hysteresis margin.
    foo = _quote(
        "FOO",
        "NMS",
        longName="Foo Corp",
        regularMarketPrice=10.0,
        averageDailyVolume3Month=4_000,
        **incumbent_overrides,
    )
    foo_de = _quote(
        "FOO.DE",
        "GER",
        longName="Foo Corp",
        currency="EUR",
        regularMarketPrice=10.0,
        averageDailyVolume3Month=5_500,
    )
    return [foo, foo_de]


def test_incumbent_between_exit_and_entry_floor_stays_in_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Holding FOO against FOO.DE must not hand it to a floor it fails: members are
    # judged on the exit band, so the entity stays in the build.
    src, _ = _source([{"quotes": _foo_pair()}, {"quotes": []}], monkeypatch)
    src.member_lookup = lambda: {("FOO", "NASDAQ")}
    assert [i["ticker"] for i in src.get_instruments()] == ["FOO"]


def test_non_member_between_exit_and_entry_floor_is_not_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lone = _quote(
        "FOO",
        "NMS",
        longName="Foo Corp",
        regularMarketPrice=10.0,
        averageDailyVolume3Month=4_000,
    )
    src, _ = _source([{"quotes": [lone]}, {"quotes": []}], monkeypatch)
    assert src.get_instruments() == []
    src, _ = _source([{"quotes": [lone]}, {"quotes": []}], monkeypatch)
    src.member_lookup = lambda: {("FOO", "NASDAQ")}
    assert [i["ticker"] for i in src.get_instruments()] == ["FOO"]


def test_challenger_carries_the_entity_when_the_incumbent_fails_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # FOO's market cap (5M) is below even the member exit band: holding it would
    # leave the entity out of the build although FOO.DE clears the floor.
    src, _ = _source(
        [{"quotes": _foo_pair(sharesOutstanding=500_000)}, {"quotes": []}],
        monkeypatch,
    )
    src.member_lookup = lambda: {("FOO", "NASDAQ")}
    assert [i["ticker"] for i in src.get_instruments()] == ["FOO.DE"]


def _etf_quote(symbol: str, exchange: str, **kw: object) -> dict:
    quote = _quote(symbol, exchange, financialCurrency=None, sharesOutstanding=None)
    quote.update(
        {
            "longName": "iShares Core Global Aggregate Bond",
            "currency": kw["currency"],
            "regularMarketPrice": kw["price"],
            "averageDailyVolume3Month": kw["volume"],
        }
    )
    if "netAssets" in kw:
        quote["netAssets"] = kw["netAssets"]
    return quote


def test_sibling_with_fundamentals_survives_a_more_liquid_line_without_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EUNA.DE wins on ADDV but has no netAssets, so the existence gate would reject
    # the whole fund even though AGGG.L (same fund) clears it.
    euna = _etf_quote("EUNA.DE", "GER", currency="EUR", price=5.0, volume=3_000_000)
    aggg = _etf_quote(
        "AGGG.L", "LSE", currency="GBp", price=500.0, volume=100_000, netAssets=4e9
    )
    src, _ = _source(
        [{"quotes": [euna, aggg]}, {"quotes": []}],
        monkeypatch,
        queries=[("Q", "ETF", _ETF_SORT_FIELD, _BIG_CAP)],
    )
    assert [i["ticker"] for i in src.get_instruments()] == ["AGGG.L"]
    # EUNA.DE is still trading, so the builder must not book a -30% delisting for
    # a one-off netAssets gap.
    assert "EUNA.DE" in src.get_live_symbols()


def test_existence_failing_line_without_a_sibling_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    euna = _etf_quote("EUNA.DE", "GER", currency="EUR", price=5.0, volume=3_000_000)
    other = _quote("OTHER", "NMS", longName="Other Corp")
    src, _ = _source(
        [{"quotes": [euna, other]}, {"quotes": []}],
        monkeypatch,
        queries=[("Q", "ETF", _ETF_SORT_FIELD, _BIG_CAP)],
    )
    assert [i["ticker"] for i in src.get_instruments()] == ["OTHER"]


def test_real_security_without_fundamentals_fields_is_kept_on_shares_and_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The line has shares and a price but no financialCurrency or netAssets, so the
    # existence gate must rest on those local-currency quantities, not on a
    # market cap that needs an FX rate.
    seqi = _quote(
        "SEQI.L",
        "LSE",
        longName="Sequoia Economic Infrastructure Income Fund",
        currency="GBp",
        financialCurrency=None,
        regularMarketPrice=70.0,
        sharesOutstanding=1_500_000_000,
    )
    src, _ = _source([{"quotes": [seqi]}, {"quotes": []}], monkeypatch)
    assert [i["ticker"] for i in src.get_instruments()] == ["SEQI.L"]
