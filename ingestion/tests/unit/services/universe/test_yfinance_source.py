"""YFinanceUniverseSource contract (SPEC D1/D9/D14, task T9).

The source implements the ``Trading212ApiClient`` seam (get_exchanges /
get_instruments) from ``yf.screen`` results: no seed lists, no ISIN (identity is
(ticker, exchange); dedup by symbol), out-of-scope venues dropped. Query
construction is patched so tests exercise the transform, not live Yahoo.

Ranking contract: stocks are screened per exchange code by market cap descending,
ETFs per region by net assets descending, each capped — so paging returns the
investable head of every venue, not the alphabetical microcap tail.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

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


def test_empty_result_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source([{"quotes": []}], monkeypatch)
    assert src.get_instruments() == []
    assert src.get_exchanges() == []


def test_screen_none_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    src, _ = _source([None], monkeypatch)
    assert src.get_instruments() == []


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
    real = _quote("OMV", "VIE", longName="OMV AG", currency="EUR", financialCurrency="EUR")
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
