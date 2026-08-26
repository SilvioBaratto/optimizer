"""YFinanceUniverseSource contract (SPEC D1/D9/D14, task T9).

The source implements the ``Trading212ApiClient`` seam (get_exchanges /
get_instruments) from ``yf.screen`` results: no seed lists, no ISIN (identity is
(ticker, exchange); dedup by symbol), out-of-scope venues dropped. Query
construction is patched so tests exercise the transform, not live Yahoo.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("yfinance")

from app.services.universe.yfinance_source import (
    PassThroughTickerMapper,
    YFinanceUniverseSource,
)


def _quote(symbol: str, exchange: str, **kw: object) -> dict:
    return {
        "symbol": symbol,
        "exchange": exchange,
        "currency": kw.get("currency", "USD"),
        "longName": kw.get("longName", symbol),
    }


def _source(
    pages: list, monkeypatch: pytest.MonkeyPatch, queries: list | None = None
) -> tuple[YFinanceUniverseSource, MagicMock]:
    screener = MagicMock()
    screener.screen.side_effect = pages
    src = YFinanceUniverseSource(screener=screener)
    monkeypatch.setattr(src, "_build_queries", lambda: queries or [("Q", "STOCK")])
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
        queries=[("QS", "STOCK"), ("QE", "ETF")],
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


def test_build_queries_constructs_equity_and_etf() -> None:
    # Exercises the real yf.EquityQuery/ETFQuery construction (offline, no network)
    # so an invalid exchange code would surface here.
    src = YFinanceUniverseSource(screener=MagicMock())
    assert [kind for _, kind in src._build_queries()] == ["STOCK", "ETF"]


def test_respects_max_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    full = {"quotes": [_quote(f"S{i}", "NMS") for i in range(250)]}
    src, screener = _source([full, full, full], monkeypatch)
    src.max_pages = 2
    src.get_instruments()
    assert screener.screen.call_count == 2
