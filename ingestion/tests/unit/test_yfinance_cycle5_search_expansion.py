"""Tests for SearchClient.search: exposes news_count and extras flags.

Forward ``news_count`` to ``yf.Search`` constructor and conditionally include
``lists``, ``research``, ``nav`` keys in the returned dict via opt-in
``include_lists`` / ``include_research`` / ``include_nav`` flags.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("yfinance")

from yfinance.data import YfData

from app.services.market_data.yfinance.market.search import SearchClient
from app.services.market_data.yfinance.protocols import SearchClientProtocol


def _build_client() -> SearchClient:
    rate_limiter = MagicMock(name="rate_limiter")
    circuit_breaker = MagicMock(name="circuit_breaker")
    circuit_breaker.check.return_value = None
    return SearchClient(
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        default_max_retries=1,
    )


def _patch_search(
    *,
    quotes: list[dict[str, Any]] | None = None,
    news: list[dict[str, Any]] | None = None,
    lists: Any = None,
    research: Any = None,
    nav: Any = None,
    drop: tuple[str, ...] = (),
):
    quotes = quotes if quotes is not None else [{"symbol": "AAPL"}]
    news = news if news is not None else [{"title": "Apple"}]
    instance = MagicMock(name="yf.Search.instance")
    instance.quotes = quotes
    instance.news = news
    instance.lists = lists if lists is not None else []
    instance.research = research if research is not None else []
    instance.nav = nav if nav is not None else []
    for attr in drop:
        delattr(instance, attr)
    return patch(
        "app.services.market_data.yfinance.market.search.yf.Search",
        return_value=instance,
    )


def test_when_search_called_with_defaults_then_news_count_5_forwarded() -> None:
    client = _build_client()
    with _patch_search() as mock_cls:
        client.search("Apple")
    assert mock_cls.call_args.kwargs.get("news_count") == 5


def test_when_news_count_set_then_forwarded_to_yf_search_constructor() -> None:
    client = _build_client()
    with _patch_search() as mock_cls:
        client.search("Apple", news_count=10)
    assert mock_cls.call_args.kwargs.get("news_count") == 10


def test_when_no_flags_then_only_quotes_and_news_keys() -> None:
    client = _build_client()
    with _patch_search():
        result = client.search("Apple")
    assert set(result) == {"quotes", "news"}


def test_when_include_lists_true_then_lists_key_present() -> None:
    client = _build_client()
    with _patch_search(lists=[{"id": "L1"}]):
        result = client.search("Apple", include_lists=True)
    assert "lists" in result
    assert result["lists"] == [{"id": "L1"}]
    assert "research" not in result
    assert "nav" not in result


def test_when_include_research_true_then_research_key_present() -> None:
    client = _build_client()
    with _patch_search(research=[{"id": "R1"}]):
        result = client.search("Apple", include_research=True)
    assert "research" in result
    assert result["research"] == [{"id": "R1"}]
    assert "lists" not in result
    assert "nav" not in result


def test_when_include_nav_true_then_nav_key_present() -> None:
    client = _build_client()
    with _patch_search(nav=[{"id": "N1"}]):
        result = client.search("Apple", include_nav=True)
    assert "nav" in result
    assert result["nav"] == [{"id": "N1"}]
    assert "lists" not in result
    assert "research" not in result


def test_when_all_flags_true_then_all_keys_present() -> None:
    client = _build_client()
    with _patch_search():
        result = client.search(
            "Apple",
            include_lists=True,
            include_research=True,
            include_nav=True,
        )
    assert set(result) == {"quotes", "news", "lists", "research", "nav"}


def test_when_lists_attribute_missing_then_value_is_none() -> None:
    client = _build_client()
    with _patch_search(drop=("lists",)):
        result = client.search("Apple", include_lists=True)
    assert result["lists"] is None


def test_when_research_attribute_missing_then_value_is_none() -> None:
    client = _build_client()
    with _patch_search(drop=("research",)):
        result = client.search("Apple", include_research=True)
    assert result["research"] is None


def test_when_nav_attribute_missing_then_value_is_none() -> None:
    client = _build_client()
    with _patch_search(drop=("nav",)):
        result = client.search("Apple", include_nav=True)
    assert result["nav"] is None


def test_when_max_results_passed_then_forwarded_to_constructor() -> None:
    client = _build_client()
    with _patch_search() as mock_cls:
        client.search("Apple", max_results=12)
    assert mock_cls.call_args.kwargs.get("max_results") == 12


def test_when_search_client_built_then_protocol_is_satisfied() -> None:
    client = _build_client()
    assert isinstance(client, SearchClientProtocol)


def test_when_inspecting_search_signature_then_matches_protocol() -> None:
    impl = inspect.signature(SearchClient.search)
    proto = inspect.signature(SearchClientProtocol.search)
    assert impl.parameters.keys() == proto.parameters.keys()
    for name, impl_param in impl.parameters.items():
        assert impl_param.default == proto.parameters[name].default, name


def test_when_lookup_called_then_signature_unchanged() -> None:
    sig = inspect.signature(SearchClient.lookup)
    params = list(sig.parameters.keys())
    assert params == ["self", "query", "asset_type", "count", "max_retries"]
    assert sig.parameters["asset_type"].default == "stock"
    assert sig.parameters["count"].default == 25


class _Resp:
    """Minimal stand-in for the HTTP response yfinance's ``YfData.get`` returns."""

    def __init__(self, payload: dict[str, Any] | None = None, text: str = "") -> None:
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def _serve(*responses: _Resp):
    """Patch yfinance's HTTP layer to return *responses* in order, then repeat the last."""
    queue = list(responses)

    def fake_get(url: str, params: Any = None, timeout: float = 30) -> _Resp:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return patch.object(YfData, "get", side_effect=fake_get)


def _news(title: str) -> _Resp:
    return _Resp({"quotes": [], "news": [{"title": title}]})


@pytest.fixture
def _real_search_http(monkeypatch: pytest.MonkeyPatch):
    """Run the real ``yf.Search`` against a clean yfinance response cache."""
    monkeypatch.setattr("app.services.infrastructure.retry.time.sleep", lambda _: None)
    YfData.cache_get.cache_clear()
    yield
    YfData.cache_get.cache_clear()


@pytest.mark.usefixtures("_real_search_http")
def test_when_first_response_is_a_down_page_then_retry_hits_the_network_again() -> None:
    # yfinance caches the Response object itself, so a cached "Will be right
    # back" page used to be replayed to every retry of the same query.
    client = SearchClient(MagicMock(), MagicMock(), default_max_retries=3)
    down = _Resp(text="<html>Will be right back</html>")
    with _serve(down, _news("fresh")) as get:
        result = client.search("Federal Reserve")

    assert result is not None
    assert [n["title"] for n in result["news"]] == ["fresh"]
    assert get.call_count == 2


@pytest.mark.usefixtures("_real_search_http")
def test_when_same_query_repeats_then_second_call_returns_fresh_news() -> None:
    # The 07:00 and 14:00 news jobs run the same queries in one daemon process.
    client = SearchClient(MagicMock(), MagicMock(), default_max_retries=1)
    with _serve(_news("morning"), _news("midday")):
        first = client.search("Federal Reserve")
        second = client.search("Federal Reserve")

    assert first is not None and second is not None
    assert first["news"][0]["title"] == "morning"
    assert second["news"][0]["title"] == "midday"
