"""Tests for issue #604: SearchClient.search exposes news_count + extras flags.

Forward ``news_count`` to ``yf.Search`` constructor and conditionally include
``lists``, ``research``, ``nav`` keys in the returned dict via opt-in
``include_lists`` / ``include_research`` / ``include_nav`` flags.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("yfinance")

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
