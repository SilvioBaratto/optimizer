"""Behaviour of ``scripts/validate_macro_queries.py``.

The validator is the gate for ``MACRO_SEARCH_QUERIES``, so it has to see exactly
what the production fetcher sees. ``yf.Search`` is replaced by a fake: no
network, database or API key is touched.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.market_data.yfinance.market.search import SearchClient

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_macro_queries.py"

_TRUSTED = "Reuters"
_UNTRUSTED = "Simply Wall St."


def _load_script() -> ModuleType:
    # Loaded by path: the repository root also carries a ``scripts`` directory,
    # so importing ``scripts.validate_macro_queries`` is ambiguous.
    spec = importlib.util.spec_from_file_location(
        "validate_macro_queries_script", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """The script module, loaded once for the whole file."""
    return _load_script()


def _article(publisher: str, *, age_days: int = 1) -> dict:
    published = datetime.now() - timedelta(days=age_days)
    return {
        "content": {
            "title": f"{publisher} {age_days}d",
            "pubDate": published.isoformat(),
            "provider": {"displayName": publisher},
        }
    }


@pytest.fixture
def search(script, monkeypatch) -> MagicMock:
    """The ``yf.Search`` constructor the validator calls; ``.news`` is settable."""
    fake = MagicMock()
    fake.return_value.news = []
    monkeypatch.setattr(script, "yf", SimpleNamespace(Search=fake))
    return fake


def _cutoff() -> datetime:
    return datetime.now() - timedelta(days=60)


def test_evaluate_query_asks_for_the_news_count_production_asks_for(script, search):
    """A larger window would show articles the production fetcher never receives."""
    production_news_count = (
        inspect.signature(SearchClient.search).parameters["news_count"].default
    )

    script.evaluate_query("Federal Reserve interest rate decision", _cutoff())

    search.assert_called_once_with(
        "Federal Reserve interest rate decision",
        max_results=8,
        news_count=production_news_count,
    )


def test_evaluate_query_counts_only_recent_articles_from_trusted_publishers(
    script, search
):
    """Production drops untrusted publishers, so they cannot make a query pass."""
    search.return_value.news = [
        _article(_UNTRUSTED, age_days=1),
        _article(_UNTRUSTED, age_days=2),
        _article(_TRUSTED, age_days=3),
        _article(_TRUSTED, age_days=90),
    ]

    result = script.evaluate_query("high yield bond market", _cutoff())

    assert result["total"] == 4
    assert result["recent"] == 1
    assert result["publishers"] == {_TRUSTED, _UNTRUSTED}


def test_main_fails_a_query_whose_recent_articles_are_all_untrusted(
    script, search, monkeypatch, capsys
):
    """The coverage regression the gate exists to catch must not exit 0."""
    search.return_value.news = [_article(_UNTRUSTED), _article(_UNTRUSTED)]
    monkeypatch.setattr(script, "MACRO_SEARCH_QUERIES", {"Britain inflation": []})
    monkeypatch.setattr(sys, "argv", ["validate_macro_queries"])

    exit_code = script.main()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL" in out
    assert "PASS" not in out


def test_main_passes_a_query_with_a_recent_trusted_article(
    script, search, monkeypatch, capsys
):
    """A trusted article inside the recency window is what a query needs."""
    search.return_value.news = [_article(_UNTRUSTED), _article(_TRUSTED)]
    monkeypatch.setattr(script, "MACRO_SEARCH_QUERIES", {"Britain inflation": []})
    monkeypatch.setattr(sys, "argv", ["validate_macro_queries"])

    exit_code = script.main()

    assert exit_code == 0
    assert "PASS" in capsys.readouterr().out
