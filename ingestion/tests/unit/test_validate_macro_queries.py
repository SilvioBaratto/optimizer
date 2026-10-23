"""Behaviour of ``scripts/validate_macro_queries.py``.

The validator is the gate for ``MACRO_SEARCH_QUERIES``, so it has to see exactly
what the production fetcher sees. ``yf.Search`` is replaced by a fake: no
network, database or API key is touched.
"""

from __future__ import annotations

import importlib.util
import inspect
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
