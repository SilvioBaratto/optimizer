"""The yfinance sweep's category registries agree with each other and with the guard.

Validates ``app/services/market_data/yfinance_data_service.py`` (the spec
registries, ``_EVENT_MARKER_CATEGORIES`` and ``_MARKERS_BEFORE_REQUESTS``) and
``app/services/market_data/yfinance/categories.py`` (``_refresh``). No
requirement id is registered for them yet.

A category's marker is written down in up to three places: its spec, the table
that remembers empty event markers, and the markers the incremental short circuit
checks. A category added to one and missed in another is either refetched for
ever or hidden as fresh, so these tests hold them together. The last test keeps
every repository write of a spec inside its savepoint: the AST guard on the
service sees ``upsert_*`` calls only, while a spec's store names its write.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

from portopt_db.repositories.market_data.yfinance_repository import (
    FINANCIAL_STATEMENT_GROUPS,
    FUNDAMENTAL_CATEGORIES,
    statement_staleness_key,
)

import app
from app.services.market_data.yfinance.categories import CategorySpec
from app.services.market_data.yfinance.fetch_plan import (
    FetchContext,
    FetchMode,
    StalenessThresholds,
)
from app.services.market_data.yfinance_data_service import (
    _EVENT_MARKER_CATEGORIES,
    _MARKERS_BEFORE_REQUESTS,
    YFinanceDataService,
)

_APP_ROOT = Path(app.__file__).resolve().parent
_CATEGORIES_MODULE = (
    _APP_ROOT / "services" / "market_data" / "yfinance" / "categories.py"
)


def _service() -> YFinanceDataService:
    return YFinanceDataService(MagicMock(), MagicMock(), news_scrape_delay=0.0)


def _ctx() -> FetchContext:
    return FetchContext(
        instrument_id=uuid4(),
        yfinance_ticker="EXMPL",
        period="5y",
        mode=FetchMode.FULL,
        exchange_name=None,
        currency_code="USD",
        asset_class="equity",
        instrument_type="STOCK",
        now=datetime(2026, 11, 13, tzinfo=timezone.utc),
        thresholds=StalenessThresholds(),
    )


def _fundamental_specs() -> tuple[CategorySpec[Any], ...]:
    return _service()._fundamental_specs(_ctx(), "USD")


def _event_specs() -> tuple[CategorySpec[Any], ...]:
    return _service()._event_specs()


def _statement_specs() -> tuple[CategorySpec[Any], ...]:
    return _service()._statement_specs(lambda: None)


def test_every_category_on_an_event_marker_is_listed_under_it() -> None:
    """An event marker is remembered as empty only from the categories listed."""
    on_event_markers = {
        (spec.marker, spec.name)
        for spec in (*_fundamental_specs(), *_event_specs())
        if spec.marker in _EVENT_MARKER_CATEGORIES
    }
    listed = {
        (marker, category)
        for marker, categories in _EVENT_MARKER_CATEGORIES.items()
        for category in categories
    }

    assert on_event_markers == listed


def test_every_fundamentals_category_is_gated_on_its_own_marker() -> None:
    """The fundamentals bookkeeping remembers ``<category>_updated_at`` per category."""
    own_marker = {
        spec.name: spec.marker
        for spec in _fundamental_specs()
        if spec.marker not in _EVENT_MARKER_CATEGORIES
    }

    assert own_marker == {
        category: f"{category}_updated_at" for category in FUNDAMENTAL_CATEGORIES
    }


def test_every_statement_group_is_gated_on_its_own_marker() -> None:
    """Each group's marker is the one the statements step remembers it under."""
    markers = [spec.marker for spec in _statement_specs()]

    assert markers == [
        statement_staleness_key(*group) for group in FINANCIAL_STATEMENT_GROUPS
    ]


def test_the_short_circuit_checks_the_marker_of_every_category() -> None:
    """A marker it skipped would let a due category pass for fresh."""
    gated = {
        spec.marker
        for spec in (*_statement_specs(), *_fundamental_specs(), *_event_specs())
    }

    assert set(_MARKERS_BEFORE_REQUESTS) == gated
    assert len(_MARKERS_BEFORE_REQUESTS) == len(gated)


def _store_reads(tree: ast.AST) -> list[ast.Attribute]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "store"
    ]


def _handed_to_the_savepoint(tree: ast.AST) -> set[int]:
    """Ids of the arguments ``_refresh`` passes to ``write_in_savepoint``."""
    return {
        id(argument)
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and function.name == "_refresh"
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "write_in_savepoint"
        for argument in call.args
    }


def test_a_category_store_runs_only_inside_its_savepoint() -> None:
    """Only ``_refresh`` reaches a spec's store, and only through the savepoint.

    A store wraps a repository write. Called anywhere else, a failed write would
    abort the ticker's shared transaction and take its other categories with it.
    """
    categories_tree = ast.parse(_CATEGORIES_MODULE.read_text(encoding="utf-8"))
    elsewhere = [
        f"{path.relative_to(_APP_ROOT)}:{node.lineno}"
        for path in sorted(_APP_ROOT.rglob("*.py"))
        if path != _CATEGORIES_MODULE
        for node in _store_reads(ast.parse(path.read_text(encoding="utf-8")))
    ]
    inside = _store_reads(categories_tree)
    savepoint_arguments = _handed_to_the_savepoint(categories_tree)

    assert elsewhere == []
    assert inside, "the guard no longer finds the read of CategorySpec.store"
    assert all(id(node) in savepoint_arguments for node in inside)
