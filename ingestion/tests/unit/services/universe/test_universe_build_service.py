"""Unit tests for ``services/universe/universe_build_service``.

The Trading 212 client, the ticker mapper, and the builder are all patched at
the import site — no network, no DB. What is asserted is the wiring the
scheduler depends on: filters applied unless skipped, progress forwarded as
kwargs, the session committed, and a missing API key raising rather than
building an unauthenticated client.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.universe.trading212 import UniverseBuildRequest
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.universe_build_service import (
    Trading212NotConfiguredError,
    _build_pipelines,
    build_trading212_client,
    run_universe_build,
)

M = "app.services.universe.universe_build_service"


def _make_result(**overrides):
    result = MagicMock()
    result.exchanges_saved = overrides.get("exchanges_saved", 2)
    result.instruments_saved = overrides.get("instruments_saved", 10)
    result.total_processed = overrides.get("total_processed", 50)
    result.filter_stats = overrides.get("filter_stats", {"market_cap": {"passed": 10}})
    result.errors = overrides.get("errors", [])
    return result


@contextmanager
def _patched(builder_cls: MagicMock, session: MagicMock):
    @contextmanager
    def _session_cm():
        yield session

    dm = MagicMock()
    dm.get_session = _session_cm

    with (
        patch(f"{M}.database_manager", dm),
        patch(f"{M}.UniverseBuilder", builder_cls),
        patch(f"{M}.UniverseRepository"),
        patch(f"{M}.YFinanceTickerMapper"),
        patch(f"{M}.TickerMappingCache"),
    ):
        yield


class TestBuildTrading212Client:
    def test_when_api_key_missing_then_raises(self) -> None:
        with patch("app.config.settings.trading_212_api_key", ""):
            with pytest.raises(
                Trading212NotConfiguredError, match="TRADING_212_API_KEY"
            ):
                build_trading212_client()

    def test_when_api_key_present_then_client_constructed_from_settings(self) -> None:
        with (
            patch("app.config.settings.trading_212_api_key", "key-1"),
            patch("app.config.settings.trading_212_secret_key", "secret-1"),
            patch("app.config.settings.trading_212_mode", "demo"),
            patch(f"{M}.Trading212Client") as client_cls,
        ):
            build_trading212_client()

        client_cls.assert_called_once_with(
            api_key="key-1", api_secret="secret-1", mode="demo"
        )


class TestRunUniverseBuild:
    def test_returns_builder_result_as_summary(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()
        session = MagicMock()

        with _patched(builder_cls, session):
            summary = run_universe_build(UniverseBuildRequest(), MagicMock())

        assert summary == {
            "exchanges_saved": 2,
            "instruments_saved": 10,
            "total_processed": 50,
            "filter_stats": {"market_cap": {"passed": 10}},
            "errors": [],
        }

    def test_commits_the_session(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()
        session = MagicMock()

        with _patched(builder_cls, session):
            run_universe_build(UniverseBuildRequest(), MagicMock())

        session.commit.assert_called_once()

    def test_when_skip_filters_then_no_filters_registered(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()

        with _patched(builder_cls, MagicMock()):
            run_universe_build(UniverseBuildRequest(skip_filters=True), MagicMock())

        pipeline = builder_cls.call_args.kwargs["filter_pipeline"]
        assert pipeline._filters == []

    def test_when_filters_enabled_then_five_filters_registered(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()

        with _patched(builder_cls, MagicMock()):
            run_universe_build(UniverseBuildRequest(skip_filters=False), MagicMock())

        pipeline = builder_cls.call_args.kwargs["filter_pipeline"]
        assert len(pipeline._filters) == 5

    def test_forwards_request_options_to_builder(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()

        with _patched(builder_cls, MagicMock()):
            run_universe_build(
                UniverseBuildRequest(exchanges=["NYSE"], max_workers=7), MagicMock()
            )

        kwargs = builder_cls.call_args.kwargs
        assert kwargs["only_exchanges"] == ["NYSE"]
        assert kwargs["max_workers"] == 7

    def test_builder_progress_is_forwarded_as_kwargs(self) -> None:
        """BuildProgress is a dataclass; on_progress takes kwargs. Bridge must adapt."""
        from app.services.universe.trading212.builder import BuildProgress

        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()
        seen: list[dict] = []

        with _patched(builder_cls, MagicMock()):
            run_universe_build(
                UniverseBuildRequest(),
                MagicMock(),
                on_progress=lambda **kw: seen.append(kw),
            )
            forward = builder_cls.call_args.kwargs["progress_callback"]
            forward(
                BuildProgress(
                    current=3, total=9, current_exchange="NYSE", current_stock="AAPL"
                )
            )

        assert {
            "current": 3,
            "total": 9,
            "current_exchange": "NYSE",
            "current_stock": "AAPL",
        } in seen

    def test_reports_completed_status_on_success(self) -> None:
        """_run_step relies on this: without it the job is marked complete defensively."""
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result(errors=["boom"])
        seen: list[dict] = []

        with _patched(builder_cls, MagicMock()):
            run_universe_build(
                UniverseBuildRequest(),
                MagicMock(),
                on_progress=lambda **kw: seen.append(kw),
            )

        completed = [c for c in seen if c.get("status") == "completed"]
        assert len(completed) == 1
        assert completed[0]["errors"] == ["boom"]


class TestBuildPipelines:
    """The composition seam: stock vs ETF filter pipelines built from config."""

    def test_equity_pipeline_filter_order(self) -> None:
        equity, _ = _build_pipelines(UniverseBuilderConfig(), skip_filters=False)
        assert [f.name for f in equity.get_filters()] == [
            "MarketCapFilter",
            "PriceFilter",
            "LiquidityFilter",
            "DataCoverageFilter",
            "HistoricalDataFilter",
        ]

    def test_etf_pipeline_filter_order(self) -> None:
        _, etf = _build_pipelines(UniverseBuilderConfig(), skip_filters=False)
        assert etf is not None
        assert [f.name for f in etf.get_filters()] == [
            "AUMFilter",
            "HistoricalDataFilter",
        ]

    def test_skip_filters_yields_empty_equity_and_no_etf(self) -> None:
        equity, etf = _build_pipelines(UniverseBuilderConfig(), skip_filters=True)
        assert equity.get_filters() == []
        assert etf is None
