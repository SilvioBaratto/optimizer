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
        patch(f"{M}.YFinanceUniverseSource"),
        patch(f"{M}.get_yfinance_client"),
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
            summary = run_universe_build(UniverseBuildRequest())

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
            run_universe_build(UniverseBuildRequest())

        session.commit.assert_called_once()

    def test_builder_is_wired_without_any_filter_pipeline(self) -> None:
        """Ingestion is filter-free: the builder takes no filter pipeline."""
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()

        with _patched(builder_cls, MagicMock()):
            run_universe_build(UniverseBuildRequest())

        kwargs = builder_cls.call_args.kwargs
        assert "filter_pipeline" not in kwargs
        assert "etf_filter_pipeline" not in kwargs
        assert "skip_filters" not in kwargs

    def test_forwards_request_options_to_builder(self) -> None:
        builder_cls = MagicMock()
        builder_cls.return_value.build.return_value = _make_result()

        with _patched(builder_cls, MagicMock()):
            run_universe_build(UniverseBuildRequest(exchanges=["NYSE"], max_workers=7))

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
                on_progress=lambda **kw: seen.append(kw),
            )

        completed = [c for c in seen if c.get("status") == "completed"]
        assert len(completed) == 1
        assert completed[0]["errors"] == ["boom"]


class TestAUMScreenRemoved:
    """The ETF AUM screen was removed — no class, no config surface, no export."""

    def test_aum_filter_is_not_importable(self) -> None:
        import app.services.universe.trading212.filters as filters

        assert not hasattr(filters, "AUMFilter")

    def test_config_has_no_aum_fields(self) -> None:
        config = UniverseBuilderConfig()
        assert not hasattr(config, "etf_min_aum")
        assert not hasattr(config, "etf_aum_fx_to_eur")
