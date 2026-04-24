"""Unit tests for ReportService and PdfBuilder.

TDD: tests written before implementation.

Covers:
  - Data aggregation per section (graceful degradation when data missing)
  - PDF builder produces non-empty bytes output
  - Section data shape
  - No raw SQL in ReportDataAggregator (repositories only)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = "Test Portfolio") -> MagicMock:
    p = MagicMock()
    p.name = name
    return p


def _make_optimization_run(weights: dict) -> MagicMock:
    run = MagicMock()
    run.weights = weights
    run.status = "completed"
    return run


def _make_backtest_run(summary_stats: dict) -> MagicMock:
    run = MagicMock()
    run.summary_stats = summary_stats
    run.status = "completed"
    return run


def _make_repos(
    portfolio=None,
    opt_runs=None,
    backtest_runs=None,
):
    """Build mock PortfolioRepository and ExecutionRepository."""
    portfolio_repo = MagicMock()
    portfolio_repo.get_by_id.return_value = portfolio

    execution_repo = MagicMock()
    execution_repo.get_optimization_runs.return_value = opt_runs or []
    execution_repo.get_backtest_runs.return_value = backtest_runs or []

    return portfolio_repo, execution_repo


class TestReportDataAggregatorNoRawSQL:
    """ReportDataAggregator must not execute raw SQL — only repository calls."""

    def test_portfolio_summary_uses_portfolio_repository_not_raw_sql(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos(
            portfolio=_make_portfolio("Alpha Fund")
        )
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        pid = str(uuid.uuid4())

        result = aggregator.get_portfolio_summary(portfolio_id=pid)

        assert result["available"] is True
        assert result["name"] == "Alpha Fund"
        portfolio_repo.get_by_id.assert_called_once()
        # raw SQL would call session.execute — no session here at all
        execution_repo.get_optimization_runs.assert_not_called()

    def test_portfolio_summary_returns_placeholder_when_not_found(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos(portfolio=None)
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_portfolio_summary(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False

    def test_weights_uses_execution_repository_not_raw_sql(self) -> None:
        from app.services.report_service import ReportDataAggregator

        opt_run = _make_optimization_run({"AAPL": 0.6, "MSFT": 0.4})
        portfolio_repo, execution_repo = _make_repos(opt_runs=[opt_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        pid = str(uuid.uuid4())

        result = aggregator.get_weights(portfolio_id=pid)

        assert result["available"] is True
        assert len(result["items"]) == 2
        tickers = {item["ticker"] for item in result["items"]}
        assert tickers == {"AAPL", "MSFT"}
        execution_repo.get_optimization_runs.assert_called_once()

    def test_weights_returns_placeholder_when_no_completed_run(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos(opt_runs=[])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_weights(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False

    def test_performance_metrics_uses_execution_repository_not_raw_sql(
        self,
    ) -> None:
        from app.services.report_service import ReportDataAggregator

        stats = {"sharpe": 1.5, "cagr": 0.12}
        backtest_run = _make_backtest_run(stats)
        portfolio_repo, execution_repo = _make_repos(backtest_runs=[backtest_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        pid = str(uuid.uuid4())

        result = aggregator.get_performance_metrics(portfolio_id=pid)

        assert result["available"] is True
        assert result["metrics"] == stats
        execution_repo.get_backtest_runs.assert_called_once()

    def test_performance_metrics_returns_placeholder_when_no_completed_run(
        self,
    ) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos(backtest_runs=[])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_performance_metrics(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False

    def test_risk_analytics_returns_placeholder_without_raw_sql(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        # Should not raise and no raw SQL called — risk_analytics_runs has no model
        result = aggregator.get_risk_analytics(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False
        # Repositories not queried for a table that has no model yet
        execution_repo.get_optimization_runs.assert_not_called()
        execution_repo.get_backtest_runs.assert_not_called()


class TestAdHocBacktestFallback:
    """Regression for issue #463.

    Ad-hoc backtests create :class:`BacktestRun` rows without a
    ``portfolio_id``. ``_fetch_performance_metrics`` previously short-
    circuited to ``_NO_DATA`` whenever ``portfolio_id`` was ``None``, so
    reports for ad-hoc runs always claimed no data was available. The
    fix is to forward ``portfolio_id=None`` to the repository (which
    then returns the latest completed run across all portfolios) and
    only call a run "no data" when ``summary_stats`` / ``weights`` is
    actually ``None`` — never when it is an empty dict.
    """

    def test_performance_metrics_falls_back_when_portfolio_id_is_none(
        self,
    ) -> None:
        from app.services.report_service import ReportDataAggregator

        stats = {"sharpe": 1.5, "cagr": 0.12}
        backtest_run = _make_backtest_run(stats)
        portfolio_repo, execution_repo = _make_repos(backtest_runs=[backtest_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_performance_metrics(portfolio_id=None)

        assert result["available"] is True
        assert result["metrics"] == stats
        # Repository must be queried with portfolio_id=None (no filter)
        # so ad-hoc runs are retrieved.
        call_kwargs = execution_repo.get_backtest_runs.call_args.kwargs
        assert call_kwargs.get("portfolio_id") is None

    def test_performance_metrics_empty_summary_stats_is_not_no_data(self) -> None:
        """Empty ``summary_stats`` dict is valid data, not a missing run."""
        from app.services.report_service import ReportDataAggregator

        # BacktestRun.summary_stats defaults to {} on creation, not None.
        backtest_run = _make_backtest_run({})
        portfolio_repo, execution_repo = _make_repos(backtest_runs=[backtest_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_performance_metrics(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is True
        assert result["metrics"] == {}

    def test_performance_metrics_none_summary_stats_is_no_data(self) -> None:
        """When ``summary_stats`` is explicitly ``None``, fall back to placeholder."""
        from app.services.report_service import ReportDataAggregator

        backtest_run = _make_backtest_run(None)  # type: ignore[arg-type]
        portfolio_repo, execution_repo = _make_repos(backtest_runs=[backtest_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_performance_metrics(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False

    def test_weights_falls_back_when_portfolio_id_is_none(self) -> None:
        from app.services.report_service import ReportDataAggregator

        opt_run = _make_optimization_run({"AAPL": 0.6, "MSFT": 0.4})
        portfolio_repo, execution_repo = _make_repos(opt_runs=[opt_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_weights(portfolio_id=None)

        assert result["available"] is True
        assert len(result["items"]) == 2
        call_kwargs = execution_repo.get_optimization_runs.call_args.kwargs
        assert call_kwargs.get("portfolio_id") is None

    def test_weights_empty_dict_is_not_no_data(self) -> None:
        """Empty ``weights`` dict is valid data (run completed, items just empty)."""
        from app.services.report_service import ReportDataAggregator

        opt_run = _make_optimization_run({})
        portfolio_repo, execution_repo = _make_repos(opt_runs=[opt_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_weights(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is True
        assert result["items"] == []

    def test_weights_none_weights_is_no_data(self) -> None:
        """When ``weights`` is explicitly ``None``, fall back to placeholder."""
        from app.services.report_service import ReportDataAggregator

        opt_run = _make_optimization_run(None)  # type: ignore[arg-type]
        portfolio_repo, execution_repo = _make_repos(opt_runs=[opt_run])
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_weights(portfolio_id=str(uuid.uuid4()))

        assert result["available"] is False


class TestReportDataAggregator:
    """ReportDataAggregator collects section data from repositories."""

    def test_portfolio_summary_returns_placeholder_when_no_data(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        result = aggregator.get_portfolio_summary(portfolio_id=None)

        assert result is not None
        assert isinstance(result, dict)

    def test_weights_section_returns_placeholder_when_no_data(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        result = aggregator.get_weights(portfolio_id=None)

        assert result is not None
        assert isinstance(result, dict)

    def test_performance_metrics_returns_placeholder_when_no_data(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        result = aggregator.get_performance_metrics(portfolio_id=None)

        assert result is not None
        assert isinstance(result, dict)

    def test_risk_analytics_returns_placeholder_when_no_data(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        result = aggregator.get_risk_analytics(portfolio_id=None)

        assert result is not None
        assert isinstance(result, dict)

    def test_missing_section_does_not_raise(self) -> None:
        """Graceful degradation: repository error returns placeholder, not exception."""
        from app.services.report_service import ReportDataAggregator

        portfolio_repo = MagicMock()
        portfolio_repo.get_by_id.side_effect = Exception("DB unavailable")
        execution_repo = MagicMock()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)

        result = aggregator.get_portfolio_summary(portfolio_id=str(uuid.uuid4()))

        assert isinstance(result, dict)
        assert result["available"] is False

    def test_collect_sections_returns_dict_per_section(self) -> None:
        from app.services.report_service import ReportDataAggregator

        portfolio_repo, execution_repo = _make_repos()
        aggregator = ReportDataAggregator(portfolio_repo, execution_repo)
        sections = ["portfolio_summary", "weights"]
        result = aggregator.collect_sections(sections, portfolio_id=None)

        assert isinstance(result, dict)
        for s in sections:
            assert s in result


class TestPdfBuilder:
    """PdfBuilder renders section data into PDF bytes."""

    def test_build_returns_bytes(self) -> None:
        from app.services.report_service import PdfBuilder

        section_data = {
            "portfolio_summary": {"available": False, "message": "No data available"},
        }
        branding = {
            "company_name": "Test Corp",
            "primary_color": "#003366",
            "include_disclaimer": False,
        }
        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data=section_data,
            branding=branding,
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_build_produces_pdf_header(self) -> None:
        from app.services.report_service import PdfBuilder

        section_data = {
            "weights": {"available": False, "message": "No data available"},
        }
        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data=section_data,
            branding={"company_name": "Corp", "primary_color": "#000", "include_disclaimer": False},
            orientation="portrait",
        )

        # PDF files start with %PDF
        assert pdf_bytes[:4] == b"%PDF"

    def test_build_landscape_does_not_raise(self) -> None:
        from app.services.report_service import PdfBuilder

        section_data = {
            "portfolio_summary": {"available": False, "message": "No data available"},
        }
        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data=section_data,
            branding={"company_name": "Corp", "primary_color": "#000", "include_disclaimer": False},
            orientation="landscape",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_build_with_real_data_produces_larger_output(self) -> None:
        from app.services.report_service import PdfBuilder

        section_data = {
            "portfolio_summary": {
                "available": True,
                "total_value": 1_000_000.0,
                "num_assets": 10,
                "currency": "USD",
            },
            "weights": {
                "available": True,
                "items": [
                    {"ticker": "AAPL", "weight": 0.25},
                    {"ticker": "MSFT", "weight": 0.15},
                ],
            },
        }
        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data=section_data,
            branding={
                "company_name": "Acme Capital",
                "primary_color": "#003366",
                "include_disclaimer": True,
            },
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100


class TestReportService:
    """ReportService orchestrates aggregation + PDF building + file persistence."""

    def test_generate_saves_pdf_to_output_dir(self, tmp_path: Any) -> None:
        from app.services.report_service import ReportService

        session = MagicMock()
        service = ReportService(session, output_dir=str(tmp_path))
        report_id = service.generate(
            sections=["portfolio_summary"],
            branding={"company_name": "X", "primary_color": "#000", "include_disclaimer": False},
            orientation="portrait",
            portfolio_id=None,
        )

        saved_files = list(tmp_path.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].suffix == ".pdf"
        assert report_id in saved_files[0].name

    def test_generate_returns_non_empty_report_id(self, tmp_path: Any) -> None:
        from app.services.report_service import ReportService

        session = MagicMock()
        service = ReportService(session, output_dir=str(tmp_path))
        report_id = service.generate(
            sections=["portfolio_summary"],
            branding={"company_name": "X", "primary_color": "#000", "include_disclaimer": False},
            orientation="portrait",
            portfolio_id=None,
        )

        assert report_id
        assert isinstance(report_id, str)

    def test_generate_produces_valid_pdf_file(self, tmp_path: Any) -> None:
        from app.services.report_service import ReportService

        session = MagicMock()
        service = ReportService(session, output_dir=str(tmp_path))
        service.generate(
            sections=["portfolio_summary", "weights"],
            branding={"company_name": "X", "primary_color": "#000", "include_disclaimer": False},
            orientation="portrait",
            portfolio_id=None,
        )

        saved_file = next(tmp_path.iterdir())
        content = saved_file.read_bytes()
        assert content[:4] == b"%PDF"
