"""Branch-coverage tests for report_service.py missed lines.

Targets:
  - Lines 85-88:  get_weights exception handler (graceful degradation)
  - Lines 94-97:  get_performance_metrics exception handler
  - Lines 103-106: get_risk_analytics exception handler
  - Lines 335-349: _render_performance_metrics non-empty metrics table path
  - Lines 354-373: _render_risk_analytics empty-analytics and table-render paths

Pure unit — no DB, no real PDF written. MagicMock repos injected directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aggregator(*, portfolio_repo=None, execution_repo=None):
    from app.services.reports.report_service import ReportDataAggregator

    return ReportDataAggregator(
        portfolio_repo=portfolio_repo or MagicMock(),
        execution_repo=execution_repo or MagicMock(),
    )


def _make_story_and_styles():
    """Return a plain list (story) and a real reportlab stylesheet."""
    from reportlab.lib.styles import getSampleStyleSheet

    return [], getSampleStyleSheet()


# ---------------------------------------------------------------------------
# FILE 1 — Section A: graceful-degradation exception handlers (lines 85-106)
# ---------------------------------------------------------------------------


class TestGracefulDegradationExceptionHandlers:
    """Each public method must swallow exceptions and return the placeholder."""

    def test_get_weights_returns_placeholder_when_execution_repo_raises(self):
        """Lines 85-88: exception inside _fetch_weights → placeholder."""
        execution_repo = MagicMock()
        execution_repo.get_optimization_runs.side_effect = RuntimeError("DB gone")
        aggregator = _make_aggregator(execution_repo=execution_repo)

        result = aggregator.get_weights(portfolio_id="irrelevant-id")

        assert result["available"] is False
        assert "message" in result

    def test_get_performance_metrics_returns_placeholder_when_repo_raises(self):
        """Lines 94-97: exception inside _fetch_performance_metrics → placeholder."""
        execution_repo = MagicMock()
        execution_repo.get_backtest_runs.side_effect = RuntimeError("timeout")
        aggregator = _make_aggregator(execution_repo=execution_repo)

        result = aggregator.get_performance_metrics(portfolio_id="irrelevant-id")

        assert result["available"] is False
        assert "message" in result

    def test_get_risk_analytics_returns_placeholder_when_internal_raises(self):
        """Lines 103-106: exception in get_risk_analytics outer try → placeholder."""
        execution_repo = MagicMock()
        portfolio_repo = MagicMock()
        aggregator = _make_aggregator(
            portfolio_repo=portfolio_repo, execution_repo=execution_repo
        )

        # Patch the private fetch to raise so the outer except fires
        with patch.object(
            aggregator,
            "_fetch_risk_analytics",
            side_effect=ValueError("unexpected"),
        ):
            result = aggregator.get_risk_analytics(portfolio_id="any-id")

        assert result["available"] is False

    def test_get_weights_error_is_logged_at_warning(self, caplog):
        """Exception in get_weights must be logged at WARNING level."""
        import logging

        execution_repo = MagicMock()
        execution_repo.get_optimization_runs.side_effect = Exception("broken")
        aggregator = _make_aggregator(execution_repo=execution_repo)

        with caplog.at_level(logging.WARNING, logger="app.services.reports.report_service"):
            aggregator.get_weights(portfolio_id="pid")

        assert any("weights unavailable" in r.message for r in caplog.records)

    def test_get_performance_metrics_error_is_logged_at_warning(self, caplog):
        """Exception in get_performance_metrics must be logged at WARNING level."""
        import logging

        execution_repo = MagicMock()
        execution_repo.get_backtest_runs.side_effect = Exception("boom")
        aggregator = _make_aggregator(execution_repo=execution_repo)

        with caplog.at_level(
            logging.WARNING, logger="app.services.reports.report_service"
        ):
            aggregator.get_performance_metrics(portfolio_id="pid")

        assert any("performance_metrics unavailable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# FILE 1 — Section B: PdfBuilder._render_performance_metrics (lines 335-349)
# ---------------------------------------------------------------------------


class TestRenderPerformanceMetrics:
    """Cover both branches in _render_performance_metrics."""

    def _builder(self):
        from app.services.reports.report_service import PdfBuilder

        b = PdfBuilder()
        # Inject a valid primary color so TableStyle does not fail
        from reportlab.lib import colors

        b._primary_color = colors.HexColor("#003366")
        return b

    def test_when_metrics_empty_then_no_data_paragraph_appended(self):
        """Lines 331-333: empty metrics dict → 'No performance data.' paragraph."""
        builder = self._builder()
        story, styles = _make_story_and_styles()

        builder._render_performance_metrics(story, styles, {"metrics": {}})

        # story must have exactly one element (the placeholder paragraph)
        assert len(story) == 1
        # The Paragraph text must contain 'No performance data.'
        para = story[0]
        assert "No performance data" in para.text

    def test_when_metrics_present_then_table_appended(self):
        """Lines 335-349: non-empty metrics → Table added to story."""
        from reportlab.platypus import Table

        builder = self._builder()
        story, styles = _make_story_and_styles()
        data = {"available": True, "metrics": {"sharpe": 1.5, "cagr": 0.12}}

        builder._render_performance_metrics(story, styles, data)

        assert len(story) == 1
        assert isinstance(story[0], Table)

    def test_when_metrics_present_then_table_contains_header_row(self):
        """Table must have ['Metric', 'Value'] as first row."""
        from reportlab.platypus import Table

        builder = self._builder()
        story, styles = _make_story_and_styles()
        builder._render_performance_metrics(
            story, styles, {"metrics": {"vol": 0.15}}
        )

        table = story[0]
        assert isinstance(table, Table)
        # reportlab Table stores its data internally as ._cellvalues
        assert table._cellvalues[0] == ["Metric", "Value"]

    def test_when_metrics_is_none_then_no_data_paragraph_appended(self):
        """metrics=None is falsy → 'No performance data.' placeholder."""
        builder = self._builder()
        story, styles = _make_story_and_styles()

        builder._render_performance_metrics(story, styles, {"metrics": None})

        assert len(story) == 1
        assert "No performance data" in story[0].text


# ---------------------------------------------------------------------------
# FILE 1 — Section C: PdfBuilder._render_risk_analytics (lines 354-373)
# ---------------------------------------------------------------------------


class TestRenderRiskAnalytics:
    """Cover both branches in _render_risk_analytics."""

    def _builder(self):
        from reportlab.lib import colors

        from app.services.reports.report_service import PdfBuilder

        b = PdfBuilder()
        b._primary_color = colors.HexColor("#003366")
        return b

    def test_when_analytics_absent_then_no_data_paragraph_appended(self):
        """Lines 354-357: data has no 'analytics' key → 'No risk analytics data.'."""
        builder = self._builder()
        story, styles = _make_story_and_styles()

        builder._render_risk_analytics(story, styles, {"available": True})

        assert len(story) == 1
        assert "No risk analytics" in story[0].text

    def test_when_analytics_empty_dict_then_no_data_paragraph_appended(self):
        """analytics={} is falsy → placeholder paragraph."""
        builder = self._builder()
        story, styles = _make_story_and_styles()

        builder._render_risk_analytics(story, styles, {"analytics": {}})

        assert len(story) == 1
        assert "No risk analytics" in story[0].text

    def test_when_analytics_present_then_table_appended(self):
        """Lines 359-373: non-empty analytics dict → Table in story."""
        from reportlab.platypus import Table

        builder = self._builder()
        story, styles = _make_story_and_styles()

        builder._render_risk_analytics(
            story, styles, {"analytics": {"var_95": -0.02, "cvar_95": -0.03}}
        )

        assert len(story) == 1
        assert isinstance(story[0], Table)

    def test_when_analytics_present_then_table_has_header_row(self):
        """Table must have ['Metric', 'Value'] as first row."""
        from reportlab.platypus import Table

        builder = self._builder()
        story, styles = _make_story_and_styles()
        builder._render_risk_analytics(
            story, styles, {"analytics": {"vol": 0.18}}
        )

        table = story[0]
        assert isinstance(table, Table)
        assert table._cellvalues[0] == ["Metric", "Value"]

    def test_risk_analytics_section_renders_via_full_build(self):
        """Integration path: PdfBuilder.build with risk_analytics available data
        must produce valid PDF bytes and reach lines 359-373."""
        from app.services.reports.report_service import PdfBuilder

        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data={
                "risk_analytics": {
                    "available": True,
                    "analytics": {"var_95": -0.02, "cvar_95": -0.03},
                }
            },
            branding={"company_name": "Risk Co", "primary_color": "#660000"},
            orientation="portrait",
        )

        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 100

    def test_performance_metrics_section_renders_via_full_build(self):
        """Integration path: PdfBuilder.build with performance_metrics available data
        must produce valid PDF bytes and reach lines 335-349."""
        from app.services.reports.report_service import PdfBuilder

        builder = PdfBuilder()
        pdf_bytes = builder.build(
            section_data={
                "performance_metrics": {
                    "available": True,
                    "metrics": {"sharpe": 1.4, "cagr": 0.11},
                }
            },
            branding={"company_name": "Perf Co", "primary_color": "#003366"},
            orientation="portrait",
        )

        assert pdf_bytes[:4] == b"%PDF"
        assert len(pdf_bytes) > 100
