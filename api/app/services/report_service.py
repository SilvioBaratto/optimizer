"""Report generation service.

SOLID design:
  - ReportDataAggregator: single responsibility — collects section data from DB
  - PdfBuilder: single responsibility — renders section data into PDF bytes
  - ReportService: orchestrates aggregation + build + file persistence

Graceful degradation: every section method returns a placeholder dict on
any exception, so a missing upstream (no backtest run, no risk analytics)
never causes the whole report to fail.
"""

from __future__ import annotations

import io
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from app.repositories.execution_repository import ExecutionRepository
    from app.repositories.portfolio_repository import PortfolioRepository

logger = logging.getLogger(__name__)

# Placeholder sentinel used when data is unavailable.
_NO_DATA: dict[str, Any] = {"available": False, "message": "No data available"}


# ---------------------------------------------------------------------------
# Data aggregator (SRP: only knows how to fetch data)
# ---------------------------------------------------------------------------


class ReportDataAggregator:
    """Collects per-section data from repositories.

    Each method returns a plain dict — either real data or a placeholder.
    Methods never raise; errors are logged and a placeholder is returned.

    Args:
        portfolio_repo: Repository for portfolio lookups.
        execution_repo: Repository for optimization and backtest run lookups.
    """

    def __init__(
        self,
        portfolio_repo: PortfolioRepository,
        execution_repo: ExecutionRepository,
    ) -> None:
        self._portfolio_repo = portfolio_repo
        self._execution_repo = execution_repo

    # ------------------------------------------------------------------
    # Section methods
    # ------------------------------------------------------------------

    def get_portfolio_summary(self, portfolio_id: str | None) -> dict[str, Any]:
        """Return portfolio-level summary data."""
        try:
            return self._fetch_portfolio_summary(portfolio_id)
        except Exception as exc:
            logger.warning("portfolio_summary unavailable: %s", exc)
            return dict(_NO_DATA)

    def get_weights(self, portfolio_id: str | None) -> dict[str, Any]:
        """Return latest portfolio weights."""
        try:
            return self._fetch_weights(portfolio_id)
        except Exception as exc:
            logger.warning("weights unavailable: %s", exc)
            return dict(_NO_DATA)

    def get_performance_metrics(self, portfolio_id: str | None) -> dict[str, Any]:
        """Return performance metrics from the latest backtest run."""
        try:
            return self._fetch_performance_metrics(portfolio_id)
        except Exception as exc:
            logger.warning("performance_metrics unavailable: %s", exc)
            return dict(_NO_DATA)

    def get_risk_analytics(self, portfolio_id: str | None) -> dict[str, Any]:
        """Return risk analytics (VaR, CVaR, volatility)."""
        try:
            return self._fetch_risk_analytics(portfolio_id)
        except Exception as exc:
            logger.warning("risk_analytics unavailable: %s", exc)
            return dict(_NO_DATA)

    def collect_sections(
        self,
        sections: list[str],
        portfolio_id: str | None,
    ) -> dict[str, dict[str, Any]]:
        """Return a dict mapping section ID → section data for all requested sections."""
        _dispatch = {
            "portfolio_summary": self.get_portfolio_summary,
            "weights": self.get_weights,
            "performance_metrics": self.get_performance_metrics,
            "risk_analytics": self.get_risk_analytics,
        }
        return {
            section_id: _dispatch[section_id](portfolio_id)
            for section_id in sections
            if section_id in _dispatch
        }

    # ------------------------------------------------------------------
    # Private fetch helpers (delegate to repositories)
    # ------------------------------------------------------------------

    def _fetch_portfolio_summary(self, portfolio_id: str | None) -> dict[str, Any]:
        """Query portfolio summary via PortfolioRepository."""
        if portfolio_id is None:
            return dict(_NO_DATA)

        portfolio = self._portfolio_repo.get_by_id(uuid.UUID(portfolio_id))
        if portfolio is None:
            return dict(_NO_DATA)

        return {
            "available": True,
            "portfolio_id": portfolio_id,
            "name": portfolio.name,
        }

    def _fetch_weights(self, portfolio_id: str | None) -> dict[str, Any]:
        """Query latest completed optimization run weights via ExecutionRepository."""
        if portfolio_id is None:
            return dict(_NO_DATA)

        runs = self._execution_repo.get_optimization_runs(
            portfolio_id=uuid.UUID(portfolio_id),
            status="completed",
            limit=1,
        )
        if not runs or not runs[0].weights:
            return dict(_NO_DATA)

        weights = runs[0].weights
        items = [{"ticker": k, "weight": v} for k, v in weights.items()]
        return {"available": True, "items": items}

    def _fetch_performance_metrics(self, portfolio_id: str | None) -> dict[str, Any]:
        """Query latest completed backtest summary_stats via ExecutionRepository."""
        if portfolio_id is None:
            return dict(_NO_DATA)

        runs = self._execution_repo.get_backtest_runs(
            portfolio_id=uuid.UUID(portfolio_id),
            status="completed",
            limit=1,
        )
        if not runs or not runs[0].summary_stats:
            return dict(_NO_DATA)

        return {"available": True, "metrics": runs[0].summary_stats}

    def _fetch_risk_analytics(self, portfolio_id: str | None) -> dict[str, Any]:
        """Return placeholder — risk_analytics_runs has no ORM model yet."""
        return dict(_NO_DATA)


# ---------------------------------------------------------------------------
# PDF builder (SRP: only knows how to render PDF)
# ---------------------------------------------------------------------------


class PdfBuilder:
    """Renders section data into a PDF document using reportlab.

    Produces text and table layout only (no charts in v1).
    """

    _SECTION_TITLES: dict[str, str] = {
        "portfolio_summary": "Portfolio Summary",
        "weights": "Asset Allocation",
        "performance_metrics": "Performance Metrics",
        "risk_analytics": "Risk Analytics",
    }

    # Primary color stored per-build so renderers access it without an extra param.
    _primary_color: Any = None

    def build(
        self,
        section_data: dict[str, dict[str, Any]],
        branding: dict[str, Any],
        orientation: str,
    ) -> bytes:
        """Render section_data into PDF bytes."""
        buffer = io.BytesIO()
        page_size = landscape(A4) if orientation == "landscape" else A4
        doc = SimpleDocTemplate(buffer, pagesize=page_size)
        styles = getSampleStyleSheet()
        story: list = []

        primary_hex = branding.get("primary_color", "#003366").lstrip("#")
        try:
            r = int(primary_hex[0:2], 16) / 255
            g = int(primary_hex[2:4], 16) / 255
            b = int(primary_hex[4:6], 16) / 255
            self._primary_color = colors.Color(r, g, b)
        except (ValueError, IndexError):
            self._primary_color = colors.HexColor("#003366")

        company_name = branding.get("company_name", "")
        if company_name:
            story.append(Paragraph(company_name, styles["Title"]))
            story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("Portfolio Report", styles["h1"]))
        story.append(Spacer(1, 0.6 * cm))

        for section_id, data in section_data.items():
            title = self._SECTION_TITLES.get(section_id, section_id.replace("_", " ").title())
            story.append(Paragraph(title, styles["h2"]))
            story.append(Spacer(1, 0.2 * cm))

            if not data.get("available", False):
                story.append(Paragraph(data.get("message", "No data available"), styles["Normal"]))
            else:
                self._render_section(story, styles, section_id, data)

            story.append(Spacer(1, 0.5 * cm))

        if branding.get("include_disclaimer"):
            story.append(Spacer(1, 1 * cm))
            disclaimer = (
                "Disclaimer: This report is for informational purposes only and does not "
                "constitute investment advice. Past performance is not indicative of future results."
            )
            story.append(Paragraph(disclaimer, styles["Normal"]))

        doc.build(story)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Private section renderers
    # ------------------------------------------------------------------

    def _render_section(
        self,
        story: list,
        styles: Any,
        section_id: str,
        data: dict[str, Any],
    ) -> None:
        """Dispatch rendering to section-specific method."""
        _renderers = {
            "portfolio_summary": self._render_portfolio_summary,
            "weights": self._render_weights,
            "performance_metrics": self._render_performance_metrics,
            "risk_analytics": self._render_risk_analytics,
        }
        renderer = _renderers.get(section_id, self._render_generic)
        renderer(story, styles, data)

    def _render_portfolio_summary(
        self, story: list, styles: Any, data: dict[str, Any]
    ) -> None:
        for key in ("name", "portfolio_id", "total_value", "num_assets", "currency"):
            if key in data:
                story.append(Paragraph(f"{key.replace('_', ' ').title()}: {data[key]}", styles["Normal"]))

    def _render_weights(
        self, story: list, styles: Any, data: dict[str, Any]
    ) -> None:
        items = data.get("items", [])
        if not items:
            story.append(Paragraph("No weights data.", styles["Normal"]))
            return

        table_data = [["Ticker", "Weight (%)"]]
        for item in items:
            table_data.append([item.get("ticker", ""), f"{item.get('weight', 0) * 100:.2f}"])

        t = Table(table_data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self._primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)

    def _render_performance_metrics(
        self, story: list, styles: Any, data: dict[str, Any]
    ) -> None:
        metrics = data.get("metrics", {})
        if not metrics:
            story.append(Paragraph("No performance data.", styles["Normal"]))
            return

        table_data = [["Metric", "Value"]]
        for k, v in metrics.items():
            table_data.append([str(k), str(v)])

        t = Table(table_data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self._primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)

    def _render_risk_analytics(
        self, story: list, styles: Any, data: dict[str, Any]
    ) -> None:
        analytics = data.get("analytics", {})
        if not analytics:
            story.append(Paragraph("No risk analytics data.", styles["Normal"]))
            return

        table_data = [["Metric", "Value"]]
        for k, v in analytics.items():
            table_data.append([str(k), str(v)])

        t = Table(table_data)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self._primary_color),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)

    def _render_generic(
        self, story: list, styles: Any, data: dict[str, Any]
    ) -> None:
        story.append(Paragraph(str(data), styles["Normal"]))


# ---------------------------------------------------------------------------
# Report service (orchestrator, SRP: persistence + coordination only)
# ---------------------------------------------------------------------------


class ReportService:
    """Orchestrates data aggregation, PDF building, and file persistence.

    Args:
        session: SQLAlchemy session for data access.
        output_dir: Directory where generated PDF files are stored.
    """

    def __init__(self, session: Session, output_dir: str) -> None:
        from app.repositories.execution_repository import ExecutionRepository
        from app.repositories.portfolio_repository import PortfolioRepository

        self._aggregator = ReportDataAggregator(
            portfolio_repo=PortfolioRepository(session),
            execution_repo=ExecutionRepository(session),
        )
        self._builder = PdfBuilder()
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        sections: list[str],
        branding: dict[str, Any],
        orientation: str,
        portfolio_id: str | None,
    ) -> str:
        """Generate a PDF report and save it to disk.

        Returns:
            report_id: A unique identifier for the generated report.
        """
        report_id = str(uuid.uuid4())
        section_data = self._aggregator.collect_sections(sections, portfolio_id)
        pdf_bytes = self._builder.build(
            section_data=section_data,
            branding=branding,
            orientation=orientation,
        )
        filename = f"report_{report_id}.pdf"
        file_path = os.path.join(self._output_dir, filename)
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info("Report %s saved to %s", report_id, file_path)
        return report_id
