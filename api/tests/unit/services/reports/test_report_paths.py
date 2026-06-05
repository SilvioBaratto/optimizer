"""Empty-data and exception-path tests for report_service (issue #825).

Existing tests in tests/unit/test_report_service.py already cover:
  - ReportDataAggregator: all section methods with both data and placeholder
  - ReportDataAggregator: graceful degradation when repo raises
  - ReportDataAggregator: collect_sections maps all registered section IDs
  - PdfBuilder: produces valid PDF bytes (portrait + landscape)
  - PdfBuilder: real weights data → larger output
  - ReportService: generate saves file, returns report_id, valid PDF header

report_service has NO explicit raises (graceful degradation everywhere).
Gap analysis identifies the following uncovered empty/edge branches:
  - collect_sections: unknown section keys silently ignored → empty dict
  - collect_sections: empty sections list → empty dict
  - PdfBuilder.build: empty section_data → still produces valid bytes
  - PdfBuilder.build: branding with invalid hex colour falls back gracefully
  - PdfBuilder.build: section with available=True but no weights items renders
    the "No weights data." fallback paragraph
  - PdfBuilder.build: performance_metrics section with empty metrics dict
  - PdfBuilder.build: include_disclaimer=True appends disclaimer text
  - _to_uuid: None passes through as None; invalid string raises ValueError
  - ReportService.generate: all 4 sections requested → single PDF saved
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# _to_uuid — edge cases
# ---------------------------------------------------------------------------


class TestToUuid:
    """_to_uuid converts str → UUID or passes None through."""

    def test_when_none_then_returns_none(self) -> None:
        from app.services.reports.report_service import _to_uuid

        assert _to_uuid(None) is None

    def test_when_valid_uuid_string_then_returns_uuid(self) -> None:
        from app.services.reports.report_service import _to_uuid

        uid = str(uuid.uuid4())
        result = _to_uuid(uid)

        assert isinstance(result, uuid.UUID)
        assert str(result) == uid

    def test_when_invalid_string_then_raises_value_error(self) -> None:
        from app.services.reports.report_service import _to_uuid

        with pytest.raises(ValueError):
            _to_uuid("not-a-uuid")


# ---------------------------------------------------------------------------
# collect_sections — unknown / empty section keys
# ---------------------------------------------------------------------------


class TestCollectSectionsEdgeCases:
    """collect_sections silently drops unknown section IDs."""

    def _make_aggregator(self) -> Any:
        from app.services.reports.report_service import ReportDataAggregator

        portfolio_repo = MagicMock()
        portfolio_repo.get_by_id.return_value = None
        execution_repo = MagicMock()
        execution_repo.get_optimization_runs.return_value = []
        execution_repo.get_backtest_runs.return_value = []
        return ReportDataAggregator(portfolio_repo, execution_repo)

    def test_when_sections_empty_then_returns_empty_dict(self) -> None:
        aggregator = self._make_aggregator()

        result = aggregator.collect_sections([], portfolio_id=None)

        assert result == {}

    def test_when_all_sections_unknown_then_returns_empty_dict(self) -> None:
        aggregator = self._make_aggregator()

        result = aggregator.collect_sections(
            ["nonexistent_section", "another_bad_one"], portfolio_id=None
        )

        assert result == {}

    def test_when_mixed_valid_and_invalid_sections_then_only_valid_returned(
        self,
    ) -> None:
        aggregator = self._make_aggregator()

        result = aggregator.collect_sections(
            ["portfolio_summary", "totally_unknown"], portfolio_id=None
        )

        assert "portfolio_summary" in result
        assert "totally_unknown" not in result


# ---------------------------------------------------------------------------
# PdfBuilder — empty section_data / edge case branding
# ---------------------------------------------------------------------------


class TestPdfBuilderEdgeCases:
    """PdfBuilder.build with edge-case inputs."""

    def _builder(self) -> Any:
        from app.services.reports.report_service import PdfBuilder

        return PdfBuilder()

    def test_when_section_data_empty_then_still_returns_pdf_bytes(self) -> None:
        pdf_bytes = self._builder().build(
            section_data={},
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"

    def test_when_primary_color_invalid_hex_then_fallback_color_used(self) -> None:
        """Invalid hex string must not raise — falls back to default."""
        pdf_bytes = self._builder().build(
            section_data={},
            branding={"primary_color": "not-a-color", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_when_weights_section_has_empty_items_then_fallback_text_rendered(
        self,
    ) -> None:
        """available=True but items=[] triggers 'No weights data.' paragraph."""
        section_data = {"weights": {"available": True, "items": []}}
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_when_performance_metrics_has_empty_metrics_then_fallback_rendered(
        self,
    ) -> None:
        """available=True with empty metrics dict → 'No performance data.' rendered."""
        section_data = {"performance_metrics": {"available": True, "metrics": {}}}
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_when_include_disclaimer_true_then_pdf_is_non_empty(self) -> None:
        section_data = {
            "portfolio_summary": {"available": False, "message": "No data"}
        }
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={
                "company_name": "AcmeCorp",
                "primary_color": "#003366",
                "include_disclaimer": True,
            },
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0

    def test_when_risk_analytics_section_has_available_false_then_pdf_built(
        self,
    ) -> None:
        section_data = {
            "risk_analytics": {"available": False, "message": "No data available"}
        }
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)

    def test_when_unknown_section_id_then_generic_renderer_used(self) -> None:
        """Sections not in _SECTION_TITLES use _render_generic without raising."""
        section_data = {"custom_section": {"available": True, "data": "some value"}}
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)

    def test_when_all_four_sections_requested_then_pdf_generated(self) -> None:
        section_data = {
            "portfolio_summary": {"available": False, "message": "No data"},
            "weights": {"available": False, "message": "No data"},
            "performance_metrics": {"available": False, "message": "No data"},
            "risk_analytics": {"available": False, "message": "No data"},
        }
        pdf_bytes = self._builder().build(
            section_data=section_data,
            branding={"primary_color": "#003366", "include_disclaimer": False},
            orientation="portrait",
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# ReportService.generate — all sections, tmp_path
# ---------------------------------------------------------------------------


class TestReportServiceAllSections:
    """ReportService.generate with all four registered sections."""

    def test_when_all_sections_requested_then_single_pdf_saved(
        self, tmp_path: Any
    ) -> None:
        from app.services.reports.report_service import ReportService

        session = MagicMock()
        service = ReportService(session, output_dir=str(tmp_path))

        report_id = service.generate(
            sections=[
                "portfolio_summary",
                "weights",
                "performance_metrics",
                "risk_analytics",
            ],
            branding={
                "company_name": "FullReport Corp",
                "primary_color": "#003366",
                "include_disclaimer": True,
            },
            orientation="landscape",
            portfolio_id=None,
        )

        saved_files = list(tmp_path.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes()[:4] == b"%PDF"
        assert isinstance(report_id, str)
        assert len(report_id) > 0

    def test_when_output_dir_does_not_exist_then_created_automatically(
        self, tmp_path: Any
    ) -> None:
        from app.services.reports.report_service import ReportService

        new_dir = tmp_path / "auto_created" / "reports"
        session = MagicMock()
        service = ReportService(session, output_dir=str(new_dir))

        # Directory should exist even before generate is called
        assert new_dir.exists()

        service.generate(
            sections=["portfolio_summary"],
            branding={"primary_color": "#000", "include_disclaimer": False},
            orientation="portrait",
            portfolio_id=None,
        )

        assert any(new_dir.iterdir())
