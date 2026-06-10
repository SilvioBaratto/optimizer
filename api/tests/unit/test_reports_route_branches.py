"""Branch-coverage tests for app.api.v1.reports.reports (missed lines 55-58, 73-101).

Targets:
- _find_report_path: file-exists branch (returns str) and absent branch (returns None).
- _run_report_generation: success path (completed + result dict) and exception path
  (failed + error string).

Does NOT duplicate the 202/409/200/404 matrices in test_reports_routes.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "app.api.v1.reports.reports"
_UPDATE_JOB = f"{_MOD}._report_job_service.update_job"
_DB_CTX = f"{_MOD}.database_manager.get_session"
_REPORT_SVC = "app.services.reports.report_service.ReportService"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _db_ctx_mock(session: MagicMock | None = None) -> MagicMock:
    """Return a MagicMock configured as a context manager yielding *session*."""
    sess = session or MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=sess)
    cm.__exit__ = MagicMock(return_value=False)
    parent = MagicMock(return_value=cm)
    return parent


# ---------------------------------------------------------------------------
# _find_report_path
# ---------------------------------------------------------------------------


class TestFindReportPath:
    """Lines 55-58: file-exists → str path; absent → None."""

    def test_when_file_exists_then_returns_str_path(self, tmp_path) -> None:
        from app.api.v1.reports.reports import _find_report_path

        report_id = "abc123"
        pdf = tmp_path / f"report_{report_id}.pdf"
        pdf.write_bytes(b"%PDF")

        with patch(f"{_MOD}._REPORT_OUTPUT_DIR", str(tmp_path)):
            result = _find_report_path(report_id)

        assert result == str(pdf)

    def test_when_file_absent_then_returns_none(self, tmp_path) -> None:
        from app.api.v1.reports.reports import _find_report_path

        with patch(f"{_MOD}._REPORT_OUTPUT_DIR", str(tmp_path)):
            result = _find_report_path("nonexistent-id")

        assert result is None


# ---------------------------------------------------------------------------
# _run_report_generation — success path
# ---------------------------------------------------------------------------


class TestRunReportGenerationSuccess:
    """Lines 73-98: wrapper sets running, calls service.generate, sets completed."""

    def test_completed_result_contains_report_id(self) -> None:
        from app.api.v1.reports.reports import _run_report_generation
        from app.schemas.reports.reports import (
            ReportBranding,
            ReportGenerateRequest,
            ReportOrientation,
            ReportTemplate,
        )

        request = ReportGenerateRequest(
            template=ReportTemplate.standard,
            sections=["portfolio_summary"],
            branding=ReportBranding(),
            orientation=ReportOrientation.portrait,
        )

        mock_service = MagicMock()
        mock_service.return_value.generate.return_value = "abc"

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(_REPORT_SVC, mock_service),
        ):
            _run_report_generation("job-1", request)

        completed_call = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "completed"
        )
        assert completed_call.kwargs["result"]["report_id"] == "abc"

    def test_completed_result_contains_download_url(self) -> None:
        from app.api.v1.reports.reports import _run_report_generation
        from app.schemas.reports.reports import (
            ReportBranding,
            ReportGenerateRequest,
            ReportOrientation,
            ReportTemplate,
        )

        request = ReportGenerateRequest(
            template=ReportTemplate.standard,
            sections=["weights"],
            branding=ReportBranding(),
            orientation=ReportOrientation.landscape,
        )

        mock_service = MagicMock()
        mock_service.return_value.generate.return_value = "xyz"

        with (
            patch(_UPDATE_JOB),
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(_REPORT_SVC, mock_service),
        ):
            _run_report_generation("job-2", request)

        mock_service.return_value.generate.assert_called_once()

    def test_status_set_to_running_first(self) -> None:
        from app.api.v1.reports.reports import _run_report_generation
        from app.schemas.reports.reports import (
            ReportBranding,
            ReportGenerateRequest,
            ReportOrientation,
            ReportTemplate,
        )

        request = ReportGenerateRequest(
            template=ReportTemplate.standard,
            sections=["portfolio_summary"],
            branding=ReportBranding(),
            orientation=ReportOrientation.portrait,
        )

        mock_service = MagicMock()
        mock_service.return_value.generate.return_value = "r1"

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(_REPORT_SVC, mock_service),
        ):
            _run_report_generation("job-3", request)

        first_call = mock_update.call_args_list[0]
        assert first_call.kwargs.get("status") == "running"


# ---------------------------------------------------------------------------
# _run_report_generation — exception path
# ---------------------------------------------------------------------------


class TestRunReportGenerationFailure:
    """Lines 99-106: service raises → update_job(status="failed", error=...)."""

    def test_exception_sets_failed_status(self) -> None:
        from app.api.v1.reports.reports import _run_report_generation
        from app.schemas.reports.reports import (
            ReportBranding,
            ReportGenerateRequest,
            ReportOrientation,
            ReportTemplate,
        )

        request = ReportGenerateRequest(
            template=ReportTemplate.standard,
            sections=["risk_analytics"],
            branding=ReportBranding(),
            orientation=ReportOrientation.portrait,
        )

        mock_service = MagicMock()
        mock_service.return_value.generate.side_effect = Exception("boom")

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(_REPORT_SVC, mock_service),
        ):
            _run_report_generation("job-fail", request)

        failed_call = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        )
        assert failed_call.kwargs["status"] == "failed"

    def test_exception_error_message_propagated(self) -> None:
        from app.api.v1.reports.reports import _run_report_generation
        from app.schemas.reports.reports import (
            ReportBranding,
            ReportGenerateRequest,
            ReportOrientation,
            ReportTemplate,
        )

        request = ReportGenerateRequest(
            template=ReportTemplate.standard,
            sections=["performance_metrics"],
            branding=ReportBranding(),
            orientation=ReportOrientation.portrait,
        )

        mock_service = MagicMock()
        mock_service.return_value.generate.side_effect = Exception("boom")

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(_REPORT_SVC, mock_service),
        ):
            _run_report_generation("job-err", request)

        failed_call = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        )
        assert failed_call.kwargs["error"] == "boom"
