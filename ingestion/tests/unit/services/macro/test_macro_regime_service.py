"""Unit tests for MacroRegimeService.fetch_country and run_bulk_macro_fetch (issue #822).

Covers gaps NOT already in test_macro_regime_service_fred_key.py:
  - fetch_country success: counts populated, repo upserts called correctly
  - fetch_country ilsole scraper raises: error surfaced, no propagation
  - fetch_country te scraper raises: error surfaced, no propagation
  - fetch_country te non-success status: errors captured, counts zeroed
  - run_bulk_macro_fetch: progress callback contract, session.commit per-country
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.macro.macro_regime_service import (
    MacroRegimeService,
    run_bulk_macro_fetch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_service(
    ilsole_forecasts: dict | Exception | None = None,
    te_response: dict | Exception | None = None,
) -> tuple[MacroRegimeService, MagicMock]:
    """Return (service, mock_repo) with pre-wired scraper behaviours."""
    mock_repo = MagicMock()
    mock_repo.upsert_economic_indicator.return_value = 1
    mock_repo.upsert_economic_indicator_observation.return_value = 1
    mock_repo.upsert_te_indicators.return_value = 3
    mock_repo.upsert_te_observations.return_value = 3
    mock_repo.upsert_bond_yields.return_value = 2
    mock_repo.upsert_bond_yield_observations.return_value = 2

    mock_ilsole = MagicMock()
    if isinstance(ilsole_forecasts, Exception):
        mock_ilsole.get_forecasts.side_effect = ilsole_forecasts
    else:
        mock_ilsole.get_forecasts.return_value = ilsole_forecasts

    mock_te = MagicMock()
    if isinstance(te_response, Exception):
        mock_te.get_country_indicators.side_effect = te_response
    else:
        mock_te.get_country_indicators.return_value = te_response

    service = MacroRegimeService(
        repo=mock_repo,
        ilsole_scraper=mock_ilsole,
        te_scraper=mock_te,
        fred_scraper=MagicMock(),
    )
    return service, mock_repo


_FORECAST_DATA = {
    "last_inflation": 3.1,
    "inflation_6m": 2.8,
    "gdp_growth_6m": 2.2,
    "timestamp": "2026-01-01T00:00:00",
}

_TE_SUCCESS = {
    "status": "success",
    "indicators": {"inflation_rate": 3.1, "gdp_growth_rate": 2.5},
    "bond_yields": {"10y": 4.2},
}

_TE_FAILED = {
    "status": "error",
    "error": "HTML parse failed",
    "indicators": {},
    "bond_yields": {},
}


# ---------------------------------------------------------------------------
# fetch_country — success
# ---------------------------------------------------------------------------


class TestFetchCountrySuccess:
    def test_when_both_scrapers_succeed_then_counts_all_populated(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA", include_bonds=True)

        assert result["errors"] == []
        assert result["counts"]["ilsole_forecast"] == 1
        assert result["counts"]["ilsole_observations"] == 1
        assert result["counts"]["te_indicators"] == 3
        assert result["counts"]["te_observations"] == 3
        assert result["counts"]["bond_yields"] == 2
        assert result["counts"]["bond_yield_observations"] == 2

    def test_when_both_scrapers_succeed_then_ilsole_upsert_called_with_country(
        self,
    ) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        service.fetch_country("USA", include_bonds=True)

        mock_repo.upsert_economic_indicator.assert_called_once()
        call_kwargs = mock_repo.upsert_economic_indicator.call_args
        assert call_kwargs.kwargs.get("country") == "USA" or (
            len(call_kwargs.args) > 0 and call_kwargs.args[0] == "USA"
        )

    def test_when_both_scrapers_succeed_then_te_upsert_called_with_country(
        self,
    ) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        service.fetch_country("Germany", include_bonds=True)

        mock_repo.upsert_te_indicators.assert_called_once()
        call_kwargs = mock_repo.upsert_te_indicators.call_args
        assert call_kwargs.kwargs.get("country") == "Germany"

    def test_when_ilsole_returns_none_then_ilsole_counts_zeroed(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=None,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert result["counts"]["ilsole_forecast"] == 0
        assert result["counts"]["ilsole_observations"] == 0
        assert result["errors"] == []

    def test_when_include_bonds_false_then_bond_counts_zeroed(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA", include_bonds=False)

        assert result["counts"]["bond_yields"] == 0
        assert result["counts"]["bond_yield_observations"] == 0
        mock_repo.upsert_bond_yields.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_country — ilsole scraper raises
# ---------------------------------------------------------------------------


class TestFetchCountryIlsoleRaises:
    def test_when_ilsole_raises_then_error_message_contains_ilsole_forecast(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=RuntimeError("timeout"),
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert any("ilsole_forecast" in e for e in result["errors"])

    def test_when_ilsole_raises_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=RuntimeError("boom"),
            te_response=_TE_SUCCESS,
        )

        # Must not raise
        result = service.fetch_country("USA")
        assert isinstance(result, dict)

    def test_when_ilsole_raises_then_te_still_processed(self) -> None:
        service, mock_repo = _build_service(
            ilsole_forecasts=RuntimeError("boom"),
            te_response=_TE_SUCCESS,
        )

        result = service.fetch_country("USA")

        assert result["counts"].get("te_indicators", 0) == 3
        mock_repo.upsert_te_indicators.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_country — te scraper raises
# ---------------------------------------------------------------------------


class TestFetchCountryTeRaises:
    def test_when_te_raises_then_error_message_contains_trading_economics(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("connection refused"),
        )

        result = service.fetch_country("USA")

        assert any("trading_economics" in e for e in result["errors"])

    def test_when_te_raises_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("boom"),
        )

        result = service.fetch_country("USA")
        assert isinstance(result, dict)

    def test_when_te_raises_then_te_counts_default_to_zero(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=RuntimeError("boom"),
        )

        result = service.fetch_country("USA")

        assert result["counts"].get("te_indicators", 0) == 0
        assert result["counts"].get("te_observations", 0) == 0
        assert result["counts"].get("bond_yields", 0) == 0
        assert result["counts"].get("bond_yield_observations", 0) == 0


# ---------------------------------------------------------------------------
# fetch_country — te non-success status
# ---------------------------------------------------------------------------


class TestFetchCountryTeNonSuccess:
    def test_when_te_status_error_then_errors_contains_trading_economics(
        self,
    ) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")

        assert any("trading_economics" in e for e in result["errors"])

    def test_when_te_status_error_then_te_counts_zeroed(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")

        assert result["counts"]["te_indicators"] == 0
        assert result["counts"]["te_observations"] == 0
        assert result["counts"]["bond_yields"] == 0
        assert result["counts"]["bond_yield_observations"] == 0

    def test_when_te_status_error_then_no_exception_propagates(self) -> None:
        service, _ = _build_service(
            ilsole_forecasts=_FORECAST_DATA,
            te_response=_TE_FAILED,
        )

        result = service.fetch_country("USA")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# run_bulk_macro_fetch — progress and session.commit contract
# ---------------------------------------------------------------------------


class TestRunBulkMacroFetch:
    def _run_bulk(
        self,
        countries: list[str],
        on_progress: MagicMock | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Run run_bulk_macro_fetch with database_manager patched at source."""
        request = SimpleNamespace(countries=countries, include_bonds=True)
        mock_session = MagicMock()
        mock_svc_cls = MagicMock()
        mock_svc_cls.return_value.fetch_country.return_value = {
            "counts": {"te_indicators": 2},
            "errors": [],
        }

        with (
            patch("app.database.database_manager") as mock_db,
            patch("app.services.macro.macro_regime_service.MacroRegimeRepository"),
            patch(
                "app.services.macro.macro_regime_service.MacroRegimeService",
                mock_svc_cls,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            kwargs: dict = {}
            if on_progress is not None:
                kwargs["on_progress"] = on_progress
            run_bulk_macro_fetch(request, **kwargs)

        return mock_session, on_progress or MagicMock()

    def test_when_two_countries_then_on_progress_total_equals_two(self) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        first_call_kwargs = on_progress.call_args_list[0].kwargs
        assert first_call_kwargs.get("total") == 2

    def test_when_two_countries_then_on_progress_fired_for_each_country(
        self,
    ) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        all_kwargs = [c.kwargs for c in on_progress.call_args_list]
        countries_seen = [kw.get("current_country") for kw in all_kwargs]
        assert "USA" in countries_seen
        assert "Germany" in countries_seen

    def test_when_two_countries_then_session_commit_called_twice(self) -> None:
        mock_session, _ = self._run_bulk(["USA", "Germany"])
        # One session.commit() per country = 2
        assert mock_session.commit.call_count == 2

    def test_when_no_errors_then_final_progress_status_is_completed(self) -> None:
        on_progress = MagicMock()
        self._run_bulk(["USA", "Germany"], on_progress=on_progress)

        # The final on_progress call carries status
        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"
