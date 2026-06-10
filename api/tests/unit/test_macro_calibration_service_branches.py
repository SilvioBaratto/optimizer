"""Branch-coverage tests for app/services/macro/macro_calibration.py.

Maps the term-missing ranges to concrete conditions not covered by
tests/unit/test_macro_calibration.py:

  Lines 100-111  — _build_macro_summary: forecast_rows (gdp/inflation/earnings),
                   te_table rows with None raw_name, bond yield map with/without
                   10Y-2Y spread computation
  Line  117->116 — te_table branch: empty te_table (no matching indicators)
  Lines 134-135  — bond_yields with yield_value=None (not added to yield_map)
  Lines 137-145  — bond 10Y-2Y spread both present → steepening, inverted
  Lines 220-221  — cached result branch (no force_refresh, cache hit)
  Line  235->242 — macro_summary_override=None + force_refresh=True → build summary + LLM call
  Lines 268-289  — upsert_macro_calibration: success path and exception path
  Lines 317-347  — run_bulk_calibrate: success + error per-country, on_progress contract

Do NOT duplicate tests from tests/unit/test_macro_calibration.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from baml_client.types import BusinessCyclePhase, MacroRegimeCalibration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_calibration(
    phase: BusinessCyclePhase = BusinessCyclePhase.MID_EXPANSION,
    delta: float = 2.75,
    tau: float = 0.025,
    confidence: float = 0.80,
    rationale: str = "Test rationale.",
) -> MacroRegimeCalibration:
    return MacroRegimeCalibration(
        phase=phase, delta=delta, tau=tau, confidence=confidence, rationale=rationale
    )


def _make_repo(
    *,
    indicators: list | None = None,
    te_rows: list | None = None,
    bond_yields: list | None = None,
    news_summary: object = None,
    calibration: object = None,
) -> MagicMock:
    mock_repo = MagicMock()
    mock_repo.get_economic_indicators.return_value = indicators or []
    mock_repo.get_te_indicators.return_value = te_rows or []
    mock_repo.get_bond_yields.return_value = bond_yields or []
    mock_repo.get_macro_news_summary.return_value = news_summary
    mock_repo.get_macro_calibration.return_value = calibration
    mock_repo.upsert_macro_calibration.return_value = None
    return mock_repo


def _make_indicator(
    gdp_growth_6m: float | None = None,
    inflation_6m: float | None = None,
    earnings_12m: float | None = None,
) -> MagicMock:
    ind = MagicMock()
    ind.gdp_growth_6m = gdp_growth_6m
    ind.inflation_6m = inflation_6m
    ind.earnings_12m = earnings_12m
    return ind


def _make_te_row(
    indicator_key: str = "manufacturing_pmi",
    value: float | None = 54.2,
    raw_name: str | None = "Manufacturing PMI",
    unit: str | None = "index",
) -> MagicMock:
    row = MagicMock()
    row.indicator_key = indicator_key
    row.value = value
    row.raw_name = raw_name
    row.unit = unit
    return row


def _make_bond(maturity: str, yield_value: float | None) -> MagicMock:
    bond = MagicMock()
    bond.maturity = maturity
    bond.yield_value = yield_value
    return bond


# ---------------------------------------------------------------------------
# _build_macro_summary — forecast rows branch (lines 100-111)
# ---------------------------------------------------------------------------


class TestBuildMacroSummaryForecastRows:
    """Cover the forecast_rows construction when indicators have values."""

    def test_when_gdp_growth_present_then_forecast_section_added(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            indicators=[_make_indicator(gdp_growth_6m=2.5)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "### Consensus Forecasts" in result
        assert "GDP 6m forecast" in result
        assert "2.50%" in result

    def test_when_inflation_6m_present_then_inflation_row_added(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            indicators=[_make_indicator(inflation_6m=3.1)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "Inflation 6m forecast" in result
        assert "3.1%" in result

    def test_when_earnings_12m_present_then_earnings_row_added(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            indicators=[_make_indicator(earnings_12m=8.5)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "Earnings 12m forecast" in result
        assert "8.5%" in result

    def test_when_all_three_forecast_fields_present_all_rows_added(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            indicators=[_make_indicator(gdp_growth_6m=2.0, inflation_6m=2.5, earnings_12m=7.0)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "GDP 6m forecast" in result
        assert "Inflation 6m forecast" in result
        assert "Earnings 12m forecast" in result

    def test_when_indicator_all_none_then_no_forecast_section(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            indicators=[_make_indicator()],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "### Consensus Forecasts" not in result


# ---------------------------------------------------------------------------
# _build_macro_summary — te_table: empty table branch (line 117->116)
# ---------------------------------------------------------------------------


class TestBuildMacroSummaryTETable:
    def test_when_te_indicator_not_in_key_set_then_no_economic_section(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        # "unknown_indicator" not in _KEY_TE_INDICATORS
        repo = _make_repo(te_rows=[_make_te_row(indicator_key="unknown_indicator")])
        result = _build_macro_summary(repo, "USA")
        assert "### Economic Indicators" not in result

    def test_when_te_value_is_none_then_row_skipped(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(te_rows=[_make_te_row(indicator_key="manufacturing_pmi", value=None)])
        result = _build_macro_summary(repo, "USA")
        assert "### Economic Indicators" not in result

    def test_when_raw_name_is_none_then_indicator_key_used_as_label(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            te_rows=[_make_te_row(indicator_key="manufacturing_pmi", raw_name=None)],
        )
        result = _build_macro_summary(repo, "USA")
        assert "manufacturing_pmi" in result

    def test_when_unit_is_none_then_empty_unit_string_used(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(te_rows=[_make_te_row(unit=None)])
        result = _build_macro_summary(repo, "USA")
        # Row should still appear; unit column is just empty
        assert "### Economic Indicators" in result


# ---------------------------------------------------------------------------
# _build_macro_summary — bond yields (lines 134-145)
# ---------------------------------------------------------------------------


class TestBuildMacroSummaryBondYields:
    def test_when_bond_yield_value_none_then_not_added_to_yield_map(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        # yield_value=None → bond skipped → yield_map empty → no bond section
        repo = _make_repo(
            bond_yields=[_make_bond("10Y", None)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "### Bond Yields" not in result

    def test_when_only_10y_present_then_no_spread_computed(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            bond_yields=[_make_bond("10Y", 4.5)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "### Bond Yields" in result
        assert "10Y-2Y spread" not in result

    def test_when_10y_above_2y_then_spread_is_steepening(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            bond_yields=[_make_bond("10Y", 4.5), _make_bond("2Y", 4.0)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "steepening" in result
        assert "10Y-2Y spread" in result

    def test_when_10y_below_2y_then_spread_is_inverted(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            bond_yields=[_make_bond("10Y", 3.5), _make_bond("2Y", 4.5)],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "inverted" in result

    def test_when_multiple_maturities_present_section_appears(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo(
            bond_yields=[
                _make_bond("2Y", 4.0),
                _make_bond("5Y", 4.2),
                _make_bond("10Y", 4.5),
            ],
            te_rows=[_make_te_row()],
        )
        result = _build_macro_summary(repo, "USA")
        assert "### Bond Yields" in result
        assert "2Y" in result
        assert "5Y" in result
        assert "10Y" in result


# ---------------------------------------------------------------------------
# _build_macro_summary — returns empty string when only header (line 163)
# ---------------------------------------------------------------------------


class TestBuildMacroSummaryEmptyResult:
    def test_when_no_data_then_returns_empty_string(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        repo = _make_repo()
        result = _build_macro_summary(repo, "USA")
        assert result == ""


# ---------------------------------------------------------------------------
# classify_macro_regime — cache hit branch (lines 220-221)
# ---------------------------------------------------------------------------


class TestClassifyMacroRegimeCacheBranch:
    def test_when_cached_result_exists_then_returned_without_llm_call(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        cached = MagicMock()
        cached.phase = "MID_EXPANSION"
        cached.delta = 2.75
        cached.tau = 0.025
        cached.confidence = 0.80
        cached.rationale = "Cached rationale."
        cached.macro_summary = "Cached summary."

        mock_repo = _make_repo(calibration=cached)
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch("app.services.macro.macro_calibration.b.ClassifyMacroRegime") as mock_llm,
        ):
            result = classify_macro_regime(mock_session, country="USA")

        mock_llm.assert_not_called()
        assert result.phase == BusinessCyclePhase.MID_EXPANSION
        assert result.rationale == "Cached rationale."
        assert result.macro_summary == "Cached summary."

    def test_when_force_refresh_true_then_cache_bypassed_and_llm_called(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_raw = _make_raw_calibration()
        mock_repo = _make_repo()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
                return_value=mock_raw,
            ) as mock_llm,
        ):
            # force_refresh=True but no override → _build_macro_summary called
            # We use override to avoid an empty-data ValueError
            classify_macro_regime(
                mock_session,
                country="USA",
                macro_summary_override="Some macro text.",
                force_refresh=True,
            )

        mock_llm.assert_called_once()

    def test_when_cache_returns_none_and_no_override_then_builds_summary(self) -> None:
        """No cache, no override → calls _build_macro_summary."""
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_raw = _make_raw_calibration()
        mock_repo = _make_repo(
            calibration=None,
            te_rows=[_make_te_row()],
        )
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
                return_value=mock_raw,
            ),
        ):
            result = classify_macro_regime(mock_session, country="USA")

        assert result.phase == BusinessCyclePhase.MID_EXPANSION


# ---------------------------------------------------------------------------
# classify_macro_regime — upsert success + exception paths (lines 268-289)
# ---------------------------------------------------------------------------


class TestClassifyMacroRegimeUpsert:
    _SUMMARY = "GDP strong; PMI 55."

    def test_when_override_provided_then_upsert_not_called(self) -> None:
        """macro_summary_override is not None → persist block skipped."""
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_raw = _make_raw_calibration()
        mock_repo = _make_repo()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
                return_value=mock_raw,
            ),
        ):
            classify_macro_regime(
                mock_session, macro_summary_override=self._SUMMARY
            )

        mock_repo.upsert_macro_calibration.assert_not_called()

    def test_when_upsert_raises_then_exception_logged_and_result_still_returned(
        self,
    ) -> None:
        """upsert failure is non-fatal; CalibrationResult is still returned."""
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_raw = _make_raw_calibration()
        mock_repo = _make_repo(
            calibration=None,
            te_rows=[_make_te_row()],
        )
        mock_repo.upsert_macro_calibration.side_effect = RuntimeError("DB write failed")
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
                return_value=mock_raw,
            ),
        ):
            result = classify_macro_regime(mock_session, country="USA")

        # Result is returned despite DB failure
        assert result.phase == BusinessCyclePhase.MID_EXPANSION

    def test_when_upsert_succeeds_then_phase_persisted(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_raw = _make_raw_calibration(phase=BusinessCyclePhase.RECESSION)
        mock_repo = _make_repo(
            calibration=None,
            te_rows=[_make_te_row()],
        )
        mock_session = MagicMock()

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            patch(
                "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
                return_value=mock_raw,
            ),
        ):
            result = classify_macro_regime(mock_session, country="USA")

        mock_repo.upsert_macro_calibration.assert_called_once()
        call_kwargs = mock_repo.upsert_macro_calibration.call_args
        persisted_data = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
        assert persisted_data["phase"] == "RECESSION"
        assert result.phase == BusinessCyclePhase.RECESSION


# ---------------------------------------------------------------------------
# run_bulk_calibrate — progress, per-country success + error (lines 317-347)
# ---------------------------------------------------------------------------


class TestRunBulkCalibrate:
    """run_bulk_calibrate imports database_manager lazily inside the function body."""

    _DB_PATCH = "app.database.database_manager"

    def test_when_all_countries_succeed_then_error_count_zero(self) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        mock_session = MagicMock()

        with (
            patch(self._DB_PATCH) as mock_db,
            patch(
                "app.services.macro.macro_calibration.classify_macro_regime",
                return_value=MagicMock(phase=BusinessCyclePhase.MID_EXPANSION),
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_calibrate(["USA", "Germany"], force_refresh=False)

        assert result["countries_processed"] == 2
        assert result["error_count"] == 0

    def test_when_one_country_fails_then_error_count_is_one_and_continues(
        self,
    ) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        def _classify_side_effect(session, country, force_refresh):
            if country == "Broken":
                raise ValueError("No macro data")
            return MagicMock(phase=BusinessCyclePhase.MID_EXPANSION)

        mock_session = MagicMock()

        with (
            patch(self._DB_PATCH) as mock_db,
            patch(
                "app.services.macro.macro_calibration.classify_macro_regime",
                side_effect=_classify_side_effect,
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_calibrate(["USA", "Broken"], force_refresh=False)

        assert result["countries_processed"] == 2
        assert result["error_count"] == 1

    def test_on_progress_receives_total_at_start(self) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        on_progress = MagicMock()
        mock_session = MagicMock()

        with (
            patch(self._DB_PATCH) as mock_db,
            patch(
                "app.services.macro.macro_calibration.classify_macro_regime",
                return_value=MagicMock(phase=BusinessCyclePhase.MID_EXPANSION),
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_calibrate(["USA"], force_refresh=False, on_progress=on_progress)

        first_kwargs = on_progress.call_args_list[0].kwargs
        assert first_kwargs.get("total") == 1

    def test_on_progress_final_call_has_completed_status(self) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        on_progress = MagicMock()
        mock_session = MagicMock()

        with (
            patch(self._DB_PATCH) as mock_db,
            patch(
                "app.services.macro.macro_calibration.classify_macro_regime",
                return_value=MagicMock(phase=BusinessCyclePhase.MID_EXPANSION),
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_calibrate(["USA"], force_refresh=False, on_progress=on_progress)

        final_kwargs = on_progress.call_args_list[-1].kwargs
        assert final_kwargs.get("status") == "completed"

    def test_session_rollback_called_when_country_raises(self) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        mock_session = MagicMock()

        with (
            patch(self._DB_PATCH) as mock_db,
            patch(
                "app.services.macro.macro_calibration.classify_macro_regime",
                side_effect=RuntimeError("LLM timeout"),
            ),
        ):
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            run_bulk_calibrate(["USA"], force_refresh=True)

        mock_session.rollback.assert_called_once()

    def test_empty_countries_list_returns_zero_counts(self) -> None:
        from app.services.macro.macro_calibration import run_bulk_calibrate

        mock_session = MagicMock()

        with patch(self._DB_PATCH) as mock_db:
            mock_db.get_session.return_value.__enter__.return_value = mock_session
            mock_db.get_session.return_value.__exit__.return_value = False

            result = run_bulk_calibrate([], force_refresh=False)

        assert result["countries_processed"] == 0
        assert result["error_count"] == 0


# ---------------------------------------------------------------------------
# build_bl_config_from_calibration — views kwarg branch
# ---------------------------------------------------------------------------


class TestBuildBlConfigViews:
    def _make_result(self) -> object:
        from app.services.macro.macro_calibration import CalibrationResult

        return CalibrationResult(
            phase=BusinessCyclePhase.MID_EXPANSION,
            delta=2.75,
            tau=0.025,
            confidence=0.80,
            rationale="Test.",
            macro_summary="Test summary.",
        )

    def test_when_views_provided_then_views_list_in_config(self) -> None:
        from app.services.macro.macro_calibration import (
            build_bl_config_from_calibration,
        )

        result = self._make_result()
        cfg = build_bl_config_from_calibration(result, views=("AAPL == 0.02", "MSFT == -0.01"))  # type: ignore[arg-type]
        assert cfg["views"] == ["AAPL == 0.02", "MSFT == -0.01"]

    def test_when_no_views_then_empty_list(self) -> None:
        from app.services.macro.macro_calibration import (
            build_bl_config_from_calibration,
        )

        result = self._make_result()
        cfg = build_bl_config_from_calibration(result)
        assert cfg["views"] == []

    def test_cov_estimator_is_ledoit_wolf(self) -> None:
        from app.services.macro.macro_calibration import (
            build_bl_config_from_calibration,
        )

        result = self._make_result()
        cfg = build_bl_config_from_calibration(result)
        assert cfg["prior_config"]["cov_estimator"] == "ledoit_wolf"
