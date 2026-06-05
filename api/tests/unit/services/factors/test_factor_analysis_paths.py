"""Unit tests for app.services.factors.factor_analysis_service — gap-fill paths.

No existing unit tests for this module (only route-level mocks in
tests/unit/test_factor_utility_routes.py).

Covers:

select_stocks_for_tickers:
  - FactorDataError when no factor scores found (explicit raise)
  - Happy path returns expected keys and is JSON-safe

build_exposure_constraints_for_tickers:
  - FactorDataError when no standardized scores found (explicit raise)
  - Happy path with mocked library call

compute_quintile_spread_for_tickers:
  - FactorDataError when no factor scores found (first explicit raise)
  - FactorDataError when prices are empty (second explicit raise)
  - Happy path returns expected keys and is JSON-safe

compute_regime_tilt:
  - FactorDataError when macro DataFrame is empty (explicit raise)
  - Happy path returns regime, tilted_weights, tilt_multipliers

Internal helpers:
  - _build_composite_scores_series: aggregates by ticker
  - _resolve_bounds: list → tuple, dict → dict-of-tuples
  - _build_scores_wide: empty list → empty DataFrame
  - _fetch_macro_dataframe: no observations → empty DataFrame
  - _parse_group_weights: unknown group key is skipped
  - _compute_buffer_zone: None current_members → all selected in entered
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_MODULE = "app.services.factors.factor_analysis_service"
_REPO_PATH = f"{_MODULE}.FactorRepository"
_MACRO_REPO_PATH = f"{_MODULE}.MacroRegimeRepository"
_SELECT_STOCKS = f"{_MODULE}.select_stocks"
_BUILD_CONSTRAINTS = f"{_MODULE}.build_factor_exposure_constraints"
_COMPUTE_QUINTILE = f"{_MODULE}.compute_quintile_spread"
_CLASSIFY_REGIME = f"{_MODULE}.classify_regime"
_APPLY_TILTS = f"{_MODULE}.apply_regime_tilts"
_GET_TILTS = f"{_MODULE}.get_regime_tilts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> MagicMock:
    return MagicMock()


def _make_score(
    ticker: str = "AAPL",
    score_date: datetime.date | None = None,
    factor_type: str = "momentum_12_1",
    raw_score: float = 0.5,
    standardized_score: float | None = 0.8,
) -> MagicMock:
    s = MagicMock()
    s.ticker = ticker
    s.score_date = score_date or datetime.date(2023, 6, 30)
    s.factor_type = factor_type
    s.raw_score = raw_score
    s.standardized_score = standardized_score
    return s


def _make_factor_repo(scores: list | None = None) -> MagicMock:
    repo = MagicMock()
    repo.get_scores_by_tickers_and_date_range.return_value = scores or []
    return repo


def _make_constraints_result() -> MagicMock:
    import numpy as np

    result = MagicMock()
    result.left_inequality = np.array([[0.1, 0.2], [0.3, 0.4]])
    result.right_inequality = np.array([0.5, 0.6])
    return result


def _make_quintile_result() -> MagicMock:
    idx = pd.date_range("2023-01-02", periods=3)
    result = MagicMock()
    result.quintile_returns = pd.DataFrame(
        {"Q1": [0.01, 0.02, 0.03], "Q5": [0.05, 0.06, 0.07]}, index=idx
    )
    result.spread_returns = pd.Series([0.04, 0.04, 0.04], index=idx)
    result.annualised_mean = 0.10
    return result


# ---------------------------------------------------------------------------
# select_stocks_for_tickers
# ---------------------------------------------------------------------------


class TestSelectStocksForTickers:
    def test_when_no_scores_then_raises_factor_data_error(self) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            select_stocks_for_tickers,
        )

        session = _make_session()
        repo = _make_factor_repo(scores=[])

        with patch(_REPO_PATH, return_value=repo):
            with pytest.raises(FactorDataError, match="No factor scores"):
                select_stocks_for_tickers(
                    session=session,
                    tickers=["AAPL", "MSFT"],
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2023, 12, 31),
                )

    def test_when_scores_exist_then_returns_expected_keys(self) -> None:
        from app.services.factors.factor_analysis_service import (
            select_stocks_for_tickers,
        )

        scores = [_make_score("AAPL"), _make_score("MSFT")]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        selected_index = pd.Index(["AAPL"])
        with (
            patch(_REPO_PATH, return_value=repo),
            patch(_SELECT_STOCKS, return_value=(selected_index, 0.1)),
        ):
            result = select_stocks_for_tickers(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 12, 31),
            )

        assert "selected_tickers" in result
        assert "count" in result
        assert "buffer_zone" in result
        assert_json_safe(result)

    def test_when_scores_exist_then_count_matches_selected(self) -> None:
        from app.services.factors.factor_analysis_service import (
            select_stocks_for_tickers,
        )

        scores = [_make_score("AAPL"), _make_score("MSFT")]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        selected_index = pd.Index(["AAPL", "MSFT"])
        with (
            patch(_REPO_PATH, return_value=repo),
            patch(_SELECT_STOCKS, return_value=(selected_index, None)),
        ):
            result = select_stocks_for_tickers(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 12, 31),
            )

        assert result["count"] == 2


# ---------------------------------------------------------------------------
# build_exposure_constraints_for_tickers
# ---------------------------------------------------------------------------


class TestBuildExposureConstraintsForTickers:
    def test_when_no_standardized_scores_then_raises_factor_data_error(
        self,
    ) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            build_exposure_constraints_for_tickers,
        )

        # Scores exist but no standardized_score
        scores = [_make_score("AAPL", standardized_score=None)]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        with patch(_REPO_PATH, return_value=repo):
            with pytest.raises(FactorDataError, match="No standardized"):
                build_exposure_constraints_for_tickers(
                    session=session,
                    tickers=["AAPL"],
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2023, 12, 31),
                    bounds=[0.0, 1.0],
                )

    def test_when_empty_scores_then_raises_factor_data_error(self) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            build_exposure_constraints_for_tickers,
        )

        session = _make_session()
        repo = _make_factor_repo(scores=[])

        with patch(_REPO_PATH, return_value=repo):
            with pytest.raises(FactorDataError):
                build_exposure_constraints_for_tickers(
                    session=session,
                    tickers=["AAPL"],
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2023, 12, 31),
                    bounds=[0.0, 1.0],
                )

    def test_when_valid_scores_then_returns_left_and_right_inequality(self) -> None:
        from app.services.factors.factor_analysis_service import (
            build_exposure_constraints_for_tickers,
        )

        scores = [_make_score("AAPL", standardized_score=0.8)]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        with (
            patch(_REPO_PATH, return_value=repo),
            patch(_BUILD_CONSTRAINTS, return_value=_make_constraints_result()),
        ):
            result = build_exposure_constraints_for_tickers(
                session=session,
                tickers=["AAPL"],
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 12, 31),
                bounds=[0.0, 1.0],
            )

        assert "left_inequality" in result
        assert "right_inequality" in result
        assert isinstance(result["left_inequality"], list)
        assert isinstance(result["right_inequality"], list)
        assert_json_safe(result)


# ---------------------------------------------------------------------------
# compute_quintile_spread_for_tickers
# ---------------------------------------------------------------------------


class TestComputeQuintileSpreadForTickers:
    def test_when_no_factor_scores_then_raises_factor_data_error(self) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            compute_quintile_spread_for_tickers,
        )

        session = _make_session()
        repo = _make_factor_repo(scores=[])

        with patch(_REPO_PATH, return_value=repo):
            with pytest.raises(FactorDataError, match="No factor scores"):
                compute_quintile_spread_for_tickers(
                    session=session,
                    tickers=["AAPL"],
                    factor_name="momentum_12_1",
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2023, 12, 31),
                )

    def test_when_no_price_history_then_raises_factor_data_error(self) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            compute_quintile_spread_for_tickers,
        )

        scores = [_make_score("AAPL")]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        # _fetch_price_rows returns [] → _build_returns_from_price_rows → empty df
        with (
            patch(_REPO_PATH, return_value=repo),
            patch(f"{_MODULE}._fetch_price_rows", return_value=[]),
        ):
            with pytest.raises(FactorDataError, match="No price history"):
                compute_quintile_spread_for_tickers(
                    session=session,
                    tickers=["AAPL"],
                    factor_name="momentum_12_1",
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2023, 12, 31),
                )

    def test_when_valid_data_then_returns_expected_keys_and_json_safe(self) -> None:
        from app.services.factors.factor_analysis_service import (
            compute_quintile_spread_for_tickers,
        )

        scores = [_make_score("AAPL")]
        session = _make_session()
        repo = _make_factor_repo(scores=scores)

        # Provide non-empty returns so the price guard passes
        idx = pd.date_range("2023-01-02", periods=3)
        returns_df = pd.DataFrame({"AAPL": [0.01, 0.02, 0.03]}, index=idx)

        with (
            patch(_REPO_PATH, return_value=repo),
            patch(f"{_MODULE}._fetch_price_rows", return_value=[MagicMock()]),
            patch(
                f"{_MODULE}._build_returns_from_price_rows", return_value=returns_df
            ),
            patch(_COMPUTE_QUINTILE, return_value=_make_quintile_result()),
        ):
            result = compute_quintile_spread_for_tickers(
                session=session,
                tickers=["AAPL"],
                factor_name="momentum_12_1",
                start_date=datetime.date(2023, 1, 1),
                end_date=datetime.date(2023, 12, 31),
            )

        assert "quintile_cumulative_returns" in result
        assert "spread_cumulative_return" in result
        assert "annualized_spread" in result
        assert_json_safe(result)


# ---------------------------------------------------------------------------
# compute_regime_tilt
# ---------------------------------------------------------------------------


class TestComputeRegimeTilt:
    def test_when_no_macro_data_then_raises_factor_data_error(self) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            compute_regime_tilt,
        )

        session = _make_session()
        macro_repo = MagicMock()
        macro_repo.get_te_observations.return_value = []

        with patch(_MACRO_REPO_PATH, return_value=macro_repo):
            with pytest.raises(FactorDataError, match="No macro data"):
                compute_regime_tilt(
                    session=session,
                    group_weights={"quality": 0.5, "momentum": 0.5},
                )

    def test_when_observations_all_none_value_then_raises_factor_data_error(
        self,
    ) -> None:
        from app.services.factors.factor_analysis_service import (
            FactorDataError,
            compute_regime_tilt,
        )

        obs = MagicMock()
        obs.value = None
        obs.date = datetime.date(2023, 12, 31)
        obs.indicator_key = "gdp_growth"

        session = _make_session()
        macro_repo = MagicMock()
        macro_repo.get_te_observations.return_value = [obs]

        with patch(_MACRO_REPO_PATH, return_value=macro_repo):
            with pytest.raises(FactorDataError, match="No macro data"):
                compute_regime_tilt(
                    session=session,
                    group_weights={"quality": 0.5},
                )

    def test_when_valid_macro_data_then_returns_regime_and_weights(self) -> None:
        from app.services.factors.factor_analysis_service import (
            compute_regime_tilt,
        )
        from optimizer.factors import FactorGroupType

        obs = MagicMock()
        obs.value = 2.5
        obs.date = datetime.date(2023, 12, 31)
        obs.indicator_key = "gdp_growth"

        session = _make_session()
        macro_repo = MagicMock()
        macro_repo.get_te_observations.return_value = [obs]

        # Build a valid macro df for classify_regime
        macro_df = pd.DataFrame(
            {"gdp_growth": [2.5]},
            index=pd.to_datetime([datetime.date(2023, 12, 31)]),
        )

        mock_regime = MagicMock()
        mock_regime.value = "expansion"

        momentum_group = FactorGroupType("momentum")
        tilted_weights = {momentum_group: 0.6}
        raw_tilts = {momentum_group: 1.2}

        with (
            patch(_MACRO_REPO_PATH, return_value=macro_repo),
            patch(
                f"{_MODULE}._fetch_macro_dataframe", return_value=macro_df
            ),
            patch(_CLASSIFY_REGIME, return_value=mock_regime),
            patch(_APPLY_TILTS, return_value=tilted_weights),
            patch(_GET_TILTS, return_value=raw_tilts),
        ):
            result = compute_regime_tilt(
                session=session,
                group_weights={"momentum": 0.5},
            )

        assert "regime" in result
        assert "tilted_weights" in result
        assert "tilt_multipliers" in result
        assert_json_safe(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestBuildCompositeScoresSeries:
    def test_when_multiple_factor_types_then_averages_per_ticker(self) -> None:
        from app.services.factors.factor_analysis_service import (
            _build_composite_scores_series,
        )

        scores = [
            _make_score("AAPL", factor_type="momentum", standardized_score=0.8),
            _make_score("AAPL", factor_type="value", standardized_score=0.4),
            _make_score("MSFT", factor_type="momentum", standardized_score=0.6),
        ]
        # raw_score is used when standardized_score is None; here standardized_score
        # is set so the service picks it via `s.standardized_score or s.raw_score`
        result = _build_composite_scores_series(scores)

        assert "AAPL" in result.index
        assert "MSFT" in result.index
        # AAPL: mean(0.8, 0.4) = 0.6
        assert abs(result["AAPL"] - 0.6) < 1e-9


class TestResolveBounds:
    def test_when_list_then_returns_tuple(self) -> None:
        from app.services.factors.factor_analysis_service import _resolve_bounds

        result = _resolve_bounds([0.0, 1.0])

        assert result == (0.0, 1.0)

    def test_when_dict_then_returns_dict_of_tuples(self) -> None:
        from app.services.factors.factor_analysis_service import _resolve_bounds

        result = _resolve_bounds({"quality": [0.1, 0.9], "momentum": [0.0, 1.0]})

        assert result == {"quality": (0.1, 0.9), "momentum": (0.0, 1.0)}


class TestBuildScoresWide:
    def test_when_empty_list_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_analysis_service import _build_scores_wide

        result = _build_scores_wide([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestFetchMacroDataframe:
    def test_when_no_observations_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_analysis_service import _fetch_macro_dataframe

        session = _make_session()
        macro_repo = MagicMock()
        macro_repo.get_te_observations.return_value = []

        with patch(_MACRO_REPO_PATH, return_value=macro_repo):
            result = _fetch_macro_dataframe(session)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_all_values_none_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_analysis_service import _fetch_macro_dataframe

        obs = MagicMock()
        obs.value = None
        obs.date = datetime.date(2023, 12, 31)
        obs.indicator_key = "gdp_growth"

        session = _make_session()
        macro_repo = MagicMock()
        macro_repo.get_te_observations.return_value = [obs]

        with patch(_MACRO_REPO_PATH, return_value=macro_repo):
            result = _fetch_macro_dataframe(session)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestParseGroupWeights:
    def test_when_unknown_key_then_it_is_skipped(self) -> None:
        from app.services.factors.factor_analysis_service import _parse_group_weights

        result = _parse_group_weights({"NOT_A_REAL_GROUP": 0.5})

        assert result == {}

    def test_when_valid_key_then_it_is_included(self) -> None:
        from app.services.factors.factor_analysis_service import _parse_group_weights
        from optimizer.factors import FactorGroupType

        any_group = next(iter(FactorGroupType))
        result = _parse_group_weights({any_group.value: 1.0})

        assert any_group in result
        assert result[any_group] == pytest.approx(1.0)


class TestComputeBufferZone:
    def test_when_current_members_is_none_then_all_selected_are_entered(self) -> None:
        from app.services.factors.factor_analysis_service import _compute_buffer_zone

        selected = pd.Index(["AAPL", "MSFT"])
        entered, exited = _compute_buffer_zone(selected, current_members=None)

        assert set(entered) == {"AAPL", "MSFT"}
        assert exited == []

    def test_when_current_members_empty_then_all_selected_are_entered(self) -> None:
        from app.services.factors.factor_analysis_service import _compute_buffer_zone

        selected = pd.Index(["AAPL"])
        entered, exited = _compute_buffer_zone(selected, current_members=pd.Index([]))

        assert set(entered) == {"AAPL"}
        assert exited == []

    def test_when_overlap_then_computes_diff_correctly(self) -> None:
        from app.services.factors.factor_analysis_service import _compute_buffer_zone

        selected = pd.Index(["AAPL", "GOOGL"])
        current = pd.Index(["AAPL", "MSFT"])
        entered, exited = _compute_buffer_zone(selected, current_members=current)

        assert set(entered) == {"GOOGL"}
        assert set(exited) == {"MSFT"}
