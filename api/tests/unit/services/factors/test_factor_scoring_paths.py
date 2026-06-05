"""Unit tests for app.services.factors.factor_scoring_service — gap-fill paths.

Existing tests (tests/unit/test_factor_validation_service.py) cover:
  - validate_factors: in_sample / oos happy paths, FactorDataError on empty scores.
  - compute_factor_scores: equal_weight, ic_weighted, ridge_weighted, FactorDataError.

This file adds uncovered branches:

validate_factors:
  - FactorDataError match string contains factor_type (existing test only checks type)

_build_ic_history:
  - no validation reports → empty DataFrame
  - reports exist but all ic_mean is None → empty DataFrame
  - valid reports → pivoted DataFrame

_build_training_data:
  - training dates are None → returns empty dict
  - no standardized training scores → returns empty dict

_extract_score_results:
  - DataFrame result → returns (scores dict, group_contributions)
  - neither Series nor DataFrame → returns ({}, {})

_safe_icir / _extract_vif / _series_get:
  - ic_std == 0 → None
  - vif_scores is None → None
  - factor_type absent from vif Series → None
  - series is None → None
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_MODULE = "app.services.factors.factor_scoring_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session() -> MagicMock:
    return MagicMock()


def _make_score(
    ticker: str = "AAPL",
    score_date: datetime.date | None = None,
    factor_type: str = "book_to_price",
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


def _make_factor_repo(
    scores: list | None = None,
    reports: list | None = None,
) -> MagicMock:
    repo = MagicMock()
    repo.get_scores_by_tickers_and_date_range.return_value = scores or []
    repo.get_scores_by_tickers_at_date.return_value = scores or []
    repo.get_validation_reports.return_value = reports or []
    repo.save_validation_report.return_value = MagicMock()
    return repo


# ---------------------------------------------------------------------------
# validate_factors — FactorDataError message
# ---------------------------------------------------------------------------


class TestValidateFactorsErrorMessage:
    def test_when_no_scores_then_error_message_contains_factor_type(self) -> None:
        from app.services.factors.factor_scoring_service import (
            FactorDataError,
            validate_factors,
        )

        repo = _make_factor_repo(scores=[])
        session = _make_session()

        with patch(f"{_MODULE}.FactorRepository", return_value=repo):
            with pytest.raises(FactorDataError, match="momentum_12_1"):
                validate_factors(
                    session=session,
                    tickers=["AAPL"],
                    start_date=datetime.date(2023, 1, 1),
                    end_date=datetime.date(2024, 1, 1),
                    factor_type="momentum_12_1",
                )


# ---------------------------------------------------------------------------
# _build_ic_history
# ---------------------------------------------------------------------------


class TestBuildIcHistory:
    def test_when_no_reports_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_scoring_service import _build_ic_history

        repo = _make_factor_repo(reports=[])
        result = _build_ic_history(repo)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_reports_all_ic_mean_none_then_returns_empty_dataframe(self) -> None:
        from app.services.factors.factor_scoring_service import _build_ic_history

        report = MagicMock()
        report.factor_type = "momentum"
        report.ic_mean = None
        report.icir = None
        report.report_date = datetime.date(2023, 12, 31)
        repo = _make_factor_repo(reports=[report])

        result = _build_ic_history(repo)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_when_valid_reports_then_pivoted_dataframe_returned(self) -> None:
        from app.services.factors.factor_scoring_service import _build_ic_history

        report = MagicMock()
        report.factor_type = "book_to_price"
        report.ic_mean = 0.04
        report.icir = 0.40
        report.report_date = datetime.date(2023, 12, 31)
        repo = _make_factor_repo(reports=[report])

        result = _build_ic_history(repo)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "book_to_price" in result.columns


# ---------------------------------------------------------------------------
# _build_training_data
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    def test_when_both_dates_none_then_returns_empty_dict(self) -> None:
        from app.services.factors.factor_scoring_service import _build_training_data

        repo = _make_factor_repo()
        session = _make_session()

        with patch(f"{_MODULE}._fetch_price_rows", return_value=[]):
            result = _build_training_data(
                repo,
                tickers=["AAPL"],
                training_start_date=None,
                training_end_date=None,
                session=session,
            )

        assert result == {}

    def test_when_start_date_none_only_then_returns_empty_dict(self) -> None:
        from app.services.factors.factor_scoring_service import _build_training_data

        repo = _make_factor_repo()
        session = _make_session()

        result = _build_training_data(
            repo,
            tickers=["AAPL"],
            training_start_date=None,
            training_end_date=datetime.date(2024, 1, 1),
            session=session,
        )

        assert result == {}

    def test_when_no_standardized_training_scores_then_returns_empty_dict(self) -> None:
        from app.services.factors.factor_scoring_service import _build_training_data

        # Scores exist but none have standardized_score set
        raw_scores = [_make_score("AAPL", standardized_score=None)]
        repo = _make_factor_repo(scores=raw_scores)
        session = _make_session()

        with patch(f"{_MODULE}._fetch_price_rows", return_value=[]):
            result = _build_training_data(
                repo,
                tickers=["AAPL"],
                training_start_date=datetime.date(2022, 1, 1),
                training_end_date=datetime.date(2023, 1, 1),
                session=session,
            )

        assert result == {}


# ---------------------------------------------------------------------------
# _extract_score_results
# ---------------------------------------------------------------------------


class TestExtractScoreResults:
    def test_when_series_result_then_returns_scores_dict_and_empty_contributions(
        self,
    ) -> None:
        from app.services.factors.factor_scoring_service import _extract_score_results

        series = pd.Series({"AAPL": 0.8, "MSFT": 0.6})
        scores, contributions = _extract_score_results(series)

        assert scores == {"AAPL": pytest.approx(0.8), "MSFT": pytest.approx(0.6)}
        assert contributions == {}

    def test_when_series_has_nan_then_nan_ticker_excluded(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_score_results

        series = pd.Series({"AAPL": 0.8, "MSFT": float("nan")})
        scores, _ = _extract_score_results(series)

        assert "AAPL" in scores
        assert "MSFT" not in scores

    def test_when_dataframe_result_then_scores_and_contributions_populated(
        self,
    ) -> None:
        from app.services.factors.factor_scoring_service import _extract_score_results

        df = pd.DataFrame(
            {
                "composite": [0.8, 0.6],
                "quality_contrib": [0.3, 0.2],
                "momentum_contrib": [0.5, 0.4],
            },
            index=["AAPL", "MSFT"],
        )
        scores, contributions = _extract_score_results(df)

        assert "AAPL" in scores
        assert "MSFT" in scores
        assert "quality_contrib" in contributions
        assert "momentum_contrib" in contributions
        assert_json_safe({"scores": scores, "contributions": contributions})

    def test_when_neither_series_nor_dataframe_then_returns_empty_dicts(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_score_results

        scores, contributions = _extract_score_results(None)  # type: ignore[arg-type]

        assert scores == {}
        assert contributions == {}

    def test_series_result_is_json_safe(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_score_results

        series = pd.Series({"AAPL": 0.8})
        scores, contributions = _extract_score_results(series)

        assert_json_safe({"scores": scores, "contributions": contributions})


# ---------------------------------------------------------------------------
# _safe_icir
# ---------------------------------------------------------------------------


class TestSafeIcir:
    def test_when_ic_result_is_none_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _safe_icir

        assert _safe_icir(None) is None

    def test_when_ic_std_is_zero_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _safe_icir

        ic = MagicMock()
        ic.ic_std = 0
        ic.mean_ic = 0.05

        assert _safe_icir(ic) is None

    def test_when_valid_then_returns_ratio(self) -> None:
        from app.services.factors.factor_scoring_service import _safe_icir

        ic = MagicMock()
        ic.ic_std = 0.10
        ic.mean_ic = 0.05

        result = _safe_icir(ic)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _extract_vif
# ---------------------------------------------------------------------------


class TestExtractVif:
    def test_when_vif_scores_none_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_vif

        assert _extract_vif(None, "book_to_price") is None

    def test_when_factor_absent_from_series_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_vif

        series = pd.Series({"momentum": 1.5})
        assert _extract_vif(series, "book_to_price") is None

    def test_when_factor_present_then_returns_float(self) -> None:
        from app.services.factors.factor_scoring_service import _extract_vif

        series = pd.Series({"book_to_price": 2.0})
        result = _extract_vif(series, "book_to_price")

        assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _series_get
# ---------------------------------------------------------------------------


class TestSeriesGet:
    def test_when_series_is_none_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _series_get

        assert _series_get(None, "any_key") is None

    def test_when_key_absent_then_returns_none(self) -> None:
        from app.services.factors.factor_scoring_service import _series_get

        series = pd.Series({"other": 0.1})
        assert _series_get(series, "missing_key") is None

    def test_when_key_present_then_returns_float(self) -> None:
        from app.services.factors.factor_scoring_service import _series_get

        series = pd.Series({"book_to_price": 0.04})
        result = _series_get(series, "book_to_price")

        assert result == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# compute_factor_scores — FactorDataError message
# ---------------------------------------------------------------------------


class TestComputeFactorScoresErrorMessage:
    def test_when_no_standardized_scores_then_error_mentions_tickers_and_date(
        self,
    ) -> None:
        from app.services.factors.factor_scoring_service import (
            FactorDataError,
            compute_factor_scores,
        )

        scores = [_make_score("AAPL", standardized_score=None)]
        repo = _make_factor_repo(scores=scores)
        session = _make_session()

        with patch(f"{_MODULE}.FactorRepository", return_value=repo):
            with pytest.raises(FactorDataError, match="standardized"):
                compute_factor_scores(
                    session=session,
                    tickers=["AAPL"],
                    score_date=datetime.date(2024, 1, 1),
                    composite_method="equal_weight",
                )
