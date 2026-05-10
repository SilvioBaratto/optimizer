"""Unit tests for app.services.factors.factor_service: validate_factors() and compute_factor_scores().

Covers:
  - validate_factors(): in-sample calls run_factor_validation with correct shapes
  - validate_factors(): out-of-sample calls run_factor_oos_validation with MultiIndex shape
  - validate_factors(): persists report via FactorRepository.save_validation_report()
  - validate_factors(): raises FactorDataError when no factor scores found
  - compute_factor_scores(): EQUAL_WEIGHT calls compute_composite_score
  - compute_factor_scores(): IC_WEIGHTED fetches ic_history from DB reports
  - compute_factor_scores(): RIDGE_WEIGHTED / GBT_WEIGHTED builds training data
  - compute_factor_scores(): raises FactorDataError when no standardized scores found
  - Data reshaping: factor_scores rows → dict[str, pd.DataFrame]
  - Data reshaping: factor_scores rows → (date, ticker) MultiIndex DataFrame

All heavy dependencies (library functions, DB session) are mocked.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_MODULE = "app.services.factors.factor_scoring_service"
_RUN_VALIDATION = f"{_MODULE}.run_factor_validation"
_RUN_OOS_VALIDATION = f"{_MODULE}.run_factor_oos_validation"
_COMPUTE_COMPOSITE = f"{_MODULE}.compute_composite_score"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_session() -> MagicMock:
    return MagicMock()


def _make_factor_repo(
    scores: list | None = None, reports: list | None = None
) -> MagicMock:
    repo = MagicMock()
    repo.get_scores_by_tickers_and_date_range.return_value = scores or []
    repo.get_scores_by_tickers_at_date.return_value = scores or []
    repo.get_validation_reports.return_value = reports or []
    repo.save_validation_report.return_value = MagicMock(
        report_date=datetime.date(2024, 1, 1),
        factor_type="book_to_price",
        validation_type="in_sample",
        ic_mean=0.05,
        ic_std=0.10,
        icir=0.50,
        t_stat=2.1,
        p_value=0.04,
        vif=1.5,
        details={"significant_factors": ["book_to_price"]},
    )
    return repo


def _make_factor_score(
    ticker: str = "AAPL",
    score_date: datetime.date | None = None,
    factor_type: str = "book_to_price",
    raw_score: float = 0.5,
    standardized_score: float | None = 0.8,
) -> MagicMock:
    score = MagicMock()
    score.ticker = ticker
    score.score_date = score_date or datetime.date(2023, 6, 30)
    score.factor_type = factor_type
    score.raw_score = raw_score
    score.standardized_score = standardized_score
    score.composite_score = None
    return score


def _make_price_rows(
    tickers: list[str] | None = None,
    dates: list[datetime.date] | None = None,
) -> list[MagicMock]:
    tickers = tickers or ["AAPL", "MSFT"]
    dates = dates or [datetime.date(2023, 6, 30), datetime.date(2023, 7, 31)]
    rows = []
    for ticker in tickers:
        for dt in dates:
            row = MagicMock()
            row.ticker = ticker
            row.date = dt
            row.close = 150.0
            rows.append(row)
    return rows


def _make_validation_report_result() -> MagicMock:
    """Build a mock FactorValidationReport (library return value)."""
    ic_result = MagicMock()
    ic_result.factor_name = "book_to_price"
    ic_result.mean_ic = 0.05
    ic_result.ic_std = 0.10
    ic_result.t_stat = 2.1
    ic_result.p_value = 0.04
    ic_result.significant = True

    report = MagicMock()
    report.ic_results = [ic_result]
    report.vif_scores = pd.Series({"book_to_price": 1.5})
    report.significant_factors = ["book_to_price"]
    return report


def _make_oos_result() -> MagicMock:
    result = MagicMock()
    result.mean_oos_ic = pd.Series({"book_to_price": 0.04})
    result.mean_oos_icir = pd.Series({"book_to_price": 0.40})
    result.n_folds = 5
    result.per_fold_ic = pd.DataFrame({"book_to_price": [0.03, 0.04, 0.05, 0.04, 0.03]})
    return result


# ===========================================================================
# Import guard
# ===========================================================================


class TestFactorServiceImports:
    def test_validate_factors_importable(self) -> None:
        from app.services.factors.factor_service import validate_factors

        assert callable(validate_factors)

    def test_compute_factor_scores_importable(self) -> None:
        from app.services.factors.factor_service import compute_factor_scores

        assert callable(compute_factor_scores)

    def test_factor_data_error_importable(self) -> None:
        from app.services.factors.factor_service import FactorDataError

        assert issubclass(FactorDataError, Exception)


# ===========================================================================
# Repository helpers for factor_service
# ===========================================================================


class TestFactorRepositoryHelpers:
    """FactorRepository gains new query methods used by validate/score services."""

    def test_get_scores_by_tickers_and_date_range_exists(self) -> None:
        from app.repositories.factors.factor_repository import FactorRepository

        assert hasattr(FactorRepository, "get_scores_by_tickers_and_date_range")

    def test_get_scores_by_tickers_at_date_exists(self) -> None:
        from app.repositories.factors.factor_repository import FactorRepository

        assert hasattr(FactorRepository, "get_scores_by_tickers_at_date")


# ===========================================================================
# validate_factors — in_sample
# ===========================================================================


class TestValidateFactorsInSample:
    """validate_factors() in_sample path calls run_factor_validation() and persists."""

    def _invoke(
        self,
        tickers: list[str] | None = None,
        factor_scores: list | None = None,
        library_result=None,
        price_rows: list | None = None,
    ) -> dict:
        from app.services.factors.factor_service import validate_factors

        tickers = tickers or ["AAPL", "MSFT"]
        if factor_scores is None:
            factor_scores = [
                _make_factor_score("AAPL"),
                _make_factor_score("MSFT"),
            ]
        if library_result is None:
            library_result = _make_validation_report_result()

        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()
        price_rows = price_rows or _make_price_rows()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_VALIDATION, return_value=library_result),
            patch(f"{_MODULE}._fetch_price_rows", return_value=price_rows),
        ):
            return validate_factors(
                session=session,
                tickers=tickers,
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="in_sample",
            )

    def test_returns_dict(self) -> None:
        result = self._invoke()
        assert isinstance(result, dict)

    def test_returns_ic_mean(self) -> None:
        result = self._invoke()
        assert "ic_mean" in result

    def test_returns_t_stat(self) -> None:
        result = self._invoke()
        assert "t_stat" in result

    def test_returns_validation_type(self) -> None:
        result = self._invoke()
        assert result["validation_type"] == "in_sample"

    def test_returns_factor_type(self) -> None:
        result = self._invoke()
        assert result["factor_type"] == "book_to_price"

    def test_run_factor_validation_called(self) -> None:
        factor_scores = [_make_factor_score("AAPL"), _make_factor_score("MSFT")]
        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(
                _RUN_VALIDATION, return_value=_make_validation_report_result()
            ) as mock_val,
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import validate_factors

            validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="in_sample",
            )

        mock_val.assert_called_once()

    def test_save_validation_report_called(self) -> None:
        factor_scores = [_make_factor_score("AAPL"), _make_factor_score("MSFT")]
        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_VALIDATION, return_value=_make_validation_report_result()),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import validate_factors

            validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="in_sample",
            )

        factor_repo.save_validation_report.assert_called_once()

    def test_raises_factor_data_error_when_no_scores(self) -> None:
        from app.services.factors.factor_service import (
            FactorDataError,
            validate_factors,
        )

        factor_repo = _make_factor_repo(scores=[])
        session = _make_session()

        with patch(f"{_MODULE}.FactorRepository", return_value=factor_repo):
            with pytest.raises(FactorDataError):
                validate_factors(
                    session=session,
                    tickers=["AAPL", "MSFT"],
                    start_date=datetime.date(2020, 1, 1),
                    end_date=datetime.date(2024, 1, 1),
                    factor_type="book_to_price",
                    validation_type="in_sample",
                )

    def test_factor_scores_reshaped_to_dict_of_dataframes(self) -> None:
        """run_factor_validation receives dict[str, pd.DataFrame] (factor→dates×tickers)."""
        factor_scores = [
            _make_factor_score("AAPL", factor_type="book_to_price"),
            _make_factor_score("MSFT", factor_type="book_to_price"),
        ]
        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()
        captured_args: dict = {}

        def capture_call(factor_scores_history, returns_history, **_):
            captured_args["factor_scores_history"] = factor_scores_history
            captured_args["returns_history"] = returns_history
            return _make_validation_report_result()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_VALIDATION, side_effect=capture_call),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import validate_factors

            validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="in_sample",
            )

        fsh = captured_args["factor_scores_history"]
        assert isinstance(fsh, dict)
        assert "book_to_price" in fsh
        assert isinstance(fsh["book_to_price"], pd.DataFrame)


# ===========================================================================
# validate_factors — out_of_sample
# ===========================================================================


class TestValidateFactorsOutOfSample:
    """validate_factors() out_of_sample path calls run_factor_oos_validation()."""

    def _invoke(self, factor_scores: list | None = None) -> dict:
        from app.services.factors.factor_service import validate_factors

        if factor_scores is None:
            factor_scores = [
                _make_factor_score("AAPL"),
                _make_factor_score("MSFT"),
            ]

        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_OOS_VALIDATION, return_value=_make_oos_result()),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            return validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="out_of_sample",
            )

    def test_returns_dict(self) -> None:
        result = self._invoke()
        assert isinstance(result, dict)

    def test_returns_validation_type_out_of_sample(self) -> None:
        result = self._invoke()
        assert result["validation_type"] == "out_of_sample"

    def test_run_oos_validation_called(self) -> None:
        factor_scores = [_make_factor_score("AAPL"), _make_factor_score("MSFT")]
        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_OOS_VALIDATION, return_value=_make_oos_result()) as mock_oos,
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import validate_factors

            validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="out_of_sample",
            )

        mock_oos.assert_called_once()

    def test_oos_receives_multiindex_dataframe(self) -> None:
        """run_factor_oos_validation receives (date, ticker) MultiIndex DataFrames."""
        factor_scores = [
            _make_factor_score("AAPL", score_date=datetime.date(2023, 6, 30)),
            _make_factor_score("MSFT", score_date=datetime.date(2023, 6, 30)),
        ]
        factor_repo = _make_factor_repo(scores=factor_scores)
        session = _make_session()
        captured: dict = {}

        def capture_call(scores, returns, **_):
            captured["scores"] = scores
            captured["returns"] = returns
            return _make_oos_result()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_RUN_OOS_VALIDATION, side_effect=capture_call),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import validate_factors

            validate_factors(
                session=session,
                tickers=["AAPL", "MSFT"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                factor_type="book_to_price",
                validation_type="out_of_sample",
            )

        scores_df = captured["scores"]
        assert isinstance(scores_df, pd.DataFrame)
        assert isinstance(scores_df.index, pd.MultiIndex)


# ===========================================================================
# compute_factor_scores
# ===========================================================================


class TestComputeFactorScores:
    """compute_factor_scores() dispatches to compute_composite_score()."""

    def _make_standardized_scores(self) -> list[MagicMock]:
        return [
            _make_factor_score("AAPL", standardized_score=0.8),
            _make_factor_score("MSFT", standardized_score=0.6),
        ]

    def _invoke(
        self,
        composite_method: str = "equal_weight",
        extra_kwargs: dict | None = None,
        composite_return=None,
        factor_scores: list | None = None,
        reports: list | None = None,
    ) -> dict:
        from app.services.factors.factor_service import compute_factor_scores

        if composite_return is None:
            composite_return = pd.Series({"AAPL": 0.8, "MSFT": 0.6})
        if factor_scores is None:
            factor_scores = self._make_standardized_scores()

        factor_repo = _make_factor_repo(scores=factor_scores, reports=reports or [])
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_COMPUTE_COMPOSITE, return_value=composite_return),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            kwargs: dict = {
                "session": session,
                "tickers": ["AAPL", "MSFT"],
                "score_date": datetime.date(2024, 1, 1),
                "composite_method": composite_method,
            }
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            return compute_factor_scores(**kwargs)

    def test_returns_dict(self) -> None:
        result = self._invoke()
        assert isinstance(result, dict)

    def test_returns_scores_key(self) -> None:
        result = self._invoke()
        assert "scores" in result

    def test_returns_group_contributions_key(self) -> None:
        result = self._invoke()
        assert "group_contributions" in result

    def test_returns_score_date(self) -> None:
        result = self._invoke()
        assert "score_date" in result

    def test_compute_composite_called(self) -> None:
        factor_repo = _make_factor_repo(scores=self._make_standardized_scores())
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(
                _COMPUTE_COMPOSITE, return_value=pd.Series({"AAPL": 0.8})
            ) as mock_comp,
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import compute_factor_scores

            compute_factor_scores(
                session=session,
                tickers=["AAPL", "MSFT"],
                score_date=datetime.date(2024, 1, 1),
                composite_method="equal_weight",
            )

        mock_comp.assert_called_once()

    def test_equal_weight_does_not_fetch_reports(self) -> None:
        factor_repo = _make_factor_repo(scores=self._make_standardized_scores())
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_COMPUTE_COMPOSITE, return_value=pd.Series({"AAPL": 0.8})),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import compute_factor_scores

            compute_factor_scores(
                session=session,
                tickers=["AAPL", "MSFT"],
                score_date=datetime.date(2024, 1, 1),
                composite_method="equal_weight",
            )

        factor_repo.get_validation_reports.assert_not_called()

    def test_ic_weighted_fetches_reports(self) -> None:
        factor_repo = _make_factor_repo(scores=self._make_standardized_scores())
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_COMPUTE_COMPOSITE, return_value=pd.Series({"AAPL": 0.8})),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import compute_factor_scores

            compute_factor_scores(
                session=session,
                tickers=["AAPL", "MSFT"],
                score_date=datetime.date(2024, 1, 1),
                composite_method="ic_weighted",
            )

        factor_repo.get_validation_reports.assert_called()

    def test_icir_weighted_fetches_reports(self) -> None:
        factor_repo = _make_factor_repo(scores=self._make_standardized_scores())
        session = _make_session()

        with (
            patch(f"{_MODULE}.FactorRepository", return_value=factor_repo),
            patch(_COMPUTE_COMPOSITE, return_value=pd.Series({"AAPL": 0.8})),
            patch(f"{_MODULE}._fetch_price_rows", return_value=_make_price_rows()),
        ):
            from app.services.factors.factor_service import compute_factor_scores

            compute_factor_scores(
                session=session,
                tickers=["AAPL", "MSFT"],
                score_date=datetime.date(2024, 1, 1),
                composite_method="icir_weighted",
            )

        factor_repo.get_validation_reports.assert_called()

    def test_raises_factor_data_error_when_no_standardized_scores(self) -> None:
        from app.services.factors.factor_service import (
            FactorDataError,
            compute_factor_scores,
        )

        # Scores with no standardized_score
        scores = [_make_factor_score("AAPL", standardized_score=None)]
        factor_repo = _make_factor_repo(scores=scores)
        session = _make_session()

        with patch(f"{_MODULE}.FactorRepository", return_value=factor_repo):
            with pytest.raises(FactorDataError):
                compute_factor_scores(
                    session=session,
                    tickers=["AAPL"],
                    score_date=datetime.date(2024, 1, 1),
                    composite_method="equal_weight",
                )

    def test_raises_factor_data_error_when_no_scores_at_date(self) -> None:
        from app.services.factors.factor_service import (
            FactorDataError,
            compute_factor_scores,
        )

        factor_repo = _make_factor_repo(scores=[])
        session = _make_session()

        with patch(f"{_MODULE}.FactorRepository", return_value=factor_repo):
            with pytest.raises(FactorDataError):
                compute_factor_scores(
                    session=session,
                    tickers=["AAPL"],
                    score_date=datetime.date(2024, 1, 1),
                    composite_method="equal_weight",
                )


# ===========================================================================
# Data reshaping helpers
# ===========================================================================


class TestDataReshaping:
    """Internal reshaping functions produce correct DataFrame shapes."""

    def test_build_factor_scores_dict_returns_dict_of_dataframes(self) -> None:
        from app.services.factors.factor_service import _build_factor_scores_dict

        scores = [
            _make_factor_score(
                "AAPL",
                score_date=datetime.date(2023, 6, 30),
                factor_type="book_to_price",
            ),
            _make_factor_score(
                "MSFT",
                score_date=datetime.date(2023, 6, 30),
                factor_type="book_to_price",
            ),
            _make_factor_score(
                "AAPL", score_date=datetime.date(2023, 7, 31), factor_type="momentum"
            ),
        ]

        result = _build_factor_scores_dict(scores)

        assert isinstance(result, dict)
        assert "book_to_price" in result
        assert "momentum" in result
        assert isinstance(result["book_to_price"], pd.DataFrame)

    def test_build_factor_scores_dict_date_index(self) -> None:
        from app.services.factors.factor_service import _build_factor_scores_dict

        scores = [
            _make_factor_score("AAPL", score_date=datetime.date(2023, 6, 30)),
            _make_factor_score("MSFT", score_date=datetime.date(2023, 6, 30)),
        ]

        result = _build_factor_scores_dict(scores)
        df = result["book_to_price"]
        assert "AAPL" in df.columns or "MSFT" in df.columns

    def test_build_multiindex_dataframe_has_multiindex(self) -> None:
        from app.services.factors.factor_service import _build_multiindex_factor_df

        scores = [
            _make_factor_score("AAPL", score_date=datetime.date(2023, 6, 30)),
            _make_factor_score("MSFT", score_date=datetime.date(2023, 6, 30)),
        ]

        result = _build_multiindex_factor_df(scores)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)

    def test_build_standardized_scores_df(self) -> None:
        from app.services.factors.factor_service import _build_standardized_scores_df

        scores = [
            _make_factor_score(
                "AAPL", factor_type="book_to_price", standardized_score=0.8
            ),
            _make_factor_score("AAPL", factor_type="momentum", standardized_score=0.5),
            _make_factor_score(
                "MSFT", factor_type="book_to_price", standardized_score=0.6
            ),
        ]

        result, coverage = _build_standardized_scores_df(scores)

        assert isinstance(result, pd.DataFrame)
        assert isinstance(coverage, pd.DataFrame)
        assert "AAPL" in result.index or "book_to_price" in result.columns
