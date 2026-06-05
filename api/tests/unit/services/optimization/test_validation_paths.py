"""Empty-data and exception-path tests for validation_service (issue #825).

No existing unit tests cover validation_service directly (only route-level
mocks exist).

Gap analysis:
  - build_cv: unknown cv_type → explicit raise ValueError (1 raise)
  - build_cv: known cv_types build without raising (walk_forward, cpcv,
    multiple_randomized)
  - fetch_returns: empty prices → empty returns DataFrame
  - serialize_population: empty iterable → empty list
  - serialize_population: non-iterable population (TypeError path)
  - _compute_aggregate_score: no sharpe values → returns 0.0
  - run_validation: on_progress called when provided
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_FETCH_PRICES = "app.services.optimization.validation_service.fetch_close_prices"
_BUILD_OPTIMIZER = "app.services.optimization.validation_service._build_optimizer"
_RUN_CROSS_VAL = "app.services.optimization.validation_service.run_cross_val"


# ---------------------------------------------------------------------------
# build_cv — explicit raise for unknown cv_type
# ---------------------------------------------------------------------------


class TestBuildCvUnknownType:
    """build_cv raises ValueError for unregistered cv_type strings."""

    def test_when_unknown_cv_type_then_raises_value_error(self) -> None:
        from app.services.optimization.validation_service import build_cv

        with pytest.raises(ValueError, match="Unknown cv_type"):
            build_cv("does_not_exist", {})

    def test_when_unknown_cv_type_then_message_contains_valid_types(self) -> None:
        from app.services.optimization.validation_service import build_cv

        with pytest.raises(ValueError, match="walk_forward"):
            build_cv("bad_type", {})


class TestBuildCvKnownTypes:
    """build_cv succeeds for every registered cv_type with default config."""

    @pytest.mark.parametrize(
        "cv_type",
        ["walk_forward", "cpcv", "multiple_randomized"],
    )
    def test_when_known_cv_type_then_returns_object_without_raising(
        self, cv_type: str
    ) -> None:
        from app.services.optimization.validation_service import build_cv

        cv = build_cv(cv_type, {})

        assert cv is not None


# ---------------------------------------------------------------------------
# fetch_returns — empty prices path
# ---------------------------------------------------------------------------


class TestFetchReturnsEmptyPrices:
    """fetch_returns with empty price data returns an empty DataFrame."""

    def test_when_prices_empty_then_returns_empty_dataframe(self) -> None:
        session = MagicMock()
        empty_prices = pd.DataFrame()

        with patch(_FETCH_PRICES, return_value=empty_prices):
            from app.services.optimization.validation_service import fetch_returns

            returns = fetch_returns(
                tickers=[],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                session=session,
            )

        assert returns.empty


# ---------------------------------------------------------------------------
# serialize_population — empty iterable / TypeError path
# ---------------------------------------------------------------------------


class TestSerializePopulation:
    """serialize_population handles empty and non-iterable inputs gracefully."""

    def test_when_population_empty_then_returns_empty_list(self) -> None:
        from app.services.optimization.validation_service import serialize_population

        result = serialize_population([])

        assert result == []

    def test_when_population_non_iterable_then_wraps_in_single_fold(self) -> None:
        """TypeError from iterating → falls back to extracting a single Portfolio."""
        portfolio = MagicMock()
        portfolio.weights_dict = {"AAPL": 0.5, "MSFT": 0.5}
        portfolio.summary.return_value = {"sharpe": 1.2}

        # Raise TypeError when iteration is attempted (non-iterable path)
        non_iterable = MagicMock()
        non_iterable.__iter__ = MagicMock(side_effect=TypeError("not iterable"))
        non_iterable.weights_dict = portfolio.weights_dict
        non_iterable.summary = portfolio.summary

        from app.services.optimization.validation_service import serialize_population

        result = serialize_population(non_iterable)

        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "weights" in result[0]

    def test_when_population_has_portfolios_then_each_fold_has_weights_and_measures(
        self,
    ) -> None:
        p = MagicMock()
        p.weights_dict = {"AAPL": 1.0}
        p.summary.return_value = {"annualized_sharpe_ratio": 0.9}

        from app.services.optimization.validation_service import serialize_population

        result = serialize_population([p])

        assert len(result) == 1
        assert "weights" in result[0]
        assert "measures" in result[0]
        assert_json_safe(result)

    def test_when_portfolio_weights_dict_raises_then_weights_is_empty(self) -> None:
        """_extract_portfolio catches exceptions on weights_dict access."""
        bad_portfolio = MagicMock()
        bad_portfolio.weights_dict = MagicMock(
            side_effect=Exception("unavailable"),
            items=MagicMock(side_effect=Exception("unavailable")),
        )
        # Make weights_dict itself raise when .items() is called
        broken = MagicMock()
        type(broken).weights_dict = property(lambda self: (_ for _ in ()).throw(Exception("broken")))  # type: ignore[assignment]
        broken.summary = MagicMock(side_effect=Exception("broken summary"))

        from app.services.optimization.validation_service import _extract_portfolio

        fold = _extract_portfolio(broken)

        assert fold["weights"] == {}
        assert fold["measures"] == {}


# ---------------------------------------------------------------------------
# _compute_aggregate_score — no valid sharpe values → 0.0
# ---------------------------------------------------------------------------


class TestComputeAggregateScore:
    """_compute_aggregate_score returns 0.0 when no portfolios have valid sharpe."""

    def test_when_population_empty_then_score_is_zero(self) -> None:
        from app.services.optimization.validation_service import (
            _compute_aggregate_score,
        )

        score = _compute_aggregate_score([])

        assert score == 0.0

    def test_when_all_sharpe_values_raise_then_score_is_zero(self) -> None:
        bad_portfolio = MagicMock()
        type(bad_portfolio).annualized_sharpe_ratio = property(  # type: ignore[assignment]
            lambda self: (_ for _ in ()).throw(Exception("nan"))
        )

        from app.services.optimization.validation_service import (
            _compute_aggregate_score,
        )

        score = _compute_aggregate_score([bad_portfolio])

        assert score == 0.0

    def test_when_population_has_valid_sharpes_then_score_is_mean(self) -> None:
        p1, p2 = MagicMock(), MagicMock()
        p1.annualized_sharpe_ratio = 1.0
        p2.annualized_sharpe_ratio = 2.0

        from app.services.optimization.validation_service import (
            _compute_aggregate_score,
        )

        score = _compute_aggregate_score([p1, p2])

        assert score == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# run_validation — on_progress callback
# ---------------------------------------------------------------------------


class TestRunValidationProgress:
    """run_validation invokes on_progress after fold serialisation."""

    def test_when_on_progress_provided_then_called_with_fold_counts(self) -> None:
        session = MagicMock()
        progress_calls: list[dict] = []

        def spy(**kwargs: object) -> None:
            progress_calls.append(dict(kwargs))

        mock_population = [MagicMock()]
        mock_population[0].weights_dict = {"AAPL": 1.0}
        mock_population[0].summary.return_value = {}
        mock_population[0].annualized_sharpe_ratio = 1.0

        import numpy as np

        prices = pd.DataFrame(
            {"AAPL": 100 * (1 + np.zeros(10))},
            index=pd.date_range("2020-01-01", periods=10, freq="B"),
        )

        with (
            patch(_FETCH_PRICES, return_value=prices),
            patch(_BUILD_OPTIMIZER, return_value=MagicMock()),
            patch(_RUN_CROSS_VAL, return_value=mock_population),
        ):
            from app.services.optimization.validation_service import run_validation

            run_validation(
                tickers=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                cv_type="walk_forward",
                cv_config={},
                optimizer_type="mean_risk",
                optimizer_config={},
                session=session,
                on_progress=spy,
            )

        assert len(progress_calls) == 1
        assert "current_fold" in progress_calls[0]
        assert "total_folds" in progress_calls[0]
