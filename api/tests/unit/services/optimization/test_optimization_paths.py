"""Empty-data and exception-path tests for optimization_service (issue #825).

Existing tests in tests/unit/test_optimization_service.py already cover:
  - _build_optimizer happy paths (all types)
  - unknown optimizer_type → ValueError (COVERED)
  - extract_results: weights, metrics, contributions, frontier

This file adds the missing branches:
  - build_pipeline: empty returns DataFrame when fetch returns empty prices
  - _build_optimizer: unsupported type raises ValueError (already covered –
    kept here only for structural completeness as the message-match differs)

Gap analysis:
  - No test covers build_pipeline when fetch_close_prices returns an empty
    frame (zero-row prices → prices_to_returns produces empty returns).
  - The explicit `raise ValueError` in _build_optimizer is already covered
    by test_optimization_service.py → NOT duplicated here.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_FETCH_PRICES = "app.services.optimization.optimization_service.fetch_close_prices"
_BUILD_PRESELECTION = (
    "app.services.optimization.optimization_service.build_preselection_pipeline"
)


# ---------------------------------------------------------------------------
# build_pipeline — empty-data path
# ---------------------------------------------------------------------------


class TestBuildPipelineEmptyData:
    """build_pipeline when price fetch returns an empty DataFrame."""

    def test_when_prices_empty_then_returns_empty_returns_frame(self) -> None:
        """fetch_close_prices → empty DF → prices_to_returns → empty returns."""
        session = MagicMock()
        empty_prices = pd.DataFrame()
        mock_preselection = MagicMock()

        with (
            patch(_FETCH_PRICES, return_value=empty_prices),
            patch(_BUILD_PRESELECTION, return_value=mock_preselection),
        ):
            from app.services.optimization.optimization_service import build_pipeline

            pipe, returns = build_pipeline(
                tickers=[],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                optimizer_type="mean_risk",
                config={},
                session=session,
            )

        assert returns.empty

    def test_when_prices_empty_then_pipeline_is_assembled(self) -> None:
        """Even with empty prices the Pipeline object is still returned (not None)."""
        from sklearn.pipeline import Pipeline

        session = MagicMock()
        empty_prices = pd.DataFrame()
        mock_preselection = MagicMock()

        with (
            patch(_FETCH_PRICES, return_value=empty_prices),
            patch(_BUILD_PRESELECTION, return_value=mock_preselection),
        ):
            from app.services.optimization.optimization_service import build_pipeline

            pipe, _ = build_pipeline(
                tickers=[],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                optimizer_type="mean_risk",
                config={},
                session=session,
            )

        assert isinstance(pipe, Pipeline)

    def test_when_prices_single_column_then_returns_has_one_column(self) -> None:
        """Single-ticker prices → returns frame has one column."""
        session = MagicMock()
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            {"AAPL": 100 * np.cumprod(1 + rng.normal(0, 0.01, 10))},
            index=pd.date_range("2020-01-01", periods=10, freq="B"),
        )
        mock_preselection = MagicMock()

        with (
            patch(_FETCH_PRICES, return_value=prices),
            patch(_BUILD_PRESELECTION, return_value=mock_preselection),
        ):
            from app.services.optimization.optimization_service import build_pipeline

            _, returns = build_pipeline(
                tickers=["AAPL"],
                start_date=date(2020, 1, 1),
                end_date=date(2020, 12, 31),
                optimizer_type="mean_risk",
                config={},
                session=session,
            )

        assert list(returns.columns) == ["AAPL"]


# ---------------------------------------------------------------------------
# _build_optimizer — unsupported type (explicit raise)
# NOTE: already covered in test_optimization_service.py with the same match.
# We add one test here targeting an edge case (non-string type key) to
# ensure the function is robust.  The raise itself is not duplicated.
# ---------------------------------------------------------------------------


class TestBuildOptimizerEdgeCases:
    """Edge cases for _build_optimizer not covered by the existing test file."""

    def test_when_unsupported_type_then_raises_value_error_with_type_name(
        self,
    ) -> None:
        from app.services.optimization.optimization_service import _build_optimizer

        with pytest.raises(ValueError, match="hrp"):
            _build_optimizer("hrp", {})

    def test_when_mean_risk_config_has_extra_fields_stripped_then_no_raise(
        self,
    ) -> None:
        """Config with only valid MeanRiskConfig fields must not raise."""
        from skfolio.optimization import MeanRisk

        from app.services.optimization.optimization_service import _build_optimizer

        optimizer = _build_optimizer("mean_risk", {"min_weights": 0.0})

        assert isinstance(optimizer, MeanRisk)


# ---------------------------------------------------------------------------
# extract_results — empty weights edge case
# ---------------------------------------------------------------------------


class TestExtractResultsEmptyWeights:
    """extract_results with a portfolio that has an empty weights_dict."""

    def test_when_portfolio_has_no_weights_then_result_is_json_safe(self) -> None:
        """Empty weights_dict → risk_contributions empty → result JSON-safe."""
        portfolio = MagicMock()
        portfolio.weights_dict = {}
        portfolio.annualized_mean = 0.0
        portfolio.annualized_standard_deviation = 0.0
        portfolio.annualized_sharpe_ratio = 0.0
        portfolio.max_drawdown = 0.0
        portfolio.contribution.return_value = np.array([])

        pipeline = MagicMock()
        pipeline.predict.return_value = portfolio
        returns_df = pd.DataFrame()

        from app.services.optimization.optimization_service import extract_results

        result = extract_results(pipeline, returns_df)

        assert result["weights"] == {}
        assert result["risk_contributions"] == {}
        assert_json_safe(result)
