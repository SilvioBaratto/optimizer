"""Empty-data and exception-path tests for entropy_pooling_service (issue #825).

Covers service-layer gaps NOT already tested in tests/unit/test_entropy_pooling.py:
  - run_entropy_pooling with empty DataFrame raises (propagated from optimizer)
  - fetch_prices_df with None start/end dates (both branches)
  - result of run_entropy_pooling is json-safe on success
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

_BUILD_EP = "app.services.views.entropy_pooling_service.build_entropy_pooling"
_FETCH_CLOSE = "app.services.views.entropy_pooling_service.fetch_close_prices"


def _make_returns(n_rows: int = 300, n_cols: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-01", periods=n_rows, freq="B")
    cols = [f"T{i}" for i in range(n_cols)]
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)) * 0.01, index=dates, columns=cols)


def _mock_estimator(n: int = 3) -> MagicMock:
    est = MagicMock()
    dist = MagicMock()
    dist.mu = np.zeros(n)
    dist.covariance = np.eye(n) * 1e-4
    est.return_distribution_ = dist
    return est


# ---------------------------------------------------------------------------
# run_entropy_pooling — empty DataFrame
# ---------------------------------------------------------------------------


class TestWhenEmptyDataFrameThenRaisesFromEstimator:
    def test_when_empty_returns_df_then_optimizer_error_propagates(self) -> None:
        """Empty DataFrame has no columns — estimator fit raises; service propagates."""
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        empty_df = pd.DataFrame()

        with patch(_BUILD_EP) as mock_build:
            mock_est = MagicMock()
            mock_est.fit.side_effect = ValueError("DataFrame has no columns")
            mock_build.return_value = mock_est

            with pytest.raises(ValueError, match="columns"):
                run_entropy_pooling(empty_df, mean_views=("A == 0.001",))

    def test_when_empty_returns_df_then_fit_called_with_it(self) -> None:
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        empty_df = pd.DataFrame()

        with patch(_BUILD_EP) as mock_build:
            mock_est = MagicMock()
            mock_est.fit.side_effect = ValueError("no columns")
            mock_build.return_value = mock_est

            with contextlib.suppress(ValueError):
                run_entropy_pooling(empty_df, mean_views=("A == 0.001",))

            mock_est.fit.assert_called_once_with(empty_df)


# ---------------------------------------------------------------------------
# run_entropy_pooling — RuntimeError from solver propagates
# ---------------------------------------------------------------------------


class TestWhenSolverFailsThenRuntimeErrorPropagates:
    def test_when_solver_raises_runtime_error_then_propagated(self) -> None:
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        returns = _make_returns()

        with patch(_BUILD_EP) as mock_build:
            mock_est = MagicMock()
            mock_est.fit.side_effect = RuntimeError("Solver did not converge")
            mock_build.return_value = mock_est

            with pytest.raises(RuntimeError, match="converge"):
                run_entropy_pooling(returns, mean_views=("T0 == 0.001",))


# ---------------------------------------------------------------------------
# run_entropy_pooling — successful result is json-safe
# ---------------------------------------------------------------------------


class TestWhenSuccessfulThenResultIsJsonSafe:
    def test_when_valid_returns_df_then_result_json_safe(self) -> None:
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        returns = _make_returns(n_rows=300, n_cols=3)

        with patch(_BUILD_EP, return_value=_mock_estimator(3)):
            result = run_entropy_pooling(returns, mean_views=("T0 == 0.001",))

        assert_json_safe(result)

    def test_when_valid_returns_df_then_tickers_match_columns(self) -> None:
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        returns = _make_returns(n_rows=300, n_cols=3)

        with patch(_BUILD_EP, return_value=_mock_estimator(3)):
            result = run_entropy_pooling(returns, mean_views=("T0 == 0.001",))

        assert result["tickers"] == list(returns.columns)

    def test_when_all_views_none_then_config_still_built(self) -> None:
        """All view keyword args None → EntropyPoolingConfig with all-None fields."""
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        returns = _make_returns()

        with patch(_BUILD_EP, return_value=_mock_estimator(3)):
            result = run_entropy_pooling(returns)  # no views at all

        assert "mu" in result
        assert "covariance" in result
        assert "tickers" in result


# ---------------------------------------------------------------------------
# fetch_prices_df — None date arguments (both branches exercised)
# ---------------------------------------------------------------------------


class TestFetchPricesDfDateBranches:
    def test_when_start_and_end_none_then_fetch_close_called_with_none_dates(
        self,
    ) -> None:
        from app.services.views.entropy_pooling_service import fetch_prices_df

        mock_session = MagicMock()
        fake_prices = pd.DataFrame({"T0": [1.0, 2.0]})

        with patch(_FETCH_CLOSE, return_value=fake_prices) as mock_fetch:
            fetch_prices_df(["T0"], None, None, mock_session)

        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args.kwargs
        assert call_kwargs["start_date"] is None
        assert call_kwargs["end_date"] is None

    def test_when_start_and_end_provided_then_parsed_as_date_objects(self) -> None:
        from datetime import date

        from app.services.views.entropy_pooling_service import fetch_prices_df

        mock_session = MagicMock()
        fake_prices = pd.DataFrame({"T0": [1.0, 2.0]})

        with patch(_FETCH_CLOSE, return_value=fake_prices) as mock_fetch:
            fetch_prices_df(["T0"], "2023-01-01", "2023-12-31", mock_session)

        call_kwargs = mock_fetch.call_args.kwargs
        assert call_kwargs["start_date"] == date(2023, 1, 1)
        assert call_kwargs["end_date"] == date(2023, 12, 31)

    def test_when_only_start_provided_then_end_is_none(self) -> None:
        from datetime import date

        from app.services.views.entropy_pooling_service import fetch_prices_df

        mock_session = MagicMock()
        fake_prices = pd.DataFrame({"T0": [1.0]})

        with patch(_FETCH_CLOSE, return_value=fake_prices) as mock_fetch:
            fetch_prices_df(["T0"], "2023-06-01", None, mock_session)

        call_kwargs = mock_fetch.call_args.kwargs
        assert call_kwargs["start_date"] == date(2023, 6, 1)
        assert call_kwargs["end_date"] is None
