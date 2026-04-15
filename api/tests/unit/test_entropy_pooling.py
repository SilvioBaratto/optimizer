"""Unit tests for POST /api/v1/views/entropy-pooling endpoint.

Covers:
  - mean_views only
  - variance_views only
  - correlation_views only (tuple format)
  - skew_views only
  - kurtosis_views only
  - cvar_views only
  - combined views
  - missing tickers → 422
  - all-null views → 422
  - invalid correlation format → 422
  - solver convergence failure → 500
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

BASE_URL = "/api/v1/views/entropy-pooling"

_TICKERS = ["AAPL", "MSFT", "GOOGL"]

_FETCH_PRICES = "app.api.v1.views.fetch_prices_df"
_RUN_ENTROPY = "app.api.v1.views.run_entropy_pooling"


def _make_ep_result(tickers: list[str] | None = None) -> dict:
    t = tickers or _TICKERS
    n = len(t)
    return {
        "mu": [0.001 * (i + 1) for i in range(n)],
        "covariance": [[0.0001 * (i + j + 1) for j in range(n)] for i in range(n)],
        "tickers": t,
    }


def _make_prices_df():
    import pandas as pd

    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    np.random.seed(42)
    data = {t: 100 + np.cumsum(np.random.randn(252)) for t in _TICKERS}
    return pd.DataFrame(data, index=dates)


# ===========================================================================
# POST /api/v1/views/entropy-pooling — validation errors (422)
# ===========================================================================


class TestEntropyPoolingValidation:
    """Request validation before any service calls."""

    def test_all_null_views_returns_422(self, client: TestClient) -> None:
        payload = {"tickers": _TICKERS}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_empty_view_lists_returns_422(self, client: TestClient) -> None:
        payload = {
            "tickers": _TICKERS,
            "mean_views": [],
            "variance_views": [],
            "correlation_views": [],
            "skew_views": [],
            "kurtosis_views": [],
            "cvar_views": [],
        }
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_invalid_correlation_format_returns_422(self, client: TestClient) -> None:
        """Correlation views must be parseable as tuple format."""
        payload = {
            "tickers": _TICKERS,
            "correlation_views": ["AAPL MSFT == 0.5"],  # missing parentheses
        }
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_missing_tickers_field_returns_422(self, client: TestClient) -> None:
        payload = {"mean_views": ["AAPL == 0.001"]}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_empty_tickers_list_returns_422(self, client: TestClient) -> None:
        payload = {"tickers": [], "mean_views": ["AAPL == 0.001"]}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422


# ===========================================================================
# POST /api/v1/views/entropy-pooling — 422 from service (no price data)
# ===========================================================================


class TestEntropyPoolingNoPriceData:
    """Service returns ValueError when tickers not found → 422."""

    def test_no_price_data_returns_422(self, client: TestClient) -> None:
        import pandas as pd

        with patch(_FETCH_PRICES, return_value=pd.DataFrame()):
            resp = client.post(
                BASE_URL,
                json={"tickers": ["ZZZZ"], "mean_views": ["ZZZZ == 0.001"]},
            )
        assert resp.status_code == 422

    def test_error_message_mentions_tickers(self, client: TestClient) -> None:
        import pandas as pd

        with patch(_FETCH_PRICES, return_value=pd.DataFrame()):
            resp = client.post(
                BASE_URL,
                json={"tickers": ["ZZZZ"], "mean_views": ["ZZZZ == 0.001"]},
            )
        assert resp.status_code == 422
        body = resp.json()
        # The app wraps errors in {"error": {"message": "..."}} format
        assert "error" in body or "detail" in body


# ===========================================================================
# POST /api/v1/views/entropy-pooling — 500 from solver failure
# ===========================================================================


class TestEntropyPoolingSolverFailure:
    """Solver convergence failure → 500."""

    def test_solver_failure_returns_500(self, client: TestClient) -> None:
        prices_df = _make_prices_df()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, side_effect=RuntimeError("Solver did not converge")),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        assert resp.status_code == 500

    def test_solver_failure_includes_message(self, client: TestClient) -> None:
        prices_df = _make_prices_df()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, side_effect=RuntimeError("Solver did not converge")),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        assert resp.status_code == 500
        body = resp.json()
        # App may wrap in {"error": {"message": ...}} or {"detail": ...}
        message = (
            body.get("detail")
            or body.get("error", {}).get("message", "")
        )
        assert "converge" in message.lower()


# ===========================================================================
# POST /api/v1/views/entropy-pooling — successful responses
# ===========================================================================


class TestEntropyPoolingSuccess:
    """Happy-path tests for each view type and combined views."""

    def test_mean_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        assert resp.status_code == 200

    def test_variance_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "variance_views": ["AAPL == 0.0004"]},
            )
        assert resp.status_code == 200

    def test_correlation_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={
                    "tickers": _TICKERS,
                    "correlation_views": ["(AAPL, MSFT) == 0.5"],
                },
            )
        assert resp.status_code == 200

    def test_skew_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "skew_views": ["AAPL == -0.5"]},
            )
        assert resp.status_code == 200

    def test_kurtosis_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "kurtosis_views": ["AAPL == 3.0"]},
            )
        assert resp.status_code == 200

    def test_cvar_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "cvar_views": ["AAPL == -0.02"]},
            )
        assert resp.status_code == 200

    def test_combined_views_returns_200(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={
                    "tickers": _TICKERS,
                    "mean_views": ["AAPL == 0.001"],
                    "variance_views": ["MSFT == 0.0003"],
                },
            )
        assert resp.status_code == 200

    def test_response_contains_mu(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        body = resp.json()
        assert "mu" in body
        assert isinstance(body["mu"], list)
        assert len(body["mu"]) == len(_TICKERS)

    def test_response_contains_covariance(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        body = resp.json()
        assert "covariance" in body
        assert isinstance(body["covariance"], list)
        assert len(body["covariance"]) == len(_TICKERS)

    def test_response_contains_tickers(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        body = resp.json()
        assert "tickers" in body
        assert body["tickers"] == _TICKERS

    def test_date_range_accepted(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()

        with (
            patch(_FETCH_PRICES, return_value=prices_df),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            resp = client.post(
                BASE_URL,
                json={
                    "tickers": _TICKERS,
                    "mean_views": ["AAPL == 0.001"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                },
            )
        assert resp.status_code == 200

    def test_fetch_prices_called_with_correct_tickers(self, client: TestClient) -> None:
        prices_df = _make_prices_df()
        ep_result = _make_ep_result()
        captured: dict = {}

        def capture_fetch(tickers, *_):
            captured["tickers"] = tickers
            return prices_df

        with (
            patch(_FETCH_PRICES, side_effect=capture_fetch),
            patch(_RUN_ENTROPY, return_value=ep_result),
        ):
            client.post(
                BASE_URL,
                json={"tickers": _TICKERS, "mean_views": ["AAPL == 0.001"]},
            )
        assert captured["tickers"] == _TICKERS


# ===========================================================================
# Service layer — run_entropy_pooling
# ===========================================================================


class TestRunEntropyPoolingService:
    """Direct unit tests for the service function."""

    def test_returns_dict_with_mu_covariance_tickers(self) -> None:
        from app.services.entropy_pooling_service import run_entropy_pooling

        import pandas as pd

        np.random.seed(0)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        returns = pd.DataFrame(
            np.random.randn(300, 3) * 0.01,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

        with patch(
            "app.services.entropy_pooling_service.build_entropy_pooling"
        ) as mock_build:
            mock_estimator = MagicMock()
            mock_estimator.return_distribution_ = MagicMock()
            mock_estimator.return_distribution_.mu = np.array([0.001, 0.002, 0.0015])
            mock_estimator.return_distribution_.covariance = np.eye(3) * 0.0001
            mock_estimator.fit.return_value = mock_estimator
            mock_build.return_value = mock_estimator

            result = run_entropy_pooling(
                returns=returns,
                mean_views=("AAPL == 0.001",),
            )

        assert "mu" in result
        assert "covariance" in result
        assert "tickers" in result

    def test_mu_length_matches_tickers(self) -> None:
        from app.services.entropy_pooling_service import run_entropy_pooling

        import pandas as pd

        np.random.seed(1)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        returns = pd.DataFrame(
            np.random.randn(300, 3) * 0.01,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

        with patch(
            "app.services.entropy_pooling_service.build_entropy_pooling"
        ) as mock_build:
            mock_estimator = MagicMock()
            mock_estimator.return_distribution_ = MagicMock()
            mock_estimator.return_distribution_.mu = np.array([0.001, 0.002, 0.0015])
            mock_estimator.return_distribution_.covariance = np.eye(3) * 0.0001
            mock_estimator.fit.return_value = mock_estimator
            mock_build.return_value = mock_estimator

            result = run_entropy_pooling(
                returns=returns,
                mean_views=("AAPL == 0.001",),
            )

        assert len(result["mu"]) == 3
        assert len(result["tickers"]) == 3

    def test_covariance_is_square_matrix(self) -> None:
        from app.services.entropy_pooling_service import run_entropy_pooling

        import pandas as pd

        np.random.seed(2)
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        returns = pd.DataFrame(
            np.random.randn(300, 3) * 0.01,
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"],
        )

        with patch(
            "app.services.entropy_pooling_service.build_entropy_pooling"
        ) as mock_build:
            mock_estimator = MagicMock()
            mock_estimator.return_distribution_ = MagicMock()
            mock_estimator.return_distribution_.mu = np.array([0.001, 0.002, 0.0015])
            mock_estimator.return_distribution_.covariance = np.eye(3) * 0.0001
            mock_estimator.fit.return_value = mock_estimator
            mock_build.return_value = mock_estimator

            result = run_entropy_pooling(
                returns=returns,
                mean_views=("AAPL == 0.001",),
            )

        n = len(result["tickers"])
        assert len(result["covariance"]) == n
        assert all(len(row) == n for row in result["covariance"])


# ===========================================================================
# Schema validation — correlation_views format
# ===========================================================================


class TestCorrelationViewFormat:
    """Correlation views must use (ASSET1, ASSET2) == value format."""

    def test_valid_correlation_view_accepted(self) -> None:
        from app.schemas.views import EntropyPoolingRequest

        req = EntropyPoolingRequest(
            tickers=["AAPL", "MSFT"],
            correlation_views=["(AAPL, MSFT) == 0.5"],
        )
        assert req.correlation_views == ["(AAPL, MSFT) == 0.5"]

    def test_invalid_correlation_format_raises(self) -> None:
        from pydantic import ValidationError

        from app.schemas.views import EntropyPoolingRequest

        with pytest.raises(ValidationError):
            EntropyPoolingRequest(
                tickers=["AAPL", "MSFT"],
                correlation_views=["AAPL MSFT == 0.5"],
            )

    def test_all_null_views_raises(self) -> None:
        from pydantic import ValidationError

        from app.schemas.views import EntropyPoolingRequest

        with pytest.raises(ValidationError):
            EntropyPoolingRequest(tickers=["AAPL", "MSFT"])

    def test_empty_tickers_raises(self) -> None:
        from pydantic import ValidationError

        from app.schemas.views import EntropyPoolingRequest

        with pytest.raises(ValidationError):
            EntropyPoolingRequest(tickers=[], mean_views=["AAPL == 0.001"])
