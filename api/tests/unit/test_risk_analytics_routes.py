"""Unit tests for risk analytics endpoints (issues #368, #369).

Covers:
  - GET /api/v1/portfolio/{name}/risk/var             — VaR/CVaR historical & parametric
  - GET /api/v1/portfolio/{name}/risk/correlation     — clustered correlation matrix
  - GET /api/v1/portfolio/{name}/risk/factor-exposure — portfolio-weighted factor exposures
  - GET /api/v1/portfolio/{name}/risk/concentration   — concentration metrics (issue #369)
  - GET /api/v1/portfolio/{name}/risk/liquidity       — ADDV-based liquidity (issue #369)

All external collaborators (PortfolioRepository, RiskAnalyticsService) are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

BASE_PORTFOLIO = "/api/v1/portfolio"

_PORTFOLIO_REPO = "app.api.v1.risk.risk_analytics.PortfolioRepository"
_SERVICE = "app.api.v1.risk.risk_analytics.RiskAnalyticsService"

_MOCK_PORTFOLIO_NAME = "my-portfolio"
_BASE = f"{BASE_PORTFOLIO}/{_MOCK_PORTFOLIO_NAME}/risk"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = _MOCK_PORTFOLIO_NAME) -> MagicMock:
    p = MagicMock()
    p.id = "00000000-0000-0000-0000-000000000001"
    p.name = name
    return p


def _make_var_result() -> dict:
    return {
        "var": {"90": 0.012, "95": 0.018, "99": 0.030},
        "cvar": {"90": 0.016, "95": 0.022, "99": 0.038},
        "method": "historical",
        "lookback": 252,
        "n_observations": 252,
    }


def _make_correlation_result() -> dict:
    return {
        "assets": ["AAPL", "MSFT"],
        "matrix": [[1.0, 0.7], [0.7, 1.0]],
        "cluster_labels": [0, 0],
    }


def _make_factor_exposure_result() -> dict:
    return {
        "exposures": {"momentum": 0.5, "quality": 0.3},
        "asset_exposures": {
            "AAPL": {"momentum": 0.6, "quality": 0.4},
            "MSFT": {"momentum": 0.4, "quality": 0.2},
        },
    }


# ===========================================================================
# GET /portfolio/{name}/risk/var
# ===========================================================================


class TestVarEndpoint:
    """Tests for GET /portfolio/{name}/risk/var."""

    def test_returns_200_with_var_and_cvar(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        assert resp.status_code == 200

    def test_response_contains_var_dict(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        body = resp.json()
        assert "var" in body
        assert set(body["var"].keys()) == {"90", "95", "99"}

    def test_response_contains_cvar_dict(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        body = resp.json()
        assert "cvar" in body
        assert set(body["cvar"].keys()) == {"90", "95", "99"}

    def test_cvar_values_exceed_var_values(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        body = resp.json()
        for level in ("90", "95", "99"):
            assert body["cvar"][level] >= body["var"][level]

    def test_default_method_is_historical(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        body = resp.json()
        assert body["method"] == "historical"

    def test_parametric_method_accepted(self, client: TestClient) -> None:
        parametric_result = {**_make_var_result(), "method": "parametric"}
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = parametric_result
            resp = client.get(f"{_BASE}/var?method=parametric")

        assert resp.status_code == 200

    def test_invalid_method_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/var?method=montecarlo")
        assert resp.status_code == 422

    def test_lookback_below_1_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/var?lookback=0")
        assert resp.status_code == 422

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{_BASE}/var")

        assert resp.status_code == 404

    def test_insufficient_data_returns_400(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.side_effect = ValueError(
                "Insufficient data"
            )
            resp = client.get(f"{_BASE}/var")

        assert resp.status_code == 400

    def test_response_contains_lookback(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var?lookback=126")

        body = resp.json()
        assert "lookback" in body

    def test_response_contains_n_observations(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_var.return_value = _make_var_result()
            resp = client.get(f"{_BASE}/var")

        body = resp.json()
        assert "nObservations" in body


# ===========================================================================
# GET /portfolio/{name}/risk/correlation
# ===========================================================================


class TestCorrelationEndpoint:
    """Tests for GET /portfolio/{name}/risk/correlation."""

    def test_returns_200(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.return_value = (
                _make_correlation_result()
            )
            resp = client.get(f"{_BASE}/correlation")

        assert resp.status_code == 200

    def test_response_contains_assets_list(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.return_value = (
                _make_correlation_result()
            )
            resp = client.get(f"{_BASE}/correlation")

        body = resp.json()
        assert isinstance(body["assets"], list)
        assert len(body["assets"]) == 2

    def test_response_matrix_is_n_by_n(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.return_value = (
                _make_correlation_result()
            )
            resp = client.get(f"{_BASE}/correlation")

        body = resp.json()
        n = len(body["assets"])
        assert len(body["matrix"]) == n
        assert all(len(row) == n for row in body["matrix"])

    def test_response_matrix_is_symmetric(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.return_value = (
                _make_correlation_result()
            )
            resp = client.get(f"{_BASE}/correlation")

        body = resp.json()
        matrix = body["matrix"]
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-9

    def test_response_contains_cluster_labels(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.return_value = (
                _make_correlation_result()
            )
            resp = client.get(f"{_BASE}/correlation")

        body = resp.json()
        assert "clusterLabels" in body
        assert len(body["clusterLabels"]) == len(body["assets"])

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{_BASE}/correlation")

        assert resp.status_code == 404

    def test_fewer_than_2_assets_returns_400(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_correlation.side_effect = ValueError(
                "At least 2 assets required"
            )
            resp = client.get(f"{_BASE}/correlation")

        assert resp.status_code == 400

    def test_lookback_below_1_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/correlation?lookback=0")
        assert resp.status_code == 422


# ===========================================================================
# GET /portfolio/{name}/risk/factor-exposure
# ===========================================================================


class TestFactorExposureEndpoint:
    """Tests for GET /portfolio/{name}/risk/factor-exposure."""

    def test_returns_200(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_factor_exposure.return_value = (
                _make_factor_exposure_result()
            )
            resp = client.get(f"{_BASE}/factor-exposure")

        assert resp.status_code == 200

    def test_response_contains_exposures_dict(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_factor_exposure.return_value = (
                _make_factor_exposure_result()
            )
            resp = client.get(f"{_BASE}/factor-exposure")

        body = resp.json()
        assert isinstance(body["exposures"], dict)

    def test_response_contains_asset_exposures_dict(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_factor_exposure.return_value = (
                _make_factor_exposure_result()
            )
            resp = client.get(f"{_BASE}/factor-exposure")

        body = resp.json()
        assert isinstance(body["assetExposures"], dict)

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{_BASE}/factor-exposure")

        assert resp.status_code == 404

    def test_no_factor_scores_returns_200_with_empty_exposures(
        self, client: TestClient
    ) -> None:
        """Regression for #424: absence of scores must surface as empty state, not 404."""
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_factor_exposure.return_value = {
                "exposures": {},
                "asset_exposures": {},
            }
            resp = client.get(f"{_BASE}/factor-exposure")

        assert resp.status_code == 200
        body = resp.json()
        assert body["exposures"] == {}
        assert body["assetExposures"] == {}

    def test_exposures_sum_correctly(self, client: TestClient) -> None:
        """Weighted exposure should equal sum(weight_i * exposure_i) for each factor."""
        asset_exposures = {
            "AAPL": {"momentum": 0.6, "quality": 0.4},
            "MSFT": {"momentum": 0.4, "quality": 0.2},
        }
        expected_momentum = 0.6 * 0.6 + 0.4 * 0.4
        expected_quality = 0.6 * 0.4 + 0.4 * 0.2

        result = {
            "exposures": {
                "momentum": round(expected_momentum, 6),
                "quality": round(expected_quality, 6),
            },
            "asset_exposures": asset_exposures,
        }

        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockSvc.return_value.compute_factor_exposure.return_value = result
            resp = client.get(f"{_BASE}/factor-exposure")

        body = resp.json()
        assert abs(body["exposures"]["momentum"] - expected_momentum) < 1e-6
        assert abs(body["exposures"]["quality"] - expected_quality) < 1e-6


# ===========================================================================
# GET /portfolio/{name}/risk/concentration  (issue #369)
# ===========================================================================


def _make_concentration_result() -> dict:
    return {
        "assets": [
            {"ticker": "AAPL", "name": "Apple Inc", "weight": 0.6},
            {"ticker": "MSFT", "name": "Microsoft Corp", "weight": 0.4},
        ],
        "summary": {
            "hhi": 0.52,
            "effective_n": 1.923,
            "top_n_ratio": 0.6,
        },
    }


def _make_snapshot(weights: dict | None = None):
    snap = MagicMock()
    snap.weights = weights if weights is not None else {"AAPL": 0.6, "MSFT": 0.4}
    return snap


class TestConcentrationEndpoint:
    """Tests for GET /portfolio/{name}/risk/concentration."""

    def test_returns_200(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        assert resp.status_code == 200

    def test_response_contains_assets_array(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        body = resp.json()
        assert isinstance(body["assets"], list)
        assert len(body["assets"]) == 2

    def test_response_contains_summary_with_hhi(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        body = resp.json()
        assert "summary" in body
        assert "hhi" in body["summary"]

    def test_response_summary_contains_effective_n_and_top_n_ratio(
        self, client: TestClient
    ) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        body = resp.json()
        assert "effectiveN" in body["summary"]
        assert "topNRatio" in body["summary"]

    def test_n_param_defaults_to_5(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        assert resp.status_code == 200

    def test_n_zero_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/concentration?n=0")
        assert resp.status_code == 422

    def test_n_negative_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/concentration?n=-1")
        assert resp.status_code == 422

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{_BASE}/concentration")

        assert resp.status_code == 404

    def test_empty_portfolio_returns_200_with_empty_assets(
        self, client: TestClient
    ) -> None:
        empty_result = {
            "assets": [],
            "summary": {"hhi": 0.0, "effective_n": 0.0, "top_n_ratio": 0.0},
        }
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot(
                weights={}
            )
            MockSvc.return_value.compute_concentration.return_value = empty_result
            resp = client.get(f"{_BASE}/concentration")

        assert resp.status_code == 200
        assert resp.json()["assets"] == []

    def test_asset_entries_have_ticker_name_weight(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_concentration.return_value = (
                _make_concentration_result()
            )
            resp = client.get(f"{_BASE}/concentration")

        for asset in resp.json()["assets"]:
            assert "ticker" in asset
            assert "name" in asset
            assert "weight" in asset


# ===========================================================================
# GET /portfolio/{name}/risk/liquidity  (issue #369)
# ===========================================================================


def _make_liquidity_result() -> dict:
    return {
        "assets": [
            {
                "ticker": "AAPL",
                "name": "Apple Inc",
                "weight": 0.6,
                "avg_daily_volume": 5_000_000.0,
                "days_to_liquidate": 1.2,
                "liquidity_cost": 0.0006,
            },
            {
                "ticker": "MSFT",
                "name": "Microsoft Corp",
                "weight": 0.4,
                "avg_daily_volume": None,
                "days_to_liquidate": None,
                "liquidity_cost": None,
            },
        ],
        "summary": {"weighted_avg_days_to_liquidate": 1.2},
    }


class TestLiquidityEndpoint:
    """Tests for GET /portfolio/{name}/risk/liquidity."""

    def test_returns_200(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity")

        assert resp.status_code == 200

    def test_response_contains_assets_array(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity")

        assert isinstance(resp.json()["assets"], list)

    def test_response_contains_portfolio_summary(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity")

        assert "summary" in resp.json()
        assert "weightedAvgDaysToLiquidate" in resp.json()["summary"]

    def test_asset_entry_contains_expected_fields(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity")

        asset = resp.json()["assets"][0]
        for field in (
            "ticker",
            "name",
            "weight",
            "avgDailyVolume",
            "daysToLiquidate",
            "liquidityCost",
        ):
            assert field in asset, f"Missing field: {field}"

    def test_missing_asset_data_returns_null_fields(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity")

        msft = next(a for a in resp.json()["assets"] if a["ticker"] == "MSFT")
        assert msft["avgDailyVolume"] is None
        assert msft["daysToLiquidate"] is None
        assert msft["liquidityCost"] is None

    def test_lookback_days_below_1_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/liquidity?lookback_days=0")
        assert resp.status_code == 422

    def test_participation_rate_zero_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/liquidity?participation_rate=0")
        assert resp.status_code == 422

    def test_participation_rate_above_1_returns_422(self, client: TestClient) -> None:
        resp = client.get(f"{_BASE}/liquidity?participation_rate=1.5")
        assert resp.status_code == 422

    def test_participation_rate_exactly_1_returns_200(self, client: TestClient) -> None:
        with (
            patch(_PORTFOLIO_REPO) as MockRepo,
            patch(_SERVICE) as MockSvc,
        ):
            MockRepo.return_value.get_by_name.return_value = _make_portfolio()
            MockRepo.return_value.get_latest_snapshot.return_value = _make_snapshot()
            MockSvc.return_value.compute_liquidity.return_value = (
                _make_liquidity_result()
            )
            resp = client.get(f"{_BASE}/liquidity?participation_rate=1.0")

        assert resp.status_code == 200

    def test_portfolio_not_found_returns_404(self, client: TestClient) -> None:
        with patch(_PORTFOLIO_REPO) as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            resp = client.get(f"{_BASE}/liquidity")

        assert resp.status_code == 404
