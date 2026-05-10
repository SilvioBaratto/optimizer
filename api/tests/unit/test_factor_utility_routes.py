"""Unit tests for factor utility endpoints:

  - POST /api/v1/factors/select
  - POST /api/v1/factors/exposure-constraints
  - POST /api/v1/factors/quintile-spread
  - POST /api/v1/factors/regime-tilt

Service functions are mocked so tests remain fast and DB-independent.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BASE_SELECT = "/api/v1/factors/select"
BASE_EXPOSURE = "/api/v1/factors/exposure-constraints"
BASE_QUINTILE = "/api/v1/factors/quintile-spread"
BASE_REGIME = "/api/v1/factors/regime-tilt"

_SERVICE_SELECT = "app.api.v1.factors.factors.select_stocks_for_tickers"
_SERVICE_EXPOSURE = "app.api.v1.factors.factors.build_exposure_constraints_for_tickers"
_SERVICE_QUINTILE = "app.api.v1.factors.factors.compute_quintile_spread_for_tickers"
_SERVICE_REGIME = "app.api.v1.factors.factors.compute_regime_tilt"

# ---------------------------------------------------------------------------
# Shared mock payloads
# ---------------------------------------------------------------------------

_MOCK_SELECT_RESULT: dict[str, Any] = {
    "selected_tickers": ["AAPL", "MSFT"],
    "count": 2,
    "turnover": 0.3,
    "buffer_zone": {"entered": ["AAPL"], "exited": ["META"]},
}

_MOCK_EXPOSURE_RESULT: dict[str, Any] = {
    "left_inequality": [[0.1, 0.2], [0.3, 0.4]],
    "right_inequality": [0.5, 0.6],
}

_MOCK_QUINTILE_RESULT: dict[str, Any] = {
    "quintile_cumulative_returns": {
        "Q1": [0.0, -0.01, -0.02],
        "Q5": [0.0, 0.01, 0.03],
    },
    "spread_cumulative_return": [0.0, 0.02, 0.05],
    "annualized_spread": 0.15,
}

_MOCK_REGIME_RESULT: dict[str, Any] = {
    "regime": "expansion",
    "tilted_weights": {"value": 0.3, "momentum": 0.2},
    "tilt_multipliers": {"value": 1.2, "momentum": 0.9},
}

# ---------------------------------------------------------------------------
# Valid request bodies
# ---------------------------------------------------------------------------

_VALID_SELECT_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT", "GOOGL"],
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
}

_VALID_EXPOSURE_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
    "bounds": [-0.5, 0.5],
}

_VALID_QUINTILE_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "factor_name": "momentum_12_1",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
}

_VALID_REGIME_REQUEST: dict[str, Any] = {
    "group_weights": {"value": 0.3, "momentum": 0.25, "profitability": 0.2},
}


# ===========================================================================
# POST /api/v1/factors/select
# ===========================================================================


class TestPostFactorSelect:
    """POST /api/v1/factors/select selects stocks and returns turnover + buffer zone."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.status_code == 200

    def test_returns_selected_tickers(self, client: TestClient) -> None:
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.json()["selected_tickers"] == ["AAPL", "MSFT"]

    def test_returns_count(self, client: TestClient) -> None:
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.json()["count"] == 2

    def test_returns_turnover(self, client: TestClient) -> None:
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.json()["turnover"] == pytest.approx(0.3)

    def test_returns_buffer_zone_entered_and_exited(self, client: TestClient) -> None:
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        bz = resp.json()["buffer_zone"]
        assert bz["entered"] == ["AAPL"]
        assert bz["exited"] == ["META"]

    def test_returns_null_turnover_when_no_previous_members(
        self, client: TestClient
    ) -> None:
        result_no_turnover = {**_MOCK_SELECT_RESULT, "turnover": None}
        with patch(_SERVICE_SELECT, return_value=result_no_turnover):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.json()["turnover"] is None

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        bad = {**_VALID_SELECT_REQUEST, "tickers": []}
        resp = client.post(BASE_SELECT, json=bad)

        assert resp.status_code == 422

    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        bad = {"tickers": ["AAPL"]}  # missing start_date / end_date
        resp = client.post(BASE_SELECT, json=bad)

        assert resp.status_code == 422

    def test_no_factor_data_returns_404(self, client: TestClient) -> None:
        from app.services.factors.factor_service import FactorDataError

        with patch(_SERVICE_SELECT, side_effect=FactorDataError("no data")):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        assert resp.status_code == 404

    def test_empty_universe_returns_empty_selection(self, client: TestClient) -> None:
        empty_result: dict[str, Any] = {
            "selected_tickers": [],
            "count": 0,
            "turnover": None,
            "buffer_zone": {"entered": [], "exited": []},
        }
        with patch(_SERVICE_SELECT, return_value=empty_result):
            resp = client.post(BASE_SELECT, json=_VALID_SELECT_REQUEST)

        body = resp.json()
        assert body["selected_tickers"] == []
        assert body["count"] == 0

    def test_optional_current_members_is_accepted(self, client: TestClient) -> None:
        req = {**_VALID_SELECT_REQUEST, "current_members": ["AAPL"]}
        with patch(_SERVICE_SELECT, return_value=_MOCK_SELECT_RESULT):
            resp = client.post(BASE_SELECT, json=req)

        assert resp.status_code == 200

    def test_fixed_count_mode_returns_exact_count(self, client: TestClient) -> None:
        result_2 = {
            **_MOCK_SELECT_RESULT,
            "selected_tickers": ["AAPL", "MSFT"],
            "count": 2,
        }
        req = {**_VALID_SELECT_REQUEST, "method": "fixed_count", "target_count": 2}
        with patch(_SERVICE_SELECT, return_value=result_2):
            resp = client.post(BASE_SELECT, json=req)

        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_quantile_mode_returns_tickers_above_threshold(
        self, client: TestClient
    ) -> None:
        result = {**_MOCK_SELECT_RESULT, "selected_tickers": ["AAPL"], "count": 1}
        req = {**_VALID_SELECT_REQUEST, "method": "quantile", "target_quantile": 0.5}
        with patch(_SERVICE_SELECT, return_value=result):
            resp = client.post(BASE_SELECT, json=req)

        assert resp.status_code == 200
        assert resp.json()["selected_tickers"] == ["AAPL"]

    def test_fixed_count_without_target_count_returns_422(
        self, client: TestClient
    ) -> None:
        req = {**_VALID_SELECT_REQUEST, "method": "fixed_count"}
        resp = client.post(BASE_SELECT, json=req)

        assert resp.status_code == 422

    def test_quantile_without_target_quantile_returns_422(
        self, client: TestClient
    ) -> None:
        req = {**_VALID_SELECT_REQUEST, "method": "quantile"}
        resp = client.post(BASE_SELECT, json=req)

        assert resp.status_code == 422


# ===========================================================================
# POST /api/v1/factors/exposure-constraints
# ===========================================================================


class TestPostExposureConstraints:
    """POST /api/v1/factors/exposure-constraints returns serializable inequality matrices."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_EXPOSURE, return_value=_MOCK_EXPOSURE_RESULT):
            resp = client.post(BASE_EXPOSURE, json=_VALID_EXPOSURE_REQUEST)

        assert resp.status_code == 200

    def test_returns_left_inequality_as_nested_list(self, client: TestClient) -> None:
        with patch(_SERVICE_EXPOSURE, return_value=_MOCK_EXPOSURE_RESULT):
            resp = client.post(BASE_EXPOSURE, json=_VALID_EXPOSURE_REQUEST)

        li = resp.json()["left_inequality"]
        assert isinstance(li, list)
        assert all(isinstance(row, list) for row in li)

    def test_returns_right_inequality_as_list(self, client: TestClient) -> None:
        with patch(_SERVICE_EXPOSURE, return_value=_MOCK_EXPOSURE_RESULT):
            resp = client.post(BASE_EXPOSURE, json=_VALID_EXPOSURE_REQUEST)

        ri = resp.json()["right_inequality"]
        assert isinstance(ri, list)

    def test_no_factor_data_returns_404(self, client: TestClient) -> None:
        from app.services.factors.factor_service import FactorDataError

        with patch(_SERVICE_EXPOSURE, side_effect=FactorDataError("no data")):
            resp = client.post(BASE_EXPOSURE, json=_VALID_EXPOSURE_REQUEST)

        assert resp.status_code == 404

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        bad = {**_VALID_EXPOSURE_REQUEST, "tickers": []}
        resp = client.post(BASE_EXPOSURE, json=bad)

        assert resp.status_code == 422

    def test_per_factor_bounds_dict_is_accepted(self, client: TestClient) -> None:
        req_dict = {
            **_VALID_EXPOSURE_REQUEST,
            "bounds": {"momentum_12_1": [-0.3, 0.3], "book_to_price": [-0.5, 0.5]},
        }
        with patch(_SERVICE_EXPOSURE, return_value=_MOCK_EXPOSURE_RESULT):
            resp = client.post(BASE_EXPOSURE, json=req_dict)

        assert resp.status_code == 200


# ===========================================================================
# POST /api/v1/factors/quintile-spread
# ===========================================================================


class TestPostQuintileSpread:
    """POST /api/v1/factors/quintile-spread returns quintile cumulative returns and spread."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_QUINTILE, return_value=_MOCK_QUINTILE_RESULT):
            resp = client.post(BASE_QUINTILE, json=_VALID_QUINTILE_REQUEST)

        assert resp.status_code == 200

    def test_returns_quintile_cumulative_returns(self, client: TestClient) -> None:
        with patch(_SERVICE_QUINTILE, return_value=_MOCK_QUINTILE_RESULT):
            resp = client.post(BASE_QUINTILE, json=_VALID_QUINTILE_REQUEST)

        qcr = resp.json()["quintile_cumulative_returns"]
        assert "Q1" in qcr
        assert "Q5" in qcr

    def test_returns_spread_cumulative_return(self, client: TestClient) -> None:
        with patch(_SERVICE_QUINTILE, return_value=_MOCK_QUINTILE_RESULT):
            resp = client.post(BASE_QUINTILE, json=_VALID_QUINTILE_REQUEST)

        spread = resp.json()["spread_cumulative_return"]
        assert isinstance(spread, list)

    def test_returns_annualized_spread(self, client: TestClient) -> None:
        with patch(_SERVICE_QUINTILE, return_value=_MOCK_QUINTILE_RESULT):
            resp = client.post(BASE_QUINTILE, json=_VALID_QUINTILE_REQUEST)

        assert resp.json()["annualized_spread"] == pytest.approx(0.15)

    def test_no_factor_data_returns_404(self, client: TestClient) -> None:
        from app.services.factors.factor_service import FactorDataError

        with patch(_SERVICE_QUINTILE, side_effect=FactorDataError("no data")):
            resp = client.post(BASE_QUINTILE, json=_VALID_QUINTILE_REQUEST)

        assert resp.status_code == 404

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        bad = {**_VALID_QUINTILE_REQUEST, "tickers": []}
        resp = client.post(BASE_QUINTILE, json=bad)

        assert resp.status_code == 422

    def test_missing_factor_name_returns_422(self, client: TestClient) -> None:
        bad = {k: v for k, v in _VALID_QUINTILE_REQUEST.items() if k != "factor_name"}
        resp = client.post(BASE_QUINTILE, json=bad)

        assert resp.status_code == 422

    def test_optional_n_quantiles_is_accepted(self, client: TestClient) -> None:
        req = {**_VALID_QUINTILE_REQUEST, "n_quantiles": 10}
        with patch(_SERVICE_QUINTILE, return_value=_MOCK_QUINTILE_RESULT):
            resp = client.post(BASE_QUINTILE, json=req)

        assert resp.status_code == 200


# ===========================================================================
# POST /api/v1/factors/regime-tilt
# ===========================================================================


class TestPostRegimeTilt:
    """POST /api/v1/factors/regime-tilt applies macro regime tilts to group weights."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_REGIME, return_value=_MOCK_REGIME_RESULT):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        assert resp.status_code == 200

    def test_returns_regime_string(self, client: TestClient) -> None:
        with patch(_SERVICE_REGIME, return_value=_MOCK_REGIME_RESULT):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        assert resp.json()["regime"] == "expansion"

    def test_returns_tilted_weights(self, client: TestClient) -> None:
        with patch(_SERVICE_REGIME, return_value=_MOCK_REGIME_RESULT):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        tw = resp.json()["tilted_weights"]
        assert isinstance(tw, dict)
        assert "value" in tw

    def test_returns_tilt_multipliers(self, client: TestClient) -> None:
        with patch(_SERVICE_REGIME, return_value=_MOCK_REGIME_RESULT):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        tm = resp.json()["tilt_multipliers"]
        assert isinstance(tm, dict)
        assert "value" in tm

    def test_no_macro_data_returns_404(self, client: TestClient) -> None:
        from app.services.factors.factor_service import FactorDataError

        with patch(_SERVICE_REGIME, side_effect=FactorDataError("no macro data")):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        assert resp.status_code == 404

    def test_empty_group_weights_returns_422(self, client: TestClient) -> None:
        bad = {"group_weights": {}}
        resp = client.post(BASE_REGIME, json=bad)

        assert resp.status_code == 422

    def test_missing_group_weights_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_REGIME, json={})

        assert resp.status_code == 422

    def test_optional_regime_tilt_config_is_accepted(self, client: TestClient) -> None:
        req = {
            **_VALID_REGIME_REQUEST,
            "enable": True,
            "max_tilt_multiplier": 2.0,
        }
        with patch(_SERVICE_REGIME, return_value=_MOCK_REGIME_RESULT):
            resp = client.post(BASE_REGIME, json=req)

        assert resp.status_code == 200

    def test_tilted_weights_sum_to_approximately_one(self, client: TestClient) -> None:
        nine_groups = {
            "value": 0.12,
            "momentum": 0.12,
            "profitability": 0.12,
            "quality": 0.11,
            "low_volatility": 0.11,
            "growth": 0.11,
            "size": 0.11,
            "liquidity": 0.10,
            "dividend": 0.10,
        }
        result_nine = {
            "regime": "expansion",
            "tilted_weights": nine_groups,
            "tilt_multipliers": dict.fromkeys(nine_groups, 1.0),
        }
        with patch(_SERVICE_REGIME, return_value=result_nine):
            resp = client.post(BASE_REGIME, json=_VALID_REGIME_REQUEST)

        tilted = resp.json()["tilted_weights"]
        assert sum(tilted.values()) == pytest.approx(1.0, abs=0.01)
