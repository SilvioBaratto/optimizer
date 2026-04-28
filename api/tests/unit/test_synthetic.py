"""Unit tests for POST /synthetic/scenario endpoint (issue #366).

TDD: Red phase — tests written before implementation.
All tests are isolated via mocks; no real DB or skfolio calls.
"""

from __future__ import annotations

import gzip
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(TICKERS)

# Fake 100-row price DataFrame (enough to pass the >=60 row validation)
_DATES = pd.date_range("2022-01-01", periods=100, freq="B")
MOCK_PRICES_DF = pd.DataFrame(
    np.random.default_rng(42).uniform(100, 500, (100, N_ASSETS)),
    index=_DATES,
    columns=TICKERS,
)

# Fake 99-row returns DataFrame (prices_to_returns reduces by 1)
MOCK_RETURNS_DF = pd.DataFrame(
    np.random.default_rng(42).uniform(-0.05, 0.05, (99, N_ASSETS)),
    index=_DATES[1:],
    columns=TICKERS,
)

# Fake scenario matrix returned by SyntheticData.fit
MOCK_SCENARIOS = np.random.default_rng(42).uniform(-0.05, 0.05, (1000, N_ASSETS))


def _make_mock_sd(scenarios: np.ndarray = MOCK_SCENARIOS) -> MagicMock:
    """Return a mock SyntheticData estimator with return_distribution_ set."""
    mock_sd = MagicMock()
    mock_rd = MagicMock()
    mock_rd.returns = scenarios
    mock_sd.return_distribution_ = mock_rd
    return mock_sd


# ---------------------------------------------------------------------------
# TestScenarioRequestSchema
# ---------------------------------------------------------------------------


class TestScenarioRequestSchema:
    """Validate that Pydantic schema enforces field constraints."""

    def test_valid_request_accepted(self) -> None:
        from app.schemas.synthetic import ScenarioRequest

        req = ScenarioRequest(tickers=["AAPL", "MSFT"], n_samples=500)
        assert req.tickers == ["AAPL", "MSFT"]
        assert req.n_samples == 500
        assert req.conditioning is None
        assert req.preset is None

    def test_empty_tickers_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(tickers=[], n_samples=100)

    def test_n_samples_zero_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(tickers=["AAPL"], n_samples=0)

    def test_n_samples_exceeds_max_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(tickers=["AAPL"], n_samples=100_001)

    def test_conditioning_value_at_boundary_raises_validation_error(self) -> None:
        """Values exactly at ±1 are excluded (must be strictly inside (-1, 1))."""
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(
                tickers=["AAPL", "MSFT"],
                n_samples=100,
                conditioning={"AAPL": 1.0},
            )

    def test_conditioning_minus_one_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(
                tickers=["AAPL", "MSFT"],
                n_samples=100,
                conditioning={"AAPL": -1.0},
            )

    def test_conditioning_value_inside_range_accepted(self) -> None:
        from app.schemas.synthetic import ScenarioRequest

        req = ScenarioRequest(
            tickers=["AAPL", "MSFT"],
            n_samples=100,
            conditioning={"AAPL": -0.15},
        )
        assert req.conditioning == {"AAPL": -0.15}

    def test_preset_scenario_generation_accepted(self) -> None:
        from app.schemas.synthetic import ScenarioRequest

        req = ScenarioRequest(
            tickers=["AAPL"],
            preset="scenario_generation",
        )
        assert req.preset == "scenario_generation"

    def test_preset_stress_test_accepted(self) -> None:
        from app.schemas.synthetic import ScenarioRequest

        req = ScenarioRequest(tickers=["AAPL"], preset="stress_test")
        assert req.preset == "stress_test"

    def test_invalid_preset_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        from app.schemas.synthetic import ScenarioRequest

        with pytest.raises(ValidationError):
            ScenarioRequest(tickers=["AAPL"], preset="invalid_preset")


# ---------------------------------------------------------------------------
# TestGenerateScenarios (service unit tests)
# ---------------------------------------------------------------------------


class TestGenerateScenarios:
    """Unit tests for generate_scenarios() pure function."""

    def test_returns_ndarray_and_columns(self) -> None:
        from app.services.synthetic_service import generate_scenarios

        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            scenarios, columns = generate_scenarios(
                tickers=TICKERS,
                n_samples=1000,
                conditioning=None,
                preset=None,
                session=mock_session,
            )

        assert isinstance(scenarios, np.ndarray)
        assert scenarios.shape == (1000, N_ASSETS)
        assert columns == TICKERS

    def test_uses_stress_test_preset_when_conditioning_provided(self) -> None:
        from app.services.synthetic_service import generate_scenarios

        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ) as mock_build,
        ):
            generate_scenarios(
                tickers=TICKERS,
                n_samples=500,
                conditioning={"AAPL": -0.10},
                preset=None,
                session=mock_session,
            )

        # sample_args should contain the conditioning dict
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("sample_args") == {
            "conditioning": {"AAPL": -0.10}
        }

    def test_conditioning_ticker_not_in_universe_raises_value_error(self) -> None:
        from app.services.synthetic_service import generate_scenarios

        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
            pytest.raises(ValueError, match="TSLA"),
        ):
            generate_scenarios(
                tickers=TICKERS,
                n_samples=500,
                conditioning={"TSLA": -0.10},  # TSLA not in TICKERS
                preset=None,
                session=mock_session,
            )

    def test_insufficient_data_raises_value_error(self) -> None:
        """Fewer than 60 rows of returns data should raise ValueError."""
        from app.services.synthetic_service import generate_scenarios

        short_prices = MOCK_PRICES_DF.iloc[:30]  # only 30 rows → 29 returns
        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=short_prices,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
            pytest.raises(ValueError, match="60"),
        ):
            generate_scenarios(
                tickers=TICKERS,
                n_samples=500,
                conditioning=None,
                preset=None,
                session=mock_session,
            )

    def test_explicit_scenario_generation_preset(self) -> None:
        from app.services.synthetic_service import generate_scenarios

        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
            patch(
                "app.services.synthetic_service.SyntheticDataConfig"
            ) as mock_config_cls,
        ):
            generate_scenarios(
                tickers=TICKERS,
                n_samples=1000,
                conditioning=None,
                preset="scenario_generation",
                session=mock_session,
            )

        mock_config_cls.for_scenario_generation.assert_called_once_with(n_samples=1000)

    def test_explicit_stress_test_preset(self) -> None:
        from app.services.synthetic_service import generate_scenarios

        mock_sd = _make_mock_sd()
        mock_session = MagicMock()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
            patch(
                "app.services.synthetic_service.SyntheticDataConfig"
            ) as mock_config_cls,
        ):
            generate_scenarios(
                tickers=TICKERS,
                n_samples=1000,
                conditioning=None,
                preset="stress_test",
                session=mock_session,
            )

        mock_config_cls.for_stress_test.assert_called_once_with(n_samples=1000)


# ---------------------------------------------------------------------------
# TestStoreAndLoadResult (service unit tests)
# ---------------------------------------------------------------------------


class TestStoreAndLoadResult:
    """Unit tests for store_result() / load_result() helpers."""

    def test_store_and_load_roundtrip(self) -> None:
        from app.services.synthetic_service import load_result, store_result

        payload = {"scenarios": [[0.01, -0.02], [0.03, 0.04]], "columns": ["A", "B"]}
        download_id, expires_at = store_result(payload)

        assert isinstance(download_id, str)
        assert len(download_id) == 36  # UUID4 string
        assert isinstance(expires_at, datetime)
        assert expires_at > datetime.now(timezone.utc)

        loaded = load_result(download_id)
        assert loaded is not None
        assert loaded["columns"] == ["A", "B"]

    def test_load_nonexistent_id_returns_none(self) -> None:
        from app.services.synthetic_service import load_result

        result = load_result(str(uuid.uuid4()))
        assert result is None

    def test_load_expired_result_returns_none(self) -> None:
        """Simulate expiry by writing a file with past timestamp metadata."""
        from app.services.synthetic_service import _TEMP_DIR, load_result

        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        download_id = str(uuid.uuid4())
        expired_time = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        payload = {
            "__expires_at": expired_time,
            "scenarios": [[0.01]],
            "columns": ["A"],
        }
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

        result = load_result(download_id)
        assert result is None

    def test_store_returns_correct_ttl(self) -> None:
        from app.services.synthetic_service import _STORE_TTL_SECONDS, store_result

        payload = {"data": "test"}
        _, expires_at = store_result(payload)
        expected_min = datetime.now(timezone.utc) + timedelta(
            seconds=_STORE_TTL_SECONDS - 5
        )
        expected_max = datetime.now(timezone.utc) + timedelta(
            seconds=_STORE_TTL_SECONDS + 5
        )
        assert expected_min < expires_at < expected_max


# ---------------------------------------------------------------------------
# TestSyntheticRouteInline (HTTP-level tests via TestClient)
# ---------------------------------------------------------------------------


class TestSyntheticRouteInline:
    """Tests for POST /api/v1/synthetic/scenario returning inline JSON."""

    def test_basic_generation_returns_200_with_scenarios(
        self, client: TestClient
    ) -> None:
        mock_scenarios = np.zeros((100, 3))
        mock_sd = _make_mock_sd(mock_scenarios)

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={"tickers": TICKERS, "n_samples": 100},
            )

        assert response.status_code == 200
        body = response.json()
        assert "scenarios" in body
        assert "columns" in body
        assert body["columns"] == TICKERS
        assert len(body["scenarios"]) == 100
        assert len(body["scenarios"][0]) == 3

    def test_conditioning_stress_test_returns_200(self, client: TestClient) -> None:
        mock_sd = _make_mock_sd(np.zeros((200, 3)))

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={
                    "tickers": TICKERS,
                    "n_samples": 200,
                    "conditioning": {"AAPL": -0.10},
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert "scenarios" in body
        assert "columns" in body

    def test_conditioning_ticker_not_in_tickers_returns_400(
        self, client: TestClient
    ) -> None:
        mock_sd = _make_mock_sd()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={
                    "tickers": TICKERS,
                    "n_samples": 100,
                    "conditioning": {"TSLA": -0.10},
                },
            )

        assert response.status_code == 400

    def test_insufficient_data_returns_422(self, client: TestClient) -> None:
        short_prices = MOCK_PRICES_DF.iloc[:30]
        mock_sd = _make_mock_sd()

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=short_prices,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={"tickers": TICKERS, "n_samples": 100},
            )

        assert response.status_code == 422

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/synthetic/scenario",
            json={"tickers": [], "n_samples": 100},
        )
        assert response.status_code == 422

    def test_n_samples_zero_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/synthetic/scenario",
            json={"tickers": TICKERS, "n_samples": 0},
        )
        assert response.status_code == 422

    def test_conditioning_value_at_one_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/synthetic/scenario",
            json={
                "tickers": TICKERS,
                "n_samples": 100,
                "conditioning": {"AAPL": 1.0},
            },
        )
        assert response.status_code == 422

    def test_preset_scenario_generation_accepted(self, client: TestClient) -> None:
        mock_sd = _make_mock_sd(np.zeros((100, 3)))

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={
                    "tickers": TICKERS,
                    "n_samples": 100,
                    "preset": "scenario_generation",
                },
            )

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestSyntheticRouteLarge (large n_samples → stored result)
# ---------------------------------------------------------------------------


class TestSyntheticRouteLarge:
    """Tests for POST /synthetic/scenario when n_samples > 10_000."""

    def test_large_n_samples_returns_201_with_download_id(
        self, client: TestClient
    ) -> None:
        large_scenarios = np.zeros((10_001, 3))
        mock_sd = _make_mock_sd(large_scenarios)

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            response = client.post(
                "/api/v1/synthetic/scenario",
                json={"tickers": TICKERS, "n_samples": 10_001},
            )

        assert response.status_code == 201
        body = response.json()
        assert "download_id" in body
        assert "expires_at" in body
        # Should NOT contain inline scenarios
        assert "scenarios" not in body

    def test_get_stored_scenario_returns_200(self, client: TestClient) -> None:
        large_scenarios = np.zeros((10_001, 3))
        mock_sd = _make_mock_sd(large_scenarios)

        with (
            patch(
                "app.services.synthetic_service._fetch_prices_df",
                return_value=MOCK_PRICES_DF,
            ),
            patch(
                "app.services.synthetic_service.build_synthetic_data",
                return_value=mock_sd,
            ),
        ):
            post_response = client.post(
                "/api/v1/synthetic/scenario",
                json={"tickers": TICKERS, "n_samples": 10_001},
            )

        assert post_response.status_code == 201
        download_id = post_response.json()["download_id"]

        get_response = client.get(f"/api/v1/synthetic/scenario/{download_id}")
        assert get_response.status_code == 200
        body = get_response.json()
        assert "scenarios" in body
        assert "columns" in body

    def test_get_invalid_download_id_returns_404(self, client: TestClient) -> None:
        response = client.get(f"/api/v1/synthetic/scenario/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_get_expired_download_id_returns_404(self, client: TestClient) -> None:
        from app.services.synthetic_service import _TEMP_DIR

        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        download_id = str(uuid.uuid4())
        expired_time = (
            datetime.now(timezone.utc) - timedelta(seconds=10)
        ).isoformat()
        payload = {
            "__expires_at": expired_time,
            "scenarios": [[0.01, 0.02, 0.03]],
            "columns": TICKERS,
        }
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

        response = client.get(f"/api/v1/synthetic/scenario/{download_id}")
        assert response.status_code == 404
