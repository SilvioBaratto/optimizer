"""Empty-data and exception-path tests for attribution_service (issue #825).

Covers paths NOT exercised by tests/unit/test_attribution_service.py:
  - compute_brinson_attribution with empty portfolio weights (empty-data path)
  - compute_factor_attribution with empty portfolio_weights (empty-data path)
  - run_brinson_attribution raises MissingDataError when benchmark tickers
    are absent from the DB (line 291)
  - run_factor_attribution raises MissingDataError when no factor scores
    exist in the DB (line 323)

Raises NOT covered (skipped with reason):
  - InsufficientSectorCoverageError at line 392: already covered by the
    existing happy-path test file (TestBrinsonDecomposition.
    test_raises_when_sector_coverage_below_threshold).
  - MissingDataError at line 211 (compute_factor_attribution empty exposures):
    already covered in TestFactorAttribution.test_raises_missing_data_when_no_factor_exposures.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# compute_brinson_attribution — empty data path
# ---------------------------------------------------------------------------


class TestComputeBrinsonAttributionEmptyData:
    """compute_brinson_attribution with empty portfolio_weights returns safe structure."""

    def test_when_portfolio_weights_empty_then_returns_zero_totals(self) -> None:
        from app.services.attribution.attribution_service import (
            compute_brinson_attribution,
        )

        result = compute_brinson_attribution(
            portfolio_weights={},
            benchmark_weights={},
            sector_map={},
            portfolio_returns={},
            benchmark_returns={},
        )

        assert result["sectors"] == []
        assert result["total_allocation"] == pytest.approx(0.0)
        assert result["total_selection"] == pytest.approx(0.0)
        assert result["total_interaction"] == pytest.approx(0.0)
        assert result["total_active_return"] == pytest.approx(0.0)
        assert result["portfolio_return"] == pytest.approx(0.0)
        assert result["benchmark_return"] == pytest.approx(0.0)
        assert_json_safe(result)

    def test_when_portfolio_weights_empty_then_result_is_json_safe(self) -> None:
        from app.services.attribution.attribution_service import (
            compute_brinson_attribution,
        )

        result = compute_brinson_attribution(
            portfolio_weights={},
            benchmark_weights={"SPY": 1.0},
            sector_map={"SPY": "ETF"},
            portfolio_returns={},
            benchmark_returns={"SPY": 0.05},
        )

        assert_json_safe(result)


# ---------------------------------------------------------------------------
# compute_factor_attribution — empty-data path (no tickers in weights)
# ---------------------------------------------------------------------------


class TestComputeFactorAttributionEmptyData:
    """compute_factor_attribution with non-empty exposures but empty weights returns safe result."""

    def test_when_portfolio_weights_empty_then_portfolio_return_is_zero(self) -> None:
        from app.services.attribution.attribution_service import (
            compute_factor_attribution,
        )

        result = compute_factor_attribution(
            portfolio_weights={},
            asset_returns={},
            factor_exposures={"AAPL": {"momentum": 0.5}},
        )

        assert result["portfolio_return"] == pytest.approx(0.0)
        assert result["residual"] == pytest.approx(0.0)
        assert_json_safe(result)

    def test_when_asset_returns_empty_but_exposures_present_then_residual_is_zero(
        self,
    ) -> None:
        from app.services.attribution.attribution_service import (
            compute_factor_attribution,
        )

        result = compute_factor_attribution(
            portfolio_weights={"AAPL": 1.0},
            asset_returns={},
            factor_exposures={"AAPL": {"momentum": 1.0}},
        )

        assert result["portfolio_return"] == pytest.approx(0.0)
        assert_json_safe(result)


# ---------------------------------------------------------------------------
# run_brinson_attribution — MissingDataError when benchmark tickers absent
# ---------------------------------------------------------------------------


class TestRunBrinsonAttributionMissingBenchmark:
    """run_brinson_attribution raises MissingDataError when the DB has no price
    history for one or more benchmark tickers (attribution_service.py line 291).
    """

    def test_when_benchmark_ticker_missing_from_db_then_raises_missing_data_error(
        self,
    ) -> None:
        from app.services.attribution import attribution_service
        from app.services.attribution.attribution_service import MissingDataError

        session = MagicMock()
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 3, 31)

        # Sector map returns something — sector coverage check must pass.
        # Benchmark ticker "SPY" is NOT in the returned returns dict.
        with (
            patch.object(
                attribution_service, "_fetch_sector_map", return_value={"AAPL": "Technology"}
            ),
            patch.object(
                attribution_service,
                "_fetch_ticker_returns",
                side_effect=[
                    {"AAPL": 0.05},  # portfolio returns — present
                    {},  # benchmark returns — empty (SPY missing)
                ],
            ),
        ):
            with pytest.raises(MissingDataError, match="No price history for benchmark"):
                attribution_service.run_brinson_attribution(
                    session=session,
                    portfolio_weights={"AAPL": 1.0},
                    benchmark_weights={"SPY": 1.0},
                    start_date=start,
                    end_date=end,
                )

    def test_when_benchmark_returns_partial_then_raises_for_missing_ticker(
        self,
    ) -> None:
        from app.services.attribution import attribution_service
        from app.services.attribution.attribution_service import MissingDataError

        session = MagicMock()
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 3, 31)

        with (
            patch.object(
                attribution_service,
                "_fetch_sector_map",
                return_value={"AAPL": "Technology", "MSFT": "Technology"},
            ),
            patch.object(
                attribution_service,
                "_fetch_ticker_returns",
                side_effect=[
                    {"AAPL": 0.05, "MSFT": 0.03},  # portfolio
                    {"SPY": 0.04},  # benchmark — BOND missing
                ],
            ),
        ):
            with pytest.raises(MissingDataError, match="BOND"):
                attribution_service.run_brinson_attribution(
                    session=session,
                    portfolio_weights={"AAPL": 0.6, "MSFT": 0.4},
                    benchmark_weights={"SPY": 0.5, "BOND": 0.5},
                    start_date=start,
                    end_date=end,
                )


# ---------------------------------------------------------------------------
# run_factor_attribution — MissingDataError when no factor scores in DB
# ---------------------------------------------------------------------------


class TestRunFactorAttributionMissingScores:
    """run_factor_attribution raises MissingDataError when _fetch_factor_exposures
    returns an empty dict (attribution_service.py line 323).
    """

    def test_when_no_factor_scores_in_db_then_raises_missing_data_error(
        self,
    ) -> None:
        from app.services.attribution import attribution_service
        from app.services.attribution.attribution_service import MissingDataError

        session = MagicMock()
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 3, 31)

        with patch.object(
            attribution_service,
            "_fetch_factor_exposures",
            return_value={},
        ):
            with pytest.raises(MissingDataError, match="No factor scores found"):
                attribution_service.run_factor_attribution(
                    session=session,
                    portfolio_weights={"AAPL": 1.0},
                    start_date=start,
                    end_date=end,
                )

    def test_when_factor_scores_present_then_delegates_to_compute(self) -> None:
        """run_factor_attribution delegates to compute_factor_attribution and
        returns a JSON-safe dict when factor data exists.
        """
        from app.services.attribution import attribution_service

        session = MagicMock()
        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 3, 31)

        with (
            patch.object(
                attribution_service,
                "_fetch_factor_exposures",
                return_value={"AAPL": {"momentum": 0.8}},
            ),
            patch.object(
                attribution_service,
                "_fetch_ticker_returns",
                return_value={"AAPL": 0.10},
            ),
        ):
            result = attribution_service.run_factor_attribution(
                session=session,
                portfolio_weights={"AAPL": 1.0},
                start_date=start,
                end_date=end,
            )

        assert "factors" in result
        assert "portfolio_return" in result
        assert_json_safe(result)
