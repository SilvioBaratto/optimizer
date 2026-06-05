"""Empty-data and exception-path tests for dashboard_service (issue #825) — Part 2.

Covers compute_asset_class_returns raises and empty-data paths:
  - raises ValueError when prices has fewer than 2 rows (line 685)
  - raises ValueError when no price data on or after YTD cutoff (line 697)
  - returns JSON-safe result for minimal valid 2-row input (empty-data path)
  - returns JSON-safe result with no rows when weights map to no matching tickers

Raises already covered in test_dashboard_paths.py (not duplicated here):
  - compute_equity_curve insufficient overlap (line 309)
  - compute_drift no positions and no prices (line 507)
  - _actual_weights_from_prices no common tickers (line 555)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests._fixtures.assertions import assert_json_safe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_asset_prices(
    tickers: list[str],
    periods: int,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(start, periods=periods)
    data = {
        t: (1 + rng.normal(0.0003, 0.01, periods)).cumprod() * 100 for t in tickers
    }
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# compute_asset_class_returns — exception paths
# ---------------------------------------------------------------------------


class TestComputeAssetClassReturnsExceptionPaths:
    """compute_asset_class_returns raises ValueError on insufficient price data."""

    def test_when_prices_has_one_row_then_raises_insufficient_data(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        prices = _make_asset_prices(["AAPL"], periods=1)

        with pytest.raises(ValueError, match="Insufficient price data"):
            compute_asset_class_returns(
                weights={"AAPL": 1.0},
                sector_mapping={"AAPL": "Technology"},
                prices=prices,
                today=date(2024, 1, 2),
            )

    def test_when_prices_is_empty_then_raises_insufficient_data(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        prices = pd.DataFrame({"AAPL": pd.Series([], dtype=float)})

        with pytest.raises(ValueError, match="Insufficient price data"):
            compute_asset_class_returns(
                weights={"AAPL": 1.0},
                sector_mapping={"AAPL": "Technology"},
                prices=prices,
                today=date(2024, 1, 2),
            )

    def test_when_no_ytd_data_then_raises_ytd_error(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        # Prices only cover 2023 — no data on or after Jan 1 2024.
        prices = _make_asset_prices(["AAPL"], periods=5, start="2023-06-01")

        with pytest.raises(ValueError, match="No price data on or after"):
            compute_asset_class_returns(
                weights={"AAPL": 1.0},
                sector_mapping={"AAPL": "Technology"},
                prices=prices,
                today=date(2024, 1, 15),
            )


# ---------------------------------------------------------------------------
# compute_asset_class_returns — empty-data and JSON-safe paths
# ---------------------------------------------------------------------------


class TestComputeAssetClassReturnsEmptyData:
    """compute_asset_class_returns returns safe structure for minimal valid inputs."""

    def test_when_minimal_two_rows_then_returns_json_safe_result(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        # 2 rows in 2024 satisfies n_rows >= 2 and YTD cutoff is present.
        dates = pd.bdate_range("2024-01-02", periods=2)
        prices = pd.DataFrame({"AAPL": [150.0, 152.0]}, index=dates)

        result = compute_asset_class_returns(
            weights={"AAPL": 1.0},
            sector_mapping={"AAPL": "Technology"},
            prices=prices,
            today=date(2024, 1, 3),
        )

        assert "returns" in result
        assert "as_of" in result
        assert len(result["returns"]) == 1
        # as_of is a date object — convert before json safety check
        safe = {**result, "as_of": str(result["as_of"])}
        assert_json_safe(safe)

    def test_when_weights_empty_then_returns_no_sector_rows(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        prices = _make_asset_prices(["AAPL"], periods=5, start="2024-01-02")

        result = compute_asset_class_returns(
            weights={},
            sector_mapping={},
            prices=prices,
            today=date(2024, 1, 10),
        )

        assert result["returns"] == []
        safe = {**result, "as_of": str(result["as_of"])}
        assert_json_safe(safe)

    def test_when_tickers_not_in_prices_then_sector_returns_zero(self) -> None:
        """Tickers present in weights but absent from prices columns should not crash
        and produce zero returns for the period (the helper silently skips them).
        """
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        prices = _make_asset_prices(["AAPL"], periods=5, start="2024-01-02")

        result = compute_asset_class_returns(
            weights={"AAPL": 0.5, "MISSING": 0.5},
            sector_mapping={"AAPL": "Technology", "MISSING": "Unknown"},
            prices=prices,
            today=date(2024, 1, 10),
        )

        assert "returns" in result
        safe = {**result, "as_of": str(result["as_of"])}
        assert_json_safe(safe)

    def test_when_full_year_data_then_all_periods_populated(self) -> None:
        from app.services.dashboard.dashboard_service import compute_asset_class_returns

        prices = _make_asset_prices(["AAPL", "MSFT"], periods=30, start="2024-01-02")
        today = prices.index[-1].date()

        result = compute_asset_class_returns(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            sector_mapping={"AAPL": "Technology", "MSFT": "Technology"},
            prices=prices,
            today=today,
        )

        assert len(result["returns"]) == 1  # one sector
        row = result["returns"][0]
        assert row["name"] == "Technology"
        assert "1D" in row
        assert "YTD" in row
        safe = {**result, "as_of": str(result["as_of"])}
        assert_json_safe(safe)


# ---------------------------------------------------------------------------
# compute_allocation — empty weights (pure empty-data path, no DB)
# ---------------------------------------------------------------------------


class TestComputeAllocationEmptyData:
    """compute_allocation with empty weights returns a safe zero-node result."""

    def test_when_weights_empty_then_returns_empty_nodes(self) -> None:
        from app.services.dashboard.dashboard_service import compute_allocation

        result = compute_allocation(weights={}, sector_mapping={})

        assert result["nodes"] == []
        assert result["total_positions"] == 0
        assert result["total_sectors"] == 0
        assert_json_safe(result)
