"""Unit tests for get_market_snapshot — pure computation, no DB."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from app.services.dashboard_service import get_market_snapshot


class TestGetMarketSnapshot:
    @pytest.fixture()
    def fred_data(self) -> dict[str, tuple[float, float]]:
        return {
            "VIXCLS": (16.8, 18.0),
            "DTWEXBGS": (103.4, 103.68),
        }

    @pytest.fixture()
    def spy_prices(self) -> list[float]:
        return [449.30, 457.48]

    @pytest.fixture()
    def bond_yield(self) -> tuple[float, float]:
        return (4.22, -0.03)

    @pytest.fixture()
    def as_of_date(self) -> date:
        return date(2026, 3, 18)

    def test_vix_value(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        assert result["vix"] == pytest.approx(16.8, abs=0.01)

    def test_vix_change_is_absolute(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        expected = 16.8 - 18.0  # -1.2
        assert result["vix_change"] == pytest.approx(expected, abs=0.01)

    def test_sp500_return_is_fraction(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        expected = (457.48 - 449.30) / 449.30
        assert result["sp500_return"] == pytest.approx(expected, abs=1e-5)
        # Ensure it's a fraction, not a percentage
        assert abs(result["sp500_return"]) < 1.0

    def test_yield_passthrough(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        assert result["ten_year_yield"] == pytest.approx(4.22, abs=0.01)
        assert result["yield_change"] == pytest.approx(-0.03, abs=0.01)

    def test_usd_index(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        assert result["usd_index"] == pytest.approx(103.4, abs=0.01)
        expected_change = 103.4 - 103.68  # -0.28
        assert result["usd_change"] == pytest.approx(expected_change, abs=0.01)

    def test_as_of_is_utc_datetime(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        assert isinstance(result["as_of"], datetime)
        assert result["as_of"].tzinfo == timezone.utc
        assert result["as_of"].date() == as_of_date

    def test_all_keys_present(self, fred_data, spy_prices, bond_yield, as_of_date):
        result = get_market_snapshot(fred_data, spy_prices, bond_yield, as_of_date)
        expected_keys = {
            "vix", "vix_change", "sp500_return", "ten_year_yield",
            "yield_change", "usd_index", "usd_change", "as_of",
        }
        assert set(result.keys()) == expected_keys
