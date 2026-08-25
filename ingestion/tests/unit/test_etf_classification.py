"""T1 — fund classifier: name (+ optional yfinance metadata) → asset-class tags.

Equity ETFs classify to ``None`` (rejected: equity exposure comes from the
STOCK universe). Fixed-income sub-class precedence is geography/inflation-first:
inflation_linked > em > high_yield > government > corporate > aggregate.
"""

from __future__ import annotations

import pytest

from app.services.universe.trading212.classification import (
    Classification,
    classify_instrument,
)


class TestStockAndReject:
    def test_stock_is_equity(self) -> None:
        assert classify_instrument("Apple Inc", "STOCK") == Classification(
            "equity", None, None
        )

    def test_equity_etf_is_rejected(self) -> None:
        assert classify_instrument("Vanguard FTSE All-World (Acc)", "ETF") is None

    def test_non_stock_non_etf_is_rejected(self) -> None:
        assert classify_instrument("Some 3x Leverage Warrant", "WARRANT") is None

    @pytest.mark.parametrize(
        "name",
        [
            "Leverage Shares -5x Short 7-10 Year Treasury Bond",
            "GraniteShares 3x Long Natural Gas",
            "WisdomTree Bund 10Y 3x Daily Leveraged",
            "Xtrackers Inverse US Treasury Bond",
        ],
    )
    def test_leveraged_and_inverse_products_are_rejected(self, name: str) -> None:
        assert classify_instrument(name, "ETF") is None


class TestMultiAsset:
    @pytest.mark.parametrize(
        "name",
        [
            "Vanguard LifeStrategy 60% Equity (Acc)",
            "SPDR Morningstar Multi-Asset Global Infrastructure (Dist)",
            "Amundi Multi-Asset Portfolio Dist",
            "IncomeShares 60-30-10 Multi-Asset Balanced (Dist)",
        ],
    )
    def test_multi_asset(self, name: str) -> None:
        assert classify_instrument(name, "ETF") == Classification(
            "multi_asset", None, None
        )


class TestFixedIncomeSubclass:
    @pytest.mark.parametrize(
        ("name", "subclass", "duration"),
        [
            ("iShares Core Global Aggregate Bond", "aggregate", "intermediate"),
            ("Invesco US Treasury Bond 10+ Year", "government", "long"),
            ("Amundi US Treasury Bond 1-3Y (Dist)", "government", "short"),
            ("Vanguard USD Corporate Bond (Dist)", "corporate", "intermediate"),
            ("iShares JP Morgan USD EM Corp Bond", "em", "intermediate"),
            ("VanEck Emerging Markets High Yield Bond", "em", "intermediate"),
            (
                "Xtrackers II Global Inflation-Linked Bond (Acc)",
                "inflation_linked",
                "intermediate",
            ),
            (
                "PIMCO US Short-Term High Yield Corporate Bond (Dist)",
                "high_yield",
                "short",
            ),
            ("iShares USD Ultrashort Bond (Dist)", "aggregate", "short"),
        ],
    )
    def test_fixed_income(self, name: str, subclass: str, duration: str) -> None:
        result = classify_instrument(name, "ETF")
        assert result is not None
        assert result.asset_class == "fixed_income"
        assert result.fi_subclass == subclass
        assert result.duration_bucket == duration


class TestMetadataBoost:
    def test_balanced_asset_classes_promote_to_multi_asset(self) -> None:
        # No multi-asset keyword in the name, but yfinance asset_classes show a
        # material stock+bond split -> multi_asset.
        meta = {"asset_classes": {"stockPosition": 0.6, "bondPosition": 0.39}}
        result = classify_instrument("Acme Global Wealth Fund", "ETF", meta)
        assert result == Classification("multi_asset", None, None)

    def test_bond_heavy_asset_classes_promote_to_fixed_income(self) -> None:
        # No bond keyword in the name, but yfinance shows ~all bonds.
        meta = {"asset_classes": {"stockPosition": 0.0, "bondPosition": 0.95}}
        result = classify_instrument("Acme Income Fund", "ETF", meta)
        assert result is not None and result.asset_class == "fixed_income"
