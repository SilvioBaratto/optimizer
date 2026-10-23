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
            "ProShares UltraShort 20+ Year Treasury",
            "ProShares Ultra 20+ Year Treasury",
            "ProShares Short 20+ Year Treasury",
            "ProShares UltraPro Short 20+ Year Treasury",
            "ProShares UltraShort Investment Grade Corporate Bond",
            "Xtrackers II US Treasuries Short Daily Swap",
            "Xtrackers II US Treasuries x2 Daily",
            "Direxion Daily 20+ Year Treasury Bear Shares",
            "UltraShort 20+ Year Treasury",
            "UltraShort 7-10 Year Treasury",
            "Ultra 20+ Year Treasury",
        ],
    )
    def test_leveraged_and_inverse_products_are_rejected(self, name: str) -> None:
        assert classify_instrument(name, "ETF") is None

    @pytest.mark.parametrize(
        ("name", "duration"),
        [
            ("iShares USD Ultrashort Bond (Dist)", "short"),
            ("Vanguard Ultra-Short Bond", "short"),
            ("Invesco Ultra Short Duration Bond", "short"),
            ("ProShares Short Term USD Emerging Markets Bond", "short"),
            ("Xtrackers USD Ultra Short 0-1 Year Treasury", "short"),
            ("Amundi Ultra-Short 0-1Y Euro Government Bond", "short"),
            ("Foo Ultrashort 0-1 Year Treasury", "short"),
            ("Foo Ultra Short Investment Grade Bond", "short"),
            ("Foo Ultra Short High Yield Bond", "short"),
        ],
    )
    def test_ultra_short_duration_bond_funds_are_kept(
        self, name: str, duration: str
    ) -> None:
        result = classify_instrument(name, "ETF")
        assert result is not None
        assert result.asset_class == "fixed_income"
        assert result.duration_bucket == duration


class TestMultiAsset:
    @pytest.mark.parametrize(
        "name",
        [
            "Vanguard LifeStrategy 60% Equity (Acc)",
            "SPDR Morningstar Multi-Asset Global Infrastructure (Dist)",
            "Amundi Multi-Asset Portfolio Dist",
            "IncomeShares 60-30-10 Multi-Asset Balanced (Dist)",
            "Some 60/40 Allocation Fund",
        ],
    )
    def test_multi_asset(self, name: str) -> None:
        assert classify_instrument(name, "ETF") == Classification(
            "multi_asset", None, None
        )

    def test_ucits_concentration_cap_name_is_not_multi_asset(self) -> None:
        # "20/35" is a UCITS concentration cap (equity ETF), not a 60/40 split.
        assert classify_instrument("UBS MSCI Switzerland 20/35 (Acc)", "ETF") is None


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
            ("iShares Short Treasury Bond ETF", "government", "short"),
            ("iShares 10-20 Year Treasury Bond ETF", "government", "long"),
            ("iShares 10-30 Year Treasury Bond ETF", "government", "long"),
        ],
    )
    def test_fixed_income(self, name: str, subclass: str, duration: str) -> None:
        result = classify_instrument(name, "ETF")
        assert result is not None
        assert result.asset_class == "fixed_income"
        assert result.fi_subclass == subclass
        assert result.duration_bucket == duration
