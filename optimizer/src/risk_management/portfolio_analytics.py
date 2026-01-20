#!/usr/bin/env python3
from optimizer.database.models.stock_signals import StockSignal
from optimizer.database.models.universe import Instrument


class PortfolioAnalytics:
    """Stock selection utility functions for ranking and country mapping."""

    @staticmethod
    def calculate_composite_score(signal: StockSignal) -> float:
        """
        Calculate composite score for ranking using available quantitative metrics.
        """
        score = 0.0

        # Momentum (35%) - based on confidence level
        if signal.confidence_level:
            confidence_map = {"high": 100, "medium": 60, "low": 30}
            score += 0.35 * confidence_map.get(signal.confidence_level.value, 30)

        # Sharpe Ratio (25%) - risk-adjusted return quality
        if signal.sharpe_ratio is not None and signal.sharpe_ratio > 0:
            sharpe_score = min(100, (signal.sharpe_ratio / 2.0) * 100)  # 2.0 Sharpe = 100
            score += 0.25 * sharpe_score

        # Volatility (20%) - inverse (lower is better)
        if signal.volatility is not None and signal.volatility > 0:
            volatility_score = max(0, 100 - (signal.volatility / 0.5) * 100)  # 50% vol = 0 score
            score += 0.20 * volatility_score
        else:
            score += 0.20 * 50  # Neutral if missing

        # Alpha (15%) - excess return vs benchmark
        if signal.alpha is not None:
            alpha_score = min(
                100, max(0, ((signal.alpha + 0.05) / 0.15) * 100)
            )  # -5% to +10% mapped to 0-100
            score += 0.15 * alpha_score
        else:
            score += 0.15 * 50  # Neutral if missing

        # Data Quality (5%)
        if signal.data_quality_score is not None:
            score += 0.05 * (signal.data_quality_score * 100)
        else:
            score += 0.05 * 50  # Neutral if missing

        return score

    @staticmethod
    def get_country(signal: StockSignal, instrument: Instrument) -> str:
        """Extract country from exchange name."""
        exchange_name = signal.exchange_name or (
            instrument.exchange.exchange_name if instrument.exchange else None
        )

        if not exchange_name:
            return "Unknown"

        exchange_to_country = {
            "NYSE": "USA",
            "NASDAQ": "USA",
            "London Stock Exchange": "UK",
            "Deutsche Börse Xetra": "Germany",
            "Gettex": "Germany",
            "Euronext Paris": "France",
            "Euronext Amsterdam": "Netherlands",
            "SIX Swiss Exchange": "Switzerland",
            "Tokyo Stock Exchange": "Japan",
        }

        return exchange_to_country.get(exchange_name, exchange_name)
