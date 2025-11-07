"""
Mathematical Signal Calculator - Core Orchestrator
==================================================

Institutional-grade signal calculator using four-factor framework (NO LLM costs).

Framework per Sections 5.2 (Factor Construction) and 5.2.6 (Multi-Factor Combination):

Four Institutional Factors (equal-weighted baseline):
1. Value Factor (25%): B/P, E/P, FCF/P (Section 5.2.2)
2. Momentum Factor (25%): 12-1 month returns (Section 5.2.5)
3. Quality Factor (25%): ROE, margins, Sharpe (Section 5.2.4)
4. Growth Factor (25%): Revenue/earnings growth (Section 5.2.3)

Z-Score Framework (Section 5.2.1):
- Each factor standardized relative to market norms
- Equal-weighted composite z-score (baseline)
- James-Stein shrinkage (Section 4.3.2)
- Regime-based adjustments (Section 2.1)

Signal Classification (percentile-based thresholds):
- LARGE_GAIN (top 20%)
- SMALL_GAIN (60th-80th percentile)
- NEUTRAL (40th-60th percentile)
- SMALL_DECLINE (20th-40th percentile)
- LARGE_DECLINE (bottom 20%)

Technical Implementation:
- EWMA volatility (84-day half-life)
- Momentum with 1-month skip
- Statistical significance (alpha t-stat, CI)
- Risk decomposition (systematic vs specific)
- Regime multipliers for risk/return
"""

from datetime import date as date_type
from typing import Optional, Tuple, Dict
import logging
import numpy as np

from baml_client.types import (
    StockSignalOutput,
    SignalType,
    ConfidenceLevel,
    SignalDrivers,
    RiskFactors,
)

# Import refactored modules
from src.stock_analyzer.data.fetchers import (
    fetch_price_data,
    fetch_macro_regime,
    get_country_from_ticker,
)
from src.stock_analyzer.technical.metrics import calculate_technical_metrics
from src.stock_analyzer.factors.calculators import (
    calculate_value_factor,
    calculate_momentum_factor,
    calculate_quality_factor,
    calculate_growth_factor,
    calculate_ic_weights,
)
from src.stock_analyzer.adjustments.macro import apply_macro_adjustments
from src.stock_analyzer.adjustments.risk import calculate_risk_factors, calculate_confidence
from src.stock_analyzer.classification.distribution import SignalClassifier
from src.stock_analyzer.classification.scoring import (
    calculate_upside_potential,
    calculate_downside_risk,
    calculate_data_quality,
    generate_analysis_notes,
)
from src.stock_analyzer.risk_free_rate import get_risk_free_rate

logger = logging.getLogger(__name__)


class MathematicalSignalCalculator:
    """
    Institutional-grade signal calculator using four-factor framework (NO LLM costs).

    This is the main orchestrator that coordinates all refactored modules to generate
    signals based on fundamental, technical, and macro analysis.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        use_ic_weighting: bool = True,
        cross_sectional_stats: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
        risk_free_rates_cache: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize mathematical signal calculator.

        Args:
            lookback_days: Days of historical data to analyze (default: 252 = 1 year)
            use_ic_weighting: Enable IC-weighted factor combination (requires history)
            cross_sectional_stats: Optional universe-level statistics for true cross-sectional
                                   standardization
            risk_free_rates_cache: Optional pre-fetched risk-free rates by country (performance optimization)
                                   Format: {'USA': 0.045, 'UK': 0.042, ...}
        """
        self.lookback_days = lookback_days
        self.use_ic_weighting = use_ic_weighting
        self.cross_sectional_stats = cross_sectional_stats
        self.risk_free_rates_cache = risk_free_rates_cache or {}
        self.historical_ics = {}  # Track historical factor ICs for dynamic weighting
        self.last_technical_metrics = None  # Store last computed metrics

        # Initialize signal classifier
        self.classifier = SignalClassifier(validation_interval=50, save_interval=100)

        # Log configuration
        if cross_sectional_stats:
            logger.info("✅ Using TRUE cross-sectional standardization (universe-level statistics)")
        else:
            logger.info("⚠️  Using static S&P 500 norms (single-stock approximation)")

        if risk_free_rates_cache:
            logger.info(f"✅ Using pre-fetched risk-free rates for {len(risk_free_rates_cache)} countries (performance optimized)")
        else:
            logger.info("⚠️  Risk-free rates will be fetched individually (slower)")

    def _get_risk_free_rate_cached(self, country: Optional[str]) -> float:
        """
        Get risk-free rate using cache-first strategy (performance optimization).

        Args:
            country: Country code (USA, UK, Germany, etc.)

        Returns:
            Risk-free rate as decimal (e.g., 0.045 for 4.5%)
        """
        if not country:
            return 0.045  # Default fallback

        # Try cache first (fast path)
        if country in self.risk_free_rates_cache:
            return self.risk_free_rates_cache[country]

        # Fall back to individual query (slow path)
        rate = get_risk_free_rate(country=country)
        # Cache it for future use
        self.risk_free_rates_cache[country] = rate
        return rate

    async def generate_signal(
        self, yf_ticker: str, target_date: Optional[date_type] = None
    ) -> Optional[StockSignalOutput]:
        """
        Generate institutional-grade signal using four-factor framework (NO LLM).

        Pipeline:
        1. Fetch price data and benchmark
        2. Calculate technical metrics
        3. Calculate institutional factors (value, momentum, quality, growth)
        4. Multi-factor combination (equal-weighted or IC-weighted)
        5. Apply James-Stein shrinkage
        6. Apply macro adjustments
        7. Classify signal using percentiles
        8. Calculate confidence and risk factors
        9. Build StockSignalOutput

        Args:
            yf_ticker: Yahoo Finance ticker symbol
            target_date: Date to generate signal for (defaults to today)

        Returns:
            StockSignalOutput with institutional factor-derived signal
        """
        if target_date is None:
            target_date = date_type.today()

        logger.info(f"Generating INSTITUTIONAL signal for {yf_ticker} on {target_date}")

        try:
            # Step 1: Fetch price data
            stock_data, benchmark_data, info = await fetch_price_data(yf_ticker, period="2y")

            if stock_data is None or len(stock_data) < 252:
                logger.warning(
                    f"Insufficient data for {yf_ticker} "
                    f"(need 252 days, got {len(stock_data) if stock_data is not None else 0})"
                )
                return None

            # Step 1.5: Get country and risk-free rate (using cache for performance)
            country = get_country_from_ticker(yf_ticker, info) if info else None
            risk_free_rate = self._get_risk_free_rate_cached(country)
            logger.debug(
                f"{yf_ticker} → Country: {country}, "
                f"Risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)"
            )

            # Step 2: Calculate technical metrics
            technical_metrics = calculate_technical_metrics(
                stock_data, benchmark_data, risk_free_rate
            )
            self.last_technical_metrics = technical_metrics

            # Step 3: Calculate institutional factors
            value_z = calculate_value_factor(info, stock_data, self.cross_sectional_stats)
            momentum_z = calculate_momentum_factor(technical_metrics, self.cross_sectional_stats)
            quality_z = calculate_quality_factor(
                info, technical_metrics, country, self.cross_sectional_stats
            )
            growth_z = calculate_growth_factor(info, country, self.cross_sectional_stats)

            logger.debug(
                f"Factor z-scores: Value={value_z:.2f}, Momentum={momentum_z:.2f}, "
                f"Quality={quality_z:.2f}, Growth={growth_z:.2f}"
            )

            # Step 4: Multi-factor combination
            if self.use_ic_weighting:
                weights = calculate_ic_weights(self.historical_ics)
                composite_z = (
                    value_z * weights['value']
                    + momentum_z * weights['momentum']
                    + quality_z * weights['quality']
                    + growth_z * weights['growth']
                )
                logger.debug(f"IC-weighted combination: {weights}")
            else:
                composite_z = (value_z + momentum_z + quality_z + growth_z) / 4.0
                logger.debug("Using equal-weighted factor combination (baseline)")

            # Step 5: Apply James-Stein shrinkage
            shrinkage_factor = 0.7
            composite_z = shrinkage_factor * composite_z + (1 - shrinkage_factor) * 0

            logger.debug(f"Composite z-score (after shrinkage): {composite_z:.2f}")

            # Step 6: Apply macro adjustments
            macro_data = await fetch_macro_regime(country) if country else None
            composite_z = apply_macro_adjustments(composite_z, macro_data, info, country)

            # Step 7: Classify signal using percentiles
            signal_type = self.classifier.classify(composite_z)
            logger.debug(f"Signal type: {signal_type.value} (z-score={composite_z:.2f})")

            # Step 8: Calculate confidence and risk
            composite_score = 50 + composite_z * 15  # Convert to 0-100 scale
            confidence_level = calculate_confidence(
                technical_metrics, composite_score, stock_data
            )
            risk_factors = calculate_risk_factors(technical_metrics, info or {}, stock_data)

            # Step 9: Calculate upside/downside
            upside_potential_pct = calculate_upside_potential(
                composite_score, technical_metrics, macro_data
            )
            downside_risk_pct = calculate_downside_risk(
                composite_score, technical_metrics, macro_data
            )

            # Step 10: Generate analysis notes
            analysis_notes = generate_analysis_notes(
                yf_ticker,
                signal_type,
                composite_z,
                value_z,
                momentum_z,
                quality_z,
                growth_z,
                technical_metrics,
                macro_data,
            )

            # Step 11: Build StockSignalOutput
            signal_output = StockSignalOutput(
                signal_date=target_date.isoformat(),
                signal_type=signal_type,
                confidence_level=confidence_level,
                # Price data
                close_price=float(stock_data['Close'].iloc[-1]),
                open_price=float(stock_data['Open'].iloc[-1]) if 'Open' in stock_data else None,
                daily_return=(
                    float(stock_data['Close'].pct_change().iloc[-1])
                    if len(stock_data) > 1
                    else 0.0
                ),
                volume=float(stock_data['Volume'].iloc[-1]) if 'Volume' in stock_data else None,
                # Technical indicators
                volatility=technical_metrics.get('volatility'),
                rsi=technical_metrics.get('rsi'),
                # Metadata
                data_quality_score=calculate_data_quality(stock_data, info or {}),
                analysis_notes=analysis_notes,
                # Signal Drivers
                signal_drivers=SignalDrivers(
                    valuation_score=np.tanh(value_z / 2),
                    valuation_summary=f"Value z-score: {value_z:.2f} (B/P, E/P, FCF/P)",
                    momentum_score=np.tanh(momentum_z / 2),
                    momentum_summary=(
                        f"Momentum z-score: {momentum_z:.2f} "
                        f"(12-1m: {technical_metrics.get('momentum_12m_minus_1m', 0)*100:.1f}%)"
                    ),
                    quality_score=(np.tanh(quality_z / 2) + 1) / 2,
                    quality_summary=(
                        f"Quality z-score: {quality_z:.2f} "
                        f"(ROE, margins, Sharpe: {technical_metrics.get('sharpe_ratio', 0):.2f})"
                    ),
                    growth_score=np.tanh(growth_z / 2),
                    growth_summary=f"Growth z-score: {growth_z:.2f} (Revenue/earnings growth)",
                    technical_score=np.tanh(composite_z / 2),
                    technical_summary=f"Composite z-score: {composite_z:.2f} (equal-weighted 4-factor)",
                    analyst_score=None,
                    analyst_summary="Institutional four-factor model (no analyst data)",
                ),
                # Risk Factors
                risk_factors=risk_factors,
                # Price targets
                upside_potential_pct=upside_potential_pct,
                downside_risk_pct=downside_risk_pct,
            )

            logger.info(
                f"Generated {signal_type.value} signal for {yf_ticker} "
                f"(composite z-score: {composite_z:+.2f}, confidence: {confidence_level.value})"
            )

            return signal_output

        except Exception as e:
            logger.error(f"Error generating signal for {yf_ticker}: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    async def fetch_raw_fundamentals(
        self, yf_ticker: str, target_date: Optional[date_type] = None
    ) -> Optional[Tuple[Dict, Optional[Dict], str]]:
        """
        Fetch raw fundamental data WITHOUT calculating z-scores (Pass 1).

        Args:
            yf_ticker: Yahoo Finance ticker symbol
            target_date: Date to generate signal for (defaults to today)

        Returns:
            Tuple of (technical_metrics, info, country)
        """
        if target_date is None:
            target_date = date_type.today()

        logger.debug(f"Fetching raw fundamentals for {yf_ticker}")

        try:
            stock_data, benchmark_data, info = await fetch_price_data(yf_ticker, period="2y")

            if stock_data is None or len(stock_data) < 252:
                logger.warning(f"Insufficient data for {yf_ticker}")
                return None

            country = get_country_from_ticker(yf_ticker, info) if info else None
            risk_free_rate = self._get_risk_free_rate_cached(country)

            technical_metrics = calculate_technical_metrics(
                stock_data, benchmark_data, risk_free_rate
            )

            return (technical_metrics, info, country or "Unknown")

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {yf_ticker}: {e}")
            return None

    async def calculate_raw_composite_zscore(
        self, yf_ticker: str, target_date: Optional[date_type] = None
    ) -> Optional[Tuple[float, Dict, Optional[Dict], str, Dict[str, float]]]:
        """
        Calculate raw composite z-score WITHOUT classification (Pass 1).

        Returns raw z-score after factor combination and macro adjustments,
        but BEFORE cross-sectional standardization.

        Args:
            yf_ticker: Yahoo Finance ticker symbol
            target_date: Date to generate signal for

        Returns:
            Tuple of (raw_composite_z, technical_metrics, info, country, factor_zscores)
        """
        if target_date is None:
            target_date = date_type.today()

        logger.debug(f"Calculating raw z-score for {yf_ticker}")

        try:
            stock_data, benchmark_data, info = await fetch_price_data(yf_ticker, period="2y")

            if stock_data is None or len(stock_data) < 252:
                logger.warning(f"Insufficient data for {yf_ticker}")
                return None

            country = get_country_from_ticker(yf_ticker, info) if info else None
            risk_free_rate = self._get_risk_free_rate_cached(country)

            technical_metrics = calculate_technical_metrics(
                stock_data, benchmark_data, risk_free_rate
            )
            self.last_technical_metrics = technical_metrics

            # Calculate factors
            value_z = calculate_value_factor(info, stock_data, self.cross_sectional_stats)
            momentum_z = calculate_momentum_factor(technical_metrics, self.cross_sectional_stats)
            quality_z = calculate_quality_factor(
                info, technical_metrics, country, self.cross_sectional_stats
            )
            growth_z = calculate_growth_factor(info, country, self.cross_sectional_stats)

            logger.debug(
                f"{yf_ticker} factor z-scores: Value={value_z:.2f}, "
                f"Momentum={momentum_z:.2f}, Quality={quality_z:.2f}, Growth={growth_z:.2f}"
            )

            # Multi-factor combination
            composite_z = (value_z + momentum_z + quality_z + growth_z) / 4.0

            # Apply shrinkage
            shrinkage_factor = 0.7
            composite_z = shrinkage_factor * composite_z + (1 - shrinkage_factor) * 0

            # Apply macro adjustments
            macro_data = await fetch_macro_regime(country) if country else None
            composite_z = apply_macro_adjustments(composite_z, macro_data, info, country)

            logger.debug(f"{yf_ticker} raw composite z-score: {composite_z:.2f}")

            factor_zscores = {
                'value': value_z,
                'momentum': momentum_z,
                'quality': quality_z,
                'growth': growth_z,
            }

            return (composite_z, technical_metrics, info, country or "Unknown", factor_zscores)

        except Exception as e:
            logger.error(f"Error calculating raw z-score for {yf_ticker}: {e}")
            return None

    def standardize_zscores_cross_sectional(
        self,
        raw_zscores: list,
        winsorize_threshold: float = 10.0
    ) -> np.ndarray:
        """
        Apply cross-sectional standardization to raw z-scores.

        Implements two-step standardization per Chapter 5:
        1. Winsorization: Clip extreme values at ±winsorize_threshold
        2. StandardScaler: Ensure mean=0, std=1 across the universe

        Args:
            raw_zscores: List of raw composite z-scores
            winsorize_threshold: Threshold for winsorization (default: ±10σ)

        Returns:
            Standardized z-scores as numpy array (mean=0, std=1)
        """
        from sklearn.preprocessing import StandardScaler

        # Convert to numpy array
        raw_array = np.array(raw_zscores).reshape(-1, 1)

        # Debug: Identify outliers
        outlier_threshold = 10.0
        outlier_mask = np.abs(raw_array) > outlier_threshold
        outlier_indices = np.where(outlier_mask)[0]

        if len(outlier_indices) > 0:
            logger.warning(f"🚨 Found {len(outlier_indices)} extreme outliers (|z| > {outlier_threshold}):")
            for idx in outlier_indices:
                z_value = float(raw_array[idx])
                logger.warning(f"  Index {idx:3d}: z-score = {z_value:+8.2f}")

        # Show top 10 most extreme values
        sorted_indices = np.argsort(np.abs(raw_array.flatten()))[-10:][::-1]
        logger.info("📊 Top 10 most extreme z-scores:")
        for idx in sorted_indices:
            z_value = float(raw_array[idx])
            logger.info(f"  Index {idx:3d}: z-score = {z_value:+8.2f}")

        # Step 1: Winsorization (clip extreme outliers)
        winsorized = np.clip(raw_array, -winsorize_threshold, winsorize_threshold)

        # Count winsorized values
        n_winsorized = np.sum(raw_array != winsorized)
        if n_winsorized > 0:
            logger.info(f"Winsorized {n_winsorized} extreme values at ±{winsorize_threshold}σ")

        # Step 2: StandardScaler (ensure mean=0, std=1)
        scaler = StandardScaler()
        standardized = scaler.fit_transform(winsorized).flatten()

        return standardized

    def finalize_run(self, universe_description: Optional[str] = None) -> None:
        """
        Finalize the current signal generation run by saving the distribution.

        Call this at the end of each batch/run to ensure cumulative tracking.

        Args:
            universe_description: Optional description of the stock universe analyzed
        """
        self.classifier.finalize_run(universe_description)
