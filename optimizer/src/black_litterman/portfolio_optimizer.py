#!/usr/bin/env python3
import logging
import asyncio
from typing import List, Tuple, Dict, Optional, Protocol, Sequence
from datetime import date as date_type, timedelta

import numpy as np
import pandas as pd

from dataclasses import dataclass

from optimizer.database.database import database_manager
from optimizer.database.models.stock_signals import StockSignal
from optimizer.database.models.universe import Instrument
from optimizer.database.models.macro_regime import CountryRegimeAssessment
from optimizer.database.models.portfolio import PortfolioPosition as DBPortfolioPosition
from optimizer.src.black_litterman.view_generator import ViewGenerator, BlackLittermanView
from optimizer.src.black_litterman.equilibrium import (
    calculate_equilibrium_prior,
    estimate_risk_aversion,
    adjust_risk_aversion_for_regime,
    fetch_market_caps_from_db,
)
from optimizer.src.black_litterman.optimizer.risk_models import ledoit_wolf_shrinkage
from optimizer.src.stock_analyzer.risk_free_rate import get_risk_free_rate
from optimizer.src.black_litterman.optimizer.black_litterman import BlackLittermanModel
from optimizer.src.yfinance.client import YFinanceClient

logger = logging.getLogger(__name__)


class PositionLike(Protocol):
    """Protocol for position-like objects that can be optimized."""

    ticker: str
    weight: float
    country: Optional[str] = None


@dataclass
class PortfolioPosition:
    """Lightweight portfolio position for optimizer output."""

    ticker: str
    weight: float
    instrument_id: Optional[str] = None
    signal_id: Optional[str] = None
    signal_type: Optional[str] = None
    conviction_tier: Optional[int] = 2  # 1=highest, 2=medium, 3=lowest
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: str = "USA"
    exchange: Optional[str] = None
    yfinance_ticker: Optional[str] = None
    price: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    volatility: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    max_drawdown: Optional[float] = None
    annualized_return: Optional[float] = None
    confidence_level: str = "medium"
    data_quality_score: Optional[float] = None
    selection_reason: str = ""


class BlackLittermanOptimizer:
    """
    Optimizes portfolio weights using Black-Litterman framework.
    """

    def __init__(
        self,
        lookback_period: str = "5y",  # 5 years of historical data for covariance
        tau: float = 0.025,  # Prior uncertainty parameter
        use_regime_adjustment: bool = True,
        max_sector_weight: float = 0.15,  # 15% max per sector
        max_position_weight: float = 0.10,  # 10% max per position
        min_position_weight: float = 0.0,  # Long-only (no shorts)
    ):
        """
        Initialize Black-Litterman optimizer with BAML view generation.
        """
        self.lookback_period = lookback_period
        self.tau = tau
        self.use_regime_adjustment = use_regime_adjustment
        self.max_sector_weight = max_sector_weight
        self.max_position_weight = max_position_weight
        self.min_position_weight = min_position_weight

        self.view_generator = ViewGenerator()
        self.yfinance_client = YFinanceClient()

        # State
        self.positions: Sequence[PositionLike] = []
        self.stock_signals: Dict[str, StockSignal] = {}
        self.instruments: Dict[str, Instrument] = {}
        self.price_history: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.equilibrium_returns: Optional[pd.Series] = None
        self.posterior_returns: Optional[pd.Series] = None
        self.optimized_weights: Optional[pd.Series] = None
        self.views: List[Tuple[StockSignal, BlackLittermanView]] = []

    def calculate_portfolio_weighted_risk_free_rate(
        self, positions: List[PositionLike]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate portfolio-weighted risk-free rate using country-specific rates.
        """
        logger.info("Calculating portfolio-weighted risk-free rate...")

        # Count positions by country (equal weight for rate calculation)
        country_counts = {}
        for pos in positions:
            country = pos.country or "USA"  # Default to USA if missing
            country_counts[country] = country_counts.get(country, 0) + 1

        total_positions = len(positions)
        if total_positions == 0:
            logger.warning("No positions provided, using USA rate as fallback")
            usa_rate = get_risk_free_rate("USA")
            return usa_rate, {"USA": usa_rate}

        # Get risk-free rate for each country
        country_rates = {}
        for country in country_counts.keys():
            try:
                rate = get_risk_free_rate(country)
                country_rates[country] = rate
                logger.info(f"  {country:15s}: {rate:.4f} ({rate*100:.2f}%)")
            except Exception as e:
                logger.warning(f"  {country:15s}: Failed to get rate, using USA fallback: {e}")
                country_rates[country] = get_risk_free_rate("USA")

        # Calculate weighted average based on position counts
        weighted_rate = sum(
            country_rates[country] * (country_counts[country] / total_positions)
            for country in country_counts.keys()
        )

        logger.info(
            f"✓ Portfolio-weighted risk-free rate: {weighted_rate:.4f} ({weighted_rate*100:.2f}%)"
        )
        logger.info(f"  Countries: {len(country_rates)} | Positions: {total_positions}")

        return weighted_rate, country_rates

    def fetch_signal_data(
        self, tickers: List[str], signal_date: Optional[date_type] = None
    ) -> Dict[str, Tuple[StockSignal, Instrument]]:
        """
        Fetch stock signals and instruments for portfolio tickers.
        """
        signal_date = signal_date or date_type.today()

        logger.info(f"Fetching signal data for {len(tickers)} tickers (target date: {signal_date})")

        with database_manager.get_session() as session:
            from sqlalchemy import select, func, desc
            from sqlalchemy.orm import joinedload

            # Try exact date first
            stmt = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .where(Instrument.ticker.in_(tickers))
                .where(StockSignal.signal_date == signal_date)
            )

            results = session.execute(stmt).all()

            signal_data = {inst.ticker: (signal, inst) for signal, inst in results}

            # For missing tickers, get most recent signal
            if len(signal_data) < len(tickers):
                missing_tickers = set(tickers) - set(signal_data.keys())
                logger.info(
                    f"  Exact date not found for {len(missing_tickers)} tickers, fetching most recent..."
                )

                # Get most recent signal for each missing ticker
                for ticker in missing_tickers:
                    recent_stmt = (
                        select(StockSignal, Instrument)
                        .join(Instrument, StockSignal.instrument_id == Instrument.id)
                        .where(Instrument.ticker == ticker)
                        .order_by(desc(StockSignal.signal_date))
                        .limit(1)
                    )

                    recent_result = session.execute(recent_stmt).first()

                    if recent_result:
                        signal, inst = recent_result
                        signal_data[ticker] = (signal, inst)
                        logger.debug(f"    {ticker}: Using signal from {signal.signal_date}")

        logger.info(f"✓ Fetched signal data for {len(signal_data)} tickers")

        if len(signal_data) < len(tickers):
            still_missing = set(tickers) - set(signal_data.keys())
            logger.warning(f"⚠️  No signals found for {len(still_missing)} tickers: {still_missing}")

        return signal_data

    def get_yfinance_ticker_mapping(self, tickers: List[str]) -> Dict[str, Optional[str]]:
        """
        Get mapping from Trading212 tickers to Yahoo Finance tickers.
        """
        logger.debug(f"Fetching yfinance ticker mapping for {len(tickers)} tickers")

        with database_manager.get_session() as session:
            from sqlalchemy import select

            stmt = select(Instrument.ticker, Instrument.yfinance_ticker).where(
                Instrument.ticker.in_(tickers)
            )

            results = session.execute(stmt).all()

            ticker_mapping = {t212_ticker: yf_ticker for t212_ticker, yf_ticker in results}

        # Log warnings for missing mappings
        missing = set(tickers) - set(ticker_mapping.keys())
        if missing:
            logger.warning(f"No instrument records found for {len(missing)} tickers: {missing}")

        # Log warnings for None yfinance tickers
        none_yf = [t for t, yf in ticker_mapping.items() if yf is None]
        if none_yf:
            logger.warning(f"No yfinance_ticker mapping for {len(none_yf)} tickers: {none_yf}")

        logger.debug(f"✓ Retrieved {len(ticker_mapping)} ticker mappings")
        return ticker_mapping

    def fetch_price_history(
        self, tickers: List[str], lookback_period: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical price data for covariance estimation.
        """
        lookback_period = lookback_period or self.lookback_period

        logger.info(f"Fetching {lookback_period} of price history for {len(tickers)} tickers")

        # Get yfinance ticker mapping
        ticker_mapping = self.get_yfinance_ticker_mapping(tickers)

        prices = {}
        failed_tickers = []

        for ticker in tickers:
            # Get yfinance ticker (or skip if not available)
            yf_ticker = ticker_mapping.get(ticker)

            if yf_ticker is None:
                logger.warning(f"Skipping {ticker}: no yfinance ticker mapping")
                failed_tickers.append(ticker)
                continue

            try:
                data = self.yfinance_client.fetch_history(yf_ticker, period=self.lookback_period)

                # Verify we got data
                if data is None or data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Use adjusted close if available, otherwise close
                if "Adj Close" in data.columns:
                    price_series = data["Adj Close"]
                else:
                    price_series = data["Close"]

                # CRITICAL FIX: Normalize timezone for cross-market portfolios
                # US stocks use America/New_York, European stocks use Europe/Paris, etc.
                # This prevents proper date alignment even when calendar dates match
                # Remove timezone before adding to price dict
                if (
                    isinstance(price_series.index, pd.DatetimeIndex)
                    and price_series.index.tz is not None
                ):
                    price_series = price_series.copy()
                    price_series.index = price_series.index.tz_localize(None)

                prices[ticker] = price_series

            except Exception as e:
                logger.warning(f"Failed to fetch price history for {ticker}: {e}")
                failed_tickers.append(ticker)

        if not prices:
            raise ValueError("Failed to fetch price history for any tickers")

        price_df = pd.DataFrame(prices)

        # Forward fill missing data (max 5 days)
        price_df = price_df.ffill(limit=5)

        # Drop any remaining NaN
        price_df = price_df.dropna()

        logger.info(
            f"✓ Fetched price history: {len(price_df)} days × {len(price_df.columns)} tickers"
        )

        if failed_tickers:
            logger.warning(f"Failed tickers: {failed_tickers}")

        return price_df

    def calculate_covariance(
        self, prices: pd.DataFrame, method: str = "ledoit_wolf"
    ) -> pd.DataFrame:
        """
        Calculate robust covariance matrix.
        """
        logger.info(f"Calculating covariance matrix using {method} method")

        if method == "ledoit_wolf":
            cov_matrix = ledoit_wolf_shrinkage(prices, frequency=252)
        else:
            from optimizer.src.black_litterman.optimizer.risk_models import sample_cov

            cov_matrix = sample_cov(prices, frequency=252)

        logger.info(f"✓ Covariance matrix: {cov_matrix.shape[0]} × {cov_matrix.shape[1]}")

        return cov_matrix

    def calculate_risk_aversion(
        self,
        market_ticker: str = "^GSPC",  # S&P 500
        regime: Optional[str] = None,
        recession_risk: Optional[float] = None,
        risk_free_rate: float = 0.045,  # Default to USA rate if not provided
    ) -> float:
        """
        Calculate risk aversion coefficient (delta).
        """
        logger.info("Calculating risk aversion coefficient")

        # Fetch market data
        try:
            market_data = self.yfinance_client.fetch_history(
                market_ticker, period=self.lookback_period
            )

            # Verify we got data
            if market_data is None or market_data.empty:
                raise ValueError("No market data returned")

            # Calculate returns
            market_returns = market_data["Close"].pct_change().dropna()

            # Estimate base risk aversion
            delta = estimate_risk_aversion(market_returns, risk_free_rate)

        except Exception as e:
            logger.warning(f"Failed to estimate risk aversion from market: {e}")
            logger.warning("Using default delta=2.5")
            delta = 2.5

        # Adjust for regime
        if self.use_regime_adjustment and regime:
            delta = adjust_risk_aversion_for_regime(delta, regime, recession_risk)

        logger.info(f"✓ Risk aversion coefficient: δ={delta:.2f}")

        return delta

    def calculate_equilibrium(
        self,
        tickers: List[str],
        cov_matrix: pd.DataFrame,
        risk_aversion: float,
        risk_free_rate: float,
    ) -> pd.Series:
        """
        Calculate market-implied equilibrium returns.
        """
        logger.info("Calculating equilibrium prior returns")

        # Convert Trading212 tickers to yfinance tickers
        ticker_mapping = self.get_yfinance_ticker_mapping(tickers)

        # Get list of valid yfinance tickers
        yf_tickers = [yf_ticker for yf_ticker in ticker_mapping.values() if yf_ticker]

        logger.info(f"Fetching market caps for {len(yf_tickers)} stocks via yfinance tickers")

        # Fetch market caps using yfinance tickers
        market_caps_yf = fetch_market_caps_from_db(tickers=yf_tickers)

        # Convert back to Trading212 tickers for alignment with covariance matrix
        # Create reverse mapping: yfinance -> Trading212
        yf_to_t212: Dict[str, str] = {}
        for t212, yf in ticker_mapping.items():
            if yf:  # Only include valid yfinance tickers
                yf_to_t212[yf] = t212

        market_caps = pd.Series(dtype=float)
        for yf_ticker, cap in market_caps_yf.items():
            t212_ticker = yf_to_t212.get(str(yf_ticker))
            if t212_ticker:
                market_caps[t212_ticker] = cap

        logger.info(f"Mapped {len(market_caps)} market caps to Trading212 tickers")

        # Check if we have market caps for all tickers
        missing_caps = set(tickers) - set(market_caps.index)
        if missing_caps:
            logger.warning(f"Missing market caps for {len(missing_caps)} tickers: {missing_caps}")
            logger.warning("Using equal weights for missing market caps")

            # Use equal weights for missing tickers
            avg_cap = market_caps.mean() if len(market_caps) > 0 else 1.0
            for ticker in missing_caps:
                market_caps[ticker] = avg_cap

        # Calculate equilibrium
        pi = calculate_equilibrium_prior(market_caps, cov_matrix, risk_aversion, risk_free_rate)

        return pi

    async def generate_views(
        self,
        signal_data: Dict[str, Tuple[StockSignal, Instrument]],
        regime: Optional[str] = None,
    ) -> List[Tuple[StockSignal, BlackLittermanView]]:
        """
        Generate Black-Litterman views using BAML with country-specific macro regimes.
        """
        logger.info(f"Generating Black-Litterman views for {len(signal_data)} stocks")

        # Group stocks by country
        from collections import defaultdict
        from optimizer.src.stock_analyzer.data.fetchers import get_country_from_ticker

        stocks_by_country = defaultdict(list)

        for ticker, (signal, inst) in signal_data.items():
            # Determine country from yfinance ticker (e.g., FRE.DE -> Germany, RR.L -> UK)
            country = (
                get_country_from_ticker(signal.yfinance_ticker, info=None)
                if signal.yfinance_ticker
                else "USA"
            )
            stocks_by_country[country].append((signal, inst))

        logger.info(f"Stocks grouped by country:")
        for country, stocks in sorted(stocks_by_country.items()):
            logger.info(f"  {country:15s}: {len(stocks):2d} stocks")

        # Fetch country-specific macro regimes from database
        country_regimes = {}

        try:
            with database_manager.get_session() as session:
                from sqlalchemy import select

                for country in stocks_by_country.keys():
                    stmt = (
                        select(CountryRegimeAssessment)
                        .where(CountryRegimeAssessment.country == country)
                        .order_by(CountryRegimeAssessment.assessment_timestamp.desc())
                        .limit(1)
                    )

                    result = session.execute(stmt).scalar_one_or_none()

                    if result:
                        country_regimes[country] = result.regime
                        logger.info(f"  ✓ {country}: {result.regime} regime")
                    else:
                        # Fallback to provided regime or MID_CYCLE
                        country_regimes[country] = regime or "MID_CYCLE"
                        logger.warning(
                            f"  ⚠️  {country}: No regime assessment found, using {country_regimes[country]}"
                        )

        except Exception as e:
            logger.warning(f"Failed to fetch country regimes from database: {e}")
            # Fallback: use provided regime or MID_CYCLE for all countries
            for country in stocks_by_country.keys():
                country_regimes[country] = regime or "MID_CYCLE"

        # Generate views with country-specific macro regimes
        from baml_client.types import MacroRegimeContext as MacroRegimeContextType

        views = []
        tasks = []

        for country, stocks in stocks_by_country.items():
            # Fetch country-specific macro regime context from ViewGenerator
            macro_regime_context = self.view_generator.fetch_macro_regime(country=country)

            # Provide default if none found
            if macro_regime_context is None:
                logger.warning(f"No macro regime context found for {country}, using default")
                macro_regime_context = MacroRegimeContextType(
                    current_regime=country_regimes[country], regime_confidence=0.5
                )

            # Generate views for each stock in this country
            for signal, inst in stocks:
                sector_context = self.view_generator.build_sector_context(
                    signal.sector or "Unknown", [s for s, _ in stocks]
                )

                task = self.view_generator._generate_single_view(
                    signal, inst, macro_regime_context, sector_context
                )
                tasks.append((signal, inst, country, task))

        # Execute all tasks
        for signal, inst, country, task in tasks:
            try:
                view = await task
                if view:
                    views.append((signal, view))
            except Exception as e:
                logger.warning(f"Failed to generate view for {inst.ticker} ({country}): {e}")

        logger.info(f"✓ Generated {len(views)} Black-Litterman views")

        return views

    def construct_bl_inputs(
        self,
        views: List[Tuple[StockSignal, BlackLittermanView]],
        universe_tickers: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct P, Q, Omega matrices for Black-Litterman.
        """
        return self.view_generator.construct_matrices(views, universe_tickers)

    def _build_sector_mapping(
        self, positions: Sequence[PositionLike], tickers: List[str]
    ) -> Dict[str, List[int]]:
        """
        Build mapping from sectors to ticker indices.
        """
        from collections import defaultdict

        sector_indices = defaultdict(list)

        for position in positions:
            if position.ticker in tickers:
                idx = tickers.index(position.ticker)
                sector = getattr(position, "sector", None) or "Unknown"
                sector_indices[sector].append(idx)

        logger.debug(f"Sector mapping: {len(sector_indices)} sectors")
        for sector, indices in sector_indices.items():
            logger.debug(f"  {sector}: {len(indices)} stocks (indices {indices})")

        return dict(sector_indices)

    def optimize_with_constraints(
        self,
        posterior_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float,
        positions: Sequence[PositionLike],
        max_sector_weight: float = 0.15,
        max_position_weight: float = 0.10,
        min_position_weight: float = 0.0,
    ) -> pd.Series:
        """
        Black-Litterman optimization with sector and position constraints.
        """
        from scipy.optimize import minimize
        from collections import defaultdict

        logger.info("Running constrained Black-Litterman optimization")
        logger.info(f"  Constraints:")
        logger.info(f"    - Fully invested: Σw = 100%")
        logger.info(
            f"    - Position limits: {min_position_weight:.1%} ≤ w_i ≤ {max_position_weight:.1%}"
        )
        logger.info(f"    - Sector limits: Σw_sector ≤ {max_sector_weight:.1%}")

        tickers = list(posterior_returns.index)
        n = len(tickers)

        # Convert to numpy arrays
        mu = np.array(posterior_returns.values, dtype=float).reshape(-1, 1)
        Sigma = np.array(covariance_matrix.values, dtype=float)

        # Build sector mapping
        sector_mapping = self._build_sector_mapping(positions, tickers)

        # Objective: minimize (1/2) w^T (δΣ) w - μ^T w
        # Equivalent to maximize: μ^T w - (δ/2) w^T Σ w
        def objective(w):
            portfolio_variance = w @ (risk_aversion * Sigma) @ w
            portfolio_return = mu.flatten() @ w
            return 0.5 * portfolio_variance - portfolio_return

        # Gradient for faster convergence
        def gradient(w):
            return (risk_aversion * Sigma) @ w - mu.flatten()

        # Constraints
        constraints = []

        # 1. Fully invested constraint: Σw = 1.0
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(n),
            }
        )

        # 2. Sector constraints: Σw_i ≤ max_sector_weight for each sector
        for sector, indices in sector_mapping.items():

            def sector_constraint(w, idx=indices, name=sector):
                return max_sector_weight - np.sum(w[idx])

            def sector_jac(w, idx=indices):
                jac = np.zeros(n)
                jac[idx] = -1.0
                return jac

            constraints.append({"type": "ineq", "fun": sector_constraint, "jac": sector_jac})

        # 3. Position bounds: min_weight ≤ w_i ≤ max_weight
        bounds = [(min_position_weight, max_position_weight) for _ in range(n)]

        # Initial guess: equal-weighted or current weights
        w0 = np.ones(n) / n

        logger.info(f"\nOptimizing {n} assets with {len(sector_mapping)} sector constraints...")

        # Optimize
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
            logger.warning("Returning equal-weighted portfolio as fallback")
            optimal_weights = np.ones(n) / n
        else:
            logger.info(f"✓ Optimization converged in {result.nit} iterations")
            optimal_weights = result.x

        # Convert to Series
        optimized_weights = pd.Series(optimal_weights, index=tickers)

        # Verify constraints
        total_weight = optimized_weights.sum()
        logger.info(f"\nVerifying constraints:")
        logger.info(f"  Total weight: {total_weight:.4f} (target: 1.0000)")

        # Check sector weights
        sector_weights = defaultdict(float)
        for ticker, weight in optimized_weights.items():
            position = next((p for p in positions if p.ticker == ticker), None)
            if position:
                sector = getattr(position, "sector", None) or "Unknown"
                sector_weights[sector] += weight

        violations = []
        logger.info(f"\n  Sector weights:")
        for sector in sorted(sector_weights.keys(), key=lambda s: sector_weights[s], reverse=True):
            weight = sector_weights[sector]
            status = "✅" if weight <= max_sector_weight + 0.001 else "❌"  # Allow tiny tolerance
            logger.info(f"    {status} {sector:25s}: {weight:5.1%}")

            if weight > max_sector_weight + 0.001:
                violations.append(f"{sector}: {weight:.1%}")

        if violations:
            logger.error(f"❌ Sector constraint violations detected: {violations}")
            logger.error("This should not happen with SLSQP. Check constraint implementation.")
        else:
            logger.info(f"  ✅ All sector constraints satisfied")

        return optimized_weights

    def optimize_portfolio(
        self, positions: List[PositionLike], signal_date: Optional[date_type] = None
    ) -> Tuple[List[PortfolioPosition], Dict]:
        """
        Optimize portfolio weights using Black-Litterman.

        Args:
            positions: Portfolio positions (must have ticker, weight, and optionally country)
            signal_date: Signal date (defaults to yesterday)

        Returns:
            Tuple of (optimized_positions, metrics_dict)
        """
        logger.info("=" * 100)
        logger.info(" " * 30 + "BLACK-LITTERMAN OPTIMIZATION")
        logger.info("=" * 100)

        self.positions = positions
        tickers = [p.ticker for p in positions]

        logger.info(f"\nOptimizing portfolio with {len(tickers)} stocks:")
        logger.info(f"  Tickers: {', '.join(tickers)}")

        # Step 1: Fetch signal data
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 1/8] FETCHING STOCK SIGNAL DATA")
        logger.info("─" * 100)

        signal_data = self.fetch_signal_data(tickers, signal_date)

        if len(signal_data) < len(tickers) * 0.8:
            raise ValueError(f"Insufficient signal data: {len(signal_data)}/{len(tickers)} tickers")

        # Step 2: Calculate portfolio-weighted risk-free rate
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 2/8] CALCULATING PORTFOLIO-WEIGHTED RISK-FREE RATE")
        logger.info("─" * 100)

        risk_free_rate, country_rates = self.calculate_portfolio_weighted_risk_free_rate(positions)

        # Step 3: Fetch price history
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 3/8] FETCHING PRICE HISTORY")
        logger.info("─" * 100)

        self.price_history = self.fetch_price_history(list(signal_data.keys()))

        # Step 4: Calculate covariance matrix
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 4/8] CALCULATING COVARIANCE MATRIX (LEDOIT-WOLF)")
        logger.info("─" * 100)

        self.covariance_matrix = self.calculate_covariance(self.price_history)

        # Step 5: Calculate risk aversion
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 5/8] CALCULATING RISK AVERSION")
        logger.info("─" * 100)

        # Fetch macro regime for risk aversion adjustment
        regime = None
        recession_risk = None

        try:
            with database_manager.get_session() as session:
                from sqlalchemy import select

                stmt = (
                    select(CountryRegimeAssessment)
                    .where(CountryRegimeAssessment.country == "USA")
                    .order_by(CountryRegimeAssessment.assessment_timestamp.desc())
                    .limit(1)
                )

                result = session.execute(stmt).scalar_one_or_none()
                if result:
                    regime = result.regime
                    recession_risk = result.recession_risk_6m

        except Exception as e:
            logger.warning(f"Failed to fetch macro regime: {e}")

        risk_aversion = self.calculate_risk_aversion(
            regime=regime, recession_risk=recession_risk, risk_free_rate=risk_free_rate
        )

        # Step 6: Calculate equilibrium prior returns
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 6/8] CALCULATING EQUILIBRIUM PRIOR RETURNS")
        logger.info("─" * 100)

        logger.info("Calculating market equilibrium returns (will be adjusted by BAML views)")
        self.equilibrium_returns = self.calculate_equilibrium(
            list(self.covariance_matrix.index),
            self.covariance_matrix,
            risk_aversion,
            risk_free_rate,
        )

        # Step 7: Generate Black-Litterman views using BAML
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 7/8] GENERATING BLACK-LITTERMAN VIEWS (BAML)")
        logger.info("─" * 100)

        # Generate AI views using BAML (costs money via LLM calls)
        self.views = asyncio.run(self.generate_views(signal_data, regime))

        if not self.views:
            logger.warning("No views generated by BAML, falling back to equilibrium-only")
            self.posterior_returns = self.equilibrium_returns
        else:
            # Step 8: Construct BL matrices and run optimization
            logger.info("\n" + "─" * 100)
            logger.info("[STEP 8/8] RUNNING BLACK-LITTERMAN BAYESIAN UPDATE")
            logger.info("─" * 100)

            P, Q, Omega = self.construct_bl_inputs(self.views, list(self.covariance_matrix.index))

            logger.info(f"View matrices: P={P.shape}, Q={Q.shape}, Omega={Omega.shape}")

            # Run Black-Litterman
            bl_model = BlackLittermanModel(
                cov_matrix=self.covariance_matrix,
                pi=self.equilibrium_returns,
                P=P,
                Q=Q,
                omega=Omega,
                tau=self.tau,
                risk_aversion=risk_aversion,
            )

            # Get posterior returns
            self.posterior_returns = bl_model.bl_returns()

        logger.info("\nPosterior expected returns:")
        for ticker, ret in self.posterior_returns.items():
            logger.info(f"  {ticker:10s}: {ret:+.2%}")

        # Get optimized weights using constrained optimization
        logger.info("\n" + "─" * 100)
        logger.info("APPLYING SECTOR-CONSTRAINED OPTIMIZATION")
        logger.info("─" * 100)

        try:
            # Use constrained optimization with sector limits
            self.optimized_weights = self.optimize_with_constraints(
                posterior_returns=self.posterior_returns,
                covariance_matrix=self.covariance_matrix,
                risk_aversion=risk_aversion,
                positions=positions,
                max_sector_weight=self.max_sector_weight,
                max_position_weight=self.max_position_weight,
                min_position_weight=self.min_position_weight,
            )

            logger.info("\nOptimized weights (sector-constrained):")
            for ticker, weight in self.optimized_weights.items():
                logger.info(f"  {ticker:10s}: {weight:.2%}")

        except Exception as e:
            logger.error(f"Failed to calculate optimized weights: {e}")
            logger.warning("Cannot optimize without weights, returning original positions")
            import traceback

            traceback.print_exc()
            # Return original positions without optimization
            return positions, {
                "method": "no_optimization",
                "reason": "weight_calculation_failed",
                "error": str(e),
            }

        # Step 8: Create optimized positions
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 9/9] CREATING OPTIMIZED PORTFOLIO")
        logger.info("─" * 100)

        # At this point optimized_weights is guaranteed to be non-None (or we returned early)
        assert self.optimized_weights is not None
        assert self.equilibrium_returns is not None
        assert self.posterior_returns is not None

        optimized_positions = []

        for position in positions:
            if position.ticker in self.optimized_weights:
                # Get optimized weight and convert to float
                optimized_weight = float(self.optimized_weights[position.ticker])

                # Create new position with optimized weight
                # Use getattr for optional attributes that may not exist in PositionLike
                optimized_pos = PortfolioPosition(
                    ticker=position.ticker,
                    weight=optimized_weight,
                    instrument_id=getattr(position, "instrument_id", None),
                    signal_id=getattr(position, "signal_id", None),
                    signal_type=getattr(position, "signal_type", None),
                    conviction_tier=getattr(position, "conviction_tier", "medium"),
                    company_name=getattr(position, "company_name", None),
                    sector=getattr(position, "sector", None),
                    industry=getattr(position, "industry", None),
                    country=getattr(position, "country", "USA"),
                    exchange=getattr(position, "exchange", None),
                    yfinance_ticker=getattr(position, "yfinance_ticker", None),
                    price=getattr(position, "price", None),
                    sharpe_ratio=getattr(position, "sharpe_ratio", None),
                    sortino_ratio=getattr(position, "sortino_ratio", None),
                    volatility=getattr(position, "volatility", None),
                    alpha=getattr(position, "alpha", None),
                    beta=getattr(position, "beta", None),
                    max_drawdown=getattr(position, "max_drawdown", None),
                    annualized_return=getattr(position, "annualized_return", None),
                    confidence_level=getattr(position, "confidence_level", "medium"),
                    data_quality_score=getattr(position, "data_quality_score", None),
                    selection_reason=f"Black-Litterman optimized: {optimized_weight:.2%}",
                )
                optimized_positions.append(optimized_pos)

        # Calculate metrics
        total_weight = sum(p.weight for p in optimized_positions)

        metrics = {
            "method": "black_litterman",
            "total_positions": len(optimized_positions),
            "total_weight": total_weight,
            "views_count": len(self.views),
            "risk_aversion": risk_aversion,
            "tau": self.tau,
            "regime": regime,
            "risk_free_rate": risk_free_rate,
            "risk_free_rate_by_country": country_rates,
            "equilibrium_returns": self.equilibrium_returns.to_dict(),
            "posterior_returns": self.posterior_returns.to_dict(),
            "optimized_weights": self.optimized_weights.to_dict(),
            "weight_changes": {
                p.ticker: float(self.optimized_weights.get(p.ticker, 0.0)) - p.weight
                for p in positions
                if p.ticker in self.optimized_weights
            },
        }

        logger.info("\n✓ Black-Litterman optimization complete")
        logger.info(f"  Method: Black-Litterman with BAML AI views")
        logger.info(f"  Total weight: {total_weight:.1%}")
        logger.info(f"  Views generated: {len(self.views)}")
        logger.info(f"  Risk aversion: {risk_aversion:.2f}")
        logger.info(f"  Portfolio risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")

        logger.info("\nWeight changes (optimized - original):")
        for ticker, change in metrics["weight_changes"].items():
            logger.info(f"  {ticker:10s}: {change:+.2%}")

        logger.info("=" * 100)

        return optimized_positions, metrics

    def save_to_database(
        self,
        optimized_positions: List[PortfolioPosition],
        original_positions: List[PortfolioPosition],
        metrics: Dict,
        portfolio_name: Optional[str] = None,
        portfolio_date: Optional[date_type] = None,
    ) -> str:
        """
        Save optimized portfolio to database.

        Args:
            optimized_positions: List of optimized portfolio positions
            original_positions: List of original positions (for weight changes)
            metrics: Metrics dictionary from optimize_portfolio()
            portfolio_name: Optional portfolio name
            portfolio_date: Optional portfolio date (defaults to today)

        Returns:
            Portfolio UUID as string
        """
        from decimal import Decimal
        from optimizer.database.models.portfolio import (
            Portfolio,
            PortfolioPosition as DBPortfolioPosition,
        )

        portfolio_date = portfolio_date or date_type.today()

        logger.info("\n" + "=" * 100)
        logger.info("SAVING PORTFOLIO TO DATABASE")
        logger.info("=" * 100)

        with database_manager.get_session() as session:
            # Create portfolio record
            portfolio = Portfolio(
                portfolio_date=portfolio_date,
                name=portfolio_name or f"BL Optimized {len(optimized_positions)}-Stock",
                optimization_method="black_litterman",
                used_baml_views=True,  # Always True now
                used_factor_priors=False,  # Never used anymore
                total_positions=len(optimized_positions),
                total_weight=Decimal(str(metrics.get("total_weight", 1.0))),
                risk_aversion=(
                    Decimal(str(metrics.get("risk_aversion", 0.0)))
                    if metrics.get("risk_aversion")
                    else None
                ),
                tau=Decimal(str(self.tau)),
                regime=metrics.get("regime"),
                metrics={
                    "views_count": metrics.get("views_count", 0),
                    "weight_changes": metrics.get("weight_changes", {}),
                },
            )

            session.add(portfolio)
            session.flush()  # Get portfolio ID

            logger.info(f"\nCreated portfolio: {portfolio.id}")
            logger.info(f"  Name: {portfolio.name}")
            logger.info(f"  Date: {portfolio.portfolio_date}")
            logger.info(f"  Method: {portfolio.optimization_method}")
            logger.info(f"  Total positions: {portfolio.total_positions}")

            # Create position records
            equilibrium_returns = metrics.get("equilibrium_returns", {})
            posterior_returns = metrics.get("posterior_returns", {})

            # Create mapping of original weights
            original_weights = {p.ticker: p.weight for p in original_positions}

            logger.info(f"\nSaving {len(optimized_positions)} positions...")

            for position in optimized_positions:
                # Get original weight for this ticker
                original_weight = original_weights.get(position.ticker, 0.0)

                # Get expected returns
                equilibrium_ret = equilibrium_returns.get(position.ticker)
                posterior_ret = posterior_returns.get(position.ticker)

                db_position = DBPortfolioPosition(
                    portfolio_id=portfolio.id,
                    ticker=position.ticker,
                    yfinance_ticker=getattr(position, "yfinance_ticker", None),
                    weight=Decimal(str(position.weight)),
                    company_name=position.company_name,
                    sector=position.sector,
                    industry=position.industry,
                    country=position.country,
                    exchange=position.exchange,
                    price=Decimal(str(position.price)) if position.price else None,
                    sharpe_ratio=(
                        Decimal(str(position.sharpe_ratio)) if position.sharpe_ratio else None
                    ),
                    sortino_ratio=(
                        Decimal(str(position.sortino_ratio)) if position.sortino_ratio else None
                    ),
                    volatility=(Decimal(str(position.volatility)) if position.volatility else None),
                    alpha=Decimal(str(position.alpha)) if position.alpha else None,
                    beta=Decimal(str(position.beta)) if position.beta else None,
                    max_drawdown=(
                        Decimal(str(position.max_drawdown)) if position.max_drawdown else None
                    ),
                    annualized_return=(
                        Decimal(str(position.annualized_return))
                        if position.annualized_return
                        else None
                    ),
                    signal_id=position.signal_id,
                    signal_type=(
                        position.signal_type if hasattr(position, "signal_type") else None
                    ),
                    confidence_level=(
                        position.confidence_level if hasattr(position, "confidence_level") else None
                    ),
                    conviction_tier=position.conviction_tier,
                    original_weight=Decimal(str(original_weight)),
                    equilibrium_return=(Decimal(str(equilibrium_ret)) if equilibrium_ret else None),
                    posterior_return=(Decimal(str(posterior_ret)) if posterior_ret else None),
                    selection_reason=position.selection_reason,
                )

                session.add(db_position)

            session.commit()

            logger.info(f"✓ Portfolio saved successfully: {portfolio.id}")
            logger.info("=" * 100)

            return str(portfolio.id)


if __name__ == "__main__":
    """Example usage with ConcentratedPortfolioBuilder."""
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from optimizer.database.database import init_db
    from optimizer.src.risk_management import ConcentratedPortfolioBuilder

    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize database
        init_db()

        # Step 1: Build 20-stock concentrated portfolio
        logger.info("Building concentrated portfolio...")
        builder = ConcentratedPortfolioBuilder(
            target_positions=20,
            max_sector_weight=0.15,  # Max 3 stocks per sector (20 * 15%)
            max_country_weight=0.60,  # Max 12 stocks per country (20 * 60%)
            max_correlation=0.75,  # Max pairwise correlation
            capital=1500.0,  # Trading212: €1 min, max €75 per stock
        )
        selected_stocks = builder.build_portfolio()  # Returns List[Tuple[StockSignal, Instrument]]

        logger.info(f"✓ Selected {len(selected_stocks)} stocks for optimization")

        # Convert to PortfolioPosition objects for Black-Litterman optimizer
        # Extract sector, industry, and other metadata from StockSignal and Instrument
        from optimizer.src.stock_analyzer.data.fetchers import get_country_from_ticker

        positions = []
        for signal, instrument in selected_stocks:
            # Determine country from yfinance ticker (e.g., FRE.DE -> Germany)
            country = (
                get_country_from_ticker(signal.yfinance_ticker, info=None)
                if signal.yfinance_ticker
                else "USA"
            )

            pos = PortfolioPosition(
                ticker=instrument.ticker,
                weight=1.0 / len(selected_stocks),  # Equal weight initially
                instrument_id=str(signal.instrument_id),
                signal_id=str(signal.id),
                signal_type=signal.signal_type.value if signal.signal_type else None,
                conviction_tier=1,  # All selected stocks are highest conviction (1=highest, 2=medium, 3=lowest)
                company_name=instrument.name,  # From Instrument
                sector=signal.sector,  # ✅ From StockSignal
                industry=signal.industry,  # ✅ From StockSignal
                country=country or "USA",  # ✅ From yfinance ticker
                exchange=signal.exchange_name,
                yfinance_ticker=signal.yfinance_ticker,
                price=(
                    float(signal.close_price) if signal.close_price else None
                ),  # Use close_price
                sharpe_ratio=(float(signal.sharpe_ratio) if signal.sharpe_ratio else None),
                sortino_ratio=(float(signal.sortino_ratio) if signal.sortino_ratio else None),
                volatility=float(signal.volatility) if signal.volatility else None,
                alpha=float(signal.alpha) if signal.alpha else None,
                beta=float(signal.beta) if signal.beta else None,
                max_drawdown=(float(signal.max_drawdown) if signal.max_drawdown else None),
                annualized_return=(
                    float(signal.annualized_return) if signal.annualized_return else None
                ),
                confidence_level=(
                    signal.confidence_level.value if signal.confidence_level else "medium"
                ),
                data_quality_score=(
                    float(signal.data_quality_score) if signal.data_quality_score else None
                ),
                selection_reason="Selected by ConcentratedPortfolioBuilder",
            )
            positions.append(pos)

        # Step 2: Optimize with Black-Litterman (using BAML AI views)
        logger.info("\nOptimizing with Black-Litterman...")
        optimizer = BlackLittermanOptimizer()
        optimized_positions, bl_metrics = optimizer.optimize_portfolio(positions)

        # Print comparison
        logger.info("\n" + "=" * 100)
        logger.info("WEIGHT COMPARISON")
        logger.info("=" * 100)
        logger.info(f"{'Ticker':<12} {'Original':<12} {'Optimized':<12} {'Change':<12}")
        logger.info("─" * 100)

        for orig_pos in positions:
            opt_pos = next((p for p in optimized_positions if p.ticker == orig_pos.ticker), None)

            if opt_pos:
                change = opt_pos.weight - orig_pos.weight
                logger.info(
                    f"{orig_pos.ticker:<12} {orig_pos.weight:>10.2%}   "
                    f"{opt_pos.weight:>10.2%}   {change:>+10.2%}"
                )

        logger.info("=" * 100)

        # Step 3: Save portfolio to database
        logger.info("\nSaving portfolio to database...")
        portfolio_id = optimizer.save_to_database(
            optimized_positions=optimized_positions,
            original_positions=positions,
            metrics=bl_metrics,
            portfolio_name="BL Optimized 20-Stock Portfolio",
        )

        logger.info(f"\n✓ Portfolio saved with ID: {portfolio_id}")

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
