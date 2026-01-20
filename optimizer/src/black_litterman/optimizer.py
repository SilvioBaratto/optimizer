import logging
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from optimizer.config.optimizer_config import OptimizerConfig
from optimizer.config.portfolio_config import PortfolioConfig
from optimizer.domain.models.portfolio import PositionDTO, PortfolioDTO
from optimizer.domain.models.view import MacroRegimeDTO
from optimizer.domain.value_objects.covariance import CovarianceMatrix

from optimizer.src.black_litterman.covariance import LedoitWolfEstimator, get_covariance_estimator
from optimizer.src.black_litterman.equilibrium import (
    EquilibriumCalculatorImpl,
    RegimeAdjustedRiskAversion,
)
from optimizer.src.black_litterman.views import ViewGeneratorImpl, ViewMatrixBuilder
from optimizer.src.black_litterman.optimization import (
    ConstrainedOptimizerImpl,
    SectorConstraintBuilder,
)

logger = logging.getLogger(__name__)


class BlackLittermanOptimizer:
    """
    Main orchestrator for Black-Litterman portfolio optimization.
    """

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        portfolio_config: Optional[PortfolioConfig] = None,
    ):
        """
        Initialize Black-Litterman optimizer.

        Args:
            config: Optimizer configuration (uses defaults if None)
            portfolio_config: Portfolio constraints configuration
        """
        self._config = config or OptimizerConfig()
        self._portfolio_config = portfolio_config or PortfolioConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self._covariance_estimator = get_covariance_estimator(self._config.covariance_method)
        self._equilibrium_calculator = EquilibriumCalculatorImpl()
        self._risk_aversion_estimator = RegimeAdjustedRiskAversion()
        self._view_generator = ViewGeneratorImpl(
            min_confidence=self._config.min_view_confidence,
            view_scaling=self._config.view_confidence_scaling,
        )
        self._constrained_optimizer = ConstrainedOptimizerImpl(
            max_iterations=self._config.max_iterations,
            tolerance=self._config.tolerance,
            regularization=self._config.regularization,
        )
        self._sector_builder = SectorConstraintBuilder()
        self._view_matrix_builder = ViewMatrixBuilder()

        # State from last optimization
        self._last_equilibrium: Optional[pd.Series] = None
        self._last_posterior: Optional[pd.Series] = None
        self._last_covariance: Optional[CovarianceMatrix] = None

    async def optimize(
        self,
        positions: List[PositionDTO],
        signal_date: Optional[date] = None,
        prices_df: Optional[pd.DataFrame] = None,
        market_caps: Optional[pd.Series] = None,
        macro_regime: Optional[MacroRegimeDTO] = None,
        views: Optional[List[Any]] = None,
    ) -> Tuple[List[PositionDTO], Dict[str, Any]]:
        """
        Optimize portfolio weights using Black-Litterman.
        """
        self._logger.info(f"Starting Black-Litterman optimization for {len(positions)} positions")

        tickers = [p.ticker for p in positions]
        n = len(tickers)

        if n == 0:
            self._logger.error("No positions to optimize")
            return [], {"error": "no_positions"}

        # Step 1: Get historical prices
        if prices_df is None:
            prices_df = await self._fetch_prices(tickers)
            if prices_df is None or prices_df.empty:
                self._logger.error("Could not fetch prices")
                return positions, {"error": "no_prices"}

        # Align tickers with available prices
        available_tickers = [t for t in tickers if t in prices_df.columns]
        if len(available_tickers) < len(tickers):
            self._logger.warning(
                f"Only {len(available_tickers)}/{len(tickers)} tickers have price data"
            )
            tickers = available_tickers
            prices_df = prices_df[tickers]

        # Step 2: Estimate covariance matrix
        self._logger.info("Step 1/5: Estimating covariance matrix...")
        cov_matrix = self._covariance_estimator.estimate(
            prices_df,
            frequency=252,
        )
        self._last_covariance = cov_matrix

        # Step 3: Determine risk aversion
        self._logger.info("Step 2/5: Calculating risk aversion...")
        base_delta = self._config.risk_aversion
        if macro_regime:
            delta = self._risk_aversion_estimator.adjust_for_regime(
                base_delta,
                macro_regime.regime.value,
                recession_risk=macro_regime.recession_risk_6m,
            )
        else:
            delta = base_delta

        # Step 4: Calculate equilibrium returns
        self._logger.info("Step 3/5: Calculating equilibrium returns...")
        if market_caps is None:
            # Use equal weights as proxy for market caps
            market_caps = pd.Series(1.0, index=tickers)

        equilibrium = self._equilibrium_calculator.calculate(
            market_caps=market_caps,
            covariance_matrix=cov_matrix,
            risk_aversion=delta,
            risk_free_rate=self._config.risk_free_rate,
        )
        self._last_equilibrium = equilibrium

        # Step 5: Get or generate views
        self._logger.info("Step 4/5: Processing views...")
        if views is None:
            # Convert positions to signals for view generation
            from domain.models.stock_signal import StockSignalDTO, SignalType
            import uuid

            signals = [
                StockSignalDTO(
                    id=uuid.uuid4(),
                    instrument_id=uuid.uuid4(),
                    ticker=p.ticker,
                    signal_date=signal_date or date.today(),
                    signal_type=(
                        SignalType(p.signal_type.lower()) if p.signal_type else SignalType.NEUTRAL
                    ),
                    yfinance_ticker=p.yfinance_ticker,
                    sector=p.sector,
                    industry=p.industry,
                    sharpe_ratio=p.sharpe_ratio,
                    volatility=p.volatility,
                    alpha=p.alpha,
                    beta=p.beta,
                    annualized_return=p.annualized_return,
                )
                for p in positions
                if p.ticker in tickers
            ]
            views = await self._view_generator.generate(signals, macro_regime)

        # Construct view matrices
        if views:
            P, Q, Omega = self._view_matrix_builder.construct(views, tickers)
            has_views = P.shape[0] > 0
        else:
            P, Q, Omega = np.zeros((0, n)), np.zeros((0, 1)), np.zeros((0, 0))
            has_views = False

        # Step 6: Calculate posterior returns
        self._logger.info("Step 5/5: Calculating posterior and optimizing...")
        if has_views:
            posterior = self._calculate_posterior(
                equilibrium=equilibrium,
                covariance=cov_matrix,
                P=P,
                Q=Q,
                Omega=Omega,
                tau=self._config.tau,
            )
        else:
            self._logger.warning("No views, using equilibrium as posterior")
            posterior = equilibrium

        self._last_posterior = posterior

        # Step 7: Build sector constraints
        sector_mapping = self._build_sector_mapping(positions, tickers)

        # Step 8: Optimize weights
        optimized_weights = self._constrained_optimizer.optimize(
            posterior_returns=posterior,
            covariance_matrix=cov_matrix,
            risk_aversion=delta,
            sector_mapping=sector_mapping,
            max_sector_weight=self._portfolio_config.max_sector_weight,
            max_position_weight=self._portfolio_config.max_position_weight,
            min_position_weight=self._portfolio_config.min_position_weight,
        )

        # Step 9: Update positions with optimized weights
        optimized_positions = self._update_positions(
            positions=positions,
            weights=optimized_weights,
            equilibrium=equilibrium,
            posterior=posterior,
        )

        # Step 10: Compute metrics
        metrics = self._compute_metrics(
            positions=optimized_positions,
            weights=optimized_weights,
            equilibrium=equilibrium,
            posterior=posterior,
            covariance=cov_matrix,
            delta=delta,
            n_views=len(views) if views else 0,
        )

        self._logger.info(
            f"Optimization complete. "
            f"Expected return: {metrics['expected_return']:.2%}, "
            f"Volatility: {metrics['volatility']:.2%}, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}"
        )

        return optimized_positions, metrics

    def _calculate_posterior(
        self,
        equilibrium: pd.Series,
        covariance: CovarianceMatrix,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
        tau: float,
    ) -> pd.Series:
        """
        Calculate posterior expected returns using Black-Litterman formula.
        """
        tickers = covariance.ticker_list
        sigma = covariance.matrix
        pi = equilibrium.reindex(tickers).values

        # τΣ
        tau_sigma = tau * sigma

        # (τΣ)^-1
        tau_sigma_inv = np.linalg.inv(tau_sigma)

        # Ω^-1
        omega_inv = np.linalg.inv(Omega)

        # Master formula
        # M = (τΣ)^-1 + P'Ω^-1 P
        M = tau_sigma_inv + P.T @ omega_inv @ P

        # M^-1
        M_inv = np.linalg.inv(M)

        # μ = M^-1 × [(τΣ)^-1 π + P'Ω^-1 Q]
        term1 = tau_sigma_inv @ pi
        term2 = (P.T @ omega_inv @ Q).flatten()

        posterior = M_inv @ (term1 + term2)

        return pd.Series(posterior, index=tickers)

    def _build_sector_mapping(
        self,
        positions: List[PositionDTO],
        tickers: List[str],
    ) -> Dict[str, List[int]]:
        """Build sector constraint mapping."""
        return self._sector_builder.build_from_positions(positions, tickers)

    def _update_positions(
        self,
        positions: List[PositionDTO],
        weights: pd.Series,
        equilibrium: pd.Series,
        posterior: pd.Series,
    ) -> List[PositionDTO]:
        """Update positions with optimized weights and BL returns."""
        optimized = []

        for pos in positions:
            if pos.ticker not in weights.index:
                continue

            new_weight = weights[pos.ticker]

            # Skip zero-weight positions
            if new_weight < 1e-6:
                continue

            # Create updated position
            updated = PositionDTO(
                ticker=pos.ticker,
                weight=float(new_weight),
                instrument_id=pos.instrument_id,
                signal_id=pos.signal_id,
                yfinance_ticker=pos.yfinance_ticker,
                company_name=pos.company_name,
                sector=pos.sector,
                industry=pos.industry,
                country=pos.country,
                exchange=pos.exchange,
                price=pos.price,
                sharpe_ratio=pos.sharpe_ratio,
                sortino_ratio=pos.sortino_ratio,
                volatility=pos.volatility,
                alpha=pos.alpha,
                beta=pos.beta,
                max_drawdown=pos.max_drawdown,
                annualized_return=pos.annualized_return,
                signal_type=pos.signal_type,
                confidence_level=pos.confidence_level,
                conviction_tier=pos.conviction_tier,
                original_weight=pos.weight,
                equilibrium_return=(
                    float(equilibrium.get(pos.ticker, 0))
                    if pos.ticker in equilibrium.index
                    else None
                ),
                posterior_return=(
                    float(posterior.get(pos.ticker, 0)) if pos.ticker in posterior.index else None
                ),
                selection_reason=pos.selection_reason,
            )
            optimized.append(updated)

        # Sort by weight descending
        optimized.sort(key=lambda p: p.weight, reverse=True)

        return optimized

    def _compute_metrics(
        self,
        positions: List[PositionDTO],
        weights: pd.Series,
        equilibrium: pd.Series,
        posterior: pd.Series,
        covariance: CovarianceMatrix,
        delta: float,
        n_views: int,
    ) -> Dict[str, Any]:
        """Compute portfolio metrics and diagnostics."""
        tickers = covariance.ticker_list
        w = weights.reindex(tickers).fillna(0).values
        mu = posterior.reindex(tickers).fillna(0).values
        sigma = covariance.matrix

        # Portfolio expected return
        port_return = float(w @ mu)

        # Portfolio variance and volatility
        port_var = float(w @ sigma @ w)
        port_vol = np.sqrt(port_var)

        # Sharpe ratio
        sharpe = (port_return - self._config.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Risk contribution
        risk_contrib = self._constrained_optimizer.get_risk_contribution(weights, covariance)

        # Sector weights
        sector_mapping = self._build_sector_mapping(positions, tickers)
        sector_weights = {
            sector: float(sum(w[i] for i in indices)) for sector, indices in sector_mapping.items()
        }

        return {
            "expected_return": port_return,
            "volatility": port_vol,
            "variance": port_var,
            "sharpe_ratio": sharpe,
            "risk_aversion": delta,
            "tau": self._config.tau,
            "n_positions": len([p for p in positions if weights.get(p.ticker, 0) > 1e-6]),
            "n_views": n_views,
            "sector_weights": sector_weights,
            "max_weight": float(weights.max()),
            "min_weight": float(weights[weights > 1e-6].min()) if any(weights > 1e-6) else 0,
            "equilibrium_mean": float(equilibrium.mean()),
            "posterior_mean": float(posterior.mean()),
            "view_impact": float(posterior.mean() - equilibrium.mean()),
        }

    async def _fetch_prices(self, tickers: List[str]) -> Optional[pd.DataFrame]:
        """Fetch historical prices for tickers."""
        try:
            from src.yfinance.client import YFinanceClient

            client = YFinanceClient.get_instance()

            # Fetch for lookback period
            prices_df = client.fetch_historical_prices(
                tickers,
                period=f"{self._config.lookback_days}d",
            )

            return prices_df

        except Exception as e:
            self._logger.error(f"Error fetching prices: {e}")
            return None

    def get_posterior_returns(self) -> Optional[pd.Series]:
        """Get the posterior expected returns from last optimization."""
        return self._last_posterior

    def get_equilibrium_returns(self) -> Optional[pd.Series]:
        """Get the equilibrium returns from last optimization."""
        return self._last_equilibrium

    def get_covariance_matrix(self) -> Optional[CovarianceMatrix]:
        """Get the covariance matrix from last optimization."""
        return self._last_covariance
