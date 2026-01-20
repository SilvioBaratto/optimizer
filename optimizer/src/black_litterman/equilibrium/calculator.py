import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from optimizer.domain.value_objects.covariance import CovarianceMatrix

logger = logging.getLogger(__name__)


class EquilibriumCalculatorImpl:
    """
    Calculate market-implied equilibrium returns (π) for Black-Litterman.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate(
        self,
        market_caps: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
        risk_free_rate: float = 0.045,
    ) -> pd.Series:
        """
        Calculate market-implied equilibrium returns.
        """
        # Align tickers
        tickers = covariance_matrix.ticker_list
        caps_aligned = market_caps.reindex(tickers).fillna(0)

        # Calculate market cap weights
        total_cap = caps_aligned.sum()
        if total_cap <= 0:
            self._logger.warning("Total market cap is zero, using equal weights")
            market_weights = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            market_weights = caps_aligned / total_cap

        # Calculate equilibrium returns: π = δ × Σ × w_mkt
        sigma = covariance_matrix.matrix
        w = market_weights.values.reshape(-1, 1)

        pi = risk_aversion * sigma @ w
        pi = pi.flatten()

        # Add risk-free rate for excess returns
        equilibrium_returns = pd.Series(pi + risk_free_rate, index=tickers)

        self._logger.info(
            f"Calculated equilibrium returns for {len(tickers)} assets. "
            f"Mean: {equilibrium_returns.mean():.2%}, "
            f"Range: [{equilibrium_returns.min():.2%}, {equilibrium_returns.max():.2%}]"
        )

        return equilibrium_returns

    def calculate_with_equal_weights(
        self,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
        risk_free_rate: float = 0.045,
    ) -> pd.Series:
        """
        Calculate equilibrium returns assuming equal market weights.
        """
        tickers = covariance_matrix.ticker_list
        n = len(tickers)

        # Equal weights
        market_weights = pd.Series(1.0 / n, index=tickers)

        # Create dummy market caps (they'll be equal-weighted anyway)
        market_caps = pd.Series(1.0, index=tickers)

        return self.calculate(market_caps, covariance_matrix, risk_aversion, risk_free_rate)

    def calculate_from_benchmark(
        self,
        benchmark_weights: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
        risk_free_rate: float = 0.045,
    ) -> pd.Series:
        """
        Calculate equilibrium returns from benchmark weights.
        """
        # Normalize weights to sum to 1
        weights_normalized = benchmark_weights / benchmark_weights.sum()

        # Treat weights as "market caps" for calculation
        return self.calculate(
            market_caps=weights_normalized,
            covariance_matrix=covariance_matrix,
            risk_aversion=risk_aversion,
            risk_free_rate=risk_free_rate,
        )

    def reverse_optimize(
        self,
        portfolio_weights: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_free_rate: float = 0.045,
    ) -> tuple[pd.Series, float]:
        """
        Reverse optimize to find implied returns and risk aversion.
        """
        tickers = covariance_matrix.ticker_list
        w = portfolio_weights.reindex(tickers).fillna(0).values.reshape(-1, 1)
        sigma = covariance_matrix.matrix

        # Portfolio variance
        port_var = float(w.T @ sigma @ w)

        # Estimate risk aversion from portfolio characteristics
        # Using: δ = E[R_portfolio - r_f] / σ²_portfolio
        # Assuming typical equity premium of 5-6%
        assumed_premium = 0.055
        implied_delta = assumed_premium / port_var if port_var > 0 else 2.5

        # Calculate implied returns: μ = δ × Σ × w + r_f
        implied_returns = implied_delta * sigma @ w
        implied_returns = pd.Series(implied_returns.flatten() + risk_free_rate, index=tickers)

        self._logger.info(
            f"Reverse optimization: implied δ = {implied_delta:.2f}, "
            f"portfolio σ = {np.sqrt(port_var):.2%}"
        )

        return implied_returns, implied_delta

    def get_diagnostics(
        self,
        equilibrium_returns: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """
        Get diagnostic information about equilibrium calculations.
        """
        tickers = covariance_matrix.ticker_list
        returns = equilibrium_returns.reindex(tickers)

        # Calculate implied Sharpe ratios
        volatilities = pd.Series(
            [covariance_matrix.get_volatility(t) for t in tickers], index=tickers
        )
        sharpe_ratios = returns / volatilities

        return {
            "n_assets": len(tickers),
            "risk_aversion": risk_aversion,
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "mean_sharpe": float(sharpe_ratios.mean()),
            "max_sharpe": float(sharpe_ratios.max()),
            "top_5_assets": returns.nlargest(5).to_dict(),
            "bottom_5_assets": returns.nsmallest(5).to_dict(),
        }
