"""
Constrained Optimizer - Quadratic programming portfolio optimization.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from optimizer.domain.value_objects.covariance import CovarianceMatrix

logger = logging.getLogger(__name__)


class ConstrainedOptimizerImpl:
    """
    Implementation of constrained portfolio optimization.
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        regularization: float = 1e-6,
    ):
        """
        Initialize constrained optimizer.

        Args:
            max_iterations: Maximum optimizer iterations
            tolerance: Convergence tolerance
            regularization: L2 regularization for numerical stability
        """
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._regularization = regularization
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def optimize(
        self,
        posterior_returns: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
        sector_mapping: Dict[str, List[int]],
        max_sector_weight: float = 0.15,
        max_position_weight: float = 0.10,
        min_position_weight: float = 0.0,
    ) -> pd.Series:
        """
        Run constrained optimization.
        """
        tickers = covariance_matrix.ticker_list
        n = len(tickers)

        # Align returns with covariance matrix tickers
        mu = posterior_returns.reindex(tickers).values
        sigma = covariance_matrix.matrix

        # Add regularization for numerical stability
        sigma_reg = sigma + self._regularization * np.eye(n)

        # Define objective function: minimize 0.5 * w^T (δΣ) w - μ^T w
        def objective(w):
            return 0.5 * risk_aversion * w @ sigma_reg @ w - mu @ w

        def gradient(w):
            return risk_aversion * sigma_reg @ w - mu

        # Constraints
        constraints = []

        # Equality constraint: weights sum to 1
        constraints.append(
            {
                "type": "eq",
                "fun": lambda w: np.sum(w) - 1.0,
                "jac": lambda w: np.ones(n),
            }
        )

        # Sector constraints
        for sector, indices in sector_mapping.items():
            if len(indices) > 0:
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda w, idx=indices: max_sector_weight - np.sum(w[idx]),
                        "jac": lambda w, idx=indices: -np.array(
                            [1.0 if i in idx else 0.0 for i in range(n)]
                        ),
                    }
                )

        # Bounds: min_weight <= w_i <= max_weight
        bounds = [(min_position_weight, max_position_weight) for _ in range(n)]

        # Initial guess: equal weights
        w0 = np.ones(n) / n

        self._logger.info(f"Starting optimization with {n} assets, δ={risk_aversion:.2f}")

        # Run optimization
        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": self._max_iterations,
                "ftol": self._tolerance,
                "disp": False,
            },
        )

        if not result.success:
            self._logger.warning(f"Optimization did not converge: {result.message}")

        # Post-process weights
        weights = result.x

        # Ensure weights sum to exactly 1
        weights = weights / weights.sum()

        # Clip small negatives (numerical artifacts)
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        self._logger.info(
            f"Optimization complete. "
            f"Objective: {result.fun:.6f}, "
            f"Iterations: {result.nit}, "
            f"Success: {result.success}"
        )

        return pd.Series(weights, index=tickers)

    def optimize_with_turnover_constraint(
        self,
        posterior_returns: pd.Series,
        covariance_matrix: CovarianceMatrix,
        risk_aversion: float,
        current_weights: pd.Series,
        max_turnover: float = 0.30,
        sector_mapping: Optional[Dict[str, List[int]]] = None,
        max_sector_weight: float = 0.15,
        max_position_weight: float = 0.10,
    ) -> pd.Series:
        """
        Optimize with turnover constraint to limit trading.
        """
        tickers = covariance_matrix.ticker_list
        n = len(tickers)

        # Align inputs
        mu = posterior_returns.reindex(tickers).values
        sigma = covariance_matrix.matrix
        w_current = current_weights.reindex(tickers).fillna(0).values

        sigma_reg = sigma + self._regularization * np.eye(n)

        def objective(w):
            return 0.5 * risk_aversion * w @ sigma_reg @ w - mu @ w

        # Constraints
        constraints = [
            # Full investment
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            # Turnover constraint: sum of absolute changes <= max_turnover
            {"type": "ineq", "fun": lambda w: max_turnover - np.sum(np.abs(w - w_current))},
        ]

        # Sector constraints
        if sector_mapping:
            for sector, indices in sector_mapping.items():
                if len(indices) > 0:
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=indices: max_sector_weight - np.sum(w[idx]),
                        }
                    )

        bounds = [(0.0, max_position_weight) for _ in range(n)]

        # Start from current weights
        w0 = w_current.copy()
        w0 = np.clip(w0, 0.0, max_position_weight)
        w0 = w0 / w0.sum()

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self._max_iterations, "ftol": self._tolerance},
        )

        weights = result.x
        weights = np.maximum(weights, 0.0)
        weights = weights / weights.sum()

        actual_turnover = np.sum(np.abs(weights - w_current))
        self._logger.info(f"Turnover-constrained optimization: turnover={actual_turnover:.2%}")

        return pd.Series(weights, index=tickers)

    def verify_constraints(
        self,
        weights: pd.Series,
        sector_mapping: Dict[str, List[int]],
        max_sector_weight: float,
        max_position_weight: float,
        min_position_weight: float,
    ) -> Dict[str, Any]:
        """
        Verify that weights satisfy all constraints.
        """
        violations = []
        w = weights.values

        # Check total weight
        total = np.sum(w)
        if abs(total - 1.0) > 0.01:
            violations.append(f"Total weight {total:.4f} != 1.0")

        # Check position bounds
        if np.any(w < min_position_weight - 1e-6):
            below_min = weights[w < min_position_weight - 1e-6]
            violations.append(f"Positions below min: {below_min.to_dict()}")

        if np.any(w > max_position_weight + 1e-6):
            above_max = weights[w > max_position_weight + 1e-6]
            violations.append(f"Positions above max: {above_max.to_dict()}")

        # Check sector constraints
        sector_weights = {}
        for sector, indices in sector_mapping.items():
            sector_weight = np.sum(w[indices])
            sector_weights[sector] = sector_weight
            if sector_weight > max_sector_weight + 1e-6:
                violations.append(
                    f"Sector {sector} weight {sector_weight:.2%} > {max_sector_weight:.2%}"
                )

        return {
            "all_satisfied": len(violations) == 0,
            "total_weight": float(total),
            "sector_weights": sector_weights,
            "max_weight": float(np.max(w)),
            "min_weight": float(np.min(w[w > 0])) if np.any(w > 0) else 0.0,
            "n_positions": int(np.sum(w > 1e-6)),
            "violations": violations,
        }

    def get_risk_contribution(
        self,
        weights: pd.Series,
        covariance_matrix: CovarianceMatrix,
    ) -> pd.Series:
        """
        Calculate marginal risk contribution of each position.
        """
        tickers = covariance_matrix.ticker_list
        w = weights.reindex(tickers).fillna(0).values
        sigma = covariance_matrix.matrix

        # Portfolio variance
        port_var = w @ sigma @ w
        port_vol = np.sqrt(port_var)

        # Marginal contribution
        marginal = sigma @ w

        # Risk contribution
        risk_contrib = w * marginal / port_vol

        return pd.Series(risk_contrib, index=tickers)
