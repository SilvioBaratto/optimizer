"""
View Matrix Builder - Construct P, Q, Omega matrices for Black-Litterman.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from optimizer.domain.models.view import BlackLittermanViewDTO

logger = logging.getLogger(__name__)


class ViewMatrixBuilder:
    """
    Construct Black-Litterman view matrices (P, Q, Omega).

    The matrices encode investor views:
    - P (KxN): Picking matrix - which assets each view concerns
    - Q (Kx1): Expected returns for each view
    - Omega (KxK): Uncertainty matrix for views (diagonal)

    Supports:
    - Absolute views: Stock X will return Y%
    - Relative views: Stock X will outperform Y by Z%
    - Basket views: Sector X will return Y%
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def construct(
        self,
        views: List[BlackLittermanViewDTO],
        universe_tickers: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct P, Q, Omega matrices from views.

        Args:
            views: List of BlackLittermanViewDTO objects
            universe_tickers: Complete list of tickers in portfolio universe

        Returns:
            Tuple of (P, Q, Omega) numpy arrays:
            - P: KxN picking matrix
            - Q: Kx1 expected returns vector
            - Omega: KxK view uncertainty matrix (diagonal)
        """
        N = len(universe_tickers)
        K = len(views)

        if K == 0:
            self._logger.warning("No views to construct matrices from")
            return np.zeros((0, N)), np.zeros((0, 1)), np.zeros((0, 0))

        # Create ticker index mapping
        ticker_index = {ticker: i for i, ticker in enumerate(universe_tickers)}

        # Initialize matrices
        P = np.zeros((K, N))
        Q = np.zeros((K, 1))
        Omega = np.zeros((K, K))

        # Track views with missing tickers
        missing_count = 0

        # Fill matrices
        for k, view in enumerate(views):
            ticker = view.ticker

            if ticker not in ticker_index:
                self._logger.debug(f"Ticker {ticker} not in universe, skipping view")
                missing_count += 1
                continue

            idx = ticker_index[ticker]

            if view.view_type == "absolute":
                # Absolute view: asset k will return Q[k]
                P[k, idx] = 1.0
            elif view.view_type == "relative" and view.benchmark_ticker:
                # Relative view: asset k will outperform benchmark by Q[k]
                if view.benchmark_ticker in ticker_index:
                    P[k, idx] = 1.0
                    P[k, ticker_index[view.benchmark_ticker]] = -1.0
                else:
                    # Fallback to absolute if benchmark not in universe
                    P[k, idx] = 1.0

            Q[k, 0] = view.expected_return
            Omega[k, k] = view.view_uncertainty ** 2

        # Remove rows for missing tickers
        if missing_count > 0:
            # Find non-zero rows
            non_zero_rows = np.any(P != 0, axis=1)
            P = P[non_zero_rows]
            Q = Q[non_zero_rows]
            Omega = Omega[np.ix_(non_zero_rows, non_zero_rows)]

            self._logger.info(f"Removed {missing_count} views with missing tickers")

        self._logger.info(
            f"Constructed BL matrices: P({P.shape[0]}x{P.shape[1]}), "
            f"Q({Q.shape[0]}x{Q.shape[1]}), Omega({Omega.shape[0]}x{Omega.shape[1]})"
        )

        return P, Q, Omega

    def construct_basket_views(
        self,
        views: List[BlackLittermanViewDTO],
        universe_tickers: List[str],
        groupby: str = "sector",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct basket views grouped by sector or industry.

        Instead of individual stock views, creates views on equal-weighted
        baskets (e.g., "Technology sector will return 12%").

        Args:
            views: List of BlackLittermanViewDTO objects
            universe_tickers: Complete list of tickers in portfolio universe
            groupby: 'sector' or 'industry' for grouping

        Returns:
            Tuple of (P, Q, Omega) matrices for basket views
        """
        # Group views by sector/industry
        groups: Dict[str, List[BlackLittermanViewDTO]] = defaultdict(list)
        for view in views:
            if groupby == "sector":
                key = view.sector or "Unknown"
            else:
                key = "Unknown"  # Would need industry info
            groups[key].append(view)

        N = len(universe_tickers)
        K = len(groups)

        if K == 0:
            return np.zeros((0, N)), np.zeros((0, 1)), np.zeros((0, 0))

        self._logger.info(f"Creating {K} basket views grouped by {groupby}")

        # Create ticker index mapping
        ticker_index = {ticker: i for i, ticker in enumerate(universe_tickers)}

        # Initialize matrices
        P = np.zeros((K, N))
        Q = np.zeros((K, 1))
        Omega = np.zeros((K, K))

        # Fill matrices
        for k, (group_name, group_views) in enumerate(groups.items()):
            # Get tickers in this group that are in universe
            group_tickers = [
                view.ticker
                for view in group_views
                if view.ticker in ticker_index
            ]

            if not group_tickers:
                continue

            n_stocks = len(group_tickers)

            # P matrix: equal-weighted basket
            for ticker in group_tickers:
                idx = ticker_index[ticker]
                P[k, idx] = 1.0 / n_stocks

            # Q matrix: average expected return
            avg_return = np.mean([v.expected_return for v in group_views])
            Q[k, 0] = avg_return

            # Omega matrix: average uncertainty (scaled by sqrt(n) for basket)
            avg_uncertainty = np.mean([v.view_uncertainty for v in group_views])
            basket_uncertainty = avg_uncertainty / np.sqrt(n_stocks)
            Omega[k, k] = basket_uncertainty ** 2

            self._logger.debug(
                f"Basket view for {group_name}: {n_stocks} stocks, "
                f"return={avg_return:.2%}, uncertainty={basket_uncertainty:.2%}"
            )

        return P, Q, Omega

    def validate_matrices(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
        universe_tickers: List[str],
    ) -> Dict[str, Any]:
        """
        Validate constructed matrices.

        Checks:
        - Dimensions consistency
        - P has no all-zero rows
        - Omega is positive definite
        - Q values are reasonable

        Returns:
            Dictionary with validation results
        """
        issues = []

        K, N = P.shape

        # Dimension checks
        if Q.shape != (K, 1):
            issues.append(f"Q dimension mismatch: expected ({K}, 1), got {Q.shape}")

        if Omega.shape != (K, K):
            issues.append(f"Omega dimension mismatch: expected ({K}, {K}), got {Omega.shape}")

        if N != len(universe_tickers):
            issues.append(f"P columns {N} != universe size {len(universe_tickers)}")

        # Check for all-zero rows in P
        zero_rows = np.where(~np.any(P != 0, axis=1))[0]
        if len(zero_rows) > 0:
            issues.append(f"P has {len(zero_rows)} all-zero rows: {zero_rows.tolist()}")

        # Check Omega is positive definite
        omega_eigenvalues = np.linalg.eigvalsh(Omega)
        if np.any(omega_eigenvalues <= 0):
            issues.append("Omega is not positive definite")

        # Check Q values are reasonable
        if np.any(np.abs(Q) > 1.0):
            extreme_views = np.where(np.abs(Q) > 1.0)[0]
            issues.append(f"Q has extreme values (>100%) at indices: {extreme_views.tolist()}")

        return {
            "valid": len(issues) == 0,
            "n_views": K,
            "n_assets": N,
            "issues": issues,
            "q_mean": float(Q.mean()) if K > 0 else None,
            "q_std": float(Q.std()) if K > 0 else None,
            "omega_trace": float(np.trace(Omega)) if K > 0 else None,
        }

    def get_view_summary(
        self,
        views: List[BlackLittermanViewDTO],
    ) -> Dict[str, Any]:
        """
        Get summary statistics for generated views.

        Args:
            views: List of view DTOs

        Returns:
            Dictionary of summary statistics
        """
        if not views:
            return {"n_views": 0}

        returns = [v.expected_return for v in views]
        confidences = [v.confidence for v in views]
        uncertainties = [v.view_uncertainty for v in views]

        # Group by sector
        sectors: Dict[str, List[float]] = {}
        for view in views:
            sector = view.sector or "Unknown"
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(view.expected_return)

        return {
            "n_views": len(views),
            "avg_expected_return": float(np.mean(returns)),
            "median_expected_return": float(np.median(returns)),
            "min_expected_return": float(np.min(returns)),
            "max_expected_return": float(np.max(returns)),
            "avg_confidence": float(np.mean(confidences)),
            "avg_uncertainty": float(np.mean(uncertainties)),
            "sectors": {
                sector: {
                    "count": len(rets),
                    "avg_return": float(np.mean(rets))
                }
                for sector, rets in sectors.items()
            }
        }
