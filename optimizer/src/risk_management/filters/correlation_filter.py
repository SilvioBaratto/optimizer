"""
Correlation Filter - Filter stocks based on pairwise correlations.
"""

from typing import Tuple, Optional, Dict, Any, List, Set
import numpy as np

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class CorrelationFilterImpl(StockFilterImpl):
    """
    Filter stocks to ensure low pairwise correlations.
    """

    def __init__(
        self,
        max_correlation: float = 0.75,
        max_cluster_size: int = 3,
        lookback_days: int = 252,
    ):
        """
        Initialize correlation filter.
        """
        super().__init__()
        self._max_correlation = max_correlation
        self._max_cluster_size = max_cluster_size
        self._lookback_days = lookback_days

        # Correlation matrix (set externally)
        self._correlation_matrix: Optional[np.ndarray] = None
        self._tickers: List[str] = []
        self._ticker_to_idx: Dict[str, int] = {}

        # Stateful tracking of selected stocks
        self._selected_indices: Set[int] = set()

    @property
    def name(self) -> str:
        """Filter name."""
        return "CorrelationFilter"

    @property
    def max_correlation(self) -> float:
        """Maximum allowed pairwise correlation."""
        return self._max_correlation

    @property
    def max_cluster_size(self) -> int:
        """Maximum stocks allowed per correlation cluster."""
        return self._max_cluster_size

    def reset(self) -> None:
        """Reset stateful tracking."""
        self._selected_indices.clear()

    def set_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        tickers: List[str],
    ) -> None:
        """
        Set the correlation matrix for filtering.

        Args:
            correlation_matrix: NxN correlation matrix
            tickers: List of tickers matching matrix indices
        """
        self._correlation_matrix = correlation_matrix
        self._tickers = tickers
        self._ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if signal passes correlation constraints.
        """
        if self._correlation_matrix is None:
            # No correlation matrix set - pass by default
            return True, None

        if signal.ticker not in self._ticker_to_idx:
            # Ticker not in correlation matrix - cannot check
            return True, "ticker_not_in_matrix"

        idx = self._ticker_to_idx[signal.ticker]

        # Check correlation with all already-selected stocks
        for selected_idx in self._selected_indices:
            corr = self._correlation_matrix[idx, selected_idx]
            if corr > self._max_correlation:
                selected_ticker = self._tickers[selected_idx]
                return (
                    False,
                    f"high_correlation({signal.ticker}-{selected_ticker}:{corr:.2f}>{self._max_correlation})",
                )

        # Signal passes - update tracking
        self._selected_indices.add(idx)
        return True, None

    def filter_batch(
        self,
        signals: List[StockSignalDTO],
    ) -> Tuple[List[StockSignalDTO], Dict[str, int]]:
        """
        Filter batch with automatic reset.
        """
        self.reset()
        return super().filter_batch(signals)

    def select_diversified(
        self,
        signals: List[StockSignalDTO],
        correlation_matrix: np.ndarray,
        target_count: int,
    ) -> List[StockSignalDTO]:
        """
        Select diversified stocks using correlation constraints.
        """
        self.reset()
        self.set_correlation_matrix(correlation_matrix, [s.ticker for s in signals])

        selected: List[StockSignalDTO] = []

        for signal in signals:
            if len(selected) >= target_count:
                break

            passes, reason = self.filter(signal)
            if passes:
                selected.append(signal)

        return selected

    def build_correlation_matrix(
        self,
        prices_df,  # pd.DataFrame
    ) -> np.ndarray:
        """
        Build correlation matrix from price DataFrame.
        """
        # Calculate returns
        returns = prices_df.pct_change().dropna()

        # Calculate correlation matrix
        corr_matrix = returns.corr().values

        # Store tickers
        self._tickers = list(prices_df.columns)
        self._ticker_to_idx = {t: i for i, t in enumerate(self._tickers)}
        self._correlation_matrix = corr_matrix

        return corr_matrix

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "max_correlation": self._max_correlation,
            "max_cluster_size": self._max_cluster_size,
            "lookback_days": self._lookback_days,
            "matrix_size": len(self._tickers) if self._tickers else 0,
        }

    def get_correlation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the correlation matrix.

        Returns:
            Dictionary with correlation statistics
        """
        if self._correlation_matrix is None:
            return {"error": "no_matrix_set"}

        # Get upper triangle (excluding diagonal)
        n = len(self._correlation_matrix)
        upper_triangle = self._correlation_matrix[np.triu_indices(n, k=1)]

        return {
            "n_assets": n,
            "n_pairs": len(upper_triangle),
            "mean_correlation": float(np.mean(upper_triangle)),
            "median_correlation": float(np.median(upper_triangle)),
            "max_correlation": float(np.max(upper_triangle)),
            "min_correlation": float(np.min(upper_triangle)),
            "std_correlation": float(np.std(upper_triangle)),
            "high_corr_pairs": int(np.sum(upper_triangle > self._max_correlation)),
        }
