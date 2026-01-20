from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from . import objective_functions

if TYPE_CHECKING:
    import pandas as pd


class BaseOptimizer:
    """
    Abstract base class for portfolio optimizers.
    """

    def __init__(self, n_assets: int, tickers: Optional[List[Union[str, int]]] = None) -> None:
        """
        Initialize the base optimizer with asset count and ticker labels.
        """
        if n_assets <= 0:
            raise ValueError("n_assets must be positive")

        self.n_assets = n_assets
        self.tickers: List[Union[str, int]] = list(range(n_assets)) if tickers is None else tickers
        self._risk_free_rate: Optional[float] = None
        self.weights: Optional[np.ndarray] = None

        # Validate tickers length
        if len(self.tickers) != n_assets:
            raise ValueError(
                f"Number of tickers ({len(self.tickers)}) must match n_assets ({n_assets})"
            )

    def _make_output_weights(self, weights: Optional[np.ndarray] = None) -> OrderedDict:
        """
        Utility function to create an ordered dict of weights from a numpy array.
        """
        if weights is None:
            if self.weights is None:
                raise AttributeError("Weights not yet computed")
            weights = self.weights

        # Convert numpy float64 to plain Python float for JSON serialization
        weight_values = [float(w) for w in weights]

        return OrderedDict(zip(self.tickers, weight_values))

    def set_weights(self, input_weights: Dict[Union[str, int], float]) -> None:
        """
        Set the weights attribute from a dictionary of ticker-weight pairs.
        """
        try:
            self.weights = np.array([input_weights[ticker] for ticker in self.tickers])
        except KeyError as e:
            raise KeyError(f"Ticker {e} not found in input_weights") from e

    def clean_weights(self, cutoff: float = 1e-4, rounding: Optional[int] = 5) -> OrderedDict:
        """
        Clean the raw weights by setting small values to zero and rounding.
        """
        if self.weights is None:
            raise AttributeError("Weights not yet computed")

        clean_weights = self.weights.copy()

        # Set weights below cutoff to zero
        clean_weights[np.abs(clean_weights) < cutoff] = 0

        # Round if requested
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)


def portfolio_performance(
    weights: Union[Dict[Union[str, int], float], np.ndarray, List[float]],
    expected_returns: Optional[Union[np.ndarray, "pd.Series"]],
    cov_matrix: Union[np.ndarray, "pd.DataFrame"],
    verbose: bool = False,
    risk_free_rate: float = 0.0,
) -> tuple[Optional[float], float, Optional[float]]:
    """
    Calculate portfolio performance metrics: return, volatility, and Sharpe ratio.
    """
    import pandas as pd

    # Convert weights to numpy array
    new_weights = _sanitize_weights(weights, expected_returns, cov_matrix)

    # Calculate volatility
    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

    # Calculate return and Sharpe ratio if returns are provided
    if expected_returns is not None:
        mu = objective_functions.portfolio_return(new_weights, expected_returns)

        sharpe = objective_functions.sharpe_ratio(
            new_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
        )

        if verbose:
            print(f"Expected annual return: {100 * mu:.1f}%")
            print(f"Annual volatility: {100 * sigma:.1f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")

        return mu, sigma, sharpe
    else:
        if verbose:
            print(f"Annual volatility: {100 * sigma:.1f}%")
        return None, sigma, None


def _sanitize_weights(
    weights: Union[Dict[Union[str, int], float], np.ndarray, List[float]],
    expected_returns: Optional[Union[np.ndarray, "pd.Series"]],
    cov_matrix: Union[np.ndarray, "pd.DataFrame"],
) -> np.ndarray:
    """
    Convert various weight formats to a numpy array.
    """
    import pandas as pd

    if isinstance(weights, dict):
        # Determine ticker order from expected_returns or cov_matrix
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = (
                list(range(len(expected_returns)))
                if expected_returns is not None
                else list(range(len(cov_matrix)))
            )

        # Map weights dict to array
        new_weights = np.zeros(len(tickers))
        for i, ticker in enumerate(tickers):
            if ticker in weights:
                new_weights[i] = weights[ticker]

        if new_weights.sum() == 0:
            raise ValueError("Weights sum to zero, or ticker names don't match")

        return new_weights
    elif weights is not None:
        return np.asarray(weights)
    else:
        raise ValueError("Weights cannot be None")
