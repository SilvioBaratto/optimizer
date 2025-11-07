#!/usr/bin/env python3
"""
Correlation Analyzer - Manages Correlation Clustering and Diversification
==========================================================================
Analyzes pairwise correlations between stocks and enforces diversification
constraints to prevent hidden concentration risk.

Key Features:
- Builds correlation matrix from historical returns
- Identifies correlation clusters using hierarchical clustering
- Enforces maximum correlation between holdings
- Limits positions per correlation cluster
- Caches price data to avoid redundant downloads

Author: Portfolio Optimization System
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta, date
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm

from app.models.stock_signals import StockSignal
from app.models.universe import Instrument

logger = logging.getLogger(__name__)


# Cache for price data (module-level to persist across analyzer instances)
@lru_cache(maxsize=20)
def _fetch_prices_cached(
    tickers_tuple: tuple,
    end_date: date,
    lookback_days: int
) -> pd.DataFrame:
    """
    Fetch and cache price data from yfinance.

    Cache key: (tickers, end_date, lookback_days)
    Cache lifetime: Until Python process restarts or LRU eviction

    Args:
        tickers_tuple: Tuple of yfinance tickers (hashable for caching)
        end_date: End date for price data
        lookback_days: Number of days to fetch

    Returns:
        DataFrame with price data (multi-index columns)
    """
    from src.yfinance import YFinanceClient

    start_date = end_date - timedelta(days=int(lookback_days * 1.5))

    logger.info(
        f"📦 Cache miss - downloading price data "
        f"({len(tickers_tuple)} tickers, {start_date} to {end_date})"
    )

    # Use tqdm to show download progress
    with tqdm(total=len(tickers_tuple), desc="Downloading stock prices", unit="stock") as pbar:
        client = YFinanceClient.get_instance()

        # Convert dates to strings (YFinanceClient expects string dates)
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

        price_data = client.bulk_download(
            symbols=list(tickers_tuple),  # Parameter name is 'symbols' not 'tickers'
            start=start_str,
            end=end_str,
            progress=False,  # Disable yfinance's progress bar (we use tqdm)
            auto_adjust=False,  # Explicit to avoid FutureWarning
            threads=True,
            group_by='ticker'
        )
        pbar.update(len(tickers_tuple))  # Mark complete after download

    # Ensure we got valid data
    if price_data is None or (hasattr(price_data, 'empty') and price_data.empty):
        raise ValueError(
            f"yfinance returned no data for {len(tickers_tuple)} tickers "
            f"(date range: {start_date} to {end_date})"
        )

    return price_data


class CorrelationAnalyzer:
    """
    Analyzes correlations and enforces diversification constraints.

    Prevents concentrated portfolio from being 20 stocks that are essentially
    the same bet (e.g., 20 gold miners, 20 biotech stocks).
    """

    def __init__(
        self,
        max_correlation: float = 0.7,
        max_cluster_size: int = 2,
        clustering_threshold: float = 0.7
    ):
        """
        Initialize correlation analyzer.

        Args:
            max_correlation: Maximum pairwise correlation allowed (default 0.7)
            max_cluster_size: Max stocks per correlation cluster (default 2)
            clustering_threshold: Correlation threshold for clustering (default 0.7)
        """
        self.max_correlation = max_correlation
        self.max_cluster_size = max_cluster_size
        self.clustering_threshold = clustering_threshold

    def build_correlation_matrix(
        self,
        signals: List[Tuple[StockSignal, Instrument]],
        lookback_days: int = 252,
        use_real_data: bool = True
    ) -> pd.DataFrame:
        """
        Build correlation matrix from historical returns.

        Args:
            signals: List of (signal, instrument) tuples
            lookback_days: Days of history to use (default 252 = 1 year)
            use_real_data: Use real price data from yfinance (default True)

        Returns:
            Correlation matrix DataFrame (NxN where N = number of stocks)
        """
        logger.info(f"Building correlation matrix for {len(signals)} stocks")

        # Extract tickers and yfinance mappings
        tickers = [inst.ticker for _sig, inst in signals]
        yf_tickers = [inst.yfinance_ticker for _sig, inst in signals]

        if use_real_data:
            # Use real historical data from yfinance
            corr_matrix = self._build_real_correlation_matrix(
                tickers, yf_tickers, lookback_days
            )
        else:
            # Use mathematically valid placeholder for testing
            logger.warning("Using placeholder correlation matrix (for testing only)")
            corr_matrix = self._build_placeholder_correlation_matrix(tickers)

        # Check for NaN values in correlation matrix
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        nan_count = np.sum(np.isnan(upper_triangle))
        total_pairs = len(upper_triangle)

        if nan_count > 0:
            logger.warning(
                f"⚠️  Correlation matrix has {nan_count}/{total_pairs} NaN values "
                f"({nan_count/total_pairs*100:.1f}%)"
            )
            logger.warning(
                "    This indicates insufficient historical data overlap for some stock pairs"
            )

        # Calculate statistics (excluding NaN)
        valid_corr = upper_triangle[~np.isnan(upper_triangle)]
        if len(valid_corr) > 0:
            logger.info(
                f"Correlation matrix built: "
                f"avg={valid_corr.mean():.3f}, "
                f"max={valid_corr.max():.3f}, "
                f"min={valid_corr.min():.3f}"
            )
        else:
            logger.error("❌ All correlations are NaN - no valid data overlap!")

        return corr_matrix

    def _build_real_correlation_matrix(
        self,
        tickers: List[str],
        yf_tickers: List[Optional[str]],
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Build correlation matrix from real historical returns using yfinance.

        Uses batch downloads with caching and ensures all stocks are aligned to
        the same date range to produce consistent correlations.

        Args:
            tickers: List of T212 tickers (for indexing)
            yf_tickers: List of yfinance tickers (for fetching data)
            lookback_days: Days of history to use (default 252 = 1 trading year)

        Returns:
            Correlation matrix DataFrame (indexed by T212 tickers)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(lookback_days * 1.5))  # Extra buffer for weekends/holidays

        logger.info(f"Fetching price history from {start_date.date()} to {end_date.date()}")

        # Create mapping from yfinance ticker to T212 ticker
        yf_to_t212 = {}
        valid_yf_tickers = []
        missing_tickers = []

        for ticker, yf_ticker in zip(tickers, yf_tickers):
            if not yf_ticker:
                logger.warning(f"No yfinance ticker mapping for {ticker}")
                missing_tickers.append(ticker)
                continue
            yf_to_t212[yf_ticker] = ticker
            valid_yf_tickers.append(yf_ticker)

        if len(valid_yf_tickers) < 2:
            raise ValueError(
                f"Insufficient stocks with yfinance mappings: {len(valid_yf_tickers)}/{len(tickers)}. "
                f"Need at least 2 stocks."
            )

        # BATCH DOWNLOAD: Use cached function (avoids redundant downloads)
        # Cache key includes tickers, date, and lookback period
        tickers_tuple = tuple(sorted(valid_yf_tickers))  # Hashable for cache
        cache_date = end_date.date()  # Cache by day (not minute)

        # Check if this will be a cache hit (simple heuristic)
        cache_info_before = _fetch_prices_cached.cache_info()

        logger.info(
            f"Fetching price data for {len(valid_yf_tickers)} stocks "
            f"(cache: {cache_info_before.hits} hits, {cache_info_before.misses} misses)..."
        )

        try:
            price_data_raw = _fetch_prices_cached(
                tickers_tuple,
                cache_date,
                lookback_days
            )

            # Check if it was a cache hit
            cache_info_after = _fetch_prices_cached.cache_info()
            if cache_info_after.hits > cache_info_before.hits:
                logger.info("✓ Cache hit - using previously downloaded data")
            else:
                logger.info("✓ Download complete")

        except Exception as e:
            logger.error(f"Failed to batch download price data: {e}")
            raise

        # Check if download returned data
        if price_data_raw is None or (hasattr(price_data_raw, 'empty') and price_data_raw.empty):
            raise ValueError("yfinance returned no price data (empty response)")

        # Handle single ticker case (yfinance returns different format)
        if len(valid_yf_tickers) == 1:
            yf_ticker = valid_yf_tickers[0]
            if 'Adj Close' in price_data_raw.columns:
                price_series = price_data_raw['Adj Close']
            else:
                price_series = price_data_raw['Close']
            price_data = {yf_ticker: price_series}
        else:
            # Extract adjusted close prices for each ticker
            price_dict = {}
            for yf_ticker in valid_yf_tickers:
                try:
                    if yf_ticker in price_data_raw.columns.get_level_values(0):
                        # Multi-ticker format: (ticker, field)
                        if (yf_ticker, 'Adj Close') in price_data_raw.columns:
                            price_dict[yf_ticker] = price_data_raw[(yf_ticker, 'Adj Close')]
                        elif (yf_ticker, 'Close') in price_data_raw.columns:
                            price_dict[yf_ticker] = price_data_raw[(yf_ticker, 'Close')]
                    else:
                        logger.warning(f"No price data returned for {yf_ticker}")
                except Exception as e:
                    logger.warning(f"Error extracting prices for {yf_ticker}: {e}")
            price_data = price_dict

        # Convert to DataFrame with T212 tickers as columns
        price_df_dict = {}
        insufficient_data = []

        logger.info(f"Processing price data for {len(price_data)} stocks...")
        for yf_ticker, prices in tqdm(price_data.items(), desc="Processing stock data", unit="stock"):
            t212_ticker = yf_to_t212[yf_ticker]

            if prices is None or len(prices) < 50:
                logger.warning(f"Insufficient data for {t212_ticker} ({yf_ticker}): {len(prices) if prices is not None else 0} days")
                insufficient_data.append(t212_ticker)
                continue

            # CRITICAL FIX: Normalize timezone for cross-market portfolios
            # US stocks use America/New_York, European stocks use Europe/Paris, etc.
            # This prevents proper date alignment even when calendar dates match
            # Remove timezone before adding to price dict
            if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
                prices = prices.copy()
                prices.index = prices.index.tz_localize(None)

            price_df_dict[t212_ticker] = prices

        if len(price_df_dict) < 2:
            raise ValueError(
                f"Insufficient stocks with valid price data: {len(price_df_dict)}/{len(tickers)}. "
                f"Need at least 2 stocks with historical data."
            )

        # Build price DataFrame (stocks aligned to common dates after timezone normalization)
        price_df = pd.DataFrame(price_df_dict)

        # Calculate returns (automatically aligned by date)
        returns_df = price_df.pct_change().dropna()

        # Take last N days (now all stocks use 

        logger.info(f"Returns data: {len(returns_df)} days × {len(returns_df.columns)} stocks")
        logger.info(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")

        # Check for stocks with too much missing data
        min_required_days = int(lookback_days * 0.5)  # At least 50% of requested period
        stocks_to_remove = []

        logger.info("Checking data quality for each stock...")
        for ticker in tqdm(returns_df.columns, desc="Validating data quality", unit="stock"):
            valid_days = returns_df[ticker].notna().sum()
            coverage = valid_days / len(returns_df) * 100

            if valid_days < min_required_days:
                logger.warning(
                    f"Insufficient data for {ticker}: "
                    f"{valid_days}/{len(returns_df)} days ({coverage:.0f}% coverage)"
                )
                stocks_to_remove.append(ticker)

        # Remove stocks with insufficient data
        if stocks_to_remove:
            logger.warning(f"Removing {len(stocks_to_remove)} stocks due to insufficient data coverage")
            returns_df = returns_df.drop(columns=stocks_to_remove)

        if len(returns_df.columns) < 2:
            raise ValueError(
                f"Insufficient stocks remaining after data quality checks: {len(returns_df.columns)}. "
                f"Need at least 2 stocks."
            )

        # Calculate correlation matrix with minimum overlap requirement
        # min_periods ensures correlations are only calculated with sufficient overlap
        logger.info(f"Calculating correlation matrix ({len(returns_df.columns)}x{len(returns_df.columns)})...")
        corr_matrix = returns_df.corr(method='pearson', min_periods=min_required_days)

        # Check for NaN values in correlation matrix
        logger.info("Checking for missing correlations...")
        nan_pairs = []
        n_stocks = len(corr_matrix)

        # Show progress bar only if checking many stocks (>20)
        if n_stocks > 20:
            pairs_to_check = [(i, j) for i in range(n_stocks) for j in range(i + 1, n_stocks)]
            for i, j in tqdm(pairs_to_check, desc="Checking correlation pairs", unit="pair"):
                if pd.isna(corr_matrix.iloc[i, j]):
                    ticker_i = corr_matrix.index[i]
                    ticker_j = corr_matrix.columns[j]
                    nan_pairs.append((ticker_i, ticker_j))
        else:
            for i in range(n_stocks):
                for j in range(i + 1, n_stocks):
                    if pd.isna(corr_matrix.iloc[i, j]):
                        ticker_i = corr_matrix.index[i]
                        ticker_j = corr_matrix.columns[j]
                        nan_pairs.append((ticker_i, ticker_j))

        if nan_pairs:
            logger.warning(
                f"⚠️  {len(nan_pairs)} stock pairs have NaN correlation "
                f"(insufficient pairwise overlap)"
            )

            # Relax min_periods requirement for remaining NaN values
            logger.warning(f"   Recalculating with relaxed requirements (min_periods={int(lookback_days * 0.3)})")
            corr_matrix = returns_df.corr(method='pearson', min_periods=int(lookback_days * 0.3))

            # If still have NaN, fill with conservative value (0.3 = modest correlation)
            remaining_nan = corr_matrix.isna().sum().sum() - len(corr_matrix)  # Exclude diagonal
            if remaining_nan > 0:
                logger.warning(
                    f"   Filling {remaining_nan} remaining NaN values with 0.3 "
                    "(assuming modest correlation for insufficient data)"
                )
                # Fill off-diagonal NaN with 0.3
                mask = ~np.eye(len(corr_matrix), dtype=bool)
                corr_matrix = corr_matrix.where(~(corr_matrix.isna() & mask), 0.3)

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix.values, 1.0)

        # Validate correlation matrix is positive semi-definite
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        min_eigenvalue = eigenvalues.min()
        max_eigenvalue = eigenvalues.max()

        # Use relative threshold (0.1% of max eigenvalue)
        relative_threshold = -0.001 * max_eigenvalue

        if min_eigenvalue < relative_threshold:
            logger.error(
                f"❌ Invalid correlation matrix: min eigenvalue = {min_eigenvalue:.10f} "
                f"(threshold: {relative_threshold:.10f})"
            )
            logger.error("   Matrix is not positive semi-definite")

            # Try to fix with regularization
            logger.warning("   Attempting to fix with ridge regularization...")
            epsilon = abs(min_eigenvalue) + abs(relative_threshold)

            # Add small value to diagonal
            corr_matrix_fixed = corr_matrix.values + np.eye(len(corr_matrix)) * epsilon

            # Rescale to correlation matrix (diagonal = 1)
            D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(corr_matrix_fixed)))
            corr_matrix_fixed = D_inv_sqrt @ corr_matrix_fixed @ D_inv_sqrt

            # Verify fix worked
            min_eigenvalue_fixed = np.linalg.eigvals(corr_matrix_fixed).min()
            if min_eigenvalue_fixed < relative_threshold:
                raise ValueError(
                    f"Failed to fix correlation matrix: "
                    f"min eigenvalue still {min_eigenvalue_fixed:.10f}"
                )
            else:
                logger.info(f"   ✓ Fixed: min eigenvalue now {min_eigenvalue_fixed:.10f}")
                corr_matrix = pd.DataFrame(
                    corr_matrix_fixed,
                    index=corr_matrix.index,
                    columns=corr_matrix.columns
                )
                min_eigenvalue = min_eigenvalue_fixed

        # Calculate correlation statistics
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        valid_corr = upper_triangle[~np.isnan(upper_triangle)]

        logger.info(f"✓ Correlation matrix built and validated")
        logger.info(f"  Stocks: {len(corr_matrix)}/{len(tickers)} with sufficient data")
        logger.info(f"  Eigenvalues: min={min_eigenvalue:.6f}, max={max_eigenvalue:.6f}")
        logger.info(f"  Correlations: avg={valid_corr.mean():.3f}, max={valid_corr.max():.3f}, min={valid_corr.min():.3f}")

        return corr_matrix

    def _build_placeholder_correlation_matrix(
        self,
        tickers: List[str],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate a mathematically valid random correlation matrix for testing.

        Uses the method from Rebonato & Jackel (2000):
        Generate random orthogonal eigenvectors with positive eigenvalues.

        Args:
            tickers: List of ticker symbols
            seed: Random seed for reproducibility

        Returns:
            Valid correlation matrix DataFrame
        """
        n = len(tickers)
        np.random.seed(seed)

        logger.warning("=" * 80)
        logger.warning("⚠️  USING PLACEHOLDER CORRELATION MATRIX (FOR TESTING ONLY)")
        logger.warning("=" * 80)
        logger.warning("This is a mathematically valid random correlation matrix.")
        logger.warning("It does NOT reflect real stock correlations.")
        logger.warning("Set use_real_data=True for production use.")
        logger.warning("=" * 80)

        # Generate random orthogonal matrix (eigenvectors)
        A = np.random.randn(n, n)
        Q, _ = np.linalg.qr(A)  # QR decomposition gives orthogonal Q

        # Generate random positive eigenvalues that sum to n
        eigenvalues = np.random.exponential(scale=1, size=n)
        eigenvalues = eigenvalues / eigenvalues.sum() * n  # Normalize

        # Construct correlation matrix: Σ = Q Λ Q^T
        Lambda = np.diag(eigenvalues)
        corr_matrix = Q @ Lambda @ Q.T

        # Scale to have 1s on diagonal
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(corr_matrix)))
        corr_matrix = D_inv_sqrt @ corr_matrix @ D_inv_sqrt

        # Verify it's valid
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1"
        assert np.allclose(corr_matrix, corr_matrix.T), "Should be symmetric"

        min_eigenvalue = np.linalg.eigvals(corr_matrix).min()
        assert min_eigenvalue >= -1e-10, f"Should be PSD (min eigenvalue: {min_eigenvalue})"

        logger.info(f"✓ Valid placeholder correlation matrix generated")
        logger.info(f"  Min eigenvalue: {min_eigenvalue:.6f} (must be ≥ 0)")

        # Convert to DataFrame
        corr_df = pd.DataFrame(
            corr_matrix,
            index=tickers,
            columns=tickers
        )

        return corr_df

    def identify_correlation_clusters(
        self,
        corr_matrix: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Identify correlation clusters using hierarchical clustering.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Dictionary mapping ticker to cluster_id
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        logger.info("Identifying correlation clusters")

        # Convert correlation to distance (1 - correlation)
        distance_matrix = 1 - corr_matrix.abs()

        # Convert to condensed distance matrix
        condensed_dist = squareform(distance_matrix.values)

        # Hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='average')

        # Cut dendrogram at threshold
        cluster_labels = fcluster(
            linkage_matrix,
            t=1 - self.clustering_threshold,
            criterion='distance'
        )

        # Map tickers to clusters
        ticker_to_cluster = dict(zip(corr_matrix.index, cluster_labels))

        # Count clusters
        cluster_counts = defaultdict(int)
        for cluster_id in cluster_labels:
            cluster_counts[cluster_id] += 1

        # Log cluster size distribution
        cluster_sizes = list(cluster_counts.values())
        logger.info(
            f"Identified {len(cluster_counts)} correlation clusters "
            f"(avg size: {sum(cluster_sizes)/len(cluster_sizes):.1f}, "
            f"max size: {max(cluster_sizes)}, "
            f"min size: {min(cluster_sizes)})"
        )

        return ticker_to_cluster

    def select_diversified_stocks(
        self,
        signals: List[Tuple[StockSignal, Instrument]],
        corr_matrix: pd.DataFrame,
        target_count: int = 40
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Select stocks enforcing correlation constraints.

        Constraints:
        - Maximum pairwise correlation <= max_correlation
        - Maximum stocks per correlation cluster <= max_cluster_size

        Args:
            signals: List of (signal, instrument) tuples
            corr_matrix: Correlation matrix
            target_count: Target number of stocks to select

        Returns:
            Selected list of (signal, instrument) tuples
        """
        logger.info(
            f"Selecting {target_count} diversified stocks from {len(signals)} candidates"
        )

        # Identify clusters
        ticker_to_cluster = self.identify_correlation_clusters(corr_matrix)

        # Track selected stocks and cluster counts
        selected: List[Tuple[StockSignal, Instrument]] = []
        cluster_counts = defaultdict(int)
        selected_tickers = set()

        # Sort signals by quality (use Sharpe ratio as proxy)
        sorted_signals = sorted(
            signals,
            key=lambda x: x[0].sharpe_ratio if x[0].sharpe_ratio is not None else 0.0,
            reverse=True
        )

        for signal, instrument in sorted_signals:
            if len(selected) >= target_count:
                break

            ticker = instrument.ticker
            cluster_id = ticker_to_cluster.get(ticker, 0)

            # Check cluster constraint
            if cluster_counts[cluster_id] >= self.max_cluster_size:
                logger.debug(
                    f"Skipping {ticker}: cluster {cluster_id} full "
                    f"({cluster_counts[cluster_id]} stocks)"
                )
                continue

            # Check pairwise correlation constraint
            violates_correlation = False
            for selected_ticker in selected_tickers:
                if ticker in corr_matrix.index and selected_ticker in corr_matrix.columns:
                    corr_value = corr_matrix.loc[ticker, selected_ticker]

                    # ═══════════════════════════════════════════════════════
                    # HANDLE MISSING DATA: Check for NaN correlations
                    # ═══════════════════════════════════════════════════════
                    if pd.isna(corr_value):
                        logger.debug(
                            f"Warning: {ticker} <-> {selected_ticker} has NaN correlation "
                            f"(insufficient data overlap)"
                        )
                        # Conservative approach: assume low correlation (allow pairing)
                        # Alternative: strict mode would skip this stock
                        corr = 0.0
                    else:
                        # Convert to float and take absolute value
                        # Type ignore: corr_value is numeric Scalar from correlation matrix
                        corr = abs(float(corr_value))  # type: ignore[arg-type]

                    if corr > self.max_correlation:
                        logger.debug(
                            f"Skipping {ticker}: correlation {corr:.3f} > "
                            f"{self.max_correlation} with {selected_ticker}"
                        )
                        violates_correlation = True
                        break

            if violates_correlation:
                continue

            # Add to selected
            selected.append((signal, instrument))
            selected_tickers.add(ticker)
            cluster_counts[cluster_id] += 1

        # Final summary
        logger.info(
            f"Selected {len(selected)}/{target_count} stocks with correlation constraints"
        )
        logger.info(f"  Max cluster size: {max(cluster_counts.values()) if cluster_counts else 0}")
        logger.info(f"  Unique clusters:  {len(cluster_counts)}")

        # Check if we met the target
        if len(selected) < target_count:
            shortage = target_count - len(selected)
            logger.warning(
                f"⚠️  Only selected {len(selected)}/{target_count} stocks "
                f"({shortage} short of target)"
            )
            logger.warning("    Consider relaxing correlation threshold or cluster size constraints")

        return selected

    def calculate_portfolio_correlation(
        self,
        positions_tickers: List[str],
        corr_matrix: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Calculate average and maximum correlation for portfolio.

        Args:
            positions_tickers: List of tickers in portfolio
            corr_matrix: Correlation matrix

        Returns:
            Tuple of (average_correlation, max_correlation)
        """
        if len(positions_tickers) < 2:
            return 0.0, 0.0

        # Extract submatrix
        sub_matrix = corr_matrix.loc[positions_tickers, positions_tickers]

        # Get upper triangle (excluding diagonal)
        upper_triangle = sub_matrix.values[np.triu_indices_from(sub_matrix.values, k=1)]

        avg_corr = upper_triangle.mean()
        max_corr = upper_triangle.max()

        return avg_corr, max_corr

    def get_correlation_report(
        self,
        positions_tickers: List[str],
        corr_matrix: pd.DataFrame,
        top_n: int = 5
    ) -> str:
        """
        Generate correlation report for portfolio.

        Args:
            positions_tickers: List of tickers in portfolio
            corr_matrix: Correlation matrix
            top_n: Number of highest correlations to report

        Returns:
            Formatted correlation report string
        """
        if len(positions_tickers) < 2:
            return "Portfolio too small for correlation analysis"

        # Extract submatrix
        sub_matrix = corr_matrix.loc[positions_tickers, positions_tickers]

        # Find highest correlations
        upper_triangle_indices = np.triu_indices_from(sub_matrix.values, k=1)
        correlations = []

        for i, j in zip(*upper_triangle_indices):
            ticker_i = sub_matrix.index[i]
            ticker_j = sub_matrix.columns[j]
            corr = sub_matrix.iloc[i, j]
            correlations.append((ticker_i, ticker_j, corr))

        # Sort by correlation (descending)
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # Build report
        avg_corr, max_corr = self.calculate_portfolio_correlation(positions_tickers, corr_matrix)

        report = [
            "CORRELATION ANALYSIS",
            "=" * 60,
            f"Average pairwise correlation: {avg_corr:.3f}",
            f"Maximum pairwise correlation: {max_corr:.3f}",
            f"\nTop {top_n} highest correlations:",
            "-" * 60
        ]

        for i, (ticker_i, ticker_j, corr) in enumerate(correlations[:top_n], 1):
            report.append(f"  {i}. {ticker_i:6s} <-> {ticker_j:6s}: {corr:6.3f}")

        return "\n".join(report)
