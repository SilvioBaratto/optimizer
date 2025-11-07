#!/usr/bin/env python3
"""
Concentrated Portfolio Builder - Stock Selection for Portfolio Optimization
============================================================================
Selects 20 high-quality stocks from LARGE_GAIN momentum signals using systematic
filters for diversification and risk management.

Key Features:
- Quality filters (Sharpe≥0.5, Vol≤40%, Drawdown≥-30%)
- Correlation constraints (max 0.7 pairwise)
- Sector/industry concentration limits
- Country allocation constraints (max 60% per country)
- Affordability filters (Trading212 constraints)

Output: List of 20 selected stocks with metadata
Note: Stock weighting is handled separately by Black-Litterman optimizer

Usage:
    from src.risk_management import ConcentratedPortfolioBuilder

    builder = ConcentratedPortfolioBuilder(target_positions=20, capital=1500.0)
    selected_stocks = builder.build_portfolio()
    # Pass selected_stocks to Black-Litterman optimizer for weighting

Author: Portfolio Optimization System
"""

import logging
import sys
from pathlib import Path
from datetime import date as date_type
from typing import List, Tuple, Optional

# Project root for default output directory
project_root = Path(__file__).parent.parent.parent

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv
load_dotenv()

# Import database and models
from app.database import database_manager, init_db
from app.models.stock_signals import StockSignal, SignalEnum
from app.models.universe import Instrument


from src.risk_management.quality_filter import QualityFilter
from src.risk_management.correlation_analyzer import CorrelationAnalyzer
from src.risk_management.sector_allocator import SectorAllocator
from src.risk_management.portfolio_analytics import PortfolioAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConcentratedPortfolioBuilder:
    """
    Selects 20 high-quality stocks from LARGE_GAIN signals for portfolio optimization.

    Applies systematic filters (quality, correlation, sector, country, affordability)
    to transform hundreds of momentum signals into a final set of 20 stocks ready
    for Black-Litterman optimization.

    Output: List of selected stocks with metadata (no weights - that's BL's job)
    """

    def __init__(
        self,
        target_positions: int = 20,
        signal_date: Optional[date_type] = None,
        max_sector_weight: float = 0.15,
        max_industry_positions: int = 2,
        max_correlation: float = 0.7,
        min_sharpe_ratio: float = 0.7,
        max_volatility: float = 0.40,
        min_close_price: float = 5.0,
        max_country_weight: float = 0.60,
        capital: Optional[float] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize portfolio builder.

        Args:
            target_positions: Target number of stocks (default 20)
            signal_date: Date to analyze (defaults to most recent)
            max_sector_weight: Maximum weight per sector (default 15%)
            max_country_weight: Maximum weight per country (default 60%)
            max_industry_positions: Max stocks per industry (default 2)
            max_correlation: Max pairwise correlation (default 0.7)
            min_sharpe_ratio: Minimum Sharpe ratio (default 0.5)
            max_volatility: Maximum volatility (default 40%)
            min_close_price: Minimum stock price (default $5)
            capital: Total capital available (EUR). If specified, filters stocks by
                    affordability: min €1 (Trading212 minimum) to capital/target_positions
            output_dir: Directory for output files (default project_root/data/portfolios)
        """
        self.target_positions = target_positions
        self.signal_date = signal_date
        self.max_sector_weight = max_sector_weight
        self.max_industry_positions = max_industry_positions
        self.max_correlation = max_correlation
        self.min_sharpe_ratio = min_sharpe_ratio
        self.max_volatility = max_volatility
        self.min_close_price = min_close_price
        self.capital = capital
        self.max_country_weight = max_country_weight

        # Calculate affordability constraints if capital specified
        if capital:
            self.max_affordable_price = capital / target_positions
            self.min_affordable_price = 1.0  # Trading212 minimum investment per position
        else:
            self.max_affordable_price = None
            self.min_affordable_price = None

        # Set output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = project_root / "data" / "portfolios"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        # Use BASIC filters per portfolio theory (Chapter 5)
        # Theory specifies: Sharpe ratio, Volatility, Max Drawdown, Price filter
        # Enhanced filters (Sortino, Information ratio, Dollar volume, Quality score, Calmar)
        # are NOT mentioned in theory and reduced pool from 84 to 12 stocks
        self.quality_filter = QualityFilter(
            # Theory-based filters (Chapter 5: Quantitative Implementation)
            min_sharpe_ratio=min_sharpe_ratio,
            max_volatility=max_volatility,
            max_max_drawdown=-0.30,
            min_close_price=min_close_price,
            require_positive_alpha=False,
            min_data_quality_score=0.6,

            # Disable Phase 1 enhanced filters (not in theory)
            min_sortino_ratio=None,
            min_information_ratio=None,
            min_daily_dollar_volume=None,
            min_quality_score=None,

            # Disable Phase 2 enhanced filters (not in theory)
            min_calmar_ratio=None,
            max_beta=None
        )

        self.correlation_analyzer = CorrelationAnalyzer(
            max_correlation=max_correlation,
            max_cluster_size=max_industry_positions
        )

        self.sector_allocator = SectorAllocator(
            max_sector_weight=max_sector_weight,
            min_sectors=8,
            defensive_min_weight=0.10
        )

        # Selection state (no weights - just filtered stocks)
        self.large_gain_signals: List[Tuple[StockSignal, Instrument]] = []
        self.filtered_signals: List[Tuple[StockSignal, Instrument]] = []
        self.selected_stocks: List[Tuple[StockSignal, Instrument]] = []  # Final 20 stocks

    def fetch_large_gain_signals(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch LARGE_GAIN signals with full instrument details.

        Logic:
        1. Try to fetch signals for specified date (or today if not specified)
        2. If no signals found, fall back to most recent signal date in database

        Returns:
            List of (StockSignal, Instrument) tuples
        """
        from sqlalchemy import func

        date = self.signal_date or date_type.today()
        logger.info(f"Query parameters:")
        logger.info(f"  Signal date: {date}")
        logger.info(f"  Signal type: {SignalEnum.LARGE_GAIN.value}")

        with database_manager.get_session() as session:
            # Try fetching for specified date
            stmt = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .options(joinedload(StockSignal.instrument))
                .where(StockSignal.signal_type == SignalEnum.LARGE_GAIN)
                .where(StockSignal.signal_date == date)
            )

            logger.info("Executing database query...")
            results = session.execute(stmt).all()
            signals = [(signal, instrument) for signal, instrument in results]

            # If no signals found, fall back to most recent date
            if not signals:
                logger.warning(f"No LARGE_GAIN signals found for {date}")
                logger.info("Searching for most recent signal date...")

                most_recent_date = session.execute(
                    select(func.max(StockSignal.signal_date))
                    .where(StockSignal.signal_type == SignalEnum.LARGE_GAIN)
                ).scalar_one_or_none()

                if most_recent_date:
                    logger.info(f"✓ Found most recent signals from: {most_recent_date}")
                    logger.info(f"  Using signals from {most_recent_date} instead of {date}")

                    # Update signal_date for consistency
                    self.signal_date = most_recent_date

                    # Fetch signals for most recent date
                    stmt = (
                        select(StockSignal, Instrument)
                        .join(Instrument, StockSignal.instrument_id == Instrument.id)
                        .options(joinedload(StockSignal.instrument))
                        .where(StockSignal.signal_type == SignalEnum.LARGE_GAIN)
                        .where(StockSignal.signal_date == most_recent_date)
                    )

                    results = session.execute(stmt).all()
                    signals = [(signal, instrument) for signal, instrument in results]
                else:
                    logger.error("No LARGE_GAIN signals found in database at all")

        self.large_gain_signals = signals

        # Log distribution statistics
        from collections import defaultdict
        sector_counts = defaultdict(int)
        exchange_counts = defaultdict(int)

        for signal, instrument in signals:
            sector_counts[signal.sector or "Unknown"] += 1
            exchange_counts[signal.exchange_name or "Unknown"] += 1

        logger.info(f"✓ Found {len(signals)} LARGE_GAIN signals")
        logger.info(f"\n  Sector distribution (top 5):")
        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"    {sector:25s}: {count:3d} signals ({count/len(signals)*100:5.1f}%)")

        logger.info(f"\n  Exchange distribution (top 3):")
        for exchange, count in sorted(exchange_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            logger.info(f"    {exchange:25s}: {count:3d} signals ({count/len(signals)*100:5.1f}%)")

        # Show top signals by Sharpe ratio
        top_by_sharpe = sorted(
            [(sig, inst) for sig, inst in signals if sig.sharpe_ratio],
            key=lambda x: x[0].sharpe_ratio,
            reverse=True
        )[:10]

        if top_by_sharpe:
            logger.info(f"\n  Top 10 by Sharpe ratio:")
            for i, (sig, inst) in enumerate(top_by_sharpe, 1):
                logger.info(
                    f"    {i:2d}. {inst.ticker:10s} (Sharpe: {sig.sharpe_ratio:5.2f}, "
                    f"Vol: {sig.volatility*100:5.1f}%, {sig.sector or 'Unknown'})"
                )

        return signals

    def apply_affordability_filter(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Filter signals by affordability constraints based on available capital.

        For Trading212 fractional shares with €1 minimum position:
        - Minimum price: €1 (can invest at least €1)
        - Maximum price: capital / target_positions (for equal-weight allocation)

        Returns:
            Filtered list of (StockSignal, Instrument) tuples
        """
        if not self.capital:
            logger.info("No capital constraint specified - skipping affordability filter")
            return self.large_gain_signals

        logger.info("Affordability filter (Trading212 fractional shares):")
        logger.info(f"  Total capital:       €{self.capital:,.2f}")
        logger.info(f"  Target positions:    {self.target_positions}")
        logger.info(f"  Max price per stock: €{self.max_affordable_price:.2f} (capital / {self.target_positions})")
        logger.info(f"  Min price per stock: €{self.min_affordable_price:.2f} (Trading212 minimum)")
        logger.info("")

        # Filter by price range
        affordable = []
        too_expensive = []
        too_cheap = []

        for signal, instrument in self.large_gain_signals:
            price = signal.close_price

            if price is None:
                too_cheap.append((signal, instrument))
                continue

            # Type guards for None checks
            if self.min_affordable_price is None or self.max_affordable_price is None:
                # Should not happen since we check capital at start of method
                affordable.append((signal, instrument))
                continue

            if price < self.min_affordable_price:
                too_cheap.append((signal, instrument))
            elif price > self.max_affordable_price:
                too_expensive.append((signal, instrument))
            else:
                affordable.append((signal, instrument))

        # Log statistics
        total = len(self.large_gain_signals)
        logger.info(
            f"✓ Affordability filtering: {total} → {len(affordable)} signals "
            f"({len(affordable)/total*100:.1f}% affordable)"
        )
        logger.info(f"  Excluded (too expensive): {len(too_expensive)} stocks (price > €{self.max_affordable_price:.2f})")
        logger.info(f"  Excluded (too cheap):     {len(too_cheap)} stocks (price < €{self.min_affordable_price:.2f})")

        # Show price distribution of affordable stocks
        if affordable:
            prices = [sig.close_price for sig, inst in affordable if sig.close_price]
            if prices:
                import numpy as np
                logger.info(f"\n  Affordable stock price distribution:")
                logger.info(f"    Min:    €{min(prices):.2f}")
                logger.info(f"    25th:   €{np.percentile(prices, 25):.2f}")
                logger.info(f"    Median: €{np.percentile(prices, 50):.2f}")
                logger.info(f"    75th:   €{np.percentile(prices, 75):.2f}")
                logger.info(f"    Max:    €{max(prices):.2f}")

        # Show examples of excluded expensive stocks
        if too_expensive:
            logger.info(f"\n  Sample stocks excluded (too expensive, first 5):")
            sorted_expensive = sorted(too_expensive, key=lambda x: x[0].close_price or 0, reverse=True)
            for i, (sig, inst) in enumerate(sorted_expensive[:5], 1):
                logger.info(
                    f"    {i}. {inst.ticker:10s} - €{sig.close_price:>8.2f} "
                    f"({sig.sector or 'Unknown':20s})"
                )

        return affordable

    def apply_quality_filters(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply quality filters to LARGE_GAIN signals.

        Returns:
            Filtered list of (StockSignal, Instrument) tuples
        """
        logger.info("Quality filter criteria:")
        logger.info(f"  Sharpe ratio     ≥ {self.min_sharpe_ratio}")
        logger.info(f"  Volatility       ≤ {self.max_volatility:.0%}")
        logger.info(f"  Max drawdown     ≥ -30%")
        logger.info(f"  Price            ≥ ${self.min_close_price:.2f}")
        logger.info("")

        filtered = self.quality_filter.filter_signals(self.large_gain_signals)
        self.filtered_signals = filtered

        pass_rate = (len(filtered) / len(self.large_gain_signals)) * 100 if self.large_gain_signals else 0
        logger.info(
            f"✓ Quality filtering: {len(self.large_gain_signals)} → {len(filtered)} signals "
            f"({pass_rate:.1f}% pass rate)"
        )

        # Show examples of passed and failed stocks
        if filtered:
            logger.info(f"\n  Sample stocks that PASSED (first 5):")
            for i, (sig, inst) in enumerate(filtered[:5], 1):
                logger.info(
                    f"    {i}. {inst.ticker:10s} - Sharpe: {sig.sharpe_ratio or 0:5.2f}, "
                    f"Vol: {(sig.volatility or 0)*100:5.1f}%, "
                    f"DD: {(sig.max_drawdown or 0)*100:6.1f}%"
                )

        failed = set(self.large_gain_signals) - set(filtered)
        if failed:
            logger.info(f"\n  Sample stocks that FAILED (first 5):")
            for i, (sig, inst) in enumerate(list(failed)[:5], 1):
                reasons = []
                if sig.sharpe_ratio and sig.sharpe_ratio < self.min_sharpe_ratio:
                    reasons.append(f"Low Sharpe ({sig.sharpe_ratio:.2f})")
                if sig.volatility and sig.volatility > self.max_volatility:
                    reasons.append(f"High Vol ({sig.volatility*100:.1f}%)")
                if sig.max_drawdown and sig.max_drawdown < -0.30:
                    reasons.append(f"Large DD ({sig.max_drawdown*100:.1f}%)")

                logger.info(
                    f"    {i}. {inst.ticker:10s} - {', '.join(reasons) if reasons else 'Multiple filters'}"
                )

        return filtered

    def apply_sector_pre_allocation(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Sector Pre-Allocation - Select stocks from each sector BEFORE correlation filtering.

        Institutional Practice (from Chapter 3):
        - Target ~20 stocks total
        - 11 GICS sectors → aim for 2-3 stocks per sector for balanced exposure
        - Pick top stocks from each sector by Sharpe ratio
        - This prevents one sector (e.g., Financial Services) from dominating

        Strategy:
        1. Group filtered stocks by sector
        2. From each sector, select top 3-4 stocks by Sharpe ratio
        3. This creates a sector-diversified pool (30-40 stocks)
        4. Then apply correlation filter within this pre-diversified pool

        Returns:
            Sector-balanced list of (StockSignal, Instrument) tuples
        """
        from collections import defaultdict

        if not self.filtered_signals:
            logger.warning("No filtered signals available for sector pre-allocation")
            return []

        # Group by sector
        sector_groups = defaultdict(list)
        for signal, instrument in self.filtered_signals:
            sector = signal.sector or "Unknown"
            sector_groups[sector].append((signal, instrument))

        logger.info(f"Sector distribution in quality-filtered pool ({len(self.filtered_signals)} stocks):")
        for sector in sorted(sector_groups.keys()):
            count = len(sector_groups[sector])
            pct = (count / len(self.filtered_signals)) * 100
            logger.info(f"  {sector:30s}: {count:3d} stocks ({pct:5.1f}%)")

        # Calculate how many stocks to take per sector
        # Target: 40 stocks pre-allocated (2x target portfolio of 20)
        # Distributed across sectors proportionally, but with min/max limits
        target_preallocated = self.target_positions * 2  # 40 stocks
        num_sectors = len(sector_groups)

        # Min 2 stocks per sector (if available), max 8 stocks per sector
        # Relaxed max from 6 to 8 to ensure we get ~40 stocks
        min_per_sector = 2
        max_per_sector = 8

        logger.info(f"\nSector pre-allocation strategy:")
        logger.info(f"  Target pre-allocated stocks: {target_preallocated}")
        logger.info(f"  Number of sectors: {num_sectors}")
        logger.info(f"  Range per sector: {min_per_sector}-{max_per_sector} stocks")
        logger.info("")

        # Pre-allocate stocks from each sector
        preallocated = []
        allocation_log = []

        for sector in sorted(sector_groups.keys()):
            stocks_in_sector = sector_groups[sector]

            # Sort by Sharpe ratio (best quality first)
            stocks_sorted = sorted(stocks_in_sector, key=lambda x: x[0].sharpe_ratio or 0, reverse=True)

            # Determine allocation for this sector
            # Start with proportional allocation
            sector_proportion = len(stocks_in_sector) / len(self.filtered_signals)
            ideal_allocation = int(target_preallocated * sector_proportion)

            # Apply min/max constraints
            actual_allocation = max(min_per_sector, min(max_per_sector, ideal_allocation))
            actual_allocation = min(actual_allocation, len(stocks_in_sector))  # Can't exceed available

            # Select top N from this sector
            selected_from_sector = stocks_sorted[:actual_allocation]
            preallocated.extend(selected_from_sector)

            allocation_log.append((sector, len(stocks_in_sector), actual_allocation))

        logger.info(f"Sector pre-allocation results:")
        for sector, available, allocated in sorted(allocation_log, key=lambda x: x[2], reverse=True):
            logger.info(f"  {sector:30s}: {allocated:2d} selected (from {available:3d} available)")

        logger.info(f"\n✓ Sector pre-allocation: {len(self.filtered_signals)} → {len(preallocated)} stocks")
        logger.info(f"  This pre-diversified pool will now undergo correlation filtering")

        return preallocated

    def apply_correlation_constraints(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply correlation clustering constraints.

        Returns:
            Decorrelated list of (StockSignal, Instrument) tuples
        """
        logger.info("Correlation analysis:")
        logger.info(f"  Max pairwise correlation allowed: {self.max_correlation:.2f}")
        logger.info(f"  Max stocks per cluster: {self.max_industry_positions}")
        logger.info(f"  Target stocks after filtering: {self.target_positions * 2}")
        logger.info("")

        # Build correlation matrix
        logger.info("Building correlation matrix...")
        corr_matrix = self.correlation_analyzer.build_correlation_matrix(
            self.filtered_signals
        )

        # Find highly correlated pairs
        import numpy as np
        n = len(corr_matrix)
        high_corr_pairs = []

        # Access underlying numpy array for proper type inference
        corr_values = corr_matrix.values

        for i in range(n):
            for j in range(i+1, n):
                corr_value = float(corr_values[i, j])
                if corr_value > self.max_correlation:
                    high_corr_pairs.append((
                        corr_matrix.index[i],
                        corr_matrix.index[j],
                        corr_value
                    ))

        if high_corr_pairs:
            logger.info(f"\n  Found {len(high_corr_pairs)} pairs exceeding correlation threshold")
            logger.info(f"  Top 5 most correlated pairs:")
            for ticker1, ticker2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
                logger.info(f"    {ticker1:10s} <-> {ticker2:10s}: {corr:5.3f}")

        # Select diversified stocks (target ~2x final positions)
        logger.info(f"\nSelecting decorrelated stocks...")
        decorrelated = self.correlation_analyzer.select_diversified_stocks(
            self.filtered_signals,
            corr_matrix,
            target_count=self.target_positions * 2
        )

        # ═══════════════════════════════════════════════════════════════════
        # VERIFICATION: Check if we got sufficient stocks
        # ═══════════════════════════════════════════════════════════════════
        target_count = self.target_positions * 2
        min_required = int(self.target_positions * 1.25)  # Need at least 1.25x for good selection (25 for 20-stock portfolio)

        if len(decorrelated) < min_required:
            logger.error("=" * 80)
            logger.error("❌ INSUFFICIENT DECORRELATED STOCKS")
            logger.error("=" * 80)
            logger.error(f"  Got:      {len(decorrelated)} stocks")
            logger.error(f"  Need:     {min_required} minimum (ideally {target_count})")
            logger.error(f"  Missing:  {min_required - len(decorrelated)} stocks")
            logger.error("")
            logger.error("Possible solutions:")
            logger.error(f"  1. Relax correlation threshold: {self.max_correlation:.2f} → {self.max_correlation + 0.05:.2f}")
            logger.error(f"  2. Relax quality filters (to get more input stocks)")
            logger.error(f"  3. Increase max_cluster_size: {self.max_industry_positions} → {self.max_industry_positions + 1}")
            logger.error(f"  4. Accept smaller portfolio (e.g., 15 stocks instead of {self.target_positions})")
            logger.error("=" * 80)
            raise ValueError(
                f"Insufficient decorrelated stocks: {len(decorrelated)} < {min_required} required. "
                f"Relax constraints or reduce portfolio size."
            )

        # ═══════════════════════════════════════════════════════════════════
        # VERIFICATION: Calculate actual max correlation in result
        # ═══════════════════════════════════════════════════════════════════
        if len(decorrelated) >= 2:
            decorrelated_tickers = [inst.ticker for _, inst in decorrelated]
            final_corr_matrix = corr_matrix.loc[decorrelated_tickers, decorrelated_tickers]

            # Get upper triangle values (excluding diagonal)
            upper_triangle = final_corr_matrix.values[
                np.triu_indices_from(final_corr_matrix.values, k=1)
            ]

            # Filter out NaN values before calculating statistics
            valid_correlations = upper_triangle[~np.isnan(upper_triangle)]

            if len(valid_correlations) > 0:
                actual_max_corr = float(valid_correlations.max())
                actual_avg_corr = float(valid_correlations.mean())
                nan_count = len(upper_triangle) - len(valid_correlations)

                logger.info(f"✓ Correlation filtering: {len(self.filtered_signals)} → {len(decorrelated)} signals")
                logger.info(f"  Actual average correlation: {actual_avg_corr:.3f}")
                logger.info(f"  Actual max correlation:     {actual_max_corr:.3f}")

                if nan_count > 0:
                    logger.warning(
                        f"  ⚠️  {nan_count} NaN correlations "
                        f"({nan_count/len(upper_triangle)*100:.1f}% of pairs)"
                    )

                # Warning if constraint violated
                if actual_max_corr > self.max_correlation:
                    logger.warning("=" * 80)
                    logger.warning("⚠️  CORRELATION CONSTRAINT VIOLATION")
                    logger.warning("=" * 80)
                    logger.warning(
                        f"  Target max correlation: {self.max_correlation:.3f}"
                    )
                    logger.warning(f"  Actual max correlation: {actual_max_corr:.3f}")
                    logger.warning(
                        f"  Exceeded by:            {actual_max_corr - self.max_correlation:.3f}"
                    )
                    logger.warning("")
                    logger.warning("This may indicate:")
                    logger.warning("  - Insufficient stocks to satisfy constraint")
                    logger.warning("  - Greedy algorithm selecting suboptimal combination")
                    logger.warning("  - Need to tighten quality filters or relax correlation threshold")
                    logger.warning("=" * 80)

                    # Only fail if significantly exceeds (allow 5% tolerance for greedy algorithm)
                    if actual_max_corr > self.max_correlation + 0.05:
                        logger.error("❌ Correlation significantly exceeds threshold!")
                        raise ValueError(
                            f"Max correlation {actual_max_corr:.3f} significantly exceeds "
                            f"threshold {self.max_correlation:.3f}"
                        )
            else:
                logger.warning("⚠️  All correlations are NaN (insufficient historical data)")
                logger.warning("     Proceeding without correlation verification")
                logger.info(f"✓ Correlation filtering: {len(self.filtered_signals)} → {len(decorrelated)} signals")
        else:
            logger.info(f"✓ Correlation filtering: {len(self.filtered_signals)} → {len(decorrelated)} signals")

        # Log sample of selected stocks
        logger.info(f"\n  Selected stocks (first 10):")
        for i, (sig, inst) in enumerate(decorrelated[:10], 1):
            logger.info(
                f"    {i:2d}. {inst.ticker:10s} ({sig.sector or 'Unknown':20s}) - "
                f"Sharpe: {sig.sharpe_ratio or 0:5.2f}"
            )

        return decorrelated

    def apply_sector_allocation(
        self,
        candidates: List[Tuple[StockSignal, Instrument]]
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply sector diversification constraints.

        Args:
            candidates: Candidate (signal, instrument) tuples

        Returns:
            Sector-balanced list of (signal, instrument) tuples
        """
        logger.info("Sector allocation constraints:")
        logger.info(f"  Max weight per sector: {self.max_sector_weight:.0%}")
        logger.info(f"  Min sectors required: 8")
        logger.info(f"  Min defensive allocation: 10%")
        logger.info(f"  Max stocks per industry: {self.max_industry_positions}")
        logger.info("")

        sector_balanced = self.sector_allocator.allocate_by_sector(
            candidates,
            target_positions=self.target_positions
        )

        # Log selected stocks by sector
        from collections import defaultdict
        sector_stocks = defaultdict(list)
        for sig, inst in sector_balanced:
            sector_stocks[sig.sector or "Unknown"].append(inst.ticker)

        logger.info(f"✓ Selected {len(sector_balanced)} stocks across {len(sector_stocks)} sectors")
        logger.info(f"\n  Stock selection by sector:")
        for sector in sorted(sector_stocks.keys(), key=lambda s: len(sector_stocks[s]), reverse=True):
            tickers = ", ".join(sector_stocks[sector])
            logger.info(f"    {sector:25s} ({len(sector_stocks[sector]):2d}): {tickers}")

        return sector_balanced

    def _backfill_geographic_diversity(
        self,
        selected: List[Tuple[StockSignal, Instrument]],
        country_counts: dict,
        max_stocks_per_country: int,
        shortfall: int
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Backfill portfolio with additional non-US stocks when country constraints cause shortfall.

        Strategy (Institutional Practice):
        - Query database for additional LARGE_GAIN signals from underrepresented countries
        - Apply relaxed quality thresholds (Sharpe >= 0.4, Vol <= 45%)
        - Prioritize geographic diversity over marginal quality differences
        - Stop when target_positions reached or no more candidates available

        Args:
            selected: Currently selected stocks
            country_counts: Current country allocation counts
            max_stocks_per_country: Maximum stocks allowed per country
            shortfall: Number of additional stocks needed

        Returns:
            Expanded list with backfilled stocks
        """
        logger.info(f"  Backfilling {shortfall} stocks with relaxed non-US constraints...")

        # Identify underrepresented countries
        selected_tickers = {inst.ticker for _, inst in selected}

        # Query database for additional LARGE_GAIN signals (not already selected)
        with database_manager.get_session() as session:
            # Get most recent signal date
            if self.signal_date:
                target_date = self.signal_date
            else:
                from sqlalchemy import desc
                latest = session.execute(
                    select(StockSignal.signal_date)
                    .order_by(desc(StockSignal.signal_date))
                    .limit(1)
                ).scalar_one_or_none()
                target_date = latest

            # Query for additional LARGE_GAIN stocks
            stmt = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .where(
                    StockSignal.signal_date == target_date,
                    StockSignal.signal_type == SignalEnum.LARGE_GAIN,
                    ~Instrument.ticker.in_(selected_tickers),  # Not already selected
                    StockSignal.close_price >= self.min_close_price,
                    # Relaxed quality thresholds for geographic diversity
                    StockSignal.sharpe_ratio >= 0.4,  # Was 0.5
                    StockSignal.volatility <= 0.45,   # Was 0.40
                    StockSignal.max_drawdown >= -0.35  # Was -0.30
                )
                .options(joinedload(StockSignal.instrument))
                .order_by(StockSignal.sharpe_ratio.desc())  # Order by quality
            )

            backfill_candidates = session.execute(stmt).all()

        logger.info(f"  Found {len(backfill_candidates)} potential backfill candidates")

        # Calculate max stocks per sector (15% of target positions)
        max_stocks_per_sector = int(self.max_sector_weight * self.target_positions)

        # Build current sector counts from already-selected stocks
        from collections import defaultdict
        sector_counts = defaultdict(int)
        for sig, _ in selected:
            sector = sig.sector or "Unknown"
            sector_counts[sector] += 1

        # Add stocks, respecting both country AND sector limits
        backfill_added = 0
        for signal, instrument in backfill_candidates:
            if len(selected) >= self.target_positions:
                break

            country = PortfolioAnalytics.get_country(signal, instrument)
            sector = signal.sector or "Unknown"

            # Skip if would violate country limit
            if country_counts[country] >= max_stocks_per_country:
                logger.debug(f"    Skipping {instrument.ticker}: would exceed country limit ({country})")
                continue

            # Skip if would violate sector limit (NEW)
            if sector_counts[sector] >= max_stocks_per_sector:
                logger.debug(f"    Skipping {instrument.ticker}: would exceed sector limit ({sector})")
                continue

            # Prioritize non-US stocks for geographic diversity
            if country == 'USA' and backfill_added < shortfall // 2:
                # Only add USA stocks after we've added some non-US
                continue

            selected.append((signal, instrument))
            country_counts[country] += 1
            sector_counts[sector] += 1
            backfill_added += 1

            logger.info(f"    ✓ Added {instrument.ticker} ({country}, {sector}) - Sharpe: {signal.sharpe_ratio:.2f}")

        logger.info(f"  Backfill complete: added {backfill_added}/{shortfall} stocks")

        # Log final sector distribution after backfill
        if backfill_added > 0:
            logger.info(f"\n  Final sector distribution after backfill:")
            for sector in sorted(sector_counts.keys(), key=lambda s: sector_counts[s], reverse=True):
                count = sector_counts[sector]
                pct = (count / len(selected)) * 100
                status = "✓" if count <= max_stocks_per_sector else "⚠️"
                logger.info(f"    {status} {sector:30s}: {count:2d} stocks ({pct:5.1f}%)")

        if len(selected) < self.target_positions:
            remaining = self.target_positions - len(selected)
            logger.warning(f"  ⚠️  Still {remaining} stocks short after backfill")
            logger.warning(f"      Consider: (1) relaxing constraints further, (2) expanding universe")

        return selected

    def apply_country_allocation(
        self,
        candidates: List[Tuple[StockSignal, Instrument]]
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply country diversification constraints.

        Theory Compliance (Ch. 3, §3.1):
        - MSCI Barra: ±5% country tilts from global benchmark
        - Institutional practice: Max 60% single country exposure
        - Our implementation: 60% maximum per country (avoids 80% US concentration)

        Args:
            candidates: Candidate (signal, instrument) tuples

        Returns:
            Country-balanced list of (signal, instrument) tuples
        """
        from collections import defaultdict

        logger.info("Country allocation constraints:")
        logger.info(f"  Max weight per country: {self.max_country_weight:.0%}")
        logger.info(f"  Min countries required: 3")
        logger.info(f"  Target geographic diversity: 40% non-US")
        logger.info("")

        # Group candidates by country
        country_stocks = defaultdict(list)
        for sig, inst in candidates:
            country = PortfolioAnalytics.get_country(sig, inst)
            country_stocks[country].append((sig, inst))

        # Log current distribution
        logger.info(f"  Available stocks by country:")
        for country in sorted(country_stocks.keys(), key=lambda c: len(country_stocks[c]), reverse=True):
            logger.info(f"    {country:20s}: {len(country_stocks[country]):2d} stocks")

        # Calculate max stocks per country based on weight constraint
        # With equal weighting (1/20 = 5% per stock), max 60% means max 12 stocks
        max_stocks_per_country = int(self.max_country_weight * self.target_positions)

        logger.info(f"\n  Max stocks per country: {max_stocks_per_country} "
                   f"(based on {self.max_country_weight:.0%} × {self.target_positions} positions)")

        # Select stocks respecting country limits
        country_balanced = []
        country_counts = defaultdict(int)

        # Sort candidates by composite score
        scored_candidates = [
            (sig, inst, PortfolioAnalytics.calculate_composite_score(sig))
            for sig, inst in candidates
        ]
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[2], reverse=True)

        for sig, inst, _ in sorted_candidates:
            country = PortfolioAnalytics.get_country(sig, inst)

            # Check if adding this stock would violate country constraint
            if country_counts[country] >= max_stocks_per_country:
                logger.debug(f"    Skipping {inst.ticker} ({country}): would exceed country limit")
                continue

            country_balanced.append((sig, inst))
            country_counts[country] += 1

            # Stop when we have enough stocks
            if len(country_balanced) >= self.target_positions:
                break

        # Backfill if we don't have enough stocks
        if len(country_balanced) < self.target_positions:
            shortfall = self.target_positions - len(country_balanced)
            logger.warning(f"⚠️  Only selected {len(country_balanced)}/{self.target_positions} stocks")
            logger.warning(f"    Country constraints caused {shortfall}-stock shortfall")
            logger.info(f"\n🔄 Attempting backfill with relaxed constraints for non-US stocks...")

            country_balanced = self._backfill_geographic_diversity(
                country_balanced,
                country_counts,
                max_stocks_per_country,
                shortfall
            )

        # Log final distribution
        final_country_stocks = defaultdict(list)
        for sig, inst in country_balanced:
            country = PortfolioAnalytics.get_country(sig, inst)
            final_country_stocks[country].append(inst.ticker)

        logger.info(f"\n✓ Selected {len(country_balanced)} stocks across {len(final_country_stocks)} countries")
        logger.info(f"\n  Stock selection by country:")
        for country in sorted(final_country_stocks.keys(), key=lambda c: len(final_country_stocks[c]), reverse=True):
            tickers = ", ".join(final_country_stocks[country])
            percentage = (len(final_country_stocks[country]) / self.target_positions) * 100
            logger.info(f"    {country:20s} ({len(final_country_stocks[country]):2d} = {percentage:5.1f}%): {tickers}")

        # Verify country constraint compliance
        for country, stocks in final_country_stocks.items():
            count = len(stocks)
            weight = count / self.target_positions
            if weight > self.max_country_weight:
                logger.error(f"❌ Country constraint violated: {country} = {weight:.1%} > {self.max_country_weight:.0%}")
                raise ValueError(
                    f"Country {country} has {count} stocks ({weight:.1%}) exceeding "
                    f"max {self.max_country_weight:.0%} limit"
                )

        return country_balanced

    def build_portfolio(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Execute stock selection pipeline - returns 20 stocks for BL optimization.

        Pipeline (Institutional Best Practice - Selection Only, NO Weighting):
        1. Fetch LARGE_GAIN signals (~500+ stocks)
        2. Apply affordability filter (if capital specified)
        3. Apply quality filters (Sharpe, volatility, drawdown)
        4. **Sector pre-allocation** (NEW - 2-6 stocks per sector, prevents dominance)
        5. Apply correlation constraints (max 0.7 pairwise, on pre-diversified pool)
        6. Final sector allocation fine-tuning (max 15% per sector)
        7. Apply country allocation (max 60% per country)
        8. Select final 20 stocks

        Theory Basis (Chapter 3 - Diversification):
        - Sector allocation comes BEFORE correlation filtering (institutional practice)
        - Ensures sector diversity prevents one sector from dominating
        - Correlation filter then works on pre-diversified pool
        - Prevents Financial Services (or any sector) from rejecting all others

        Returns:
            List of 20 (StockSignal, Instrument) tuples selected for optimization

        Note: Weighting is handled by Black-Litterman optimizer, not here!
        """
        logger.info("=" * 100)
        logger.info(" " * 30 + "STOCK SELECTION PIPELINE")
        logger.info(" " * 25 + "(For Black-Litterman Optimization)")
        logger.info("=" * 100)

        # Step 1: Fetch signals
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 1/7] FETCHING LARGE_GAIN SIGNALS")
        logger.info("─" * 100)
        self.fetch_large_gain_signals()

        if not self.large_gain_signals:
            raise ValueError("No LARGE_GAIN signals found. Cannot select stocks.")

        # Step 2: Affordability filter (if capital specified)
        if self.capital:
            logger.info("\n" + "─" * 100)
            logger.info("[STEP 2/7] APPLYING AFFORDABILITY FILTER")
            logger.info("─" * 100)
            affordable_signals = self.apply_affordability_filter()

            if not affordable_signals:
                raise ValueError(
                    f"No affordable signals found (capital: €{self.capital:,.2f}, "
                    f"max price: €{self.max_affordable_price:.2f}). "
                    f"Try increasing capital or relaxing constraints."
                )

            self.large_gain_signals = affordable_signals
        else:
            logger.info("\n[STEP 2/7] SKIPPING AFFORDABILITY FILTER (no capital specified)")

        # Step 3: Quality filters
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 3/7] APPLYING QUALITY FILTERS")
        logger.info("─" * 100)
        self.apply_quality_filters()

        # Step 4: Sector pre-allocation (NEW - before correlation)
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 4/7] SECTOR PRE-ALLOCATION (Institutional Practice)")
        logger.info("─" * 100)
        preallocated = self.apply_sector_pre_allocation()

        if not preallocated:
            raise ValueError("No stocks available after sector pre-allocation")

        # Step 5: Correlation analysis (on pre-diversified pool)
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 5/7] ANALYZING CORRELATIONS (within sector-diversified pool)")
        logger.info("─" * 100)
        # Temporarily store preallocated stocks for correlation analysis
        original_filtered = self.filtered_signals
        self.filtered_signals = preallocated
        decorrelated = self.apply_correlation_constraints()
        self.filtered_signals = original_filtered  # Restore

        # Step 6: Final sector allocation (fine-tuning)
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 6/7] FINAL SECTOR ALLOCATION")
        logger.info("─" * 100)
        sector_balanced = self.apply_sector_allocation(decorrelated)

        # Step 7: Country allocation
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 7/8] APPLYING COUNTRY ALLOCATION")
        logger.info("─" * 100)
        country_balanced = self.apply_country_allocation(sector_balanced)

        # Step 8: Select final 20 stocks
        logger.info("\n" + "─" * 100)
        logger.info("[STEP 8/8] SELECTING FINAL 20 STOCKS")
        logger.info("─" * 100)

        final_stocks = country_balanced[:self.target_positions]
        self.selected_stocks = final_stocks

        logger.info(f"✓ Selected {len(final_stocks)} stocks for portfolio optimization")
        logger.info("")
        logger.info("Stock list:")
        for i, (signal, instrument) in enumerate(final_stocks, 1):
            logger.info(
                f"  {i:2d}. {instrument.ticker:12s} | {signal.sector or 'Unknown':25s} | "
                f"Sharpe: {signal.sharpe_ratio:5.2f} | Vol: {signal.volatility:5.1%}"
            )

        # Selection complete
        logger.info("\n" + "─" * 100)
        logger.info("[COMPLETE] STOCK SELECTION COMPLETE")
        logger.info("─" * 100)
        logger.info(f"✓ {len(final_stocks)} stocks ready for Black-Litterman optimization")
        logger.info("=" * 100)

        return final_stocks



def main():
    """Main function - select 20 stocks for Black-Litterman optimization."""
    print("=" * 100)
    print(" " * 30 + "STOCK SELECTOR")
    print(" " * 25 + "(For BL Optimization)")
    print("=" * 100)

    try:
        # Initialize database
        logger.info("Initializing database connection...")
        init_db()
        logger.info("Database connection established")

        # Select 20 stocks with quality and diversification constraints
        # No weighting here - that's Black-Litterman's job!
        builder = ConcentratedPortfolioBuilder(
            target_positions=20,
            max_sector_weight=0.15,      # Max 3 stocks per sector (20 * 15%)
            max_country_weight=0.60,     # Max 12 stocks per country (20 * 60%)
            max_correlation=0.7,         # Max pairwise correlation
            capital=1500.0               # Trading212: €1 min, max €75 per stock
        )

        # Run selection pipeline
        selected_stocks = builder.build_portfolio()

        print(f"\n✓ {len(selected_stocks)} stocks selected and ready for Black-Litterman optimization")
        print("\nNext step: Pass these stocks to Black-Litterman optimizer for weight calculation")

    except KeyboardInterrupt:
        logger.info("\nStock selection interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Stock selection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
