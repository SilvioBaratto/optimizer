#!/usr/bin/env python3
import sys
from pathlib import Path
from datetime import date as date_type
from typing import List, Tuple, Optional
from collections import defaultdict

# Project root for default output directory
project_root = Path(__file__).parent.parent.parent

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from dotenv import load_dotenv

load_dotenv()

# Import database and models
from optimizer.database.database import database_manager, init_db
from optimizer.database.models.stock_signals import StockSignal, SignalEnum
from optimizer.database.models.universe import Instrument


from optimizer.src.risk_management.quality_filter import QualityFilter
from optimizer.src.risk_management.correlation_analyzer import CorrelationAnalyzer
from optimizer.src.risk_management.sector_allocator import SectorAllocator
from optimizer.src.risk_management.portfolio_analytics import PortfolioAnalytics


class ConcentratedPortfolioBuilder:
    """
    Selects 20 high-quality stocks from LARGE_GAIN signals for portfolio optimization.
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
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize portfolio builder.
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
            max_beta=None,
        )

        self.correlation_analyzer = CorrelationAnalyzer(
            max_correlation=max_correlation, max_cluster_size=max_industry_positions
        )

        self.sector_allocator = SectorAllocator(
            max_sector_weight=max_sector_weight,
            min_sectors=8,
            defensive_min_weight=0.10,
        )

        # Selection state (no weights - just filtered stocks)
        self.large_gain_signals: List[Tuple[StockSignal, Instrument]] = []
        self.filtered_signals: List[Tuple[StockSignal, Instrument]] = []
        self.selected_stocks: List[Tuple[StockSignal, Instrument]] = []  # Final 20 stocks

    def fetch_large_gain_signals(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Fetch LARGE_GAIN signals with full instrument details.
        """
        from sqlalchemy import func

        date = self.signal_date or date_type.today()

        with database_manager.get_session() as session:
            # Try fetching for specified date
            stmt = (
                select(StockSignal, Instrument)
                .join(Instrument, StockSignal.instrument_id == Instrument.id)
                .options(joinedload(StockSignal.instrument))
                .where(StockSignal.signal_type == SignalEnum.LARGE_GAIN)
                .where(StockSignal.signal_date == date)
            )

            results = session.execute(stmt).all()
            signals = [(signal, instrument) for signal, instrument in results]

            # If no signals found, fall back to most recent date
            if not signals:
                most_recent_date = session.execute(
                    select(func.max(StockSignal.signal_date)).where(
                        StockSignal.signal_type == SignalEnum.LARGE_GAIN
                    )
                ).scalar_one_or_none()

                if most_recent_date:
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

        self.large_gain_signals = signals
        return signals

    def apply_affordability_filter(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Filter signals by affordability constraints based on available capital.
        """
        if not self.capital:
            return self.large_gain_signals

        # Filter by price range
        affordable = []

        for signal, instrument in self.large_gain_signals:
            price = signal.close_price

            if price is None:
                continue

            # Type guards for None checks
            if self.min_affordable_price is None or self.max_affordable_price is None:
                # Should not happen since we check capital at start of method
                affordable.append((signal, instrument))
                continue

            if price < self.min_affordable_price:
                continue
            elif price > self.max_affordable_price:
                continue
            else:
                affordable.append((signal, instrument))

        return affordable

    def apply_quality_filters(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply quality filters to LARGE_GAIN signals.
        """
        filtered = self.quality_filter.filter_signals(self.large_gain_signals)
        self.filtered_signals = filtered
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
        if not self.filtered_signals:
            return []

        # Group by sector
        sector_groups = defaultdict(list)
        for signal, instrument in self.filtered_signals:
            sector = signal.sector or "Unknown"
            sector_groups[sector].append((signal, instrument))

        # Calculate how many stocks to take per sector
        # Target: 40 stocks pre-allocated (2x target portfolio of 20)
        # Distributed across sectors proportionally, but with min/max limits
        target_preallocated = self.target_positions * 2  # 40 stocks

        # Min 2 stocks per sector (if available), max 8 stocks per sector
        # Relaxed max from 6 to 8 to ensure we get ~40 stocks
        min_per_sector = 2
        max_per_sector = 8

        # Pre-allocate stocks from each sector
        preallocated = []

        for sector in sorted(sector_groups.keys()):
            stocks_in_sector = sector_groups[sector]

            # Sort by Sharpe ratio (best quality first)
            stocks_sorted = sorted(
                stocks_in_sector, key=lambda x: x[0].sharpe_ratio or 0, reverse=True
            )

            # Determine allocation for this sector
            # Start with proportional allocation
            sector_proportion = len(stocks_in_sector) / len(self.filtered_signals)
            ideal_allocation = int(target_preallocated * sector_proportion)

            # Apply min/max constraints
            actual_allocation = max(min_per_sector, min(max_per_sector, ideal_allocation))
            actual_allocation = min(
                actual_allocation, len(stocks_in_sector)
            )  # Can't exceed available

            # Select top N from this sector
            selected_from_sector = stocks_sorted[:actual_allocation]
            preallocated.extend(selected_from_sector)

        return preallocated

    def apply_correlation_constraints(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply correlation clustering constraints.

        Returns:
            Decorrelated list of (StockSignal, Instrument) tuples
        """
        import numpy as np

        # Build correlation matrix
        corr_matrix = self.correlation_analyzer.build_correlation_matrix(self.filtered_signals)

        # Select diversified stocks (target ~2x final positions)
        decorrelated = self.correlation_analyzer.select_diversified_stocks(
            self.filtered_signals, corr_matrix, target_count=self.target_positions * 2
        )

        # Verification: Check if we got sufficient stocks
        min_required = int(
            self.target_positions * 1.25
        )  # Need at least 1.25x for good selection (25 for 20-stock portfolio)

        if len(decorrelated) < min_required:
            raise ValueError(
                f"Insufficient decorrelated stocks: {len(decorrelated)} < {min_required} required. "
                f"Relax constraints or reduce portfolio size."
            )

        # Verification: Calculate actual max correlation in result
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

                # Only fail if significantly exceeds (allow 5% tolerance for greedy algorithm)
                if actual_max_corr > self.max_correlation + 0.05:
                    raise ValueError(
                        f"Max correlation {actual_max_corr:.3f} significantly exceeds "
                        f"threshold {self.max_correlation:.3f}"
                    )

        return decorrelated

    def apply_sector_allocation(
        self, candidates: List[Tuple[StockSignal, Instrument]]
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Apply sector diversification constraints.
        """
        sector_balanced = self.sector_allocator.allocate_by_sector(
            candidates, target_positions=self.target_positions
        )
        return sector_balanced

    def _backfill_geographic_diversity(
        self,
        selected: List[Tuple[StockSignal, Instrument]],
        country_counts: dict,
        max_stocks_per_country: int,
        shortfall: int,
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Backfill portfolio with additional non-US stocks when country constraints cause shortfall.
        """
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
                    select(StockSignal.signal_date).order_by(desc(StockSignal.signal_date)).limit(1)
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
                    StockSignal.volatility <= 0.45,  # Was 0.40
                    StockSignal.max_drawdown >= -0.35,  # Was -0.30
                )
                .options(joinedload(StockSignal.instrument))
                .order_by(StockSignal.sharpe_ratio.desc())  # Order by quality
            )

            backfill_candidates = session.execute(stmt).all()

        # Calculate max stocks per sector (15% of target positions)
        max_stocks_per_sector = int(self.max_sector_weight * self.target_positions)

        # Build current sector counts from already-selected stocks
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
                continue

            # Skip if would violate sector limit
            if sector_counts[sector] >= max_stocks_per_sector:
                continue

            # Prioritize non-US stocks for geographic diversity
            if country == "USA" and backfill_added < shortfall // 2:
                # Only add USA stocks after we've added some non-US
                continue

            selected.append((signal, instrument))
            country_counts[country] += 1
            sector_counts[sector] += 1
            backfill_added += 1

        return selected

    def apply_country_allocation(
        self, candidates: List[Tuple[StockSignal, Instrument]]
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
        # Group candidates by country
        country_stocks = defaultdict(list)
        for sig, inst in candidates:
            country = PortfolioAnalytics.get_country(sig, inst)
            country_stocks[country].append((sig, inst))

        # Calculate max stocks per country based on weight constraint
        # With equal weighting (1/20 = 5% per stock), max 60% means max 12 stocks
        max_stocks_per_country = int(self.max_country_weight * self.target_positions)

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
                continue

            country_balanced.append((sig, inst))
            country_counts[country] += 1

            # Stop when we have enough stocks
            if len(country_balanced) >= self.target_positions:
                break

        # Backfill if we don't have enough stocks
        if len(country_balanced) < self.target_positions:
            shortfall = self.target_positions - len(country_balanced)

            country_balanced = self._backfill_geographic_diversity(
                country_balanced, country_counts, max_stocks_per_country, shortfall
            )

        # Verify country constraint compliance
        final_country_stocks = defaultdict(list)
        for sig, inst in country_balanced:
            country = PortfolioAnalytics.get_country(sig, inst)
            final_country_stocks[country].append(inst.ticker)

        for country, stocks in final_country_stocks.items():
            count = len(stocks)
            weight = count / self.target_positions
            if weight > self.max_country_weight:
                raise ValueError(
                    f"Country {country} has {count} stocks ({weight:.1%}) exceeding "
                    f"max {self.max_country_weight:.0%} limit"
                )

        return country_balanced

    def build_portfolio(self) -> List[Tuple[StockSignal, Instrument]]:
        """
        Execute stock selection pipeline - returns 20 stocks for BL optimization.
        """
        # Step 1: Fetch signals
        self.fetch_large_gain_signals()

        if not self.large_gain_signals:
            raise ValueError("No LARGE_GAIN signals found. Cannot select stocks.")

        # Step 2: Affordability filter (if capital specified)
        if self.capital:
            affordable_signals = self.apply_affordability_filter()

            if not affordable_signals:
                raise ValueError(
                    f"No affordable signals found (capital: €{self.capital:,.2f}, "
                    f"max price: €{self.max_affordable_price:.2f}). "
                    f"Try increasing capital or relaxing constraints."
                )

            self.large_gain_signals = affordable_signals

        # Step 3: Quality filters
        self.apply_quality_filters()

        # Step 4: Sector pre-allocation (NEW - before correlation)
        preallocated = self.apply_sector_pre_allocation()

        if not preallocated:
            raise ValueError("No stocks available after sector pre-allocation")

        # Step 5: Correlation analysis (on pre-diversified pool)
        # Temporarily store preallocated stocks for correlation analysis
        original_filtered = self.filtered_signals
        self.filtered_signals = preallocated
        decorrelated = self.apply_correlation_constraints()
        self.filtered_signals = original_filtered  # Restore

        # Step 6: Final sector allocation (fine-tuning)
        sector_balanced = self.apply_sector_allocation(decorrelated)

        # Step 7: Country allocation
        country_balanced = self.apply_country_allocation(sector_balanced)

        # Step 8: Select final 20 stocks
        final_stocks = country_balanced[: self.target_positions]
        self.selected_stocks = final_stocks

        return final_stocks


def main():
    """Main function - select 20 stocks for Black-Litterman optimization."""
    try:
        # Initialize database
        init_db()

        # Select 20 stocks with quality and diversification constraints
        # No weighting here - that's Black-Litterman's job!
        builder = ConcentratedPortfolioBuilder(
            target_positions=20,
            max_sector_weight=0.15,  # Max 3 stocks per sector (20 * 15%)
            max_country_weight=0.60,  # Max 12 stocks per country (20 * 60%)
            max_correlation=0.75,  # Max pairwise correlation
            capital=1500.0,  # Trading212: €1 min, max €75 per stock
        )

        # Run selection pipeline
        selected_stocks = builder.build_portfolio()

        print(
            f"\n{len(selected_stocks)} stocks selected and ready for Black-Litterman optimization"
        )
        print("\nNext step: Pass these stocks to Black-Litterman optimizer for weight calculation")

    except KeyboardInterrupt:
        sys.exit(1)

    except Exception as e:
        print(f"Stock selection failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
