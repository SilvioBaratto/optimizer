#!/usr/bin/env python3
"""
Sector Allocator - Manages Sector Diversification Constraints
==============================================================
Enforces sector-level diversification to prevent concentrated portfolios
from being dominated by a single sector or industry.

Constraints:
- Maximum weight per sector (default 15%)
- Minimum number of sectors (default 8)
- Minimum defensive sector allocation (default 10%)
- Maximum positions per industry (default 2)

Defensive Sectors: Consumer Staples, Healthcare, Utilities

Author: Portfolio Optimization System
"""

import logging
from typing import List, Tuple, Dict
from collections import defaultdict

from app.models.stock_signals import StockSignal, ConfidenceLevelEnum
from app.models.universe import Instrument

logger = logging.getLogger(__name__)


# GICS Sector Classification
# Note: These sectors match yfinance sector taxonomy (verified against stock_signals table)
# Database: 7,759 signals across 11 sectors + 143 industries
# Coverage: 99.2% of signals have sector data (62 signals have empty/None sectors)

DEFENSIVE_SECTORS = {
    "Consumer Defensive",  # Staples
    "Healthcare",
    "Utilities"
}

CYCLICAL_SECTORS = {
    "Consumer Cyclical",  # Discretionary
    "Financial Services",
    "Industrials",
    "Basic Materials",  # Materials
    "Technology",
    "Energy"  # Mixed but often cyclical
}

COMMUNICATION_SECTORS = {
    "Communication Services"
}

REAL_ESTATE_SECTORS = {
    "Real Estate"
}

ALL_SECTORS = DEFENSIVE_SECTORS | CYCLICAL_SECTORS | COMMUNICATION_SECTORS | REAL_ESTATE_SECTORS

# Note: Empty/None sectors are handled by converting to "Unknown" in _group_by_sector()
# This affects ~0.8% of signals (62 out of 7,759) where yfinance returns empty sector data


class SectorAllocator:
    """
    Manages sector-level diversification for concentrated portfolios.

    Ensures portfolio isn't dominated by single sector (e.g., all Technology)
    and includes defensive exposure for downside protection.
    """

    def __init__(
        self,
        max_sector_weight: float = 0.15,  # 15% max per sector
        min_sectors: int = 8,  # Require at least 8 sectors
        defensive_min_weight: float = 0.10,  # Min 10% in defensive
        max_industry_positions: int = 2  # Max 2 stocks per industry
    ):
        """
        Initialize sector allocator.

        Args:
            max_sector_weight: Maximum weight per sector (default 15%)
            min_sectors: Minimum number of sectors required (default 8)
            defensive_min_weight: Minimum allocation to defensive sectors (default 10%)
            max_industry_positions: Maximum positions per industry (default 2)
        """
        self.max_sector_weight = max_sector_weight
        self.min_sectors = min_sectors
        self.defensive_min_weight = defensive_min_weight
        self.max_industry_positions = max_industry_positions

    def is_defensive_sector(self, sector: str) -> bool:
        """Check if sector is defensive."""
        return sector in DEFENSIVE_SECTORS

    def allocate_by_sector(
        self,
        candidates: List[Tuple[StockSignal, Instrument]],
        target_positions: int = 20
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Select stocks enforcing sector diversification constraints.

        Args:
            candidates: List of (signal, instrument) tuples
            target_positions: Target number of positions (default 20)

        Returns:
            Sector-balanced list of (signal, instrument) tuples
        """
        logger.info(
            f"Allocating {target_positions} positions across sectors "
            f"from {len(candidates)} candidates"
        )

        # Group candidates by sector
        sector_candidates = self._group_by_sector(candidates)

        logger.info(
            f"Available sectors: {len(sector_candidates)} "
            f"({list(sector_candidates.keys())})"
        )

        # Check minimum sector requirement
        if len(sector_candidates) < self.min_sectors:
            logger.warning(
                f"Only {len(sector_candidates)} sectors available, "
                f"need {self.min_sectors} minimum"
            )

        # Calculate max positions per sector
        # FIX: Was dividing by (1.0/target_positions) which multiplied by target_positions again!
        # Old (WRONG): int(20 * 0.15 / (1.0/20)) = int(20 * 0.15 * 20) = 60 positions
        # New (CORRECT): int(20 * 0.15) = 3 positions
        max_positions_per_sector = int(target_positions * self.max_sector_weight)

        logger.info(f"Max positions per sector: {max_positions_per_sector} (15% of {target_positions})")

        # Calculate minimum defensive positions
        # FIX: Same math error as above
        # Old (WRONG): int(20 * 0.10 / (1.0/20)) = int(20 * 0.10 * 20) = 40 positions
        # New (CORRECT): int(20 * 0.10) = 2 positions
        min_defensive_positions = int(target_positions * self.defensive_min_weight)

        logger.info(f"Min defensive sector positions: {min_defensive_positions} (10% of {target_positions})")

        # Select positions
        selected: List[Tuple[StockSignal, Instrument]] = []
        sector_counts = defaultdict(int)
        industry_counts = defaultdict(int)
        defensive_count = 0

        # PASS 1: Ensure minimum defensive exposure
        defensive_candidates = [
            (sig, inst) for sig, inst in candidates
            if sig.sector and self.is_defensive_sector(sig.sector)
        ]

        defensive_candidates_sorted = sorted(
            defensive_candidates,
            key=lambda x: self._get_quality_score(x[0]),
            reverse=True
        )

        logger.info("")
        logger.info("PASS 1: Selecting defensive sector positions...")

        for signal, instrument in defensive_candidates_sorted:
            if defensive_count >= min_defensive_positions:
                break

            sector = signal.sector or "Unknown"
            industry = signal.industry or "Unknown"
            quality = self._get_quality_score(signal)

            # Check sector limit
            if sector_counts[sector] >= max_positions_per_sector:
                logger.debug(
                    f"  ⏭️  {instrument.ticker:6s}: Sector {sector} full "
                    f"({sector_counts[sector]}/{max_positions_per_sector})"
                )
                continue

            # Check industry limit
            if industry_counts[industry] >= self.max_industry_positions:
                logger.debug(
                    f"  ⏭️  {instrument.ticker:6s}: Industry {industry} full "
                    f"({industry_counts[industry]}/{self.max_industry_positions})"
                )
                continue

            # SELECTED!
            selected.append((signal, instrument))
            sector_counts[sector] += 1
            industry_counts[industry] += 1
            defensive_count += 1

            logger.info(
                f"  ✅ {instrument.ticker:6s} | {sector:20s} | {industry:30s} | "
                f"Quality: {quality:+.2f} | Sharpe: {signal.sharpe_ratio or 0:.2f}"
            )

        logger.info("")
        logger.info(f"Selected {defensive_count} defensive sector positions")

        # PASS 2: Fill remaining positions from all sectors
        remaining_candidates = [
            (sig, inst) for sig, inst in candidates
            if (sig, inst) not in selected
        ]

        remaining_candidates_sorted = sorted(
            remaining_candidates,
            key=lambda x: self._get_quality_score(x[0]),
            reverse=True
        )

        logger.info("")
        logger.info(f"PASS 2: Filling remaining positions (need {target_positions - len(selected)} more)...")

        for signal, instrument in remaining_candidates_sorted:
            if len(selected) >= target_positions:
                break

            sector = signal.sector or "Unknown"
            industry = signal.industry or "Unknown"
            quality = self._get_quality_score(signal)

            # Check sector limit
            if sector_counts[sector] >= max_positions_per_sector:
                logger.debug(
                    f"  ⏭️  {instrument.ticker:6s}: Sector {sector} full "
                    f"({sector_counts[sector]}/{max_positions_per_sector})"
                )
                continue

            # Check industry limit
            if industry_counts[industry] >= self.max_industry_positions:
                logger.debug(
                    f"  ⏭️  {instrument.ticker:6s}: Industry {industry} full "
                    f"({industry_counts[industry]}/{self.max_industry_positions})"
                )
                continue

            # SELECTED!
            selected.append((signal, instrument))
            sector_counts[sector] += 1
            industry_counts[industry] += 1

            logger.info(
                f"  ✅ {instrument.ticker:6s} | {sector:20s} | {industry:30s} | "
                f"Quality: {quality:+.2f} | Sharpe: {signal.sharpe_ratio or 0:.2f}"
            )

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"SECTOR ALLOCATION COMPLETE: {len(selected)} positions across {len(sector_counts)} sectors")
        logger.info("=" * 100)

        # Log sector distribution
        self._log_sector_distribution(sector_counts, len(selected))

        # Log detailed factor breakdown
        self._log_factor_breakdown(selected)

        return selected

    def _group_by_sector(
        self,
        signals: List[Tuple[StockSignal, Instrument]]
    ) -> Dict[str, List[Tuple[StockSignal, Instrument]]]:
        """Group signals by sector."""
        sector_groups = defaultdict(list)

        for signal, instrument in signals:
            sector = signal.sector or "Unknown"
            sector_groups[sector].append((signal, instrument))

        return dict(sector_groups)

    def _get_quality_score(self, signal: StockSignal) -> float:
        """
        Calculate composite ranking score using existing factor scores from stock_analyzer.

        Uses multi-factor framework (Value, Momentum, Quality, Growth) with scores
        already calculated and stored in the database.

        STRICT MODE: No fallbacks - raises ValueError if factor scores are missing.

        Returns:
            Composite score (typically -1 to +1 range, higher is better)

        Raises:
            ValueError: If required factor scores are missing
        """
        # ═══════════════════════════════════════════════════════════════
        # VALIDATE: All 4 factor scores must be present (no fallbacks!)
        # ═══════════════════════════════════════════════════════════════

        missing_factors = []

        if signal.valuation_score is None:
            missing_factors.append("valuation_score")
        if signal.momentum_score is None:
            missing_factors.append("momentum_score")
        if signal.quality_score is None:
            missing_factors.append("quality_score")
        if signal.growth_score is None:
            missing_factors.append("growth_score")

        if missing_factors:
            raise ValueError(
                f"Stock {signal.ticker} missing required factor scores: {', '.join(missing_factors)}. "
                f"All stocks must have complete factor data from stock_analyzer. "
                f"Re-run signal generation to populate missing factors."
            )

        # ═══════════════════════════════════════════════════════════════
        # CALCULATE: Multi-factor composite score (equal weighting)
        # ═══════════════════════════════════════════════════════════════

        # Type assertions for Pylance (we already validated above)
        assert signal.valuation_score is not None
        assert signal.momentum_score is not None
        assert signal.quality_score is not None
        assert signal.growth_score is not None

        # Valuation Factor (25% weight)
        # Range: -1 (overvalued) to +1 (undervalued)
        valuation_component = 0.25 * signal.valuation_score

        # Momentum Factor (25% weight)
        # Range: -1 (downtrend) to +1 (uptrend)
        momentum_component = 0.25 * signal.momentum_score

        # Quality Factor (25% weight)
        # Range: 0 (poor) to 1 (excellent) - need to center at 0
        # Convert 0-1 range to -0.5 to +0.5, then scale to -1 to +1
        centered_quality = (signal.quality_score - 0.5) * 2.0
        quality_component = 0.25 * centered_quality

        # Growth Factor (25% weight)
        # Range: -1 (contracting) to +1 (high growth)
        growth_component = 0.25 * signal.growth_score

        # Composite score
        score = valuation_component + momentum_component + quality_component + growth_component

        # ═══════════════════════════════════════════════════════════════
        # CONFIDENCE ADJUSTMENT: Boost/penalty based on LLM confidence
        # ═══════════════════════════════════════════════════════════════

        if signal.confidence_level:
            confidence_adjustment = {
                ConfidenceLevelEnum.HIGH: 0.2,      # +20% bonus
                ConfidenceLevelEnum.MEDIUM: 0.0,    # neutral
                ConfidenceLevelEnum.LOW: -0.2       # -20% penalty
            }
            adjustment = confidence_adjustment.get(signal.confidence_level, 0.0)
            score = score * (1.0 + adjustment)

        return score

    def _log_sector_distribution(
        self,
        sector_counts: Dict[str, int],
        total_positions: int
    ) -> None:
        """Log sector distribution statistics."""
        logger.info("")
        logger.info("SECTOR DISTRIBUTION:")
        logger.info("-" * 100)

        # Sort by count (descending)
        sorted_sectors = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for sector, count in sorted_sectors:
            weight = count / total_positions if total_positions > 0 else 0
            sector_type = "DEFENSIVE" if sector in DEFENSIVE_SECTORS else "CYCLICAL"
            logger.info(f"  {sector:25s}: {count:2d} positions ({weight:5.1%}) [{sector_type}]")

    def _log_factor_breakdown(
        self,
        selected: List[Tuple[StockSignal, Instrument]]
    ) -> None:
        """Log factor score breakdown for selected portfolio."""
        logger.info("")
        logger.info("FACTOR SCORE BREAKDOWN:")
        logger.info("-" * 100)

        # Calculate average factor scores
        valuation_scores = [sig.valuation_score for sig, _ in selected if sig.valuation_score is not None]
        momentum_scores = [sig.momentum_score for sig, _ in selected if sig.momentum_score is not None]
        quality_scores = [sig.quality_score for sig, _ in selected if sig.quality_score is not None]
        growth_scores = [sig.growth_score for sig, _ in selected if sig.growth_score is not None]

        # Calculate averages
        avg_valuation = sum(valuation_scores) / len(valuation_scores) if valuation_scores else 0
        avg_momentum = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_growth = sum(growth_scores) / len(growth_scores) if growth_scores else 0

        # Calculate quantitative metrics
        sharpe_ratios = [sig.sharpe_ratio for sig, _ in selected if sig.sharpe_ratio is not None]
        volatilities = [sig.volatility for sig, _ in selected if sig.volatility is not None]
        alphas = [sig.alpha for sig, _ in selected if sig.alpha is not None]

        avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
        avg_vol = sum(volatilities) / len(volatilities) if volatilities else 0
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0

        # Log factor scores
        logger.info(f"  Valuation (avg):  {avg_valuation:+.3f}  (-1=overvalued, +1=undervalued)")
        logger.info(f"  Momentum (avg):   {avg_momentum:+.3f}  (-1=downtrend, +1=uptrend)")
        logger.info(f"  Quality (avg):    {avg_quality:+.3f}  (0=poor, 1=excellent)")
        logger.info(f"  Growth (avg):     {avg_growth:+.3f}  (-1=contracting, +1=high growth)")
        logger.info("")
        logger.info(f"  Sharpe Ratio:     {avg_sharpe:+.2f}")
        logger.info(f"  Volatility:       {avg_vol:.1%}")
        logger.info(f"  Alpha:            {avg_alpha:+.2%}")
        logger.info("")

        # Count confidence levels
        confidence_counts = defaultdict(int)
        for sig, _ in selected:
            if sig.confidence_level:
                confidence_counts[sig.confidence_level.value] += 1

        logger.info(f"  Confidence: HIGH={confidence_counts.get('high', 0)}, "
                   f"MEDIUM={confidence_counts.get('medium', 0)}, "
                   f"LOW={confidence_counts.get('low', 0)}")
        logger.info("=" * 100)

    def get_allocation_summary(
        self,
        positions: List[Tuple[StockSignal, Instrument]]
    ) -> str:
        """Get formatted sector allocation summary."""
        sector_counts = defaultdict(int)
        industry_counts = defaultdict(int)
        defensive_count = 0

        for signal, _instrument in positions:
            sector = signal.sector or "Unknown"
            industry = signal.industry or "Unknown"

            sector_counts[sector] += 1
            industry_counts[industry] += 1

            if self.is_defensive_sector(sector):
                defensive_count += 1

        total = len(positions)

        lines = [
            "SECTOR ALLOCATION SUMMARY",
            "=" * 60,
            f"Total positions: {total}",
            f"Defensive positions: {defensive_count} ({defensive_count/total:.1%})",
            f"Unique sectors: {len(sector_counts)}",
            f"Unique industries: {len(industry_counts)}",
            "",
            "Sector breakdown:",
            "-" * 60
        ]

        # Sort by count
        sorted_sectors = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for sector, count in sorted_sectors:
            weight = count / total if total > 0 else 0
            sector_type = "DEF" if sector in DEFENSIVE_SECTORS else "CYC"
            lines.append(
                f"  {sector:25s}: {count:2d} ({weight:5.1%}) [{sector_type}]"
            )

        return "\n".join(lines)
