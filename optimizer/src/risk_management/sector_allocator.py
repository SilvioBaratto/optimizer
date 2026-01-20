#!/usr/bin/env python3
from typing import List, Tuple, Dict
from collections import defaultdict

from optimizer.database.models.stock_signals import StockSignal, ConfidenceLevelEnum
from optimizer.database.models.universe import Instrument

DEFENSIVE_SECTORS = {"Consumer Defensive", "Healthcare", "Utilities"}  # Staples

CYCLICAL_SECTORS = {
    "Consumer Cyclical",  # Discretionary
    "Financial Services",
    "Industrials",
    "Basic Materials",  # Materials
    "Technology",
    "Energy",  # Mixed but often cyclical
}

COMMUNICATION_SECTORS = {"Communication Services"}

REAL_ESTATE_SECTORS = {"Real Estate"}

ALL_SECTORS = DEFENSIVE_SECTORS | CYCLICAL_SECTORS | COMMUNICATION_SECTORS | REAL_ESTATE_SECTORS

# Note: Empty/None sectors are handled by converting to "Unknown" in _group_by_sector()
# This affects ~0.8% of signals (62 out of 7,759) where yfinance returns empty sector data


class SectorAllocator:
    """
    Manages sector-level diversification for concentrated portfolios.
    """

    def __init__(
        self,
        max_sector_weight: float = 0.15,  # 15% max per sector
        min_sectors: int = 8,  # Require at least 8 sectors
        defensive_min_weight: float = 0.10,  # Min 10% in defensive
        max_industry_positions: int = 2,  # Max 2 stocks per industry
    ):
        """
        Initialize sector allocator.
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
        target_positions: int = 20,
    ) -> List[Tuple[StockSignal, Instrument]]:
        """
        Select stocks enforcing sector diversification constraints.
        """
        # Group candidates by sector
        sector_candidates = self._group_by_sector(candidates)

        max_positions_per_sector = int(target_positions * self.max_sector_weight)
        min_defensive_positions = int(target_positions * self.defensive_min_weight)

        # Select positions
        selected: List[Tuple[StockSignal, Instrument]] = []
        sector_counts = defaultdict(int)
        industry_counts = defaultdict(int)
        defensive_count = 0

        # PASS 1: Ensure minimum defensive exposure
        defensive_candidates = [
            (sig, inst)
            for sig, inst in candidates
            if sig.sector and self.is_defensive_sector(sig.sector)
        ]

        defensive_candidates_sorted = sorted(
            defensive_candidates,
            key=lambda x: self._get_quality_score(x[0]),
            reverse=True,
        )

        for signal, instrument in defensive_candidates_sorted:
            if defensive_count >= min_defensive_positions:
                break

            sector = signal.sector or "Unknown"
            industry = signal.industry or "Unknown"

            # Check sector limit
            if sector_counts[sector] >= max_positions_per_sector:
                continue

            # Check industry limit
            if industry_counts[industry] >= self.max_industry_positions:
                continue

            # SELECTED!
            selected.append((signal, instrument))
            sector_counts[sector] += 1
            industry_counts[industry] += 1
            defensive_count += 1

        # PASS 2: Fill remaining positions from all sectors
        remaining_candidates = [
            (sig, inst) for sig, inst in candidates if (sig, inst) not in selected
        ]

        remaining_candidates_sorted = sorted(
            remaining_candidates,
            key=lambda x: self._get_quality_score(x[0]),
            reverse=True,
        )

        for signal, instrument in remaining_candidates_sorted:
            if len(selected) >= target_positions:
                break

            sector = signal.sector or "Unknown"
            industry = signal.industry or "Unknown"

            # Check sector limit
            if sector_counts[sector] >= max_positions_per_sector:
                continue

            # Check industry limit
            if industry_counts[industry] >= self.max_industry_positions:
                continue

            # SELECTED!
            selected.append((signal, instrument))
            sector_counts[sector] += 1
            industry_counts[industry] += 1

        return selected

    def _group_by_sector(
        self, signals: List[Tuple[StockSignal, Instrument]]
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
        """

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

        if signal.confidence_level:
            confidence_adjustment = {
                ConfidenceLevelEnum.HIGH: 0.2,  # +20% bonus
                ConfidenceLevelEnum.MEDIUM: 0.0,  # neutral
                ConfidenceLevelEnum.LOW: -0.2,  # -20% penalty
            }
            adjustment = confidence_adjustment.get(signal.confidence_level, 0.0)
            score = score * (1.0 + adjustment)

        return score

    def get_allocation_summary(self, positions: List[Tuple[StockSignal, Instrument]]) -> str:
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
            "-" * 60,
        ]

        # Sort by count
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)

        for sector, count in sorted_sectors:
            weight = count / total if total > 0 else 0
            sector_type = "DEF" if sector in DEFENSIVE_SECTORS else "CYC"
            lines.append(f"  {sector:25s}: {count:2d} ({weight:5.1%}) [{sector_type}]")

        return "\n".join(lines)
