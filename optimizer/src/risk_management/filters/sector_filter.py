"""
Sector Filter - Filter stocks to ensure sector diversification.
"""

from typing import Tuple, Optional, Dict, Any, List, Set

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.src.risk_management.filters.protocol import StockFilterImpl


class SectorFilter(StockFilterImpl):
    """
    Filter stocks to ensure sector diversification.

    This filter limits the number of stocks from any single sector
    to prevent over-concentration.

    Features:
    - Maximum stocks per sector
    - Minimum sectors required
    - Sector blocklist (exclude certain sectors)
    - Stateful tracking (updates as signals pass)

    Note: This filter is stateful and must be reset between runs.
    """

    def __init__(
        self,
        max_per_sector: int = 3,
        min_sectors: int = 5,
        blocked_sectors: Optional[List[str]] = None,
        max_sector_weight: Optional[float] = None,
    ):
        """
        Initialize sector filter.

        Args:
            max_per_sector: Maximum stocks allowed per sector
            min_sectors: Minimum number of sectors (used for validation)
            blocked_sectors: Sectors to exclude entirely
            max_sector_weight: Maximum portfolio weight per sector (for context)
        """
        super().__init__()
        self._max_per_sector = max_per_sector
        self._min_sectors = min_sectors
        self._blocked_sectors = set(blocked_sectors or [])
        self._max_sector_weight = max_sector_weight

        # Stateful tracking
        self._sector_counts: Dict[str, int] = {}
        self._selected_tickers: Set[str] = set()

    @property
    def name(self) -> str:
        """Filter name."""
        return "SectorFilter"

    def reset(self) -> None:
        """Reset stateful tracking for a new filtering run."""
        self._sector_counts.clear()
        self._selected_tickers.clear()

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """
        Check if signal passes sector constraints.

        This is stateful - it tracks which sectors have been filled.
        """
        sector = signal.sector or "Unknown"

        # Check blocklist
        if sector in self._blocked_sectors:
            return False, f"blocked_sector({sector})"

        # Check sector count
        current_count = self._sector_counts.get(sector, 0)
        if current_count >= self._max_per_sector:
            return False, f"sector_full({sector}:{current_count}/{self._max_per_sector})"

        # Signal passes - update tracking
        self._sector_counts[sector] = current_count + 1
        self._selected_tickers.add(signal.ticker)

        return True, None

    def filter_batch(
        self,
        signals: List[StockSignalDTO],
    ) -> Tuple[List[StockSignalDTO], Dict[str, int]]:
        """
        Filter batch with automatic reset.

        Overrides parent to ensure state is reset before filtering.
        """
        self.reset()
        return super().filter_batch(signals)

    def get_sector_distribution(self) -> Dict[str, int]:
        """Get current sector distribution."""
        return dict(self._sector_counts)

    def get_selected_tickers(self) -> Set[str]:
        """Get tickers that passed the filter."""
        return self._selected_tickers.copy()

    def has_minimum_sectors(self) -> bool:
        """Check if minimum sector count is met."""
        return len(self._sector_counts) >= self._min_sectors

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "max_per_sector": self._max_per_sector,
            "min_sectors": self._min_sectors,
            "blocked_sectors": list(self._blocked_sectors),
            "max_sector_weight": f"{self._max_sector_weight:.0%}" if self._max_sector_weight else None,
        }


class SectorBalancer(StockFilterImpl):
    """
    Alternative sector filter that balances across sectors.

    Instead of a hard limit, this filter prioritizes sector diversity
    by selecting the best stock from each sector first.
    """

    def __init__(
        self,
        target_positions: int = 15,
        min_sectors: int = 5,
    ):
        """
        Initialize sector balancer.

        Args:
            target_positions: Target number of positions
            min_sectors: Minimum sectors to include
        """
        super().__init__()
        self._target_positions = target_positions
        self._min_sectors = min_sectors

    @property
    def name(self) -> str:
        """Filter name."""
        return "SectorBalancer"

    def _filter_single(self, signal: StockSignalDTO) -> Tuple[bool, Optional[str]]:
        """Not used for balancer - use select_balanced instead."""
        return True, None

    def select_balanced(
        self,
        signals: List[StockSignalDTO],
        sort_key: str = "sharpe_ratio",
    ) -> List[StockSignalDTO]:
        """
        Select signals with sector balance.

        Prioritizes:
        1. At least one stock from each sector
        2. Best stocks by sort_key within each sector

        Args:
            signals: Candidate signals
            sort_key: Attribute to sort by within each sector

        Returns:
            Selected signals with sector balance
        """
        # Group by sector
        by_sector: Dict[str, List[StockSignalDTO]] = {}
        for signal in signals:
            sector = signal.sector or "Unknown"
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(signal)

        # Sort within each sector
        def get_sort_value(s: StockSignalDTO) -> float:
            val = getattr(s, sort_key, None)
            return val if val is not None else float('-inf')

        for sector in by_sector:
            by_sector[sector].sort(key=get_sort_value, reverse=True)

        selected: List[StockSignalDTO] = []
        sectors = list(by_sector.keys())

        # Round-robin selection
        round_num = 0
        while len(selected) < self._target_positions and sectors:
            for sector in sectors[:]:
                if len(selected) >= self._target_positions:
                    break

                if round_num < len(by_sector[sector]):
                    selected.append(by_sector[sector][round_num])
                else:
                    sectors.remove(sector)

            round_num += 1

        return selected

    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of filter configuration."""
        return {
            "name": self.name,
            "target_positions": self._target_positions,
            "min_sectors": self._min_sectors,
        }
