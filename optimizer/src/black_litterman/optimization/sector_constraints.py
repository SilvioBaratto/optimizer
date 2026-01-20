"""
Sector Constraints - Build sector weight constraints for optimization.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from optimizer.domain.models.stock_signal import StockSignalDTO
from optimizer.domain.models.portfolio import PositionDTO

logger = logging.getLogger(__name__)


class SectorConstraintBuilder:
    """
    Build sector constraint mappings for optimization.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def build_from_signals(
        self,
        signals: List[StockSignalDTO],
        tickers: List[str],
    ) -> Dict[str, List[int]]:
        """
        Build sector mapping from signals.
        """
        # Create ticker to index mapping
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}

        # Build sector mapping
        sector_mapping: Dict[str, List[int]] = {}

        for signal in signals:
            if signal.ticker not in ticker_to_idx:
                continue

            sector = signal.sector or "Unknown"
            idx = ticker_to_idx[signal.ticker]

            if sector not in sector_mapping:
                sector_mapping[sector] = []
            sector_mapping[sector].append(idx)

        self._logger.info(
            f"Built sector mapping: {len(sector_mapping)} sectors, " f"{len(signals)} signals"
        )

        for sector, indices in sorted(sector_mapping.items()):
            self._logger.debug(f"  {sector}: {len(indices)} stocks")

        return sector_mapping

    def build_from_positions(
        self,
        positions: List[PositionDTO],
        tickers: List[str],
    ) -> Dict[str, List[int]]:
        """
        Build sector mapping from positions.
        """
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        sector_mapping: Dict[str, List[int]] = {}

        for pos in positions:
            if pos.ticker not in ticker_to_idx:
                continue

            sector = pos.sector or "Unknown"
            idx = ticker_to_idx[pos.ticker]

            if sector not in sector_mapping:
                sector_mapping[sector] = []
            sector_mapping[sector].append(idx)

        return sector_mapping

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        ticker_column: str = "ticker",
        sector_column: str = "sector",
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, List[int]]:
        """
        Build sector mapping from DataFrame.
        """
        if tickers is None:
            tickers = df[ticker_column].tolist()

        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        sector_mapping: Dict[str, List[int]] = {}

        for _, row in df.iterrows():
            ticker = row[ticker_column]
            if ticker not in ticker_to_idx:
                continue

            sector = row.get(sector_column, "Unknown") or "Unknown"
            idx = ticker_to_idx[ticker]

            if sector not in sector_mapping:
                sector_mapping[sector] = []
            sector_mapping[sector].append(idx)

        return sector_mapping

    def get_sector_weights(
        self,
        weights: pd.Series,
        sector_mapping: Dict[str, List[int]],
    ) -> Dict[str, float]:
        """
        Calculate current sector weights.
        """
        w = weights.values
        return {
            sector: float(sum(w[i] for i in indices)) for sector, indices in sector_mapping.items()
        }

    def validate_sector_constraints(
        self,
        weights: pd.Series,
        sector_mapping: Dict[str, List[int]],
        max_sector_weight: float,
    ) -> Dict[str, bool]:
        """
        Check which sector constraints are satisfied.
        """
        sector_weights = self.get_sector_weights(weights, sector_mapping)
        return {sector: weight <= max_sector_weight for sector, weight in sector_weights.items()}


class CountryConstraintBuilder:
    """
    Build country constraint mappings for optimization.
    """

    def __init__(self):
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def build_from_positions(
        self,
        positions: List[PositionDTO],
        tickers: List[str],
    ) -> Dict[str, List[int]]:
        """
        Build country mapping from positions.
        """
        ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        country_mapping: Dict[str, List[int]] = {}

        for pos in positions:
            if pos.ticker not in ticker_to_idx:
                continue

            country = pos.country or "USA"  # Default to USA
            idx = ticker_to_idx[pos.ticker]

            if country not in country_mapping:
                country_mapping[country] = []
            country_mapping[country].append(idx)

        return country_mapping

    def infer_country(
        self,
        ticker: str,
        yfinance_ticker: Optional[str] = None,
    ) -> str:
        """
        Infer country from ticker.
        """
        # Check yfinance suffix
        if yfinance_ticker:
            suffix_to_country = {
                ".L": "UK",
                ".DE": "Germany",
                ".PA": "France",
                ".AS": "Netherlands",
                ".MI": "Italy",
                ".MC": "Spain",
                ".SW": "Switzerland",
                ".TO": "Canada",
                ".AX": "Australia",
                ".HK": "Hong Kong",
                ".T": "Japan",
            }
            for suffix, country in suffix_to_country.items():
                if yfinance_ticker.endswith(suffix):
                    return country

        # Check T212 ticker pattern
        if "_US_" in ticker:
            return "USA"
        if "_UK_" in ticker or "_LSE_" in ticker:
            return "UK"
        if "_DE_" in ticker:
            return "Germany"

        return "USA"  # Default
