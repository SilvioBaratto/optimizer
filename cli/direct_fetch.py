"""Direct yfinance fetching without API — fallback when API is unavailable."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the api directory to Python path for imports
api_path = Path(__file__).parent.parent / "api"
if str(api_path) not in sys.path:
    sys.path.insert(0, str(api_path))

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.database import DatabaseManager
from app.models.universe import Instrument
from app.services.yfinance import YFinanceClient, get_yfinance_client
from app.services.yfinance_data_service import YFinanceDataService

logger = logging.getLogger(__name__)


class DirectFetcher:
    """Direct database and yfinance fetcher for CLI fallback."""

    def __init__(self):
        self._db_manager: Optional[DatabaseManager] = None
        self._yf_client: Optional[YFinanceClient] = None

    def _ensure_initialized(self) -> None:
        """Initialize database and yfinance client if not already done."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager()
            self._db_manager.initialize()
        if self._yf_client is None:
            self._yf_client = get_yfinance_client()

    def close(self) -> None:
        """Cleanup resources."""
        if self._db_manager:
            self._db_manager.close()
            self._db_manager = None

    def get_instruments_with_tickers(self) -> List[Dict[str, Any]]:
        """Get all instruments that have yfinance_ticker set."""
        self._ensure_initialized()

        with self._db_manager.get_session() as session:
            instruments = (
                session.execute(
                    select(Instrument)
                    .where(Instrument.yfinance_ticker.isnot(None))
                    .where(Instrument.yfinance_ticker != "")
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": str(inst.id),
                    "ticker": inst.ticker,
                    "yfinance_ticker": inst.yfinance_ticker,
                    "short_name": inst.short_name,
                }
                for inst in instruments
            ]

    def fetch_all(
        self,
        period: str = "5y",
        mode: str = "incremental",
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Fetch yfinance data for all instruments directly.

        Args:
            period: Price history period (e.g. "5y")
            mode: "incremental" or "full"
            progress_callback: Optional callback(current, total, ticker) for progress

        Returns:
            Dict with total_counts, errors, and skipped counts.
        """
        self._ensure_initialized()

        all_errors: List[str] = []
        total_counts: Dict[str, int] = {}
        total_skipped: int = 0

        with self._db_manager.get_session() as session:
            instruments = (
                session.execute(
                    select(Instrument)
                    .options(joinedload(Instrument.exchange))
                    .where(Instrument.yfinance_ticker.isnot(None))
                    .where(Instrument.yfinance_ticker != "")
                )
                .scalars()
                .unique()
                .all()
            )

            total = len(instruments)

            for idx, instrument in enumerate(instruments, 1):
                ticker = instrument.yfinance_ticker

                if progress_callback:
                    progress_callback(idx, total, ticker)

                try:
                    service = YFinanceDataService(session, self._yf_client)
                    result = service.fetch_and_store(
                        instrument_id=instrument.id,
                        yfinance_ticker=ticker,
                        period=period,
                        mode=mode,
                        exchange_name=instrument.exchange_name,
                    )
                    session.commit()

                    # Accumulate counts
                    for key, val in result.get("counts", {}).items():
                        total_counts[key] = total_counts.get(key, 0) + val

                    # Accumulate errors
                    for err in result.get("errors", []):
                        all_errors.append(f"{ticker}: {err}")

                    # Count skipped
                    total_skipped += len(result.get("skipped", []))

                except Exception as e:
                    logger.exception("Error fetching %s", ticker)
                    all_errors.append(f"{ticker}: {e}")
                    session.rollback()

        return {
            "processed": total,
            "counts": total_counts,
            "errors": all_errors,
            "skipped_categories": total_skipped,
        }

    def fetch_ticker(
        self,
        yfinance_ticker: str,
        period: str = "5y",
        mode: str = "incremental",
    ) -> Dict[str, Any]:
        """
        Fetch yfinance data for a single ticker directly.

        Args:
            yfinance_ticker: Yahoo Finance ticker symbol
            period: Price history period
            mode: "incremental" or "full"

        Returns:
            Dict with ticker, instrument_id, counts, errors, skipped.
        """
        self._ensure_initialized()

        with self._db_manager.get_session() as session:
            # Find instrument by yfinance_ticker (eager-load exchange)
            instrument = session.execute(
                select(Instrument)
                .options(joinedload(Instrument.exchange))
                .where(Instrument.yfinance_ticker == yfinance_ticker)
            ).scalar_one_or_none()

            if not instrument:
                return {
                    "ticker": yfinance_ticker,
                    "instrument_id": None,
                    "counts": {},
                    "errors": [f"No instrument found with yfinance_ticker={yfinance_ticker}"],
                    "skipped": [],
                }

            try:
                service = YFinanceDataService(session, self._yf_client)
                result = service.fetch_and_store(
                    instrument_id=instrument.id,
                    yfinance_ticker=yfinance_ticker,
                    period=period,
                    mode=mode,
                    exchange_name=instrument.exchange_name,
                )
                session.commit()

                return {
                    "ticker": yfinance_ticker,
                    "instrument_id": str(instrument.id),
                    "counts": result.get("counts", {}),
                    "errors": result.get("errors", []),
                    "skipped": result.get("skipped", []),
                }

            except Exception as e:
                logger.exception("Error fetching %s", yfinance_ticker)
                session.rollback()
                return {
                    "ticker": yfinance_ticker,
                    "instrument_id": str(instrument.id),
                    "counts": {},
                    "errors": [str(e)],
                    "skipped": [],
                }


# Module-level singleton for reuse
_direct_fetcher: Optional[DirectFetcher] = None


def get_direct_fetcher() -> DirectFetcher:
    """Get or create the direct fetcher singleton."""
    global _direct_fetcher
    if _direct_fetcher is None:
        _direct_fetcher = DirectFetcher()
    return _direct_fetcher
