"""Sub-client for ticker metadata: ISIN, fast_info, calendar, options, earnings dates."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any

import pandas as pd

from .._base import BaseClient
from ._isin import has_exchange_suffix, is_trusted_isin

logger = logging.getLogger(__name__)

_VALUATION_CURRENT_LABEL = "Current"
_VALUATION_PERIOD_LABEL_FORMAT = "%m/%d/%Y"


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _valuation_label_to_datetime(label: Any, as_of: date) -> datetime | None:
    if label == _VALUATION_CURRENT_LABEL:
        return datetime.combine(as_of, time.min)
    if isinstance(label, str):
        try:
            return datetime.strptime(label, _VALUATION_PERIOD_LABEL_FORMAT)
        except ValueError:
            return None
    if isinstance(label, datetime) and not pd.isna(label):
        return label
    return None


def _date_valuation_columns(panel: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Key a valuation panel's columns by date, as the statement repository expects.

    yfinance labels the panel's columns ``"Current"`` (the latest trailing value)
    and ``"M/D/YYYY"`` period ends. Neither parses as an ISO date, so the
    date-keyed repository would skip every column and store nothing.

    Args:
        panel: Panel as returned by ``Ticker.valuation``.
        as_of: Day to stamp on the ``"Current"`` column, normally the fetch day.

    Returns:
        The panel with a ``DatetimeIndex`` of columns. Columns whose label is not
        a date are dropped, and a period column that falls on *as_of* replaces
        ``"Current"`` so each date appears once.
    """
    if not any(isinstance(label, str) for label in panel.columns):
        return panel
    parsed = [
        (label, _valuation_label_to_datetime(label, as_of)) for label in panel.columns
    ]
    kept = [(label, stamp) for label, stamp in parsed if stamp is not None]
    dated = panel[[label for label, _ in kept]].copy()
    dated.columns = pd.DatetimeIndex([stamp for _, stamp in kept])
    return dated.loc[:, ~dated.columns.duplicated(keep="last")]


class MetadataClient(BaseClient):
    """Wraps ``yf.Ticker`` metadata-related attributes."""

    def fetch_isin(
        self,
        symbol: str,
        max_retries: int | None = None,
        *,
        raise_on_lost: bool = False,
    ) -> str | None:
        """Fetch ISIN for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.
            raise_on_lost: Raise instead of returning None when no attempt got an
                answer, so a caller can tell an unreachable source from a symbol
                Yahoo has no ISIN for.

        Returns:
            ISIN string, or None if unavailable. Symbols with an exchange suffix
            always give None without a request: yfinance's Business Insider scrape
            can only match them by name and answers with a secondary line (a
            depositary receipt, preferred share or bond). For other symbols
            yfinance's ``"-"`` placeholder and malformed codes are dropped too.

        Raises:
            RequestLostError: *raise_on_lost* is set and every attempt failed.
        """
        if has_exchange_suffix(symbol):
            logger.debug("Skipping ISIN scrape for '%s': not a US symbol", symbol)
            return None
        logger.debug("Fetching ISIN for '%s'", symbol)

        def _action() -> str | None:
            return self._get_ticker(symbol).isin

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: isinstance(v, str) and is_trusted_isin(v, symbol),
            raise_on_lost=raise_on_lost,
        )

    def fetch_fast_info(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch lightweight info snapshot for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Fast-info dict, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching fast_info for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            fi = self._get_ticker(symbol).fast_info
            if fi is None:
                return None
            return dict(fi)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_calendar(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch earnings and dividend calendar for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Calendar dict, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching calendar for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            cal = self._get_ticker(symbol).calendar
            if cal is None:
                return None
            if isinstance(cal, pd.DataFrame):
                return cal.to_dict()
            return dict(cal) if not isinstance(cal, dict) else cal

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_options_expirations(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> tuple[str, ...] | None:
        """Fetch available options expiration dates for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Tuple of ISO date strings, or None if no options are listed or all retries fail.
        """
        logger.debug("Fetching options expirations for '%s'", symbol)

        def _action() -> tuple[str, ...] | None:
            return self._get_ticker(symbol).options

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_option_chain(
        self,
        symbol: str,
        date: str | None = None,
        max_retries: int | None = None,
    ) -> Any | None:
        """Fetch option chain for symbol, optionally for a specific expiration date.

        Args:
            symbol: Ticker symbol.
            date: Expiration date (ISO format); None fetches the nearest expiration.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Named-tuple with ``.calls`` and ``.puts`` DataFrames, or None on failure.
        """
        logger.debug("Fetching option chain for '%s' (date=%s)", symbol, date)

        def _action() -> Any | None:
            ticker = self._get_ticker(symbol)
            if date is not None:
                return ticker.option_chain(date)
            return ticker.option_chain()

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None,
        )

    def fetch_earnings_dates(
        self,
        symbol: str,
        limit: int = 12,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch upcoming and past earnings dates for symbol.

        Args:
            symbol: Ticker symbol.
            limit: Maximum number of earnings events to return.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Earnings dates DataFrame, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching earnings dates for '%s' (limit=%d)", symbol, limit)

        def _action() -> pd.DataFrame | None:
            return self._get_ticker(symbol).get_earnings_dates(limit=limit)

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )

    def fetch_history_metadata(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch price history metadata (exchange, timezone, currency) for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            History metadata dict, or None if unavailable or all retries fail.
        """
        logger.debug("Fetching history metadata for '%s'", symbol)

        def _action() -> dict[str, Any] | None:
            return self._get_ticker(symbol).history_metadata

        return self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda v: v is not None and len(v) > 0,
        )

    def fetch_valuation_measures(
        self,
        symbol: str,
        max_retries: int | None = None,
    ) -> pd.DataFrame | None:
        """Fetch valuation measures panel for symbol.

        Args:
            symbol: Ticker symbol.
            max_retries: Retry attempts; None uses the client default.

        Returns:
            Valuation measures DataFrame whose columns are dates (yfinance's
            ``"Current"`` column is dated with the fetch day), or None if
            unavailable or all retries fail.
        """
        logger.debug("Fetching valuation measures for '%s'", symbol)

        def _action() -> pd.DataFrame | None:
            # yfinance names this panel ``Ticker.valuation`` (not
            # ``valuation_measures``); property access keeps parity with all
            # other MetadataClient methods.
            return self._get_ticker(symbol).valuation

        panel = self._fetch_with_resilience(
            _action,
            max_retries,
            is_valid=lambda df: df is not None and not df.empty,
        )
        if panel is None:
            return None
        return _date_valuation_columns(panel, _utc_today())
