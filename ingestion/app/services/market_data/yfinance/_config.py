"""Process-wide yfinance logging, applied once by each entry point."""

from __future__ import annotations

import logging

_YFINANCE_LOGGER = "yfinance"


def configure_yfinance(*, log_level: str) -> None:
    """Set how much of yfinance's own logging reaches the operator.

    yfinance logs every final "no data" answer from Yahoo (a 404 for a data
    module a symbol lacks, missing earnings dates) at ``ERROR`` before
    returning empty data. Those answers are routine for most non-US listings,
    and the client layer already treats an empty result as final, so the
    library's records are hidden unless *log_level* asks for them.

    yfinance's ``hide_exceptions`` mode is deliberately left at its default:
    only that mode salvages partial results, such as growth estimates for a
    symbol whose own EPS-trend module is missing.

    Args:
        log_level: Logging level name for the ``yfinance`` logger, matched
            case-insensitively. ``"CRITICAL"`` hides the library's errors;
            ``"DEBUG"`` shows all of its diagnostics.

    Raises:
        ValueError: If *log_level* is not a standard logging level name.
    """
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level!r}")
    logging.getLogger(_YFINANCE_LOGGER).setLevel(level)
