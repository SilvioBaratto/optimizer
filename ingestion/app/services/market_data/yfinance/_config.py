"""Process-wide yfinance logging, applied once by each entry point."""

from __future__ import annotations

import logging

_YFINANCE_LOGGER = "yfinance"

# Remembered so the level can be re-applied after yfinance's price repair has
# overwritten it; None until an entry point calls configure_yfinance.
_configured_level: int | None = None


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
    global _configured_level
    level = getattr(logging, log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {log_level!r}")
    _configured_level = level
    logging.getLogger(_YFINANCE_LOGGER).setLevel(level)


def reapply_yfinance_log_level() -> None:
    """Put the ``yfinance`` logger back at the level chosen by ``configure_yfinance``.

    yfinance's price repair lowers the shared logger to ``CRITICAL`` around a
    nested history call and then restores the level it read beforehand, with
    no lock and no ``finally``. Overlapping repairs in parallel fetch workers
    can leave ``CRITICAL`` behind as the "original" level, and a rate limit
    raised from the nested call skips the restore, so the operator's level is
    lost for the rest of the process. Fetches call this once the library has
    returned. It does nothing when ``configure_yfinance`` was never called.
    """
    if _configured_level is None:
        return
    logger = logging.getLogger(_YFINANCE_LOGGER)
    if logger.level != _configured_level:
        logger.setLevel(_configured_level)
