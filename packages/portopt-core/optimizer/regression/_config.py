"""Names and settings shared by the regression tools.

The regression functions take plain arguments, so this module holds what they have
in common: the label of the intercept, the standard Fama-French factor names, and
the checks on the three settings (annualisation, Newey-West lags and rolling
window) that more than one function accepts.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

import numpy as np

from optimizer.exceptions import ConfigurationError

CONSTANT_NAME: Final = "const"


class FamaFrenchFactor(str, Enum):
    """Column names of the factor returns in Kenneth French's data library.

    ``Mkt-RF`` is the market return in excess of the risk-free rate, ``SMB`` size,
    ``HML`` value, ``RMW`` profitability, ``CMA`` investment and ``MOM`` momentum.
    """

    MKT_RF = "Mkt-RF"
    SMB = "SMB"
    HML = "HML"
    RMW = "RMW"
    CMA = "CMA"
    MOM = "MOM"


def _is_whole_number(value: object) -> bool:
    """Return whether ``value`` is an integer and not a bool."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def require_periods_per_year(periods_per_year: int) -> None:
    """Reject an annualisation factor that is not a positive whole number.

    Args:
        periods_per_year: Number of return periods in a year.

    Raises:
        ConfigurationError: If it is not an integer of at least 1.
    """
    if not _is_whole_number(periods_per_year) or periods_per_year < 1:
        msg = f"periods_per_year must be a positive integer, got {periods_per_year!r}"
        raise ConfigurationError(msg)


def require_hac_lags(hac_lags: int | None) -> None:
    """Reject a Newey-West lag count that is not ``None`` or a non-negative integer.

    Args:
        hac_lags: Number of Bartlett-weighted lags, or ``None`` for classical errors.

    Raises:
        ConfigurationError: If it is neither ``None`` nor an integer of at least 0.
    """
    if hac_lags is None:
        return
    if not _is_whole_number(hac_lags) or hac_lags < 0:
        msg = f"hac_lags must be None or a non-negative integer, got {hac_lags!r}"
        raise ConfigurationError(msg)


def require_window(window: int) -> None:
    """Reject a rolling window that cannot hold a covariance.

    Args:
        window: Number of trailing observations in each window.

    Raises:
        ConfigurationError: If it is not an integer of at least 2.
    """
    if not _is_whole_number(window) or window < 2:
        msg = f"window must be an integer of at least 2, got {window!r}"
        raise ConfigurationError(msg)
