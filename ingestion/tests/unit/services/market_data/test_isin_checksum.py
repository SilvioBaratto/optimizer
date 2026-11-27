"""Unit tests for the ISO 6166 check digit that ``add-instrument --isin`` relies on.

Targets: app/services/market_data/yfinance/ticker/isin_checksum.py. The function
is pure, so no mock is involved.
"""

from __future__ import annotations

import pytest

from app.services.market_data.yfinance.ticker.isin_checksum import has_valid_checksum

# Codes of real, widely listed securities: US, Dutch, Irish and Canadian prefixes,
# and a code whose body holds letters, which the check digit expands to numbers.
_VALID_ISINS = (
    "US0378331005",
    "US5949181045",
    "NL0010273215",
    "IE00B3XXRP09",
    "CA7800871021",
    "USN070592100",
)


@pytest.mark.criterion("REQ-ING-020")
@pytest.mark.parametrize("isin", _VALID_ISINS)
def test_has_valid_checksum_accepts_known_isin(isin: str) -> None:
    """Validates ``has_valid_checksum``: a correct ISO 6166 check digit passes."""
    assert has_valid_checksum(isin) is True


@pytest.mark.criterion("REQ-ING-020")
@pytest.mark.parametrize(
    "isin",
    [
        "US0378331006",
        "US0378331004",
        "NL0010273216",
        "IE00B3XXRP08",
        "US0378331050",
    ],
    ids=["off-by-one-up", "off-by-one-down", "nl-typo", "ie-typo", "transposition"],
)
def test_has_valid_checksum_rejects_a_wrong_check_digit(isin: str) -> None:
    """Validates ``has_valid_checksum``: a wrong check digit or a typo fails."""
    assert has_valid_checksum(isin) is False
