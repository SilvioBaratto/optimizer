"""The leveraged / inverse name screen of the K&E ``no_leverage`` restriction.

The instrument table carries no leverage flag, so a product is judged by its name.
The screen errs towards excluding: a restricted client losing a plain fund is cheaper
than holding a leveraged one.
"""

from __future__ import annotations

import pytest

from fund.tools.mandate import _is_leveraged


@pytest.mark.parametrize(
    "name",
    [
        "Lyxor Daily ShortDAX x2 UCITS ETF",
        "Lyxor Daily LevDAX UCITS ETF",
        "ProShares Short S&P500",
        "Xtrackers DAX Daily Short UCITS ETF",
        "Direxion Daily S&P 500 Bear 3X Shares",
        "ProShares UltraPro QQQ",
        "ProShares UltraShort Dow30",
        "WisdomTree NASDAQ 100 3x Daily Short",
        "WisdomTree Brent Crude Oil 2x Daily Leveraged",
        "Leverage Shares 3x Tesla ETP",
        "Boost FTSE 100 -2x Inverse",
        "Short QQQ Daily ETF",
    ],
)
def test_leveraged_and_inverse_funds_are_recognised(name: str) -> None:
    """Every common naming style for a geared or inverse fund is caught."""
    assert _is_leveraged("ETF", name) is True


@pytest.mark.parametrize(
    "name",
    [
        "iShares Core MSCI World UCITS ETF",
        "SPDR S&P 500 ETF Trust",
        "Vanguard Short-Term Bond ETF",
        "iShares Short Treasury Bond ETF",
        "iShares Level Income Fund",
        "Xtrackers MSCI World Swap UCITS ETF 1C",
        "Invesco Short Term Corporate Bond ETF",
    ],
)
def test_plain_funds_are_not_flagged(name: str) -> None:
    """A fund that merely contains a near-miss word is left alone."""
    assert _is_leveraged("ETF", name) is False


def test_common_stock_is_never_judged_by_its_name() -> None:
    """A company called "Ultra Bull" is not a leveraged product."""
    assert _is_leveraged("EQUITY", "Lev Short Ultra Bull Holdings") is False
