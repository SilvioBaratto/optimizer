"""Trust rules for ISINs that yfinance scrapes from Business Insider.

``yf.Ticker.isin`` is experimental. It searches Business Insider by company
name and keeps the hit whose ticker equals the Yahoo symbol. Business Insider
tickers carry no Yahoo exchange suffix, so a suffixed symbol (``ASML.AS``,
``RY.TO``) never matches and yfinance falls back to the first hit whose ticker
is *empty*. Such an entry is a secondary line by construction: a depositary
receipt (CEDEAR, CDR), a preferred share or a bond of the same issuer. Its
code is well formed, passes the check digit and often carries the listing's own
country prefix, so no checksum or prefix rule can tell it from the right one.
"""

from __future__ import annotations

import re

_ISIN_FORMAT = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_LUHN_DOUBLED_DIGIT_MAX = 9


def has_valid_checksum(isin: str) -> bool:
    """Check the ISIN check digit (ISO 6166, Luhn over the expanded code).

    Args:
        isin: Twelve-character ISIN in upper case.

    Returns:
        True when the Luhn sum of the letter-expanded digits is divisible by 10.
    """
    digits = "".join(str(int(char, 36)) for char in isin)
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > _LUHN_DOUBLED_DIGIT_MAX:
                value -= _LUHN_DOUBLED_DIGIT_MAX
        total += value
    return total % 10 == 0


def has_exchange_suffix(symbol: str) -> bool:
    """Report whether *symbol* is a Yahoo listing outside the US exchanges.

    Yahoo marks every non-US venue with a suffix (``.AS``, ``.T``, ``.TO``) and
    writes share classes with a dash (``BRK-B``), so any dot is a venue suffix.
    """
    return "." in symbol


def is_trusted_isin(isin: str, symbol: str) -> bool:
    """Report whether a scraped ISIN may be stored as *symbol*'s identifier.

    Only a suffix-less symbol is trusted, because Business Insider matches those
    by ticker; its code must also be well formed and pass its check digit. A
    suffixed symbol is never trusted: the scrape cannot be tied to that venue,
    and a missing ISIN is safer than the ISIN of another security. A suffix-less
    symbol that Business Insider lacks takes the same name-search fallback, so
    for those the checks above are the only safeguard.

    Args:
        isin: Candidate ISIN as returned by yfinance.
        symbol: Yahoo ticker symbol the ISIN was requested for.

    Returns:
        True when the ISIN may be stored for the symbol.
    """
    if has_exchange_suffix(symbol):
        return False
    return _ISIN_FORMAT.fullmatch(isin) is not None and has_valid_checksum(isin)
