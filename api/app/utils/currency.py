"""Currency utilities shared between API and CLI layers."""

from __future__ import annotations

# Maps minor-unit ISO / yfinance currency codes to their major-unit equivalent.
MINOR_TO_MAJOR: dict[str, str] = {
    "GBX": "GBP",
    "GBp": "GBP",
    "ILA": "ILS",
    "ZAC": "ZAR",
}


def to_major_currency(code: str | None) -> str | None:
    """Convert a minor-unit currency code to its major-unit equivalent.

    Returns the code unchanged if it is already a major-unit code or ``None``.

    >>> to_major_currency("GBX")
    'GBP'
    >>> to_major_currency("USD")
    'USD'
    >>> to_major_currency(None) is None
    True
    """
    if code is None:
        return None
    return MINOR_TO_MAJOR.get(code, code)
