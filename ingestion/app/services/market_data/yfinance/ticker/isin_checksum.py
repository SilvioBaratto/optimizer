"""Check digit of an International Securities Identification Number (ISIN)."""

from __future__ import annotations

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
