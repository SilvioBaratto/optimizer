"""Currency utilities shared between API and CLI layers."""

from __future__ import annotations

# Every spelling of a minor-unit code seen in the wild, mapped to its major. Yahoo
# emits the sub-unit with a lower-case final letter (``GBp``, ``ZAc``, ``ILa``);
# ``GBX``, ``ZAC`` and ``ILA`` are the upper-cased variants other feeds use. Kept in
# step with ``optimizer.fx._minor_units`` by hand because ingestion must not import
# the optimizer package.
MINOR_TO_MAJOR: dict[str, str] = {
    "GBp": "GBP",
    "GBX": "GBP",
    "GBx": "GBP",
    "ZAc": "ZAR",
    "ZAC": "ZAR",
    "ZAX": "ZAR",
    "ZAx": "ZAR",
    "ILA": "ILS",
    "ILa": "ILS",
}

# Upper-cased fallback so a differently-cased minor code still resolves, except
# those that upper-case onto their own major (``GBp`` -> ``GBP``): matching
# those case-insensitively would read a plain pound as pence.
_MINOR_TO_MAJOR_CI: dict[str, str] = {
    code.upper(): major
    for code, major in MINOR_TO_MAJOR.items()
    if code.upper() != major
}


def minor_unit_major(code: str | None) -> str | None:
    """Return the major-unit code if ``code`` is a minor unit, else ``None``.

    The exact spelling is tried first so ``GBp`` (pence) is distinguished from
    ``GBP`` (pounds); any other minor code also matches case-insensitively.
    """
    if not isinstance(code, str):
        return None
    exact = MINOR_TO_MAJOR.get(code)
    if exact is not None:
        return exact
    return _MINOR_TO_MAJOR_CI.get(code.upper())


def to_major_currency(code: str | None) -> str | None:
    """Convert a minor-unit currency code to its major-unit equivalent.

    Returns the code unchanged if it is already a major-unit code or ``None``.

    >>> to_major_currency("GBX")
    'GBP'
    >>> to_major_currency("ZAc")
    'ZAR'
    >>> to_major_currency("USD")
    'USD'
    >>> to_major_currency(None) is None
    True
    """
    if code is None:
        return None
    return minor_unit_major(code) or code
