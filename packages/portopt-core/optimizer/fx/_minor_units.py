"""Minor-unit (sub-currency) normalization for mixed-scale price data.

Some listings are quoted in a currency's **minor** unit rather than its major
unit, and market-data providers store the price *as quoted* together with a
sub-unit currency code.  yfinance (the ingestion source) is explicit about
this: as of 1.6.0 its price-repair path deliberately keeps GBp / ZAc / ILA
prices in their quoted sub-unit rather than folding them into the main
currency, and ``price_history.price_unit`` stores that code verbatim.

Known sub-units (all are 1/100 of the major unit):

===========  ==================  ==============  ============
Minor code   Major currency      Units / major   Market
===========  ==================  ==============  ============
``GBp``      GBP (pound)         100 (pence)     London (LSE)
``GBX``      GBP (pound)         100 (pence)     London (synonym)
``ZAc``      ZAR (rand)          100 (cents)     Johannesburg (JSE)
``ILA``      ILS (shekel)        100 (agorot)    Tel Aviv (TASE)
===========  ==================  ==============  ============

A naive FX conversion keyed on the currency code alone treats ``GBp`` prices
(pence) as if they were GBP (pounds) and applies the GBP FX rate directly — a
**100x error**.  :func:`normalize_currency_code` maps a (possibly minor) code
to its ``(major_code, minor_units_per_major)`` pair so that a reader can divide
prices by the scale *before* applying an FX rate quoted in the major unit.

Case handling: the pence code ``GBp`` upper-cases to ``GBP`` (the pound code),
so the two are distinguished **case-sensitively**.  Every other minor code
upper-cases to a token that does not collide with a real currency, so those are
matched case-insensitively for robustness against caller casing.
"""

from __future__ import annotations

# (minor codes as emitted / commonly seen, major ISO code, minor units per major)
_MINOR_UNIT_DEFS: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (("GBp", "GBX", "GBx"), "GBP", 100),  # pence — London Stock Exchange
    (("ZAc", "ZAX", "ZAx"), "ZAR", 100),  # cents — Johannesburg
    (("ILA", "ILa"), "ILS", 100),  # agorot — Tel Aviv
)

# Public, verbatim registry: exact code (as stored) -> (major code, scale).
MINOR_UNIT_SCALES: dict[str, tuple[str, int]] = {
    code: (major, scale) for codes, major, scale in _MINOR_UNIT_DEFS for code in codes
}

# Case-insensitive lookup keyed on the upper-cased minor code, EXCLUDING any
# whose upper-cased form collides with its own major code (i.e. ``GBp`` ->
# ``GBP``).  Those must be matched case-sensitively via ``MINOR_UNIT_SCALES``
# so that a plain ``GBP`` (pounds) is never mis-read as pence.
_MINOR_UNIT_SCALES_CI: dict[str, tuple[str, int]] = {
    code.upper(): (major, scale)
    for codes, major, scale in _MINOR_UNIT_DEFS
    for code in codes
    if code.upper() != major
}


def normalize_currency_code(code: str) -> tuple[str, int]:
    """Resolve a (possibly minor-unit) currency code to major code + scale.

    Parameters
    ----------
    code : str
        A currency / price-unit code as stored in the DB, e.g. ``"USD"``,
        ``"GBP"``, ``"GBp"`` (pence), ``"ZAc"`` (cents), ``"ILA"`` (agorot).

    Returns
    -------
    tuple[str, int]
        ``(major_code, minor_units_per_major)``.  ``major_code`` is the
        upper-cased ISO code of the *major* unit; the scale is the number of
        minor units in one major unit (``100`` for pence/cents/agorot, ``1``
        for a code that is already a major unit).  Dividing a quoted price by
        the scale expresses it in the major unit.

    Examples
    --------
    >>> normalize_currency_code("GBp")
    ('GBP', 100)
    >>> normalize_currency_code("GBP")
    ('GBP', 1)
    >>> normalize_currency_code("ZAc")
    ('ZAR', 100)
    >>> normalize_currency_code("usd")
    ('USD', 1)
    """
    raw = str(code).strip()
    # Case-sensitive exact match first (handles GBp/GBP ambiguity).
    if raw in MINOR_UNIT_SCALES:
        return MINOR_UNIT_SCALES[raw]
    upper = raw.upper()
    if upper in _MINOR_UNIT_SCALES_CI:
        return _MINOR_UNIT_SCALES_CI[upper]
    return upper, 1
