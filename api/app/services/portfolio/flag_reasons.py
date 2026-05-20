"""Single source of truth for per-`PositionFlag` human reason copy (issue #793).

Used by the normalizer (#795), drift_service (#796), and any other emission
site to construct a `FlagInstance` with consistent UI copy. Pure module — no
I/O, no service dependencies.
"""

from __future__ import annotations

from app.schemas.portfolio import FlagInstance, PositionFlag

FLAG_REASONS: dict[PositionFlag, str] = {
    PositionFlag.UNMAPPED: (
        "T212 ticker `{reference}` has no yfinance mapping; "
        "position cannot be drift-compared."
    ),
    PositionFlag.FX_MISSING: (
        "FX rate for `{reference}`→EUR unavailable; position valued at 0 EUR."
    ),
    PositionFlag.STALE_PRICE: (
        "Last price tick for `{reference}` exceeds the freshness threshold."
    ),
    PositionFlag.RECONCILIATION_MISMATCH: (
        "Normalized EUR sum diverges from T212 invested beyond ±1.5%."
    ),
    PositionFlag.TARGET_NOT_ON_BROKER: (
        "Target ticker `{reference}` is not in the T212 instrument universe; "
        "cannot buy via this broker."
    ),
}


def make_flag_instance(code: PositionFlag, **ctx: object) -> FlagInstance:
    """Build a `FlagInstance` from `code` and optional template kwargs.

    Raises `KeyError` if `code` has no `FLAG_REASONS` entry — guards against
    `PositionFlag` enum additions shipping without matching copy.
    """
    template = FLAG_REASONS[code]
    reason = template.format(**ctx) if ctx else template
    reference = ctx.get("reference")
    reference_str = str(reference) if reference is not None else None
    return FlagInstance(code=code, reason=reason, reference=reference_str)
