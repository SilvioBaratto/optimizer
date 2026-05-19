"""T212 ticker suffix resolver (issue #774).

Pure function. Strips Trading 212 venue suffixes (`_l_EQ`, `_p_EQ`, ...) to
recover a base symbol and infer the probable Yahoo Finance exchange suffix.
No I/O, no DB, no logging.
"""

from __future__ import annotations

from dataclasses import dataclass

_VENUE_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_US_EQ", ""),
    ("_pp_EQ", ".PA"),
    ("_l_EQ", ".L"),
    ("_p_EQ", ".PA"),
    ("_d_EQ", ".DE"),
    ("_m_EQ", ".MI"),
)
_FALLBACK_SUFFIX = "_EQ"


@dataclass(frozen=True)
class ResolvedSymbol:
    """Result of parsing a raw T212 ticker."""

    base_symbol: str
    t212_suffix: str
    probable_yf_suffix: str
    is_recognized: bool


def resolve_t212_suffix(t212_ticker: str) -> ResolvedSymbol:
    """Strip the T212 venue suffix and infer the Yahoo Finance suffix."""
    for suffix, yf_suffix in _VENUE_SUFFIXES:
        if t212_ticker.endswith(suffix):
            return ResolvedSymbol(
                base_symbol=t212_ticker[: -len(suffix)],
                t212_suffix=suffix,
                probable_yf_suffix=yf_suffix,
                is_recognized=True,
            )

    if t212_ticker.endswith(_FALLBACK_SUFFIX):
        return ResolvedSymbol(
            base_symbol=t212_ticker[: -len(_FALLBACK_SUFFIX)],
            t212_suffix=_FALLBACK_SUFFIX,
            probable_yf_suffix="",
            is_recognized=False,
        )

    return ResolvedSymbol(
        base_symbol=t212_ticker,
        t212_suffix="",
        probable_yf_suffix="",
        is_recognized=False,
    )
