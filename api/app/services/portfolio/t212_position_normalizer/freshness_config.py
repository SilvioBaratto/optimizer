"""FreshnessConfig — time-based stale-price threshold (issue #794).

Frozen domain value object wrapping `stale_price_threshold_hours`. Env-var
plumbing lives on the central `Settings` singleton in `app.config`; this
module exposes a `default_freshness_config()` factory that reads the current
settings value at call time (never at import time).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreshnessConfig:
    """Per-environment freshness rules for the position normalizer."""

    stale_price_threshold_hours: int


def default_freshness_config() -> FreshnessConfig:
    """Build a `FreshnessConfig` from the current `Settings` snapshot."""
    from app.config import settings

    return FreshnessConfig(
        stale_price_threshold_hours=settings.stale_price_threshold_hours
    )
