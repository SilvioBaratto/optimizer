"""Cycle 3 protocol signature regression for YFinanceClientProtocol.

Validates that ``fetch_history`` exposes the eight price-quality kwargs and
that ``bulk_download`` flips ``auto_adjust`` default to ``True`` and gains
``max_retries`` to align with the facade signature.

Issue: #597
"""

from __future__ import annotations

import inspect

from app.services.yfinance.protocols.interfaces import YFinanceClientProtocol

_HISTORY_PRICE_QUALITY_DEFAULTS: dict[str, object] = {
    "auto_adjust": True,
    "back_adjust": False,
    "repair": True,
    "actions": False,
    "prepost": False,
    "keepna": False,
    "rounding": False,
    "timeout": None,
}


def test_when_inspecting_fetch_history_then_price_quality_kwargs_present() -> None:
    sig = inspect.signature(YFinanceClientProtocol.fetch_history)
    for name, expected_default in _HISTORY_PRICE_QUALITY_DEFAULTS.items():
        assert name in sig.parameters, f"fetch_history missing kwarg {name!r}"
        assert sig.parameters[name].default == expected_default, (
            f"fetch_history.{name} default expected {expected_default!r}, "
            f"got {sig.parameters[name].default!r}"
        )


def test_when_inspecting_bulk_download_then_auto_adjust_default_true() -> None:
    sig = inspect.signature(YFinanceClientProtocol.bulk_download)
    assert sig.parameters["auto_adjust"].default is True


def test_when_inspecting_bulk_download_then_max_retries_present() -> None:
    sig = inspect.signature(YFinanceClientProtocol.bulk_download)
    assert "max_retries" in sig.parameters
    assert sig.parameters["max_retries"].default is None


def test_when_inspecting_fetch_history_then_kwargs_are_keyword_only() -> None:
    sig = inspect.signature(YFinanceClientProtocol.fetch_history)
    for name in _HISTORY_PRICE_QUALITY_DEFAULTS:
        param = sig.parameters[name]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"fetch_history.{name} must be keyword-only, got {param.kind}"
        )
