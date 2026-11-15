"""The text-safety helpers under the ``fund.tui.safe_text`` import path.

The helpers live in :mod:`fund.text_safety`, which the Ask agent can import without
importing the cockpit; this module re-exports them for the cockpit modules and tests
that import them from here. At import time it needs only the standard library and
that module.
"""

from __future__ import annotations

from fund.text_safety import (
    MAX_MESSAGE_CHARS,
    clip,
    defang_markdown,
    describe_error,
    strip_control,
)

__all__ = [
    "MAX_MESSAGE_CHARS",
    "clip",
    "defang_markdown",
    "describe_error",
    "strip_control",
]
