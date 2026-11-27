"""Upstream defects the suite tolerates until a fixed release ships.

Each gap is a strict expected failure: when the dependency is fixed the test
passes, strict mode turns that pass into a failure, and the entry is deleted
here, which is the only reminder the gap needs.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest

TEL_AVIV_EXCHANGE = "Tel Aviv Stock Exchange"

# ``exchange_calendars`` 4.13.3 cannot build its Tel Aviv calendar on pandas 3.1:
# pandas renamed the scalar offset hook the calendar's custom offsets override,
# so it rejects the Sunday special closes as "not sessions". Upstream's fix is an
# open pull request (633, pandas 3.1 support) with no release yet. The failure is
# caught in ``get_expected_trading_sessions``, so for Tel Aviv listings the
# history-sufficiency check is skipped (with a warning) instead of the run
# failing. ``raises`` keeps any other error from hiding behind the marker.
_TEL_AVIV_CALENDAR_GAP = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="exchange_calendars 4.13.3 cannot build XTAE on pandas 3.1",
)


def tolerate_tel_aviv_calendar_gap(exchanges: Iterable[str]) -> list[object]:
    """Wrap the Tel Aviv entry of a parametrised exchange list as an expected failure.

    Args:
        exchanges: Exchange names a test is parametrised over.

    Returns:
        The same names, with the Tel Aviv one carrying the expected-failure
        marker, ready to pass to ``pytest.mark.parametrize``.
    """
    return [
        (
            pytest.param(name, marks=_TEL_AVIV_CALENDAR_GAP)
            if name == TEL_AVIV_EXCHANGE
            else name
        )
        for name in exchanges
    ]
