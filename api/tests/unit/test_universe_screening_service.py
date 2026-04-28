"""Unit tests for universe_screening_service (issue #426).

Covers the service-layer data-assembly boundary:

- ``test_empty_body``: an empty request body must not blow up the pivot path.
- ``test_duplicate_instruments_dont_crash_reshape``: when two
  ``instruments`` rows map to the same ``yfinance_ticker``, the
  ``price_history`` join produces duplicate ``(ticker, date)`` tuples. The
  service must deduplicate before calling ``df.pivot`` instead of surfacing
  pandas's ``Index contains duplicate entries, cannot reshape`` as an HTTP
  422 (which misrepresents a server-side data bug as a validation error).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from app.schemas.universe_screen import UniverseScreenRequest
from app.services.universe_screening_service import (
    _assemble_price_volume,
    run_universe_screen,
)

# ---------------------------------------------------------------------------
# _assemble_price_volume — dedup guard at hydration boundary
# ---------------------------------------------------------------------------


class TestAssemblePriceVolumeDedup:
    """_assemble_price_volume must dedupe duplicate (ticker, date) rows."""

    def test_duplicate_instruments_dont_crash_reshape(self) -> None:
        """Two rows with the same (ticker, date) must not raise on pivot."""
        rows = [
            ("ADM", date(2021, 3, 25), 50.0, 1_000_000),
            ("ADM", date(2021, 3, 25), 50.5, 1_100_000),  # duplicate
            ("ADM", date(2021, 3, 26), 51.0, 1_050_000),
            ("MSFT", date(2021, 3, 25), 230.0, 30_000_000),
        ]
        session = _make_session_returning(rows)

        close_df, volume_df = _assemble_price_volume(session)

        assert not close_df.empty
        assert not volume_df.empty
        # Pivot succeeded: one row per date, one column per ticker
        assert close_df.loc[date(2021, 3, 25), "ADM"] in (50.0, 50.5)
        assert close_df.loc[date(2021, 3, 26), "ADM"] == 51.0
        assert close_df.loc[date(2021, 3, 25), "MSFT"] == 230.0

    def test_empty_rows_returns_empty_frames(self) -> None:
        session = _make_session_returning([])

        close_df, volume_df = _assemble_price_volume(session)

        assert close_df.empty
        assert volume_df.empty

    def test_unique_rows_pivot_unchanged(self) -> None:
        """Regression guard: dedup does not drop unique rows."""
        rows = [
            ("AAPL", date(2021, 1, 4), 133.0, 10_000_000),
            ("AAPL", date(2021, 1, 5), 131.5, 11_000_000),
            ("MSFT", date(2021, 1, 4), 218.0, 20_000_000),
            ("MSFT", date(2021, 1, 5), 220.0, 21_000_000),
        ]
        session = _make_session_returning(rows)

        close_df, _ = _assemble_price_volume(session)

        assert close_df.shape == (2, 2)
        assert close_df.loc[date(2021, 1, 4), "AAPL"] == 133.0
        assert close_df.loc[date(2021, 1, 5), "MSFT"] == 220.0


# ---------------------------------------------------------------------------
# run_universe_screen — empty body should not crash
# ---------------------------------------------------------------------------


class TestRunUniverseScreenEmptyBody:
    """Empty body (``{}``) must reach a real response, not a 500-class error."""

    def test_empty_body(self) -> None:
        """Empty request + empty DB → 200 with empty passing_tickers."""
        session = _make_session_returning([])  # no fundamentals, no prices

        result = run_universe_screen(UniverseScreenRequest(), session)

        assert result == {
            "passing_tickers": [],
            "total_screened": 0,
            "diagnostics": {},
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_returning(rows: list[tuple]) -> MagicMock:
    """Return a MagicMock Session whose ``.execute(stmt).all()`` yields *rows* and ``.scalars().all()`` yields []."""
    session = MagicMock()
    exec_result = MagicMock()
    exec_result.all.return_value = rows
    exec_result.scalars.return_value.all.return_value = []
    session.execute.return_value = exec_result
    return session
