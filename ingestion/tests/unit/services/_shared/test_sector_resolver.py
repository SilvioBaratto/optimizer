"""Unit tests for ``_shared/_sector_resolver.resolve_sector_map`` (issue #826).

Session is mocked — the ``TickerProfile`` join is exercised by stubbing
``session.execute(...).all()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from app.services._shared._sector_resolver import UNCLASSIFIED, resolve_sector_map


def _profile_rows(*pairs: tuple[str, str | None]) -> MagicMock:
    session = MagicMock()
    rows = [SimpleNamespace(yfinance_ticker=t, sector=s) for t, s in pairs]
    session.execute.return_value.all.return_value = rows
    return session


class TestResolveSectorMap:
    def test_when_no_tickers_then_empty_dict(self) -> None:
        session = MagicMock()
        assert resolve_sector_map(session, []) == {}
        session.execute.assert_not_called()

    def test_when_snapshot_covers_all_then_profile_not_queried(self) -> None:
        session = MagicMock()
        result = resolve_sector_map(
            session, ["AAPL"], snapshot_mapping={"AAPL": "Technology"}
        )
        assert result == {"AAPL": "Technology"}
        session.execute.assert_not_called()  # snapshot short-circuits the join

    def test_when_unknown_ticker_then_defaults_to_unclassified(self) -> None:
        session = _profile_rows()  # join returns nothing
        result = resolve_sector_map(session, ["GHOST"])
        assert result == {"GHOST": UNCLASSIFIED}

    def test_when_profile_has_sector_then_used_as_fallback(self) -> None:
        session = _profile_rows(("AAPL", "Technology"))
        result = resolve_sector_map(session, ["AAPL"])
        assert result == {"AAPL": "Technology"}

    def test_when_snapshot_entry_empty_then_falls_through_to_profile(self) -> None:
        session = _profile_rows(("AAPL", "Technology"))
        result = resolve_sector_map(session, ["AAPL"], snapshot_mapping={"AAPL": ""})
        assert result == {"AAPL": "Technology"}

    def test_when_result_then_covers_every_input_ticker(self) -> None:
        session = _profile_rows(("AAPL", "Technology"))
        result = resolve_sector_map(
            session,
            ["AAPL", "MSFT", "GHOST"],
            snapshot_mapping={"MSFT": "Software"},
        )
        assert result == {
            "AAPL": "Technology",  # profile
            "MSFT": "Software",  # snapshot
            "GHOST": UNCLASSIFIED,  # default bucket
        }
        assert all(isinstance(v, str) for v in result.values())
