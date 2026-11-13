"""Whole result, progress and abort of ``run_t212_annotate``.

Validates ``app/services/universe/t212_annotate.py`` (``run_t212_annotate``): the
exact summary of a mixed run, key order included, the exact progress reports,
where the batch commits fall between them, which catalogue entries count, the
exact abort message after an ISIN outage and the exact skipped result.
``test_t212_annotate.py`` checks the matching rules one by one. No requirement
id is registered for the Trading 212 overlay yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from app.services.universe import t212_annotate
from app.services.universe.universe_build_service import Trading212NotConfiguredError

_M = "app.services.universe.t212_annotate"

_SUMMARY_KEYS = [
    "annotated",
    "total",
    "skipped",
    "lookup_errors",
    "cleared",
    "pinned_skipped",
    "tradeable",
    "untradeable",
    "isin_stored",
]


def _inst(ticker: str, **fields: Any) -> SimpleNamespace:
    row = {
        "ticker": ticker,
        "exchange_id": 1,
        "isin": None,
        "currency_code": None,
        "t212_ticker": None,
        "yfinance_ticker": ticker,
        "is_t212_tradeable": None,
    }
    row.update(fields)
    return SimpleNamespace(**row)


@contextmanager
def _patched(
    instruments: list[SimpleNamespace],
    catalogue: list[dict[str, Any]],
    events: list[str],
    manual_pins: frozenset[str] = frozenset(),
) -> Iterator[MagicMock]:
    """Patch the step's collaborators and log each session open, commit and close."""
    repo = MagicMock(name="universe_repo")
    repo.get_active_instruments.return_value = instruments
    repo.set_t212_ticker.return_value = True
    repo.set_isin_if_missing.return_value = True
    pins = MagicMock(name="pin_repo")
    pins.pinned_tickers.return_value = set(manual_pins)
    client = MagicMock(name="t212_client")
    client.get_instruments.return_value = catalogue

    @contextmanager
    def session() -> Iterator[MagicMock]:
        """Yield a session whose commit is logged between its open and close."""
        events.append("open")
        handle = MagicMock(name="session")
        handle.commit.side_effect = lambda: events.append("commit")
        try:
            yield handle
        finally:
            events.append("close")

    manager = MagicMock(name="database_manager")
    manager.get_session = session
    with (
        patch(f"{_M}.database_manager", manager),
        patch(f"{_M}.UniverseRepository", return_value=repo),
        patch(f"{_M}.PinnedInstrumentRepository", return_value=pins),
        patch(f"{_M}.build_trading212_client", return_value=client),
    ):
        yield repo


_CATALOGUE = [
    {"ticker": "AAA_US_EQ", "isin": "US1", "currencyCode": "USD"},
    {"ticker": "BBB_EQ", "isin": "GB1", "currencyCode": "GBX"},
    {"isin": "NOTICKER1"},
    {"ticker": "NOISIN_EQ", "currencyCode": "USD"},
]


def _mixed_universe() -> list[SimpleNamespace]:
    return [
        _inst("AAA", isin="US1"),
        _inst("BBB"),
        _inst("CCC"),
        _inst("DDD"),
        _inst("EEE", isin="XX9", t212_ticker="OLD_EQ", is_t212_tradeable=True),
        _inst("FFF", isin="US1", t212_ticker="HAND_EQ"),
        _inst("GGG", isin="NOTICKER1"),
    ]


def _lookup(ticker: str) -> str | None:
    if ticker == "CCC":
        raise t212_annotate.IsinLookupError("throttled")
    return {"BBB": "GB1"}.get(ticker)


class TestMixedRun:
    """One run over every kind of instrument the step distinguishes."""

    def test_the_summary_is_exactly(self) -> None:
        """Pin every count of the summary and the order of its keys."""
        on_progress = MagicMock()
        with _patched(_mixed_universe(), _CATALOGUE, [], frozenset({"FFF"})):
            summary = t212_annotate.run_t212_annotate(
                isin_lookup=_lookup, on_progress=on_progress
            )

        assert summary == {
            "annotated": 2,
            "total": 7,
            "skipped": False,
            "lookup_errors": 1,
            "cleared": 1,
            "pinned_skipped": 1,
            "tradeable": 3,
            "untradeable": 2,
            "isin_stored": 1,
        }
        assert list(summary) == _SUMMARY_KEYS

    def test_progress_reports_each_instrument_then_the_summary(self) -> None:
        """One report per instrument in universe order, then the completed one."""
        on_progress = MagicMock()
        with _patched(_mixed_universe(), _CATALOGUE, [], frozenset({"FFF"})):
            summary = t212_annotate.run_t212_annotate(
                isin_lookup=_lookup, on_progress=on_progress
            )

        tickers = [i.ticker for i in _mixed_universe()]
        assert on_progress.call_args_list == [
            *(
                call(current=i, total=7, current_stock=ticker)
                for i, ticker in enumerate(tickers, start=1)
            ),
            call(status="completed", result=summary),
        ]

    def test_the_writes_follow_the_matching_rules(self) -> None:
        """Pin each write the mixed run makes, in order."""
        with _patched(_mixed_universe(), _CATALOGUE, [], frozenset({"FFF"})) as repo:
            t212_annotate.run_t212_annotate(isin_lookup=_lookup)

        assert repo.set_t212_ticker.call_args_list == [
            call(ticker="AAA", exchange_id=1, t212_ticker="AAA_US_EQ"),
            call(ticker="BBB", exchange_id=1, t212_ticker="BBB_EQ"),
            call(ticker="EEE", exchange_id=1, t212_ticker=None),
        ]
        assert repo.set_isin_if_missing.call_args_list == [
            call(ticker="BBB", exchange_id=1, isin="GB1")
        ]
        assert repo.set_t212_tradeable.call_args_list == [
            call(ticker="AAA", exchange_id=1, tradeable=True),
            call(ticker="BBB", exchange_id=1, tradeable=True),
            call(ticker="EEE", exchange_id=1, tradeable=False),
            call(ticker="FFF", exchange_id=1, tradeable=True),
            call(ticker="GGG", exchange_id=1, tradeable=False),
        ]


class TestBatchCommits:
    """A batch is written after its last lookup and before its progress report."""

    def test_sessions_commits_lookups_and_reports_interleave_exactly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the event order for batches of two over three instruments."""
        monkeypatch.setattr(t212_annotate, "_COMMIT_EVERY", 2)
        events: list[str] = []

        def lookup(ticker: str) -> str:
            """Log the lookup and answer the one ISIN the catalogue lists."""
            events.append(f"lookup:{ticker}")
            return "US1"

        def on_progress(**report: Any) -> None:
            """Log the position, or the status of the final report."""
            events.append(f"progress:{report.get('current', report.get('status'))}")

        universe = [_inst(t) for t in ("T0", "T1", "T2")]
        with _patched(universe, _CATALOGUE[:1], events):
            t212_annotate.run_t212_annotate(isin_lookup=lookup, on_progress=on_progress)

        assert events == [
            "open",
            "close",
            "lookup:T0",
            "progress:1",
            "lookup:T1",
            "open",
            "commit",
            "close",
            "progress:2",
            "lookup:T2",
            "progress:3",
            "open",
            "commit",
            "close",
            "progress:completed",
        ]


class TestOutage:
    """Too many failed lookups in a row stop the run with what it wrote kept."""

    def test_the_abort_message_and_the_last_report_are_exactly(self) -> None:
        """The failing instrument is not reported; the earlier writes are kept."""
        limit = t212_annotate._MAX_CONSECUTIVE_LOOKUP_ERRORS
        universe = [_inst("T0", isin="US1")] + [
            _inst(f"T{i}") for i in range(1, limit + 5)
        ]
        on_progress = MagicMock()
        events: list[str] = []

        def lookup(ticker: str) -> str:
            """Fail every lookup, as an unreachable ISIN source does."""
            raise t212_annotate.IsinLookupError(f"{ticker} throttled")

        with (
            _patched(universe, _CATALOGUE, events) as repo,
            pytest.raises(t212_annotate.IsinLookupError) as raised,
        ):
            t212_annotate.run_t212_annotate(isin_lookup=lookup, on_progress=on_progress)

        assert str(raised.value) == (
            f"{limit} consecutive ISIN lookups failed; stopped after annotating "
            f"1/{len(universe)} instruments"
        )
        assert on_progress.call_args_list == [
            call(current=i, total=len(universe), current_stock=f"T{i - 1}")
            for i in range(1, limit + 1)
        ]
        assert repo.set_t212_ticker.call_args_list == [
            call(ticker="T0", exchange_id=1, t212_ticker="AAA_US_EQ")
        ]
        assert events[-3:] == ["open", "commit", "close"]


class TestSkipped:
    """Without Trading 212 the step reports a skip and nothing else."""

    def test_the_skipped_result_is_exactly(self) -> None:
        """Three keys only, and one skipped report."""
        on_progress = MagicMock()
        with patch(
            f"{_M}.build_trading212_client",
            side_effect=Trading212NotConfiguredError("no key"),
        ):
            summary = t212_annotate.run_t212_annotate(on_progress=on_progress)

        assert summary == {"annotated": 0, "total": 0, "skipped": True}
        assert list(summary) == ["annotated", "total", "skipped"]
        assert on_progress.call_args_list == [call(status="skipped")]
