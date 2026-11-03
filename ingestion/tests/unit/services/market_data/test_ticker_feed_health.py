"""Behaviour of the gate that stops probing a ticker news feed that keeps failing."""

from __future__ import annotations

import logging
import threading

import pytest

from app.services.market_data.yfinance.news.feed_health import TickerFeedHealth

LOGGER = "app.services.market_data.yfinance.news.feed_health"


class _Clock:
    """Hand-advanced monotonic clock so cooldowns need no real waiting."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward by *seconds* without waiting."""
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    """A fresh fake clock for one test."""
    return _Clock()


@pytest.fixture
def gate(clock: _Clock) -> TickerFeedHealth:
    """A gate with small numbers: opens after 3 failures, 100 s cooldown, 10 s probe."""
    return TickerFeedHealth(
        failure_threshold=3,
        cooldown_seconds=100.0,
        probe_window_seconds=10.0,
        clock=clock,
    )


def _fail(gate: TickerFeedHealth, times: int) -> None:
    for _ in range(times):
        gate.record_failure()


class TestClosedGate:
    """A feed that answers, or has failed only a little, keeps being used."""

    def test_new_gate_allows_requests(self, gate: TickerFeedHealth) -> None:
        """Nothing is known about a feed yet, so it is tried."""
        assert gate.allows_request() is True

    def test_failures_below_threshold_keep_it_closed(
        self, gate: TickerFeedHealth
    ) -> None:
        """A couple of failures in a row are not yet a dead feed."""
        _fail(gate, 2)

        assert gate.allows_request() is True

    def test_success_resets_the_consecutive_failure_count(
        self, gate: TickerFeedHealth
    ) -> None:
        """Only failures in a row count, so a good answer in between starts over."""
        _fail(gate, 2)
        gate.record_success()
        _fail(gate, 2)

        assert gate.allows_request() is True


class TestOpenGate:
    """A feed that keeps failing is skipped until its cooldown has passed."""

    def test_threshold_failures_in_a_row_open_the_gate(
        self, gate: TickerFeedHealth
    ) -> None:
        """Once the threshold is reached no further request is allowed."""
        _fail(gate, 3)

        assert gate.allows_request() is False

    def test_stays_open_for_the_whole_cooldown(
        self, gate: TickerFeedHealth, clock: _Clock
    ) -> None:
        """Callers keep being turned away until the cooldown has elapsed."""
        _fail(gate, 3)
        clock.advance(99.0)

        assert gate.allows_request() is False

    def test_opening_logs_one_warning_not_one_per_failure(
        self, gate: TickerFeedHealth, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An operator needs one line saying the feed is being skipped."""
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            _fail(gate, 10)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "fall" in warnings[0].getMessage().lower()


class TestHalfOpenProbe:
    """After the cooldown a single caller tests the feed on everyone's behalf."""

    def test_one_caller_is_admitted_as_probe_after_the_cooldown(
        self, gate: TickerFeedHealth, clock: _Clock
    ) -> None:
        """Four sweep workers must not all hit a dead endpoint at once."""
        _fail(gate, 3)
        clock.advance(100.0)

        assert gate.allows_request() is True
        assert gate.allows_request() is False

    def test_successful_probe_closes_the_gate_and_logs_recovery(
        self,
        gate: TickerFeedHealth,
        clock: _Clock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A feed that answers again is trusted again by every caller."""
        _fail(gate, 3)
        clock.advance(100.0)
        gate.allows_request()

        with caplog.at_level(logging.INFO, logger=LOGGER):
            gate.record_success()

        assert gate.allows_request() is True
        assert gate.allows_request() is True
        assert any("recover" in r.getMessage().lower() for r in caplog.records)

    def test_failed_probe_reopens_for_a_full_cooldown_without_a_new_warning(
        self,
        gate: TickerFeedHealth,
        clock: _Clock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A probe against a still-dead feed costs one request, then silence."""
        _fail(gate, 3)
        clock.advance(100.0)
        gate.allows_request()
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            gate.record_failure()

        assert gate.allows_request() is False
        clock.advance(99.0)
        assert gate.allows_request() is False
        clock.advance(1.0)
        assert gate.allows_request() is True
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    def test_a_probe_that_reports_nothing_is_replaced_after_the_probe_window(
        self, gate: TickerFeedHealth, clock: _Clock
    ) -> None:
        """A probe that died on an unrelated error must not wedge the gate shut."""
        _fail(gate, 3)
        clock.advance(100.0)
        assert gate.allows_request() is True

        clock.advance(9.0)
        assert gate.allows_request() is False
        clock.advance(1.0)
        assert gate.allows_request() is True


class TestConcurrency:
    """The bulk fetch calls the gate from every worker thread."""

    def test_concurrent_failures_open_the_gate_and_warn_exactly_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Racing workers must not each announce that the feed went down."""
        gate = TickerFeedHealth(failure_threshold=5)
        barrier = threading.Barrier(8)

        def worker() -> None:
            """Wait for every thread, then hammer the gate with failing requests."""
            barrier.wait()
            for _ in range(20):
                gate.allows_request()
                gate.record_failure()

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert gate.allows_request() is False
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
