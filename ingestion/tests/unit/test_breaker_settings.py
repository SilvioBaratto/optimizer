"""External-call breakers, attempts and timeouts come from the daemon's settings.

Covers the ``breaker_*``, ``*_request_attempts``, ``*_request_timeout_seconds`` and
``fred_retry_*`` fields of ``app.config.Settings``; ``build_breaker`` and the
process-wide breakers in ``app.services.infrastructure.external_calls``; the
scrapers reading those settings; ``timed_request``; the
``external_breaker_open`` gauge and its binding in ``app.worker._serve_metrics``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests
from prometheus_client import REGISTRY
from pydantic import ValidationError

from app import metrics
from app import worker as worker_module
from app.config import Settings, settings
from app.services.infrastructure import CircuitBreaker
from app.services.infrastructure.external_calls import (
    ExternalService,
    build_breaker,
    process_breaker,
    timed_request,
)
from app.services.macro.scrapers import (
    fred_scraper,
    ilsole_scraper,
    tradingeconomics_scraper,
)

_ATTEMPT_DEFAULTS = {
    "BREAKER_FRED_MAX_ATTEMPTS": 8,
    "BREAKER_TRADINGECONOMICS_MAX_ATTEMPTS": 5,
    "BREAKER_ILSOLE_MAX_ATTEMPTS": 5,
    "BREAKER_YFINANCE_MAX_ATTEMPTS": 10,
    "FRED_REQUEST_ATTEMPTS": 5,
    "TRADINGECONOMICS_REQUEST_ATTEMPTS": 3,
    "ILSOLE_REQUEST_ATTEMPTS": 3,
}
_SECONDS_DEFAULTS = {
    "FRED_REQUEST_TIMEOUT_SECONDS": 30.0,
    "TRADINGECONOMICS_REQUEST_TIMEOUT_SECONDS": 20.0,
    "ILSOLE_REQUEST_TIMEOUT_SECONDS": 10.0,
    "FRED_RETRY_BASE_DELAY_SECONDS": 1.0,
    "FRED_RETRY_MAX_DELAY_SECONDS": 60.0,
}
_BREAKER_VARS = {
    ExternalService.FRED: "BREAKER_FRED_MAX_ATTEMPTS",
    ExternalService.TRADING_ECONOMICS: "BREAKER_TRADINGECONOMICS_MAX_ATTEMPTS",
    ExternalService.IL_SOLE: "BREAKER_ILSOLE_MAX_ATTEMPTS",
}


@pytest.fixture(autouse=True)
def _unset_external_call_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the tests from the defaults, whatever the shell exports."""
    for name in (*_ATTEMPT_DEFAULTS, *_SECONDS_DEFAULTS):
        monkeypatch.delenv(name, raising=False)


def _field(env_name: str) -> str:
    return env_name.lower()


class TestSettingsFields:
    """Documented defaults, values read from the variables, bad values refused."""

    @pytest.mark.parametrize(
        ("env_name", "default"),
        [*_ATTEMPT_DEFAULTS.items(), *_SECONDS_DEFAULTS.items()],
    )
    def test_an_unset_knob_takes_its_listed_default(
        self, env_name: str, default: float
    ) -> None:
        """Unset, the field equals the default listed for its variable."""
        assert getattr(Settings(_env_file=None), _field(env_name)) == default

    @pytest.mark.parametrize("env_name", list(_ATTEMPT_DEFAULTS))
    def test_an_attempt_count_is_read_from_the_environment(
        self, env_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The variable named after the field sets it."""
        monkeypatch.setenv(env_name, "7")

        assert getattr(Settings(_env_file=None), _field(env_name)) == 7

    @pytest.mark.parametrize("env_name", list(_SECONDS_DEFAULTS))
    def test_a_duration_is_read_from_the_environment(
        self, env_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fractional number of seconds is accepted."""
        monkeypatch.setenv(env_name, "12.5")

        assert getattr(Settings(_env_file=None), _field(env_name)) == 12.5

    @pytest.mark.parametrize("env_name", list(_ATTEMPT_DEFAULTS))
    @pytest.mark.parametrize("value", ["0", "-1", "2.5", "abc", ""])
    def test_an_attempt_count_below_one_or_not_an_integer_is_refused(
        self, env_name: str, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails with an error that names the variable."""
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ValidationError, match=env_name):
            Settings(_env_file=None)

    @pytest.mark.parametrize("env_name", list(_SECONDS_DEFAULTS))
    @pytest.mark.parametrize("value", ["0", "-1", "inf", "nan", "abc"])
    def test_a_duration_that_is_not_finite_and_positive_is_refused(
        self, env_name: str, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails with an error that names the variable."""
        monkeypatch.setenv(env_name, value)

        with pytest.raises(ValidationError, match=env_name):
            Settings(_env_file=None)


class TestBuildBreaker:
    """``build_breaker`` sizes a source's breaker from the settings it is given."""

    @pytest.mark.parametrize(("service", "env_name"), list(_BREAKER_VARS.items()))
    def test_the_setting_sets_the_breaker_max_attempts(
        self, service: ExternalService, env_name: str
    ) -> None:
        """Changing the setting changes the breaker's ``max_attempts``."""
        config = Settings(_env_file=None, **{env_name: 3})

        assert build_breaker(service, config).max_attempts == 3

    @pytest.mark.parametrize(("service", "env_name"), list(_BREAKER_VARS.items()))
    def test_a_breaker_from_default_settings_has_the_listed_max_attempts(
        self, service: ExternalService, env_name: str
    ) -> None:
        """Its ``max_attempts`` equals its variable's entry in ``_ATTEMPT_DEFAULTS``."""
        breaker = build_breaker(service, Settings(_env_file=None))

        assert breaker.max_attempts == _ATTEMPT_DEFAULTS[env_name]

    @pytest.mark.parametrize(
        ("service", "name"),
        [
            (ExternalService.FRED, "FRED API"),
            (ExternalService.TRADING_ECONOMICS, "Trading Economics"),
            (ExternalService.IL_SOLE, "IlSole24Ore"),
        ],
    )
    def test_each_breaker_keeps_its_service_name(
        self, service: ExternalService, name: str
    ) -> None:
        """The name in a refusal message is the one the scrapers used."""
        assert build_breaker(service).service_name == name

    @pytest.mark.parametrize(
        "service",
        [
            ExternalService.FRED,
            ExternalService.TRADING_ECONOMICS,
            ExternalService.IL_SOLE,
        ],
    )
    def test_scraper_breakers_keep_the_shared_back_off_windows(
        self, service: ExternalService
    ) -> None:
        """The attempt count moved to settings; the windows did not change."""
        breaker = build_breaker(service)
        default = CircuitBreaker()

        assert breaker.base_wait_minutes == default.base_wait_minutes
        assert breaker.max_wait_minutes == default.max_wait_minutes


class TestProcessBreakers:
    """One breaker per source per process, shared by the scraper instances."""

    @pytest.mark.parametrize(
        ("module", "attribute", "service"),
        [
            (fred_scraper, "_fred_circuit_breaker", ExternalService.FRED),
            (
                tradingeconomics_scraper,
                "_te_circuit_breaker",
                ExternalService.TRADING_ECONOMICS,
            ),
            (ilsole_scraper, "_ilsole_circuit_breaker", ExternalService.IL_SOLE),
        ],
    )
    def test_each_scraper_uses_the_process_breaker_of_its_source(
        self, module: object, attribute: str, service: ExternalService
    ) -> None:
        """The metrics gauge reads the same object the scraper trips."""
        assert getattr(module, attribute) is process_breaker(service)

    @pytest.mark.parametrize(("service", "env_name"), list(_BREAKER_VARS.items()))
    def test_process_breakers_are_sized_from_the_process_settings(
        self, service: ExternalService, env_name: str
    ) -> None:
        """The process breakers read ``app.config.settings``."""
        expected = getattr(settings, _field(env_name))

        assert process_breaker(service).max_attempts == expected


class TestScrapersReadTheirSettings:
    """The scrapers take their timeout and attempt budget from the settings."""

    def test_fred_timeout_attempts_and_delays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FRED reads its timeout, attempts and both backoff delays."""
        monkeypatch.setattr(settings, "fred_request_timeout_seconds", 12.5)
        monkeypatch.setattr(settings, "fred_request_attempts", 2)
        monkeypatch.setattr(settings, "fred_retry_base_delay_seconds", 0.5)
        monkeypatch.setattr(settings, "fred_retry_max_delay_seconds", 7.0)
        retry = MagicMock(return_value=None)
        monkeypatch.setattr(fred_scraper, "retry_with_backoff", retry)
        scraper = fred_scraper.FredScraper(api_key="test-key")

        scraper.get_series_observations("VIXCLS")

        assert scraper.timeout == 12.5
        assert retry.call_args.kwargs["max_retries"] == 2
        assert retry.call_args.kwargs["base_delay"] == 0.5
        assert retry.call_args.kwargs["max_delay"] == 7.0

    def test_an_explicit_fred_timeout_wins_over_the_setting(self) -> None:
        """A caller that passes a timeout keeps it."""
        assert fred_scraper.FredScraper(api_key="test-key", timeout=4).timeout == 4

    def test_trading_economics_timeout_and_attempts_on_every_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Indicators, bond yields and the country lists read one attempt setting."""
        monkeypatch.setattr(settings, "tradingeconomics_request_timeout_seconds", 12.5)
        monkeypatch.setattr(settings, "tradingeconomics_request_attempts", 2)
        retry = MagicMock(return_value=None)
        monkeypatch.setattr(tradingeconomics_scraper, "retry_with_backoff", retry)
        scraper = tradingeconomics_scraper.TradingEconomicsIndicatorsScraper(
            rate_limit_delay=0.0
        )

        scraper.get_country_indicators("USA", include_bonds=False)
        scraper.get_bond_yields("USA")
        scraper._fetch_page("https://tradingeconomics.com/country-list/x")

        assert scraper.timeout == 12.5
        assert [c.kwargs["max_retries"] for c in retry.call_args_list] == [2, 2, 2]

    def test_il_sole_timeout_and_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Il Sole reads its timeout and attempts."""
        monkeypatch.setattr(settings, "ilsole_request_timeout_seconds", 12.5)
        monkeypatch.setattr(settings, "ilsole_request_attempts", 2)
        retry = MagicMock(return_value=None)
        monkeypatch.setattr(ilsole_scraper, "retry_with_backoff", retry)
        scraper = ilsole_scraper.IlSoleScraper()

        scraper._fetch_page("previsione-economica")

        assert scraper.timeout == 12.5
        assert retry.call_args.kwargs["max_retries"] == 2

    def test_an_explicit_trading_economics_timeout_wins_over_the_setting(
        self,
    ) -> None:
        """A caller that passes a timeout keeps it."""
        scraper = tradingeconomics_scraper.TradingEconomicsIndicatorsScraper(timeout=4)

        assert scraper.timeout == 4

    def test_an_explicit_il_sole_timeout_wins_over_the_setting(self) -> None:
        """A caller that passes a timeout keeps it."""
        assert ilsole_scraper.IlSoleScraper(timeout=4).timeout == 4

    def test_the_yfinance_breaker_reads_its_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Yahoo facade sizes its breaker from ``BREAKER_YFINANCE_MAX_ATTEMPTS``."""
        from app.services.market_data.yfinance._facade import YFinanceClient

        monkeypatch.setattr(settings, "breaker_yfinance_max_attempts", 4)
        YFinanceClient.reset_instance()
        try:
            breaker = YFinanceClient.get_instance().circuit_breaker
        finally:
            YFinanceClient.reset_instance()

        assert breaker.max_attempts == 4


def _request_count(service: str, outcome: str) -> float:
    value = REGISTRY.get_sample_value(
        "external_request_seconds_count", {"service": service, "outcome": outcome}
    )
    return value or 0.0


class TestScrapersTimeTheirRequests:
    """Scraper requests land on ``external_request_seconds``."""

    def test_a_fred_request_is_timed_under_its_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An answered FRED request counts as ``ok``."""
        monkeypatch.setattr(fred_scraper, "_fred_circuit_breaker", MagicMock())
        scraper = fred_scraper.FredScraper(api_key="test-key")
        response = MagicMock(status_code=200)
        response.json.return_value = {"observations": []}
        monkeypatch.setattr(scraper.session, "get", MagicMock(return_value=response))
        before = _request_count("FRED API", "ok")

        scraper.get_series_observations("VIXCLS")

        assert _request_count("FRED API", "ok") == before + 1

    def test_a_failed_trading_economics_request_is_timed_as_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request that raises counts as ``error``."""
        monkeypatch.setattr(
            tradingeconomics_scraper, "_te_circuit_breaker", MagicMock()
        )
        monkeypatch.setattr(settings, "tradingeconomics_request_attempts", 1)
        scraper = tradingeconomics_scraper.TradingEconomicsIndicatorsScraper(
            rate_limit_delay=0.0
        )
        refused = requests.exceptions.ConnectionError("refused")
        monkeypatch.setattr(scraper.session, "get", MagicMock(side_effect=refused))
        before = _request_count("Trading Economics", "error")

        assert scraper._fetch_page("https://tradingeconomics.com/x") is None

        assert _request_count("Trading Economics", "error") == before + 1

    def test_an_il_sole_request_is_timed_under_its_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An answered Il Sole request counts as ``ok``."""
        monkeypatch.setattr(ilsole_scraper, "_ilsole_circuit_breaker", MagicMock())
        monkeypatch.setattr(ilsole_scraper, "_ilsole_rate_limiter", MagicMock())
        scraper = ilsole_scraper.IlSoleScraper()
        response = MagicMock(status_code=200, content=b"<html></html>")
        monkeypatch.setattr(scraper.session, "get", MagicMock(return_value=response))
        before = _request_count("IlSole24Ore", "ok")

        scraper._fetch_page("previsione-economica")

        assert _request_count("IlSole24Ore", "ok") == before + 1


class TestTimedRequest:
    """``timed_request`` records one observation per block, labelled by outcome."""

    def test_a_block_that_completes_is_an_ok_request(self) -> None:
        """No exception means ``ok``."""
        before = _request_count("FRED API", "ok")

        with timed_request(ExternalService.FRED):
            pass

        assert _request_count("FRED API", "ok") == before + 1

    @pytest.mark.criterion("REQ-ING-018")
    def test_a_block_that_raises_is_an_error_and_the_exception_propagates(
        self,
    ) -> None:
        """Validates ``app.services.infrastructure.external_calls.timed_request``
        records a raising block as an ``error`` request and lets the exception
        propagate."""
        before = _request_count("Trading Economics", "error")

        with (
            pytest.raises(requests.exceptions.Timeout),
            timed_request(ExternalService.TRADING_ECONOMICS),
        ):
            raise requests.exceptions.Timeout("read timed out")

        assert _request_count("Trading Economics", "error") == before + 1

    def test_the_duration_is_observed_in_seconds(self) -> None:
        """The observation is the wall time the block took."""
        labels = {"service": "IlSole24Ore", "outcome": "ok"}
        before = REGISTRY.get_sample_value("external_request_seconds_sum", labels)

        with (
            patch(
                "app.services.infrastructure.external_calls.time.perf_counter",
                side_effect=[10.0, 12.5],
            ),
            timed_request(ExternalService.IL_SOLE),
        ):
            pass

        after = REGISTRY.get_sample_value("external_request_seconds_sum", labels)
        assert after == pytest.approx((before or 0.0) + 2.5)


class _Clock:
    """Stand-in for the ``time`` module whose ``sleep`` advances ``time``."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def time(self) -> float:
        """Return the fake current time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time instead of blocking."""
        self.now += seconds


_GAUGE_PROBE = "external-breaker-gauge-probe"


def _gauge(service: str) -> float | None:
    return REGISTRY.get_sample_value("external_breaker_open", {"service": service})


class TestExternalBreakerGauge:
    """``external_breaker_open{service}`` reflects the breaker at scrape time."""

    def test_the_gauge_reads_the_breaker_at_each_scrape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0 while closed or backing off, 1 while refusing, 0 once the window ends."""
        clock = _Clock()
        monkeypatch.setattr("app.services.infrastructure.circuit_breaker.time", clock)
        breaker = CircuitBreaker(service_name="probe", max_attempts=2)
        metrics.track_external_breaker(_GAUGE_PROBE, breaker)

        assert _gauge(_GAUGE_PROBE) == 0
        breaker.trigger()
        assert _gauge(_GAUGE_PROBE) == 0
        clock.now = breaker._until
        breaker.trigger()
        assert _gauge(_GAUGE_PROBE) == 1
        clock.now = breaker._until
        assert _gauge(_GAUGE_PROBE) == 0

    def test_serving_metrics_binds_every_process_breaker(self) -> None:
        """The worker exposes the process-wide breakers."""
        with (
            patch.object(worker_module.settings, "enable_metrics", True),
            patch.object(worker_module, "start_http_server"),
            patch.object(metrics, "track_db_breaker"),
            patch.object(metrics, "track_external_breaker") as track,
        ):
            worker_module._serve_metrics()

        assert track.call_args_list == [
            call(service.value, process_breaker(service)) for service in ExternalService
        ]
