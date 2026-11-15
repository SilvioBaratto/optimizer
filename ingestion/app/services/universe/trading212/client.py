import base64
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from app.config import settings
from app.services.infrastructure import CircuitBreaker, retry_with_backoff
from app.services.infrastructure.external_calls import (
    TRADING_212_MAX_WAIT_SECONDS,
    ExternalService,
    build_breaker,
    process_breaker,
    timed_request,
)

# Trading 212's metadata endpoints sit behind Cloudflare, which 403s the default
# python-requests User-Agent. A browser UA (paired with an explicit JSON Accept)
# clears the challenge.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# T212 reports the end of a rate-limit window in ``x-ratelimit-reset`` (epoch
# seconds) and sends no Retry-After. The window on /metadata/instruments is
# about 50 s, so a fixed 2+4+8+16 s backoff gives up before it reopens; waiting
# for the reported reset is capped so a bogus header cannot stall the daemon.
_MAX_RATE_LIMIT_WAIT_SECONDS = TRADING_212_MAX_WAIT_SECONDS
_RESET_MARGIN_SECONDS = 1

# Answers worth repeating: the request was fine, T212 (or Cloudflare in front of
# it) was not ready for it. Another 4xx is an answer a retry cannot change.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})
_RATE_LIMITED = 429
_FIRST_SERVER_ERROR = 500


def _status_of(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _trips_breaker(exc: Exception) -> bool:
    """Whether ``exc`` shows Trading 212, or the way to it, failing for now.

    True for a 408, 429 or 5xx answer and for a ``requests`` failure that is not
    an HTTP error (connection, timeout, undecodable body); False for another
    HTTP error (a different 4xx, or one without a response) and for anything
    raised outside ``requests``, such as :class:`CircuitOpenError`.
    """
    if isinstance(exc, requests.exceptions.HTTPError):
        status = _status_of(exc)
        if status is None:
            return False
        return status in _RETRYABLE_CLIENT_STATUSES or status >= _FIRST_SERVER_ERROR
    return isinstance(exc, requests.exceptions.RequestException)


def _ends_the_call(exc: Exception) -> bool:
    return not _trips_breaker(exc)


@dataclass
class Trading212Client:
    """HTTP client for the Trading 212 metadata endpoints.

    Covers only the exchange and instrument lists used by universe
    ingestion; account/portfolio/order endpoints are intentionally absent
    (the daemon ingests market data, it does not trade).

    Attributes:
        max_retries: Attempts per request, the first included;
            ``TRADING_212_REQUEST_ATTEMPTS`` unless given.
        timeout_seconds: Timeout passed to ``requests.get``;
            ``TRADING_212_REQUEST_TIMEOUT_SECONDS`` unless given.
        breaker: Circuit breaker :meth:`_send` checks before it sends. Unless given, a
            client gets its own, sized by ``BREAKER_TRADING_212_MAX_ATTEMPTS``;
            :meth:`from_settings` passes the process-wide one, which the worker
            reports on ``external_breaker_open`` when metrics are enabled.
    """

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    mode: str = "live"
    max_retries: int = field(
        default_factory=lambda: settings.trading_212_request_attempts
    )
    timeout_seconds: float = field(
        default_factory=lambda: settings.trading_212_request_timeout_seconds
    )
    breaker: CircuitBreaker = field(
        default_factory=lambda: build_breaker(ExternalService.TRADING_212),
        repr=False,
    )
    base_url: str = field(init=False)
    rate_limit: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _instruments_cache: list[dict[str, Any]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if self.mode == "demo":
            self.base_url = "https://demo.trading212.com"
        else:
            self.base_url = "https://live.trading212.com"

    @property
    def headers(self) -> dict[str, str]:
        """Basic-auth plus the browser headers that clear Cloudflare.

        T212 requires Base64(api_key:api_secret) Basic auth; the metadata
        endpoints additionally 403 the default requests UA, so every request
        also carries a browser User-Agent and a JSON Accept.
        """
        credentials = base64.b64encode(
            f"{self.api_key}:{self.api_secret}".encode()
        ).decode()
        return {
            "Authorization": f"Basic {credentials}",
            "User-Agent": _BROWSER_USER_AGENT,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Metadata endpoints — the ONLY Trading 212 surface the ingestion
    # daemon uses. The account / portfolio / order-history / dividend-history
    # endpoints were removed: the daemon ingests market data, it does not
    # trade (see CLAUDE.md and docs/api.md). Several of those paths were also
    # stale vs the current API (e.g. /equity/portfolio -> /equity/positions,
    # /history/dividends -> /equity/history/dividends) and their cursor
    # pagination assumed `nextPageCursor` where the API now returns
    # `nextPagePath` — dead code that would have broken if ever revived.
    # ------------------------------------------------------------------

    def get_exchanges(self) -> list[dict[str, Any]]:
        return self._fetch_json("/api/v0/equity/metadata/exchanges")

    def get_instruments(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Return the T212 instrument universe, cached per client instance.

        T212 rate-limits ``/metadata/instruments`` aggressively (one call per
        account per window), so the list is fetched once and cached for the life
        of the client — the natural span of a single universe build. This is the
        primary way the strict per-account limit is respected; the latest
        ``x-ratelimit-*`` headers are also captured onto :attr:`rate_limit`.

        Args:
            force_refresh: When True, bypass the cache and re-fetch.

        Returns:
            The list of instrument-metadata dicts.
        """
        if self._instruments_cache is None or force_refresh:
            self._instruments_cache = self._fetch_json(
                "/api/v0/equity/metadata/instruments"
            )
        return self._instruments_cache

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """GET ``path`` through the circuit breaker, with retries.

        An attempt calls ``breaker.check()`` before it sends (:meth:`_send`). A
        failure :func:`_trips_breaker` accepts is passed to ``breaker.trigger()``
        and retried, up to :attr:`max_retries` attempts, unless ``check()``
        refuses the next attempt. Any other 4xx is raised at once.

        Args:
            path: Path under :attr:`base_url`.
            params: Query parameters.

        Returns:
            The parsed JSON body, ``None`` for a JSON ``null``.

        Raises:
            ValueError: :attr:`max_retries` is below 1.
            CircuitOpenError: The breaker refused an attempt; that attempt sent
                no request.
            requests.RequestException: At once for a 4xx other than 408 or 429
                and for an HTTP error without a response; otherwise the last
                failure, once the attempts have run out.
        """
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {self.max_retries}")
        url = f"{self.base_url}{path}"
        failures: list[Exception] = []

        def attempt() -> tuple[Any]:
            """Run one attempt, keeping its error for the caller."""
            try:
                return (self._send(url, params),)
            except Exception as exc:
                failures.append(exc)
                raise

        # The body travels in a tuple so that a JSON null is not mistaken for the
        # None that retry_with_backoff returns after the last failed attempt.
        answer = retry_with_backoff(
            attempt,
            self.max_retries,
            is_permanent_error=_ends_the_call,
            # _send has passed these to breaker.trigger(); the next attempt's
            # check() then sleeps out a window that is still open, or refuses
            # the attempt while it is open and the count is at max_attempts.
            is_rate_limit_error=_trips_breaker,
            on_success=lambda _: self.breaker.reset(),
        )
        if answer is None:
            raise failures[-1]
        return answer[0]

    def _send(self, url: str, params: dict[str, Any] | None) -> Any:
        """Send one GET through the breaker and return the parsed JSON body.

        A failure that :func:`_trips_breaker` accepts goes to
        ``breaker.trigger()``, with a 429's reported wait passed as
        ``min_wait_seconds``, before it propagates. ``trigger()`` changes nothing
        while a window is already open, as with two calls running at once.
        """
        self.breaker.check()
        try:
            return self._request(url, params)
        except requests.exceptions.RequestException as exc:
            if _trips_breaker(exc):
                self.breaker.trigger(min_wait_seconds=self._reported_wait(exc))
            raise

    def _request(self, url: str, params: dict[str, Any] | None) -> Any:
        with timed_request(ExternalService.TRADING_212):
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=self.timeout_seconds,
            )
            self._capture_rate_limit(response)
            response.raise_for_status()
            body = response.json()
        return body

    @classmethod
    def _reported_wait(cls, exc: Exception) -> float:
        response = getattr(exc, "response", None)
        if _status_of(exc) != _RATE_LIMITED or response is None:
            return 0.0
        return cls._rate_limit_wait(response.headers)

    @staticmethod
    def _rate_limit_wait(headers: Any) -> float:
        """Seconds a 429 asked for, or 0 when it gave nothing usable.

        A non-empty ``Retry-After`` decides alone: its seconds when it is a whole
        number, 0 when it is negative or not a whole number (an HTTP date,
        ``1.5``), and ``x-ratelimit-reset`` is then not read. Without one, the
        time left until ``x-ratelimit-reset`` plus a one-second margin, or 0 when
        that header is missing, unreadable or already past. The result is capped
        at ``_MAX_RATE_LIMIT_WAIT_SECONDS``.
        """
        if not isinstance(headers, Mapping):
            return 0.0
        lowered = {str(key).lower(): value for key, value in headers.items()}

        retry_after = lowered.get("retry-after")
        if retry_after:
            try:
                seconds = int(retry_after)
            except ValueError:
                return 0.0
            # The server's value is untrusted: an hour-scale one would stall the
            # universe build.
            return min(max(seconds, 0), _MAX_RATE_LIMIT_WAIT_SECONDS)

        try:
            until_reset = float(lowered["x-ratelimit-reset"]) - time.time()
        except (KeyError, TypeError, ValueError):
            return 0.0
        if not math.isfinite(until_reset) or until_reset <= 0:
            return 0.0
        return min(until_reset + _RESET_MARGIN_SECONDS, _MAX_RATE_LIMIT_WAIT_SECONDS)

    def _fetch_json(self, path: str) -> list[dict[str, Any]]:
        return self._get(path)

    def _capture_rate_limit(self, resp: Any) -> None:
        """Record the latest ``x-ratelimit-*`` headers for backpressure.

        Preserves the last-known limit across responses that omit the headers
        (e.g. non-metadata calls) rather than clobbering it with an empty map.
        Guards against mock/non-mapping ``headers`` in tests.
        """
        headers = getattr(resp, "headers", None)
        if not isinstance(headers, Mapping):
            return
        captured = {
            key.lower(): value
            for key, value in headers.items()
            if key.lower().startswith("x-ratelimit")
        }
        if captured:
            self.rate_limit = captured

    @staticmethod
    def missing_credentials() -> list[str]:
        """Names of the Trading 212 env vars that are unset in settings.

        Basic auth is ``Base64(API_KEY:SECRET_KEY)``, so a key without its
        secret authenticates as ``key:`` and every request 401s. This is the
        single definition of "configured" shared by :meth:`from_settings` and
        the scheduler's skip gate.
        """
        return [
            name
            for name, value in (
                ("TRADING_212_API_KEY", settings.trading_212_api_key),
                ("TRADING_212_SECRET_KEY", settings.trading_212_secret_key),
            )
            if not value
        ]

    @classmethod
    def from_settings(cls, mode: str | None = None) -> Optional["Trading212Client"]:
        """Construct a client from ``app.config.settings``, or return ``None``.

        Args:
            mode: Override for ``live``/``demo``; falls back to
                ``settings.trading_212_mode`` when omitted.

        Returns:
            A configured client on the process-wide Trading 212 breaker, or
            ``None`` when either API credential is absent from settings (the
            ``t212_annotate`` step skips without claiming a slot; the universe
            build does not need this client).
        """
        if cls.missing_credentials():
            return None
        api_key = settings.trading_212_api_key
        api_secret = settings.trading_212_secret_key
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            mode=mode or settings.trading_212_mode,
            breaker=process_breaker(ExternalService.TRADING_212),
        )
