import base64
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from app.config import settings

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
_MAX_RATE_LIMIT_WAIT_SECONDS = 120
_RESET_MARGIN_SECONDS = 1


@dataclass
class Trading212Client:
    """HTTP client for the Trading 212 metadata endpoints.

    Covers only the exchange and instrument lists used by universe
    ingestion; account/portfolio/order endpoints are intentionally absent
    (the daemon ingests market data, it does not trade).
    """

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    mode: str = "live"
    max_retries: int = 5
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
        """General-purpose GET with retry/rate-limit logic. Returns parsed JSON."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{path}"
                resp = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30,
                )
                self._capture_rate_limit(resp)
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as e:
                last_error = e
                response = e.response
                if response is None:
                    raise
                if response.status_code == 429:
                    if attempt >= self.max_retries - 1:
                        raise
                    time.sleep(self._rate_limit_wait(response.headers, attempt))
                    continue
                if response.status_code == 408 or response.status_code >= 500:
                    # Cloudflare in front of T212 answers 502/503/504 under load.
                    if attempt >= self.max_retries - 1:
                        raise
                    time.sleep((2**attempt) * 2)
                    continue
                raise

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt >= self.max_retries - 1:
                    raise
                time.sleep((2**attempt) * 2)
                continue

        raise Exception(
            f"Failed to fetch {path} after {self.max_retries} attempts"
        ) from last_error

    @staticmethod
    def _rate_limit_wait(headers: Any, attempt: int) -> float:
        """Seconds to sleep after a 429.

        Precedence: ``Retry-After`` (seconds), then the time left until
        ``x-ratelimit-reset`` (capped), then exponential backoff when neither
        header is usable.
        """
        backoff = (2**attempt) * 2
        if not isinstance(headers, Mapping):
            return backoff
        lowered = {str(key).lower(): value for key, value in headers.items()}

        retry_after = lowered.get("retry-after")
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                return backoff

        try:
            until_reset = float(lowered["x-ratelimit-reset"]) - time.time()
        except (KeyError, TypeError, ValueError):
            return backoff
        if not math.isfinite(until_reset) or until_reset <= 0:
            return backoff
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
            A configured client, or ``None`` when either API credential is
            absent from settings (the ``t212_annotate`` step skips without
            claiming a slot; the universe build does not need this client).
        """
        if cls.missing_credentials():
            return None
        api_key = settings.trading_212_api_key
        api_secret = settings.trading_212_secret_key
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            mode=mode or settings.trading_212_mode,
        )
