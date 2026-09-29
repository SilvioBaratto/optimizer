import base64
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


@dataclass
class Trading212Client:
    api_key: str
    api_secret: str
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

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

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
                if e.response is not None and e.response.status_code == 429:
                    if attempt >= self.max_retries - 1:
                        raise
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            wait_time = (2**attempt) * 2
                    else:
                        wait_time = (2**attempt) * 2
                    time.sleep(wait_time)
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

    def _fetch_json(self, path: str) -> list[dict[str, Any]]:
        """Fetch a JSON array from a metadata endpoint (via ``_get``)."""
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

    @classmethod
    def from_settings(cls, mode: str | None = None) -> Optional["Trading212Client"]:
        api_key = settings.trading_212_api_key
        api_secret = settings.trading_212_secret_key
        if not api_key or not api_secret:
            return None
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            mode=mode or settings.trading_212_mode,
        )
