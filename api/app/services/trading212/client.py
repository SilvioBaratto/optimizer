import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from app.config import settings


@dataclass
class Trading212Client:
    api_key: str
    mode: str = "live"
    max_retries: int = 5
    base_url: str = field(init=False)

    def __post_init__(self):
        if self.mode == "demo":
            self.base_url = "https://demo.trading212.com"
        else:
            self.base_url = "https://live.trading212.com"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    # ------------------------------------------------------------------
    # Metadata endpoints (existing)
    # ------------------------------------------------------------------

    def get_exchanges(self) -> list[dict[str, Any]]:
        return self._fetch_json("/api/v0/equity/metadata/exchanges")

    def get_instruments(self) -> list[dict[str, Any]]:
        return self._fetch_json("/api/v0/equity/metadata/instruments")

    # ------------------------------------------------------------------
    # Account endpoints
    # ------------------------------------------------------------------

    def get_account_cash(self) -> dict[str, Any]:
        """GET /api/v0/equity/account/cash → {blocked, free, invested, pieCash, result, total}"""
        return self._get("/api/v0/equity/account/cash")

    def get_account_info(self) -> dict[str, Any]:
        """GET /api/v0/equity/account/info → {currencyCode, id}"""
        return self._get("/api/v0/equity/account/info")

    # ------------------------------------------------------------------
    # Portfolio endpoints
    # ------------------------------------------------------------------

    def get_portfolio_positions(self) -> list[dict[str, Any]]:
        """GET /api/v0/equity/portfolio → list of open positions."""
        return self._get("/api/v0/equity/portfolio")

    # ------------------------------------------------------------------
    # Order history (cursor-paginated)
    # ------------------------------------------------------------------

    def get_order_history(
        self,
        *,
        cursor: int | None = None,
        ticker: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v0/equity/history/orders → {items, nextPageCursor}"""
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        if ticker is not None:
            params["ticker"] = ticker
        return self._get("/api/v0/equity/history/orders", params=params)

    def get_all_order_history(
        self, *, ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of order history."""
        return self._get_paginated(
            "/api/v0/equity/history/orders",
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # Dividend history (cursor-paginated)
    # ------------------------------------------------------------------

    def get_dividend_history(
        self,
        *,
        cursor: int | None = None,
        ticker: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """GET /api/v0/history/dividends → {items, nextPageCursor}"""
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        if ticker is not None:
            params["ticker"] = ticker
        return self._get("/api/v0/history/dividends", params=params)

    def get_all_dividend_history(
        self, *, ticker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch all pages of dividend history."""
        return self._get_paginated(
            "/api/v0/history/dividends",
            ticker=ticker,
        )

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _get(
        self, path: str, *, params: dict[str, Any] | None = None,
    ) -> Any:
        """General-purpose GET with retry/rate-limit logic. Returns parsed JSON."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{path}"
                resp = requests.get(
                    url, headers=self.headers, params=params, timeout=30,
                )
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

    def _get_paginated(
        self,
        path: str,
        *,
        ticker: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Auto-paginate a cursor-based endpoint. Returns all items."""
        all_items: list[dict[str, Any]] = []
        cursor: int | None = None

        while True:
            params: dict[str, Any] = {"limit": limit}
            if cursor is not None:
                params["cursor"] = cursor
            if ticker is not None:
                params["ticker"] = ticker

            page = self._get(path, params=params)
            items = page.get("items", [])
            all_items.extend(items)

            next_cursor = page.get("nextPageCursor")
            if not next_cursor or not items:
                break
            cursor = next_cursor

        return all_items

    def _fetch_json(self, path: str) -> list[dict[str, Any]]:
        """Legacy method — delegates to _get for backwards compatibility."""
        return self._get(path)

    @classmethod
    def from_settings(cls, mode: str | None = None) -> Optional["Trading212Client"]:
        api_key = settings.trading_212_api_key
        if not api_key:
            return None
        return cls(api_key=api_key, mode=mode or settings.trading_212_mode)
