"""Single-responsibility HTTP client for the Optimizer API."""

from __future__ import annotations

from typing import Any

import httpx
import typer
from rich.console import Console

console = Console(stderr=True)


class ApiClient:
    """Synchronous HTTP adapter for the Optimizer API.

    All HTTP interaction is isolated here so command modules
    depend on this abstraction rather than ``httpx`` directly.
    """

    API_PREFIX = "/api/v1"

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url + self.API_PREFIX,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the API server is reachable."""
        try:
            self._client.get("/health", timeout=2.0)
            return True
        except (httpx.ConnectError, httpx.HTTPError, Exception):
            return False

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        silent_connect_error: bool = False,
    ) -> Any:
        """Issue an HTTP request and return parsed JSON (or None for 204)."""
        try:
            resp = self._client.request(method, path, params=params, json=json)
        except httpx.ConnectError:
            if silent_connect_error:
                raise  # Let caller handle it
            console.print(
                f"[bold red]Error:[/] Cannot connect to {self._base_url}. "
                "Is the API server running?"
            )
            raise typer.Exit(code=1)
        except httpx.HTTPError as exc:
            console.print(f"[bold red]HTTP error:[/] {exc}")
            raise typer.Exit(code=1)

        if resp.status_code == 204:
            return None

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            console.print(f"[bold red]API error {resp.status_code}:[/] {detail}")
            raise typer.Exit(code=1)

        return resp.json()

    def _get(self, path: str, **params: Any) -> Any:
        cleaned = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=cleaned)

    def _post(self, path: str, *, json: dict[str, Any] | None = None) -> Any:
        return self._request("POST", path, json=json)

    def _delete(self, path: str, **params: Any) -> Any:
        cleaned = {k: v for k, v in params.items() if v is not None}
        return self._request("DELETE", path, params=cleaned)

    # ------------------------------------------------------------------
    # Universe endpoints
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return self._get("/universe/stats")

    def get_exchanges(self) -> list[dict[str, Any]]:
        return self._get("/universe/exchanges")

    def get_instruments(
        self,
        exchange: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self._get(
            "/universe/instruments",
            exchange=exchange,
            skip=skip,
            limit=limit,
        )

    def build_universe(
        self,
        exchanges: list[str] | None = None,
        skip_filters: bool = False,
        max_workers: int = 20,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "max_workers": max_workers,
            "skip_filters": skip_filters,
        }
        if exchanges:
            body["exchanges"] = exchanges
        return self._post("/universe/build", json=body)

    def get_build_status(self, build_id: str) -> dict[str, Any]:
        return self._get(f"/universe/build/{build_id}")

    def get_cache_stats(self) -> dict[str, Any]:
        return self._get("/universe/cache/stats")

    def clear_cache(self) -> None:
        self._delete("/universe/cache")

    # ------------------------------------------------------------------
    # YFinance endpoints
    # ------------------------------------------------------------------

    def start_fetch(
        self,
        max_workers: int = 4,
        period: str = "5y",
        mode: str = "incremental",
    ) -> dict[str, Any]:
        return self._post(
            "/yfinance-data/fetch",
            json={"max_workers": max_workers, "period": period, "mode": mode},
        )

    def get_fetch_status(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/yfinance-data/fetch/{job_id}")

    def fetch_ticker(
        self,
        ticker: str,
        period: str = "5y",
        mode: str = "incremental",
    ) -> dict[str, Any]:
        return self._post(
            f"/yfinance-data/fetch/ticker/{ticker}",
            json={"period": period, "mode": mode},
        )

    def get_profile(self, instrument_id: str) -> dict[str, Any]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/profile")

    def get_prices(
        self,
        instrument_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/yfinance-data/instruments/{instrument_id}/prices",
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    def get_financials(
        self,
        instrument_id: str,
        statement_type: str | None = None,
        period_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._get(
            f"/yfinance-data/instruments/{instrument_id}/financials",
            statement_type=statement_type,
            period_type=period_type,
        )

    def get_dividends(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/dividends")

    def get_splits(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/splits")

    def get_recommendations(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/recommendations")

    def get_price_targets(self, instrument_id: str) -> dict[str, Any]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/price-targets")

    def get_institutional_holders(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(
            f"/yfinance-data/instruments/{instrument_id}/institutional-holders"
        )

    def get_mutualfund_holders(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(
            f"/yfinance-data/instruments/{instrument_id}/mutualfund-holders"
        )

    def get_insider_transactions(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(
            f"/yfinance-data/instruments/{instrument_id}/insider-transactions"
        )

    def get_news(self, instrument_id: str) -> list[dict[str, Any]]:
        return self._get(f"/yfinance-data/instruments/{instrument_id}/news")

    # ------------------------------------------------------------------
    # Macro data endpoints
    # ------------------------------------------------------------------

    def start_macro_fetch(
        self,
        countries: list[str] | None = None,
        include_bonds: bool = True,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"include_bonds": include_bonds}
        if countries:
            body["countries"] = countries
        return self._post("/macro-data/fetch", json=body)

    def get_macro_fetch_status(self, job_id: str) -> dict[str, Any]:
        return self._get(f"/macro-data/fetch/{job_id}")

    def fetch_macro_country(
        self,
        country: str,
        include_bonds: bool = True,
    ) -> dict[str, Any]:
        return self._post(
            f"/macro-data/fetch/{country}",
            json={"include_bonds": include_bonds},
        )

    def get_country_summary(self, country: str) -> dict[str, Any]:
        return self._get(f"/macro-data/countries/{country}")

    def get_economic_indicators(
        self, country: str | None = None
    ) -> list[dict[str, Any]]:
        return self._get("/macro-data/economic-indicators", country=country)

    def get_te_indicators(self, country: str | None = None) -> list[dict[str, Any]]:
        return self._get("/macro-data/te-indicators", country=country)

    def get_bond_yields(self, country: str | None = None) -> list[dict[str, Any]]:
        return self._get("/macro-data/bond-yields", country=country)

    # ------------------------------------------------------------------
    # Database management endpoints
    # ------------------------------------------------------------------

    def db_health(self) -> dict[str, Any]:
        return self._get("/database/health")

    def db_status(self) -> dict[str, Any]:
        return self._get("/database/status")

    def db_tables(self) -> list[dict[str, Any]]:
        return self._get("/database/tables")

    def db_clear_table(self, name: str, confirm: bool = True) -> dict[str, Any]:
        return self._delete(f"/database/tables/{name}", confirm=confirm)

    def db_clear_all(self, confirm: bool = True) -> dict[str, Any]:
        return self._delete("/database/tables", confirm=confirm)
