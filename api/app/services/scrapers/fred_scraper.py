"""FRED (Federal Reserve Economic Data) REST API client.

Fetches time-series observations for credit spreads, yield spreads,
volatility indices, business cycle indicators, and recession
probabilities via the public FRED API (api.stlouisfed.org).
"""

import logging
from datetime import date as date_type
from datetime import datetime

import requests

from app.services.infrastructure import CircuitBreaker, retry_with_backoff
from app.services.infrastructure.retry import is_transient_network_error

_fred_circuit_breaker = CircuitBreaker(service_name="FRED API", max_attempts=8)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Series IDs tracked for the macroeconomic analysis pipeline.
# Grouped by purpose so downstream consumers can pull subsets if needed.
# ---------------------------------------------------------------------------

# Credit & yield spreads (daily)
FRED_SPREAD_SERIES: dict[str, str] = {
    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",
    "BAMLC0A0CM": "ICE BofA US Corporate IG OAS",
    "T10Y2Y": "10Y-2Y Treasury Constant Maturity Spread",
    "BAA10Y": "Moody's Seasoned Baa Corporate Bond vs 10Y Treasury",
}

# Volatility (daily)
FRED_VOLATILITY_SERIES: dict[str, str] = {
    "VIXCLS": "CBOE Volatility Index (VIX) Daily Close",
}

# OECD Composite Leading Indicators — amplitude adjusted (monthly)
# CLI > 100 and rising → expansion; > 100 and falling → slowdown;
# < 100 and falling → contraction; < 100 and rising → recovery.
FRED_CLI_SERIES: dict[str, str] = {
    "USALOLITOAASTSAM": "OECD CLI Amplitude Adjusted - USA",
    "DEULOLITOAASTSAM": "OECD CLI Amplitude Adjusted - Germany",
    "FRALOLITOAASTSAM": "OECD CLI Amplitude Adjusted - France",
    "GBRLOLITOAASTSAM": "OECD CLI Amplitude Adjusted - UK",
}

# US recession / business cycle indicators (monthly/quarterly)
FRED_RECESSION_SERIES: dict[str, str] = {
    "RECPROUSM156N": "Smoothed US Recession Probability (%)",
    "JHGDPBRINDX": "GDP-Based Recession Indicator Index",
    "USREC": "NBER Recession Indicator (0=expansion, 1=recession)",
}

# Risk-free rate proxy (daily)
FRED_RATE_SERIES: dict[str, str] = {
    "DGS3MO": "3-Month Treasury Constant Maturity Rate",
}

# FX / Dollar Index (daily)
FRED_FX_SERIES: dict[str, str] = {
    "DTWEXBGS": "Trade Weighted US Dollar Index: Broad, Goods and Services",
}

# Combined registry used by default fetch operations.
FRED_SERIES: dict[str, str] = {
    **FRED_SPREAD_SERIES,
    **FRED_VOLATILITY_SERIES,
    **FRED_CLI_SERIES,
    **FRED_RECESSION_SERIES,
    **FRED_RATE_SERIES,
    **FRED_FX_SERIES,
}

_FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FredScraper:
    """REST client for the FRED series/observations endpoint.

    Args:
        api_key: FRED API key (from ``FRED_API_KEY`` env var).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, api_key: str, timeout: int = 30) -> None:
        if not api_key:
            raise ValueError("FRED API key is required. Set FRED_API_KEY in .env")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_series_observations(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Fetch all observations for a single FRED series.

        Args:
            series_id: FRED series identifier (e.g. ``"BAMLH0A0HYM2"``).
            start_date: ISO date ``"YYYY-MM-DD"`` for incremental fetch.
            end_date: ISO date ``"YYYY-MM-DD"`` upper bound.

        Returns:
            Dict with keys: ``status``, ``series_id``, ``observations``,
            ``count``, ``timestamp``.
        """
        params: dict[str, str | int] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": 100_000,
            "sort_order": "asc",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date

        url = f"{_FRED_BASE_URL}/series/observations"

        def _fetch() -> requests.Response:
            _fred_circuit_breaker.check()
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp

        raw = retry_with_backoff(
            _fetch,
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            is_rate_limit_error=is_transient_network_error,
            on_rate_limit=_fred_circuit_breaker.trigger,
            on_success=lambda _: _fred_circuit_breaker.reset(),
        )
        if raw is None:
            return {
                "status": "error",
                "series_id": series_id,
                "error": "Fetch failed after retries",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            data = raw.json()

            if "error_code" in data:
                return {
                    "status": "error",
                    "series_id": series_id,
                    "error": data.get("error_message", "FRED API error"),
                    "timestamp": datetime.now().isoformat(),
                }

            observations = self._parse_observations(data.get("observations", []))

            return {
                "status": "success",
                "series_id": series_id,
                "observations": observations,
                "count": len(observations),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as exc:
            return {
                "status": "error",
                "series_id": series_id,
                "error": f"Parsing failed: {exc!s}",
                "timestamp": datetime.now().isoformat(),
            }

    def get_all_series(
        self,
        series_ids: list[str] | None = None,
        start_date: str | None = None,
    ) -> dict[str, dict]:
        """Fetch all configured FRED series.

        Args:
            series_ids: Override list. ``None`` means ``FRED_SERIES`` keys.
            start_date: ISO date for incremental fetch across all series.

        Returns:
            Dict mapping ``series_id`` → result dict.
        """
        ids = series_ids if series_ids is not None else list(FRED_SERIES.keys())
        results: dict[str, dict] = {}
        for series_id in ids:
            logger.info("Fetching FRED series: %s", series_id)
            results[series_id] = self.get_series_observations(
                series_id, start_date=start_date
            )
        return results

    @staticmethod
    def _parse_observations(raw: list[dict]) -> list[dict]:
        """Convert raw FRED JSON observations to typed dicts.

        FRED returns ``{"date": "YYYY-MM-DD", "value": "3.14" | "."}``.
        ``"."`` means missing data — returned as ``None``.
        """
        parsed: list[dict] = []
        for obs in raw:
            raw_value = obs.get("value", "")
            value: float | None = None
            if raw_value not in (".", "", None):
                try:
                    value = float(raw_value)
                except ValueError:
                    value = None

            try:
                obs_date = date_type.fromisoformat(obs["date"])
            except (KeyError, ValueError):
                continue

            parsed.append({"date": obs_date, "value": value})

        return parsed
