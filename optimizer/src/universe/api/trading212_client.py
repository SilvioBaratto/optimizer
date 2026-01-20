"""
Trading212 API Client - Fetches exchange and instrument metadata.

Single Responsibility: HTTP communication with Trading212 API.

Note: This module is intentionally silent (no logging/print).
All user-facing output is handled by the CLI layer.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import requests


@dataclass
class Trading212Client:
    """
    Client for Trading212 API.

    Fetches exchange and instrument metadata with:
    - Rate limiting with exponential backoff
    - Retry logic for transient failures
    - Support for demo and live environments

    Attributes:
        api_key: Trading212 API key
        mode: 'demo' or 'live' environment
        max_retries: Maximum retry attempts for API calls
        base_url: Base URL (computed from mode)
    """

    api_key: str
    mode: str = "live"
    max_retries: int = 5
    base_url: str = field(init=False)

    def __post_init__(self):
        """Set base URL based on mode."""
        if self.mode == "demo":
            self.base_url = "https://demo.trading212.com"
        else:
            self.base_url = "https://live.trading212.com"

    @property
    def headers(self) -> Dict[str, str]:
        """Get HTTP headers with authorization."""
        return {"Authorization": self.api_key}

    def get_exchanges(self) -> List[Dict[str, Any]]:
        """
        Fetch all exchanges from Trading212 API.

        Returns:
            List of exchange dictionaries with keys:
                - id: Exchange ID
                - name: Exchange name
                - workingSchedules: List of schedule dicts with 'id' key

        Raises:
            Exception: If API call fails after all retries
        """
        return self._fetch_json("/api/v0/equity/metadata/exchanges")

    def get_instruments(self) -> List[Dict[str, Any]]:
        """
        Fetch all instruments from Trading212 API.

        Returns:
            List of instrument dictionaries with keys:
                - ticker: Trading212 ticker (e.g., 'AAPL_US_EQ')
                - shortName: Short name (e.g., 'AAPL')
                - name: Full company name
                - isin: ISIN code
                - type: 'STOCK', 'ETF', etc.
                - currencyCode: Trading currency
                - workingScheduleId: Schedule ID (links to exchange)
                - maxOpenQuantity: Position limit
                - addedOn: Date added

        Raises:
            Exception: If API call fails after all retries
        """
        return self._fetch_json("/api/v0/equity/metadata/instruments")

    def _fetch_json(self, path: str) -> List[Dict[str, Any]]:
        """
        Fetch JSON from Trading212 API with retry logic.

        Args:
            path: API endpoint path

        Returns:
            JSON response as list of dictionaries

        Raises:
            Exception: If API call fails after all retries
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{path}"
                resp = requests.get(url, headers=self.headers, timeout=30)
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code == 429:
                    # Rate limit - exponential backoff
                    wait_time = (2**attempt) * 2
                    time.sleep(wait_time)
                    if attempt == self.max_retries - 1:
                        raise
                else:
                    raise

            except requests.exceptions.RequestException as e:
                last_error = e
                error_str = str(e).lower()

                # Check for rate limit or timeout errors in message
                if any(
                    x in error_str
                    for x in ["rate limit", "too many requests", "timeout", "timed out"]
                ):
                    if attempt < self.max_retries - 1:
                        # Progressive backoff: 60s, 300s, 900s, 1800s, 3600s
                        wait_times = [60, 300, 900, 1800, 3600]
                        wait_time = (
                            wait_times[attempt] if attempt < len(wait_times) else 3600
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        raise
                else:
                    # Other request errors - shorter retry
                    time.sleep(5)
                    if attempt == self.max_retries - 1:
                        raise

        raise Exception(f"Failed to fetch {path} after {self.max_retries} attempts") from last_error

    @classmethod
    def from_env(cls, mode: str = "live") -> Optional["Trading212Client"]:
        """
        Create client from environment variable.

        Args:
            mode: 'demo' or 'live' environment

        Returns:
            Trading212Client if TRADING_212_API_KEY is set, None otherwise
        """
        import os

        api_key = os.getenv("TRADING_212_API_KEY")
        if not api_key:
            return None
        return cls(api_key=api_key, mode=mode)
