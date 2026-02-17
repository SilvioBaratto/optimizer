import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

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
    def headers(self) -> Dict[str, str]:
        return {"Authorization": self.api_key}

    def get_exchanges(self) -> List[Dict[str, Any]]:
        return self._fetch_json("/api/v0/equity/metadata/exchanges")

    def get_instruments(self) -> List[Dict[str, Any]]:
        return self._fetch_json("/api/v0/equity/metadata/instruments")

    def _fetch_json(self, path: str) -> List[Dict[str, Any]]:
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
                    if attempt >= self.max_retries - 1:
                        raise
                    # Respect Retry-After header, fall back to exponential backoff
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                        except ValueError:
                            wait_time = (2 ** attempt) * 2
                    else:
                        wait_time = (2 ** attempt) * 2
                    time.sleep(wait_time)
                    continue
                raise

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt >= self.max_retries - 1:
                    raise
                time.sleep((2 ** attempt) * 2)
                continue

        raise Exception(f"Failed to fetch {path} after {self.max_retries} attempts") from last_error

    @classmethod
    def from_settings(cls, mode: Optional[str] = None) -> Optional["Trading212Client"]:
        api_key = settings.trading_212_api_key
        if not api_key:
            return None
        return cls(api_key=api_key, mode=mode or settings.trading_212_mode)
