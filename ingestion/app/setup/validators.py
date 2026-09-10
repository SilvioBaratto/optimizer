"""Live API-key validators for the portopt install wizard (SPEC D8).

Each validator issues one cheap request and returns ``True`` on HTTP 200,
``False`` on auth/other non-200 (invalid key), or raises
``ValidationNetworkError`` when the request itself fails (network/timeout) so the
wizard can tell "bad key" (retry) apart from "network down" (abort/inform).
"""

from __future__ import annotations

import base64

import requests

_TIMEOUT = 10

_T212_BASE = {
    "live": "https://live.trading212.com",
    "demo": "https://demo.trading212.com",
}


class ValidationNetworkError(Exception):
    """Raised when a validation request cannot reach the service."""


def _get(url: str, **kwargs: object) -> requests.Response:
    try:
        return requests.get(url, timeout=_TIMEOUT, **kwargs)  # type: ignore[arg-type]
    except requests.RequestException as exc:
        raise ValidationNetworkError(str(exc)) from exc


def validate_t212(api_key: str, secret_key: str, *, mode: str = "live") -> bool:
    """True if the Trading212 metadata endpoint accepts the key/secret pair."""
    base = _T212_BASE.get(mode, _T212_BASE["live"])
    token = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()
    resp = _get(
        f"{base}/api/v0/equity/metadata/exchanges",
        headers={"Authorization": f"Basic {token}"},
    )
    return resp.status_code == 200


def validate_fred(api_key: str) -> bool:
    """True if the FRED API accepts the key on a minimal series request."""
    resp = _get(
        "https://api.stlouisfed.org/fred/series",
        params={"series_id": "GNPCA", "api_key": api_key, "file_type": "json"},
    )
    return resp.status_code == 200
