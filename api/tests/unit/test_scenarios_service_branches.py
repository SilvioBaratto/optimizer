"""Branch-coverage tests for scenarios service modules.

Targets the missed lines in synthetic_service.py:
  48       — _fetch_prices_df delegates to fetch_close_prices
  168-169  — load_result: OSError / json.JSONDecodeError caught → returns None
  173      — load_result: __expires_at key absent → returns None
  178      — load_result: naive expires_at datetime → tzinfo replaced before compare
"""

from __future__ import annotations

import gzip
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# ===========================================================================
# L48 — _fetch_prices_df delegates to fetch_close_prices
# ===========================================================================


class TestFetchPricesDfDelegates:
    """_fetch_prices_df must delegate to fetch_close_prices unchanged."""

    def test_delegates_call_to_fetch_close_prices(self) -> None:
        import numpy as np
        import pandas as pd

        from app.services.scenarios.synthetic_service import _fetch_prices_df

        mock_df = pd.DataFrame(
            np.zeros((10, 2)),
            columns=["AAPL", "MSFT"],
        )
        session = MagicMock()

        with patch(
            "app.services.scenarios.synthetic_service.fetch_close_prices",
            return_value=mock_df,
        ) as mock_fetch:
            result = _fetch_prices_df(["AAPL", "MSFT"], session)

        mock_fetch.assert_called_once_with(["AAPL", "MSFT"], session)
        assert result is mock_df

    def test_returns_dataframe_from_delegate(self) -> None:
        import pandas as pd

        from app.services.scenarios.synthetic_service import _fetch_prices_df

        empty_df = pd.DataFrame()
        session = MagicMock()

        with patch(
            "app.services.scenarios.synthetic_service.fetch_close_prices",
            return_value=empty_df,
        ):
            result = _fetch_prices_df([], session)

        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# L168-169 — load_result: OSError / json.JSONDecodeError → returns None
# ===========================================================================


class TestLoadResultCorruptFile:
    """When gzip.open raises OSError or json.load raises JSONDecodeError, return None."""

    def _write_valid_gz(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def test_os_error_on_open_returns_none(self) -> None:
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        # Write a valid placeholder so the file-exists check passes
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")  # empty file — gzip.open read will raise

        result = load_result(download_id)

        # Corrupt/empty file raises OSError or json.JSONDecodeError → None
        assert result is None

    def test_invalid_json_inside_gz_returns_none(self) -> None:
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with gzip.open(file_path, "wt", encoding="utf-8") as fh:
            fh.write("NOT VALID JSON {{{")

        result = load_result(download_id)

        assert result is None


# ===========================================================================
# L173 — load_result: __expires_at key absent → returns None
# ===========================================================================


class TestLoadResultNoExpiresAt:
    """When the stored payload lacks __expires_at, load_result returns None."""

    def test_missing_expires_at_returns_none(self) -> None:
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Payload without __expires_at key
        payload = {"scenarios": [[0.01]], "columns": ["A"]}
        with gzip.open(file_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

        result = load_result(download_id)

        assert result is None


# ===========================================================================
# L178 — load_result: naive expires_at datetime → tzinfo replaced
# ===========================================================================


class TestLoadResultNaiveExpiresAt:
    """When expires_at is stored without timezone, it must be treated as UTC."""

    def test_naive_iso_expires_in_future_loads_successfully(self) -> None:
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Naive datetime (no +00:00 suffix) that expires in the future
        future_naive = (
            datetime.utcnow() + timedelta(hours=2)
        ).isoformat()  # no tzinfo
        payload = {
            "__expires_at": future_naive,
            "scenarios": [[0.1, -0.2]],
            "columns": ["AAPL", "MSFT"],
        }
        with gzip.open(file_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

        result = load_result(download_id)

        # Naive future expiry should be accepted (treated as UTC → still in future)
        assert result is not None
        assert result["columns"] == ["AAPL", "MSFT"]

    def test_naive_iso_expires_in_past_returns_none(self) -> None:
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Naive datetime that expired 10 seconds ago
        past_naive = (datetime.utcnow() - timedelta(seconds=10)).isoformat()
        payload = {
            "__expires_at": past_naive,
            "scenarios": [[0.1]],
            "columns": ["A"],
        }
        with gzip.open(file_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

        result = load_result(download_id)

        assert result is None

    def test_aware_iso_expires_in_future_still_loads(self) -> None:
        """Sanity check: timezone-aware future timestamp also works."""
        from app.services.scenarios.synthetic_service import _TEMP_DIR, load_result

        download_id = str(uuid.uuid4())
        file_path = _TEMP_DIR / f"{download_id}.json.gz"
        _TEMP_DIR.mkdir(parents=True, exist_ok=True)

        future_aware = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat()  # includes +00:00
        payload = {
            "__expires_at": future_aware,
            "data": "payload",
        }
        with gzip.open(file_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

        result = load_result(download_id)

        assert result is not None
        assert result["data"] == "payload"
