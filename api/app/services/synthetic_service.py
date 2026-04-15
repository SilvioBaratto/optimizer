"""Synthetic scenario generation service (issue #366).

Single Responsibility: each function does exactly one thing.
  - _fetch_prices_df   — data access (delegate to fetch_close_prices)
  - generate_scenarios — orchestrate fit + extract
  - store_result       — persist large result to temp gzip file
  - load_result        — retrieve and validate stored result
"""

from __future__ import annotations

import gzip
import json
import logging
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from skfolio.preprocessing import prices_to_returns
from sqlalchemy.orm import Session

from app.services._price_fetcher import fetch_close_prices
from optimizer.synthetic._config import SyntheticDataConfig
from optimizer.synthetic._factory import build_synthetic_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LARGE_THRESHOLD: int = 10_000
_STORE_TTL_SECONDS: int = 3600
_MIN_RETURN_ROWS: int = 60
_TEMP_DIR: Path = Path(tempfile.gettempdir()) / "optimizer_synthetic"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _fetch_prices_df(tickers: list[str], session: Session) -> pd.DataFrame:
    """Return a prices DataFrame indexed by date for the requested tickers."""
    return fetch_close_prices(tickers, session)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_scenarios(
    tickers: list[str],
    n_samples: int,
    conditioning: dict[str, float] | None,
    preset: str | None,
    session: Session,
) -> tuple[np.ndarray, list[str]]:
    """Fit a SyntheticData estimator and return (scenarios_matrix, columns).

    Args:
        tickers: Asset tickers to include in the scenario universe.
        n_samples: Number of synthetic return scenarios to generate.
        conditioning: Optional dict mapping ticker → return shock for
            conditional (stress-test) generation.  Every ticker in this
            dict must appear in *tickers*.
        preset: One of ``"scenario_generation"`` or ``"stress_test"``.
            When ``None`` the preset is inferred: ``"stress_test"`` if
            *conditioning* is provided, ``"scenario_generation"`` otherwise.
        session: SQLAlchemy session used for price fetching.

    Returns:
        ``(scenarios, columns)`` where *scenarios* is an ndarray of shape
        ``(n_samples, n_assets)`` and *columns* is the ordered ticker list.

    Raises:
        ValueError: If a conditioning ticker is absent from *tickers*, or if
            the historical price data contains fewer than 60 rows.
    """
    # 1. Validate conditioning tickers against requested universe
    if conditioning:
        unknown = set(conditioning) - set(tickers)
        if unknown:
            raise ValueError(
                f"Conditioning tickers not in universe: {sorted(unknown)}"
            )

    # 2. Fetch prices from the database
    prices = _fetch_prices_df(tickers, session)

    # 3. Convert to returns (reduces rows by 1)
    returns = pd.DataFrame(prices_to_returns(prices))

    # 4. Enforce minimum history requirement
    if len(returns) < _MIN_RETURN_ROWS:
        raise ValueError(
            f"Insufficient historical data: need at least {_MIN_RETURN_ROWS} "
            f"return observations, got {len(returns)}."
        )

    # 5. Select config based on preset (or infer from presence of conditioning)
    resolved_preset = preset or ("stress_test" if conditioning else "scenario_generation")
    if resolved_preset == "stress_test":
        config = SyntheticDataConfig.for_stress_test(n_samples=n_samples)
    else:
        config = SyntheticDataConfig.for_scenario_generation(n_samples=n_samples)

    # 6. Build estimator, passing conditioning as sample_args when provided
    sample_args = {"conditioning": conditioning} if conditioning else None
    sd = build_synthetic_data(config, sample_args=sample_args)

    # 7. Fit and extract scenario matrix
    sd.fit(returns)
    scenarios: np.ndarray = sd.return_distribution_.returns
    columns: list[str] = list(returns.columns)

    logger.info(
        "Generated %d synthetic scenarios for %d assets (preset=%s)",
        n_samples,
        len(columns),
        resolved_preset,
    )
    return scenarios, columns


# ---------------------------------------------------------------------------
# Storage helpers for large results
# ---------------------------------------------------------------------------


def store_result(result: dict) -> tuple[str, datetime]:
    """Persist *result* as a gzip-compressed JSON file.

    Args:
        result: Arbitrary JSON-serialisable dict (without ``__expires_at`` key).

    Returns:
        ``(download_id, expires_at)`` where *download_id* is a UUID4 string
        and *expires_at* is a timezone-aware UTC datetime.
    """
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    download_id = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=_STORE_TTL_SECONDS)
    payload = {**result, "__expires_at": expires_at.isoformat()}
    file_path = _TEMP_DIR / f"{download_id}.json.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return download_id, expires_at


def load_result(download_id: str) -> dict | None:
    """Load a previously stored result by *download_id*.

    Returns:
        The stored dict (without the internal ``__expires_at`` key) if the
        file exists and has not expired, otherwise ``None``.
    """
    file_path = _TEMP_DIR / f"{download_id}.json.gz"
    if not file_path.exists():
        return None
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    expires_at_str = payload.pop("__expires_at", None)
    if expires_at_str is None:
        return None

    expires_at = datetime.fromisoformat(expires_at_str)
    # Ensure timezone-aware comparison
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) >= expires_at:
        return None

    return payload
