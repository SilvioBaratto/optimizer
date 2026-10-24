"""Liveness heartbeat shared by the worker and the container healthcheck.

Stdlib only, so the probe (spawned every thirty seconds by Docker) starts without
importing the scheduler, SQLAlchemy or the agent stack.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_FILE = "/tmp/fund-worker.heartbeat"  # noqa: S108 - container-local
# Several monitor ticks, so one slow tick does not flap the container to unhealthy.
_MAX_AGE_SECONDS = 150.0


def heartbeat_path() -> Path:
    """Where the worker records its heartbeat (``FUND_HEARTBEAT_FILE`` overrides)."""
    return Path(os.environ.get("FUND_HEARTBEAT_FILE", _DEFAULT_HEARTBEAT_FILE))


def touch_heartbeat(path: Path | None = None) -> None:
    """Record that the worker is alive now; a failed write logs instead of raising.

    The monitor loop must keep running whatever the filesystem does, and a missing
    heartbeat already fails the probe.
    """
    target = path if path is not None else heartbeat_path()
    try:
        target.touch()
    except OSError as exc:
        logger.warning("could not touch the worker heartbeat %s: %s", target, exc)


def heartbeat_is_fresh(path: Path, *, max_age_seconds: float) -> bool:
    """Whether ``path`` exists and was touched within ``max_age_seconds``."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age <= max_age_seconds


def main() -> int:
    """Probe entry point: exit status 0 when the heartbeat is fresh, else 1."""
    fresh = heartbeat_is_fresh(heartbeat_path(), max_age_seconds=_MAX_AGE_SECONDS)
    return 0 if fresh else 1


if __name__ == "__main__":
    sys.exit(main())
