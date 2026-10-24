"""Compose `logging:` contract.

The daemons log JSON by default and Docker's default `json-file` driver keeps
every line forever. A weekend with Postgres down (heartbeat and reaper tracebacks
every few seconds) or a systematic write failure would grow the log files on the
same VM disk as the database volume until it is full, so each service caps its
log files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _services() -> dict[str, dict]:
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return doc["services"]


@pytest.mark.parametrize("name", sorted(_services()))
def test_service_logs_are_rotated(name: str) -> None:
    """Every service rotates its json-file logs at a bounded size and file count."""
    logging = _services()[name].get("logging")

    assert logging is not None, f"{name} has no logging options"
    assert logging["driver"] == "json-file"
    options = logging["options"]
    assert options["max-size"], f"{name} logs have no size cap"
    assert int(options["max-file"]) >= 2, f"{name} keeps no rotated files"
