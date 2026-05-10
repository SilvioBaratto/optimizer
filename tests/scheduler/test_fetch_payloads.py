"""Structural tests for scheduler shell payloads (issue #574).

Verifies that ``scheduler/fetch.sh`` and ``scheduler/refetch_all.sh`` send the
renamed ``workers`` field to ``/yfinance-data/fetch`` instead of the legacy
``max_workers`` field, and that the ``MAX_POLL_SECONDS`` default is preserved.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FETCH_SCRIPT = REPO_ROOT / "scheduler" / "fetch.sh"
REFETCH_SCRIPT = REPO_ROOT / "scheduler" / "refetch_all.sh"

YFINANCE_PAYLOAD_RE = re.compile(
    r"/yfinance-data/fetch.*?\'(?P<payload>\{[^\']+\})",
    re.DOTALL,
)


def _yfinance_payload(script: Path) -> str:
    text = script.read_text()
    match = YFINANCE_PAYLOAD_RE.search(text)
    assert match, f"no /yfinance-data/fetch payload found in {script}"
    return match.group("payload")


@pytest.mark.parametrize("script", [FETCH_SCRIPT, REFETCH_SCRIPT])
class TestYfinancePayload:
    def test_when_payload_inspected_then_workers_field_present(self, script: Path):
        payload = _yfinance_payload(script)
        assert '"workers":1' in payload, payload

    def test_when_payload_inspected_then_legacy_max_workers_absent(self, script: Path):
        payload = _yfinance_payload(script)
        assert "max_workers" not in payload, payload


@pytest.mark.parametrize("script", [FETCH_SCRIPT, REFETCH_SCRIPT])
class TestShellHarness:
    def test_when_bash_n_run_then_exits_zero(self, script: Path):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_when_inspecting_max_poll_seconds_then_default_is_21600(self, script: Path):
        text = script.read_text()
        assert 'MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-21600}"' in text
