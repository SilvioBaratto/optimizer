"""METRICS_PORT contract: the daemon, the published port and the healthchecks agree.

`Settings.metrics_port` (env `METRICS_PORT`, default 9000) decides where the
daemon serves Prometheus. If compose still publishes a literal 9000 and the
healthchecks still probe a literal 9000, an operator who sets METRICS_PORT sees an
unreachable port and a container that never turns healthy, which makes
`docker compose up --wait` (used by `portopt start`) fail.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_DOCKERFILE = _REPO_ROOT / "ingestion" / "Dockerfile"

_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _scheduler() -> dict[str, Any]:
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return doc["services"]["scheduler"]


def _interpolate(text: str, env: dict[str, str]) -> str:
    """Resolve compose `${VAR}` / `${VAR:-default}` the way `docker compose` does."""
    return _VAR.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), text)


def _compose_probe_code() -> str:
    test = _scheduler()["healthcheck"]["test"]
    assert test[:3] == ["CMD", "python", "-c"]
    return test[3]


def _dockerfile_probe_code() -> str:
    """The python source of the Dockerfile HEALTHCHECK, continuations collapsed."""
    text = _DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        if line.lstrip().startswith("HEALTHCHECK"):
            words = shlex.split(line)
            return words[words.index("-c") + 1]
    raise AssertionError("ingestion/Dockerfile has no HEALTHCHECK")


def _probed_url(
    code: str, monkeypatch: pytest.MonkeyPatch, metrics_port: str | None
) -> str:
    """Run a healthcheck snippet in-process with a stubbed urlopen; return its URL.

    The probe executes inside the container, so the only configuration it can see
    is the container environment, which is what the env var below stands in for.
    """
    import urllib.request

    seen: list[str] = []

    def _urlopen(url: str, *args: Any, **kwargs: Any) -> SimpleNamespace:
        seen.append(url)
        return SimpleNamespace(read=lambda: b"")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    if metrics_port is None:
        monkeypatch.delenv("METRICS_PORT", raising=False)
    else:
        monkeypatch.setenv("METRICS_PORT", metrics_port)
    exec(code, {})  # noqa: S102 - the snippet under test, taken from our own compose file
    assert len(seen) == 1
    return seen[0]


@pytest.mark.parametrize("port", ["9000", "9100"])
def test_compose_publishes_the_configured_metrics_port(port: str) -> None:
    """The host mapping must follow METRICS_PORT, or the port the daemon listens on
    is not the one that is published (and it stays bound to loopback only)."""
    published = _scheduler()["ports"]
    assert [_interpolate(p, {"METRICS_PORT": port}) for p in published] == [
        f"127.0.0.1:{port}:{port}"
    ]


def test_compose_publishes_9000_when_metrics_port_is_unset() -> None:
    """Without an override the documented default port stays published."""
    assert [_interpolate(p, {}) for p in _scheduler()["ports"]] == [
        "127.0.0.1:9000:9000"
    ]


def test_compose_forwards_metrics_port_to_the_container() -> None:
    """A value set only in the shell (not in the root .env that env_file loads)
    still has to reach the daemon, or the published port and the listener diverge."""
    env = _scheduler()["environment"]
    assert _interpolate(env["METRICS_PORT"], {"METRICS_PORT": "9100"}) == "9100"
    assert _interpolate(env["METRICS_PORT"], {}) == "9000"


@pytest.mark.parametrize(
    "code_for",
    [_compose_probe_code, _dockerfile_probe_code],
    ids=["compose", "dockerfile"],
)
def test_healthcheck_probes_the_configured_metrics_port(
    code_for: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe on a hard-coded 9000 fails once the daemon listens elsewhere, so the
    container is reported unhealthy although it is serving metrics."""
    code = code_for()
    assert _probed_url(code, monkeypatch, "9100") == "http://localhost:9100/metrics"
    assert _probed_url(code, monkeypatch, None) == "http://localhost:9000/metrics"
