"""Example test for scope-4: the running scheduler exposes Prometheus
families on ``:9000/metrics`` and publishes no other (HTTP-API-shaped) port.

Source-blind by construction: hits the real ``/metrics`` endpoint over a
real socket against the real running container — no mock of the HTTP
response, no assumption about which families beyond the documented ones
are present.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from .conftest import _REPO_ROOT, _compose_ps_record

# Prometheus counters/gauge named in CLAUDE.md's Observability section —
# the minimum family set the daemon's own metrics module is documented to
# register (jobs_started_total, jobs_completed_total, jobs_failed_total,
# job_duration_seconds, jobs_in_progress).
_EXPECTED_METRIC_FAMILIES = (
    "jobs_started_total",
    "jobs_completed_total",
    "jobs_failed_total",
    "jobs_in_progress",
)


@pytest.mark.criterion("scope-4")
@pytest.mark.integration
def test_when_metrics_endpoint_is_curled_then_prometheus_families_are_returned(
    running_scheduler_container,
):
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("curl not available on PATH")

    result = subprocess.run(  # noqa: S603
        [curl, "-fsS", "http://localhost:9000/metrics"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    for family in _EXPECTED_METRIC_FAMILIES:
        assert family in result.stdout


@pytest.mark.criterion("scope-4")
@pytest.mark.integration
def test_when_scheduler_service_ports_are_inspected_then_only_the_metrics_port_is_published(
    running_scheduler_container,
):
    docker = running_scheduler_container["docker"]
    record = _compose_ps_record(docker, cwd=_REPO_ROOT)
    assert record is not None, "expected exactly one running 'scheduler' service"

    # ``docker compose ps --format json`` emits ``Publishers`` as a list of
    # dicts, each carrying the host-side ``PublishedPort`` for one mapping.
    published_host_ports = {
        str(entry["PublishedPort"])
        for entry in record.get("Publishers") or []
        if entry.get("PublishedPort")
    }

    assert published_host_ports == {"9000"}
