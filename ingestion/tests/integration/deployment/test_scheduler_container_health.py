"""Example test for scope-3: the built ``scheduler`` container reaches
Docker Compose status ``healthy``.

Source-blind by construction: drives the real ``docker compose`` CLI
against the real ``docker-compose.yml`` and the real healthcheck it
declares — no mock of Docker, no assumption about the healthcheck's
implementation.
"""

from __future__ import annotations

import pytest


@pytest.mark.criterion("scope-3")
@pytest.mark.integration
def test_when_the_scheduler_service_is_built_and_started_then_compose_reports_it_healthy(
    running_scheduler_container,
):
    assert running_scheduler_container["final_status"] == "healthy"
