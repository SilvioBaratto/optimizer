"""Unit-test conftest: reset ``pipeline_builder`` module state per test.

The session store deliberately holds module-level mutable state (single
in-process registry). Tests get isolation by clearing those globals
between functions — not by reloading the module, which would recreate
the locks and break any thread that still holds a reference.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from app.services.pipeline_builder import session_store as _session_store


@pytest.fixture(autouse=True)
def _reset_pipeline_builder_state() -> Iterator[None]:
    """Clear ``_sessions`` / ``_job_services`` and reset the daemon flag."""
    _session_store._sessions.clear()
    _session_store._job_services.clear()
    _session_store._daemon_started = False
    yield
    _session_store._sessions.clear()
    _session_store._job_services.clear()
    _session_store._daemon_started = False
