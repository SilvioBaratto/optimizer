"""Example test for scope-3: the ``docker compose ps --format json scheduler``
argument list must be assembled in exactly one place. ``conftest.py``'s
``_scheduler_status`` already builds it; ``test_metrics_endpoint_exposure.py``
must call into that shared logic for its own port-inspection assertion
rather than re-listing the same argv a second time.

Source-blind by construction: this test reads only the two test files' own
source text (never anything under ``app/``) and checks that the argv literal
exists in exactly one of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOYMENT_DIR = Path(__file__).resolve().parent
_CONFTEST = _DEPLOYMENT_DIR / "conftest.py"
_METRICS_TEST = _DEPLOYMENT_DIR / "test_metrics_endpoint_exposure.py"

_COMPOSE_PS_ARGV_FRAGMENT = '"ps", "--format", "json"'


@pytest.mark.criterion("scope-3")
def test_when_the_compose_ps_argv_is_searched_for_then_it_is_defined_only_in_conftest():
    """
    Assumption: "shared with _scheduler_status rather than hand-rolled a
    second time" was read as "the argument list that names the compose-ps
    invocation appears exactly once repository-wide, in conftest.py" — the
    simplest, most direct reading of "not duplicated" for two files that both
    need the same subprocess call.
    """
    conftest_source = _CONFTEST.read_text()
    metrics_test_source = _METRICS_TEST.read_text()

    assert _COMPOSE_PS_ARGV_FRAGMENT in conftest_source, (
        "expected the shared compose-ps argv to still live in conftest.py's "
        "_scheduler_status"
    )
    assert _COMPOSE_PS_ARGV_FRAGMENT not in metrics_test_source, (
        "test_metrics_endpoint_exposure.py must reuse the shared compose-ps "
        "helper instead of re-listing the same argv locally"
    )
