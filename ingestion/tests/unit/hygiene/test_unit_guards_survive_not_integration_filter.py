"""
Pins acceptance criterion scope-4: the two `@pytest.mark.integration` decorators on
modules under `tests/unit/` are corrected (or those modules removed), so a
`-m "not integration"` expression does not silently drop unit guards.
"""

import subprocess
import sys
from pathlib import Path

import pytest

INGESTION_ROOT = Path(__file__).resolve().parents[3]
UNIT_TESTS = INGESTION_ROOT / "tests" / "unit"


def _collected_node_ids(extra_args):
    # -qq (not -q): pytest.ini bakes a single -v into addopts, so one -q nets to
    # plain verbosity and prints the tree diagram (no "::" node ids at all) — both
    # sides of the comparison would collect to an empty set and pass vacuously.
    # -qq forces the flat `path::test` listing regardless of addopts.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            str(UNIT_TESTS),
            "--collect-only",
            "-qq",
            *extra_args,
        ],
        cwd=INGESTION_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {line.strip() for line in result.stdout.splitlines() if "::" in line}


@pytest.mark.criterion("scope-4")
def test_when_not_integration_filter_applied_then_every_unit_test_is_still_collected():
    """
    Assumption: "does not silently drop unit guards" is read as set-equality between
    the full `tests/unit` collection and the `-m "not integration"` collection — any
    unit-tree module wrongly tagged `integration` would shrink the filtered set.
    `--collect-only` never executes a test body, so this is safe even though it targets
    the directory this very file lives in.
    """
    all_unit_tests = _collected_node_ids([])
    filtered_unit_tests = _collected_node_ids(["-m", "not integration"])

    assert filtered_unit_tests == all_unit_tests
