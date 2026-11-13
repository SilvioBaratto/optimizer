"""Guard: the non-functional requirements and the test plan exist and stay truthful.

Standard 2.13 asks for documented scalability, maintainability and performance with
results, and 6.1 to 6.4 ask for a test plan that maps the critical code to its tests,
names the identification scheme and lists the acceptance tests. A plan that cites a test
file that was renamed misleads the next reader, so every cited path is checked.
"""

from __future__ import annotations

import re
from pathlib import Path

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_REPO_ROOT = _DOCS.parent

_CRITICAL_FUNCTIONS = (
    "risk_check",
    "place_orders",
    "build_nav_series",
    "resolve_mandate_eligibility",
    "fund_drift_monitor",
    "_upsert",
    "_fetch_and_store",
    "retry_with_backoff",
    "benjamini_hochberg",
)
_TEST_PATH = re.compile(
    r"`((?:tests|[a-z-]+/tests|packages/[a-z-]+/tests)/[A-Za-z0-9_./-]+\.py)`"
)


def _read(name: str) -> str:
    return (_DOCS / name).read_text(encoding="utf-8")


def test_nfr_covers_scalability_performance_and_maintainability() -> None:
    """Each aspect standard 2.13 names has its own section."""
    text = _read("nfr.md")

    for aspect in ("Scalability", "Performance", "Maintainability"):
        assert f"\n## {aspect}\n" in text, aspect


def test_nfr_reports_dated_measurements_with_targets() -> None:
    """Numbers are measured, dated and set against a target, not asserted."""
    text = _read("nfr.md")

    assert "Measured on 2026-11-13" in text
    assert text.count("Target") >= 3


def test_test_plan_has_the_sections_standard_6_asks_for() -> None:
    """Scope, risks, levels, identification, acceptance tests and defect handling."""
    text = _read("test-plan.md")

    for section in (
        "Scope",
        "Risks",
        "Critical functions",
        "Test levels",
        "Test case identification",
        "Acceptance tests",
        "Defects",
    ):
        assert f"\n## {section}\n" in text, section


def test_test_plan_maps_every_critical_function_to_tests() -> None:
    """The critical code from standard 6.2 is named in the plan."""
    text = _read("test-plan.md")

    missing = [name for name in _CRITICAL_FUNCTIONS if f"`{name}`" not in text]

    assert missing == []


def test_every_test_file_the_plan_cites_exists() -> None:
    """A renamed test file must not leave a dangling row in the plan."""
    cited = sorted(set(_TEST_PATH.findall(_read("test-plan.md"))))

    missing = [path for path in cited if not (_REPO_ROOT / path).is_file()]

    assert len(cited) >= 9
    assert missing == []


def test_requirements_registry_has_its_header() -> None:
    """The registry the traceability check will read exists, even when still empty."""
    text = _read("requirements.md")

    assert "| id | statement | module | standard |" in text
    assert "REQ-" in text
