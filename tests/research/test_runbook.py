"""Structural tests for ``research/RUNBOOK.md`` (issue #576).

Verifies the runbook exists at the expected path, contains every section
required by the acceptance criteria, names the prohibited Bash patterns,
documents the approved alternatives (``Monitor`` + ``until``; background
``Bash`` + bounded poller), and references the ``MAX_POLL_SECONDS`` env
var with its default of ``21600`` seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RUNBOOK = Path(__file__).resolve().parents[2] / "research" / "RUNBOOK.md"


@pytest.fixture(scope="module")
def runbook_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


class TestRunbookExists:
    def test_when_runbook_path_inspected_then_file_is_present(self):
        assert RUNBOOK.is_file(), f"missing {RUNBOOK}"


class TestRunbookSections:
    @pytest.mark.parametrize(
        "heading",
        [
            "Purpose",
            "Harness limitation",
            "Prohibited pattern",
            "Approved alternatives",
            "Scheduler poll reference",
        ],
    )
    def test_when_runbook_inspected_then_required_section_heading_present(
        self, runbook_text: str, heading: str
    ):
        assert f"## {heading}" in runbook_text, f"missing section: {heading}"


class TestProhibitedPatternsNamed:
    @pytest.mark.parametrize(
        "pattern",
        ["sleep N && tail", "sleep N; tail", "command && sleep N"],
    )
    def test_when_runbook_inspected_then_prohibited_pattern_quoted(
        self, runbook_text: str, pattern: str
    ):
        assert pattern in runbook_text, f"missing prohibited pattern: {pattern}"


class TestApprovedAlternativesNamed:
    @pytest.mark.parametrize(
        "term",
        ["Monitor", "until", "run_in_background=true"],
    )
    def test_when_runbook_inspected_then_approved_alternative_named(
        self, runbook_text: str, term: str
    ):
        assert term in runbook_text, f"missing approved-alternative term: {term}"


class TestSchedulerPollReference:
    def test_when_runbook_inspected_then_max_poll_seconds_documented(
        self, runbook_text: str
    ):
        assert "MAX_POLL_SECONDS" in runbook_text

    def test_when_runbook_inspected_then_default_21600_documented(
        self, runbook_text: str
    ):
        assert "21600" in runbook_text

    @pytest.mark.parametrize(
        "script",
        ["scheduler/fetch.sh", "scheduler/refetch_all.sh"],
    )
    def test_when_runbook_inspected_then_scheduler_script_referenced(
        self, runbook_text: str, script: str
    ):
        assert script in runbook_text, f"missing reference to {script}"

    def test_when_runbook_inspected_then_clarifies_env_var_not_schema_field(
        self, runbook_text: str
    ):
        text = runbook_text.lower()
        assert "env" in text
        assert (
            "not a schema field" in text
            or "not propagate" in text
            or "not a request" in text
        )
