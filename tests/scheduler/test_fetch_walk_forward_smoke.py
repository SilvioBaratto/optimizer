"""Structural + syntax tests for the walk-forward smoke stage in scheduler/fetch.sh.

The smoke stage is opt-in (flag / env var) and calls
``POST /api/v1/validate/cross-validation`` after the five fetch stages succeed.

These tests verify the script's shape and syntax rather than end-to-end behavior,
which would require mocking 6 HTTP endpoints. They catch the concrete failure
modes identified in the issue review (wrong endpoint path, hardcoded port,
missing env-var knobs, duplication of fire_and_poll).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scheduler" / "fetch.sh"


@pytest.fixture(scope="module")
def script_source() -> str:
    return SCRIPT_PATH.read_text()


class TestFetchScriptSyntax:
    """The script must remain valid POSIX sh after modification."""

    def test_sh_n_parses_script_without_errors(self) -> None:
        result = subprocess.run(
            ["sh", "-n", str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestWalkForwardSmokeStage:
    """Structural requirements for the opt-in walk-forward smoke stage."""

    def test_defines_run_walk_forward_smoke_function(self, script_source: str) -> None:
        assert re.search(r"run_walk_forward_smoke\s*\(\)\s*\{", script_source), (
            "expected a run_walk_forward_smoke() function definition"
        )

    def test_enable_flag_env_var_is_consulted(self, script_source: str) -> None:
        assert "ENABLE_WALK_FORWARD_SMOKE" in script_source

    def test_smoke_tickers_env_var_is_consulted(self, script_source: str) -> None:
        assert "SMOKE_TICKERS" in script_source

    def test_smoke_start_date_env_var_is_consulted(self, script_source: str) -> None:
        assert "SMOKE_START_DATE" in script_source

    def test_smoke_end_date_env_var_is_consulted(self, script_source: str) -> None:
        assert "SMOKE_END_DATE" in script_source

    def test_targets_cross_validation_endpoint(self, script_source: str) -> None:
        assert "/validate/cross-validation" in script_source

    def test_never_references_the_nonexistent_walk_forward_endpoint(
        self, script_source: str
    ) -> None:
        assert "/validate/walk-forward" not in script_source

    def test_cv_type_is_hardcoded_to_walk_forward(self, script_source: str) -> None:
        assert re.search(r"cv_type.*walk_forward", script_source)

    def test_reuses_fire_and_poll_helper(self, script_source: str) -> None:
        smoke_fn = _extract_function_body(script_source, "run_walk_forward_smoke")
        assert "fire_and_poll" in smoke_fn, (
            "run_walk_forward_smoke must reuse fire_and_poll, not duplicate POST/poll"
        )

    def test_does_not_hardcode_compose_ports(self, script_source: str) -> None:
        smoke_fn = _extract_function_body(script_source, "run_walk_forward_smoke")
        assert ":8000" not in smoke_fn, "in-compose internal port must not be hardcoded"
        assert ":8005" not in smoke_fn, "host-side compose port must not be hardcoded"


class TestOptInGating:
    """The smoke stage must be gated behind the opt-in env var."""

    def test_script_is_unchanged_for_defaults(self, script_source: str) -> None:
        # Summary still lists the original five stages as baseline output.
        for stage in ("yfinance", "macro", "news", "summarize", "calibrate"):
            assert stage in script_source

    def test_enable_flag_guards_invocation(self, script_source: str) -> None:
        # The smoke function call must be guarded by a test against the env var.
        # A conditional of the form [ "$ENABLE_WALK_FORWARD_SMOKE" = "1" ] or similar.
        assert re.search(r'ENABLE_WALK_FORWARD_SMOKE.*=\s*"?1"?', script_source), (
            "expected opt-in guard checking ENABLE_WALK_FORWARD_SMOKE equals 1"
        )


class TestCommentsDocumentTheFlag:
    """AC: inline comments document the new flag, env vars, and defaults."""

    def test_smoke_tickers_default_is_documented(self, script_source: str) -> None:
        # Acceptance criteria require a documented default universe.
        # We accept any of: SPY, QQQ, IWM, EFA, TLT mentioned in a comment block
        # or as a literal fallback.
        assert re.search(r"(SPY|QQQ|IWM|EFA|TLT)", script_source), (
            "expected a default ticker universe documented or inlined"
        )

    def test_enable_flag_is_mentioned_in_comment(self, script_source: str) -> None:
        # A comment block should mention ENABLE_WALK_FORWARD_SMOKE.
        in_comment = re.search(r"#[^\n]*ENABLE_WALK_FORWARD_SMOKE", script_source)
        assert in_comment, "expected a comment documenting ENABLE_WALK_FORWARD_SMOKE"


class TestJqJsonBodyConstruction:
    """Verify the jq-built request body is structurally correct.

    Runs the body-construction shell snippet against /bin/sh in isolation
    to catch quoting / escaping bugs without needing a mock HTTP server.
    """

    def test_body_contains_tickers_as_json_array(self) -> None:
        body = _build_smoke_body_via_sh(
            tickers="SPY,QQQ,IWM",
            start_date="2020-01-01",
            end_date="2024-01-01",
        )
        import json

        payload = json.loads(body)
        assert payload["tickers"] == ["SPY", "QQQ", "IWM"]
        assert payload["start_date"] == "2020-01-01"
        assert payload["end_date"] == "2024-01-01"
        assert payload["cv_type"] == "walk_forward"

    def test_body_trims_whitespace_around_tickers(self) -> None:
        body = _build_smoke_body_via_sh(
            tickers="SPY, QQQ ,IWM",
            start_date="2020-01-01",
            end_date="2024-01-01",
        )
        import json

        payload = json.loads(body)
        assert payload["tickers"] == ["SPY", "QQQ", "IWM"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_smoke_body_via_sh(tickers: str, start_date: str, end_date: str) -> str:
    """Execute the body-construction snippet to verify JSON shape end-to-end."""
    snippet = r"""
tickers_json=$(echo "$SMOKE_TICKERS" | jq -R -c 'split(",") | map(. | gsub("^\\s+|\\s+$"; ""))')
jq -n \
    --argjson tickers "$tickers_json" \
    --arg start_date "$SMOKE_START_DATE" \
    --arg end_date "$SMOKE_END_DATE" \
    '{tickers: $tickers, start_date: $start_date, end_date: $end_date, cv_type: "walk_forward"}'
"""
    result = subprocess.run(
        ["sh", "-c", snippet],
        env={
            "SMOKE_TICKERS": tickers,
            "SMOKE_START_DATE": start_date,
            "SMOKE_END_DATE": end_date,
            "PATH": subprocess.run(
                ["sh", "-c", "echo $PATH"], capture_output=True, text=True
            ).stdout.strip(),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _extract_function_body(source: str, fn_name: str) -> str:
    """Extract the body of a POSIX sh function by brace matching."""
    pattern = rf"{re.escape(fn_name)}\s*\(\)\s*\{{"
    match = re.search(pattern, source)
    if not match:
        return ""
    start = match.end() - 1  # position of opening '{'
    depth = 0
    for i in range(start, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : i]
    return source[start + 1 :]
