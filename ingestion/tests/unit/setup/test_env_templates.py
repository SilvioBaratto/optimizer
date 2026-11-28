"""Contract for the two checked-in env templates (`.env.example`).

A user who copies a template verbatim must get the behaviour the template
describes: every active key is one the code actually reads, a key whose absence
is documented to skip work is not pre-filled with a truthy placeholder, and the
example `DATABASE_URL` points at the same database the code and compose default to.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

from app.config import Settings

# unit -> setup -> tests -> ingestion -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROOT_TEMPLATE = _REPO_ROOT / ".env.example"
_INGESTION_TEMPLATE = _REPO_ROOT / "ingestion" / ".env.example"
_TEMPLATES = [_ROOT_TEMPLATE, _INGESTION_TEMPLATE]

_SOURCE_TREES = [
    _REPO_ROOT / "ingestion" / "app",
    _REPO_ROOT / "fund" / "src",
    _REPO_ROOT / "packages" / "portopt-db" / "src",
]
_QUOTED_ENV_NAME = re.compile(r"""["']([A-Z][A-Z0-9_]+)["']""")

# Keys whose absence is a documented state (skip / clear "key required" error), so
# a truthy placeholder would defeat the gate and fail later with an opaque 401.
_GATING_KEYS = ["FRED_API_KEY"]


def _read_names() -> set[str]:
    """Env var names the code reads: Settings fields plus quoted literals."""
    names: set[str] = set()
    for field_name, field in Settings.model_fields.items():
        names.add(field_name.upper())
        if isinstance(field.alias, str):
            names.add(field.alias.upper())
    for tree in _SOURCE_TREES:
        for path in tree.rglob("*.py"):
            names.update(_QUOTED_ENV_NAME.findall(path.read_text(encoding="utf-8")))
    return names


@pytest.mark.parametrize(
    "template", _TEMPLATES, ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_every_active_key_is_read_by_the_code(template: Path) -> None:
    """A documented variable that nothing reads misleads whoever sets it."""
    unread = sorted(set(dotenv_values(template)) - _read_names())
    assert not unread, f"{template.name} sets variables no code reads: {unread}"


@pytest.mark.parametrize("dead_key", ["DATABASE_COMMAND_TIMEOUT"])
def test_removed_settings_are_not_documented_even_in_comments(dead_key: str) -> None:
    """Dead names stay out of the template entirely, commented or not."""
    assert dead_key not in _ROOT_TEMPLATE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "template", _TEMPLATES, ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
@pytest.mark.parametrize("key", _GATING_KEYS)
def test_gating_keys_are_not_prefilled_with_a_truthy_placeholder(
    template: Path, key: str
) -> None:
    """Copying the template verbatim must leave these keys unset, not `your-...`."""
    assert not dotenv_values(template).get(key)


# LLM, Azure and AWS entries whose example value is a stand-in the operator must
# replace. The host CLI loads the checkout's `.env`, and the setup wizard treats
# any value it finds there as the operator's own: it skips the prompt, so a
# `your-resource...` endpoint is stored (or fails validation) instead of asked for.
_LLM_PLACEHOLDER_KEYS = [
    "OLLAMA_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "AWS_REGION",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
]


@pytest.mark.parametrize("key", _LLM_PLACEHOLDER_KEYS)
def test_llm_placeholders_are_not_active_in_the_root_template(key: str) -> None:
    """Copying the template verbatim must not pre-answer the wizard's prompts."""
    assert key not in dotenv_values(_ROOT_TEMPLATE)


@pytest.mark.parametrize("key", _LLM_PLACEHOLDER_KEYS)
def test_llm_placeholders_stay_documented_as_comments(key: str) -> None:
    """Commented, not deleted: the template is where an operator finds the names."""
    assert re.search(rf"^#\s*{key}=", _ROOT_TEMPLATE.read_text("utf-8"), re.MULTILINE)


def test_root_template_has_no_active_placeholder_value() -> None:
    """Nothing active may be a `your-...` stand-in, whatever its key."""
    placeholders = {
        key: value
        for key, value in dotenv_values(_ROOT_TEMPLATE).items()
        if value and "your-" in value
    }
    assert not placeholders


@pytest.mark.parametrize(
    "template", _TEMPLATES, ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_example_database_url_matches_the_code_default(template: Path) -> None:
    """The verbatim-copied URL reaches the compose Postgres, not a stub host."""
    default = Settings.model_fields["database_url"].default
    assert dotenv_values(template)["DATABASE_URL"] == default


# Settings fields that are labels or developer toggles, not operator knobs.
_NOT_KNOBS = {"project_name", "version", "debug"}
# Read with os.getenv rather than through Settings (the auto-migrate opt-out by
# portopt_db.migration_state, in the entrypoint and on the host).
_EXTRA_KNOBS = {"PORTOPT_SECRETS_DIR": "/run/secrets", "PORTOPT_AUTO_MIGRATE": "false"}
# Knobs whose names and defaults are documented in the files mapped to them until
# the owner, who alone edits the env templates, lists them in the template. The
# checks below fail once the template names one, so an entry cannot outlive its
# reason.
_DB_BREAKER_DOCS = (_REPO_ROOT / "CLAUDE.md",)
_EXTERNAL_CALL_DOCS = (_REPO_ROOT / "CHANGELOG.md",)
_DOCUMENTED_OUTSIDE_THE_TEMPLATE: dict[str, tuple[Path, ...]] = {
    "DATABASE_BREAKER_FAILURE_THRESHOLD": _DB_BREAKER_DOCS,
    "DATABASE_BREAKER_RESET_TIMEOUT_S": _DB_BREAKER_DOCS,
    "BREAKER_FRED_MAX_ATTEMPTS": _EXTERNAL_CALL_DOCS,
    "BREAKER_YFINANCE_MAX_ATTEMPTS": _EXTERNAL_CALL_DOCS,
    "FRED_REQUEST_TIMEOUT_SECONDS": _EXTERNAL_CALL_DOCS,
    "FRED_REQUEST_ATTEMPTS": _EXTERNAL_CALL_DOCS,
    "FRED_RETRY_BASE_DELAY_SECONDS": _EXTERNAL_CALL_DOCS,
    "FRED_RETRY_MAX_DELAY_SECONDS": _EXTERNAL_CALL_DOCS,
}
_EXEMPTION_DOCS = tuple(
    dict.fromkeys(
        doc for docs in _DOCUMENTED_OUTSIDE_THE_TEMPLATE.values() for doc in docs
    )
)

# A `NAME=value` line, active or commented out.
_ASSIGNMENT = re.compile(r"^#?[ \t]*([A-Z][A-Z0-9_]*)=(.*)$", re.MULTILINE)


def _documented_knobs() -> dict[str, str]:
    """Name -> value of every assignment in the ingestion template, comments included."""
    text = _INGESTION_TEMPLATE.read_text(encoding="utf-8")
    return {
        name: value.strip().strip("\"'") for name, value in _ASSIGNMENT.findall(text)
    }


def _setting_knobs() -> dict[str, object]:
    """Env name -> code default for every operator-facing Settings field."""
    knobs: dict[str, object] = {}
    for field_name, field in Settings.model_fields.items():
        if field_name in _NOT_KNOBS:
            continue
        env_name = field.alias if isinstance(field.alias, str) else field_name.upper()
        knobs[env_name] = field.default
    return knobs


def test_ingestion_template_lists_every_setting_the_daemon_reads() -> None:
    """CLAUDE.md sends operators here for the full list, SCHEDULER_* crons included."""
    documented = set(_documented_knobs()) | set(_DOCUMENTED_OUTSIDE_THE_TEMPLATE)
    missing = sorted((set(_setting_knobs()) | set(_EXTRA_KNOBS)) - documented)
    assert not missing, f"ingestion/.env.example does not mention: {missing}"


def test_ingestion_template_defaults_match_the_code() -> None:
    """A documented default that drifted from config.py misleads whoever relies on it."""
    documented = _documented_knobs()
    defaults = {**_setting_knobs(), **_EXTRA_KNOBS}
    drifted = {
        name: (documented[name], default)
        for name, default in defaults.items()
        if name in documented
        and default is not None
        and documented[name].lower() != str(default).lower()
    }
    assert not drifted, f"template value vs code default: {drifted}"


def _exempt_names_in(text: str) -> list[str]:
    return sorted(name for name in _DOCUMENTED_OUTSIDE_THE_TEMPLATE if name in text)


def test_the_template_exemption_retires_once_the_template_names_a_variable() -> None:
    """A name the template now carries must leave ``_DOCUMENTED_OUTSIDE_THE_TEMPLATE``."""
    listed = _exempt_names_in(_INGESTION_TEMPLATE.read_text(encoding="utf-8"))
    assert not listed, (
        f"ingestion/.env.example now names {listed}: remove them from "
        "_DOCUMENTED_OUTSIDE_THE_TEMPLATE"
    )


def test_the_retirement_check_notices_a_named_variable() -> None:
    """Guard on the guard: a template line, commented out or not, is noticed."""
    for name in _DOCUMENTED_OUTSIDE_THE_TEMPLATE:
        assert _exempt_names_in(f"# {name}=5\n") == [name]


@pytest.mark.parametrize("doc", _EXEMPTION_DOCS, ids=lambda p: p.name)
def test_an_exempt_variable_stays_documented_while_exempt(doc: Path) -> None:
    """While the template lacks them, a file mapped to a name must name it."""
    text = doc.read_text(encoding="utf-8")
    expected = {
        name for name, docs in _DOCUMENTED_OUTSIDE_THE_TEMPLATE.items() if doc in docs
    }
    missing = sorted(expected - set(_exempt_names_in(text)))
    assert not missing, f"{doc.name} no longer names {missing}"
