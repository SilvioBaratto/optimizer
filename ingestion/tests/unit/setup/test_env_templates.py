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
_GATING_KEYS = ["TRADING_212_API_KEY", "TRADING_212_SECRET_KEY", "FRED_API_KEY"]


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


@pytest.mark.parametrize(
    "dead_key", ["DATABASE_COMMAND_TIMEOUT", "TRADING_ECONOMICS_API_KEY"]
)
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


@pytest.mark.parametrize(
    "template", _TEMPLATES, ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_example_database_url_matches_the_code_default(template: Path) -> None:
    """The verbatim-copied URL reaches the compose Postgres, not a stub host."""
    default = Settings.model_fields["database_url"].default
    assert dotenv_values(template)["DATABASE_URL"] == default


def test_ingestion_template_describes_the_real_trading_212_gate() -> None:
    """Only `t212_annotate` skips without a key; the universe build never needs one."""
    text = _INGESTION_TEMPLATE.read_text(encoding="utf-8")
    assert "universe_build" not in text
    assert "t212_annotate" in text


# Every prose copy of the Trading 212 gate a reader can land on.
_GATE_COPIES = [
    "CLAUDE.md",
    "README.md",
    "ingestion/README.md",
    "ingestion/app/services/universe/trading212/client.py",
]

# "`universe_build` skips", "the universe-build step skips", "universe build skips".
_UNIVERSE_SKIP_CLAIM = re.compile(
    r"universe[_\s-]build[`*\s]*(?:(?:step|job)[`*\s]+)?skips?\b", re.IGNORECASE
)
_NOTHING_REFRESHES_CLAIM = re.compile(r"nothing\s+(?:will\s+)?refresh", re.IGNORECASE)


@pytest.mark.parametrize("relpath", _GATE_COPIES)
def test_docs_do_not_repeat_the_stale_universe_build_skip_claim(relpath: str) -> None:
    """Prose must match the template: T212 gates annotate only, never the build."""
    text = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert not _UNIVERSE_SKIP_CLAIM.search(text), relpath
    assert not _NOTHING_REFRESHES_CLAIM.search(text), relpath


@pytest.mark.parametrize("relpath", _GATE_COPIES)
def test_docs_name_t212_annotate_as_the_step_the_key_gates(relpath: str) -> None:
    """A copy that explains the missing-key skip must name the step that skips."""
    text = (_REPO_ROOT / relpath).read_text(encoding="utf-8")
    assert "t212_annotate" in text, relpath
