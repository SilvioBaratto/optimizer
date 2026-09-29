"""Contract: the ``all-llm`` aggregate extra fans out to every provider extra.

``all-llm`` is the single extra that installs *every* optional
LLM backend at once — it is what CI's ``uv sync --all-extras`` pulls in. It is
declared as a self-referential extra (``portopt-fund[openai,anthropic,...]``) so
each provider's package list stays defined in exactly one place (its own extra).

This guard fails the build if a future provider extra is added to
``[project.optional-dependencies]`` without being wired into ``all-llm`` — the
exact regression that would let ``--all-extras`` silently skip a backend.

``Path(__file__).resolve().parents[2]`` resolves to the ``fund/`` directory
(unit → tests → fund), never ``Path.cwd()``.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

_FUND_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_FILE = _FUND_ROOT / "pyproject.toml"

# Optional-dependency groups that are NOT provider backends and so are not part
# of the ``all-llm`` fan-out.
_NON_PROVIDER_EXTRAS = {"test", "all-llm"}


def _load_optional_dependencies() -> dict[str, list[str]]:
    data = tomllib.loads(_PYPROJECT_FILE.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _provider_extra_names(optional: dict[str, list[str]]) -> set[str]:
    """Every optional-dependency group that is a provider backend."""
    return set(optional) - _NON_PROVIDER_EXTRAS


def _aggregated_extra_names(all_llm_specs: list[str]) -> set[str]:
    """Extract the extra names a self-referential aggregate references.

    e.g. ``["portopt-fund[openai,anthropic]"]`` -> ``{"openai", "anthropic"}``.
    """
    names: set[str] = set()
    for spec in all_llm_specs:
        match = re.search(r"\[(?P<extras>[^\]]+)\]", spec)
        if match:
            names.update(part.strip() for part in match["extras"].split(","))
    return names


def test_when_pyproject_is_read_then_all_llm_extra_is_declared():
    optional = _load_optional_dependencies()
    assert "all-llm" in optional


def test_when_all_llm_is_read_then_it_aggregates_every_provider_extra():
    optional = _load_optional_dependencies()
    providers = _provider_extra_names(optional)
    aggregated = _aggregated_extra_names(optional["all-llm"])
    assert aggregated == providers


def test_when_all_llm_is_read_then_every_ref_targets_the_fund_distribution():
    optional = _load_optional_dependencies()
    assert all(spec.startswith("portopt-fund[") for spec in optional["all-llm"])
