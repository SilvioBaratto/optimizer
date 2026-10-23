"""Contract: the fund image ships every LLM provider SDK the wizard can select.

``portopt setup`` offers all ten providers, but only ``langchain-ollama`` (plus the
two SDKs ``deepagents`` drags in) is a hard dependency; the rest sit behind
optional extras. The runtime stage has no ``uv`` and ``/venv`` is copied from the
builder, so a provider extra missing at build time can never be added later and
the first model build fails with an ``ImportError`` inside the container.

The guard reads the builder's ``uv sync`` instruction and checks that the extras it
requests cover every provider extra declared in ``pyproject.toml``.

``Path(__file__).resolve().parents[3]`` resolves to the repo root
(unit -> tests -> fund -> optimizer), never ``Path.cwd()``.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOCKERFILE = _REPO_ROOT / "fund" / "Dockerfile"
_PYPROJECT = _REPO_ROOT / "fund" / "pyproject.toml"

_NON_PROVIDER_EXTRAS = {"test", "all-llm"}


def _uv_sync_instruction() -> str:
    """Return the builder's ``uv sync`` command with continuations collapsed."""
    text = _DOCKERFILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if "uv sync" in line:
            return line
    return ""


def _optional_dependencies() -> dict[str, list[str]]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _installed_extras(instruction: str) -> set[str]:
    """Resolve the extras ``uv sync`` installs, expanding aggregate extras."""
    optional = _optional_dependencies()
    if re.search(r"--all-extras\b", instruction):
        return set(optional)
    requested = set(re.findall(r"--extra[ =]([A-Za-z0-9_.-]+)", instruction))
    if "all-llm" in requested:
        for spec in optional["all-llm"]:
            match = re.search(r"\[(?P<extras>[^\]]+)\]", spec)
            if match:
                requested.update(p.strip() for p in match["extras"].split(","))
    return requested


def test_when_image_is_built_then_uv_sync_targets_the_fund_package() -> None:
    instruction = _uv_sync_instruction()
    assert "--package portopt-fund" in instruction


def test_when_image_is_built_then_every_provider_extra_is_installed() -> None:
    providers = set(_optional_dependencies()) - _NON_PROVIDER_EXTRAS
    installed = _installed_extras(_uv_sync_instruction())
    missing = providers - installed
    assert not missing, f"fund image omits provider extras: {sorted(missing)}"
