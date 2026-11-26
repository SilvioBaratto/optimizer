"""Guard: development tooling that can reach the network stays out of the shipped set.

virtualenv and nodeenv download wheels and Node.js builds, virtualenv also asks the
package index about yanked releases (21.15.9 reads pip's index settings, ``PIP_*`` and
``~/.netrc`` for it), pyright's wrapper checks PyPI for a newer release, and pip,
pip-api, pip-audit and cachecontrol drive the index and its HTTP cache. They all reach
the lock through the ``dev`` tooling extras only (pre-commit, pyright, pip-audit).

The Docker images sync with ``--no-dev`` and a library install resolves only the
declared dependencies, so whatever the lock lets the runtime roots reach is what ships
and what can phone out from a user's machine. A release of one of these tools may add
an update check or a lookup, so the isolation is pinned here instead of assumed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pytest
import tomllib

_UV_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

# The roots as the images and a library install see them: `uv sync --no-dev --package
# portopt` (ingestion image), `--package portopt-fund --extra all-llm` (fund image),
# and the two library packages a consumer can install on their own.
_SHIPPED_ROOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("portopt", ()),
    ("portopt-fund", ("all-llm",)),
    ("portopt-db", ()),
    ("portopt-core", ()),
)

_NETWORK_CAPABLE_DEV_TOOLS = (
    "cachecontrol",
    "nodeenv",
    "pip",
    "pip-api",
    "pip-audit",
    "pre-commit",
    "pyright",
    "python-discovery",
    "virtualenv",
)


def _reachable(
    lock_text: str, roots: Iterable[tuple[str, tuple[str, ...]]]
) -> set[str]:
    """Return every package name the roots pull in, following the requested extras."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for package in tomllib.loads(lock_text)["package"]:
        by_name[package["name"]].append(package)
    seen: set[tuple[str, tuple[str, ...]]] = set()
    pending = list(roots)
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        name, extras = node
        for package in by_name.get(name, []):
            requirements = list(package.get("dependencies", []))
            for extra in extras:
                requirements += package.get("optional-dependencies", {}).get(extra, [])
            pending.extend(
                (req["name"], tuple(sorted(req.get("extra", []))))
                for req in requirements
            )
    return {name for name, _ in seen}


@pytest.fixture(scope="module")
def shipped_packages() -> set[str]:
    """The packages the shipped roots reach in the committed ``uv.lock``."""
    return _reachable(_UV_LOCK.read_text(encoding="utf-8"), _SHIPPED_ROOTS)


def test_every_shipped_root_is_locked() -> None:
    """A renamed or removed root would make the isolation check below vacuous."""
    locked = {
        package["name"]
        for package in tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))["package"]
    }
    missing = [name for name, _ in _SHIPPED_ROOTS if name not in locked]
    assert not missing, f"update _SHIPPED_ROOTS: {missing} are not in uv.lock"


@pytest.mark.parametrize("tool", _NETWORK_CAPABLE_DEV_TOOLS)
def test_network_capable_dev_tool_is_not_reachable_from_the_shipped_roots(
    tool: str, shipped_packages: set[str]
) -> None:
    """No runtime root reaches the tool, so it ships only through the dev extras."""
    assert tool not in shipped_packages, (
        f"{tool} can phone out and is now reachable from a shipped root; "
        "keep it in the dev/lint/typecheck extras"
    )


_SYNTHETIC_LOCK = """
[[package]]
name = "app"
version = "1"
dependencies = [{ name = "lib" }]

[package.optional-dependencies]
dev = [{ name = "virtualenv" }]
llm = [{ name = "sdk", extra = ["fast"] }]

[[package]]
name = "lib"
version = "1"

[[package]]
name = "sdk"
version = "1"

[package.optional-dependencies]
fast = [{ name = "fast-dep" }]

[[package]]
name = "fast-dep"
version = "1"

[[package]]
name = "virtualenv"
version = "1"
"""


def test_default_resolution_skips_the_dev_extra() -> None:
    """Without the extra, an optional dependency such as virtualenv is not reached."""
    assert _reachable(_SYNTHETIC_LOCK, [("app", ())]) == {"app", "lib"}


def test_requested_extra_is_followed() -> None:
    """The check catches a dev tool that a requested extra pulls in."""
    reached = _reachable(_SYNTHETIC_LOCK, [("app", ("dev",))])
    assert "virtualenv" in reached


def test_extra_requested_on_a_dependency_is_followed() -> None:
    """A dependency declared with extras brings those extras' own dependencies."""
    reached = _reachable(_SYNTHETIC_LOCK, [("app", ("llm",))])
    assert {"sdk", "fast-dep"} <= reached
