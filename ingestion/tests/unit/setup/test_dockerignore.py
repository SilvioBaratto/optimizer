"""Build-context ignore contract for the compose-built service images.

Both images build from the repo root (`context: .`) and `COPY` whole member trees,
so anything the effective ignore file fails to drop lands in an image layer. BuildKit
reads `<dockerfile>.dockerignore` when it exists and the context-root `.dockerignore`
otherwise — a `.dockerignore` placed next to the Dockerfile is never read, and
patterns without a `**/` prefix only match at the context root. The matcher below
mirrors Docker's pattern semantics so a regression is caught without a daemon.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"

_SERVICES = ["scheduler", "fund"]

_NESTED_SECRETS_AND_CACHES = [
    "ingestion/.env",
    "ingestion/.env.local",
    "fund/.env",
    "secrets/fred_api_key",
    ".certs/corporate-root.pem",
    "fund/secrets/openai_api_key",
    "ingestion/secrets/fred_api_key",
    "fund/.certs/corporate-root.pem",
    "packages/portopt-db/.certs/corporate-root.pem",
    "ingestion/app/__pycache__/worker.cpython-312.pyc",
    "packages/portopt-db/src/portopt_db/__pycache__/models.cpython-312.pyc",
    "ingestion/.coverage",
    "ingestion/coverage.xml",
    "ingestion/.hypothesis/unicode_data/x",
    "ingestion/.pytest_cache/v/cache/lastfailed",
    "ingestion/.ruff_cache/CACHEDIR.TAG",
    "packages/portopt-db/portopt_db.egg-info/PKG-INFO",
]


def _compile(pattern: str) -> re.Pattern[str]:
    """Translate one Docker ignore pattern to an anchored regex.

    `*` and `?` stay inside a path segment; `**` spans segments, and `**/` also
    matches zero directories (so `**/.env` matches a root-level `.env`).
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i + 1 : i + 2] == "*":
                i += 1
                if pattern[i + 1 : i + 2] == "/":
                    i += 1
                    out.append("(.*/)?")
                else:
                    out.append(".*")
            else:
                out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            end = pattern.index("]", i)
            out.append(pattern[i : end + 1])
            i = end
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _excluded(patterns: list[str], rel_path: str) -> bool:
    """Whether Docker drops `rel_path` from the context given ordered `patterns`.

    A pattern matches the path itself or any parent directory; the last matching
    pattern decides, and a leading `!` re-includes.
    """
    parts = rel_path.split("/")
    prefixes = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for raw in patterns:
        negate = raw.startswith("!")
        body = raw[1:] if negate else raw
        regex = _compile(body.strip("/"))
        if any(regex.match(p) for p in prefixes):
            excluded = not negate
    return excluded


def _effective_ignore_patterns(service: str) -> list[str]:
    """Patterns BuildKit applies to `service`'s build, per compose `build:`."""
    build = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"][service][
        "build"
    ]
    context = (_REPO_ROOT / build["context"]).resolve()
    dockerfile = context / build["dockerfile"]
    sibling = dockerfile.with_name(dockerfile.name + ".dockerignore")
    ignore_file = sibling if sibling.is_file() else context / ".dockerignore"
    return [
        line.strip()
        for line in ignore_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("service", _SERVICES)
@pytest.mark.parametrize("rel_path", _NESTED_SECRETS_AND_CACHES)
def test_effective_ignore_excludes_nested_secrets_and_caches(
    service: str, rel_path: str
) -> None:
    """A nested `.env`, rendered secret or cache never reaches the build context."""
    assert _excluded(_effective_ignore_patterns(service), rel_path), (
        f"{rel_path} would be sent to the {service} builder"
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "pyproject.toml",
        "uv.lock",
        "ingestion/pyproject.toml",
        "ingestion/entrypoint.sh",
        "ingestion/app/worker.py",
        "packages/portopt-db/README.md",
        "packages/portopt-db/alembic.ini",
        "packages/portopt-db/alembic/env.py",
        "packages/portopt-db/src/portopt_db/__init__.py",
    ],
)
def test_scheduler_build_inputs_stay_in_context(rel_path: str) -> None:
    """Tightening the ignore rules must not drop files the Dockerfile COPYs."""
    assert not _excluded(_effective_ignore_patterns("scheduler"), rel_path)


@pytest.mark.parametrize(
    "rel_path",
    [
        "pyproject.toml",
        "README.md",
        "uv.lock",
        "fund/pyproject.toml",
        "fund/entrypoint.sh",
        "fund/src/fund/__init__.py",
        "optimizerwiki/index.md",
        "packages/portopt-core/optimizer/__init__.py",
        "packages/portopt-db/README.md",
    ],
)
def test_fund_build_inputs_stay_in_context(rel_path: str) -> None:
    """The fund image still receives its markdown skills, wiki and workspace members."""
    assert not _excluded(_effective_ignore_patterns("fund"), rel_path)
