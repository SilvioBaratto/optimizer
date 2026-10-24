"""Build-context hygiene: developer secrets must never reach an image layer.

Both service images build with the repo root as context and `COPY` whole member
trees (`ingestion/`, `fund/`, `packages/*`). Docker ignore patterns are anchored
at the context root and (unlike .gitignore) are not recursive, so a bare `.env`
only hides `./.env` and a developer's `ingestion/.env` is baked into the image and
auto-loaded at runtime by `Settings(env_file=".env")`. These tests evaluate the
ignore file BuildKit would really read, with Docker's matching rules, against the
paths a contributor's checkout can contain.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _compile(pattern: str) -> re.Pattern[str]:
    """Translate one ignore pattern to a regex using Docker's (moby) rules.

    `*` stays inside a path segment, `**` crosses segments (and may match zero of
    them), and the whole pattern is anchored at the context root.
    """
    pattern = pattern.strip().strip("/")
    out = ["^"]
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "*":
            if pattern[i + 1 : i + 2] == "*":
                while pattern[i + 1 : i + 2] == "*":
                    i += 1
                if pattern[i + 1 : i + 2] == "/":
                    i += 1
                out.append(".*" if i + 1 >= len(pattern) else "(.*/)?")
            else:
                out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    out.append("$")
    return re.compile("".join(out))


def _ignored(ignore_file: Path, path: str) -> bool:
    """Whether `path` is excluded from the build context by `ignore_file`.

    The last matching rule wins and a match on any parent directory excludes the
    whole subtree, mirroring Docker's `MatchesOrParentMatches`.
    """
    parts = path.split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    ignored = False
    for raw in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        regex = _compile(line[1:] if negate else line)
        if any(regex.match(c) for c in candidates):
            ignored = not negate
    return ignored


def _effective_ignore_file(service: str) -> Path:
    """The ignore file BuildKit reads for `service`'s compose build.

    `<dockerfile>.dockerignore` wins when present; otherwise only the
    `.dockerignore` at the context root applies. Per-directory ignore files
    inside the tree are never read.
    """
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    build = doc["services"][service]["build"]
    context = (_REPO_ROOT / build["context"]).resolve()
    dockerfile = context / build["dockerfile"]
    specific = dockerfile.with_name(dockerfile.name + ".dockerignore")
    return specific if specific.is_file() else context / ".dockerignore"


_SECRET_PATHS = [
    ".env",
    ".env.local",
    "ingestion/.env",
    "ingestion/.env.local",
    "fund/.env",
    "packages/portopt-db/.env",
    "packages/portopt-core/.env.production",
    "secrets/fred_api_key",
    "ingestion/secrets/fred_api_key",
    ".certs/corp-ca.pem",
    "ingestion/.certs/corp-ca.pem",
]


# Gitignored build/test output a real checkout accumulates inside the member
# trees the Dockerfiles COPY whole; none of it belongs in an image layer.
_DEVELOPER_ARTIFACT_PATHS = [
    "ingestion/build/lib/app/worker.py",
    "packages/portopt-db/build/lib/portopt_db/__init__.py",
    "ingestion/dist/portopt-0.1.0.tar.gz",
    "ingestion/.hypothesis/examples/0a1b2c",
    "fund/.hypothesis/unicode_data/15.0.0",
    "ingestion/.coverage",
    "ingestion/.coverage.host.1234.567890",
    "fund/.coverage",
    "packages/portopt-core/.coverage",
    "ingestion/coverage.xml",
    "fund/coverage.xml",
    "ingestion/htmlcov/index.html",
]


def test_matcher_follows_docker_anchoring_semantics(tmp_path: Path) -> None:
    """Guards the helper itself: a bare `.env` rule must not reach a nested file,
    otherwise the contract tests below could not see the leak."""
    rules = tmp_path / ".dockerignore"
    rules.write_text(".env\n**/secrets/\n!**/.env.example\n", encoding="utf-8")
    assert _ignored(rules, ".env")
    assert not _ignored(rules, "ingestion/.env")
    assert _ignored(rules, "ingestion/secrets/key")
    assert not _ignored(rules, "ingestion/.env.example")


@pytest.mark.parametrize("path", _SECRET_PATHS)
def test_scheduler_build_context_excludes_developer_secrets(path: str) -> None:
    """`COPY ingestion/ ingestion/` must not carry a developer's real keys, or
    they land in an image layer and shadow the encrypted-store configuration."""
    assert _ignored(_effective_ignore_file("scheduler"), path)


@pytest.mark.parametrize("path", _SECRET_PATHS)
def test_fund_build_context_excludes_developer_secrets(path: str) -> None:
    """`COPY fund/ fund/` and `COPY packages/*` carry the same exposure."""
    assert _ignored(_effective_ignore_file("fund"), path)


@pytest.mark.parametrize("path", _DEVELOPER_ARTIFACT_PATHS)
def test_scheduler_build_context_excludes_developer_artifacts(path: str) -> None:
    """`COPY ingestion/ ingestion/` would otherwise ship a contributor's stale
    build output and coverage/hypothesis state into the image."""
    assert _ignored(_effective_ignore_file("scheduler"), path)


@pytest.mark.parametrize("path", _DEVELOPER_ARTIFACT_PATHS)
def test_fund_build_context_excludes_developer_artifacts(path: str) -> None:
    """`COPY fund/ fund/` and `COPY packages/*` carry the same developer state."""
    assert _ignored(_effective_ignore_file("fund"), path)


def test_scheduler_build_context_keeps_runtime_sources_and_env_template() -> None:
    """The fix must not starve the image: the daemon sources, entrypoint, Alembic
    tree and the documented `.env.example` template all stay in the context."""
    rules = _effective_ignore_file("scheduler")
    for path in (
        "ingestion/app/worker.py",
        "ingestion/entrypoint.sh",
        "ingestion/pyproject.toml",
        "ingestion/.env.example",
        "packages/portopt-db/alembic/env.py",
        "packages/portopt-db/alembic.ini",
    ):
        assert not _ignored(rules, path), path


def test_fund_build_context_keeps_runtime_knowledge() -> None:
    """The fund agent reads the wiki and skill markdown at runtime."""
    rules = _effective_ignore_file("fund")
    for path in (
        "fund/src/fund/worker.py",
        "fund/src/fund/skills/optimization-objective-map/SKILL.md",
        "optimizerwiki/docs/index.md",
    ):
        assert not _ignored(rules, path), path


def test_no_unread_nested_dockerignore_gives_false_assurance() -> None:
    """Docker never reads a `.dockerignore` below the context root, so a nested one
    only looks protective. The scheduler builds from the repo root."""
    assert not (_REPO_ROOT / "ingestion" / ".dockerignore").exists()
