"""Fingerprints of the files each built image is made from.

``docker compose up`` builds an image only when its tag is missing, so after the
checkout is upgraded the old image keeps running old code, while ``portopt setup``
has already migrated the database to the new schema head. ``portopt start`` stamps
every build with :func:`source_hash` (the ``build.labels`` entry in
``docker-compose.yml`` reads it from :attr:`ImageSource.env`) and rebuilds any image
whose stamp no longer matches the checkout.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SOURCE_HASH_LABEL = "io.portopt.source-hash"


@dataclass(frozen=True)
class ImageSource:
    """What one built compose image is made from.

    Attributes:
        ref: The tag compose builds and runs.
        env: The environment variable ``docker-compose.yml`` interpolates into the
            image's :data:`SOURCE_HASH_LABEL`.
        inputs: Repo-relative files and directories the Dockerfile copies in.
    """

    ref: str
    env: str
    inputs: tuple[str, ...]


IMAGES = {
    "scheduler": ImageSource(
        ref="optimizer-scheduler:latest",
        env="PORTOPT_SCHEDULER_SOURCE_HASH",
        inputs=("pyproject.toml", "uv.lock", "packages/portopt-db", "ingestion"),
    ),
    "fund": ImageSource(
        ref="optimizer-fund:latest",
        env="PORTOPT_FUND_SOURCE_HASH",
        inputs=(
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "packages/portopt-core",
            "packages/portopt-db",
            "fund",
            "optimizerwiki",
        ),
    ),
}

# What `.dockerignore` and `fund/Dockerfile.dockerignore` keep out of the build
# context. Mirrored loosely: a name wrongly skipped here could leave a stale image
# running, so it lists only paths that can never reach an image, while a name wrongly
# kept only costs one cached rebuild.
_SKIPPED_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".github",
        ".venv",
        "venv",
        "node_modules",
        "tests",
        "build",
        "dist",
        "htmlcov",
        "secrets",
        ".certs",
        ".runtime",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
    }
)
_CHUNK_BYTES = 1 << 20


def _is_skipped_file(name: str) -> bool:
    if name.startswith(".env"):
        return name != ".env.example"
    return (
        name.endswith(".pyc") or name.startswith(".coverage") or name == "coverage.xml"
    )


def _files(root: Path) -> Iterator[Path]:
    """Yield the image-relevant files under ``root`` in a stable order."""
    if root.is_file():
        yield root
        return
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in _SKIPPED_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            if not _is_skipped_file(name):
                yield Path(current, name)


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                digest.update(chunk)
    except OSError:
        # Unreadable or vanished mid-walk: Docker could not copy it either, and a
        # failed fingerprint must never block `portopt start`.
        return b""
    return digest.digest()


def source_hash(repo: Path, image: str) -> str:
    """Return a hex fingerprint of the files ``image`` is built from.

    Args:
        repo: The checkout that holds ``docker-compose.yml``.
        image: A key of :data:`IMAGES` (``"scheduler"`` or ``"fund"``).

    Returns:
        A SHA-256 over the relative path and content of every file under the image's
        inputs, minus what the dockerignore files drop (tests, caches, ``.env``
        files, secrets). Inputs missing from ``repo`` contribute nothing, so a
        partial tree still hashes.

    Raises:
        KeyError: ``image`` is not a built image.
    """
    digest = hashlib.sha256()
    for rel in IMAGES[image].inputs:
        for path in _files(repo / rel):
            digest.update(path.relative_to(repo).as_posix().encode())
            digest.update(b"\0")
            digest.update(_file_digest(path))
    return digest.hexdigest()
