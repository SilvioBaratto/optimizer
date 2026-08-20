"""Shared filesystem-scan primitives for the hygiene guard suite (issue #15).

``_iter_python_files`` and ``_EXCLUDE_DIR_PARTS`` used to be copy-pasted
verbatim into ``test_synchronous_only.py`` and ``test_no_optimizer_import.py``
(and a third, differently-shaped variant in ``test_no_http_surface.py``,
which scans more than ``.py`` files and stays local to that guard). A single
shared definition means the excluded-directory set can only drift by an
explicit, reviewed change to this one file, not silently between two guards
that happened to diverge.

Leading-underscore module name: pytest's ``python_files = test_*.py`` glob
in ``ingestion/pytest.ini`` never collects it, so it is an ordinary import,
not a test module in its own right.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

_EXCLUDE_DIR_PARTS = {"__pycache__", "baml_client"}


def _iter_python_files(root: Path, *, exclude: Path | None = None) -> Iterator[Path]:
    """Yield every ``.py`` file under ``root``, skipping generated noise dirs.

    Args:
        root: Directory tree to walk (typically an ``app/``-shaped root).
        exclude: A resolved file path to skip — a guard passes its own
            ``__file__`` so a scan of the tree it lives under never
            self-matches the markers it exists to forbid.

    Yields:
        Each ``.py`` file path under ``root`` not inside an excluded
        directory and not equal to ``exclude``.
    """
    for path in root.rglob("*.py"):
        if _EXCLUDE_DIR_PARTS.intersection(path.parts):
            continue
        if exclude is not None and path.resolve() == exclude:
            continue
        yield path


def _iter_git_tracked_relpaths(root: Path) -> list[str] | None:
    """Return git-tracked relative paths under ``root``.

    Args:
        root: Directory to run ``git ls-files`` from.

    Returns:
        One relative path per tracked file, or ``None`` when ``git`` is
        absent from ``PATH`` or ``root`` is not a git checkout — callers
        fall back to a plain filesystem walk in that case, since the
        synthetic trees fail-injection tests scan are never git
        repositories themselves.
    """
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(  # noqa: S603
        [git, "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()
