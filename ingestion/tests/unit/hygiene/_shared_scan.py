"""Shared filesystem-scan primitives for the hygiene guard suite.

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

import ast
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import tomllib

# "build" catches ``ingestion/build/`` — a gitignored setuptools/``uv build``
# artifact (``uv tool install --from ./ingestion`` during onboarding creates it)
# that duplicates the source tree and would otherwise false-positive every scan.
_EXCLUDE_DIR_PARTS = {"__pycache__", "build"}

_DYNAMIC_IMPORT_CALLEES = frozenset({"import_module", "__import__"})

# PEP 508: the distribution name is the leading token of a requirement string,
# before any extras, version specifier, environment marker or ``@ url``.
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _is_in_excluded_dir(path: Path, root: Path) -> bool:
    """Return whether ``path`` sits in a noise directory *below* ``root``.

    Only the part of the path under ``root`` is inspected: a checkout that
    itself lives beneath a ``build`` directory (``/home/ci/build/<repo>``,
    a Docker ``WORKDIR /build``) must not make every guard scan zero files.

    Args:
        path: A file found by walking ``root``.
        root: The directory the walk started from.

    Returns:
        ``True`` when a directory component between ``root`` and the file is
        in ``_EXCLUDE_DIR_PARTS``.
    """
    return not _EXCLUDE_DIR_PARTS.isdisjoint(path.relative_to(root).parent.parts)


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
        if _is_in_excluded_dir(path, root):
            continue
        if exclude is not None and path.resolve() == exclude:
            continue
        yield path


def _imported_module_names(source: str, filename: str = "<string>") -> Iterator[str]:
    """Yield the dotted module name of every absolute import in ``source``.

    Parsing the syntax tree instead of regex-matching lines is what lets a
    guard see ``import os, optimizer``, ``x = 1; import optimizer`` and
    ``from  optimizer import y`` while ignoring a docstring that merely quotes
    an import statement.

    Args:
        source: Python source text.
        filename: Name reported by ``SyntaxError`` so an unparsable file is
            identifiable; a guard must fail loudly on it, not skip it.

    Yields:
        ``a.b`` for ``import a.b`` and ``from a.b import c``, plus the
        string-literal name handed to ``importlib.import_module`` /
        ``__import__``. Relative imports are skipped: they cannot leave the
        package being scanned. A computed (non-literal) dynamic import name
        cannot be resolved statically and is not reported.
    """
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else None
            if isinstance(callee, ast.Name):
                name = callee.id
            target = node.args[0]
            if (
                name in _DYNAMIC_IMPORT_CALLEES
                and isinstance(target, ast.Constant)
                and isinstance(target.value, str)
                and not target.value.startswith(".")
            ):
                yield target.value


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


def _declared_distribution_names(pyproject_text: str) -> set[str]:
    """Return the normalized distribution names a ``pyproject.toml`` requires.

    Reading the parsed manifest instead of substring-matching its text means a
    dependency spelled ``Portopt_Core[x]>=1`` is seen as ``portopt-core``, while
    a comment, description or lookalike such as ``portopt-core-docs`` is not
    mistaken for the distribution a boundary guard forbids.

    Args:
        pyproject_text: Raw contents of a ``pyproject.toml``.

    Returns:
        PEP 503-normalized names (lowercase, runs of ``-_.`` collapsed to
        ``-``) from ``project.dependencies``, every ``project.optional-
        dependencies`` extra, every ``dependency-groups`` group,
        ``tool.uv.dev-dependencies`` and ``build-system.requires``.

    Raises:
        tomllib.TOMLDecodeError: ``pyproject_text`` is not valid TOML — a guard
            must fail loudly on a manifest it cannot read.
    """
    data = tomllib.loads(pyproject_text)
    project = data.get("project", {})
    requirement_lists: list[list[object]] = [
        project.get("dependencies", []),
        *project.get("optional-dependencies", {}).values(),
        *data.get("dependency-groups", {}).values(),
        data.get("tool", {}).get("uv", {}).get("dev-dependencies", []),
        data.get("build-system", {}).get("requires", []),
    ]
    names: set[str] = set()
    for requirements in requirement_lists:
        for requirement in requirements:
            # dependency-groups also holds ``{include-group = "..."}`` tables.
            if not isinstance(requirement, str):
                continue
            match = _REQUIREMENT_NAME.match(requirement)
            if match:
                names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names
