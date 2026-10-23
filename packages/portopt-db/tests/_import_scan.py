"""Import and manifest scanning shared by the portopt-db boundary guards.

Leading-underscore module name: pytest never collects it, so it is an
ordinary import beside the guard tests that use it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

import tomllib

_DYNAMIC_IMPORT_CALLEES = frozenset({"import_module", "__import__"})

# PEP 508: the distribution name is the leading token of a requirement string,
# before any extras, version specifier, environment marker or ``@ url``.
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def imported_module_names(source: str, filename: str = "<string>") -> Iterator[str]:
    """Yield the dotted module name of every absolute import in ``source``.

    Walking the syntax tree instead of regex-matching lines lets a guard see
    ``import os, optimizer`` and ``x = 1; import optimizer`` while ignoring a
    docstring that merely quotes an import statement.

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


def declared_distribution_names(pyproject_text: str) -> set[str]:
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
