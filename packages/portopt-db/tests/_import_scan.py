"""Syntax-tree import scanning shared by the portopt-db boundary guards.

Leading-underscore module name: pytest never collects it, so it is an
ordinary import beside the guard tests that use it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator


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
        ``a.b`` for ``import a.b`` and ``from a.b import c``. Relative
        imports are skipped: they cannot leave the package being scanned.
    """
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
