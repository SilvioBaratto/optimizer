"""Static scan of ``portopt_db`` for ``_upsert`` calls that name a conflict constraint.

Leading-underscore module name: pytest never collects it, so it is an ordinary
import beside the guard tests that use it.

``_upsert`` builds ``ON CONFLICT ON CONSTRAINT <name>``. SQLite compiles that to a
bare ``ON CONFLICT DO UPDATE`` and drops the name, so a misspelled constraint is
invisible to every SQLite-backed test and only fails on PostgreSQL at runtime.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint

# ``_upsert_estimates`` wraps ``_upsert`` and forwards its own ``constraint_name``
# parameter, so the literal names live at the wrapper's callers.
_UPSERT_METHODS = frozenset({"_upsert", "_upsert_estimates"})

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src"
_SOURCE = _PACKAGE_ROOT / "portopt_db"


@dataclass(frozen=True)
class UpsertSite:
    """One ``self._upsert(Model, rows, constraint_name="...")`` call."""

    file: Path
    line: int
    module: str
    model_name: str
    constraint_name: str


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(_PACKAGE_ROOT).with_suffix("").parts)


def _site_from_call(path: Path, call: ast.Call) -> UpsertSite | None:
    constraint = next(
        (k.value for k in call.keywords if k.arg == "constraint_name"), None
    )
    if constraint is None:
        return None
    location = f"{path}:{call.lineno}"
    if not (isinstance(constraint, ast.Constant) and isinstance(constraint.value, str)):
        raise AssertionError(f"{location}: constraint_name must be a string literal")
    if not (call.args and isinstance(call.args[0], ast.Name)):
        raise AssertionError(f"{location}: the model must be passed as a plain name")
    return UpsertSite(
        file=path,
        line=call.lineno,
        module=_module_name(path),
        model_name=call.args[0].id,
        constraint_name=constraint.value,
    )


def _forwarding_calls(tree: ast.AST) -> set[int]:
    """Ids of calls that pass on the enclosing function's ``constraint_name`` parameter."""
    forwarded: set[int] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        parameters = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
        if "constraint_name" not in parameters:
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "constraint_name"
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "constraint_name"
                ):
                    forwarded.add(id(node))
    return forwarded


def collect_upsert_sites(root: Path = _SOURCE) -> list[UpsertSite]:
    """Return every ``_upsert`` call under ``root`` that passes ``constraint_name``.

    Args:
        root: Directory scanned recursively for ``*.py`` files.

    Returns:
        The call sites in file and line order.

    Raises:
        AssertionError: A call passes a non-literal constraint name or a model that
            is not a plain name, so the scan could not check it and must not skip it.
    """
    sites: list[UpsertSite] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forwarded = _forwarding_calls(tree)
        for node in ast.walk(tree):
            is_upsert = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _UPSERT_METHODS
            )
            if is_upsert and isinstance(node, ast.Call) and id(node) not in forwarded:
                site = _site_from_call(path, node)
                if site is not None:
                    sites.append(site)
    return sites


def unique_constraint_names(model: type) -> set[str]:
    """Names PostgreSQL accepts after ``ON CONFLICT ON CONSTRAINT`` for ``model``.

    Only unique and primary-key constraints qualify; a bare unique ``Index`` does
    not, even though it enforces the same uniqueness.
    """
    table = model.__table__  # type: ignore[attr-defined]
    return {
        constraint.name
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint | PrimaryKeyConstraint)
        and constraint.name is not None
    }


def constraint_problems(sites: list[UpsertSite]) -> list[str]:
    """Describe each site whose constraint name its model does not declare."""
    problems: list[str] = []
    for site in sites:
        model = getattr(importlib.import_module(site.module), site.model_name)
        known = unique_constraint_names(model)
        if site.constraint_name not in known:
            problems.append(
                f"{site.file.name}:{site.line} {site.model_name} has no unique or "
                f"primary-key constraint {site.constraint_name!r} (declares "
                f"{sorted(known)})"
            )
    return problems
