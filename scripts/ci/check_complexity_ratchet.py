"""Hold function length and cyclomatic complexity to a baseline that only shrinks.

Standard 2.8 targets at most 30 lines per function and a cyclomatic complexity below
10. The shipped code predates the rule, so today's offenders are recorded in
``scripts/ci/baselines/long_functions.json`` and ``complex_functions.json``. The check
fails a function that is over the limit and not in its baseline, and a baselined
function that has grown. Fixing an offender needs no edit here: ``--update`` lowers
its stored value or drops the entry, and it can never raise a value or add an entry,
so the debt can only go down.

Usage: ``check_complexity_ratchet.py [--update]``. Keys are ``path::qualname``, so an
entry does not move when other lines of the file change. A long function is measured
in body lines without the docstring; a complex one by ruff's C901 score. Standard
library plus ``python -m ruff``, which the lint job already installs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    "packages/portopt-core/optimizer",
    "packages/portopt-db/src",
    "ingestion/app",
    "fund/src",
)
MAX_LINES = 30
# Ruff's `max-complexity` is the highest score still allowed; the standard wants
# a score below 10.
MAX_COMPLEXITY = 9

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_TOO_COMPLEX = re.compile(r"is too complex \((\d+) > \d+\)")


@dataclass(frozen=True)
class Layout:
    """Where the ratchet reads sources and keeps its baselines.

    Attributes:
        base: Directory the ``path`` half of every key is relative to.
        roots: Source trees to scan.
        baseline_dir: Directory holding ``long_functions.json`` and
            ``complex_functions.json``.
    """

    base: Path
    roots: tuple[Path, ...]
    baseline_dir: Path


DEFAULT_LAYOUT = Layout(
    base=REPO_ROOT,
    roots=tuple(REPO_ROOT / root for root in SOURCE_ROOTS),
    baseline_dir=REPO_ROOT / "scripts" / "ci" / "baselines",
)


@dataclass(frozen=True)
class Metric:
    """One thing the ratchet measures.

    Attributes:
        baseline_file: Name of the baseline file inside the layout's directory.
        unit: Word that follows a value in a violation message.
        limit: Highest value that is still acceptable.
    """

    baseline_file: str
    unit: str
    limit: int


LONG = Metric("long_functions.json", "lines", MAX_LINES)
COMPLEX = Metric("complex_functions.json", "complexity", MAX_COMPLEXITY)


def _python_files(roots: Iterable[Path]) -> list[Path]:
    return sorted(path for root in roots for path in root.rglob("*.py"))


def _key(path: Path, base: Path, qualname: str) -> str:
    return f"{path.resolve().relative_to(base.resolve()).as_posix()}::{qualname}"


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        # Skipping the file would let its offenders through unmeasured.
        raise SystemExit(f"cannot parse {path}: {error}") from error


def _functions(tree: ast.Module) -> list[tuple[str, _FunctionNode]]:
    """Every function in a module with its qualified name, nested ones included."""
    found: list[tuple[str, _FunctionNode]] = []
    pending: list[tuple[ast.AST, tuple[str, ...]]] = [(tree, ())]
    while pending:
        parent, scope = pending.pop()
        for child in ast.iter_child_nodes(parent):
            inner = scope
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                inner = (*scope, child.name)
                found.append((".".join(inner), child))
            elif isinstance(child, ast.ClassDef):
                inner = (*scope, child.name)
            pending.append((child, inner))
    return found


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _body_lines(function: _FunctionNode) -> int:
    """Lines from the first statement after the docstring to the function's end."""
    body = function.body[1:] if _is_docstring(function.body[0]) else function.body
    if not body:
        return 0
    return (function.end_lineno or body[0].lineno) - body[0].lineno + 1


def _keep_largest(sizes: dict[str, int], key: str, value: int) -> None:
    # A property getter and setter, or an `if`/`else` pair of definitions, share a
    # qualified name; the larger one is the one that must not grow.
    sizes[key] = max(value, sizes.get(key, 0))


def find_long_functions(
    roots: Iterable[Path], limit: int = MAX_LINES, *, base: Path = REPO_ROOT
) -> dict[str, int]:
    """Measure the functions whose body is longer than ``limit`` lines.

    Args:
        roots: Source trees to scan for ``*.py`` files.
        limit: Longest body, in lines, that is still acceptable.
        base: Directory the path in each key is relative to.

    Returns:
        ``path::qualname`` mapped to the body line count without the docstring,
        for each function over ``limit``. Nested functions count in their parent too.

    Raises:
        SystemExit: A source file cannot be parsed.
    """
    sizes: dict[str, int] = {}
    for path in _python_files(roots):
        for qualname, function in _functions(_parse(path)):
            lines = _body_lines(function)
            if lines > limit:
                _keep_largest(sizes, _key(path, base, qualname), lines)
    return sizes


def _ruff_c901(roots: Sequence[Path]) -> list[dict[str, Any]]:
    # `--ignore-noqa` so a suppression comment cannot hide an offender from the ratchet.
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--isolated",
        "--select",
        "C901",
        "--ignore-noqa",
        "--config",
        f"lint.mccabe.max-complexity={MAX_COMPLEXITY}",
        "--output-format",
        "json",
        *map(str, roots),
    ]
    # Ruff exits 1 whenever it reports a diagnostic, which is the normal case here.
    done = subprocess.run(  # noqa: S603 - argv is built here, not from user input
        command, capture_output=True, text=True, check=False
    )
    try:
        report: list[dict[str, Any]] = json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"ruff produced no JSON report: {done.stderr.strip()}"
        ) from error
    return report


def _qualname_at(path: Path, row: int, cache: dict[Path, dict[int, str]]) -> str:
    if path not in cache:
        cache[path] = {fn.lineno: name for name, fn in _functions(_parse(path))}
    try:
        return cache[path][row]
    except KeyError:
        raise SystemExit(
            f"ruff flagged {path}:{row}, which starts no function"
        ) from None


def find_complex_functions(
    roots: Sequence[Path], *, base: Path = REPO_ROOT
) -> dict[str, int]:
    """Measure the functions ruff's C901 finds too complex.

    Args:
        roots: Source trees to hand to ruff.
        base: Directory the path in each key is relative to.

    Returns:
        ``path::qualname`` mapped to the cyclomatic complexity, for each function
        scoring above ``MAX_COMPLEXITY``.

    Raises:
        SystemExit: Ruff gives no report, or flags a row that starts no function.
    """
    if not roots:
        return {}
    cache: dict[Path, dict[int, str]] = {}
    scores: dict[str, int] = {}
    for diagnostic in _ruff_c901(roots):
        found = _TOO_COMPLEX.search(str(diagnostic["message"]))
        if found is None:
            continue
        path = Path(str(diagnostic["filename"]))
        row = int(diagnostic["location"]["row"])
        qualname = _qualname_at(path, row, cache)
        _keep_largest(scores, _key(path, base, qualname), int(found[1]))
    return scores


def find_violations(
    measured: dict[str, int], baseline: dict[str, int], metric: Metric
) -> list[str]:
    """Describe every function that is new over the limit or larger than recorded.

    Args:
        measured: Offenders found now, as ``find_long_functions`` returns them.
        baseline: The values recorded so far.
        metric: Unit and limit to quote in the messages.

    Returns:
        One line per violation, sorted by key. An entry that is only in the baseline
        (the function is gone or back under the limit) is not a violation.
    """
    lines = []
    for key, value in sorted(measured.items()):
        stored = baseline.get(key)
        if stored is None:
            lines.append(
                f"{key}: {value} {metric.unit}, over the limit of {metric.limit}"
                " and not in the baseline"
            )
        elif value > stored:
            lines.append(f"{key}: {value} {metric.unit}, baseline allows {stored}")
    return lines


def shrink_baseline(
    stored: dict[str, int] | None, measured: dict[str, int]
) -> dict[str, int]:
    """Return the baseline after an update: lowered or trimmed, never grown.

    Args:
        stored: The recorded values, or None when no baseline exists yet.
        measured: Offenders found now.

    Returns:
        ``measured`` itself for the initial write. Otherwise each recorded entry
        keeps the smaller of its stored and measured value, and an entry whose
        function is gone or back under the limit is dropped. An offender missing
        from ``stored`` is never added.
    """
    if stored is None:
        return dict(measured)
    return {
        key: min(value, measured[key])
        for key, value in stored.items()
        if key in measured
    }


def _read(path: Path) -> dict[str, int] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path: Path, entries: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _measure(layout: Layout) -> dict[Metric, dict[str, int]]:
    return {
        LONG: find_long_functions(layout.roots, LONG.limit, base=layout.base),
        COMPLEX: find_complex_functions(layout.roots, base=layout.base),
    }


def run(layout: Layout, *, update: bool) -> int:
    """Measure the tree, optionally shrink the baselines, and apply the policy.

    Args:
        layout: Source roots and baseline directory to use.
        update: Rewrite each baseline through ``shrink_baseline`` before checking.

    Returns:
        0 when no function is new over the limit or larger than recorded, else 1.
        With ``update`` the check runs against the baselines just written, so a new
        offender is still reported.
    """
    failures: list[str] = []
    for metric, measured in _measure(layout).items():
        path = layout.baseline_dir / metric.baseline_file
        stored = _read(path)
        if update:
            stored = shrink_baseline(stored, measured)
            _write(path, stored)
            print(f"{path.name}: {len(stored)} entries")
        elif stored is None:
            raise SystemExit(f"{path} is missing; create it with --update")
        failures += find_violations(measured, stored, metric)
    if failures:
        print("\n".join(failures), file=sys.stderr)
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None, *, layout: Layout = DEFAULT_LAYOUT) -> int:
    """Entry point: check the baselines, or shrink them with ``--update``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="lower or remove baseline entries (creates a missing baseline)",
    )
    args = parser.parse_args(argv)
    return run(layout, update=args.update)


if __name__ == "__main__":
    sys.exit(main())
