"""Bind tests to the requirement registry and fail when the matrix has holes.

Standard 6.6 asks each test to cite the requirement it verifies and CI to keep the
links. This module holds both halves, so the plugin and the check share one
definition of the CSV and of a requirement id:

* a pytest plugin. A suite's config puts ``scripts/ci`` on its ``pythonpath`` and
  loads it with ``-p traceability``; ``--traceability-out=PATH`` then writes one CSV
  row per (``criterion`` id, collected test), carrying the issue numbers of the
  test's ``defect`` markers, plus one row with an empty id for a test that has
  ``defect`` markers and no ``criterion`` marker;
* ``traceability.py check --requirements docs/requirements.md --csv A.csv
  [--csv B.csv ...]``, which exits 1 when a registered requirement has no test, a
  test cites a ``REQ-`` id the registry does not list, a test carries a ``defect``
  marker but no ``REQ-`` id, or an input cannot be read or parsed.

An id is a requirement id when it starts with ``REQ-``. The ingestion and
portopt-db suites also tag tests with acceptance-criterion labels such as
``scope-3`` through the same ``criterion`` marker; the CSV keeps those rows, and
the check does not look the labels up in the registry. A ``defect`` marker on a test
whose only ids are such labels is still a hole. pytest is imported for type hints
only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENT_PREFIX = "REQ-"
_REQUIREMENT_ID = re.compile(r"REQ-(?:CORE|DB|ING|FUND|OPS)-\d{3}")
_TABLE_HEADER = ("id", "statement", "module", "standard")
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_CSV_FIELDS = ("req_id", "nodeid", "module", "defects")
_ISSUE_NUMBER = re.compile(r"[1-9][0-9]*")
_CRITERION = "criterion"
_DEFECT = "defect"
_EXIT_OK = 0
_EXIT_FAILED = 1


class RegistryError(ValueError):
    """The requirements table is missing, empty or malformed."""


@dataclass(frozen=True)
class Requirement:
    """One row of the registry table.

    Attributes:
        req_id: The ``REQ-<AREA>-NNN`` id.
        statement: The behaviour the citing tests assert.
        module: Repository-relative path of the code the statement is about.
        standard: The standard clause, or the test plan risk, the row answers.
    """

    req_id: str
    statement: str
    module: str
    standard: str


@dataclass(frozen=True)
class TraceRow:
    """One (id, test) pair of the matrix.

    Attributes:
        req_id: The ``criterion`` id; empty for a test that carries only ``defect``.
        nodeid: The test's pytest node id, relative to its suite's rootdir.
        module: The test's file relative to the repository root, which tells the
            suites apart where their node ids are equal.
        defects: Sorted GitHub issue numbers from the test's ``defect`` markers.
    """

    req_id: str
    nodeid: str
    module: str
    defects: tuple[int, ...] = ()


def collect(items: Iterable[pytest.Item], *, root: Path = _REPO_ROOT) -> list[TraceRow]:
    """Turn the ``criterion`` and ``defect`` markers of ``items`` into rows.

    Args:
        items: Collected test items, in collection order.
        root: Directory the ``module`` of each row is relative to.

    Returns:
        One row per distinct ``criterion`` id of each item, in the order
        ``iter_markers`` yields them, each carrying the item's issue numbers. An
        item with ``defect`` markers and no ``criterion`` gets one row with an empty
        id; an item with neither gets no row.

    Raises:
        ValueError: A ``criterion`` marker whose arguments are not one non-empty
            string without surrounding spaces, or a ``defect`` marker whose
            arguments are not one positive integer.
    """
    rows: list[TraceRow] = []
    for item in items:
        marks = item.iter_markers(name=_CRITERION)
        ids = dict.fromkeys(_criterion_id(item, mark) for mark in marks)
        issues = {_issue(item, mark) for mark in item.iter_markers(name=_DEFECT)}
        defects = tuple(sorted(issues))
        module = item.path.relative_to(root, walk_up=True).as_posix()
        for req_id in ids or ([""] if defects else []):
            rows.append(TraceRow(req_id, item.nodeid, module, defects))
    return rows


def _criterion_id(item: pytest.Item, mark: pytest.Mark) -> str:
    value = mark.args[0] if len(mark.args) == 1 else None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{item.nodeid}: @pytest.mark.criterion takes one id")
    return value


def _issue(item: pytest.Item, mark: pytest.Mark) -> int:
    value = mark.args[0] if len(mark.args) == 1 else None
    # ``True`` is an ``int``; an issue number written as a flag is a typo.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{item.nodeid}: @pytest.mark.defect takes one positive issue number"
        )
    return value


def write_csv(rows: Iterable[TraceRow], path: Path) -> None:
    """Write ``rows`` under the header ``req_id,nodeid,module,defects``.

    The parent directory is created when missing; ``defects`` holds the issue
    numbers separated by spaces.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_FIELDS)
        for row in rows:
            issues = " ".join(str(issue) for issue in row.defects)
            writer.writerow([row.req_id, row.nodeid, row.module, issues])


def read_csv(path: Path) -> list[TraceRow]:
    """Read the rows :func:`write_csv` wrote.

    Raises:
        OSError: ``path`` cannot be opened.
        ValueError: The header is not ``req_id,nodeid,module,defects``, a
            ``defects`` cell holds something other than positive issue numbers in
            plain digits without leading zeros, or the csv reader rejects the file.
            The message starts with ``path``.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
                columns = ",".join(_CSV_FIELDS)
                raise ValueError(f"{path}: expected the columns {columns}")
            return [_trace_row(record, path) for record in reader]
        except csv.Error as error:
            raise ValueError(f"{path}: {error}") from error


def _trace_row(record: dict[str, str | None], path: Path) -> TraceRow:
    cell = record["defects"] or ""
    issues = cell.split()
    # The plugin writes only numbers ``_issue`` accepted, in plain digits, so any
    # other cell was not written by it and must not link a test to an issue.
    if not all(_ISSUE_NUMBER.fullmatch(issue) for issue in issues):
        raise ValueError(
            f"{path}: the defects cell {cell!r} is not a list of positive issue "
            "numbers in plain digits without leading zeros"
        )
    return TraceRow(
        record["req_id"] or "",
        record["nodeid"] or "",
        record["module"] or "",
        tuple(int(issue) for issue in issues),
    )


def parse_registry(text: str) -> list[Requirement]:
    """Read the requirements table of ``docs/requirements.md``.

    The table is the first one headed ``| id | statement | module | standard |``
    and ends at the first line that does not start with ``|``, so a cell cannot
    hold a ``|``. Backticks around the module path are dropped.

    Raises:
        RegistryError: No such table, no row in it, a row without four non-empty
            cells, an id not of the form ``REQ-<AREA>-NNN``, or an id listed twice.
    """
    lines = text.splitlines()
    rows = _table_rows(lines, _table_start(lines))
    requirements = [_requirement(number, cells) for number, cells in rows]
    if not requirements:
        raise RegistryError("the requirements table has no requirement rows")
    seen: set[str] = set()
    for requirement in requirements:
        if requirement.req_id in seen:
            raise RegistryError(f"{requirement.req_id} is listed twice")
        seen.add(requirement.req_id)
    return requirements


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_start(lines: Sequence[str]) -> int:
    """Index of the first row under the registry header and its separator."""
    for index, line in enumerate(lines[:-1]):
        separator = _cells(lines[index + 1])
        if tuple(_cells(line)) == _TABLE_HEADER and all(
            _SEPARATOR_CELL.fullmatch(cell) for cell in separator
        ):
            return index + 2
    header = " | ".join(_TABLE_HEADER)
    raise RegistryError(f"no requirements table with the header | {header} |")


def _table_rows(lines: Sequence[str], start: int) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(line number, cells)`` for each row, stopping at the table's end."""
    for index in range(start, len(lines)):
        if not lines[index].lstrip().startswith("|"):
            return
        yield index + 1, _cells(lines[index])


def _requirement(number: int, cells: list[str]) -> Requirement:
    if len(cells) != len(_TABLE_HEADER):
        raise RegistryError(
            f"line {number}: a row needs {len(_TABLE_HEADER)} cells, not {len(cells)}"
        )
    req_id, statement, module, standard = cells
    if not _REQUIREMENT_ID.fullmatch(req_id):
        raise RegistryError(
            f"line {number}: {req_id!r} is not an id of the form REQ-<AREA>-NNN"
        )
    module = module.strip("`")
    if not (statement and module and standard):
        raise RegistryError(f"line {number}: {req_id} has an empty cell")
    return Requirement(req_id, statement, module, standard)


def _is_requirement_id(value: str) -> bool:
    return value.startswith(_REQUIREMENT_PREFIX)


def _test_label(row: TraceRow) -> str:
    """The test as ``<file from the repository root>::<name>``."""
    _, separator, name = row.nodeid.partition("::")
    return f"{row.module}{separator}{name}"


def find_holes(
    requirements: Sequence[Requirement], rows: Sequence[TraceRow]
) -> list[str]:
    """Name each way the matrix of ``rows`` falls short of the registry.

    Args:
        requirements: The registered requirements.
        rows: The trace rows of the suites to check, merged.

    Returns:
        One message per registered id that no row cites, per row that cites a
        ``REQ-`` id the registry does not list, and per issue of a test that has no
        ``REQ-`` id; empty when the matrix is complete. An id that does not start
        with ``REQ-`` is not looked up in the registry, but the ``defect`` issues
        on its row still count against a test that has no ``REQ-`` id.
    """
    registered = {requirement.req_id for requirement in requirements}
    cited = {row.req_id for row in rows}
    holes = [
        f"{req_id} is registered, but no test cites it"
        for req_id in sorted(registered - cited)
    ]
    holes += [
        f"{_test_label(row)} cites {row.req_id}, which the registry does not list"
        for row in rows
        if _is_requirement_id(row.req_id) and row.req_id not in registered
    ]
    return holes + _unlinked_defects(rows)


def _unlinked_defects(rows: Sequence[TraceRow]) -> list[str]:
    linked = {
        (row.module, row.nodeid) for row in rows if _is_requirement_id(row.req_id)
    }
    unlinked: dict[tuple[str, str], TraceRow] = {}
    for row in rows:
        if row.defects and (row.module, row.nodeid) not in linked:
            unlinked.setdefault((row.module, row.nodeid), row)
    return [
        f"{_test_label(row)} carries defect {issue} but cites no requirement id"
        for row in unlinked.values()
        for issue in row.defects
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check the matrix against the registry")
    check.add_argument(
        "--requirements", type=Path, required=True, help="the registry markdown file"
    )
    check.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="a CSV written by --traceability-out; repeat it for each suite",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``check`` and return its exit code.

    Returns:
        0 when the matrix is complete; 1 when it has a hole or an input cannot be
        read or parsed. A usage error exits 2 from argparse instead.
    """
    args = _parser().parse_args(argv)
    try:
        requirements = parse_registry(args.requirements.read_text(encoding="utf-8"))
        rows = [row for path in args.csv for row in read_csv(path)]
    except (OSError, ValueError) as error:
        print(f"traceability: {error}", file=sys.stderr)
        return _EXIT_FAILED
    holes = find_holes(requirements, rows)
    for hole in holes:
        print(f"traceability: {hole}", file=sys.stderr)
    if holes:
        return _EXIT_FAILED
    tests = {(row.module, row.nodeid) for row in rows if _is_requirement_id(row.req_id)}
    print(
        f"traceability: {len(requirements)} requirements cited by {len(tests)} "
        "tests, no holes"
    )
    return _EXIT_OK


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--traceability-out``."""
    parser.addoption(
        "--traceability-out",
        metavar="CSV",
        default=None,
        help=(
            "write one row per (criterion id, collected test), plus one row with an "
            "empty id per test with defect markers and no criterion marker, to this "
            "CSV file"
        ),
    )


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write the trace of the session's items when ``--traceability-out`` is given.

    pytest calls this after ``-k`` and ``-m`` have deselected items, so the file
    lists the tests the session selected. A relative path is taken from the
    directory pytest was started in.
    """
    target = session.config.getoption("traceability_out")
    if target:
        write_csv(collect(session.items), session.config.invocation_params.dir / target)


if __name__ == "__main__":
    sys.exit(main())
