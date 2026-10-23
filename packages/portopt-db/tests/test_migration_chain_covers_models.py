"""Guard: Alembic is the single schema author, so the revision chain must
produce every table, column, index and unique constraint the ORM declares.

Anything only ``Base.metadata.create_all`` builds is missing from an
Alembic-only database (a restore, CI, a new environment). ``bond_yields`` was
one: no revision created it, so every bond-yield upsert raised
``UndefinedTable`` until some process happened to run ``create_all``.

The check replays the literal ``op.*`` calls of each revision's ``upgrade()``
in chain order (``ast``, no Postgres needed), drops and renames included, and
compares the resulting schema with ``Base.metadata``. It is an approximation:
conditional guards count as taken, calls behind helper functions or
non-literal names are not seen, and DDL passed to ``op.execute`` is understood
only for index and constraint renames. ``alembic check`` against a live
Postgres stays the authoritative parity test.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Column, Index, Integer, MetaData, Table, UniqueConstraint

from portopt_db.models import Base

_VERSIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"
_BOND_YIELDS_REVISION = "f5a7c9e1b3d2"

_RENAME_INDEX = re.compile(
    r"\bALTER\s+INDEX\s+(\w+)\s+RENAME\s+TO\s+(\w+)", re.IGNORECASE
)
_RENAME_CONSTRAINT = re.compile(
    r"\bRENAME\s+CONSTRAINT\s+(\w+)\s+TO\s+(\w+)", re.IGNORECASE
)


@dataclass
class _Schema:
    """Per table, the column, index and unique-constraint names a replay leaves."""

    columns: dict[str, set[str]] = field(default_factory=dict)
    indexes: dict[str, set[str]] = field(default_factory=dict)
    uniques: dict[str, set[str]] = field(default_factory=dict)

    def create(
        self,
        table: str,
        columns: set[str],
        indexes: set[str],
        uniques: set[str],
    ) -> None:
        self.columns[table] = columns
        self.indexes[table] = indexes
        self.uniques[table] = uniques

    def drop(self, table: str) -> None:
        for per_table in (self.columns, self.indexes, self.uniques):
            per_table.pop(table, None)

    def rename(self, old: str, new: str) -> None:
        if old not in self.columns:
            return
        for per_table in (self.columns, self.indexes, self.uniques):
            per_table[new] = per_table.pop(old)


@dataclass(frozen=True)
class _Revision:
    revision: str
    down: tuple[str, ...]
    source: str


def _call_name(node: ast.AST, owner: str) -> str | None:
    """Return ``attr`` when ``node`` is a call to ``owner.attr(...)``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    ):
        return node.func.attr
    return None


def _str(node: ast.expr | None) -> str | None:
    """Return a string literal, looking through ``op.f("name")`` wrappers."""
    if _call_name(node, "op") == "f" and isinstance(node, ast.Call) and node.args:
        node = node.args[0]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _arg(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    return call.args[position] if len(call.args) > position else _keyword(call, keyword)


def _walk_with_helpers(
    root: ast.AST,
    functions: Mapping[str, ast.FunctionDef],
    seen: frozenset[str] = frozenset(),
) -> Iterator[ast.AST]:
    """Walk ``root``, descending into same-module helpers it calls.

    Revisions splat shared id/timestamp columns from helpers such as
    ``_id_and_timestamps()``; without descending, those columns go unseen.
    """
    for node in ast.walk(root):
        yield node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            helper = functions.get(node.func.id)
            if helper is not None and helper.name not in seen:
                yield from _walk_with_helpers(helper, functions, seen | {helper.name})


def _declared(
    call: ast.Call, table: str, functions: Mapping[str, ast.FunctionDef]
) -> tuple[set[str], set[str], set[str]]:
    """Return the column, index and unique-constraint names ``call`` declares."""
    columns: set[str] = set()
    indexes: set[str] = set()
    uniques: set[str] = set()
    for node in _walk_with_helpers(call, functions):
        if not isinstance(node, ast.Call):
            continue
        match _call_name(node, "sa"):
            case "Column":
                column = _str(node.args[0]) if node.args else None
                if column:
                    columns.add(column)
                    flag = _keyword(node, "index")
                    # SQLAlchemy names an ``index=True`` column's index this way.
                    if isinstance(flag, ast.Constant) and flag.value is True:
                        indexes.add(f"ix_{table}_{column}")
            case "Index":
                if name := _str(node.args[0] if node.args else None):
                    indexes.add(name)
            case "UniqueConstraint":
                if name := _str(_keyword(node, "name")):
                    uniques.add(name)
    return columns, indexes, uniques


def _rename_in(per_table: dict[str, set[str]], old: str, new: str) -> None:
    for names in per_table.values():
        if old in names:
            names.discard(old)
            names.add(new)


def _apply_op(
    schema: _Schema,
    op_name: str,
    call: ast.Call,
    functions: Mapping[str, ast.FunctionDef],
) -> None:
    first = _str(call.args[0] if call.args else None)
    second = _str(_arg(call, 1, "table_name"))
    match op_name:
        case "create_table" if first:
            schema.create(first, *_declared(call, first, functions))
        case "drop_table" if first:
            schema.drop(first)
        case "rename_table" if first and second:
            schema.rename(first, second)
        case "add_column" if first in schema.columns:
            columns, indexes, uniques = _declared(call, first, functions)
            schema.columns[first] |= columns
            schema.indexes[first] |= indexes
            schema.uniques[first] |= uniques
        case "drop_column" if first in schema.columns:
            schema.columns[first].discard(second or "")
        case "alter_column" if first in schema.columns:
            renamed = _str(_keyword(call, "new_column_name"))
            if renamed and second in schema.columns[first]:
                schema.columns[first].discard(second)
                schema.columns[first].add(renamed)
        case "create_index" if first and second in schema.indexes:
            schema.indexes[second].add(first)
        case "drop_index" if first:
            for names in schema.indexes.values():
                names.discard(first)
        case "create_unique_constraint" if first and second in schema.uniques:
            schema.uniques[second].add(first)
        case "drop_constraint" if first and second in schema.uniques:
            schema.uniques[second].discard(first)
        case "execute" if first:
            for old, new in _RENAME_INDEX.findall(first):
                _rename_in(schema.indexes, old, new)
            for old, new in _RENAME_CONSTRAINT.findall(first):
                _rename_in(schema.uniques, old, new)


def _apply_upgrade(schema: _Schema, source: str) -> None:
    """Replay the literal ``op.*`` calls of ``upgrade()`` in source order.

    ``downgrade()`` is skipped on purpose: a drop revision's downgrade
    re-creates the tables it dropped, which must not count as creation.
    """
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }
    upgrade = functions.get("upgrade")
    if upgrade is None:
        return
    calls = [
        (node, op_name)
        for node in ast.walk(upgrade)
        if isinstance(node, ast.Call) and (op_name := _call_name(node, "op"))
    ]
    for call, op_name in sorted(calls, key=lambda c: (c[0].lineno, c[0].col_offset)):
        _apply_op(schema, op_name, call, functions)


def _parse_revision(source: str) -> _Revision:
    ids: dict[str, Any] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in {"revision", "down_revision"}:
            ids[target.id] = ast.literal_eval(value)
    down = ids["down_revision"]
    if down is None:
        parents: tuple[str, ...] = ()
    elif isinstance(down, str):
        parents = (down,)
    else:
        parents = tuple(down)
    return _Revision(str(ids["revision"]), parents, source)


def _in_chain_order(sources: Iterable[str]) -> list[_Revision]:
    """Order revisions parent-first; file names carry no chain order."""
    pending = {rev.revision: rev for rev in map(_parse_revision, sources)}
    ordered: list[_Revision] = []
    done: set[str] = set()
    while pending:
        ready = sorted(rid for rid, rev in pending.items() if set(rev.down) <= done)
        if not ready:
            unresolved = {rid: pending[rid].down for rid in sorted(pending)}
            raise ValueError(f"revisions whose parents never resolve: {unresolved}")
        ordered.extend(pending.pop(rid) for rid in ready)
        done.update(ready)
    return ordered


def _replay(sources: Iterable[str]) -> _Schema:
    schema = _Schema()
    for revision in _in_chain_order(sources):
        _apply_upgrade(schema, revision.source)
    return schema


def _drift(orm_tables: Mapping[str, Table], schema: _Schema) -> list[str]:
    """List the ORM declarations the replayed schema lacks, one line each."""
    problems: list[str] = []
    for name, table in sorted(orm_tables.items()):
        if name not in schema.columns:
            problems.append(f"{name}: table is not produced by the revision chain")
            continue
        declared = (
            ("column", {c.name for c in table.columns}, schema.columns[name]),
            ("index", {i.name for i in table.indexes if i.name}, schema.indexes[name]),
            (
                "unique constraint",
                {
                    c.name
                    for c in table.constraints
                    if isinstance(c, UniqueConstraint) and c.name
                },
                schema.uniques[name],
            ),
        )
        for kind, wanted, built in declared:
            problems.extend(
                f"{name}: {kind} {missing} is not produced by the revision chain"
                for missing in sorted(wanted - built)
            )
    return problems


def _revision(revision: str, down: str | tuple[str, ...] | None, body: str) -> str:
    return (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        f"revision = {revision!r}\n"
        f"down_revision = {down!r}\n"
        "def upgrade():\n"
        f"{textwrap.indent(textwrap.dedent(body), '    ')}\n"
        "def downgrade():\n"
        "    pass\n"
    )


def _replay_bodies(*bodies: str) -> _Schema:
    sources = [
        _revision(f"r{i}", f"r{i - 1}" if i else None, body)
        for i, body in enumerate(bodies)
    ]
    return _replay(sources)


def _head_id(sources: list[str]) -> str:
    revisions = [_parse_revision(source) for source in sources]
    parents = {parent for revision in revisions for parent in revision.down}
    return next(r.revision for r in revisions if r.revision not in parents)


def test_create_and_rename_targets_exist_after_upgrade() -> None:
    """Create and rename targets count; the rename source does not."""
    schema = _replay_bodies(
        """
        op.create_table('made', sa.Column('id', sa.Integer()))
        op.create_table('old_name', sa.Column('id', sa.Integer()))
        op.rename_table('old_name', 'renamed')
        """
    )

    assert set(schema.columns) == {"made", "renamed"}


def test_a_later_revision_can_drop_and_recreate_a_table() -> None:
    """A table a later revision drops is gone until something recreates it."""
    create = "op.create_table('t', sa.Column('id', sa.Integer()))"

    dropped = _replay_bodies(create, "op.drop_table('t')")
    recreated = _replay_bodies(create, "op.drop_table('t')", create)

    assert "t" not in dropped.columns
    assert "t" in recreated.columns


def test_ops_apply_in_source_order_within_one_upgrade() -> None:
    """Create-then-drop leaves nothing; drop-then-create leaves the table."""
    create = "op.create_table('t', sa.Column('id', sa.Integer()))"
    drop = "op.drop_table('t')"

    created_first = _replay_bodies(f"{create}\n{drop}")
    dropped_first = _replay_bodies(f"{drop}\n{create}")

    assert "t" not in created_first.columns
    assert "t" in dropped_first.columns


def test_downgrade_bodies_are_not_replayed() -> None:
    """A drop revision's downgrade re-creates the table; that must not count."""
    create = _revision(
        "r0", None, "op.create_table('t', sa.Column('id', sa.Integer()))"
    )
    drop = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "revision = 'r1'\n"
        "down_revision = 'r0'\n"
        "def upgrade():\n"
        "    op.drop_table('t')\n"
        "def downgrade():\n"
        "    op.create_table('t', sa.Column('id', sa.Integer()))\n"
    )

    schema = _replay([create, drop])

    assert "t" not in schema.columns


def test_column_ops_are_replayed() -> None:
    """Adds, drops and renames leave the net column set."""
    schema = _replay_bodies(
        """
        op.create_table(
            't',
            sa.Column('id', sa.Integer()),
            sa.Column('a', sa.Integer()),
            sa.Column('b', sa.Integer()),
        )
        op.add_column('t', sa.Column('c', sa.Integer()))
        op.drop_column('t', 'a')
        op.alter_column('t', 'b', new_column_name='b2')
        op.alter_column('t', 'c', nullable=False)
        """
    )

    assert schema.columns["t"] == {"id", "c", "b2"}


def test_column_ops_on_a_table_no_revision_created_are_ignored() -> None:
    """An add_column must not make a never-created table look present."""
    schema = _replay_bodies("op.add_column('ghost', sa.Column('a', sa.Integer()))")

    assert "ghost" not in schema.columns


def test_columns_from_same_module_helpers_are_counted() -> None:
    """Revisions splat shared id/timestamp columns from a local helper."""
    source = (
        "import sqlalchemy as sa\n"
        "from alembic import op\n"
        "revision = 'r0'\n"
        "down_revision = None\n"
        "def _stamps():\n"
        "    return [sa.Column('created_at', sa.DateTime())]\n"
        "def upgrade():\n"
        "    op.create_table('t', sa.Column('id', sa.Integer()), *_stamps())\n"
        "def downgrade():\n"
        "    pass\n"
    )

    assert _replay([source]).columns["t"] == {"id", "created_at"}


def test_index_and_unique_constraint_ops_are_replayed() -> None:
    """Inline, explicit and op.f-wrapped names count; dropped ones do not."""
    schema = _replay_bodies(
        """
        op.create_table(
            't',
            sa.Column('id', sa.Integer()),
            sa.Column('a', sa.Integer(), index=True),
            sa.Column('b', sa.Integer()),
            sa.UniqueConstraint('a', 'b', name='uq_inline'),
            sa.Index('ix_inline', 'b'),
        )
        op.create_index(op.f('ix_made'), 't', ['a'])
        op.create_unique_constraint('uq_made', 't', ['b'])
        op.create_index('ix_gone', 't', ['b'])
        op.create_index('ix_loose', 't', ['b'])
        op.create_unique_constraint('uq_gone', 't', ['a'])
        op.drop_index('ix_gone', table_name='t')
        op.drop_index('ix_loose')
        op.drop_constraint('uq_gone', 't', type_='unique')
        """
    )

    assert schema.indexes["t"] == {"ix_t_a", "ix_inline", "ix_made"}
    assert schema.uniques["t"] == {"uq_inline", "uq_made"}


def test_rename_table_carries_columns_indexes_and_uniques() -> None:
    """Postgres keeps index and constraint names across a table rename."""
    schema = _replay_bodies(
        """
        op.create_table(
            'old',
            sa.Column('id', sa.Integer()),
            sa.Column('a', sa.Integer(), index=True),
            sa.UniqueConstraint('a', name='uq_a'),
        )
        op.rename_table('old', 'new')
        """
    )

    assert "old" not in schema.columns
    assert schema.columns["new"] == {"id", "a"}
    assert schema.indexes["new"] == {"ix_old_a"}
    assert schema.uniques["new"] == {"uq_a"}


def test_raw_sql_renames_of_indexes_and_constraints_are_followed() -> None:
    """Revisions rename these through ``op.execute``; the new names count."""
    schema = _replay_bodies(
        """
        op.create_table(
            't',
            sa.Column('a', sa.Integer(), index=True),
            sa.UniqueConstraint('a', name='uq_old'),
        )
        op.execute("ALTER INDEX ix_t_a RENAME TO ix_new")
        op.execute(
            "ALTER TABLE t "
            "RENAME CONSTRAINT uq_old TO uq_new"
        )
        """
    )

    assert schema.indexes["t"] == {"ix_new"}
    assert schema.uniques["t"] == {"uq_new"}


def test_revisions_replay_in_chain_order_not_filename_order() -> None:
    """The drop sorts first by id but runs second by ``down_revision``."""
    first = _revision("zz", None, "op.create_table('t', sa.Column('id', sa.Integer()))")
    second = _revision("aa", "zz", "op.drop_table('t')")

    assert "t" not in _replay([second, first]).columns


def test_a_merge_revision_runs_after_every_parent() -> None:
    """A tuple ``down_revision`` waits for all of its parents."""
    sources = [
        _revision("m", ("b", "c"), "pass"),
        _revision("c", "a", "pass"),
        _revision("b", "a", "pass"),
        _revision("a", None, "pass"),
    ]

    order = [revision.revision for revision in _in_chain_order(sources)]

    assert order[0] == "a"
    assert order[-1] == "m"


def test_an_unknown_parent_revision_is_reported() -> None:
    """A broken chain fails loudly instead of silently skipping revisions."""
    orphan = _revision("b", "missing", "pass")

    with pytest.raises(ValueError, match="missing"):
        _replay([orphan])


def test_drift_names_each_missing_table_column_index_and_unique_constraint() -> None:
    """Every kind of ORM declaration the chain lacks gets its own line."""
    metadata = MetaData()
    Table(
        "has_all",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("a", Integer),
        UniqueConstraint("a", name="uq_a"),
        Index("ix_a", "a"),
    )
    Table("never_created", metadata, Column("id", Integer, primary_key=True))
    thin = _Schema(
        columns={"has_all": {"id"}},
        indexes={"has_all": set()},
        uniques={"has_all": set()},
    )
    full = _Schema(
        columns={"has_all": {"id", "a"}, "never_created": {"id"}},
        indexes={"has_all": {"ix_a"}, "never_created": set()},
        uniques={"has_all": {"uq_a"}, "never_created": set()},
    )

    assert _drift(metadata.tables, thin) == [
        "has_all: column a is not produced by the revision chain",
        "has_all: index ix_a is not produced by the revision chain",
        "has_all: unique constraint uq_a is not produced by the revision chain",
        "never_created: table is not produced by the revision chain",
    ]
    assert _drift(metadata.tables, full) == []


@pytest.fixture(scope="module")
def chain_sources() -> list[str]:
    """Return the source text of every revision in ``alembic/versions``."""
    return [
        path.read_text(encoding="utf-8") for path in sorted(_VERSIONS_DIR.glob("*.py"))
    ]


def test_revision_chain_builds_the_schema_the_orm_declares(
    chain_sources: list[str],
) -> None:
    """No ORM table, column, index or unique constraint is left to create_all."""
    drift = _drift(Base.metadata.tables, _replay(chain_sources))

    assert drift == []


def test_chain_without_the_bond_yields_revision_is_flagged(
    chain_sources: list[str],
) -> None:
    """The original defect: ``bond_yields`` created by no revision."""
    removed = next(
        source
        for source in chain_sources
        if _parse_revision(source).revision == _BOND_YIELDS_REVISION
    )
    (parent,) = _parse_revision(removed).down
    # Later revisions chain on top of it; splice it out so they stay reachable.
    without = [
        source.replace(f'"{_BOND_YIELDS_REVISION}"', f'"{parent}"')
        for source in chain_sources
        if source is not removed
    ]

    drift = _drift(Base.metadata.tables, _replay(without))

    assert "bond_yields: table is not produced by the revision chain" in drift


def test_a_mapped_unique_constraint_the_chain_never_creates_is_flagged(
    chain_sources: list[str],
) -> None:
    """Column and constraint drift is caught even when the table exists."""
    renamed = [
        source.replace("uq_bond_yield_country_maturity", "uq_renamed")
        for source in chain_sources
    ]

    drift = _drift(Base.metadata.tables, _replay(renamed))

    assert (
        "bond_yields: unique constraint uq_bond_yield_country_maturity "
        "is not produced by the revision chain"
    ) in drift


def test_a_later_drop_of_a_mapped_table_column_or_index_is_flagged(
    chain_sources: list[str],
) -> None:
    """A revision that drops what the ORM still maps must not pass."""
    dropper = _revision(
        "zz_dropper",
        _head_id(chain_sources),
        """
        op.drop_table('bond_yields')
        op.drop_column('instruments', 'ticker')
        op.drop_index('ix_instruments_ticker', table_name='instruments')
        """,
    )

    drift = _drift(Base.metadata.tables, _replay([*chain_sources, dropper]))

    assert "bond_yields: table is not produced by the revision chain" in drift
    assert "instruments: column ticker is not produced by the revision chain" in drift
    assert (
        "instruments: index ix_instruments_ticker is not produced by the revision chain"
    ) in drift
