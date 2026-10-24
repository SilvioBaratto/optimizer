"""Guard: the portopt-db Alembic tree has exactly one head.

Two branches that each add a revision on the same parent merge cleanly in git,
because they create different files, yet leave Alembic with two heads, and
``alembic upgrade head`` then aborts with "Multiple head revisions". Nothing
else in the suite loads the revision graph, so without this guard the break
only surfaces when a database is migrated. The fix at integration is a merge
revision (``alembic merge``), not a rebase of one branch onto the other.

Anchored on ``Path(__file__).resolve().parents[1]`` (the portopt-db package
root), never ``Path.cwd()``.
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_ALEMBIC_DIR = Path(__file__).resolve().parents[1] / "alembic"

_REVISION_TEMPLATE = '''"""{rev}"""

revision = "{rev}"
down_revision = {down!r}
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
'''


def _heads(script_location: Path) -> list[str]:
    """Return the sorted head revision ids of the tree at ``script_location``."""
    config = Config()
    config.set_main_option("script_location", str(script_location))
    return sorted(ScriptDirectory.from_config(config).get_heads())


def _write_tree(root: Path, revisions: dict[str, str | tuple[str, ...] | None]) -> Path:
    """Write a throwaway Alembic tree mapping each revision id to its parent(s)."""
    versions = root / "versions"
    versions.mkdir(parents=True)
    for rev, down in revisions.items():
        (versions / f"{rev}_stub.py").write_text(
            _REVISION_TEMPLATE.format(rev=rev, down=down)
        )
    return root


def test_the_portopt_db_alembic_tree_has_a_single_head() -> None:
    """Parallel branches must be joined by a merge revision before they land."""
    heads = _heads(_ALEMBIC_DIR)

    assert len(heads) == 1, (
        f"Multiple Alembic heads: {heads}. Two branches added a revision on the "
        "same parent; join them with `alembic merge -m <message> <head> <head>`."
    )


def test_the_head_probe_reports_every_head_of_a_forked_tree(tmp_path: Path) -> None:
    """The guard must be able to fail: a fork yields both child revisions."""
    root = _write_tree(
        tmp_path, {"aaa111": None, "bbb222": "aaa111", "ccc333": "aaa111"}
    )

    assert _heads(root) == ["bbb222", "ccc333"]


def test_the_head_probe_treats_a_merge_revision_as_the_single_head(
    tmp_path: Path,
) -> None:
    """Joining the fork with a merge revision restores a single head."""
    root = _write_tree(
        tmp_path,
        {
            "aaa111": None,
            "bbb222": "aaa111",
            "ccc333": "aaa111",
            "ddd444": ("bbb222", "ccc333"),
        },
    )

    assert _heads(root) == ["ddd444"]
