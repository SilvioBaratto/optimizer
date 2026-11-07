"""Source fingerprint stamped on the built images.

`portopt start` rebuilds an image only when its tag is missing, yet every `portopt
setup` migrates the database with the checkout's migrations. After an upgrade the old
images would keep running old code against the new schema, so each image carries a
fingerprint of the files it is built from and `start` rebuilds when it differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.setup import image_source

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _write(root: Path, rel: str, text: str = "x") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A minimal tree holding one file from every directory an image copies."""
    for rel in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "packages/portopt-core/optimizer/__init__.py",
        "packages/portopt-db/src/portopt_db/__init__.py",
        "packages/portopt-db/alembic/versions/0001_init.py",
        "ingestion/app/worker.py",
        "ingestion/Dockerfile",
        "fund/src/fund/worker.py",
        "fund/Dockerfile",
        "optimizerwiki/docs/theory.md",
    ):
        _write(tmp_path, rel, rel)
    return tmp_path


def test_the_hash_is_stable_for_an_unchanged_tree(checkout: Path) -> None:
    """Two reads of an untouched tree agree, or every start would rebuild."""
    assert image_source.source_hash(checkout, "scheduler") == image_source.source_hash(
        checkout, "scheduler"
    )


def test_a_new_migration_changes_both_images(checkout: Path) -> None:
    """The migration tree is the reason the stale-image bug bites: the host
    migrates the database to a head the old image has never heard of."""
    before = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}
    _write(checkout, "packages/portopt-db/alembic/versions/0002_next.py")
    after = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}

    assert after["scheduler"] != before["scheduler"]
    assert after["fund"] != before["fund"]


def test_a_changed_lock_changes_both_images(checkout: Path) -> None:
    """A dependency bump alone makes both images stale."""
    before = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}
    _write(checkout, "uv.lock", "bumped")
    after = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}

    assert after["scheduler"] != before["scheduler"]
    assert after["fund"] != before["fund"]


def test_an_edit_the_image_never_copies_leaves_its_hash_alone(checkout: Path) -> None:
    """Rebuilding the scheduler for a fund-only edit would re-pull its base image
    for nothing, and the reverse for the fund."""
    scheduler = image_source.source_hash(checkout, "scheduler")
    fund = image_source.source_hash(checkout, "fund")

    _write(checkout, "fund/src/fund/worker.py", "changed")
    assert image_source.source_hash(checkout, "scheduler") == scheduler
    assert image_source.source_hash(checkout, "fund") != fund

    fund = image_source.source_hash(checkout, "fund")
    _write(checkout, "ingestion/app/worker.py", "changed")
    assert image_source.source_hash(checkout, "fund") == fund


@pytest.mark.parametrize(
    "noise",
    [
        "ingestion/app/__pycache__/worker.cpython-312.pyc",
        "ingestion/tests/unit/test_something.py",
        "ingestion/.env",
        "ingestion/.pytest_cache/v/cache/lastfailed",
        "fund/.venv/lib/site.py",
        "fund/secrets/ollama_api_key",
        "fund/.runtime/agent-fs/notes.txt",
        "packages/portopt-db/build/lib/x.py",
    ],
)
def test_files_the_dockerignore_drops_do_not_move_the_hash(
    checkout: Path, noise: str
) -> None:
    """A test edit, a stray .env or a bytecode cache never reaches the image, so it
    must not trigger a rebuild."""
    before = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}
    _write(checkout, noise)
    after = {k: image_source.source_hash(checkout, k) for k in ("scheduler", "fund")}

    assert after == before


def test_a_tree_missing_every_input_still_hashes(tmp_path: Path) -> None:
    """A partial checkout yields a hash instead of raising."""
    assert image_source.source_hash(tmp_path, "fund")


def test_an_unknown_image_is_rejected(tmp_path: Path) -> None:
    """Only the two built images have a fingerprint; adminer and db are pulled."""
    with pytest.raises(KeyError):
        image_source.source_hash(tmp_path, "adminer")


def test_the_hash_covers_the_real_checkout() -> None:
    """Smoke-run against the repo itself: the inputs named here must exist."""
    for key in ("scheduler", "fund"):
        for rel in image_source.IMAGES[key].inputs:
            assert (_REPO_ROOT / rel).exists(), rel


def test_compose_stamps_each_image_with_the_hash_portopt_passes() -> None:
    """The label is how `docker image inspect` later tells a stale image from a
    current one; the env var is how `portopt start` hands the hash to the build."""
    services = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]
    for service, key in (("scheduler", "scheduler"), ("fund", "fund")):
        spec = image_source.IMAGES[key]
        labels = services[service]["build"]["labels"]
        assert spec.env in labels[image_source.SOURCE_HASH_LABEL]
        assert services[service]["image"] == spec.ref
