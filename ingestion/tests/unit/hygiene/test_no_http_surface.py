"""Guard: no HTTP surface anywhere under ``ingestion/`` (issue #5).

Source-blind by construction: scans the raw text of tracked files for dead
FastAPI/ASGI markers (``fastapi``, ``uvicorn``, ``TestClient``, references to
``app.main``, and route decorators). No implementation module is imported —
this is a pure text-content guard, authored from the acceptance criteria
before any implementation existed.

Ambiguity resolutions recorded here (per the "choose the simplest behaviour
consistent with the criterion text" instruction):

* ``[scope-4]`` says paths anchor on ``Path(__file__).resolve().parents[3]``.
  From this file's location (``ingestion/tests/unit/hygiene/``), that index
  lands on the ``ingestion/`` directory itself — used directly as the scan
  root, with no separate climb to the repository root and no ``Path.cwd()``.
* ``[scope-3]`` says the test "excludes its own source file and allowlists
  ``test_database_manager.py``". Applied *only* that literally, every sibling
  hygiene guard that legitimately names these markers to prove their absence
  (e.g. ``test_fastapi_dependency_removed.py``) — and the deployment
  integration suite under ``tests/integration/deployment/`` that installs a
  fastapi-free venv and asserts ``pip show fastapi`` fails — would trip the
  scan and violate ``[scope-1]`` ("the test passes against the current
  tree"). Resolved by exempting those two directories wholesale (the same
  "guards are allowed to name what they guard against" rule the criterion
  already applies to this file), and reserving the explicit
  ``(path, line-substring, reason)`` allowlist for the one match that sits
  outside both: ``tests/unit/test_database_manager.py``'s module docstring,
  which states in prose that no FastAPI dependency exists.
* Marker matching splits by kind. The word tokens (``fastapi``, ``uvicorn``)
  match case-insensitively — the regressions this guard exists to catch are
  capitalised prose ("FastAPI application", "FastAPI/SQLAlchemy"), not just
  lowercase imports, and a case-sensitive scan would miss them. The
  identifier tokens (``TestClient``, ``from app.main``, ``import app.main``,
  ``app.main:``) stay case-sensitive, since they name a specific symbol and a
  lowercase match would false-positive on unrelated prose.
* Any path component starting with ``.`` is skipped (``.pytest_cache/``,
  ``.mypy_cache/``, ``.ruff_cache/``, ``.venv/``), not only the two literal
  directory names the issue names — a stray cache artifact must not make the
  guard flake between a dev machine and a clean CI checkout.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_SELF = Path(__file__).resolve()
_HYGIENE_DIR = _SELF.parent
_DEPLOYMENT_VERIFICATION_DIR = _INGESTION_ROOT / "tests" / "integration" / "deployment"
_EXEMPT_DIRS = (_HYGIENE_DIR, _DEPLOYMENT_VERIFICATION_DIR)

_CASE_INSENSITIVE_MARKERS = ("fastapi", "uvicorn")
_CASE_SENSITIVE_MARKERS = (
    "TestClient",
    "from app.main",
    "import app.main",
    "app.main:",
)
_ROUTE_DECORATOR_PATTERN = re.compile(r"@\w+\.(get|post|put|delete|patch|websocket)\(")
_SCAN_SUFFIXES = {".py", ".toml", ".ini", ".cfg", ".txt"}
_EXCLUDE_DIR_PARTS = {"__pycache__", "baml_client"}

# (path-suffix, line-substring, reason)
_ALLOWLIST = (
    (
        "tests/unit/test_database_manager.py",
        "no FastAPI",
        "module docstring states in prose that no FastAPI dependency backs "
        "the session manager; not a live import or route",
    ),
)


def _is_allowlisted(relative_path: str, line: str) -> bool:
    return any(
        relative_path.endswith(path_suffix) and substring in line
        for path_suffix, substring, _reason in _ALLOWLIST
    )


def _scan_text_for_markers(relative_path: str, text: str) -> list[str]:
    offending: list[str] = []
    for line in text.splitlines():
        if _is_allowlisted(relative_path, line):
            continue
        lowered = line.lower()
        for marker in _CASE_INSENSITIVE_MARKERS:
            if marker in lowered:
                offending.append(f"{relative_path}: {marker!r}")
        for marker in _CASE_SENSITIVE_MARKERS:
            if marker in line:
                offending.append(f"{relative_path}: {marker!r}")
        if _ROUTE_DECORATOR_PATTERN.search(line):
            offending.append(f"{relative_path}: route decorator")
    return offending


def _has_dotted_component(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_exempt(path: Path) -> bool:
    resolved = path.resolve()
    if resolved == _SELF:
        return True
    return any(exempt in resolved.parents for exempt in _EXEMPT_DIRS)


def _iter_files_under(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in _SCAN_SUFFIXES:
            continue
        if _EXCLUDE_DIR_PARTS.intersection(path.parts):
            continue
        relative_parts = path.relative_to(root).parts
        if _has_dotted_component(Path(*relative_parts)):
            continue
        if _is_exempt(path):
            continue
        yield path


def _iter_scanned_files() -> Iterator[Path]:
    # Reads the module-level _INGESTION_ROOT at call time (not as a default
    # argument), so a test that monkeypatches it observes the new value.
    yield from _iter_files_under(_INGESTION_ROOT)


def find_http_violations(root: Path) -> list[str]:
    """Scan ``root`` for dead FastAPI/ASGI markers.

    Injectable-root sibling of ``_iter_scanned_files`` (issue #7): callers
    outside this module — the cross-guard checkpoint — point the scan at a
    synthetic tree instead of the real ``ingestion/`` root.

    Args:
        root: Directory tree to scan.

    Returns:
        One string per offending marker occurrence, empty when clean.
    """
    offending: list[str] = []
    for path in _iter_files_under(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        # Normalise to POSIX separators so the allowlist (which uses "/"
        # suffixes) matches on Windows, where relative_to yields "\".
        relative_path = path.resolve().relative_to(root).as_posix()
        offending.extend(_scan_text_for_markers(relative_path, text))
    return offending


@pytest.mark.criterion("global-4")
def test_when_ingestion_tree_is_scanned_then_no_forbidden_http_marker_is_found():
    assert find_http_violations(_INGESTION_ROOT) == []


@pytest.mark.criterion("scope-2")
@pytest.mark.parametrize("suffix", sorted(_SCAN_SUFFIXES))
def test_when_marker_appears_in_a_scanned_file_suffix_then_it_is_detected(suffix):
    text = "# leftover marker\nfastapi\n"
    offending = _scan_text_for_markers(f"probe{suffix}", text)
    assert offending


@pytest.mark.criterion("scope-2")
def test_when_route_decorator_appears_then_it_is_detected():
    text = '@app.get("/health")\ndef health():\n    return {}\n'
    offending = _scan_text_for_markers("probe.py", text)
    assert any("route decorator" in item for item in offending)


@pytest.mark.criterion("scope-1")
def test_when_hygiene_package_is_scanned_then_init_file_is_present():
    assert (_HYGIENE_DIR / "__init__.py").is_file()


@pytest.mark.criterion("scope-3")
def test_when_own_source_file_is_scanned_then_it_is_excluded():
    assert _SELF not in set(_iter_scanned_files())


@pytest.mark.criterion("scope-3")
def test_when_allowlisted_database_manager_line_is_scanned_then_it_is_not_flagged():
    text = "There is no request scope and no FastAPI dependency to lean on.\n"
    offending = _scan_text_for_markers("tests/unit/test_database_manager.py", text)
    assert offending == []


@pytest.mark.criterion("scope-3")
def test_when_same_line_appears_in_a_non_allowlisted_file_then_it_is_flagged():
    text = "There is no request scope and no FastAPI dependency to lean on.\n"
    offending = _scan_text_for_markers("app/some_other_module.py", text)
    assert offending


@pytest.mark.criterion("scope-4")
def test_when_ingestion_root_is_resolved_then_it_points_at_the_ingestion_directory():
    assert _INGESTION_ROOT.name == "ingestion"


@pytest.mark.criterion("scope-5")
def test_when_fastapi_marker_is_reintroduced_into_a_toml_file_then_scan_fails(tmp_path):
    reintroduced = tmp_path / "pyproject.toml"
    reintroduced.write_text('dependencies = ["fastapi>=0.100"]\n', encoding="utf-8")
    text = reintroduced.read_text(encoding="utf-8")
    offending = _scan_text_for_markers("pyproject.toml", text)
    reintroduced.unlink()
    assert offending, (
        "reintroducing fastapi into a scanned toml file must fail the scan"
    )


@pytest.mark.criterion("scope-5")
def test_when_capitalized_fastapi_prose_is_reintroduced_into_pytest_ini_then_scan_fails():
    text = "# Pytest configuration for FastAPI application\n[pytest]\n"
    offending = _scan_text_for_markers("pytest.ini", text)
    assert offending, (
        "capitalised 'FastAPI' prose must fail the scan, not just lowercase imports"
    )


@pytest.mark.criterion("scope-2")
def test_when_testclient_identifier_appears_lowercase_then_it_is_not_flagged():
    # Identifier markers (TestClient, app.main forms) stay case-sensitive: a
    # lowercase match would false-positive on unrelated prose using the same
    # letters, unlike the word tokens fastapi/uvicorn which regress as prose.
    text = "the testclient fixture name here is unrelated lowercase text\n"
    offending = _scan_text_for_markers("probe.py", text)
    assert offending == []


@pytest.mark.criterion("scope-1")
def test_when_ingestion_root_has_a_dotted_cache_dir_then_its_contents_are_excluded(
    tmp_path, monkeypatch
):
    dotted_cache = tmp_path / ".pytest_cache"
    dotted_cache.mkdir()
    (dotted_cache / "leftover.txt").write_text("fastapi\n", encoding="utf-8")
    monkeypatch.setattr(f"{__name__}._INGESTION_ROOT", tmp_path)

    scanned = set(_iter_scanned_files())

    assert scanned == set()
