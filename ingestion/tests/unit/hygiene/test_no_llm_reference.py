"""Guard: no LLM stack anywhere under ``ingestion/``.

The LLM steps (news summarisation, macro-regime calibration) were removed from
the ingestion daemon — they will return as a separate ``packages/`` member. No
BAML / LangChain / OpenAI / Anthropic / Ollama reference must survive in the
ingestion tree, or a dead dependency (or a re-added one) would slip back in
unnoticed.

Source-blind by construction: scans the raw text of files under ``ingestion/``
for the forbidden markers. No implementation module is imported — this is a
pure text-content guard, a sibling of ``test_no_http_surface.py`` and modelled
on it.

Anchored on ``Path(__file__).resolve().parents[3]`` — from
``ingestion/tests/unit/hygiene/``, that index lands on ``ingestion/`` itself.
Never anchored on ``Path.cwd()`` — CI may invoke pytest from either the repo
root or ``ingestion/``.

The one legitimate ``BAML*`` token in the tree is a FRED series identifier
(``BAMLH0A0HYM2`` / ``BAMLC0A0CM``: ICE BofA OAS indices) — a substring
collision with the BAML LLM library, allowlisted by exact path + substring so
the collision is silenced without blinding the guard to a real regression
elsewhere.
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

_SCAN_SUFFIXES = {".py", ".toml", ".ini", ".cfg", ".txt", ".yaml", ".yml"}
_EXCLUDE_DIR_PARTS = {"__pycache__"}

# Case-insensitive: regressions are as likely to be capitalised prose
# ("OpenAI provider", "LangChain structured output") as lowercase imports.
_CASE_INSENSITIVE_MARKERS = (
    "baml",
    "langchain",
    "openai",
    "anthropic",
    "ollama",
)
# ``llm`` matched on a word boundary so it catches ``llm_provider`` / ``LLM``
# without false-positiving on unrelated substrings.
_LLM_WORD_PATTERN = re.compile(r"\bllm\b", re.IGNORECASE)

# (path-suffix, line-substring, reason) — matched by path suffix + exact
# substring-in-line, so an entry silences only the one documented occurrence.
_ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    (
        "app/services/macro/scrapers/fred_scraper.py",
        "BAMLH0A0HYM2",
        "FRED ICE BofA US High Yield OAS series id, not the BAML LLM library",
    ),
    (
        "app/services/macro/scrapers/fred_scraper.py",
        "BAMLC0A0CM",
        "FRED ICE BofA US Corporate IG OAS series id, not the BAML LLM library",
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
        if _LLM_WORD_PATTERN.search(line):
            offending.append(f"{relative_path}: 'llm'")
    return offending


def _has_dotted_component(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_build_artifact(path: Path) -> bool:
    # egg-info/dist-info are generated on build (gitignored); their requires.txt
    # mirrors pyproject and lags a stale copy until the next rebuild.
    return any(part.endswith((".egg-info", ".dist-info")) for part in path.parts)


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
        if _is_build_artifact(path):
            continue
        relative_parts = path.relative_to(root).parts
        if _has_dotted_component(Path(*relative_parts)):
            continue
        if _is_exempt(path):
            continue
        yield path


def _iter_scanned_files() -> Iterator[Path]:
    # Reads module-level _INGESTION_ROOT at call time (not as a default
    # argument), so a test that monkeypatches it observes the new value.
    yield from _iter_files_under(_INGESTION_ROOT)


def find_llm_violations(root: Path) -> list[str]:
    """Scan ``root`` for forbidden LLM-stack markers.

    Injectable-root sibling of ``_iter_scanned_files``: callers point the scan
    at a synthetic tree instead of the real ``ingestion/`` root.

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


def test_when_ingestion_tree_is_scanned_then_no_llm_marker_is_found():
    assert find_llm_violations(_INGESTION_ROOT) == []


@pytest.mark.parametrize("marker", sorted(_CASE_INSENSITIVE_MARKERS))
def test_when_a_forbidden_marker_appears_then_it_is_detected(marker):
    offending = _scan_text_for_markers("probe.py", f"# leftover {marker.upper()}\n")
    assert offending


def test_when_llm_word_appears_then_it_is_detected():
    offending = _scan_text_for_markers("probe.py", "llm_provider = 'openai'\n")
    assert offending


def test_when_fred_oas_series_id_is_scanned_then_it_is_not_flagged():
    text = '    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",\n'
    offending = _scan_text_for_markers(
        "app/services/macro/scrapers/fred_scraper.py", text
    )
    assert offending == []


def test_when_baml_token_appears_in_a_non_allowlisted_file_then_it_is_flagged():
    text = '    "BAMLH0A0HYM2": "ICE BofA US High Yield OAS",\n'
    offending = _scan_text_for_markers("app/some_other_module.py", text)
    assert offending


def test_when_own_source_file_is_scanned_then_it_is_excluded():
    assert _SELF not in set(_iter_scanned_files())
