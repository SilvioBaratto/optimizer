"""Guard: no LLM *stack* (agent-stack import / provider SDK) under ``ingestion/``.

The LLM steps (news summarisation, macro-regime calibration) were removed from
the ingestion daemon — they will return as a separate ``packages/`` member. The
real, load-bearing boundary is that ingestion must never **import** the agent
stack (``baml`` / ``langchain`` / a provider SDK) or ``optimizer``, so a dead
(or re-added) dependency cannot slip back in unnoticed.

The one legitimate re-entry of LLM *configuration* (not the stack) is the
``portopt setup`` wizard: it configures the ``fund`` service's switchable-LLM
backend — provider selection, per-provider file-based secrets, and httpx-only
validation. Those are config identifiers (``"openai"``, ``ollama_api_key``,
``llm_provider``, ``validate_llm``), not a dependency. So the guard is scoped:

* Everywhere in ``ingestion/``: forbid the LLM *libraries* (``baml`` /
  ``langchain`` as substrings) and any provider-SDK **import** line
  (``import openai`` / ``from anthropic import …`` / ``import ollama``) — an SDK
  import is a re-added dependency even inside the wizard (validation is
  httpx-only).
* Only within the wizard LLM-config surface (``app/setup/`` + ``app/cli.py``):
  allow provider *names* and the ``llm`` word as bare config text. Elsewhere they
  stay forbidden.

Source-blind by construction: scans the raw text of files under ``ingestion/``.
No implementation module is imported — a pure text-content guard, sibling of
``test_no_http_surface.py``.

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

# LLM *libraries*: never a legitimate config string — forbidden anywhere.
# Case-insensitive: regressions are as likely to be capitalised prose
# ("LangChain structured output") as lowercase imports.
_ALWAYS_FORBIDDEN_MARKERS = (
    "baml",
    "langchain",
)
# Provider names: allowed as config identifiers (secret ids, ``"openai"``) ONLY
# inside the wizard LLM-config surface; forbidden as bare text elsewhere. Their
# SDK *imports* are caught everywhere by ``_PROVIDER_IMPORT_PATTERN`` below.
_PROVIDER_NAME_MARKERS = (
    "openai",
    "anthropic",
    "ollama",
)
# An actual provider-SDK dependency creeping back in (``import openai`` /
# ``from anthropic import …`` / ``import ollama``) — forbidden EVERYWHERE,
# including the wizard, which must validate over httpx only, never via an SDK.
_PROVIDER_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+(?:openai|anthropic|ollama)\b", re.IGNORECASE
)
# ``llm`` matched on a word boundary so it catches ``LLM`` / ``--llm-provider``
# (config, allowed in the wizard surface) without false-positiving on unrelated
# substrings. ``llm_provider`` has no boundary before ``_`` and is never matched.
_LLM_WORD_PATTERN = re.compile(r"\bllm\b", re.IGNORECASE)

# The setup-wizard LLM-config surface: ``portopt setup`` configures the fund's
# switchable-LLM backend without importing the agent stack. Provider *names* and
# the ``llm`` word are legitimate config here; SDK imports + LLM libraries are
# not (those stay forbidden everywhere). The surface is the wizard/CLI config
# code (``app/setup/`` + ``app/cli.py``) AND its tests (``tests/unit/setup/`` +
# ``tests/unit/test_cli.py``), which must name providers / ``--llm-*`` flags to
# exercise the config — the SDK-import + library checks still run there, so a
# real ``import openai`` in any of them is still caught. Paths are POSIX,
# relative to ingestion/.
_WIZARD_LLM_CONFIG_PREFIXES = ("app/setup/", "tests/unit/setup/")
_WIZARD_LLM_CONFIG_FILES = ("app/cli.py", "tests/unit/test_cli.py")


def _is_wizard_llm_config(relative_path: str) -> bool:
    return relative_path in _WIZARD_LLM_CONFIG_FILES or any(
        relative_path.startswith(prefix) for prefix in _WIZARD_LLM_CONFIG_PREFIXES
    )


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
    in_wizard = _is_wizard_llm_config(relative_path)
    for line in text.splitlines():
        if _is_allowlisted(relative_path, line):
            continue
        lowered = line.lower()
        # LLM libraries: never legitimate, anywhere.
        for marker in _ALWAYS_FORBIDDEN_MARKERS:
            if marker in lowered:
                offending.append(f"{relative_path}: {marker!r}")
        # A provider-SDK import is a re-added dependency — forbidden even in the
        # wizard (its validation is httpx-only, never an SDK).
        if _PROVIDER_IMPORT_PATTERN.search(line):
            offending.append(f"{relative_path}: 'provider-sdk-import'")
        # Provider names + the `llm` word as bare text are config: allowed only
        # in the setup-wizard LLM-config surface, forbidden elsewhere.
        if not in_wizard:
            for marker in _PROVIDER_NAME_MARKERS:
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


@pytest.mark.parametrize(
    "marker", sorted(_ALWAYS_FORBIDDEN_MARKERS + _PROVIDER_NAME_MARKERS)
)
def test_when_a_forbidden_marker_appears_in_a_normal_file_then_it_is_detected(marker):
    # A non-wizard module: every LLM library + provider name is flagged.
    offending = _scan_text_for_markers("probe.py", f"# leftover {marker.upper()}\n")
    assert offending


def test_when_llm_word_appears_in_a_normal_file_then_it_is_detected():
    offending = _scan_text_for_markers("app/services/foo.py", "# a stray LLM step\n")
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


# --- import-scoped guard: wizard LLM-config surface vs. the rest ---------------


def test_provider_name_allowed_in_wizard_config_surface():
    # `portopt setup` configures the fund's per-provider secrets; the names are
    # config identifiers, not a dependency. The surface includes the wizard's own
    # tests, which must name providers to exercise validate_llm.
    for path in (
        "app/setup/compose_secrets.py",
        "app/setup/validators.py",
        "app/cli.py",
        "tests/unit/setup/test_validators.py",
        "tests/unit/test_cli.py",
    ):
        assert (
            _scan_text_for_markers(path, '    validate_llm("openai", key="k")\n') == []
        )


def test_llm_word_allowed_in_wizard_config_surface():
    # `--llm-provider` / `llm_provider` are legitimate wizard config.
    assert _scan_text_for_markers("app/cli.py", "'--llm-provider',\n") == []


def test_provider_name_flagged_outside_wizard_surface():
    # The same string in a non-wizard module is still a regression.
    assert _scan_text_for_markers("app/services/foo.py", '    x = "ollama"\n')


def test_provider_sdk_import_flagged_even_in_wizard_surface():
    # An SDK import is a re-added dependency — forbidden even in the wizard, which
    # must validate over httpx only.
    for line in ("import openai\n", "from anthropic import Anthropic\n"):
        assert _scan_text_for_markers("app/setup/wizard.py", line)


def test_llm_library_flagged_even_in_wizard_surface():
    # baml / langchain are the LLM libraries — never legitimate, even in setup.
    assert _scan_text_for_markers("app/setup/wizard.py", "import langchain\n")


def test_when_own_source_file_is_scanned_then_it_is_excluded():
    assert _SELF not in set(_iter_scanned_files())
