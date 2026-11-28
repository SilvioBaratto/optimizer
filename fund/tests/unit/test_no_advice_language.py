"""User-visible text never frames the software as advice or an advising service.

The fund computes candidate portfolios that the person running it reviews and
approves themselves, on paper. Nothing it shows that person, nor anything its agents
read, may present it as a suitability assessment, an investment adviser, a fund
manager or an agent that chooses the portfolio, and the person is "you" or "the user",
never "the adviser". Identifiers, stored data and CLI flags keep their names.

The phrase scan reads raw source on purpose (comments, docstrings and CLAUDE.md
included), so an old phrase cannot survive anywhere; it covers ``portopt_db`` too,
because the Ask agent answers from that source as well as from ``fund``'s. The adviser
scan reads only the text a person or an agent sees: non-docstring string literals,
``@tool`` docstrings, the bound tool descriptions, the skills, the READMEs and the
``## [Unreleased]`` section of CHANGELOG.md, where the quotations of removed wording are
listed with their reason. The model-facing scan adds the schemas the agents are bound
to, and the package scan reads the ``description`` every ``pyproject.toml`` ships as
its wheel summary. A further scan reads the developer prose of ``fund`` and
``portopt_db`` (comments and docstrings), which the Ask agent stages as ``/code`` and
quotes to the user, so the word cannot return there either.
"""

from __future__ import annotations

import ast
import functools
import io
import json
import re
import tokenize
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import tomllib
from _fund_fakes import ASOF, PORTFOLIO_ID
from langchain_core.utils.function_calling import convert_to_openai_tool

from fund import schemas
from fund.agents.ask_agent import build_ask_tools
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext, bind_toolset
from fund.config import settings

_REPO = Path(__file__).resolve().parents[3]
_FUND_SRC = _REPO / "fund" / "src" / "fund"
_PORTOPT_DB_SRC = _REPO / "packages" / "portopt-db" / "src" / "portopt_db"
_SKILLS = _FUND_SRC / "skills"
_READMES = (_REPO / "README.md", _REPO / "fund" / "README.md")
_CLAUDE_MD = _REPO / "CLAUDE.md"
_CHANGELOG = _REPO / "CHANGELOG.md"
_ROOT_PYPROJECT = _REPO / "pyproject.toml"

_OLD_PHRASES = (
    "MiFID suitability questionnaire",
    "suitability band",
    "GUARDRAILS (from MiFID profile)",
    "Persist MiFID suitability profile",
    "Approve to write the profile",
    "EU investment adviser",
    "portfolio manager (PM) and orchestrator of an EU investment fund",
    "where an LLM chooses, the optimizer computes",
    "the LLM chooses, the optimizer computes",
    "LLM chooses, optimizer computes",
    "investment-fund bridge",
    "MiFID II",
    "ESMA pillar",
    "MiFID record",
    "MiFID suitability",
    "suitability record",
    "suitability profile",
    "in the suitability assessment",
    "record-keeping",
    "regulatory double-binding",
    "compliance record",
    "compliance-safe",
    "compliance guarantee",
    "amended assessment",
    "whose suitability is being assessed",
    "investor questionnaire",
    "derived from the assessment",
    "suitability classification",
    "the suitability check",
    "suitability reasoning",
    "read as full suitability",
    "suitability breach",
    "suitability ceiling",
    "active assessment",
    "always suitable",
    "client is suited for",
    "legally blocked",
    "suitability block",
    "suitability rows",
    "suitability flags",
)
_ADVISER = re.compile(r"\badvisers?\b", re.IGNORECASE)
# Identifiers keep the old name (``adviser_edit``, ``current_adviser_choice``), so an
# underscore or a digit next to the word makes it part of a name, not prose; a quote
# or an apostrophe does not, so ``adviser's`` is still the word.
_ADVISER_PROSE = re.compile(
    r"(?<![A-Za-z0-9_])advis(?:er|ers|or|ors)(?![A-Za-z0-9_])", re.IGNORECASE
)
# A wheel summary is the first line a stranger reads about a package.
_PACKAGE_FRAMING = re.compile(
    r"LLM\s+chooses|investment[\s-]+funds?|\badvis\w*|suitability|fund\s+manag\w*",
    re.IGNORECASE,
)
# The changelog entry that removed the wording has to quote it to say what changed.
_CHANGELOG_QUOTATIONS = {
    '"the adviser"': "the relabelling entry names the term the user is never called",
    '"adviser"': "the relabelling entry names the word its guard fails on",
    "`chosen by the adviser: <id>`": "the `fund report` line the relabelling replaced",
    "`ADVISER DIRECTIVE`": "the steer heading the relabelling replaced",
}
# Word boundaries keep identifiers such as ``mandate_excluded`` and ``MiFIDAnswers``
# out of the match: only the prose words are the old framing.
_OLD_TERMS = re.compile(
    r"\b(advisers?|clients?|suitability|esma|pillars?|mandates?|guardrails?|mifid)\b",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text)


def _is_tool(node: ast.AST) -> bool:
    # A LangChain ``@tool`` hands its docstring to the model as the tool description.
    decorators = getattr(node, "decorator_list", [])
    targets = [d.func if isinstance(d, ast.Call) else d for d in decorators]
    return any(
        (isinstance(t, ast.Name) and t.id == "tool")
        or (isinstance(t, ast.Attribute) and t.attr == "tool")
        for t in targets
    )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body or _is_tool(node):
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            found.add(id(first.value))
    return found


def _string_literals(path: Path, *, docstrings: bool) -> Iterator[tuple[int, int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = set() if docstrings else _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
        ):
            yield node.lineno, node.col_offset, node.value


def _visible_literals(path: Path, label: str) -> dict[str, str]:
    # Keyed by line and column: two literals on one line must both be scanned.
    return {
        f"{label}:{line}:{col}": text
        for line, col, text in _string_literals(path, docstrings=False)
    }


def _python_sources() -> list[Path]:
    return sorted(_FUND_SRC.rglob("*.py"))


def _markdown_sources() -> list[Path]:
    return [*sorted(_SKILLS.rglob("*.md")), *_READMES]


def _phrase_python_sources() -> list[Path]:
    return [*_python_sources(), *sorted(_PORTOPT_DB_SRC.rglob("*.py"))]


@functools.cache
def _phrase_haystacks() -> dict[str, str]:
    # Raw text finds a phrase in a comment or across a wrapped line; the parsed
    # literals find one split over implicitly concatenated string parts.
    haystacks = {
        str(path.relative_to(_REPO)): _normalise(
            "\n".join(
                [
                    path.read_text(encoding="utf-8"),
                    *(text for *_, text in _string_literals(path, docstrings=True)),
                ]
            )
        )
        for path in _phrase_python_sources()
    }
    for path in (*_markdown_sources(), _CLAUDE_MD):
        haystacks[str(path.relative_to(_REPO))] = _normalise(
            path.read_text(encoding="utf-8")
        )
    return haystacks


def _developer_prose(source: str) -> list[tuple[str, int, str]]:
    """Return ``(kind, first line, text)`` for each comment and docstring."""
    prose = [
        ("comment", token.start[0], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        text = first.value if isinstance(first, ast.Expr) else None
        if isinstance(text, ast.Constant) and isinstance(text.value, str):
            prose.append(("docstring", text.lineno, text.value))
    return prose


def _adviser_prose_hits(
    prose: list[tuple[str, int, str]], label: str
) -> dict[str, str]:
    hits: dict[str, str] = {}
    for kind, line, text in prose:
        for match in _ADVISER_PROSE.finditer(text):
            at = line + text.count("\n", 0, match.start())
            hits[f"{label}:{at}"] = f"{kind}: {match.group(0)}"
    return hits


@functools.cache
def _developer_prose_index() -> dict[str, list[tuple[str, int, str]]]:
    return {
        str(path.relative_to(_REPO)): _developer_prose(path.read_text(encoding="utf-8"))
        for path in _phrase_python_sources()
    }


def _tool_descriptions() -> dict[str, str]:
    # Binding builds closures only; no tool runs, so neither context needs a database.
    ctx = RunContext(
        session=None,  # type: ignore[arg-type]
        asof=ASOF,
        store=None,
        config=settings,
        run_id=uuid.uuid4(),
        portfolio_id=PORTFOLIO_ID,
    )
    run_tools = [tool for role in TOOLS_BY_AGENT for tool in bind_toolset(role, ctx)]
    ask_tools = build_ask_tools(
        session_factory=None,  # type: ignore[arg-type]
        saver=None,
        focus=None,
    )
    return {
        f"tool {tool.name}": f"{tool.description}\n{json.dumps(tool.args)}"
        for tool in (*run_tools, *ask_tools)
    }


def _structured_call_schema_names() -> set[str]:
    names: set[str] = set()
    for path in _python_sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            func = node.func if isinstance(node, ast.Call) else None
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if called == "structured_call" and len(node.args) >= 2:
                schema = node.args[1]
                if isinstance(schema, ast.Name):
                    names.add(schema.id)
    return names


def _structured_output_schemas() -> dict[str, str]:
    # ``structured_call`` binds with ``method="function_calling"``, so each model's
    # class docstrings and field descriptions reach the LLM as the tool schema.
    return {
        f"schema {name}": json.dumps(convert_to_openai_tool(getattr(schemas, name)))
        for name in sorted(_structured_call_schema_names())
    }


def _user_visible_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in _python_sources():
        texts.update(_visible_literals(path, str(path.relative_to(_REPO))))
    texts.update(_tool_descriptions())
    for path in _markdown_sources():
        texts[str(path.relative_to(_REPO))] = path.read_text(encoding="utf-8")
    return texts


def _hits(pattern: re.Pattern[str], texts: dict[str, str]) -> dict[str, list[str]]:
    return {
        source: sorted({match.group(0).lower() for match in pattern.finditer(text)})
        for source, text in texts.items()
        if pattern.search(text)
    }


def _package_descriptions() -> dict[str, str]:
    # The workspace members are every package this repository builds a wheel from.
    root = tomllib.loads(_ROOT_PYPROJECT.read_text(encoding="utf-8"))
    members = root["tool"]["uv"]["workspace"]["members"]
    paths = [_ROOT_PYPROJECT, *(_REPO / name / "pyproject.toml" for name in members)]
    descriptions: dict[str, str] = {}
    for path in paths:
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        descriptions[str(path.relative_to(_REPO))] = project["description"]
    return descriptions


def _old_phrases_in(texts: dict[str, str]) -> dict[str, list[str]]:
    found = {
        source: [phrase for phrase in _OLD_PHRASES if phrase.lower() in text.lower()]
        for source, text in texts.items()
    }
    return {source: phrases for source, phrases in found.items() if phrases}


def _unreleased_changelog() -> str:
    text = _CHANGELOG.read_text(encoding="utf-8")
    section = re.search(r"^## \[Unreleased\]\s*$(.*?)(?=^## \[)", text, re.M | re.S)
    assert section, "CHANGELOG.md has no '## [Unreleased]' section"
    return _normalise(section.group(1))


def _without_quotations(text: str) -> str:
    for quotation in _CHANGELOG_QUOTATIONS:
        text = text.replace(quotation, "")
    return text


@pytest.mark.criterion("REQ-FUND-052")
@pytest.mark.parametrize("phrase", _OLD_PHRASES)
def test_old_user_visible_phrases_are_gone(phrase: str) -> None:
    """Validates the user-visible text of fund (the TUI, the CLI, the agent prompts,
    the tool and role descriptions, the skills), the portopt_db source the Ask agent
    reads, and README.md, fund/README.md and CLAUDE.md: none keeps a phrase that
    framed the fund as a regulated suitability service, an investment adviser, a fund
    manager or an LLM that chooses."""
    needle = phrase.lower()
    hits = sorted(
        source for source, text in _phrase_haystacks().items() if needle in text.lower()
    )
    assert hits == []


@pytest.mark.criterion("REQ-FUND-052")
def test_user_visible_text_never_calls_the_user_an_adviser() -> None:
    """Validates every non-docstring string literal under fund/src/fund, the
    docstring of every ``@tool`` there, the descriptions of the tools
    fund.agents.toolsets.bind_toolset and fund.agents.ask_agent.build_ask_tools bind,
    the skills, README.md and fund/README.md: the person running the tool is never
    called an adviser."""
    assert _hits(_ADVISER, _user_visible_texts()) == {}


@pytest.mark.criterion("REQ-FUND-052")
def test_two_adviser_literals_on_one_line_are_both_found(tmp_path: Path) -> None:
    """Validates the literal scan behind
    test_user_visible_text_never_calls_the_user_an_adviser: literals sharing a line
    are kept apart, so neither can hide the other."""
    source = tmp_path / "two_on_a_line.py"
    source.write_text('PAIR = ("the adviser", "an adviser")\n', encoding="utf-8")

    hits = _hits(_ADVISER, _visible_literals(source, "synthetic"))

    assert set(hits) == {"synthetic:1:8", "synthetic:1:23"}


@pytest.mark.criterion("REQ-FUND-052")
def test_model_facing_schemas_and_tool_descriptions_drop_the_old_framing() -> None:
    """Validates the descriptions and argument schemas of every tool
    fund.agents.toolsets.bind_toolset and fund.agents.ask_agent.build_ask_tools bind,
    and the JSON schema of every model passed to fund.schemas.structured_call (the
    profiler's ``MiFIDAnswers``): none says adviser, client, suitability, ESMA,
    pillar, mandate, guardrail or MiFID as prose."""
    schema_texts = _structured_output_schemas()
    assert "schema MiFIDAnswers" in schema_texts

    assert _hits(_OLD_TERMS, {**_tool_descriptions(), **schema_texts}) == {}


@pytest.mark.criterion("REQ-FUND-052")
def test_package_descriptions_drop_the_old_framing() -> None:
    """Validates the ``description`` of the root pyproject.toml and of every
    workspace member's: no wheel summary presents a package as an investment fund, an
    adviser, a suitability service or an LLM that chooses."""
    descriptions = _package_descriptions()
    assert "fund/pyproject.toml" in descriptions

    assert _hits(_PACKAGE_FRAMING, descriptions) == {}
    assert _old_phrases_in(descriptions) == {}


@pytest.mark.criterion("REQ-FUND-052")
@pytest.mark.parametrize(
    "summary",
    [
        "deepagents investment-fund bridge: LLM chooses, optimizer computes",
        "an LLM  chooses the weights",
        "an investment fund in a box",
        "advisory portfolios",
    ],
)
def test_the_package_scan_flags_each_old_summary(summary: str) -> None:
    """Validates the pattern behind test_package_descriptions_drop_the_old_framing:
    each old way of summarising the fund package is caught."""
    assert _hits(_PACKAGE_FRAMING, {"synthetic": summary})


@pytest.mark.criterion("REQ-FUND-052")
def test_unreleased_changelog_never_calls_the_user_an_adviser() -> None:
    """Validates the ``## [Unreleased]`` section of CHANGELOG.md: outside the listed
    quotations of removed wording no entry calls the user an adviser, and every
    listed quotation is still there to need its entry."""
    section = _unreleased_changelog()

    assert [quote for quote in _CHANGELOG_QUOTATIONS if quote not in section] == []
    assert all(reason.strip() for reason in _CHANGELOG_QUOTATIONS.values())
    assert _hits(_ADVISER, {"CHANGELOG.md": _without_quotations(section)}) == {}


@pytest.mark.criterion("REQ-FUND-052")
def test_the_changelog_scan_flags_prose_beside_a_quotation() -> None:
    """Validates ``_without_quotations`` behind
    test_unreleased_changelog_never_calls_the_user_an_adviser: removing a quotation
    leaves the prose around it to be scanned."""
    entry = 'never "the adviser"; the adviser edit is unchanged'

    assert _hits(_ADVISER, {"entry": _without_quotations(entry)}) == {
        "entry": ["adviser"]
    }


@pytest.mark.criterion("REQ-FUND-052")
def test_developer_prose_never_calls_the_user_an_adviser() -> None:
    """Validates every comment and docstring under fund/src/fund and
    packages/portopt-db/src/portopt_db, the source the Ask agent stages as ``/code``
    and quotes to the user: none says adviser or advisor, in any number or
    possessive. Identifiers that keep the old name are not prose and are not read."""
    index = _developer_prose_index()
    kinds = {kind for prose in index.values() for kind, *_ in prose}
    # A scan that read nothing, or only one kind of prose, would pass for nothing.
    assert len(index) >= 100
    assert kinds == {"comment", "docstring"}

    hits = {
        key: found
        for path, prose in index.items()
        for key, found in _adviser_prose_hits(prose, path).items()
    }

    assert hits == {}


@pytest.mark.criterion("REQ-FUND-052")
def test_the_developer_prose_scan_reads_comments_and_docstrings_only() -> None:
    """Validates ``_adviser_prose_hits`` behind
    test_developer_prose_never_calls_the_user_an_adviser: a comment and a docstring
    that say adviser are found at the line that says it, and identifiers that keep
    the old name are not."""
    source = (
        '"""Module for adviser_edit."""\n'
        "\n"
        "\n"
        "def adviser_edit() -> None:\n"
        '    """Shown to the adviser."""\n'
        "    current_adviser_choice = _NO_ADVISER_CHOICE  # adviser_edit stays\n"
        "    # the adviser approves\n"
        "\n"
        "\n"
        "def gate() -> None:\n"
        '    """Pause the run.\n'
        "\n"
        "    The adviser's choice stays in force.\n"
        '    """\n'
    )

    hits = _adviser_prose_hits(_developer_prose(source), "synthetic")

    assert hits == {
        "synthetic:5": "docstring: adviser",
        "synthetic:7": "comment: adviser",
        "synthetic:13": "docstring: adviser",
    }


@pytest.mark.criterion("REQ-FUND-052")
@pytest.mark.parametrize(
    "prose",
    [
        "the adviser approves",
        "The Adviser approves",
        "the adviser's choice",
        "both advisers agree",
        "the advisers' vote",
        "an advisor",
        "the advisor's note",
        "advisors",
        "the adviser-confirmation gate",
        "``adviser``",
        "ADVISER DIRECTIVE",
    ],
)
def test_the_developer_prose_pattern_flags_each_spelling(prose: str) -> None:
    """Validates ``_ADVISER_PROSE``: the singular, plural and possessive of both
    spellings are caught, whatever the case or the punctuation beside them."""
    assert _ADVISER_PROSE.search(prose)


@pytest.mark.criterion("REQ-FUND-052")
@pytest.mark.parametrize(
    "name",
    [
        "adviser_edit",
        "current_adviser_choice",
        "adviser_choice_in_force",
        "_NO_ADVISER_CHOICE",
        "AdviserChoice",
        "adviser2",
        "advisory",
    ],
)
def test_the_developer_prose_pattern_leaves_identifiers_and_other_words(
    name: str,
) -> None:
    """Validates ``_ADVISER_PROSE``: an identifier that keeps the old name, and a
    different word that shares its stem, are not flagged."""
    assert _ADVISER_PROSE.search(name) is None
