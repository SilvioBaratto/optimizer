"""User-visible text never frames the software as advice or an advising service.

The fund computes candidate portfolios that the person running it reviews and
approves themselves, on paper. Nothing it shows that person, nor anything its agents
read, may present it as a suitability assessment, an investment adviser, a fund
manager or an agent that chooses the portfolio, and the person is "you" or "the user",
never "the adviser". Identifiers, stored data, CLI flags and developer docstrings keep
their names, so the scans below read only the text a person or an agent sees.
"""

from __future__ import annotations

import ast
import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from _fund_fakes import ASOF, PORTFOLIO_ID

from fund.agents.ask_agent import build_ask_tools
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext, bind_toolset
from fund.config import settings

_REPO = Path(__file__).resolve().parents[3]
_FUND_SRC = _REPO / "fund" / "src" / "fund"
_SKILLS = _FUND_SRC / "skills"
_READMES = (_REPO / "README.md", _REPO / "fund" / "README.md")
_CLAUDE_MD = _REPO / "CLAUDE.md"

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
)
_ADVISER = re.compile(r"\badvisers?\b", re.IGNORECASE)
# Each entry quotes the exact text that may keep the word, with the reason it does.
_ADVISER_ALLOWED: dict[str, str] = {}
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


def _string_literals(path: Path, *, docstrings: bool) -> Iterator[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skipped = set() if docstrings else _docstring_nodes(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
        ):
            yield node.lineno, node.value


def _python_sources() -> list[Path]:
    return sorted(_FUND_SRC.rglob("*.py"))


def _markdown_sources() -> list[Path]:
    return [*sorted(_SKILLS.rglob("*.md")), *_READMES]


def _phrase_haystacks() -> dict[str, str]:
    # Raw text finds a phrase in a comment or across a wrapped line; the parsed
    # literals find one split over implicitly concatenated string parts.
    haystacks = {
        str(path.relative_to(_REPO)): _normalise(
            "\n".join(
                [
                    path.read_text(encoding="utf-8"),
                    *(text for _, text in _string_literals(path, docstrings=True)),
                ]
            )
        )
        for path in _python_sources()
    }
    for path in (*_markdown_sources(), _CLAUDE_MD):
        haystacks[str(path.relative_to(_REPO))] = _normalise(
            path.read_text(encoding="utf-8")
        )
    return haystacks


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
    return {f"tool {tool.name}": tool.description for tool in (*run_tools, *ask_tools)}


def _user_visible_texts() -> dict[str, str]:
    texts = {
        f"{path.relative_to(_REPO)}:{line}": text
        for path in _python_sources()
        for line, text in _string_literals(path, docstrings=False)
    }
    texts.update(_tool_descriptions())
    for path in _markdown_sources():
        texts[str(path.relative_to(_REPO))] = path.read_text(encoding="utf-8")
    return texts


@pytest.mark.criterion("REQ-FUND-052")
@pytest.mark.parametrize("phrase", _OLD_PHRASES)
def test_old_user_visible_phrases_are_gone(phrase: str) -> None:
    """Validates the user-visible text of fund (the TUI, the CLI, the agent prompts,
    the tool and role descriptions, the skills) and README.md, fund/README.md and
    CLAUDE.md: none keeps a phrase that framed the fund as a suitability service, an
    investment adviser, a fund manager or an LLM that chooses."""
    needle = phrase.lower()
    hits = sorted(
        source for source, text in _phrase_haystacks().items() if needle in text.lower()
    )
    assert hits == []


@pytest.mark.criterion("REQ-FUND-052")
def test_user_visible_text_never_calls_the_user_an_adviser() -> None:
    """Validates every non-docstring string literal under fund/src/fund, the
    docstring of every ``@tool`` there and the descriptions of the tools
    fund.agents.toolsets.bind_toolset and fund.agents.ask_agent.build_ask_tools
    bind, the skills, README.md and
    fund/README.md: the person running the tool is never called an adviser, outside
    the quoted exceptions in ``_ADVISER_ALLOWED``."""
    hits = {
        source: sorted({match.group(0).lower() for match in _ADVISER.finditer(text)})
        for source, text in _user_visible_texts().items()
        if _ADVISER.search(text)
        and not any(quote in text for quote in _ADVISER_ALLOWED)
    }
    assert hits == {}
