"""The text a fund run puts in front of its agents never picks a candidate.

The four candidates are presented side by side and the user alone chooses one,
so nothing the run's agents read may rank one or call one recommended, best or
matching the user: not the role prompts, the role descriptions, the run and
respond instructions, the bound tools' docstrings, nor the role skills. The same
text must tell the agents to present the candidates without choosing.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from _fund_fakes import ASOF, PORTFOLIO_ID, make_constraint_set, make_mandate

from fund.agents import graph
from fund.agents.graph_assembly import _general_purpose_subagent
from fund.agents.graph_hitl import _RESPOND_FRAME
from fund.agents.graph_prompts import _run_instruction
from fund.agents.prompts import (
    ALLOCATOR_SYSTEM_PROMPT,
    ECONOMIST_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    PM_SYSTEM_PROMPT,
    RISK_SYSTEM_PROMPT,
)
from fund.agents.skills import SKILLS_BY_AGENT, skill_paths
from fund.agents.toolsets import TOOLS_BY_AGENT, RunContext, bind_toolset
from fund.config import settings
from fund.tools.eligibility import Eligibility

_RECOMMENDATION = re.compile(
    r"recommend|best|prefer|suggest|optimal|ideal|winner"
    r"|matching the (client|profile|user|inputs)",
    re.IGNORECASE,
)
_RUN_ROLES = ("orchestrator", "economist", "allocator", "risk", "executor")
# A mandate with standing directives and an eligible universe, so the run instruction
# carries the directives and eligibility blocks too.
_DIRECTIVES = ("Keep 5% cash.", "Avoid energy.")
_ALLOWED = ("AAA", "BBB")
_BARRED = (("CCC", "delisted"),)
_NO_CHOICE = "never choose"
_USER_CHOOSES = "the user chooses"


def _bound_tool_docstrings() -> dict[str, str]:
    # Binding builds closures only; no tool runs, so the context needs no database.
    ctx = RunContext(
        session=None,  # type: ignore[arg-type]
        asof=ASOF,
        store=None,
        config=settings,
        run_id=uuid.uuid4(),
        portfolio_id=PORTFOLIO_ID,
    )
    return {
        f"tool {tool.name}": tool.description
        for role in TOOLS_BY_AGENT
        for tool in bind_toolset(role, ctx)
    }


def _skill_texts() -> dict[str, str]:
    return {
        f"skill {path.parent.name}/{path.name}": path.read_text(encoding="utf-8")
        for role in _RUN_ROLES
        for directory in skill_paths(role)
        for path in sorted(Path(directory).glob("*.md"))
    }


def _fund_run_text() -> dict[str, str]:
    return {
        "PM_SYSTEM_PROMPT": PM_SYSTEM_PROMPT,
        "ECONOMIST_SYSTEM_PROMPT": ECONOMIST_SYSTEM_PROMPT,
        "ALLOCATOR_SYSTEM_PROMPT": ALLOCATOR_SYSTEM_PROMPT,
        "RISK_SYSTEM_PROMPT": RISK_SYSTEM_PROMPT,
        "EXECUTOR_SYSTEM_PROMPT": EXECUTOR_SYSTEM_PROMPT,
        "economist description": graph._ECONOMIST_DESCRIPTION,
        "allocator description": graph._ALLOCATOR_DESCRIPTION,
        "risk description": graph._RISK_DESCRIPTION,
        "executor description": graph._EXECUTOR_DESCRIPTION,
        "general-purpose description": _general_purpose_subagent(settings)[
            "description"
        ],
        "run instruction": _run_instruction(
            make_mandate(),
            str(PORTFOLIO_ID),
            ASOF,
            constraint_set=make_constraint_set(),
        ),
        "run instruction with directives and an eligible universe": _run_instruction(
            make_mandate().model_copy(update={"remembered_directives": _DIRECTIVES}),
            str(PORTFOLIO_ID),
            ASOF,
            constraint_set=make_constraint_set(),
            eligible=Eligibility(allowed=_ALLOWED, barred=_BARRED),
        ),
        "respond frame": _RESPOND_FRAME,
        **_bound_tool_docstrings(),
        **_skill_texts(),
    }


@pytest.mark.criterion("REQ-FUND-052")
def test_fund_run_text_has_no_recommendation_language() -> None:
    """Validates fund.agents.prompts, the role descriptions in fund.agents.graph, the
    run and respond instructions, the docstrings of every tool
    fund.agents.toolsets.bind_toolset binds and the role skills: none ranks a
    candidate or calls one recommended, best, optimal or matching the user, and the
    PM, allocator, risk and executor prompts and the ``place_orders`` tool tell the
    agents the user chooses while they never choose."""
    texts = _fund_run_text()
    assert set(SKILLS_BY_AGENT) >= set(_RUN_ROLES)

    hits = {
        source: sorted({match.group(0) for match in _RECOMMENDATION.finditer(text)})
        for source, text in texts.items()
        if _RECOMMENDATION.search(text)
    }
    assert hits == {}
    for source in (
        "PM_SYSTEM_PROMPT",
        "ALLOCATOR_SYSTEM_PROMPT",
        "RISK_SYSTEM_PROMPT",
        "EXECUTOR_SYSTEM_PROMPT",
    ):
        assert _NO_CHOICE in texts[source].lower(), source
    assert _USER_CHOOSES in texts["PM_SYSTEM_PROMPT"].lower()
    assert _USER_CHOOSES in texts["tool place_orders"].lower()
