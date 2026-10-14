"""Structured read-outs for the Deliberation detail pane.

Turn a recorded agent decision's payload into readable, plain-text sub-panels — what
the agent decided and why — in place of the raw JSON dump. Import-light by contract:
stdlib only at runtime (no ``optimizer`` / ``fund.tools`` / agent stack), so importing
:mod:`fund.tui.app` stays agent-stack-light. Live recompute (frontier / backtest / risk
verdict) is the view worker's job; this module only formats data already read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

# Keyed on the acting role, not the step, so the replay ``optimize_portfolio`` decision
# and any future allocator step both route to the build/results read-out.
_ALLOCATOR_AGENT = "allocator"

# Signals / factors / views are never generated or persisted (the ``views`` column is
# always NULL); label them rather than fabricate numbers.
_MOCK_LABEL = "mock — no live source yet"

# Cap inline names so a wide universe stays legible; the rest collapse to a count.
_MAX_NAMES = 8


def render_readout(
    agent: str, step: str | None, payload: Mapping[str, Any] | None
) -> str | None:
    """Render an agent step's recorded artifacts as plain-text sub-panels.

    Args:
        agent: The acting agent (e.g. ``"allocator"``).
        step: The step identity, or ``None``.
        payload: The recorded decision payload, or ``None``.

    Returns:
        The rendered sub-panels joined by newlines, or ``None`` when the step has no
        structured read-out — the caller then falls back to the raw JSON dump.
    """
    if agent == _ALLOCATOR_AGENT:
        return "\n".join(_build_block(payload or {}))
    return None


def _build_block(payload: Mapping[str, Any]) -> list[str]:
    """The *build* family: the universe, objective, and constraints the allocator used.

    Args:
        payload: The allocator decision payload.

    Returns:
        The ``build`` sub-panel lines; aspirational sources are labelled mock.
    """
    lines = ["build"]
    lines += _indent(_universe_lines(payload))
    lines += _indent(_constraints_lines(payload.get("constraint_set")))
    lines += _indent(_mock_lines())
    return lines


def _universe_lines(payload: Mapping[str, Any]) -> list[str]:
    """Render the post-filter universe (size + names) from the recorded weights keys."""
    names = list(payload.get("weights") or {})
    shown = ", ".join(names[:_MAX_NAMES])
    extra = len(names) - _MAX_NAMES
    if extra > 0:
        shown = f"{shown}, … +{extra} more"
    tail = f" · {shown}" if shown else ""
    return [f"universe (post-filter) · {len(names)} selected{tail}"]


def _constraints_lines(constraint_set: Mapping[str, Any] | None) -> list[str]:
    """Render the objective and the solver constraints from the recorded constraint set.

    Args:
        constraint_set: ``ConstraintSet.model_dump(mode="json")``, or ``None`` when the
            allocator decision recorded no constraint set.

    Returns:
        The objective/bounds/regularisation/ESG lines, or a single ``unavailable`` line.
    """
    if not constraint_set:
        return ["objective · unavailable (not recorded)"]
    cs = constraint_set
    bounds = cs.get("bounds") or {}
    cardinality = cs.get("cardinality")
    esg = cs.get("esg") or {}
    return [
        f"objective · {cs.get('objective', '—')} · risk {cs.get('risk_measure', '—')}",
        f"risk aversion a_gamma · {_num(cs.get('a_gamma'))}"
        f" · beta {_num(cs.get('beta'))}",
        f"bounds · min {_num(bounds.get('min_weights'))}"
        f" · max {_num(bounds.get('max_weights'))}"
        f" · budget {_num(bounds.get('budget'))}",
        f"regularisation · l1 {_num(cs.get('l1_coef'))} · l2 {_num(cs.get('l2_coef'))}"
        f" · cardinality {'none' if cardinality is None else cardinality}",
        f"esg · {len(esg.get('exclusions') or [])} exclusions",
    ]


def _mock_lines() -> list[str]:
    """Label the aspirational build sources that have no live data source yet."""
    return [
        f"candidate universe (pre-filter) · {_MOCK_LABEL}",
        f"factors / signals · {_MOCK_LABEL}",
        f"views · {_MOCK_LABEL}",
    ]


def _indent(lines: list[str]) -> list[str]:
    return [f"  {line}" for line in lines]


def _num(value: Any) -> str:
    """Format a numeric constraint to two decimals; fall back to its string form."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
