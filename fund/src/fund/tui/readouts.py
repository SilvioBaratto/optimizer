"""Structured read-outs for the Deliberation detail pane.

Turn a recorded agent decision's payload into readable, plain-text sub-panels — what
the agent decided and why — in place of the raw JSON dump. Import-light by contract:
stdlib only at runtime (no ``optimizer`` / ``fund.tools`` / agent stack), so importing
:mod:`fund.tui.app` stays agent-stack-light. Live recompute (frontier / backtest / risk
verdict) is the view worker's job; this module only formats data already read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    # Annotation-only: these recompute dataclasses live in ``fund.readouts_compute``,
    # which imports ``optimizer`` at its top. Importing them for real would break the
    # import-light contract, so this module duck-types ``compute`` at runtime (the
    # helper params below are ``Any``) and only names the types for the checker.
    from fund.readouts_compute import ResultsCompute, RiskCompute

# Keyed on the acting role, not the step, so the replay ``optimize_portfolio`` decision
# and any future allocator step both route to the build/results read-out.
_ALLOCATOR_AGENT = "allocator"
_RISK_AGENT = "risk"

# Signals / factors / views are never generated or persisted (the ``views`` column is
# always NULL); label them rather than fabricate numbers.
_MOCK_LABEL = "mock — no live source yet"

# Cap inline names so a wide universe stays legible; the rest collapse to a count.
_MAX_NAMES = 8

# Frontier / backtest are live-computable (not aspirational): mark them pending until
# the adviser expands the step and the recompute worker fills them in (T6).
_PENDING_LABEL = "compute to view"

# Risk verdict tiers as (dot, label) — the word always travels with the glyph so status
# never rides on colour/shape alone (mirrors views._HEALTH_*). The live ok/warn/breach
# tiers derive from risk_check violations in T6; it is pending until expanded.
_VERDICT_PENDING = ("○", "pending")


def render_readout(
    agent: str,
    step: str | None,
    payload: Mapping[str, Any] | None,
    compute: ResultsCompute | RiskCompute | None = None,
) -> str | None:
    """Render an agent step's recorded artifacts as plain-text sub-panels.

    Args:
        agent: The acting agent (e.g. ``"allocator"``).
        step: The step identity, or ``None``.
        payload: The recorded decision payload, or ``None``.
        compute: The worker's recompute result — a ``ResultsCompute`` for an allocator
            step or a ``RiskCompute`` for a risk step — which fills the frontier /
            backtest / verdict / sector lines. ``None`` leaves them as pending markers.

    Returns:
        The rendered sub-panels joined by newlines, or ``None`` when the step has no
        structured read-out — the caller then falls back to the raw JSON dump.
    """
    if agent == _ALLOCATOR_AGENT and (payload or compute is not None):
        safe = payload or {}
        return "\n".join([*_build_block(safe), *_results_block(safe, compute)])
    if agent == _RISK_AGENT:
        return "\n".join(_risk_block(compute))
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


def _results_block(payload: Mapping[str, Any], compute: Any = None) -> list[str]:
    """The *results* family: recorded weights + the frontier point and backtest.

    Args:
        payload: The allocator decision payload.
        compute: A ``ResultsCompute`` from the worker, or ``None`` for pending markers.

    Returns:
        The ``results`` sub-panel lines.
    """
    lines = ["results"]
    lines += _indent(_weights_lines(payload))
    lines += _indent(_frontier_lines(compute))
    lines += _indent(_backtest_lines(compute))
    return lines


def _frontier_lines(compute: Any) -> list[str]:
    """The frontier point: pending until recomputed, then the solved point."""
    if compute is None:
        return [f"frontier point · {_PENDING_LABEL}"]
    if compute.frontier is None:
        return [f"frontier point · unavailable ({compute.frontier_error})"]
    point = compute.frontier
    return [
        f"frontier point · mean {_num(point.mean)}"
        f" · vol {_num(point.standard_deviation)}"
        f" · sharpe {_num(point.sharpe_ratio)}"
    ]


def _backtest_lines(compute: Any) -> list[str]:
    """The backtest: pending until recomputed, then its metrics + window summary."""
    if compute is None:
        return [f"backtest · {_PENDING_LABEL}"]
    if compute.backtest is None:
        return [f"backtest · unavailable ({compute.backtest_error})"]
    result = compute.backtest
    mode = "out-of-sample" if result.out_of_sample else "in-sample"
    return [
        f"backtest · sharpe {_num(result.sharpe_ratio)}"
        f" · maxDD {_num(result.max_drawdown)}",
        f"  {result.n_observations} obs · {result.n_folds} folds · {mode}"
        f" · {result.window_start} → {result.window_end}",
    ]


def _risk_block(compute: Any = None) -> list[str]:
    """The *risk* family: the verdict + the exposures that drove it.

    The verdict is pending (dot+label) until the worker re-runs ``risk_check`` and
    :func:`risk_verdict` maps its violations to ok/warn/breach — never the risk role's
    prose. Factor exposures stay mock.

    Args:
        compute: A ``RiskCompute`` from the worker, or ``None`` for pending markers.

    Returns:
        The ``risk`` sub-panel lines.
    """
    lines = ["risk"]
    lines += _indent(_verdict_lines(compute))
    lines += _indent(_sector_lines(compute))
    lines += _indent([f"factor exposures · {_MOCK_LABEL}"])
    return lines


def _verdict_lines(compute: Any) -> list[str]:
    """The risk verdict: pending, unavailable on tool failure, else the live tier."""
    if compute is None:
        dot, label = _VERDICT_PENDING
        return [f"verdict · {dot} {label} · {_PENDING_LABEL}"]
    if compute.error is not None or compute.passed is None:
        return [f"verdict · unavailable ({compute.error})"]
    verdict = risk_verdict(list(compute.violations))
    header = f"verdict · {verdict.dot} {verdict.label}"
    return [header, *(f"  {line}" for line in verdict.lines)]


def _sector_lines(compute: Any) -> list[str]:
    """Sector exposures: pending, unavailable without a mapping, else the roll-up."""
    if compute is None:
        return [f"sector exposures · {_PENDING_LABEL}"]
    if compute.sector_exposures is None:
        return ["sector exposures · unavailable"]
    lines = ["sector exposures"]
    lines += [
        f"  {sector} · {_pct(weight)}" for sector, weight in compute.sector_exposures
    ]
    return lines


@dataclass(frozen=True)
class RiskVerdict:
    """A risk read-out's rendered verdict, derived from ``risk_check`` violations.

    Attributes:
        dot: Status glyph (``●`` ok/breach, ``◐`` warn) — never shown without ``label``.
        label: The tier word, so a reader never relies on the glyph alone.
        tier: The machine tier (``ok``/``warn``/``breach``) — a headless-test hook.
        lines: The violation lines that drove the verdict, pre-rendered.
    """

    dot: str
    label: str
    tier: str
    lines: tuple[str, ...]


# breach ⊐ warn ⊐ ok: one hard-bound violation outranks any number of budget ones,
# because a name/exposure breach is actionable where an off-target budget is advisory.
_HARD = frozenset({"min_weight", "max_weight", "non_finite"})


def risk_verdict(violations: list[dict[str, Any]]) -> RiskVerdict:
    """Map ``risk_check`` violations to the tri-state the drill-down renders.

    Args:
        violations: The ``data["violations"]`` list from ``fund.tools.risk.risk_check``.

    Returns:
        The rendered :class:`RiskVerdict`; empty ``violations`` is ``● ok``, a sole
        off-target budget is ``◐ warn``, and any hard-bound violation is ``● breach``.
    """
    lines = tuple(_violation_line(v) for v in violations)
    if any(v.get("type") in _HARD for v in violations):
        return RiskVerdict("●", "breach", "breach", lines)
    if any(v.get("type") == "budget" for v in violations):
        return RiskVerdict("◐", "warn", "warn", lines)
    return RiskVerdict("●", "ok", "ok", lines)


def _violation_line(violation: Mapping[str, Any]) -> str:
    """One readable line for a ``risk_check`` violation (type, asset/target, value)."""
    kind = str(violation.get("type", "?"))
    if kind == "budget":
        value = _num(violation.get("value"))
        return f"budget · {value} vs {_num(violation.get('target'))}"
    asset = str(violation.get("asset", "?"))
    limit = violation.get("limit")
    tail = f" (limit {_num(limit)})" if limit is not None else ""
    return f"{kind} · {asset} {_num(violation.get('value'))}{tail}"


def _weights_lines(payload: Mapping[str, Any]) -> list[str]:
    """Render the recorded weights ranked by size, capped to the top names."""
    weights = payload.get("weights") or {}
    if not weights:
        return ["weights · none recorded"]
    # Null / non-numeric weights (valid JSON from an LLM) sink to the bottom rather
    # than raising from the mixed-type comparison the bare sort would attempt.
    ranked = sorted(
        weights.items(), key=lambda item: _sort_weight(item[1]), reverse=True
    )
    lines = [f"weights · {len(ranked)} names (top by weight)"]
    lines += [f"  {name} · {_pct(weight)}" for name, weight in ranked[:_MAX_NAMES]]
    extra = len(ranked) - _MAX_NAMES
    if extra > 0:
        lines.append(f"  … +{extra} more")
    return lines


def _pct(value: Any) -> str:
    """Format a portfolio weight as a one-decimal percentage, else its raw string.

    A non-numeric or boolean weight renders raw (e.g. ``"None"``, ``"True"``) so a type
    anomaly in the recorded JSON is visible, never silently shown as a percentage.
    """
    numeric = _as_float(value)
    return str(value) if numeric is None else f"{numeric * 100:.1f}%"


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
    numeric = _as_float(value)
    return str(value) if numeric is None else f"{numeric:.2f}"


def _as_float(value: Any) -> float | None:
    """Coerce a JSON scalar to ``float``, or ``None`` for bool / null / non-numbers.

    ``bool`` is rejected despite subclassing ``int`` so a JSON ``true`` is never read as
    the number ``1`` (mirrors ``observe._metrics_from_response``'s bool guard).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _sort_weight(value: Any) -> float:
    """Sort key for a weight: its float value, or ``-inf`` for null / non-numeric."""
    numeric = _as_float(value)
    return float("-inf") if numeric is None else numeric


# Eighth-block ramp for the attribution meter: index == eighths filled (0 is a blank
# cell, 8 a full block), so a magnitude maps to full blocks plus one partial ramp cell.
_METER_CELLS = " ▏▎▍▌▋▊▉█"


def attribution_bar(value: float, scale: float, width: int = 10) -> str:
    """Render one attribution contribution as a left-aligned eighth-block meter.

    The bar length is proportional to ``|value| / scale`` and never exceeds ``width``
    full cells, so the largest-magnitude contribution fills the bar. A negative
    contribution carries a trailing ``(-)`` marker — the sign never rides on colour
    alone (mirrors ``views._HEALTH_*``).

    Args:
        value: The signed contribution (a factor contribution or the residual).
        scale: The normalising magnitude — ``max(abs(...))`` over the contributions and
            residual. A non-positive scale yields an empty meter.
        width: The full-cell length of a bar at ``|value| == scale``.

    Returns:
        The meter string (eighth-block cells, left-aligned), with ``" (-)"`` appended
        when ``value`` is negative; an empty string for a zero magnitude.
    """
    fraction = abs(value) / scale if scale > 0 else 0.0
    eighths = round(min(fraction, 1.0) * width * 8)
    bar = "█" * (eighths // 8)
    remainder = eighths % 8
    if remainder:
        bar += _METER_CELLS[remainder]
    if value < 0:
        bar += " (-)"
    return bar
