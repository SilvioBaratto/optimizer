"""Jinja-based ``report.md`` renderer for the research pipeline (Cycle 5 §12).

Consolidates a research run into one Markdown artefact: regime + tilts,
factor IS/OOS IC, optimizer config diff, binding constraints, Top-4
retighten trace, hybrid rebalance decision, 17-rule checklist, gross /
net / after-tax metrics, and chart links.

Public surface
--------------
- :func:`render_report` — renders ``research/output/report.md`` (or any
  ``output_dir``) from the bundled Jinja template.
- :func:`compute_binding_constraints` — labels rows where
  ``A @ w - b >= -tol`` (within numeric tolerance, default ``1e-4``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    import pandas as pd

    from optimizer.factors import FactorValidationReport

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.md.j2"
_REPORT_FILENAME = "report.md"


def render_report(
    *,
    output_dir: Path,
    regime: str,
    tilts: dict[str, float],
    validation_report: FactorValidationReport,
    oos_per_fold_ic: pd.DataFrame,
    optimizer_diff: dict[str, Any],
    binding_constraints: list[str],
    retighten_trace: list[dict[str, Any]],
    rebalance_decision: tuple[bool, str],
    checklist_rows: list[dict[str, Any]],
    metrics: dict[str, dict[str, float]],
    chart_paths: list[Path],
    assembly_hash: str,
) -> Path:
    """Render the research-run report.md to ``output_dir / 'report.md'``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_ic_rows = _build_factor_ic_rows(validation_report, oos_per_fold_ic)
    env = _build_environment()
    template = env.get_template(_TEMPLATE_NAME)
    text = template.render(
        regime=regime,
        tilts=tilts,
        factor_ic_rows=factor_ic_rows,
        optimizer_diff=optimizer_diff,
        binding_constraints=binding_constraints,
        retighten_trace=retighten_trace,
        rebalance_decision=rebalance_decision,
        checklist_rows=checklist_rows,
        metrics=metrics,
        chart_paths=chart_paths,
        assembly_hash=assembly_hash,
    )
    path = output_dir / _REPORT_FILENAME
    path.write_text(text)
    return path


def compute_binding_constraints(
    A: np.ndarray,  # noqa: N803  -- sklearn-style matrix name
    b: np.ndarray,
    weights: np.ndarray,
    *,
    labels: list[str] | None = None,
    tol: float = 1e-4,
) -> list[str]:
    """Return labels for constraint rows where ``A @ w - b >= -tol``.

    The slack ``s = b - A @ w`` is non-negative for any feasible point;
    rows with ``s <= tol`` are treated as binding.  Each binding row is
    annotated with the realised LHS value (``w_i = A_i @ w``).
    """
    a_arr = np.asarray(A, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    lhs = a_arr @ w_arr
    slack = b_arr - lhs
    binding_idx = np.where(slack <= tol)[0]
    out: list[str] = []
    for i in binding_idx:
        base = labels[i] if labels is not None else f"row_{i}"
        out.append(f"{base} (binding, w={lhs[i]:.4f})")
    return out


def _build_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(default_for_string=False, default=False),
        keep_trailing_newline=True,
    )


def _build_factor_ic_rows(
    validation_report: FactorValidationReport,
    oos_per_fold_ic: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Merge IS results with OOS column means into one row dict per factor."""
    oos_mean = oos_per_fold_ic.mean(axis=0).to_dict()
    rows: list[dict[str, Any]] = []
    for r in validation_report.ic_results:
        rows.append(
            {
                "factor": r.factor_name,
                "is_mean_ic": r.mean_ic,
                "is_t_stat": r.t_stat,
                "is_significant": r.significant,
                "oos_mean_ic": float(oos_mean.get(r.factor_name, float("nan"))),
            }
        )
    return rows


__all__ = ["compute_binding_constraints", "render_report"]
