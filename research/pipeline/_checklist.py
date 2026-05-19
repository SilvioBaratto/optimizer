"""Step 7b — Checklist validation and terminal gate.

Extracted from ``stock_selection_pipeline.py`` lines 272–413, 1098–1436.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from research.optimization._config import _REGION_MAP
from research.pipeline._factors import _GICS_SECTORS
from research.pipeline._metrics import (
    _project_metrics,
)

console = Console()
logger = logging.getLogger(__name__)

_CHECKLIST_TOTAL = 17


# ---------------------------------------------------------------------------
# Rule primitives
# ---------------------------------------------------------------------------


def _rule(rule: str, *, ok: bool, measured: str | float, target: str) -> dict[str, Any]:
    """Build a single checklist rule result dict."""
    return {"rule": rule, "pass": bool(ok), "measured": measured, "target": target}


def _sector_weights(
    all_weights: list[tuple[str, float]], sector_mapping: dict[str, str]
) -> dict[str, float]:
    from collections import defaultdict

    out: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        out[sector_mapping.get(ticker, "Unknown")] += w
    return dict(out)


def _country_weights(
    all_weights: list[tuple[str, float]], country_map: dict[str, str]
) -> dict[str, float]:
    from collections import defaultdict

    out: dict[str, float] = defaultdict(float)
    for ticker, w in all_weights:
        out[country_map.get(ticker, "Unknown")] += w
    return dict(out)


def _sector_lookup(sector_w: dict[str, float], *names: str) -> float:
    """Sum sector weights across alternative spellings."""
    return sum(sector_w.get(n, 0.0) for n in names)


def _eval_metric_threshold(
    metrics: dict[str, dict[str, float]],
    label: str,
    key: str,
    rule: str,
    target: str,
    *,
    pass_pred: Any,
    fmt: str = "{:.3f}",
) -> dict[str, Any]:
    """Evaluate a metric-bound rule with NaN → pass=False, measured='N/A'."""
    value = metrics.get(label, {}).get(key, float("nan"))
    if isinstance(value, float) and np.isnan(value):
        return _rule(rule, ok=False, measured="N/A", target=target)
    return _rule(rule, ok=pass_pred(value), measured=fmt.format(value), target=target)


# ---------------------------------------------------------------------------
# JSON projection helpers
# ---------------------------------------------------------------------------


def _project_rule_for_json(rule: dict[str, Any]) -> dict[str, Any]:
    """Convert checklist rule dict to JSON-safe form (NaN floats → null)."""
    measured = rule.get("measured")
    if isinstance(measured, float) and np.isnan(measured):
        measured = None
    return {
        "rule": rule["rule"],
        "pass": bool(rule["pass"]),
        "measured": measured,
        "target": rule["target"],
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_metrics_json(
    metrics_by_label: dict[str, dict[str, float]], output_dir: Path
) -> Path:
    """Persist Cycle 4 §9.3 metrics block to ``metrics.json`` (Cycle 5 input)."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "rf_assumption": "FRED DGS3MO daily forward-fill ÷ 252",
    }
    if "Portfolio" in metrics_by_label:
        payload["net_of_cost"] = _project_metrics(metrics_by_label["Portfolio"])
    if "Portfolio (after-tax)" in metrics_by_label:
        payload["after_tax"] = _project_metrics(
            metrics_by_label["Portfolio (after-tax)"]
        )
    out_path = output_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path


def write_checklist_json(
    *,
    rules: list[dict[str, Any]],
    gross_metrics: dict[str, float] | None,
    net_metrics: dict[str, float] | None,
    after_tax_metrics: dict[str, float] | None,
    output_dir: Path,
) -> Path:
    """Persist Cycle 4 §10 checklist results to ``checklist.json``."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in rules if r.get("pass"))
    payload: dict[str, Any] = {
        "rules": [_project_rule_for_json(r) for r in rules],
        "summary": {"passed": passed, "total": len(rules)},
        "breakdown": {
            "gross": _project_metrics(gross_metrics or {}),
            "net_of_cost": _project_metrics(net_metrics or {}),
            "after_tax": _project_metrics(after_tax_metrics or {}),
        },
    }
    out_path = output_dir / "checklist.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path


def write_weights_csv(weights: pd.Series, output_dir: Path) -> Path:
    """Persist final portfolio weights to ``weights.csv`` sorted desc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_w = weights.sort_values(ascending=False)
    df = pd.DataFrame(
        {"ticker": list(sorted_w.index), "weight": sorted_w.to_numpy(dtype=float)}
    )
    out_path = output_dir / "weights.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_failure_table(failed_rules: list[dict[str, Any]]) -> None:
    """Print the measured-vs-target table for failing checklist rules only."""
    table = Table(
        title="Failing Rules",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Rule", style="dim", width=42)
    table.add_column("Target", justify="right")
    table.add_column("Measured", justify="right")
    for r in failed_rules:
        table.add_row(r["rule"], r["target"], str(r["measured"]))
    console.print(table)


def _render_checklist_table(rules: list[dict[str, Any]]) -> None:
    """Render a Rich Table summary of checklist rules to stdout."""
    table = Table(
        title="Portfolio Checklist Validation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rule", style="dim", width=42)
    table.add_column("Target", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Status", justify="center")
    pass_count = 0
    for r in rules:
        status = "[green]PASS[/green]" if r["pass"] else "[red]FAIL[/red]"
        table.add_row(r["rule"], r["target"], str(r["measured"]), status)
        pass_count += int(bool(r["pass"]))
    console.print(table)
    total = len(rules)
    color = "green" if pass_count == total else "yellow" if pass_count >= 13 else "red"
    console.print(f"  [{color}]Checklist: {pass_count}/{total} passed[/{color}]")


# ---------------------------------------------------------------------------
# Terminal gate
# ---------------------------------------------------------------------------


def _apply_terminal_gate(
    *,
    rules: list[dict[str, Any]],
    weights: pd.Series,
    output_dir: Path,
) -> None:
    """Cycle 4 §10 terminal gate: 17/17 → exit 0 + weights.csv; else exit 1."""
    assert len(rules) == _CHECKLIST_TOTAL, (  # noqa: S101  -- invariant guard
        f"checklist must have exactly {_CHECKLIST_TOTAL} rules, got {len(rules)}"
    )
    pass_count = sum(1 for r in rules if r.get("pass"))
    total = len(rules)
    if pass_count == total:
        console.print(f"  [green]Checklist: {pass_count}/{total} PASS[/green]")
        weights_path = write_weights_csv(weights, output_dir)
        console.print(f"  [cyan]Saved weights:[/cyan] {weights_path}")
        raise SystemExit(0)
    failed = [r for r in rules if not r.get("pass")]
    _render_failure_table(failed)
    # Diagnostic-only dump: the blessed weights.csv is gated on 17/17, but a
    # near-miss portfolio is still worth inspecting. Distinct filename so it
    # can never be mistaken for the gate-passed artefact.
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_w = weights.sort_values(ascending=False)
    diag_path = output_dir / "weights_diagnostic.csv"
    pd.DataFrame(
        {"ticker": list(sorted_w.index), "weight": sorted_w.to_numpy(dtype=float)}
    ).to_csv(diag_path, index=False)
    console.print(
        f"  [yellow]Diagnostic weights (not gate-passed):[/yellow] {diag_path}"
    )
    console.print(
        f"  [red]Checklist: {pass_count}/{total} — "
        f"{total - pass_count} rule(s) failed[/red]"
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Checklist validation (17 rules)
# ---------------------------------------------------------------------------


def _validate_checklist(
    all_weights: list[tuple[str, float]],
    sector_mapping: dict[str, str],
    country_map: dict[str, str],
    metrics: dict[str, dict[str, float]],
    *,
    benchmark_returns: pd.Series | None,
    net_returns: pd.Series | None,
    after_tax_returns: pd.Series | None,
    cost_bps_actual: float | None,
    currency_map: dict[str, str],
    sector_bands: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the 17 §10 portfolio checklist rules.

    Returns a list of ``{"rule", "pass", "measured", "target"}`` dicts in
    deterministic order.  Rules 12-15 evaluate the after-tax series
    (``metrics["Portfolio (after-tax)"]``).  Currency-hedge advisory is
    logged only — not a rule entry.

    When ``sector_bands`` is provided (from the active macro regime), Rules 2,
    5, and 6 are tightened:

    * **Rule 2** — The per-sector cap check uses ``bands[s][1]`` instead of a
      static 15% cap.  A sector weight exceeding its regime cap triggers FAIL
      and the worst violator is reported.
    * **Rule 5** — Healthcare floor uses ``bands["Healthcare"][0]`` (regime-
      specific minimum) when the regime prescribes a non-zero floor.
    * **Rule 6** — Technology floor uses ``bands["Technology"][0]`` similarly.

    When ``sector_bands`` is ``None`` the original static thresholds are used.
    """
    rules: list[dict[str, Any]] = []
    sorted_w = sorted((w for _, w in all_weights), reverse=True)
    sector_w = _sector_weights(all_weights, sector_mapping)
    country_w = _country_weights(all_weights, country_map)
    region_w: dict[str, float] = {}
    for country, w in country_w.items():
        region_w[_REGION_MAP.get(country, "Other")] = (
            region_w.get(_REGION_MAP.get(country, "Other"), 0.0) + w
        )
    label_at = "Portfolio (after-tax)"

    # Rule 1 — region ≤ 60%
    max_region = max(region_w.values()) if region_w else 0.0
    rules.append(
        _rule(
            "No single region > 60%",
            ok=max_region <= 0.60,
            measured=f"{max_region:.1%}",
            target="≤ 60%",
        )
    )
    # Rule 2 — sector cap check.
    # When sector_bands is provided, each sector is checked against its
    # regime-specific cap; the worst violator (largest excess) is reported.
    # When sector_bands is None, the static 15% cap is used.
    if sector_bands is not None and sector_w:
        worst_sector = ""
        worst_excess = 0.0
        for s, w in sector_w.items():
            cap = sector_bands.get(s, (0.0, 0.15))[1]
            excess = w - cap
            if excess > 1e-9 and excess > worst_excess:
                worst_excess = excess
                worst_sector = s
        if worst_sector:
            worst_w = sector_w[worst_sector]
            worst_cap = sector_bands.get(worst_sector, (0.0, 0.15))[1]
            rules.append(
                _rule(
                    "No sector > regime cap",
                    ok=False,
                    measured=(f"{worst_sector}: {worst_w:.1%} > cap {worst_cap:.1%}"),
                    target="all sectors ≤ regime cap",
                )
            )
        else:
            max_sector = max(sector_w.values()) if sector_w else 0.0
            max_sector_name = (
                max(sector_w, key=lambda k: sector_w[k]) if sector_w else "N/A"
            )
            rules.append(
                _rule(
                    "No sector > regime cap",
                    ok=True,
                    measured=f"{max_sector:.1%} ({max_sector_name})",
                    target="all sectors ≤ regime cap",
                )
            )
    else:
        max_sector = max(sector_w.values()) if sector_w else 0.0
        max_sector_name = (
            max(sector_w, key=lambda k: sector_w[k]) if sector_w else "N/A"
        )
        rules.append(
            _rule(
                "No single sector > 15%",
                ok=max_sector <= 0.15,
                measured=f"{max_sector:.1%} ({max_sector_name})",
                target="≤ 15%",
            )
        )
    # Rule 3 — HHI < 0.12
    hhi = sum(w**2 for w in sorted_w)
    rules.append(
        _rule("HHI < 0.12", ok=hhi < 0.12, measured=f"{hhi:.4f}", target="< 0.12")
    )
    # Rule 4 — Top-4 < 30%
    top4 = sum(sorted_w[:4])
    rules.append(
        _rule(
            "Top-4 holdings < 30%",
            ok=top4 < 0.30,
            measured=f"{top4:.1%}",
            target="< 30%",
        )
    )
    # Rule 5 — Health Care exposure (Yahoo: "Healthcare"; GICS: "Health Care")
    # When sector_bands is provided, use the regime-specific Healthcare floor;
    # otherwise fall back to the static 8% minimum.
    health_w = _sector_lookup(sector_w, "Healthcare", "Health Care")
    if sector_bands is not None:
        health_floor = sector_bands.get("Healthcare", (0.08, 1.0))[0]
        # Use max of regime floor and static floor for safety
        health_floor = max(health_floor, 0.0)
        health_target = f"≥ {health_floor:.0%}"
        health_ok = health_w >= health_floor
    else:
        health_floor = 0.08
        health_target = "≥ 8%"
        health_ok = health_w >= health_floor
    rules.append(
        _rule(
            "Health Care exposure ≥ regime floor",
            ok=health_ok,
            measured=f"{health_w:.1%}",
            target=health_target,
        )
    )
    # Rule 6 - Information Technology exposure
    # Yahoo: "Technology"; GICS: "Information Technology"
    # When sector_bands is provided, use the regime-specific Technology floor.
    tech_w = _sector_lookup(sector_w, "Technology", "Information Technology")
    if sector_bands is not None:
        tech_floor = sector_bands.get("Technology", (0.10, 1.0))[0]
        tech_floor = max(tech_floor, 0.0)
        tech_target = f"≥ {tech_floor:.0%}"
        tech_ok = tech_w >= tech_floor
    else:
        tech_floor = 0.10
        tech_target = "≥ 10%"
        tech_ok = tech_w >= tech_floor
    rules.append(
        _rule(
            "Information Technology exposure ≥ regime floor",
            ok=tech_ok,
            measured=f"{tech_w:.1%}",
            target=tech_target,
        )
    )
    # Rule 7 - at least 8 of 11 sectors present
    # (20-stock portfolio with 2 factor-alpha signals)
    min_sectors = 8
    present = {s for s in sector_w if sector_w.get(s, 0.0) > 0.0}
    missing = [s for s in _GICS_SECTORS if s not in present]
    n_present = 11 - len(missing)
    rules.append(
        _rule(
            f"At least {min_sectors}/11 sectors present",
            ok=n_present >= min_sectors,
            measured=(f"{n_present}/11 ({', '.join(missing) or 'none'})"),
            target=f"≥ {min_sectors}/11",
        )
    )
    # Rule 8 — Single-stock cap ≤ 10%
    max_w = sorted_w[0] if sorted_w else 0.0
    rules.append(
        _rule(
            "Single-stock cap ≤ 10%",
            ok=max_w <= 0.10,
            measured=f"{max_w:.1%}",
            target="≤ 10%",
        )
    )
    # Rule 9 — Min position ≥ 2%
    min_w = sorted_w[-1] if sorted_w else 0.0
    rules.append(
        _rule(
            "Min position ≥ 2%",
            ok=min_w >= 0.02,
            measured=f"{min_w:.1%}",
            target="≥ 2%",
        )
    )
    # Rule 10 — Max drawdown > -22%
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Max Drawdown",
            "Max drawdown > -22%",
            "> -22%",
            pass_pred=lambda v: v > -0.22,
            fmt="{:.1%}",
        )
    )
    # Rule 11 — Vol ≤ benchmark vol
    p_vol = metrics.get(label_at, {}).get("Ann. Vol", float("nan"))
    b_vol = metrics.get("SPY (benchmark)", {}).get("Ann. Vol", float("nan"))
    if np.isnan(p_vol) or np.isnan(b_vol):
        rules.append(
            _rule(
                "Vol ≤ benchmark vol",
                ok=False,
                measured="N/A",
                target="≤ benchmark",
            )
        )
    else:
        rules.append(
            _rule(
                "Vol ≤ benchmark vol",
                ok=p_vol <= b_vol,
                measured=f"{p_vol:.1%} vs {b_vol:.1%}",
                target="≤ benchmark",
            )
        )
    # Rule 12 — Sharpe ∈ (1.0, 2.0)
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Sharpe (rf)",
            "Sharpe ∈ (1.0, 2.0)",
            "∈ (1.0, 2.0)",
            pass_pred=lambda v: 1.0 < v < 2.0,
        )
    )
    # Rule 13 — Sortino > 1.5
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Sortino",
            "Sortino > 1.5",
            "> 1.5",
            pass_pred=lambda v: v > 1.5,
        )
    )
    # Rule 14 — IR > 0.5
    rules.append(
        _eval_metric_threshold(
            metrics,
            label_at,
            "Info Ratio",
            "Info Ratio > 0.5",
            "> 0.5",
            pass_pred=lambda v: v > 0.5,
        )
    )
    # Rule 15 — Downside vol < 75% × total vol
    if np.isnan(p_vol):
        rules.append(
            _rule(
                "Downside vol < 75% x total vol",
                ok=False,
                measured="N/A",
                target="< 75% total",
            )
        )
    else:
        d_vol = metrics.get(label_at, {}).get("Downside Vol", float("nan"))
        if np.isnan(d_vol):
            rules.append(
                _rule(
                    "Downside vol < 75% x total vol",
                    ok=False,
                    measured="N/A",
                    target="< 75% total",
                )
            )
        else:
            rules.append(
                _rule(
                    "Downside vol < 75% x total vol",
                    ok=d_vol < 0.75 * p_vol,
                    measured=(f"{d_vol:.1%} vs 75% x {p_vol:.1%} = {0.75 * p_vol:.1%}"),
                    target="< 75% total",
                )
            )
    # Rule 16 — Total cost ≤ 100 bps
    if cost_bps_actual is None or (
        isinstance(cost_bps_actual, float) and np.isnan(cost_bps_actual)
    ):
        rules.append(
            _rule(
                "Total cost ≤ 100 bps",
                ok=False,
                measured="N/A",
                target="≤ 100 bps",
            )
        )
    else:
        rules.append(
            _rule(
                "Total cost ≤ 100 bps",
                ok=cost_bps_actual <= 100.0,
                measured=f"{cost_bps_actual:.1f} bps",
                target="≤ 100 bps",
            )
        )
    # Rule 17 - OOS span >= 1.5 years (5-yr history, ~2-yr OOS, 3-yr train)
    if net_returns is None or net_returns.empty:
        rules.append(
            _rule(
                "OOS span ≥ 1.5 years",
                ok=False,
                measured="N/A",
                target="≥ 1.5 yrs",
            )
        )
    else:
        years = (net_returns.index[-1] - net_returns.index[0]).days / 365.25
        rules.append(
            _rule(
                "OOS span ≥ 1.5 years",
                ok=years >= 1.5,
                measured=f"{years:.2f} yrs",
                target="≥ 1.5 yrs",
            )
        )

    # Currency-hedge advisory (log-only, no rule entry)
    fx_w: dict[str, float] = {}
    for ticker, w in all_weights:
        ccy = currency_map.get(ticker)
        if ccy and ccy != "EUR":
            fx_w[ccy] = fx_w.get(ccy, 0.0) + w
    for ccy, w in fx_w.items():
        if w > 0.30:
            logger.warning(
                "Currency exposure %s = %.1f%% > 30%% — consider hedging.",
                ccy,
                w * 100.0,
            )

    return rules
