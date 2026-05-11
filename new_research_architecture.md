# Research Architecture v2 — Domain-Folder Restructure

## Current State

```
research/                        # Flat — all .py files in one directory
├── data_assembly.py             # 2218 lines  ← monolithic
├── stock_selection_pipeline.py  # 1948 lines  ← monolithic
├── portfolio.py                 # 478 lines
├── _factors.py                  # 421 lines
├── _optimization.py             # 376 lines
├── _backtest_plots.py           # 372 lines
├── _preflight.py                # 313 lines
├── _currency.py                 # 234 lines
├── _display.py                  # 137 lines
├── _report.py                   # 129 lines
├── _preprocessing.py            # 123 lines
├── _cli.py                      # 109 lines
├── _persistence.py              # 107 lines
├── _returns.py                  # 96 lines
├── __init__.py                  # 14 lines
├── py.typed
├── RUNBOOK.md
├── portfolio_checklist.md
├── templates/
│   └── report.md.j2
└── output/
```

**Problems:**
- `data_assembly.py` (2218 lines): 20 functions in one file — DB queries, pivoting, dedup, currency normalization all mixed
- `stock_selection_pipeline.py` (1948 lines): 25 functions — orchestration, metrics, checklist, display, persistence all in one module
- Flat directory: no domain grouping, hard to locate specific functionality
- Implicit layering via underscore convention only (no architectural enforcement)

**Strengths to preserve:**
- Zero circular imports
- Clean hub-and-spoke dependency graph (orchestrator → leaf modules)
- Most leaf modules <500 lines, single responsibility
- Private underscore convention already signals internal vs public

---

## Target Architecture

```
research/
├── __init__.py
├── py.typed
├── RUNBOOK.md
├── portfolio_checklist.md
│
├── cli/                          # Entry points — thin, no business logic
│   ├── __init__.py
│   ├── _parser.py                # argparse builder (ex _cli.py)
│   └── _main.py                  # main() orchestrator + __main__ guard
│
├── data/                         # Data assembly — DB → DataFrames
│   ├── __init__.py
│   ├── _container.py             # DataAssembly frozen container
│   ├── _equity.py                # Prices, volumes, fundamentals, financials, delisting
│   ├── _macro.py                 # Macro, FRED, Treasury yields, bond observations
│   ├── _sentiment.py             # News sentiment assembly
│   ├── _history.py               # Fundamental history + delisting returns
│   ├── _regime.py                # Regime data assembly (multi-source merge)
│   ├── _currency.py              # Minor-unit currency normalization
│   ├── _helpers.py               # Shared pivot/dedup/float helpers
│   └── _orchestrator.py          # assemble_all() — calls all assemblers
│
├── pipeline/                     # Pipeline steps — one module per step
│   ├── __init__.py
│   ├── _load.py                  # Step 1: load_data + _materialise_clean_returns
│   ├── _screen.py                # Step 2: screen_investable
│   ├── _factors.py               # Steps 3-5: build_history + validate_is + validate_oos
│   ├── _regime.py                # Step 6: classify_and_tilt
│   ├── _optimize.py              # Step 7: optimize_portfolio
│   ├── _checklist.py             # 17-rule checklist + terminal gate + output writers
│   └── _metrics.py               # Metric computation + benchmark fetching
│
├── optimization/                 # Optimizer primitives — config, retighten, rebalance
│   ├── __init__.py
│   ├── _config.py                # _make_opt_config, _select_optimizer, _make_builder
│   ├── _retighten.py             # _solve_with_retighten, _top4_weight
│   └── _rebalance.py             # _decide_rebalance, _hockey_stick_warn
│
├── factors/                      # Factor research — history building, validation
│   ├── __init__.py
│   ├── _history.py               # build_factor_scores_history (PIT rolling)
│   └── _validator.py             # validate_factors, _slice_fundamentals_at, coverage
│
├── reporting/                    # Reports + visualization — rendering only
│   ├── __init__.py
│   ├── _report.py                # Jinja2 report.md render + binding constraints
│   ├── _plots.py                 # Matplotlib chart generation (6 chart types)
│   └── _display.py               # Rich console panels/tables/progress
│
├── returns/                      # Return computation — preprocessing + tax
│   ├── __init__.py
│   ├── _preprocessing.py         # apply_fx_to_prices, build_return_preprocessing_pipeline
│   └── _tax.py                   # compute_after_tax_returns
│
├── preflight/                    # DB health checks — pre-run validation
│   ├── __init__.py
│   └── _checks.py                # run_db_preflight + 6 check functions
│
├── persistence/                  # DB write-back — research run snapshots
│   ├── __init__.py
│   └── _snapshot.py              # persist_research_run, _diff_from_default, _flatten_metrics
│
├── strategies/                   # Alternative strategies — from portfolio.py
│   ├── __init__.py
│   ├── _runner.py                # Strategy enum + optimize()
│   └── _inspect.py               # data_summary(), strategies()
│
├── templates/                    # Jinja2 templates (unchanged)
│   └── report.md.j2
│
└── output/                       # Generated artifacts (unchanged)
    ├── cumulative_returns.png
    ├── drawdowns.png
    ├── rolling_sharpe.png
    ├── sector_allocation.png
    ├── country_allocation.png
    ├── factor_ic.png
    ├── report.md
    ├── metrics.json
    ├── checklist.json
    ├── weights.csv
    └── last_review_date.txt
```

### Line-count budget (every module ≤600 lines)

| Module | Est. Lines | Source |
|--------|-----------|--------|
| `cli/_parser.py` | ~110 | `_cli.py` (unchanged) |
| `cli/_main.py` | ~200 | Orchestrator extracted from `stock_selection_pipeline.py` |
| `data/_container.py` | ~80 | `DataAssembly` class from `data_assembly.py` |
| `data/_equity.py` | ~450 | `assemble_prices`, `assemble_volumes`, `assemble_fundamentals`, `assemble_financial_statements`, `assemble_delisting_returns`, `_apply_delisting_returns` |
| `data/_macro.py` | ~400 | `assemble_macro_data`, `assemble_fred_series`, `assemble_macro_timeseries`, `assemble_te_observations`, `assemble_bond_observations` |
| `data/_sentiment.py` | ~120 | `assemble_sentiment` |
| `data/_history.py` | ~150 | `assemble_fundamental_history`, `assemble_fx_rates` |
| `data/_regime.py` | ~200 | `assemble_regime_data` + merge logic |
| `data/_currency.py` | ~234 | `_currency.py` (unchanged) |
| `data/_helpers.py` | ~200 | `_to_float`, `_build_ticker_map`, `_build_ticker_rank_map`, `_pivot_with_dedup`, `_dedup_fundamentals_df`, `REGION_MAP` |
| `data/_orchestrator.py` | ~300 | `assemble_all`, `_compute_assembly_hash` |
| `pipeline/_load.py` | ~250 | `load_data`, `_materialise_clean_returns`, `_assert_assembly_size`, `_assert_universe_size` |
| `pipeline/_screen.py` | ~150 | `screen_investable`, `_validate_n_selected`, `_missing_gics_sectors` |
| `pipeline/_factors.py` | ~350 | `build_history`, `validate_is`, `validate_oos`, `_check_factor_coverage` |
| `pipeline/_regime.py` | ~200 | `classify_and_tilt`, `_cache_regime_classification` |
| `pipeline/_optimize.py` | ~300 | `optimize_portfolio`, `_print_diversification`, `_read_last_review_date`, `_write_last_review_date` |
| `pipeline/_checklist.py` | ~400 | `_validate_checklist`, `_rule`, `_apply_terminal_gate`, `write_metrics_json`, `write_checklist_json`, `write_weights_csv`, render helpers |
| `pipeline/_metrics.py` | ~250 | `_annualized_return`, `_sharpe`, `_sortino`, `_downside_vol`, `_information_ratio`, `compute_weighted_cost_bps`, `_fetch_benchmark_returns`, `_build_country_map`, `_daily_rf` |
| `optimization/_config.py` | ~200 | `_make_opt_config`, `_select_optimizer`, `_make_builder`, `build_research_optimizer`, `_annualized_sharpe` |
| `optimization/_retighten.py` | ~150 | `_solve_with_retighten`, `_top4_weight` |
| `optimization/_rebalance.py` | ~100 | `_decide_rebalance`, `_hockey_stick_warn`, `_REGION_MAP`, `_TOP_N` |
| `factors/_history.py` | ~280 | `build_factor_scores_history` (unchanged core) |
| `factors/_validator.py` | ~180 | `validate_factors`, `_slice_fundamentals_at` |
| `reporting/_report.py` | ~200 | `render_report`, `compute_binding_constraints`, `_build_environment`, `_build_factor_ic_rows` |
| `reporting/_plots.py` | ~400 | All 6 `plot_*` functions + `generate_backtest_plots` (unchanged) |
| `reporting/_display.py` | ~137 | `_display.py` (unchanged) |
| `returns/_preprocessing.py` | ~123 | `_preprocessing.py` (unchanged) |
| `returns/_tax.py` | ~96 | `_returns.py` (unchanged) |
| `preflight/_checks.py` | ~313 | `_preflight.py` (unchanged) |
| `persistence/_snapshot.py` | ~107 | `_persistence.py` (unchanged) |
| `strategies/_runner.py` | ~300 | `Strategy`, `optimize`, `_build_optimizer`, `_display_weights`, `_display_backtest` |
| `strategies/_inspect.py` | ~200 | `data_summary`, `strategies`, `_get_db_manager`, `_display_summary` |

**Total: 34 modules across 12 domain folders. Every module <500 lines. 10 modules unchanged (just moved).**

---

## SOLID Alignment

### S — Single Responsibility

Each folder = one reason to change:

| Folder | Single Responsibility | Changes when... |
|--------|----------------------|-----------------|
| `cli/` | Parse args, wire pipeline | CLI surface needs new flags |
| `data/` | DB → DataFrame transformation | DB schema changes, new data sources |
| `pipeline/` | End-to-end step orchestration | Pipeline flow reordered, new steps |
| `optimization/` | Optimizer config + solve + rebalance decisions | Optimization methodology changes |
| `factors/` | Factor score history + validation | Factor definitions or methodology change |
| `reporting/` | Render charts + reports + console display | Output format or visualization changes |
| `returns/` | Return preprocessing + after-tax computation | Tax rate or preprocessing methodology changes |
| `preflight/` | Pre-run DB health validation | New health checks added |
| `persistence/` | Write research runs to API DB | Snapshot schema changes |
| `strategies/` | Alternative named-strategy path | New strategies added |

### O — Open/Closed

Pipeline steps are **open for extension** (add new step module in `pipeline/`) but **closed for modification** (existing steps untouched). New data assemblers added as new modules in `data/` — `_orchestrator.py` imports them, no existing modules edited.

### L — Liskov Substitution

`DataAssembly` is a frozen container — any code receiving it can trust the contract. Pipeline step functions all accept `DataAssembly` + `args` and return consistent result types (`pd.DataFrame`, `dict`, `tuple[Portfolio, ...]`).

### I — Interface Segregation

Pipeline steps receive only what they need (not the full `args` namespace):
- `_load(args, db_manager)` → `DataAssembly`
- `_screen(assembly, args)` → `pd.Index` of tickers
- `_factors(assembly, tickers, dates, args)` → factor history dict
- `_optimize(assembly, history, regime, args)` → weights + metrics

No step depends on methods it doesn't call.

### D — Dependency Inversion

High-level pipeline (`pipeline/`) depends on abstractions from `optimizer/` library, not on concrete DB queries. Low-level `data/` modules depend on SQLAlchemy models. Dependency direction:

```
cli/ ──→ pipeline/ ──→ optimization/ ──→ optimizer/ (library)
                  ──→ factors/ ────────→ optimizer/
                  ──→ returns/ ────────→ optimizer/
                  ──→ reporting/
     ──→ data/ ────────────────────────→ api/ (DB models)
                  ──→ preflight/
                  ──→ persistence/
```

`pipeline/` never imports from `api/`. `data/` never imports from `pipeline/` or `optimization/`. Clean layering enforced by folder boundaries.

---

## Import Rules

1. **`data/` modules** — import only from `api/` (DB models) and `optimizer/` (FX rates). Never import from `pipeline/`, `optimization/`, `factors/`, `reporting/`.
2. **`pipeline/` modules** — import from `data/` (container type only), `optimization/`, `factors/`, `returns/`, `reporting/`, and `optimizer/` library. Never import from `api/`.
3. **`cli/` modules** — import from `pipeline/` (main orchestrator) and `data/` (for standalone data inspection). Thinnest layer — no business logic.
4. **`reporting/` modules** — zero imports from `api/` or `data/`. Pure rendering, receives already-computed data structures.
5. **Leaf technical folders** (`returns/`, `preflight/`, `persistence/`) — import from `api/` OR `optimizer/`, never both. `returns/` → `optimizer/` only. `preflight/` + `persistence/` → `api/` only.

---

## Dependency Graph

```
cli/_main.py
  ├── cli/_parser.py
  ├── pipeline/_load.py
  ├── pipeline/_screen.py
  ├── pipeline/_factors.py
  ├── pipeline/_regime.py
  ├── pipeline/_optimize.py
  ├── pipeline/_checklist.py
  └── pipeline/_metrics.py

pipeline/_load.py
  ├── data/_orchestrator.py       (assemble_all)
  ├── data/_container.py          (DataAssembly type)
  ├── preflight/_checks.py        (run_db_preflight)
  └── returns/_preprocessing.py   (apply_fx_to_prices, build_return_preprocessing_pipeline)

pipeline/_screen.py
  ├── data/_container.py
  └── optimizer.universe          (InvestabilityScreenConfig)

pipeline/_factors.py
  ├── factors/_history.py         (build_factor_scores_history)
  └── factors/_validator.py       (validate_factors)

pipeline/_regime.py
  ├── data/_regime.py             (assemble_regime_data)
  └── optimizer.factors           (classify_regime, apply_regime_tilts)

pipeline/_optimize.py
  ├── optimization/_config.py     (_make_opt_config, _make_builder)
  ├── optimization/_retighten.py  (_solve_with_retighten)
  ├── optimization/_rebalance.py  (_decide_rebalance)
  └── optimizer.pipeline          (run_full_pipeline_with_selection)

pipeline/_checklist.py
  └── (self-contained — reads portfolio_checklist.md rules)

pipeline/_metrics.py
  ├── returns/_tax.py             (compute_after_tax_returns)
  ├── reporting/_plots.py         (generate_backtest_plots)
  └── reporting/_report.py        (render_report, compute_binding_constraints)

data/_orchestrator.py
  ├── data/_container.py
  ├── data/_equity.py
  ├── data/_macro.py
  ├── data/_sentiment.py
  ├── data/_history.py
  ├── data/_regime.py
  ├── data/_currency.py
  └── data/_helpers.py

strategies/_runner.py
  ├── data/_orchestrator.py
  ├── reporting/_display.py
  ├── optimizer.optimization
  └── optimizer.pipeline
```

**Zero circular dependencies.** All arrows point from orchestration toward data or toward the optimizer library.

---

## Migration Plan

### Phase 1: Split monoliths (no folder creation yet)

1. Split `data_assembly.py` into 10 modules under `research/data/`. Extract `DataAssembly` container, one module per domain (equity, macro, sentiment, history, regime, currency, helpers), plus `_orchestrator.py` calling all assemblers.
2. Split `stock_selection_pipeline.py` into 8 pipeline step modules. Extract `main()` as thin orchestrator calling each step. Extract metrics, checklist, and benchmark logic.

### Phase 2: Group leaf modules into folders

3. Move `_optimization.py` → `optimization/` split into `_config.py`, `_retighten.py`, `_rebalance.py`.
4. Move `_factors.py` → `factors/` split into `_history.py`, `_validator.py`.
5. Move `_report.py` + `_backtest_plots.py` + `_display.py` → `reporting/`.
6. Move `_returns.py` + `_preprocessing.py` → `returns/`.
7. Move `_cli.py` → `cli/_parser.py`.
8. Move `_preflight.py` → `preflight/_checks.py`.
9. Move `_persistence.py` → `persistence/_snapshot.py`.
10. Move `portfolio.py` → `strategies/` split into `_runner.py`, `_inspect.py`.

### Phase 3: Update imports and __init__ files

11. Add `__init__.py` to each folder with clean re-exports.
12. Update cross-module imports to use new paths.
13. Verify `python -m research.cli._main` runs end-to-end.
14. Run full pipeline to validate output parity.

### Phase 4: Cleanup

15. Remove old flat files.
16. Update `research/__init__.py` public API.
17. Update any external importers (scheduler scripts, docs).

---

## Public API (`research/__init__.py`)

```python
# Data
from research.data import DataAssembly, assemble_all

# Pipeline
from research.pipeline import run_pipeline

# Strategies
from research.strategies import Strategy, optimize, data_summary, strategies

# Factors (for external consumers)
from research.factors import build_factor_scores_history, validate_factors

# Preflight (standalone use)
from research.preflight import run_db_preflight
```

---

## Verification Checklist

- [ ] All 34 modules exist with `__init__.py` in each folder
- [ ] Every module ≤600 lines (`wc -l **/*.py` confirms)
- [ ] Zero circular imports (`python -c "import research"` succeeds)
- [ ] `python -m research.cli._main` runs full pipeline
- [ ] `python -m research.cli._main --persist` writes to DB
- [ ] `research.strategies.strategies()` lists all 10 strategies
- [ ] `research.data.assemble_all(db_manager)` returns valid `DataAssembly`
- [ ] `research.preflight.run_db_preflight(db_manager)` runs all 6 checks
- [ ] Output files identical to pre-migration run (bit-exact for charts)
- [ ] `ruff check research/` passes
- [ ] `mypy research/` passes (strict mode)
- [ ] Existing scheduler scripts still work (no import path changes in `api/` or `optimizer/`)
