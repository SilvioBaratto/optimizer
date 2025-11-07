# CLAUDE.md - Orchestrator

Guidance for Claude Code across the entire portfolio optimizer repository.

## CRITICAL RULES - YOU MUST FOLLOW THESE

### Rule 1: Context Loading (Orchestration)

**BEFORE starting ANY task, read the relevant context files:**

| Task Type | Read Files |
|-----------|------------|
| Backend/optimizer task | `optimizer/.claude/CLAUDE.md` + `optimizer/README.md` |
| Frontend task | `frontend/.claude/CLAUDE.md` + `frontend/README.md` |
| Full-stack task | ALL four files above |

**Purpose:** Sub-CLAUDE.md files contain specific commands, patterns, and implementation details. This file orchestrates WHEN to read them.

### Rule 2: Documentation Maintenance

- **NEVER** create new .md files
- **ALWAYS** update `optimizer/README.md` or `frontend/README.md` after completing tasks
- Include: features added, config changes, implementation decisions, usage examples

### Rule 3: Theory Before Implementation

**Before coding optimizer features, read relevant theory from `portfolio_guideline/chapters/`:**

| Feature Type | Read |
|--------------|------|
| Portfolio construction | `01_portfolio_construction.md` |
| Macro regime | `02_macroeconomic_analysis.md` |
| Risk budgeting | `03_diversification_risk_budgeting.md` + `05_quantitative_implementation.md` |
| Portfolio optimizer | `04_portfolio_optimization.md` |

Contains: MSCI Barra standards, mathematical formulas, thresholds (ISM>52 = cyclical), institutional best practices.

### Rule 4: Database Schema Verification

**Before accessing database data:**

1. Read relevant model files from `optimizer/app/models/`
2. Verify data exists with bash queries
3. Check relationships and constraints

| Data Type | Model Files |
|-----------|-------------|
| Macro regime | `macro_regime.py` + `trading_economics.py` |
| Universe | `universe.py` |
| News | `news.py` |
| Signals | `stock_signals.py` + `signal_distribution.py` |

### Rule 5: VS Code Launch Configurations

**All optimizer scripts executed via `.vscode/launch.json`, NOT command line.**

- All Python configs: `"cwd": "${workspaceFolder}/optimizer"`
- All set: `PYTHONPATH=${workspaceFolder}/optimizer`
- Always use absolute imports from optimizer root

### Rule 6: Output File Location

**All output files (CSV, plots, TXT, MD, JSON) MUST be saved in `optimizer/outputs/` directory.**

### Rule 7: Cache File Location

**All cache files (pickled data, temporary files, intermediate results) MUST be saved in `optimizer/.cache/` directory.**

## Project Overview

**Quantitative portfolio optimizer system** combining:

- **Macro regime classification:** Business cycle positioning (EARLY_CYCLE, MID_CYCLE, LATE_CYCLE, RECESSION) using BAML LLM framework
- **Mathematical stock signals:** 7-pass cross-sectional standardization, zero LLM cost
- **Portfolio construction:** 20-stock concentrated portfolios with risk budgeting
- **Technology:** FastAPI backend + Supabase PostgreSQL + Angular 20 frontend

## System Architecture

```
portfolio-optimizer/
├── optimizer/              # Backend (FastAPI + SQLAlchemy 2.0 + BAML)
│   ├── app/               # API core (config, database, models)
│   ├── src/               # Business logic (macro, signals, risk)
│   ├── baml_src/          # LLM prompts (681-line institutional framework)
│   ├── alembic/           # Database migrations
│   ├── .cache/            # Cache files (Rule 7)
│   ├── outputs/           # Output files (Rule 6)
│   └── .claude/CLAUDE.md  # Backend-specific guidance
│
├── frontend/              # Frontend (Angular 20 + Tailwind + Supabase auth)
│   └── .claude/CLAUDE.md  # Frontend-specific guidance
│
├── portfolio_guideline/   # Theory documentation (Rule 3)
│   └── chapters/          # MSCI Barra standards, quantitative methods
│
├── .vscode/launch.json    # All scripts launched here (Rule 5)
└── CLAUDE.md              # THIS FILE - Orchestrator
```

## Key System Components

**Backend Modules (`optimizer/src/`):**

- **macro_regime:** Daily LLM-based regime classification (BAML, has cost)
- **stock_analyzer:** Daily mathematical signal generation (free, 100+ stocks parallel)
- **risk_management:** Portfolio construction (20 stocks, risk-based sizing)
- **universe:** Universe management (exchanges, instruments)
- **data_visualization:** Charts and analysis reports

**Frontend Features:**
- Supabase authentication
- Real-time portfolio dashboard
- Signal analysis visualization
- Regime classification display

## Workflow Guidance

**For backend development:**
1. Read `optimizer/.claude/CLAUDE.md` for commands, patterns, database config
2. Check `optimizer/README.md` for current features
3. Follow Rules 3-7 above

**For frontend development:**
1. Read `frontend/.claude/CLAUDE.md` for Angular patterns
2. Check `frontend/README.md` for current features
3. Coordinate with backend API endpoints

**For full-stack features:**
1. Read both sub-CLAUDE.md files
2. Read both README files
3. Ensure backend API + frontend UI alignment

**For database work:**
1. Follow Rule 4 (read models, verify data)
2. Check `optimizer/.claude/CLAUDE.md` for Alembic commands
3. Use `optimizer/.claude/CLAUDE.md` for session patterns

**For deployment:**
- Backend: Check `optimizer/.claude/CLAUDE.md` for Docker scheduler
- Frontend: Check `frontend/.claude/CLAUDE.md` for build/deploy

## Critical Cross-Cutting Constraints

- **Database:** NEVER use prepared statements (Supabase pooler incompatible)
- **LLM costs:** Only macro regime uses LLM; signals are 100% mathematical
- **Timezone:** All timestamps UTC
- **BAML:** Never edit `baml_client/` (auto-generated)
- **Execution:** All scripts via `.vscode/launch.json` (Rule 5)
- **Files:** Outputs in `outputs/`, cache in `.cache/` (Rules 6-7)

## Environment Files

`.env`, `.env.dev`, `.env.staging`, `.env.prod` at repository root

Required variables documented in `optimizer/.claude/CLAUDE.md`

## Quick Reference

| Need | Check |
|------|-------|
| Backend commands | `optimizer/.claude/CLAUDE.md` |
| Frontend commands | `frontend/.claude/CLAUDE.md` |
| Database patterns | `optimizer/.claude/CLAUDE.md` |
| Theory/formulas | `portfolio_guideline/chapters/*.md` |
| Current features | `optimizer/README.md` or `frontend/README.md` |
| Launch configs | `.vscode/launch.json` |
| Model schemas | `optimizer/app/models/*.py` |
