# CLAUDE.md - Backend/Optimizer

Portfolio optimizer backend guidance.

## Tech Stack

- FastAPI (sync) + SQLAlchemy 2.0 + psycopg2 + Supabase PostgreSQL
- BAML (baml-py) for type-safe LLM calls
- Riskfolio-Lib, yfinance (with circuit breaker), numpy, pandas, scikit-learn
- Docker + cron scheduler (22:00 Italy time daily)

## Project Structure

```
optimizer/
├── app/                    # FastAPI core (config, database, models, dependencies)
├── src/                    # Business logic (macro_regime, stock_analyzer, risk_management)
├── baml_src/               # LLM prompts (DO NOT EDIT baml_client/)
├── alembic/                # DB migrations
├── .cache/                 # Temporary/pickle files
├── outputs/                # User-facing outputs (CSV, plots, JSON, MD)
└── docker/                 # Scheduler deployment
```

## Critical Execution Rules

**All scripts launched via VS Code `.vscode/launch.json`:**

- `cwd`: Always `${workspaceFolder}/optimizer`
- `PYTHONPATH`: Always `${workspaceFolder}/optimizer`
- `envFile`: Always `${workspaceFolder}/.env`

**File locations (MANDATORY):**

```python
# Cache: .cache/
cache = Path(".cache/pass1_raw_data_2024-10-27.pkl")

# Outputs: outputs/
output = Path("outputs/portfolio_2024-10-27.json")

# Imports: absolute from optimizer root
from app.database import database_manager, init_db
from app.models import StockSignal, Instrument
```

## Commands

**Analysis:**

```bash
python src/macro_regime/run_regime_analysis.py        # LLM cost
python src/stock_analyzer/run_signal_analysis.py      # free
python src/stock_analyzer/run_signal_analysis.py --skip-pass1  # use cache
```

**Database:**

```bash
alembic revision --autogenerate -m "msg"
alembic upgrade head
alembic downgrade -1
```

**Testing:**

```bash
pytest -v
pytest src/stock_analyzer/
pytest -k "test_name"
```

**Docker:**

```bash
cd docker && docker-compose up -d
docker-compose logs -f
docker exec portfolio-optimizer-scheduler /app/run_daily_analysis.sh
```

## Database

**CRITICAL:**

- Port 6543 (Transaction Pooler mode)
- NEVER use prepared statements (`database_statement_cache_size=0`)
- NullPool dev, QueuePool prod (pool_size=15, max_overflow=3)
- Pool recycle: 300s, pre_ping: True
- All timestamps UTC

**Session patterns:**

```python
# Standalone scripts
init_db()
with database_manager.get_session() as session:
    results = session.query(Model).all()

# FastAPI routes
@router.get("/")
def endpoint(db: Session = Depends(get_db)):
    return db.query(Model).all()
```

**Adding models:**

1. Create `app/models/new_model.py`
2. Import in `app/models/__init__.py`
3. `alembic revision --autogenerate -m "msg"`
4. Review, then `alembic upgrade head`

**Conventions:**

- Use `BaseModel` from `app/models/base` (UUID id, timestamps)
- Add indexes for foreign keys/query filters
- Use PostgreSQL types: `UUID`, `JSONB`, `ARRAY`
- Add `comment=` to all columns

## Code Organization

**CRITICAL: Files MUST NOT exceed 500 lines**

**Before implementing, if file would exceed 500 lines:**

1. Break into package structure with logical modules
2. Target 200-300 lines per module
3. Create `__init__.py` for packages

**Module structure:**

```
feature/
├── __init__.py
├── main.py           # Entry point (orchestration only)
├── config.py         # Constants/configuration
├── data_processing.py
├── models.py
├── utils.py
```

**Split by:**

- Single responsibility (one module = one concern)
- Related classes → separate module per class/group
- Different concerns → separate modules (database, API, business logic)
- Reusable code → utility modules

**Goal:** Maintainability and readability, not arbitrary line counting

### Code Reuse (DRY Principle)

**CRITICAL: NEVER repeat code - always check existing implementations first**

**Before implementing new functionality:**
1. Search codebase for similar implementations
2. Grep for relevant keywords (e.g., `grep -r "yfinance" src/`)
3. Reuse existing modules/functions

**Common reusable modules:**
- `src/stock_analyzer/yfinance_fetcher.py` - yfinance data fetching
- `src/macro_regime/news_fetcher.py` - News fetching
- `src/stock_analyzer/utils.py` - Utility functions
- `app/database.py` - Database session management

**If found:** Import and use existing code
**If not found:** Check if it should be extracted to shared module

### Type Checking

**CRITICAL: Always check and fix typing errors after implementation**

VS Code configuration: `python.analysis.typeCheckingMode": "standard"` (`.vscode/settings.json`)

**After implementing/modifying files:**

1. Check IDE for type errors (red squiggles, Problems panel)
2. Fix all typing issues before committing
3. Add type hints to function signatures and return types

**Common patterns:**

```python
from typing import List, Dict, Optional, Tuple

def process_signals(
    signals: List[StockSignal],
    date: date_type
) -> Dict[str, float]:
    ...
```

## Code Patterns

### 7-Pass Cross-Sectional Standardization

`stock_analyzer/pipeline/cross_sectional.py`:

1. Pass 1: Fetch raw fundamentals (yfinance)
2. Pass 1.5: Robust cross-sectional stats (iterative outlier removal)
3. Pass 1B: Recalculate z-scores with robust stats
4. Pass 2: Winsorization (2.5/97.5 percentiles) + scaling
5. Pass 2.5: Robust factor statistics
6. Pass 2.6: Factor correlation analysis
7. Pass 3: Classify + save signals (momentum filters)

Key: Winsorize 2.5/97.5, remove z-scores beyond 3σ, denormalize ticker/sector

### Macro Regime Classification

`baml_src/cycle_classifier.baml` - 681-line institutional framework

**Data weights:** Country 50%, market 30%, global 15%, news 5%

**Regimes:** EARLY_CYCLE (ISM>52, steep curve), MID_CYCLE (sustained growth), LATE_CYCLE (ISM declining, flat curve), RECESSION (ISM<45, inverted curve), UNCERTAIN

**Output:** Regime + confidence, sector tilts (-0.10 to +0.10), factor exposure, recession risk (6M/12M)

**Requires macroeconomic news** - errors if missing

### BAML Integration

**Files:** `baml_src/*.baml` (cycle_classifier, clients, generators)

**Usage:**

```python
from baml_client import b
result = b.ClassifyBusinessCycleWithLLM(
    today="2024-10-27", economic_indicators=econ,
    market_indicators=market, news_signals=news, country="USA"
)
```

**Regenerate:** `baml generate` (or auto-saves)

**DO NOT EDIT:** `baml_client/` is auto-generated

## Performance

- 100+ stocks parallel (`SIGNAL_CONCURRENT_BATCHES`, `SIGNAL_STOCKS_PER_BATCH`)
- True async via thread pool executor (3-4x faster than sequential)
- Global circuit breaker for Yahoo Finance rate limiting
  - Exponential backoff: 2min → 4min → 8min → 16min...
  - Zero data loss (all stocks eventually processed)
- Conservative DB pooling (5 connections, 3 overflow)
- Cache Pass 1: `.cache/pass1_raw_data_YYYY-MM-DD.pkl`
- Batch commits: 50 signals (`SIGNAL_BATCH_SIZE`)
- Smart resume: Auto-detects incomplete runs

## Critical Constraints

- **NO prepared statements** (Supabase pooler incompatible)
- **LLM costs:** Only macro regime (stock signals 100% mathematical)
- **Data sources:** Il Sole/Trading Economics (rate limits), FRED (API), yfinance (free)
- **Timezone:** UTC only
- **BAML client:** Never edit generated code
- **NEVER use fallback data** - Prefer errors over fallback data; remove fallback data if found in existing code

## Environment Variables

Required in `.env`, `.env.dev`, `.env.staging`, `.env.prod`:

```bash
SUPABASE_DB_URL=postgresql+psycopg2://user:pass@host:6543/postgres
SUPABASE_URL=https://project.supabase.co
SUPABASE_KEY=anon-key
OPENAI_API_KEY=sk-...
FRED_API_KEY=...
ENVIRONMENT=development|staging|production
```

## Common Tasks

**Query signals:**

```python
with database_manager.get_session() as session:
    signals = session.query(StockSignal).filter(
        StockSignal.signal_date == date.today(),
        StockSignal.signal_type == "LARGE_GAIN"
    ).all()
```

**Add country:** Update `PORTFOLIO_COUNTRIES` in `src/macro_regime/ilsole_scraper.py`

**Add indicator:** Implement in `src/stock_analyzer/technical/indicators.py`, integrate in `mathematical_signal_calculator.py`

**Debug:** Use `--skip-pass1` to load from cache

## Docker Scheduler

Daily 22:00 Italy time (Europe/Rome timezone) execution:

1. Macro regime analysis → generates classifications
2. Stock signal analysis → uses latest regime

Config: `docker/crontab` (`0 22 * * *`), script: `docker/run_daily_analysis.sh`, logs: `docker/logs/`

Timezone: `TZ=Europe/Rome` (docker-compose.yml) - automatically handles DST

Resources: 2 cores max (0.5 reserved), 4GB memory (1GB reserved)

## Launch Configurations

| Config                          | Purpose            | Args                      |
| ------------------------------- | ------------------ | ------------------------- |
| API                             | FastAPI dev        | Port 8000                 |
| Signal Analyzer - Run All       | Full pipeline      | None                      |
| Signal Analyzer - Skip Pass 1   | Debug mode         | `--skip-pass1`            |
| Concentrated Portfolio Builder  | 20-stock portfolio | None                      |
| Walk-Forward Backtest           | Auto-detect dates  | None                      |
| Walk-Forward Backtest (October) | Single month       | `--start-date 2025-10-01` |
| Portfolio Growth Charts         | Backtest viz       | None                      |
| Large Gain Charts               | Signal viz         | None                      |

## Key Files

| Purpose         | File                                        |
| --------------- | ------------------------------------------- |
| Config          | `app/config.py`                             |
| DB manager      | `app/database.py`                           |
| Models          | `app/models/*.py`                           |
| Macro entry     | `src/macro_regime/run_regime_analysis.py`   |
| Signal entry    | `src/stock_analyzer/run_signal_analysis.py` |
| Signal pipeline | `src/stock_analyzer/pipeline/analyzer.py`   |
| BAML prompt     | `baml_src/cycle_classifier.baml`            |
| Migrations      | `alembic/versions/*.py`                     |
