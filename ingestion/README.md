# Ingestion daemon

Headless data-ingestion service for the optimizer database. APScheduler runs in-process;
there is **no HTTP API**. It fetches market, fundamental, and macro data into PostgreSQL on
a schedule, and exposes Prometheus metrics.

It does not depend on the `optimizer` library — this side ingests, it does not optimize.

The database layer (models, repositories, connection manager, and the Alembic migration
tree) lives in the shared **`portopt-db`** package (`packages/portopt-db/`), a `uv`-workspace
sibling. The daemon imports it as `portopt_db` and keeps only services, the scheduler, the
CLI, schemas, and the `jobs` repository behavior. Migrations are owned by `portopt-db` and
run from there.

## Run

```bash
docker compose up -d          # db + adminer + scheduler
docker compose logs -f scheduler
```

Locally (from the workspace root):

```bash
uv sync --all-packages --all-extras
(cd packages/portopt-db && alembic upgrade head)   # migration owner
uv run --package portopt python -m app.worker      # blocks until SIGTERM/SIGINT
```

## Scheduled jobs

| Job | Default | What it does |
|-----|---------|--------------|
| `daily_pipeline` | `0 7 * * *` | ref-indices → yfinance → macro → news → summarize → calibrate |
| `midday_news` | `0 14 * * *` | news + summarize (afternoon refresh) |
| `universe_build` | `0 2 * * 0` | Trading 212 instrument universe |
| `weekly_refetch` | `0 3 * * 0` | full yfinance + macro rebuild (5y) |
| `weekly_market_wide` | `0 4 * * 0` | sector/industry structure, calendars, market summaries, full option chains |
| `fred_monthly` | `0 8 1 * *` | FRED economic series |
| `news_refresh` | every 30 min | incremental re-summarization |
| `orphan_reaper` | every 300s | fails (or reclaims) jobs whose heartbeat lease expired |

News, summarize, and calibrate form a dependency chain — each consumes what the previous one
wrote, so a failure upstream skips the rest rather than summarizing stale articles.
`universe_build` runs before `weekly_refetch` because every other step iterates the
`instruments` table it writes. `weekly_market_wide` runs after `weekly_refetch` so the
option-chain sweep sees the freshly rebuilt universe.

## Manual runs

Same job-slot, heartbeat, and progress path as the scheduler — so a manual run is refused
with `JobAlreadyRunningError` if the scheduler is already running that step, instead of
double-fetching.

```bash
docker compose exec scheduler python -m app.cli daily
docker compose exec scheduler python -m app.cli refetch-all
docker compose exec scheduler python -m app.cli yfinance --mode full --period 5y
docker compose exec scheduler python -m app.cli universe
docker compose exec scheduler python -m app.cli macro
docker compose exec scheduler python -m app.cli fred
docker compose exec scheduler python -m app.cli news
docker compose exec scheduler python -m app.cli summarize
docker compose exec scheduler python -m app.cli calibrate
docker compose exec scheduler python -m app.cli reference-indices
docker compose exec scheduler python -m app.cli market-structure
docker compose exec scheduler python -m app.cli calendars
docker compose exec scheduler python -m app.cli market-summary
docker compose exec scheduler python -m app.cli options
```

Single-step commands exit non-zero when the step did not complete, so shell drivers can gate
on them. `scheduler/fetch.sh` and `scheduler/refetch_all.sh` are thin wrappers over `daily`
and `refetch-all`.

## Observability

No polling endpoint. Job state lives in three places:

- **Logs** — `docker compose logs -f scheduler`
- **`background_jobs` table** — status, progress, errors, heartbeat, worker host/pid
- **Prometheus** — `http://localhost:9000/metrics`: `jobs_started_total`,
  `jobs_completed_total`, `jobs_failed_total`, `job_duration_seconds`, `jobs_in_progress`,
  all labeled by `domain`. Also the container healthcheck target.

Set `NOTIFICATION_WEBHOOK_URL` for a Discord/Slack POST on job failure.

## Operational notes

- **Run exactly one daemon per database.** APScheduler 3.x forbids sharing a job store
  between schedulers, so one process owns the cron triggers. This is the correct standard
  mitigation, not a defect — see *Scaling* below for the sanctioned multi-replica path.
- **Liveness is a heartbeat lease.** A running job renews `last_heartbeat_at` every
  `SCHEDULER_HEARTBEAT_CADENCE_SECONDS` (30s); the orphan reaper only touches a claim once the
  lease TTL (`SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS`, 300s = 10× cadence) elapses with no
  renewal. It is the renewal — not a flat timeout — that keeps a long synchronous step (the
  multi-minute yfinance fetch, the reference-index seed) from being falsely reaped. Host and
  PID are no longer part of the decision, so the daemon is portable (Windows/macOS dev) and
  two daemons never reap each other's live jobs.
- **Orphan strategy — `SCHEDULER_ORPHAN_STRATEGY` (`fail` | `reclaim`, default `fail`).**
  `fail` marks a dead-worker job failed; the next cron re-runs it. `reclaim` additionally
  re-dispatches the step immediately (at-least-once self-healing), capped by
  `SCHEDULER_ORPHAN_MAX_RECLAIM_ATTEMPTS` (default 3). **Only enable `reclaim` once every
  fetch write is an idempotent upsert** — re-running a non-idempotent job duplicates rows.
- **Clean shutdown.** On SIGTERM the daemon stops claiming new jobs and drains in-flight work
  for up to `SCHEDULER_SHUTDOWN_DRAIN_TIMEOUT_SECONDS` (30s), then force-exits. Size the
  container `stop_grace_period` / `terminationGracePeriodSeconds` above that (compose ships
  `60s`; the compose default is only 10s) so the drain is not SIGKILLed mid-write. Raise both
  together to drain long fetches cleanly.
- **`TRADING_212_API_KEY` absent** ⇒ `universe_build` skips without claiming a job slot. That
  is a configuration state, not a failure — but nothing will refresh `instruments`.
- **Migration `d1e2f3a4b5c6` is one-way.** It drops the 17 non-ingestion tables and its
  `downgrade()` raises. Restore from a dump taken before the upgrade.

## Scaling

One daemon per database is the design now. The workload is a few dozen long-running jobs a
day (a yfinance fetch over thousands of tickers, a reference-index seed); the cost is network
latency to the sources, which a bigger scheduler does not remove — so a distributed task
queue would add operational weight for no gain.

If fetch wall-clock time ever exceeds the daily window, the sanctioned path is **leader
election with a PostgreSQL advisory lock** (`pg_try_advisory_xact_lock`): replicas compete,
only the lock-holder fires the cron triggers, the rest stay hot standby or fetch-only
workers. It reuses the existing Postgres — no Redis, no Redlock. Not built; documented so the
next person does not re-derive it.

> **Deployment caveat:** PgBouncer *transaction* pooling breaks session-level advisory locks.
> Use the transaction-scoped variant of the lock.

APScheduler 4.x sanctions multi-scheduler job stores explicitly, but was pre-release at this
project's cutoff — **do not plan a scale-out on 4.x without re-validating this section.**

## Layout

```
app/
  worker.py        daemon entrypoint (metrics → init_db → reap orphans → bootstrap → scheduler)
  cli.py           manual runs (Typer)
  database.py      thin factory: builds DbConfig from settings → portopt_db.DatabaseManager
  services/
    jobs/          APScheduler wiring + BackgroundJobService
    market_data/   yfinance client, bulk fetch, reference-index seeding
    macro/         FRED / Il Sole / Trading Economics scrapers, LLM summary + calibration
    universe/      Trading 212 universe build
    infrastructure/ circuit breaker, rate limiter, retry, TTL cache
  repositories/
    jobs/          BackgroundJobRepository (behavior stays here); domain repos re-exported
                   from portopt_db.repositories
  schemas/         typed step arguments + progress payloads
baml_src/          LLM functions (SummarizeCountryNews, ClassifyMacroRegime)

# Models, domain repositories, engine, and Alembic live in the shared package:
../packages/portopt-db/src/portopt_db/   base, models/, repositories/, engine, config, coerce
../packages/portopt-db/alembic/          the single migration tree
```

## Tests

```bash
pytest                                              # in ingestion/
pytest --cov=app --cov-branch --cov-fail-under=80   # CI gate (line ≥80% and branch ≥0.80)
```

SQLite in-memory, SAVEPOINT-per-test. There is no `client` fixture — tests drive service,
repository, and scheduler functions directly.
