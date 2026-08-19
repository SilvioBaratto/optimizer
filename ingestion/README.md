# Ingestion daemon

Headless data-ingestion service for the optimizer database. APScheduler runs in-process;
there is **no HTTP API**. It fetches market, fundamental, and macro data into PostgreSQL on
a schedule, and exposes Prometheus metrics.

It does not depend on the `optimizer` library — this side ingests, it does not optimize.

## Run

```bash
docker compose up -d          # db + adminer + scheduler
docker compose logs -f scheduler
```

Locally:

```bash
pip install -r requirements.txt
alembic upgrade head
python -m app.worker          # blocks until SIGTERM/SIGINT
```

## Scheduled jobs

| Job | Default | What it does |
|-----|---------|--------------|
| `daily_pipeline` | `0 7 * * *` | ref-indices → yfinance → macro → news → summarize → calibrate |
| `midday_news` | `0 14 * * *` | news + summarize (afternoon refresh) |
| `universe_build` | `0 2 * * 0` | Trading 212 instrument universe |
| `weekly_refetch` | `0 3 * * 0` | full yfinance + macro rebuild (5y) |
| `fred_monthly` | `0 8 1 * *` | FRED economic series |
| `news_refresh` | every 30 min | incremental re-summarization |
| `orphan_reaper` | every 300s | fails jobs whose worker died |

News, summarize, and calibrate form a dependency chain — each consumes what the previous one
wrote, so a failure upstream skips the rest rather than summarizing stale articles.
`universe_build` runs before `weekly_refetch` because every other step iterates the
`instruments` table it writes.

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

- **Run exactly one daemon per database.** The orphan reaper fails any active job whose
  `worker_host` differs from its own hostname, so two containers on one DB will reap each
  other's jobs.
- **`TRADING_212_API_KEY` absent** ⇒ `universe_build` skips without claiming a job slot. That
  is a configuration state, not a failure — but nothing will refresh `instruments`.
- **Migration `d1e2f3a4b5c6` is one-way.** It drops the 17 non-ingestion tables and its
  `downgrade()` raises. Restore from a dump taken before the upgrade.

## Layout

```
app/
  worker.py        daemon entrypoint (metrics → init_db → reap orphans → bootstrap → scheduler)
  cli.py           manual runs (Typer)
  services/
    jobs/          APScheduler wiring + BackgroundJobService
    market_data/   yfinance client, bulk fetch, reference-index seeding
    macro/         FRED / Il Sole / Trading Economics scrapers, LLM summary + calibration
    universe/      Trading 212 universe build
    infrastructure/ circuit breaker, rate limiter, retry, TTL cache
  repositories/    typed DB access
  models/          SQLAlchemy tables
  schemas/         typed step arguments + progress payloads
baml_src/          LLM functions (SummarizeCountryNews, ClassifyMacroRegime)
```

## Tests

```bash
pytest                                              # in ingestion/
pytest --cov=app --cov-branch --cov-fail-under=80   # CI gate (line ≥80% and branch ≥0.80)
```

SQLite in-memory, SAVEPOINT-per-test. There is no `client` fixture — tests drive service,
repository, and scheduler functions directly.
