# Non-functional requirements

Scalability, performance and maintainability, with what was measured and the target each
number is held against (standard 2.13). A target marked *proposed* has not been agreed
and is the maintainer's to confirm or change. A target marked *enforced* fails the build.

Measured on 2026-11-13 on the development Mac (arm64) against its local PostgreSQL 16
database, which holds job history from 2026-10-23 only. The numbers are a baseline, not a
load test; no load test has been run.

## Scalability

| Aspect | Measured on 2026-11-13 | Target | Status |
|---|---|---|---|
| Universe size | 29,241 rows in `instruments` | The daily sweep covers the active universe | proposed |
| Ingestion processes | One daemon per database; one active row per job type (`uq_background_jobs_active_type`) | One daemon per database (ADR-0004) | enforced by the unique index |
| Sweep concurrency | `YFINANCE_FETCH_WORKERS`, 1 to 16, default 4 | Raise it only while Yahoo does not throttle | proposed |
| Fund worker | One per database; its job slot mirrors the ingestion lease | One per database | by design |

Scaling out is documented, not built: `docs/ARCHITECTURE.md` §5.2 ("percorso di scala")
records the path and `ingestion/README.md` repeats it. There is no horizontal scaling
today, and `CLAUDE.md` requires exactly one daemon per database.

## Performance

Completed job runs recorded in `background_jobs`: 115 completed runs of the last 60 days,
out of 125 jobs of all statuses since 2026-10-23. Duration from `started_at` to `finished_at`, rounded to a tenth of a minute:

| Job type | Runs | p50 | p95 | Longest |
|---|---|---|---|---|
| `fund_drift_monitor` | 105 | under 3 s | under 10 s | under 10 s |
| `fx_fetch` | 1 | 0.7 min | 0.7 min | 0.7 min |
| `macro_news_fetch` | 1 | 1.2 min | 1.2 min | 1.2 min |
| `news_retention` | 6 | under 3 s | under 3 s | under 3 s |
| `fund_nav_mark` | 2 | under 3 s | under 3 s | under 3 s |

The jobs that matter most have no completed run in this history:

- `yfinance_fetch`, the daily price and fundamentals sweep: the run of 2026-11-12 was
  reaped as orphaned after 522 minutes, and the run started 2026-11-13 07:00 had been
  running for 241 minutes when measured.
- `fred_fetch`: six runs on 2026-11-13 failed within about a minute, every series ending in
  "Fetch failed after retries". The cause was not logged; `retry_with_backoff` now logs it,
  once the daemon image is rebuilt.

Optimizer solve time, one default `skfolio.MeanRisk` fit on synthetic normal returns of
756 days:

| Assets | Fit time |
|---|---|
| 20 | 0.05 s |
| 50 | 0.01 s |
| 100 | 0.01 s |
| 250 | 0.06 s |

Fund decision rounds: two completed rounds are recorded in `agent_runs` (2026-10-30 and
2026-11-04), each between `started_at` and `finished_at` taking a minute or less. Two
samples give no percentile. Rounds are bounded by the provider deadline; see the
integrations document.

| Target | Value | Status |
|---|---|---|
| Daily `yfinance_fetch` completes | Inside 24 hours, so it finishes before the next 07:00 start | proposed; not met in the measured history |
| `fred_fetch` completes | With every series fetched | proposed; failing on 2026-11-13 |
| Optimizer solve | Not the bottleneck at universe sizes the fund uses | proposed; met at 250 assets |

To reproduce the table, run this read-only against the database:

```sql
SELECT job_type, count(*),
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY extract(epoch FROM finished_at - started_at)) AS p50_s,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY extract(epoch FROM finished_at - started_at)) AS p95_s
FROM background_jobs
WHERE status = 'completed' AND started_at >= now() - interval '60 days'
GROUP BY job_type;
```

## Maintainability

| Aspect | Measured on 2026-11-13 | Target | Status |
|---|---|---|---|
| Coverage | Four suites with branch coverage, floors in CI | Library 90, `portopt-db` 90, ingestion 80, fund 80 (standard goal 75) | enforced |
| Tests | Library 3,765, ingestion 3,736, `portopt-db` 1,381 (+8 Postgres integration), fund 3,635 passing | All pass on every CI run | enforced |
| Static analysis | ruff (check and format), mypy strict, pyright with 0 errors, shellcheck, `pip-audit --strict` | Clean | enforced |
| Function size | 264 of 2,958 functions are over 30 lines, each listed in `scripts/ci/baselines/long_functions.json` | At most 30 lines (standard 2.8) | enforced by `scripts/ci/check_complexity_ratchet.py`: no new offender, no listed one grows |
| Cyclomatic complexity | 44 functions at CC 10 or more, each listed in `scripts/ci/baselines/complex_functions.json` | Below 10 (standard 2.8) | enforced by `scripts/ci/check_complexity_ratchet.py`: no new offender, no listed one grows |
| Duplication | About 2.8% in the library by an approximate scan; the other packages are not measured | At most 3 to 5% (standard 2.8) | proposed |
| Dependency vulnerabilities | `pip-audit --strict` over the exported lock on every push and pull request | None known | enforced |

Coding conventions, the layering rules and the import boundaries are in `CLAUDE.md`,
`CONTRIBUTING.md` and `docs/adr/`.
