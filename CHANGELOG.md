# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> **The `portopt` library API is unchanged.** Everything below concerns the
> repository's `api/` service and its tooling. If you install `portopt` from
> PyPI, nothing here affects you.

### Added

- Structured logging with `NullHandler` pattern across all modules (#36)
- CI Python version matrix testing (3.10, 3.11, 3.12) (#38)
- Test coverage reporting with Codecov integration (#39)
- This CHANGELOG file (#40)
- CI restructure: parallel jobs, concurrency control, format checking (#41)
- Packaging: project URLs, dependency groups, version strategy (#45)
- Community files: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CITATION.cff (#46)
- GitHub issue and PR templates (#47)
- Dependabot for dependency and GitHub Actions updates (#48)
- Property-based testing with Hypothesis (#49)
- Developer experience: Makefile (#50)
- Automated release workflow (tag to PyPI publish) (#52)
- Pyright compatibility configuration (#53)
- **api**: `app/worker.py` — the ingestion daemon entrypoint. Serves Prometheus
  metrics, reaps orphaned jobs, bootstraps benchmarks off-thread, then runs
  APScheduler until SIGTERM
- **api**: `app/cli.py` — manual runs (`python -m app.cli daily`, `yfinance`,
  `macro`, `fred`, `news`, `summarize`, `calibrate`, `universe`,
  `reference-indices`, `refetch-all`). Each goes through the same job-slot and
  heartbeat path the scheduler uses, so a manual run is refused rather than
  double-fetching when the scheduler is mid-step
- **api**: `universe_build` promoted to a scheduled job (Sun 02:00, ahead of
  `weekly_refetch` — every other step iterates the `instruments` table, so a
  stale universe silently caps what yfinance fetches)

### Changed

- **api**: the service is now a headless ingestion daemon. APScheduler runs
  in-process; there is no HTTP API. Job state is read from the `background_jobs`
  table, the logs, and Prometheus on `METRICS_PORT` (default 9000, also the
  container healthcheck target)
- **api**: the scheduler is composed of public per-step functions
  (`run_yfinance_step`, `run_macro_step`, …). The cron pipelines and the CLI both
  build on those, instead of maintaining parallel code paths that drift
- **api**: no longer depends on the `optimizer` library. The daemon ingests; it
  does not optimize. The image no longer carries the sklearn/skfolio/scipy stack,
  and CI enforces the split by installing only `api/requirements.txt`
- **api**: `scheduler/fetch.sh` and `refetch_all.sh` are thin wrappers over the
  CLI rather than curl-and-poll drivers against HTTP endpoints
- **docker**: compose service `api` → `scheduler`; the API port is gone and 9000
  (metrics) is published instead
- Type-check target raised to Python 3.12 (see Fixed). Runtime 3.10 support is
  unchanged, still enforced by ruff, pyright, and the 3.10 CI matrix leg

### Removed

- **api**: the entire HTTP layer — every `/api/v1/*` route, FastAPI, uvicorn, the
  API-key auth middleware, and the `api_keys` table
- **api**: the portfolio, optimization, backtest, factor, risk, rebalancing,
  attribution, dashboard, reports, scenarios, and views domains, across routes,
  services, repositories, models, and schemas
- **api**: six yfinance sub-clients that wrote nothing to the database — `market`,
  `sectors`, `screener`, `calendars`, `streaming`, `funds`
- **db**: migration `d1e2f3a4b5c6` drops the 17 tables those domains owned. It is
  **destructive and one-way** — `downgrade()` raises rather than recreate empty
  tables whose rows and owning code are both gone. Restore from a pre-upgrade dump
- `scheduler/smoke.sh` and `smoke.yml`, which drove `/optimize` and `/backtest`
- `TRADING_ECONOMICS_API_KEY`, plus Azure OpenAI / `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY` from `.env.example`. None were read by any code — Trading
  Economics and Il Sole 24 Ore are scraped from HTML and take no key
- The `examples/` directory and the MkDocs documentation site (#50, #51). Both were
  added and removed while unreleased, so the entries above are corrected rather
  than carried forward as history

### Fixed

- **api**: synchronous scheduler steps now run a heartbeat companion. Only the
  heartbeat stamps `last_heartbeat_at`, so a step running in the scheduler thread
  went silent to the orphan reaper and was failed mid-run once it outlived the
  300s timeout. The reference-index seed (~12 minutes) was reaped at 5/12 tickers
  while still fetching successfully — the row said `failed`, the data said fine,
  and nothing reconciled the two
- **api**: the Il Sole scraper now pins `Accept-Encoding: gzip, deflate` instead of
  inheriting requests' default, which silently becomes `gzip, deflate, br` as soon
  as brotli is importable anywhere on `sys.path`. That is the same undecodable-body
  failure that froze the Trading Economics table for 16 days; TE was pinned
  afterwards, Il Sole never was
- **build**: `mypy` was configured for Python 3.10, so it aborted on numpy's PEP 695
  stubs before reaching any project file — the typecheck job had been passing while
  checking nothing. It now checks all 98 source files

## [0.1.0] - 2026-02-21

### Added

- **preprocessing**: Data validation, outlier treatment, sector imputation, regression imputation, delisting adjustment
- **pre_selection**: Pipeline assembly with skfolio selectors (SelectComplete, DropZeroVariance, DropCorrelated, SelectKExtremes, SelectNonDominated, SelectNonExpiring)
- **moments**: Expected return and covariance estimation (empirical, shrinkage, denoised), HMM regime blending, DMM via Pyro, lognormal scaling
- **views**: Black-Litterman, Entropy Pooling, Opinion Pooling integration frameworks
- **optimization**: Mean-Risk, Risk Budgeting, HRP, HERC, NCO, Maximum Diversification, Benchmark Tracking, Equal-Weighted, Inverse-Volatility, Stacking; robust and distributionally robust variants; regime-conditional risk
- **synthetic**: Vine copula models, synthetic data generation, conditional stress testing
- **validation**: Walk-Forward, Combinatorial Purged CV, Multiple Randomized CV
- **scoring**: Performance scoring for model selection
- **tuning**: Grid search and randomized search with temporal cross-validation
- **rebalancing**: Calendar-based, threshold-based, and hybrid rebalancing; turnover and cost computation
- **pipeline**: End-to-end orchestration from prices to validated weights
- **universe**: Investability screening with hysteresis-based entry/exit thresholds
- **factors**: Factor construction, standardization, composite scoring, stock selection, regime tilts, validation, mimicking portfolios, integration with optimization

[Unreleased]: https://github.com/SilvioBaratto/optimizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SilvioBaratto/optimizer/releases/tag/v0.1.0
