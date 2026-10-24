# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> **Most entries below concern the repository's ingestion service (`ingestion/`,
> formerly `api/`) and its tooling.** The exceptions are the library breaking
> changes in **Removed (BREAKING — library)** and **Changed (BREAKING — library)**
> immediately below.

### Removed (BREAKING — library)

- **`optimizer.pipeline` deleted entirely.** The library is now a set of
  composable, DB-agnostic primitives with no fixed end-to-end runner. Removed
  public symbols: `run_full_pipeline`, `run_full_pipeline_with_selection`,
  `optimize`, `backtest`, `tune_and_optimize`, `compute_net_backtest_returns`,
  `PortfolioResult`. `build_portfolio_pipeline` (pre-selection → optimiser
  composition) moved to `optimizer.pre_selection`. Opinionated, DB-connected
  orchestration (FX conversion, delisting correction, universe/factor selection,
  rebalancing decisions, persistence) now belongs to the planned `fund/` bridge,
  keeping `optimizer` DB-agnostic. Compose runs from primitives:
  `build_portfolio_pipeline(optimizer)` + `optimizer.validation.run_cross_val`.
- **`optimizer.factors` config knobs that no code path read.** Setting them
  changed nothing while implying a protection or a switch, so they are removed
  and passing them now raises `TypeError` (or `ImportError` for the names).
  - `FactorIntegrationConfig.max_weight` — never applied; cap positions with
    `MeanRiskConfig.max_weights`.
  - `FactorConstructionConfig.publication_lag` — never read, so it gave no
    point-in-time protection; shift fundamentals with
    `align_to_pit(lag_days=...)`, taking the lag from the matching
    `PublicationLagConfig` field, before computing factors.
  - `FactorValidationConfig.fmp_top_pct` / `fmp_bottom_pct` — mimicking-portfolio
    legs are set through the builder's `quantile` argument.
  - `GroupICAggregationConfig`, `ICWeightingMethod`, `ICNegativeFilterPolicy`,
    `GroupICResult` — no group-IC aggregation by t-stat weight or negative-IC
    filter existed.

### Changed (BREAKING — library)

- **`optimizer.factors` configs now validate at construction and raise
  `ConfigurationError`** for values that previously failed late or produced
  silently wrong output:
  - `FactorConstructionConfig`: every lookback must be `>= 1` and
    `0 <= momentum_skip < momentum_lookback - 1`.
  - `CompositeScoringConfig`: `ic_lookback >= 1`, tier weights finite and
    non-negative with a positive sum.
  - `SelectionConfig`: `target_count >= 1`, quantiles and `sector_tolerance` in
    `[0, 1]`, `exit_quantile <= target_quantile`, `buffer_fraction >= 0`.
  - `SectorRegimeBandsConfig`: repeated `(regime, sector)` rows are rejected, and
    a regime listing all 11 sectors must have caps summing to at least 1.0.
- **`ICIR` composite weighting now honours `ic_lookback` and
  `ic_decay_halflife`**, as `IC_WEIGHTED` already did. With the default
  `ic_lookback=36`, group IC histories longer than 36 periods are trimmed, so
  existing ICIR-weighted composites change; set a larger `ic_lookback` to keep
  the old whole-history behaviour.
- **Factor exposure band is relative to the selected set.**
  `FactorIntegrationConfig.exposure_lower_bound` / `exposure_upper_bound` now
  bound the exposure around the equal-weight exposure of the selected tickers,
  not around the universe mean (z-score 0). The old absolute bounds were
  infeasible for any top-N selection; a full-universe selection is unchanged.
- **`resolve_sector_bands` honours `SectorRegimeBandsConfig.enable`.** A disabled
  config (`SectorRegimeBandsConfig.disabled()`) now yields no sector constraints
  instead of an 11-sector, 33%-floor set, and an enabled config with no bands is
  filled with the default regime matrix instead of a flat 3%-16% band.
- **`standardize_factor` honours `neutralize_country` on its own.**
  `neutralize_country=True` with `neutralize_sector=False` now demeans within
  each country instead of returning un-neutralized scores.
- **`RegimeThresholdConfig.for_rolling_percentile` keeps
  `spread_2s10s_inversion` at its default of 0**; only the steep threshold is
  percentile-based, so a normal upward-sloping curve is no longer classified as
  inverted.

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
- **api**: renamed to **`ingestion/`** — the folder is a schedule-driven worker,
  not an HTTP service, so the name now describes the domain rather than a
  removed transport layer. Docker (`ingestion/Dockerfile`), CI (job renamed
  `api-test` → `ingestion-test`), the `scheduler/*.sh` wrappers, and the docs
  were all updated to match; no code paths changed

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

- **docker**: `docker-compose.yml` now pins the project name with a top-level
  `name: optimizer`. Without it Compose took the name from the checkout
  directory, so re-cloning or unzipping under another basename (for example
  `optimizer-development`) created a new, empty `<basename>_postgres_data`
  volume: the database looked wiped and a full refetch started. The pin matches
  the `optimizer_postgres_data` volume of installs cloned as `optimizer` (what
  the README says to do), which need no action. **Upgrade note — checkouts
  under any other name:** their data lives in `<basename>_postgres_data`, and
  the old containers still hold the fixed `container_name`s (`optimizer_db`,
  …), so the next `portopt start` would either fail with a name conflict or
  come up on an empty database. Before upgrading, either
  - keep the old project: set `COMPOSE_PROJECT_NAME=<basename>` (it overrides
    `name:`, as does `-p`) and nothing moves; or
  - move to the pinned name: run `docker compose -p <basename> --profile
    ingestion --profile fund --profile tools down` (it frees the container names
    and keeps the volume; never use `down -v`, which deletes the database), then
    copy the data across with `docker volume create optimizer_postgres_data &&
    docker run --rm -v <basename>_postgres_data:/from -v
    optimizer_postgres_data:/to postgres:16-alpine cp -a /from/. /to/`. Compose
    may warn that it did not create `optimizer_postgres_data`; it still uses it
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
