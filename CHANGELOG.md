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
- **Pre-selection no longer drops low-volatility funds, and clips real crashes
  less.** `PreSelectionConfig.zero_variance_threshold` now defaults to `0.0`
  (was `1e-8`) and drops only columns that are constant up to float noise, so
  money-market and overnight-rate ETFs stay in the universe. The default
  `winsorize_threshold` (`PreSelectionConfig`, `OutlierTreater`, `CleaningConfig`)
  is now 8 sigma (was 3): a genuine single-name crash is no longer flattened
  before CVaR and drawdown are estimated. The `for_conservative` and
  `for_low_volatility` presets keep 2.5.
- **`RobustMeanRiskConfig.for_conservative`, `for_moderate` and `for_aggressive`
  maximise worst-case mean-variance utility, not the Sharpe ratio.** The ratio
  problem is infeasible whenever the worst-case excess return is non-positive,
  which is the normal case on 1-2 years of daily data.
- **`implied_annualization_factor` is required for an implied covariance.**
  `build_cov_estimator` raises `ConfigurationError` without a positive value;
  skfolio never inferred it from the index (it uses 252), so weekly data was
  de-annualised by the wrong factor.
- **`MomentEstimationConfig.for_regime_adjusted_ew()` uses the EW mean estimator**,
  so it fits a ragged panel like its covariance does.
- **Rebalancing treats weights that sum below 1 as cash.** `apply_no_trade_band`
  works to the target's invested level and `compute_drifted_weights` lets the
  cash sleeve earn 0, so a partly invested book no longer trades its cash away or
  reports drift with no market move.
- **Stricter inputs that used to fail silently.** A positive `min_sector_weights`
  floor for a sector absent from the mapping raises `ValueError`; a benchmark or
  factor `y` whose index differs from `X` (same length) raises `DataError`; the
  online grid and randomized searches raise `OptimizationError` when every
  candidate scores NaN and now apply `risk_free_rate=0.0` to every candidate;
  `to_returns` warns when `prices_to_returns` drops all-NaN columns and raises
  when none or the benchmark is left.
- **A weight cap that cannot reach the budget is reported at fit time.**
  `build_mean_risk` and `build_max_diversification` return estimators that warn
  (`UserWarning`) when `n_assets * max_weights` is below the budget, naming the cap,
  the asset count and the number of assets needed, instead of leaving only the
  solver's generic infeasibility error. The presets that default to
  `max_weights=0.10` need at least ten assets. Fitting, cloning and failure handling
  are otherwise unchanged.
- **`beta_neutral` factor-mimicking portfolios hedge with a lagged expanding
  beta** (NaN for the first 12 shared dates) instead of a full-sample beta, which
  used later data on earlier dates. `scores` must lead `returns`.
- **`CPCVConfig` purges and embargoes by default.** `purged_size` and
  `embargo_size` default to 5 (they were 0), presets included, so out-of-sample
  scores are no longer inflated by autocorrelated returns at the fold boundaries.
  The sizes count observations of whatever frequency is fed in; pass `1` for a
  monthly panel, or `0` to switch the guard off.
- **Black-Litterman helpers**: `market_weights` also accepts a ticker-indexed
  `pd.Series` (aligned to the fitted columns, so a pre-selection drop no longer
  breaks the shape); `AvailableAssetViewsPrior` subsets `empirical_omega` with
  the views and falls back to the equilibrium prior when no view is left.
  `SyntheticDataConfig()` always builds a vine copula with the log-transform
  guard.

### Added

- **Decision log, integration docs and a changelog check.** `docs/adr/` starts an
  architecture decision log with seven records (the strip to an ingestion pipeline,
  the four-package workspace, the removal of `optimizer.pipeline`, the heartbeat-lease
  reaper, the LangGraph schema, the provider registry and the branching, review and
  secrets policy), and `tests/test_adr_format.py` keeps their shape. `docs/integrations.md`
  documents every external system (Yahoo Finance, FRED, Trading 212, Trading
  Economics, Il Sole 24 Ore, Business Insider and the LLM providers) with its API,
  authentication, failure impact, risk and mitigation, plus a communication diagram
  and two sequence diagrams, and a test checks that each cited source file exists.
  `docs/nfr.md` records dated scalability, performance and maintainability baselines
  against their targets, `docs/test-plan.md` maps the critical functions to their
  tests, and `docs/requirements.md` holds the (still empty) requirement registry.
  An advisory `changelog` CI job (`scripts/ci/check_changelog.py`, pull requests only)
  fails a pull request that changes shipped code (sources, migrations, Dockerfiles,
  entrypoints, installers, dependency pins, agent prompts) without a `CHANGELOG.md`
  entry; the `skip-changelog` label overrides it and Dependabot is exempt. The `lint`
  job, `make lint` and pre-commit now also cover `scripts/`.
- **portopt-db PostgreSQL integration suite and migration checks in CI.** SQLite
  drops the `ON CONFLICT ON CONSTRAINT` target and never installs the
  `human_audit_log` triggers, so neither had a test. `packages/portopt-db/tests/
  integration` now runs `_upsert` (constraint and index targets, SAVEPOINT
  isolation), the append-only UPDATE/DELETE/TRUNCATE guards and a
  constraint-exists-in-the-migrated-schema check against a real server, keyed on
  `PORTOPT_TEST_DATABASE_URL` (skips without it, refuses a database not named
  `*_test`). A static test checks the 37 `_upsert` conflict names against the
  models, and the `postgres-integration` CI job now also runs `alembic check` and a
  downgrade/upgrade round trip over the five most recent revisions. The six
  revisions whose `downgrade` raises now declare an `IRREVERSIBLE_REASON`, which a
  test requires of any future one-way revision.
- **Ask is a read-only deep agent with a saved conversation.** A question is now
  answered by an agent that reads the fund's data (read-only tools over
  `fund.observe`) and a read-only staged copy of the docs and source, deciding its
  own steps (shown live on the status line), capped by `FUND_ASK_MAX_STEPS` and
  `FUND_ASK_TIMEOUT_SECONDS`. Follow-up questions work: each scope keeps one
  checkpointer thread that deepagents compacts automatically. The exact
  conversation is saved per portfolio (and for the fund-wide view) and shown when
  Ask opens; `/clear`, behind a confirm, deletes the saved messages, the agent's
  memory and its offloaded history files.
- **The Daily brief and Ask text is laid out to be read.** The market digest is a
  table (one macro series per line with value, date and signed change, then the
  markets and news themes), and Ask answers render their headings, bold, bullets
  and dividers instead of showing the markdown marks, with a blank line between
  turns.
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
- **fund / portopt-db / ingestion**: the **eligible universe**, so a portfolio can
  be restricted to a hand-picked set of instruments (the first real ETF portfolio
  could not run: its ETFs ranked below 7,000 by market cap, had no sector and
  failed every ESG exclusion)
  - `PortfolioMandate.eligible_tickers` — a hard allowlist, empty meaning
    unrestricted; set with `fund mandate set --eligible T ... | --clear-eligible`
    or the TUI Mandate view's "eligible tickers" field. Saving warns once about
    tickers without a priced instrument and syncs the portfolio's pins
  - `fund.tools.eligibility` resolves the allowlist once per run (run, resume and
    the adviser edit) into `RunContext.eligible`; `list_universe` returns only the
    allowed set (with `instrument_type` / `asset_class`), `universe_filter`,
    `estimate_moments` and `optimize_portfolio` refuse an outside ticker,
    `risk_check` records a `not_eligible` violation, and `place_orders` and the
    adviser edit refuse it. The PM seed lists the eligible and barred names
  - ESG look-through for ETFs: a fund is judged by its latest `etf_sector_weights`
    snapshot against the client's new `look_through_tolerance` (strict 0 by
    default); commodity and government / inflation-linked bond ETFs pass, other
    funds with a missing, stale (`FUND_ESG_SECTOR_WEIGHTS_MAX_AGE_DAYS`, 180
    days), partial or unmappable snapshot are barred `esg_unclassified`. Under an
    allowlist the stored-sector screen leaves ETFs to the resolver
  - `pinned_instruments` table (Alembic `e3a5c7d9b1f4`) and
    `PinnedInstrumentRepository`: a pinned ticker is never marked delisted by the
    universe build and keeps its classification and `t212_ticker`
  - `portopt add-instrument <TICKER> [--asset-class ...] [--pin]` validates one
    Yahoo ticker, stores the instrument with 5y of prices and the ETF fund data,
    and takes no job slot; ETF fund data is now fetched for every ETF, and
    `commodity` is a new asset class
  - `ETFMetadataRepository.get_latest_sector_weights` batches the latest sector
    snapshots for a list of tickers
  - `PortfolioMandate.tradeable_only`, an opt-in broker scope on top of the
    allowlist (default off, mandate JSON only, no migration): `fund mandate set
    --tradeable-only | --no-tradeable-only` or the TUI Mandate view's "Tradeable
    on Trading 212 only" box. The resolver bars an instrument whose
    `is_t212_tradeable` is an explicit False as `not_t212_tradeable`; NULL is not
    barred, and when nothing has been evaluated the universe stays whole with a
    note in the PM seed and the mandate-save warning. It also works without an
    allowlist, as a refusal list over the whole universe (`list_universe` drops the
    False rows). `UniverseRepository.get_t212_untradeable_tickers` is the new read
- **Function length and complexity ratchet.** `scripts/ci/check_complexity_ratchet.py`
  (a step of the `lint` job) holds standard 2.8 (at most 30 lines per function,
  cyclomatic complexity below 10) over the four source trees. The 274 functions
  that are already too long and the 50 that are too complex are recorded in
  `scripts/ci/baselines/`; the check fails a new offender and a recorded one that
  grew. `--update` on an existing baseline can only lower a stored value or drop
  an entry, never raise a value or add one, so the recorded debt shrinks as
  functions are split. Deleting a baseline file and running `--update` starts a
  fresh baseline from the current measurement, so a deleted or emptied baseline
  file in a diff deserves a reviewer's attention. A source root that is missing
  fails the check instead of measuring nothing.

### Changed

- **ci**: the CI workflow also runs on pushes to `feature/**`, `bugfix/**`,
  `hotfix/**` and `release/**`, so a change merged without a pull request still
  carries the status checks that branch protection on `main` and `development`
  requires; `tests/test_ci_triggers.py` guards the trigger list.
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
- **docs**: the architecture and decision records (`ARCHITECTURE.md`,
  `DECISIONS.md`, `SPEC-RECONCILIATION.md`, `VALIDATION.md`, `WRITE-PATH-AUDIT.md`)
  moved from the git-ignored `todo/` folder to a tracked `docs/`, so they are
  versioned and reviewable. Pointers to `SPEC.md`, `deep_agent.md` and
  `tasks/plan.md`, which were never committed, were dropped from `CLAUDE.md`,
  `fund/README.md`, `ci.yml` and code comments, and
  `tests/test_doc_references_resolve.py` fails on any new one. `CONTRIBUTING.md`
  now describes the uv workspace, the per-member test commands, branching from
  `development` and releasing through a `release/` branch, and the `portopt-db` README
  documents the migration process (validation, rollback, one-way revisions) with an ER
  diagram; new Alembic revisions get a dated file name.
- **ci**: the `fund-test` job timeout goes from 30 to 45 minutes. The job takes 25 to
  29 minutes, and a pull-request run competing with the push run was cancelled at
  30m17s although the same job passed on the same commit.

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

### Security

- **deps**: `soupsieve` 2.9.2 -> 4.1 in `uv.lock`. 4.0.0 is the first release that
  fixes CVE-2026-126190 and CVE-2026-126192 (super-linear CSS selector matching on
  untrusted HTML, reachable through the news and macro scrapers); the lock takes
  the current 4.1. `pip-audit --strict` failed on both, which turned the `lint`
  job red on every branch; `tests/test_lockfile_security_floors.py` now pins the
  4.0.0 floor.

### Fixed

- **fund, ingestion, portopt-db**: API keys, the broker secret, database
  passwords and the notification webhook URL no longer appear in `repr()` or
  `str()` of `FundConfig`, `DbConfig` or the ingestion `Settings`. A traceback that renders locals or a stray
  `logger.info(config)` used to print every provider key and the DB URL.
- **fund**: the risk gate and the weighting tools no longer judge a book on its
  youngest holding. The drawdown ceiling, `optimize_portfolio` and
  `estimate_moments` share one sample-depth rule (a fit needs at least half the
  median own history); an unverifiable drawdown now blocks instead of warning;
  `universe_filter` drops only assets that are thin on their own and no longer
  loses any asset that starts a bar after the oldest. A risk verdict is
  re-checked against the current constraint set before orders are placed, and the
  ESG look-through fails closed on non-finite weights, counts unreported weight
  as possible exposure at a strict tolerance, and ignores snapshots dated after
  the decision bar.
- **fund**: the risk gate compares the client's drawdown ceilings with the worst
  loss inside any one-year window of the history, because the stated tolerance is a
  one-year figure (the full-history drawdown is still reported). A common history
  shorter than one year blocks with `drawdown_unverified`, and a stated tolerance of
  zero is reported as `zero_loss_tolerance` instead of an opaque breach. Bars per
  year are derived from each panel's calendar (about 261 on a weekday union of
  venues, not a fixed 252) for moments, the optimizer, the risk-free rate and the
  Performance strip, so annualised figures move by up to about 3%. The profile's
  `moments_estimator` now reaches the optimizer's prior, so the covariance the PM
  reads is the covariance `optimize_portfolio` solves under. A legacy percent-style
  drift threshold that was clamped to "never fires" is shown as a warning on save,
  in `fund mandate show` and on the Posture view.
- **fund**: NAV starts at capital when the first ticket leaves cash undeployed
  (a later ticket used to lever the returns), a re-approved identical ticket
  becomes the active book again, holding contributions sum to the NAV return, a
  held currency with no FX rate fails loudly, and the NAV upsert drops rows the
  rebuilt series no longer emits.
- **fund**: an approve, edit or Ask approve resolves the gate the adviser was shown:
  the Decisions view remembers the id of the gate it rendered and the decision is
  refused (`GateMovedError`) when the run has since moved to another gate.
  `tradeable_only` keeps an allowlisted name the Trading 212 overlay could not judge
  (no ISIN) but names it in the PM seed and in the mandate-save warnings instead of
  letting it through silently.
- **fund**: approving a second gate on one run works (the idempotency key now
  names the gate, not the run), a database error inside a tool no longer rolls
  back the round's audit rows, a failed round is recorded after its rollback and
  its checkpoint thread deleted, mandate saves detect a concurrent change
  (`MandateConflictError`), and the first-round flag no longer blocks a freeze.
- **fund**: `OLLAMA_API_KEY` is sent only to `ollama.com` over https; a keyed
  gateway elsewhere needs `OLLAMA_BASE_URL_SEND_KEY=true`. The LangGraph
  bootstrap lock polls `pg_try_advisory_lock`, so two processes starting on a
  fresh database can no longer deadlock behind `CREATE INDEX CONCURRENTLY`.
- **ingestion**: `t212_annotate` judges mandate-pinned instruments, so
  `tradeable_only` bars a pinned name the broker does not list; the broker scope
  ignores delisted twin rows; a mandate save no longer revives delisted sibling
  rows or erases `delisting_return`; an empty Trading 212 catalogue no longer
  clears stored tickers; "Gold Explorers" and natural-gas equity funds are no
  longer labelled commodity (they skip the ESG look-through) and minimum-volatility
  equity ETFs are no longer dropped as volatility products.
- **ingestion**: a failed Yahoo fetch is no longer cached as "no data" for 72
  hours; the news search fallback goes through the shared rate limiter and
  circuit breaker; the news sweep reads only the ids it needs; an options sweep
  through an outage fails instead of reporting zero errors; a full-history fetch
  stops before today's unfinished bar; a delisted instrument with no bar for a
  year leaves the daily sweep. Migration `b8d0f2a4c6e1` turns JSON `null` values
  in `background_jobs` and `portfolio_journal` into SQL `NULL`.
- **ingestion**: `ticker_news` is pruned weekly (`NEWS_RETENTION_DAYS`, default
  365, 0 disables) in bounded batches as its own job slot; a prune error is
  recorded on that slot and never fails the refetch. The directly imported
  `certifi`, `urllib3`, `prompt-toolkit` and `python-dateutil` are declared in
  `ingestion/pyproject.toml`, with a guard test.
- **ops**: `install.sh` installs the CLI against constraints exported from the
  checkout's `uv.lock` (a failed pinned install is retried without them), and
  `install.ps1` now updates an existing checkout (fast-forward only, refusing a
  checkout of another repo) and stops when it cannot, as `install.sh` does. The CI
  boot test also starts and stops the fund container on a placeholder provider key,
  and a `windows-install` job runs `install.ps1` against stub git, uv and portopt on
  a Windows runner under Windows PowerShell 5.1 and PowerShell 7.
- **ops**: on a Linux host the fund container runs as the invoking user
  (`PORTOPT_FUND_UID` / `PORTOPT_FUND_GID`, handed to every compose call and
  defaulting to the image's 1000), so TUI exports to `./exports` are writable and
  readable by that user; macOS, a uid-1000 host and root keep the image's user. A
  leftover root-owned `./exports` is reported with the command that fixes it.
- **ops**: `portopt start` rebuilds an image whose source-hash label no longer
  matches the checkout (the first start after upgrading rebuilds once, because
  older images carry no label), so old containers no longer run against a newly
  migrated database. The ingestion entrypoint handles SIGTERM during boot,
  `portopt status` reports a deliberately disabled fund and times out docker
  calls, `./exports` is created for the invoking user, `install.sh` stops when
  the checkout cannot be updated, and the LLM placeholder keys in `.env.example`
  are commented out.
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
- **core**: `benjamini_hochberg` delegates to `correct_pvalues`, so a NaN p-value is
  left out of the family size instead of shrinking every threshold. With
  `[0.02, 0.04, NaN]` it used to mark nothing significant; it now marks the first
  two, as `correct_pvalues` always did
- **ingestion**: a Trading 212 `Retry-After` is capped at the same ceiling as
  `x-ratelimit-reset` (120 s) and a negative value falls back to the backoff, so a
  server-sent hour-scale value can no longer stall the universe build or make
  `time.sleep` raise
- **ingestion**: `retry_with_backoff` logs a WARNING with the last exception (or
  "result rejected by is_valid") when it gives up. Its callers only report "failed
  after retries", so the cause used to be lost. The text is passed through
  `redact_secrets` and no traceback is attached, because FRED sends its key as
  `?api_key=` and `requests` echoes the URL in its errors

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
