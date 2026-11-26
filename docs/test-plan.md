# Test plan

Owner and single point of contact: the maintainer. The plan follows standard 6: it maps
the critical code to its tests (6.2), fixes how a test case is identified (6.4), lists
the acceptance tests (6.5) and says how defects and coverage are tracked (6.3, 6.6).

## Scope

In scope: the four packages (`portopt-core`, `portopt-db`, `ingestion`, `fund`), the
installers and the container images. Tools: pytest (plus hypothesis in the library),
in-memory SQLite for unit tests, PostgreSQL 16 for integration tests, GitHub Actions as
the CI.

Out of scope: usability tests (there is a CLI and a TUI, no GUI) and load tests. The
performance baseline is in `docs/nfr.md`.

## Risks

Ordered by what a failure costs. Each maps to the critical functions below.

1. **A wrong order or a wrong NAV.** The fund places paper tickets and marks NAV from
   model-assisted decisions. A weight vector must come only from the optimizer, pass the
   risk gate and a human approval.
2. **Corrupted or lost ingestion data.** An upsert aimed at the wrong constraint, a failed
   write that discards earlier ones, or a mutated audit trail.
3. **Silent ingestion stalls.** A job reaped or retried without a visible cause.
4. **Leaked credentials.** API keys in `repr`, tracebacks and logs.
5. **A wrong statistic.** Multiple-testing correction that disagrees with itself.

## Critical functions

The last column names the requirements of `docs/requirements.md` registered for the
function; the tests that cite one carry its id in a `criterion` marker.

| Function | Why it is critical | Unit tests | Integration tests | Requirement ids |
|---|---|---|---|---|
| `risk_check` | Gate between a proposal and an order (risk 1) | `fund/tests/unit/test_risk.py`, `fund/tests/unit/test_rolling_loss_gate.py`, `fund/tests/unit/test_toolsets.py` | | REQ-FUND-001, REQ-FUND-002, REQ-FUND-003 |
| `place_orders`, `price_ticket`, `PaperVenue` | Prices and writes the paper ticket, idempotently (risk 1) | `fund/tests/unit/test_orders_reapprove.py`, `fund/tests/unit/test_tool_savepoint_recovery.py`, `fund/tests/unit/test_price_ticket.py`, `fund/tests/unit/test_execution_venue.py` | | REQ-FUND-009, REQ-FUND-010, REQ-FUND-011 |
| `build_nav_series`, `load_nav_inputs`, `compute_nav` | Marks the portfolio value (risk 1) | `fund/tests/unit/test_performance_nav.py`, `fund/tests/unit/test_performance_nav_chain.py`, `fund/tests/unit/test_performance_nav_inputs.py` | | REQ-FUND-012, REQ-FUND-013 |
| `resolve_mandate_eligibility`, `resolve_eligibility`, `not_eligible_violations` | Bars instruments outside the mandate (risk 1) | `fund/tests/unit/test_eligibility.py`, `fund/tests/unit/test_eligible_universe_e2e.py` | | REQ-FUND-004, REQ-FUND-005, REQ-FUND-006, REQ-FUND-007, REQ-FUND-008 |
| `fund_drift_monitor` | Daily job that watches every portfolio (risk 3) | `fund/tests/unit/test_scheduler_steps.py`, `fund/tests/unit/test_worker.py`, `fund/tests/unit/test_fund_job_repo.py` | `fund/tests/integration/test_fund_job_claim_race.py` | REQ-FUND-014, REQ-FUND-015, REQ-FUND-016 |
| `_upsert` | Every repository write goes through it (risk 2) | `packages/portopt-db/tests/test_upsert_constraint_names.py` | `packages/portopt-db/tests/integration/test_upsert_postgres.py`, `packages/portopt-db/tests/integration/test_upsert_constraints_exist_in_db.py` | REQ-DB-001 |
| `_fetch_and_store` | The per-instrument sweep (risks 2, 3) | `ingestion/tests/unit/test_fetch_and_store_error_paths.py`, `ingestion/tests/unit/test_run_category.py`, `ingestion/tests/unit/test_fetch_plan.py`, `ingestion/tests/unit/test_price_policy.py`, `ingestion/tests/unit/test_yfinance_data_service_savepoints.py`, `ingestion/tests/unit/test_market_data_service_branches.py`, `ingestion/tests/unit/test_empty_fetch_memory.py` | | REQ-ING-001, REQ-ING-002, REQ-ING-003 |
| `DbCircuitBreaker` | Refuses new sessions to a dead database and decides when to try again (risk 3) | `packages/portopt-db/tests/test_db_circuit_breaker.py`, `ingestion/tests/unit/services/jobs/test_db_breaker_consumer.py`, `fund/tests/unit/test_tui_db_breaker.py` | `packages/portopt-db/tests/integration/test_circuit_breaker_postgres.py` | REQ-DB-002, REQ-DB-003, REQ-DB-004 |
| `retry_with_backoff`, `Trading212Client._rate_limit_wait`, `CircuitBreaker` | Retries calls to external sources and backs off while they fail (risks 3, 4) | `ingestion/tests/unit/test_shared_infrastructure.py`, `ingestion/tests/unit/test_retry_exhaustion_logging.py`, `ingestion/tests/unit/test_breaker_settings.py`, `ingestion/tests/unit/test_trading212_client.py` | | REQ-ING-004, REQ-ING-005, REQ-ING-006, REQ-ING-007 |
| `benjamini_hochberg` | Factor significance (risk 5) | `tests/factors/test_validation.py` | | REQ-CORE-001 |

Also critical and tested:

- The append-only audit trail (`human_audit_log`): `packages/portopt-db/tests/integration/test_audit_log_triggers.py`.
  No requirement id.
- Secrets kept out of `repr`: `fund/tests/unit/test_config_secrets_not_in_repr.py`,
  `fund/tests/unit/test_model_anthropic_secret_not_in_repr.py`,
  `fund/tests/unit/test_model_anthropic_wire.py`,
  `fund/tests/unit/test_model_factory.py`, `fund/tests/unit/test_model_openai_family.py`,
  `ingestion/tests/unit/test_settings_repr.py`,
  `packages/portopt-db/tests/test_db_config_repr.py`.
  REQ-FUND-017, REQ-FUND-018, REQ-FUND-019, REQ-FUND-020, REQ-FUND-021, REQ-FUND-022,
  REQ-FUND-023, REQ-FUND-025, REQ-FUND-029, REQ-ING-008, REQ-DB-005.
- Provider request lines kept out of the daemon log (`silence_http_request_logs`):
  `fund/tests/unit/test_config.py`, `fund/tests/unit/test_model_factory.py`. REQ-FUND-028.
- No usage telemetry from a provider SDK (data minimisation, standard 2.15.11): the
  `nvidia` chat model is built with the SDK's telemetry switched off, whatever the
  environment holds. `fund/tests/unit/test_model_factory.py`. REQ-FUND-026.
- What the `anthropic` SDK does with a request is pinned, so its OpenTelemetry spans
  stay inert and a release of it cannot add trace context, another host or a new place
  for the key unnoticed (data minimisation, standard 2.15.11):
  `fund/tests/unit/test_model_anthropic_wire.py`. REQ-FUND-029.
- What the `groq` SDK sends with a request is pinned, so a release of it cannot add a
  header, an endpoint or a new place for the key unnoticed (data minimisation,
  standard 2.15.11): `fund/tests/unit/test_model_groq_sdk_wire.py`. REQ-FUND-030.
- No chat model is built while `LANGSMITH_GATEWAY` is set, so prompts and keys never reach
  the LangSmith gateway: `fund/tests/unit/test_model_openai_family.py`. REQ-FUND-027.
- Irreversible migrations declare why: `packages/portopt-db/tests/test_irreversible_migrations_declare_reason.py`.
  REQ-DB-006.
- A fresh install builds its universe on the first daily run (issue 1125):
  `ingestion/tests/unit/test_scheduler_universe_bootstrap.py`. REQ-ING-009.

## Test levels

- **Unit.** Every suite, in-memory SQLite, external calls mocked. Run from each package
  directory.
- **Integration.** The `postgres-integration` CI job builds an empty PostgreSQL 16 database,
  applies the migrations, checks model and migration parity, downgrades and upgrades a
  window of revisions, and runs the `integration`-marked tests of `portopt-db`
  (`PORTOPT_TEST_DATABASE_URL`) and `fund` (`FUND_TEST_DATABASE_URL`).
- **System.** The `docker-images` CI job builds both images, boots both stacks with
  `up --wait` and checks that they stop cleanly.
- **Security.** `pip-audit --strict` over the exported lock on every push and pull request.

## Test case identification

A test case is identified by its pytest node id, `path::Class::test_name`. A test can
also cite a requirement of `docs/requirements.md` by its `REQ-<AREA>-NNN` id with
`@pytest.mark.criterion("REQ-...")`. Requirements are registered for the critical
functions above, for the items under the table that show an id, and for the traceability
check and its wiring (REQ-OPS-001, REQ-OPS-002). Tests written before the registry were
tagged only where they assert one of those requirements, so most tests cite no id. That
is a recorded deviation from standard 6.6 (see `CONTRIBUTING.md`).

The `lint` CI job collects the four suites with `--traceability-out` and runs
`scripts/ci/traceability.py check`. It fails when a registered requirement has no test,
when a test cites a `REQ-` id the registry does not list, or when a test carries a
`defect` marker but no `REQ-` id. The job uploads the four CSV files as the
`traceability` artifact of the run. Each file has one row per (`criterion` id, test),
and one row with an empty id for a test that carries a `defect` marker and no
`criterion` marker.

Each test follows the template of standard 6.4 in code form: the fixtures are the
preconditions, the calls are the actions, the assertions are the expected results and a green
run is the pass criterion. The test suite is the test-case repository.

## Acceptance tests

- **Automated.** The `docker-images` and `windows-install` CI jobs accept that both stacks
  deploy and that the Windows installer runs under PowerShell 5.1 and 7.
- **Not yet run in CI.** The seven test files in `ingestion/tests/integration/deployment/` need
  `--run-deployment`, which no CI job passes.
- **Manual, before a release.** With a throwaway portfolio, an operator runs
  `fund run <portfolio_id> --asof DATE`, reviews the proposal in the TUI, approves it with
  `fund approve <run_id>`, and checks `fund status` and `fund report <run_id>`: one paper
  ticket exists, the audit trail has the approval, and nothing was placed without it.

## Defects

Defects are GitHub issues. A fix pull request closes its issue and adds a regression test
that fails without the change and names the function it guards. The regression test also
carries `@pytest.mark.defect(<issue>)` beside the `criterion` id of the requirement it
guards, so the trace links the defect to its test, its requirement and, through the
registry row, its module; the check fails on a `defect` marker on a test that cites no
`REQ-` id. Issue 1125 is linked this way (REQ-ING-009); no other fix made before the
marker existed carries one.

Coverage is checked on every CI build against floors of 90 (library, `portopt-db`) and 80
(`ingestion`, `fund`), against a standard goal of 75.
