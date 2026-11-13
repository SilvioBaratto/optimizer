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

| Function | Why it is critical | Unit tests | Integration tests |
|---|---|---|---|
| `risk_check` | Gate between a proposal and an order (risk 1) | `fund/tests/unit/test_risk.py`, `fund/tests/unit/test_rolling_loss_gate.py`, `fund/tests/unit/test_toolsets.py` | |
| `place_orders` | Writes the paper ticket, idempotently (risk 1) | `fund/tests/unit/test_orders_reapprove.py`, `fund/tests/unit/test_tool_savepoint_recovery.py` | |
| `build_nav_series` | Marks the portfolio value (risk 1) | `fund/tests/unit/test_performance_nav.py`, `fund/tests/unit/test_performance_nav_chain.py` | |
| `resolve_mandate_eligibility` | Bars instruments outside the mandate (risk 1) | `fund/tests/unit/test_eligibility.py`, `fund/tests/unit/test_eligible_universe_e2e.py` | |
| `fund_drift_monitor` | Daily job that watches every portfolio (risk 3) | `fund/tests/unit/test_scheduler_steps.py`, `fund/tests/unit/test_worker.py`, `fund/tests/unit/test_fund_job_repo.py` | `fund/tests/integration/test_fund_job_claim_race.py` |
| `_upsert` | Every repository write goes through it (risk 2) | `packages/portopt-db/tests/test_upsert_constraint_names.py` | `packages/portopt-db/tests/integration/test_upsert_postgres.py`, `packages/portopt-db/tests/integration/test_upsert_constraints_exist_in_db.py` |
| `_fetch_and_store` | The 974-line per-instrument sweep (risks 2, 3) | `ingestion/tests/unit/test_yfinance_data_service_savepoints.py`, `ingestion/tests/unit/test_market_data_service_branches.py`, `ingestion/tests/unit/test_empty_fetch_memory.py` | |
| `retry_with_backoff` | Wraps every external call (risks 3, 4) | `ingestion/tests/unit/test_shared_infrastructure.py`, `ingestion/tests/unit/test_retry_exhaustion_logging.py` | |
| `benjamini_hochberg` | Factor significance (risk 5) | `tests/factors/test_validation.py` | |

Also critical and tested:

- The append-only audit trail (`human_audit_log`): `packages/portopt-db/tests/integration/test_audit_log_triggers.py`.
- Secrets kept out of `repr`: `fund/tests/unit/test_config_secrets_not_in_repr.py`,
  `ingestion/tests/unit/test_settings_repr.py`, `packages/portopt-db/tests/test_db_config_repr.py`.
- Irreversible migrations declare why: `packages/portopt-db/tests/test_irreversible_migrations_declare_reason.py`.

Known gap: the error handlers of the individual categories inside `_fetch_and_store` are
not pinned by a test. They are covered by characterization tests planned before that
function is split.

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

A test case is identified by `REQ-<AREA>-NNN` ids registered in `docs/requirements.md` and
bound with `@pytest.mark.criterion("REQ-...")`. No test carries one yet. That is a recorded
deviation from standard 6.6 (see `CONTRIBUTING.md`). Until the registry is filled and the CI
check exists, a test is identified by its pytest node id, `path::Class::test_name`, and its
docstring names the code it validates.

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
that fails without the change and names the function it guards. Linking the test to the
issue automatically (a `defect(issue)` marker and a CI report) is part of the traceability
work that also fills the registry above.

Coverage is checked on every CI build against floors of 90 (library, `portopt-db`) and 80
(`ingestion`, `fund`), against a standard goal of 75.
