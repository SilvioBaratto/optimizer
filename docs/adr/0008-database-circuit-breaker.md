# ADR-0008: Guard database sessions with a circuit breaker

- Status: Accepted
- Date: 2026-11-14

## Context

Standard 3.3 asks for a circuit breaker on database calls, with a half-open state,
configurable thresholds, monitoring and a fallback. Before this record a
`DatabaseManager.get_session` session tried to connect at its first statement and failed
at once for a refused connection, or after the driver's `connect_timeout` for an
unreachable host (10 s, `config.py:71`). Ingestion steps met a dead database one after
another, and the fund cockpit let the adviser confirm a write that then failed. This record
puts a breaker in front of `get_session`; the paths known to reach the database another
way are listed under [Unguarded paths](#unguarded-paths).

Counting failures by exception class does not work. SQLAlchemy wraps driver errors, and
psycopg2's `QueryCanceled` (statement timeout, SQLSTATE 57014) and `LockNotAvailable`
(lock timeout, 55P03) subclass the driver's `OperationalError`. The ingestion daemon sets
a 60 s `lock_timeout` (`ingestion/app/config.py:61`), so a breaker counting each
`OperationalError` would open on a healthy database whenever a write waits on a lock.

`engine.py`, `resilience.py` and `config.py` below are in
`packages/portopt-db/src/portopt_db/`, `scheduler.py` is
`ingestion/app/services/jobs/scheduler.py`, and other paths start at the repository root.

## Decision

Each `DatabaseManager` builds its own `DbCircuitBreaker` unless one is injected
(`engine.py:89-107`), with no module global. The manager depends on the `CircuitBreaker`
protocol (`resilience.py:214-242`), and `get_session` asks it before opening a session
(`engine.py:226`).

- **Closed.** Sessions connect lazily and record their outcome at exit
  (`engine.py:227-236`). Consecutive connectivity failures are counted; a body that exits
  without raising resets the count (`engine.py:235-236`, `resilience.py:374-391`).
- **Open.** After `failure_threshold` consecutive connectivity failures
  (`resilience.py:436-454`), `before_call` raises `DatabaseUnavailableError`, a
  `RuntimeError`, before a session exists. Its message names no host, credentials or SQL
  (`resilience.py:462-471`).
- **Half-open.** Once `reset_timeout_s` has passed, `before_call` admits the next caller as
  the probe with a `ProbeTicket` (`resilience.py:346-372`). `get_session` connects that
  session before the caller's code runs, and the connect decides the probe
  (`engine.py:240-260`): success closes the breaker, a connectivity failure reopens it for
  a full timeout. Other callers are refused while the probe connects
  (`resilience.py:459-464`); the slot is free once the connect returns, so a long probe
  session holds nobody back, and its body does not decide the probe.

State, failure count, opening time and probe ticket change under one `threading.Lock`
(`resilience.py:316`). The clock is injected, so the unit tests do not sleep.

**What counts.** An error is classified in this order (`resilience.py:251-288`,
`393-424`):

1. a `DisconnectionError` counts;
2. a `DBAPIError` that SQLAlchemy marked `connection_invalidated` counts, tested before
   the SQLSTATE is read;
3. a SQLSTATE in class 08 or equal to 57P01, 57P02 or 57P03 counts;
4. an `OperationalError` without a SQLSTATE raised outside a statement counts;
5. any other driver error with a SQLSTATE is recorded as a success, because the server
   answered;
6. anything else, without a SQLSTATE, is inconclusive and leaves the count alone.

Step 4 is how a refused or timed-out connect at pool checkout surfaces. psycopg2's
`pgcode` and psycopg 3's `sqlstate` are both read (`resilience.py:245-248`). Statement
and lock timeouts, integrity, programming and data errors fall under step 5 unless
SQLAlchemy marked them invalidated. The installed psycopg2 dialect marks an error
invalidated when the connection is closed, or when the first line of its message contains
one of the libpq and SSL disconnect messages it lists with no double quote before it, such
as `terminating connection`, `closed the connection` or `could not receive data from
server`
(`.venv/lib/python3.12/site-packages/sqlalchemy/dialects/postgresql/psycopg2.py:815-863`).
So `server closed the connection unexpectedly`, which carries no SQLSTATE, counts under
step 2, and a server-side idle timeout such as `idle_in_transaction_session_timeout` (off
by default, `config.py:74`) may count.

**Inconclusive and late outcomes.** Step 6 covers, among others, a statement's driver
error without a SQLSTATE that SQLAlchemy did not mark `connection_invalidated` (SQLite
reports none), a pool-exhaustion `TimeoutError`, an exception from the caller's code,
`GeneratorExit`, `KeyboardInterrupt` and a closed manager. An inconclusive outcome that
carries the probe's ticket releases its slot (`resilience.py:423-424`, `431-434`).
While the breaker is open or half-open, an outcome without the probe's ticket is ignored
(`resilience.py:426-429`), so a session admitted before the breaker opened can neither
close it, extend it nor free the probe's slot.

**Configuration.** The defaults are 5 failures and 30 s (`resilience.py:47-48`).
`BreakerConfig` and `DbConfig` validate through `validate_breaker_settings`
(`resilience.py:96-123`, `191`; `config.py:86`): an `int` threshold above zero (a `bool`
is refused) and a finite timeout above zero.

| Variable | Default | Ingestion field | Fund field |
|---|---|---|---|
| `DATABASE_BREAKER_FAILURE_THRESHOLD` | 5 | `Settings.database_breaker_failure_threshold` | `FundConfig.db_breaker_failure_threshold` |
| `DATABASE_BREAKER_RESET_TIMEOUT_S` | 30.0 | `Settings.database_breaker_reset_timeout_s` | `FundConfig.db_breaker_reset_timeout_s` |

Both daemons parse the text with `parse_failure_threshold` and `parse_reset_timeout`
(`resilience.py:133-171`; `ingestion/app/config.py:79`, `88`;
`fund/src/fund/config.py:370`, `375`), so a value starts both or neither: plain digits for
the threshold, digits with an optional fraction for the timeout, blanks around either
allowed, both above zero. Other text, such as `5.0` for the threshold, `+3`, `1_0`,
`1e3`, `true`, `inf`, `nan` or an empty value, fails when the settings load, naming the
variable.

The env templates are edited by the repository owner. Until he lists the two variables
there, they are documented here and in `CLAUDE.md`, and the ingestion template guard
exempts the two names (`ingestion/tests/unit/setup/test_env_templates.py:170-180`); it
fails once the template names either, so the exemption is then removed.

**Why on by default.** Statement and lock timeouts (57014, 55P03) are outside the counted
SQLSTATEs (`resilience.py:52-53`). A false open refuses sessions for `reset_timeout_s`,
then a probe decides. `pool_pre_ping` (`config.py:68`) reconnects at checkout after a
failed ping (`.venv/lib/python3.12/site-packages/sqlalchemy/pool/base.py:1319-1366`), so
five consecutive connect failures are taken to mean the server is gone. The 30 s timeout is
short next to the 300 s ingestion orphan lease (`ingestion/app/config.py:165-166`).

## Alternatives considered

- **Do nothing.** Rejected: during an outage each call pays a connect attempt, the
  cockpit offers writes that cannot land, and standard 3.3 makes a breaker mandatory.
- **pgbouncer in front of PostgreSQL.** A pooler caps and reuses server connections, but
  clients keep trying, and the daemon and the cockpit are not told that the database is
  gone. It adds a service to a single-host deployment and solves connection count, a
  different problem; it could complement the breaker later.
- **Per-call retry with backoff**, as `retry_with_backoff` does for external APIs.
  Rejected: during an outage it multiplies the wait of each call. `pool_pre_ping` handles
  a dropped connection at checkout, and retrying a write that may have committed risks
  running it twice.

## Consequences

- Once the ingestion breaker is open, a step's job-slot claim is refused before a session
  exists and the step is reported as under [Guarded paths](#guarded-paths). Once the
  cockpit's breaker is open, its writes are greyed out from the next poll tick.
- The breaker is consulted when a session opens (`engine.py:226`); a session admitted
  before it opened keeps running (see [Known limits](#known-limits)).
- Each `DatabaseManager` has its own breaker (`engine.py:100-107`): the ingestion worker,
  the fund worker, the cockpit and each CLI run share no state.
- While the breaker is closed, a body that swallows its own connection error and exits
  cleanly records a success (`engine.py:235-236`), which delays the open.
- Access outside `get_session` bypasses the breaker. New access that cannot use it gets an
  entry under [Unguarded paths](#unguarded-paths) in the same change.
- Deploying needs the ingestion and fund images rebuilt and restarted; no migration, and
  the two variables keep the defaults when unset. When the live daemons pick it up is the
  owner's decision.

## Guarded paths

`DatabaseManager.get_session` (`engine.py:206-238`), and `health_check`, which opens its
session through it (`engine.py:190-191`). A lazy `initialize` reached through
`get_session` runs after `before_call` (`engine.py:226`, `262-273`).

Consumers that handle a refusal on purpose:

- Ingestion step stop: `_report_claim_failure` and `_fail_step` pass a refusal to
  `_report_db_unavailable`, which logs it without a traceback (`scheduler.py:288-324`,
  log call at 298); the step runs again at its next scheduled time, unless its row is
  still active (see [Known limits](#known-limits)).
- Counters `jobs_db_unavailable_total{domain}` for a stopped step
  (`ingestion/app/metrics.py:46`, `scheduler.py:303`) and
  `jobs_failed_status_refused_total{domain}` for a refused failed-status write
  (`ingestion/app/metrics.py:52`, `scheduler.py:346-347`, `352-377`).
- A failed-status write that lands for a refused step starts `db_unavailable:`
  (`scheduler.py:82`, `281-285`).
- A refused progress write or heartbeat is dropped with a WARNING without a traceback
  (`ingestion/app/services/jobs/background_job.py:29-39`, `236-238`, `377`); a refused
  reaper tick logs a WARNING (`scheduler.py:1108-1109`).
- Gauge `portopt_db_breaker_state`, 0 closed, 1 open, 2 half-open, read at each scrape
  (`ingestion/app/metrics.py:59-80`), bound at `ingestion/app/worker.py:97`.
- Cockpit gate: each poll tick (default 2.0 s, `fund/src/fund/tui/app.py:236`) re-reads
  the breaker (`app.py:476`, `fund/src/fund/tui/_mixins/dbgate.py:102-110`), and
  `check_action` disables the `DB_WRITE_ACTIONS` bindings while the gate is shut
  (`dbgate.py:28-43`, `112-125`).
- Ask's decision verbs check the same gate before a confirm is offered and when a confirm,
  the fund re-arm rationale or a steer directive is answered
  (`fund/src/fund/tui/_mixins/ask.py:143`, `214`, `240`, `257`; check at `173-183`), so
  an answer given after the gate closed at a poll tick is refused (`dbgate.py:92-100`).
- The attention feed logs a refused read without a traceback, once while that read keeps
  failing (`fund/src/fund/tui/_mixins/attention.py:199-209`).

## Unguarded paths

Not guarded: the breaker neither counts these paths' failures nor refuses them.

- Container start, `pg_isready` wait and `alembic upgrade head`:
  `ingestion/entrypoint.sh:31-44`, `fund/entrypoint.sh:31-53`, Alembic's own engine at
  `packages/portopt-db/alembic/env.py:73`. Compose restarts the `scheduler` and `fund`
  services (`docker-compose.yml:123`, `227`).
- `DatabaseManager.initialize`'s direct `engine.connect()` test (`engine.py:127`,
  `171-178`), called by `init_db` (`ingestion/app/database.py:64`,
  `fund/src/fund/worker.py:296`). In ingestion's default `development` environment
  `init_db` logs the error and returns (`ingestion/app/database.py:81-84`,
  `ingestion/app/config.py:289`).
- `_require_migrated_schema`, `inspect(engine).has_table`:
  `ingestion/app/database.py:45-57`.
- APScheduler job stores on the manager's engine: `scheduler.py:1268` (started at 1283),
  `fund/src/fund/scheduler.py:1111`. A failing `get_due_jobs` is logged as a WARNING and
  no job is taken from that store on that pass
  (`.venv/lib/python3.12/site-packages/apscheduler/schedulers/base.py:1160-1175`).
- fund `setup_langgraph`, two `psycopg.connect` calls
  (`fund/src/fund/audit/persistence.py:118`, `159`), called by the fund worker
  (`fund/src/fund/worker.py:130`), the cockpit (`fund/src/fund/tui/app.py:660`), the
  scheduler's runtime (`fund/src/fund/scheduler.py:1185`) and the CLI
  (`fund/src/fund/cli.py:432`, `493`, `568`).
- fund LangGraph `ConnectionPool` for the checkpointer and the Store
  (`fund/src/fund/audit/persistence.py:194`), used among others by the Ask history and the
  MiFID constraint-set cache (`fund/src/fund/audit/mifid_repository.py:147`, `318`).
- fund round bookkeeping on `Session(bind=session.get_bind())`:
  `fund/src/fund/agents/graph_failures.py:33`.
- `DatabaseManager.create_all_tables` (`engine.py:283-289`); no code under
  `ingestion/app` or `fund/src` calls it.
- Generic: `database_manager.engine` used directly, a `Session(bind=...)` on an existing
  engine, an engine of its own, a psycopg connection or pool, an Alembic run.

## Not verified

- What the cockpit's other six views (Posture, Deliberation, Decisions, Mandate, Audit,
  Ask) show while the breaker is open.
- Whether the breaker opens during a parallel yfinance sweep (see the clean-exit
  consequence above).
- What the ingestion CLI does after a failed `init_db`.
- Whether ingestion steps other than the yfinance sweep absorb a refusal in their own code.
- When the fund worker's breaker opens. Its job store is on the same engine
  (`fund/src/fund/scheduler.py:1104-1111`), so no new tick is taken while the store is
  unreadable (`apscheduler/schedulers/base.py:1160-1175`); a tick in flight records
  failures through its heartbeat pulses (`fund/src/fund/scheduler.py:158-172`, cadence at
  `fund/src/fund/config.py:174`) and its other sessions.
- Whether a fund caller that opens a `get_session` session before touching the LangGraph
  pool is refused before the pool waits (not re-read for this revision).

## Known limits

- A refused failed-status write leaves the row active: `running`, or `pending` when its
  `running` write (`scheduler.py:443`) was refused too, since the claim stamps `pending`
  (`ingestion/app/repositories/jobs/background_job_repository.py:248`). Either status
  blocks a re-run (`background_job_repository.py:29`, `240-243`). Two paths fail the row,
  both through `get_session`: the periodic orphan reaper (`scheduler.py:1067-1111`), whose
  error under the default `fail` strategy (`ingestion/app/config.py:181-182`) reads
  `orphaned — heartbeat stale, reaped by periodic reaper` (`scheduler.py:1093`), and
  `_reconcile_orphans` at worker boot (`ingestion/app/worker.py:102-121`), whose error
  reads `orphaned at startup — process restarted` (`worker.py:112`).
- Reap timing: the reaper runs every 300 s through `get_session` (`scheduler.py:1198-1202`,
  `ingestion/app/config.py:165-166`) and fails rows whose last heartbeat is more than
  300 s old (`background_job_repository.py:399-409`). The row is reaped at the first
  reaper tick at least 300 s after its last heartbeat at which the database answers and
  the breaker admits that tick's session. If the first tick after the lease expired gets
  through, that is 300 to 600 s after the last heartbeat; otherwise a later tick that
  gets through reaps it.
- The yfinance sweep (`ingestion/app/services/market_data/yfinance_data_service.py`) is
  not cut short. In parallel mode (`YFINANCE_FETCH_WORKERS=4`) each ticker opens its own
  session before it fetches (`2221`) and `_worker` turns an exception into that ticker's
  error (`2308-2325`); once the breaker is open a starting ticker is refused, except a
  probe, admitted one at a time once the timeout has passed (`resilience.py:346-372`).
  In serial mode
  (`YFINANCE_FETCH_WORKERS=1`) the sweep runs in one session admitted before its loop
  (`2072`), with an `except Exception` per ticker (`2133`). A refused progress write or
  heartbeat is dropped with a WARNING (`background_job.py:236-238`, `377`). The final
  `status="completed"` write (`2029-2034`) is a status write, which raises when refused
  (`background_job.py:236-237`), so the step can stop after its list. Follow-ups that
  would cut it short: re-raise `DatabaseUnavailableError` and connection errors past the
  per-ticker handler, or set the step's cancel event when a progress write is refused.
- A fund tick writes its status with `mark_done` on its own session
  (`fund/src/fund/scheduler.py:385`, `434`, `564`, `781`, `783`), admitted when it opened
  (`414`, `544`, `765`). The breaker does not refuse that write, and no counter records it
  if it fails.
- Cockpit texts once reads are refused. Performance shows `(performance unavailable: ...)`
  with the risk line blanked (`fund/src/fund/tui/_views/performance.py:273-277`) and
  `holdings unavailable: ...` in the detail pane (`performance.py:320-321`); its read
  returns the error instead of raising (`performance.py:358-365`). Daily brief shows
  `brief unavailable: ...`, followed by the last good brief of the same portfolio when
  there is one (`fund/src/fund/tui/_views/daily_brief.py:148-158`, error text from
  `254-261`).
- A stopped ingestion step logs lines without a traceback; how many depends on where the
  refusal surfaces:
  - at the claim, the `step stopped` line (`scheduler.py:298`), since the refusal is
    raised at `engine.py:226`, before the session's `try`;
  - at the `running` write or in step code outside a session, the `step stopped` line,
    then `could not record failed status` (`scheduler.py:367`) when the failed-status
    write is refused too;
  - inside a session the step holds open (calendars,
    `ingestion/app/services/market_data/calendars_service.py:41`, `78-83`; macro news,
    `ingestion/app/services/macro/macro_regime_service.py:704`, `723-728`; the serial
    yfinance sweep, `yfinance_data_service.py:2072`, `2144`), first the ERROR
    `Unexpected error in database session: database unavailable: ...`
    (`engine.py:44-49`, `229-231`, `279`), then the lines of the previous case.
- `_universe_is_empty`, the daily pipeline's first-run bootstrap check, logs a WARNING with
  a traceback when its session is refused, then skips the bootstrap
  (`scheduler.py:775-791`).
- The fund CLI starts a process per command, whose breaker starts closed
  (`resilience.py:317`), so an outage meets the driver's error. `profile`, `run`,
  `approve` and `reject` run `setup_langgraph` before their `try`
  (`fund/src/fund/cli.py:432`, `493`, `568`), so an outage ends them in a traceback.
  Handling it with a short message and a CLI test is a follow-up.
- The fund worker serves no metrics (no `prometheus` import under `fund/src`); its breaker
  state shows in its log.
