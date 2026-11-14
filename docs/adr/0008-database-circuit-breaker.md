# ADR-0008: Guard database sessions with a circuit breaker

- Status: Accepted
- Date: 2026-11-14

## Context

Standard 3.3 requires every database call to go through a circuit breaker with a
half-open state, configurable thresholds, monitoring and a fallback. This record puts
one in front of `DatabaseManager.get_session`, which the repositories, the services
and the fund cockpit's reads and writes go through. It does not cover every database
call. The paths that still reach the database another way, such as the APScheduler
job stores and fund's LangGraph pool, are listed under
[Unguarded paths](#unguarded-paths), each with its outage behaviour and its fallback.
Before this record nothing stood between a caller and a dead database. Every
`DatabaseManager.get_session` built a session, the first statement waited out the
driver's `connect_timeout` (10 s), and the error crossed the session with a
traceback. An ingestion pipeline runs six steps back to back, each with worker,
progress and heartbeat threads, so an outage cost minutes per run and a traceback
per call. The fund cockpit let the adviser read a gate and answer a confirm, only for
the write to fail afterwards.

Counting failures by exception class does not work. SQLAlchemy wraps driver errors,
and psycopg2's `QueryCanceled` (statement timeout, SQLSTATE 57014) and
`LockNotAvailable` (lock timeout, 55P03) subclass the driver's `OperationalError`.
The daemon runs with a 60 s `lock_timeout`, so a breaker that counts every
`OperationalError` would open on a healthy database whenever a write waits on a lock.

## Decision

Every `DatabaseManager` is guarded by its own breaker, one per manager instance with
no module global. `DatabaseManager` depends on the `portopt_db.resilience.CircuitBreaker`
protocol; `DbCircuitBreaker` is the default implementation. `get_session` asks the
breaker before it builds anything.

- **Closed.** Sessions pass and connect lazily on first use. Each session records one
  outcome when it exits. Consecutive connectivity failures are counted, and a session
  that reaches the database resets the count.
- **Open.** After `failure_threshold` consecutive connectivity failures, sessions are
  refused at once with `DatabaseUnavailableError` (a `RuntimeError`). No engine or
  session is created. The error message is one line without host, credentials or SQL.
- **Half-open.** Once `reset_timeout_s` has passed, `before_call` admits the next caller
  as the single probe and hands it a `ProbeTicket`. `get_session` connects that session
  eagerly, before the caller's code runs, and the connect decides the probe. Success
  closes the breaker. A connectivity failure reopens it, and the full timeout restarts
  from that failure; the probe's body never runs. Other callers are refused only while
  the probe connects. A probe session that stays open for a long time (a fund round
  runs in one session) therefore holds nobody back. A probe whose body runs no SQL, or
  swallows its own connection error, cannot close the breaker without evidence.

Every state change happens under one `threading.Lock`: the state, the failure count,
the time the breaker opened and the ticket of the probe in flight. The clock is
injected, so the tests never sleep.

**What counts.** `DisconnectionError` (the pool's pre-ping), any `DBAPIError` that
SQLAlchemy marked `connection_invalidated`, a SQLSTATE in class 08 or equal to 57P01,
57P02 or 57P03 (the server shutting down, crashing or still starting), and an
`OperationalError` without a SQLSTATE raised outside a statement, which is how a
refused or timed-out connect at pool checkout surfaces. psycopg2's `pgcode` and
psycopg 3's `sqlstate` are both read.

**What proves the server answered.** A driver error that carries any other SQLSTATE:
statement and lock timeouts, integrity, programming and data errors. It is recorded as
a success, so it resets the count and closes a half-open breaker.

**Inconclusive outcomes.** These say nothing about the database:

- any other driver error without a SQLSTATE (SQLite reports none);
- a pool-exhaustion `TimeoutError`;
- an exception from the caller's own code;
- `GeneratorExit`, `KeyboardInterrupt`;
- a closed manager.

The count is left alone. When the outcome carries the probe's ticket, the probe's slot
is released so the next caller probes.

**Late outcomes.** Only an outcome recorded with the probe's ticket decides a half-open
breaker. While the breaker is open or half-open, an outcome from any other session is
ignored, so a session admitted before the breaker opened can neither close it, extend
it nor free the probe's slot. Because the probe is settled at its connect, the slot is
free again as soon as the connect returns, whatever the body does afterwards.

**Defaults, on by default, set from the environment.** The defaults are 5 failures and
30 s. They are defined once, in `portopt_db.resilience`. `validate_breaker_settings` is
the one validator: the threshold must be an `int` above zero (a `bool` is refused), and
the timeout a finite number above zero. `BreakerConfig` and `DbConfig` both call it.

Both daemons read the settings from the environment and pass them into `DbConfig`:

| Variable | Default | Ingestion field | Fund field |
|---|---|---|---|
| `DATABASE_BREAKER_FAILURE_THRESHOLD` | 5 | `Settings.database_breaker_failure_threshold` | `FundConfig.db_breaker_failure_threshold` |
| `DATABASE_BREAKER_RESET_TIMEOUT_S` | 30.0 | `Settings.database_breaker_reset_timeout_s` | `FundConfig.db_breaker_reset_timeout_s` |

Both daemons parse the text with the same strict parsers, `parse_failure_threshold`
and `parse_reset_timeout` in `portopt_db.resilience`, so a value starts both or
neither. The threshold must be plain digits and the timeout digits with an optional
fraction, blanks around either allowed, and both must be above zero. Anything else
fails when the settings load, with the variable's name in the message: `5.0` for the
threshold, `+3`, `1_0`, `1e3`, `true`, `inf`, `nan`, or an empty value.

The env templates are edited by the repository owner only. Until he lists the two
variables there, they are documented here and in `CLAUDE.md`. The ingestion template
guard accepts exactly these two names as documented outside the template. It fails
once the template names either of them, so the exemption must then be removed.

The breaker is on by default because it only counts connectivity failures, so a
healthy database never sees it. A false open costs at most 30 s of refused sessions,
and then one probe decides. `pool_pre_ping` already absorbs a single dropped connection
by reconnecting at checkout. Five consecutive connect failures across the daemon's
threads therefore mean the server is gone, not a blip. 30 s matches the heartbeat
cadence and the health-check cache. It is short next to the 300 s orphan lease, so a
recovered database is noticed long before the reaper acts, and long enough that a
restarting server is not hammered.

**Fallbacks.**

- *Ingestion daemon.*
  - A step that meets the open breaker stops. It writes one ERROR line, without a
    traceback, and increments `jobs_db_unavailable_total{domain}`. The next scheduled
    run retries it.
  - A step's failed-status write that the breaker refuses is logged at that moment and
    increments `jobs_failed_status_refused_total{domain}`, whatever stopped the step.
    Such a row stays `running` until the heartbeat-lease orphan reaper (ADR-0004) fails
    it, once the database answers again. Its error then reads `orphaned - ...`, not
    `db_unavailable`, so the refusal is visible only in that log line and counter.
  - When the write does get through, the error starts `db_unavailable:`.
  - The reaper tick, refused progress writes and refused heartbeats each log one line
    without a traceback.
  - Known limit: `_universe_is_empty`, the first-run bootstrap check at the start of
    every daily pipeline, logs a WARNING with a traceback when its session is refused.
    It then skips the bootstrap, which is the intended reading of an unknown answer.
  - The gauge `portopt_db_breaker_state` reports 0 (closed), 1 (open) and 2
    (half-open), read at each scrape.
- *Fund cockpit (`fund-tui`).*
  - While the breaker of `fund.database`'s manager is not closed, the bindings that
    write are disabled: approve, reject, killswitch, re-arm, save mandate, guardrail
    override, new portfolio, profile, run, barge-in, edit and steer. They show greyed
    out and do not run.
  - Ask's decision verbs read the same gate. They are refused before any confirm or
    overlay is offered. They are refused again when an answer given after the breaker
    opened would dispatch: a confirm, the fund re-arm rationale, or a steer directive.
  - The status bar and the Ask log show the refusal as `describe_error` renders it.
  - Reads are refused at once by the same breaker, and each view keeps its last paint,
    which gives a read-only view of the last known state. The attention feed logs the
    refusal once per outage, in one line.
  - Once the timeout has passed, the next poll read is the probe. When its connect
    succeeds, the writes come back and the status bar says the database is reachable
    again.
- *Fund CLI (`fund`), a known limit.*
  - Each command is a fresh process with its own breaker, which starts closed. In an
    outage the first failed connect therefore raises the driver's error, not
    `DatabaseUnavailableError`. The breaker would refuse only after 5 failures within
    one process, and these commands open one session or a few.
  - `profile`, `run`, `approve` and `reject` call `setup_langgraph` before their
    `try`. Its `psycopg.OperationalError` escapes as an uncaught Typer traceback
    (exit 1).
    - Their `except (RuntimeError, ValueError, LookupError)` would print
      `error: database unavailable: ...` for a refusal, but an outage never reaches
      it.
    - If the LangGraph setup succeeded and the session then failed, the driver's
      `sqlalchemy.exc.OperationalError` is not in that tuple either, so it is also a
      traceback.
  - `mandate set` catches only `ValueError`. `mandate show`, `status` and `report`
    catch nothing. `nav-mark` runs `run_nav_mark`, whose first session is opened
    outside its `try`. Each prints a traceback and exits 1.
  - That is the behaviour these commands had before this record. Handling the driver
    error, `psycopg.OperationalError` and the refusal with one clear line, with a CLI
    test, is a follow-up.
- *Fund worker daemon (`fund-worker`), a known limit.*
  - The rebalance sweep, the drift monitor and the NAV mark
    (`fund/src/fund/scheduler.py`) open their first session outside their `try`. The
    refusal, or the driver error before the breaker opens, reaches APScheduler, which
    logs `Job "..." raised an exception` with a traceback.
  - Only the orphan reaper catches it, and it logs
    `orphan reaper: tick failed (non-fatal)` with a traceback.
  - Either way each tick logs one traceback while the breaker is open. If the database
    dies mid-tick, the per-portfolio handlers log one traceback per portfolio.
  - Their failed-status writes are refused the same way as the ingestion daemon's.
  - The fund worker serves no metrics, so its breaker state is visible only in the log.

## Alternatives considered

- **Do nothing.** Keep relying on `connect_timeout`, `pool_pre_ping` and the orphan
  reaper. Rejected: every call keeps paying the connect timeout during an outage, every
  call logs a traceback, the cockpit offers writes that cannot land, and standard 3.3
  makes a breaker mandatory.
- **pgbouncer in front of PostgreSQL.** A pooler caps and reuses server connections,
  but it does not stop clients from trying. When the server is down the clients still
  wait and fail one by one, and nothing tells the daemon or the cockpit that the
  database is gone. It also adds a service to a single-host deployment. It solves
  connection count, a different problem, and could complement the breaker later.
- **Per-call retry with backoff**, as `retry_with_backoff` does for external APIs.
  Rejected: during an outage it multiplies the wait of every call instead of removing
  it, so it amplifies the cost the breaker removes. The transient case it helps with,
  a dropped connection, is already handled by `pool_pre_ping` at checkout. Retrying a
  write that may have committed also risks running it twice.

## Consequences

- An outage fails each ingestion step in milliseconds with one log line and a counter.
  Operators see it on `portopt_db_breaker_state`, and the cockpit says it instead of
  failing after a confirm.
- The breaker gates the entry to a session and sees only the outcomes that cross
  `get_session`. The yfinance sweep is therefore not cut short when the database dies
  mid-run, for two reasons:
  - it does its per-ticker work inside one long session, opened before the breaker
    opened, so it is never asked again;
  - each ticker runs under its own `except Exception`, which absorbs the failure
    inside that session.

  The sweep keeps fetching from Yahoo until the step ends, its progress writes refused
  one line each.

  Two follow-ups would cut it short: re-raise `DatabaseUnavailableError` and connection
  errors past the per-ticker handler, or set the step's cancel event when a progress
  write is refused. Either is a change to the sweep, outside this record.
- While the breaker is closed, a body that swallows its own connection error and exits
  cleanly records a success and resets the count, which delays the open. The probe is
  immune because its connect, not its exit, decides it.
- Each process has its own breaker: the ingestion worker, the fund worker, the cockpit
  and each CLI run. They do not share state.
- A manual re-run of a step whose row the breaker left `running` is refused as already
  running until the 300 s lease expires and the reaper fails the row.
- Only `DatabaseManager.get_session` is guarded. Any other access bypasses the
  breaker:
  - `database_manager.engine` used directly;
  - a `Session(bind=...)` on an existing session's engine;
  - an engine of its own;
  - a psycopg connection or pool;
  - an Alembic run.

  New database access should go through `get_session`. Access that cannot gets an
  entry under [Unguarded paths](#unguarded-paths) in the same change, saying what it
  does during an outage and what the operator sees.
- Deploying this needs the running ingestion and fund images rebuilt and restarted.
  The change needs no migration. The two environment variables are optional and keep
  the defaults when unset. This record does not deploy it: when the live daemons pick
  it up is the owner's decision.

## Unguarded paths

These reach the database without passing through `DatabaseManager.get_session`, so
the breaker neither counts their failures nor refuses them. Each entry gives what it
does during an outage and the fallback, as read in the installed libraries
(APScheduler 3.11.3, psycopg 3.3.5, psycopg_pool 3.3.1) and in the repository code.
Guarded, for contrast: `DatabaseManager.health_check`, every repository and service
session, `fund.observe` reads (they get their session from the caller), and the fund
tools.

- **Container start: database wait and migrations.**
  - Where: `ingestion/entrypoint.sh`, `fund/entrypoint.sh`. `alembic upgrade head` runs
    on an engine of its own, built from `DATABASE_URL` (`packages/portopt-db/alembic/env.py`).
  - In an outage: the script polls `pg_isready` for up to `DB_WAIT_TIMEOUT` (120 s),
    then prints `Database not ready after 120s; giving up.` and exits 1. If the
    database is reachable but the migration fails, `set -e` exits non-zero.
  - Fallback: compose restarts the container (`restart: unless-stopped`) and the wait
    starts again. No session is ever opened, so there is nothing for the breaker to
    guard.
- **Process start: `init_db()`.**
  - Where: `DatabaseManager.initialize` tests the connection with a direct
    `engine.connect()`. The ingestion `init_db` also runs `_require_migrated_schema`
    (`inspect(engine).has_table`) in `ingestion/app/database.py`. Both the ingestion
    worker and CLI and the fund worker call it at start.
  - In an outage: the connect fails, at once for a refused connection, or after the
    driver's 10 s `connect_timeout` for an unreachable host. Ingestion logs
    `Database initialization failed: ...` and re-raises outside development; the fund
    worker does not catch it. Either way the process exits.
  - Fallback: the container restart above. A lazy `initialize` reached through
    `get_session` is guarded; only the explicit start-up call is not.
- **APScheduler job stores.**
  - Where:
    - `SQLAlchemyJobStore(engine=database_manager.engine)` in `create_scheduler` of
      `ingestion/app/services/jobs/scheduler.py` (table `apscheduler_jobs`);
    - `fund/src/fund/scheduler.py` (table `fund_apscheduler_jobs`).

    The store polls and writes through the engine's pool, not through a session.
  - In an outage, at start:
    - Ingestion's `create_scheduler` calls `jobstore.start(scheduler, "default")`
      before any lookup, which creates the table (`jobs_t.create(engine, True)`).
      Nothing catches it, so the process exits from `create_scheduler` with a
      traceback. The persisted-next-run lookup, and its `could not read persisted
      job` WARNING, are never reached.
    - fund's `scheduler.start()` has the store create its table and write the jobs.
      The worker does not guard that call, and the process exits.
  - In an outage, while running: `_process_jobs` (`apscheduler/schedulers/base.py`)
    catches a failing `get_due_jobs`. It logs one WARNING,
    `Error getting due jobs from job store 'default': ...`, and retries after
    `jobstore_retry_interval`, 10 s by default and not overridden here. Each retry is a
    pool checkout with pre-ping: a refused connect fails at once, and an unreachable
    host waits out the 10 s connect timeout. A cron fire missed
    meanwhile runs once the store answers if it is still within its misfire grace
    (ingestion 3600 s, fund 21600 s), coalesced to one run; past the grace it is
    skipped.
  - Writes outside that guard end the scheduler thread on an error:
    - `update_job`, after a due job was handed to the executor;
    - `remove_job`, for a job with no next fire such as ingestion's one-shot reclaim
      jobs (`remove_job` catches only `JobLookupError`);
    - `get_next_run_time`.

    The orphan reaper's re-dispatch under the `reclaim` strategy also writes to the
    store (`_scheduler.add_job`), but it runs inside the reaper's `try`, so an error
    there is logged as a failed tick instead.
    - Ingestion's `SupervisedScheduler` logs `APScheduler loop crashed; no scheduled
      job will fire`. The worker's 30 s supervision then logs `APScheduler thread died
      — exiting so the container restarts` and exits.
    - fund uses a plain `BackgroundScheduler`, so the thread's traceback goes to
      stderr. The worker's 30 s liveness poll logs `APScheduler thread died; exiting so
      the container restarts`, tries to fail its in-flight job slots (refused by the
      breaker while it is open), and calls `os._exit`.
  - Fallback: the container restart. The operator sees the WARNING every ~10 s while
    the store is unreadable, and a restart when a write fails.
- **fund LangGraph persistence: setup connections.**
  - Where: `setup_langgraph` in `fund/src/fund/audit/persistence.py`. The bootstrap
    advisory lock and `create_schema` each open a `psycopg.connect` with
    `connect_timeout` 10 s. It runs at the start of the fund worker and the cockpit,
    in the `profile`, `run`, `approve` and `reject` CLI commands, and in a manual
    scheduler run.
  - In an outage: `psycopg.OperationalError`, at once for a refused connection or after
    the connect timeout for an unreachable host, with no retry. The worker and the
    cockpit exit at start, and those CLI commands fail with a traceback (see *Fund
    CLI* above).
  - Fallback: the worker restarts with its container. The cockpit and the CLI are
    started again by the operator.
- **fund LangGraph persistence: the pool.**
  - Where: `build_pool`, a `ConnectionPool(min_size=1, max_size=4, open=True,
    check=ConnectionPool.check_connection)` with the library defaults `timeout=30.0`
    and `reconnect_timeout=300.0`. The checkpointer (`PostgresSaver`) and the store
    (`PostgresStore`) use it.
  - Who uses it: a fund round's checkpoint writes; the Ask history (the Store, on the
    cockpit's `_ask_io` thread); the Ask agent's checkpointer thread; the gate reads
    that take the saver (`observe.pending_hitl`, the cockpit's gate text); and the
    MiFID constraint-set cache in `fund/src/fund/audit/mifid_repository.py`, which
    writes and reads the Store (`store.put`, `store.get`) from the round tools and
    from the scheduler's rebalance step (`_drive_rebalance`, run by the sweep and the
    drift monitor).
  - In an outage, for a caller: each call waits up to 30 s for a connection, then
    raises `PoolTimeout("couldn't get a connection after 30.00 sec")`, a
    `psycopg.OperationalError`.
  - In an outage, in the background: the pool tries to reconnect. It logs one WARNING,
    `error connecting in 'pool-N': ...`, per attempt, backing off from 1 s, doubling,
    with 10 % jitter. After 300 s it logs `reconnection attempt in pool ... failed after
    300.0 sec` and gives that slot up; no `reconnect_failed` callback is set. A later
    caller that waits while the pool is below `max_size` starts a new attempt.
  - What softens it: the gate reads, the round, the scheduler's sweep and drift monitor
    (and the MiFID cache inside them), and the cockpit's poll all open a `get_session`
    session first. While the breaker is open, that session is refused
    before the saver is touched, so they fail at once instead of waiting 30 s.
  - What does not: the Ask history, which goes to the Store directly. Each read or save
    waits out the 30 s on the `_ask_io` thread, off the event loop. The cockpit then
    shows `saved Ask history unavailable: ...` and retries the load after 30 s.
- **fund round bookkeeping on its own session.**
  - Where: `_own_session` in `fund/src/fund/agents/graph_failures.py` opens
    `Session(bind=session.get_bind())`. On that session the round spends and hands back
    the first-round gate and records a failed round. It runs only inside a round whose
    main session the breaker admitted.
  - In an outage that begins mid-round: each of these opens a connection straight
    from the engine's pool (pre-ping, 10 s connect timeout), and the breaker does not
    count the failure.
    - Spending the gate raises, and the round fails.
    - Handing it back and recording the failed round catch the error, log it with a
      traceback (`could not hand back the first-round gate of ...`, `could not record
      failed fund run ...`) and give up. The failed run is then not recorded.
  - Fallback: none beyond the log. The round's own session was admitted before the
    outage, so the next round is refused at its first session by the breaker.
- **Unused by the daemons.** `DatabaseManager.create_all_tables` creates tables
  straight on the engine. No ingestion or fund code calls it.
