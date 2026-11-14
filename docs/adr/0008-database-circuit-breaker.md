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
`DatabaseManager.get_session` built a session, its first statement tried to connect
(failing at once for a refused connection, or after the driver's 10 s
`connect_timeout` for an unreachable host), and the error crossed the session with a
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

**Fallbacks.** Read in the code, for the default configuration. Each process has its
own breaker, which starts closed and opens only after 5 connectivity failures recorded
through that process's `get_session`. Until then the process fails on the driver's
error, with a traceback, as it did before this record.

- *Ingestion daemon.*
  - Once the breaker is open, a step stops when its job-slot claim, a status write or
    its own code meets the refusal. It writes one ERROR line, without a traceback,
    increments `jobs_db_unavailable_total{domain}`, and runs again at its next
    scheduled time.
  - A failed-status write that the breaker refuses is logged at that moment and
    increments `jobs_failed_status_refused_total{domain}`, whatever stopped the step.
  - Such a row stays `running` until the orphan reaper fails it. The lease is 300 s
    and the reaper runs every 300 s, so that happens 300 to 600 s after the last
    heartbeat, at a reaper tick when the database answers. Under the default `fail`
    strategy its error then reads `orphaned — heartbeat stale, reaped by periodic
    reaper`, not `db_unavailable`.
  - When the failed-status write of a step stopped by the refusal does land, the error
    starts `db_unavailable:`.
  - A refused reaper tick, progress write or heartbeat logs one line without a
    traceback.
  - Known limit: `_universe_is_empty`, the first-run bootstrap check at the start of
    every daily pipeline, logs a WARNING with a traceback when its session is refused.
    It then skips the bootstrap.
  - With metrics on (the default), the gauge `portopt_db_breaker_state` reports 0
    (closed), 1 (open) and 2 (half-open), read at each scrape.
- *Fund cockpit (`fund-tui`).*
  - The gate is re-read at each poll tick (every 2 s). While the breaker is not
    closed, the bindings that write are disabled: approve, reject, killswitch,
    re-arm, save mandate, guardrail override, new portfolio, profile, run, barge-in,
    edit and steer. They show greyed out and do not run.
  - Ask's decision verbs read the same gate. They are refused before any confirm or
    overlay is offered. They are refused again when an answer given after the breaker
    opened would dispatch: a confirm, the fund re-arm rationale, or a steer directive.
  - The status bar and the Ask log show the refusal as `describe_error` renders it.
  - Once the breaker is open, reads are refused at once.
    - The Performance and Daily brief views read off the event loop and keep their
      last paint when a read fails.
    - What the other views show then is not verified.
    - The attention feed logs the refusal once per outage, in one line.
  - Once the timeout has passed, the next poll read is the probe. When its connect
    succeeds, the writes come back at the next poll tick and the status bar says the
    database is reachable again.
- *Fund CLI (`fund`), a known limit.*
  - Each command is a fresh process whose breaker starts closed, so an outage raises
    the driver's error, not `DatabaseUnavailableError`. The breaker would refuse only
    after 5 failures within one process.
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
  - The rebalance sweep, the drift monitor and the scheduled NAV mark
    (`fund/src/fund/scheduler.py`) open their first session outside their `try`. The
    refusal, or the driver error before the breaker opens, reaches APScheduler, which
    logs `Job "..." raised an exception` with a traceback.
  - The orphan reaper catches it and logs `orphan reaper: tick failed (non-fatal)`
    with a traceback.
  - Each of these ticks logs one traceback when it opens a session during an outage.
    The scheduled NAV mark and its interval re-check (`run_scheduled_nav_mark`) open
    no session before the day's due time.
  - If the database dies mid-tick:
    - the per-portfolio handlers log one traceback per portfolio;
    - the step's heartbeat thread logs `heartbeat[...]: pulse failed; retrying next
      tick` with a traceback every 30 s.
  - The breaker never refuses a tick's failed-status write:
    - The tick logs `... aborted` with a traceback, then writes `failed` with
      `job_repo.mark_done` on its own session. The breaker admitted that session
      before the outage and is consulted only when a session is opened.
    - On a dead database that write fails with a database error, which escapes the
      `except` to APScheduler (`Job "..." raised an exception`).
    - No counter records it, and the row stays `running` until the fund reaper fails
      it after its 300 s lease.
    - Only `_fail_in_flight_slots` goes through `get_session`. It runs when the worker
      exits on a dead scheduler thread or a timed-out drain, and logs one line if it
      fails.
  - The fund worker serves no metrics, so its breaker state is visible only in the log.

## Alternatives considered

- **Do nothing.** Keep relying on `connect_timeout`, `pool_pre_ping` and the orphan
  reaper. Rejected: during an outage every call keeps paying a connect attempt (the
  full 10 s connect timeout when the host is unreachable) and logs a traceback, the
  cockpit offers writes that cannot land, and standard 3.3 makes a breaker mandatory.
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

- Once the ingestion daemon's breaker is open, a step's job-slot claim is refused at
  once, so the step ends in milliseconds with one log line and a counter. Operators see
  the state on `portopt_db_breaker_state`. Once the cockpit's breaker is open, it greys
  its writes out instead of failing after a confirm.
- The breaker gates the entry to a session and sees only the outcomes that cross
  `get_session`. How the yfinance sweep behaves when the database dies mid-run depends
  on its path:
  - *Parallel, the default* (`YFINANCE_FETCH_WORKERS=4`).
    - Each ticker opens its own session in `_fetch_one_ticker` before
      `fetch_and_store` sends anything to Yahoo, and `_worker` turns an exception
      into that ticker's error.
    - Once the breaker is open, every ticker that starts after that is refused before
      it fetches, and the sweep goes through the rest of the list with one error per
      ticker.
    - Whether the breaker opens at all during a parallel sweep is not verified. A
      session body that absorbs its own error and exits cleanly resets the count (see
      the next point).
  - *Serial* (`YFINANCE_FETCH_WORKERS=1`). The sweep runs in one session opened
    before its loop, with an `except Exception` per ticker. The breaker is never asked
    again, and the sweep keeps fetching from Yahoo until the step ends.

  Two follow-ups would cut it short: re-raise `DatabaseUnavailableError` and
  connection errors past the per-ticker handler, or set the step's cancel event when a
  progress write is refused. Either is a change to the sweep, outside this record.
- While the breaker is closed, a body that swallows its own connection error and exits
  cleanly records a success and resets the count, which delays the open. The probe is
  immune because its connect, not its exit, decides it.
- Each process has its own breaker: the ingestion worker, the fund worker, the cockpit
  and each CLI run. They do not share state.
- A manual re-run of an ingestion step whose row was left `running` is refused as
  already running until the orphan reaper fails the row, 300 to 600 s after its last
  heartbeat.
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
the breaker neither counts their failures nor refuses them. Each entry was read in the
installed libraries (APScheduler 3.11.3, psycopg 3.3.5, psycopg_pool 3.3.1) and the
repository code, for the default configuration.

Guarded, for contrast: `DatabaseManager.health_check`, every repository and service
session, `fund.observe` reads (they get their session from the caller), and the fund
tools.

- **Container start: database wait and migrations.**
  - Where: `ingestion/entrypoint.sh`, `fund/entrypoint.sh`. `alembic upgrade head` runs
    on an engine of its own, built from `DATABASE_URL` (`packages/portopt-db/alembic/env.py`).
  - In an outage: the script polls `pg_isready` for up to `DB_WAIT_TIMEOUT` (120 s),
    then prints `Database not ready after 120s; giving up.` and exits 1. If the
    database is reachable but the migration fails, `set -e` exits non-zero.
  - Fallback: compose restarts the `scheduler` and `fund` services
    (`restart: unless-stopped`), and the wait starts again.
- **Process start: `init_db()`.**
  - Where: `DatabaseManager.initialize` tests the connection with a direct
    `engine.connect()`. The ingestion `init_db` also runs `_require_migrated_schema`
    (`inspect(engine).has_table`) in `ingestion/app/database.py`. The ingestion worker
    and CLI and the fund worker call it at start.
  - In an outage, the connect fails at once for a refused connection, or after the
    driver's 10 s `connect_timeout` for an unreachable host.
    - The fund worker does not catch it, and exits.
    - The ingestion worker runs with `ENVIRONMENT` at its code default,
      `development`, which compose does not override. There `init_db` logs
      `Database initialization failed: ...` and
      `Continuing in development mode despite database errors`, then returns. The
      start fails a moment later at `create_scheduler`, which raises
      `RuntimeError: create_scheduler() called before database_manager.initialize()`
      because no engine exists.
    - With `ENVIRONMENT=production`, `init_db` re-raises instead. Either way the
      process exits.
    - What the ingestion CLI does after a failed `init_db` is not verified.
  - Fallback: the container restart above. A lazy `initialize` reached through
    `get_session` is guarded; only the explicit start-up call is not.
- **APScheduler job stores.**
  - Where:
    - `SQLAlchemyJobStore(engine=database_manager.engine)` in `create_scheduler` of
      `ingestion/app/services/jobs/scheduler.py` (table `apscheduler_jobs`);
    - `fund/src/fund/scheduler.py` (table `fund_apscheduler_jobs`).

    The store reads and writes through the engine's pool, not through a session.
  - At start, the store is first touched by ingestion's
    `jobstore.start(scheduler, "default")` in `create_scheduler`, before any lookup,
    and by fund's `scheduler.start()`. Each creates the table, and fund's also writes
    the jobs. Neither call is guarded.
    - In an outage both workers have already exited at the start-up step above:
      ingestion at the engine check that opens `create_scheduler`, before
      `jobstore.start`; fund at `init_db`.
    - These calls fail only if the database drops after that step, and the process
      then exits.
  - While running: `_process_jobs` (`apscheduler/schedulers/base.py`) catches a
    failing `get_due_jobs`.
    - It logs one WARNING, `Error getting due jobs from job store 'default': ...`,
      and retries after `jobstore_retry_interval`, 10 s by default and not overridden
      here.
    - Each retry is a pool checkout with pre-ping: a refused connect fails at once,
      and an unreachable host waits out the 10 s connect timeout.
    - A cron fire missed meanwhile runs once the store answers if it is still within
      its misfire grace (ingestion 3600 s, fund 21600 s), coalesced to one run. Past
      the grace it is skipped.
  - Calls outside that guard end the scheduler thread on an error:
    - the writes `update_job` (after a due job was handed to the executor) and
      `remove_job` (for a job with no next fire; it catches only `JobLookupError`);
    - the read `get_next_run_time`.

    What happens then:
    - Ingestion's `SupervisedScheduler` logs `APScheduler loop crashed; no scheduled
      job will fire`. The worker's 30 s supervision then logs `APScheduler thread died
      — exiting so the container restarts` and exits.
    - fund uses a plain `BackgroundScheduler`, so the thread's traceback goes to
      stderr. The worker's 30 s liveness poll logs `APScheduler thread died; exiting so
      the container restarts`, tries to fail its in-flight job slots through
      `get_session`, and calls `os._exit`.
  - The reaper's re-dispatch (`_scheduler.add_job`) also writes to the store. It runs
    only under `SCHEDULER_ORPHAN_STRATEGY=reclaim`, not the default `fail`, and inside
    the reaper's `try`.
  - Fallback: the container restart. The operator sees the WARNING every ~10 s while
    the store is unreadable, and a restart when a guarded-off call fails.
- **fund LangGraph persistence: setup connections.**
  - Where: `setup_langgraph` in `fund/src/fund/audit/persistence.py`. The bootstrap
    advisory lock and `create_schema` each open a `psycopg.connect` with
    `connect_timeout` 10 s. It runs:
    - at the start of the fund worker and the cockpit;
    - in the `profile`, `run`, `approve` and `reject` CLI commands;
    - in `python -m fund.scheduler rebalance-sweep` and `drift-monitor`. The
      `nav-mark` step builds no runtime.
  - In an outage: `psycopg.OperationalError`, at once for a refused connection or after
    the connect timeout for an unreachable host, with no retry.
    - The cockpit exits at start with a traceback.
    - The fund worker has already exited at `init_db`, which runs first.
    - Those CLI commands fail with a traceback (see *Fund CLI* above).
  - Fallback: the worker restarts with its container. The cockpit and the CLI are
    started again by the operator.
- **fund LangGraph persistence: the pool.**
  - Where: `build_pool`, a `ConnectionPool(min_size=1, max_size=4, open=True,
    check=ConnectionPool.check_connection)` with the library defaults `timeout=30.0`
    and `reconnect_timeout=300.0`. The checkpointer (`PostgresSaver`) and the store
    (`PostgresStore`) use it.
  - Who uses it:
    - a fund round's checkpoint writes;
    - the Ask history (the Store, on the cockpit's `_ask_io` thread);
    - the Ask agent's checkpointer thread;
    - the gate reads that take the saver (`observe.pending_hitl`, the cockpit's gate
      text);
    - the MiFID constraint-set cache in `fund/src/fund/audit/mifid_repository.py`,
      which writes and reads the Store (`store.put`, `store.get`). It runs from the
      round tools and from the scheduler's rebalance step (`_drive_rebalance`, run by
      the sweep and the drift monitor).
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
    session first. Once the breaker is open, that session is refused before the saver
    is touched, so they fail at once instead of waiting 30 s.
  - What does not:
    - The Ask history goes to the Store directly. A load waits out the 30 s on the
      `_ask_io` thread, off the event loop. The cockpit then shows
      `saved Ask history unavailable: ...` and retries the load after 30 s.
    - A save that fails is logged and dropped.
- **fund round bookkeeping on its own session.**
  - Where: `_own_session` in `fund/src/fund/agents/graph_failures.py` opens
    `Session(bind=session.get_bind())`. On that session the round spends and hands back
    the first-round gate and records a failed round. It runs only inside a round whose
    main session the breaker admitted.
  - In an outage that begins mid-round, each of these opens a connection straight from
    the engine's pool (pre-ping, 10 s connect timeout), and the breaker does not count
    the failure.
    - Spending the gate raises, and the round fails.
    - Handing it back and recording the failed round catch the error, log it with a
      traceback (`could not hand back the first-round gate of ...`, `could not record
      failed fund run ...`) and give up. The failed run is then not recorded.
  - Fallback: none beyond the log.
    - Once the breaker has opened, the next round is refused at its first session. When
      that happens depends on the process.
    - A CLI round's next round is a new process with a closed breaker, which fails at
      `setup_langgraph` with `psycopg.OperationalError`.
    - In the fund worker a tick that fails fast records about one failure, so the next
      ticks are admitted and fail on the driver's error until five have built up.
    - The cockpit polls every 2 s with several reads per tick, so its breaker is the
      one most likely to be open.
- **Unused by the daemons.** `DatabaseManager.create_all_tables` creates tables
  straight on the engine. No ingestion or fund code calls it.
