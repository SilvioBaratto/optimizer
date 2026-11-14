# ADR-0008: Guard database sessions with a circuit breaker

- Status: Accepted
- Date: 2026-11-14

## Context

Standard 3.3 requires every database call to go through a circuit breaker with a
half-open state, configurable thresholds, monitoring and a fallback. Before this
record nothing stood between a caller and a dead database. Every
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

- a driver error without a SQLSTATE (SQLite reports none);
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

A zero, negative, non-numeric or non-finite value fails when the settings load. The
env templates are edited by the repository owner only. Until he lists the two
variables there, they are documented here and in `CLAUDE.md`, and the ingestion
template guard accepts exactly these two names as documented outside the template.

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
  - The gauge `portopt_db_breaker_state` reports 0 (closed), 1 (open) and 2
    (half-open), read at each scrape.
- *Fund cockpit (`fund-tui`).*
  - While the breaker of `fund.database`'s manager is not closed, the bindings that
    write are disabled: approve, reject, killswitch, re-arm, save mandate, guardrail
    override, new portfolio, profile, run, barge-in, edit and steer. They show greyed
    out and do not run.
  - Ask's decision verbs read the same gate. They are refused before any confirm is
    offered, and again when a confirm answered after the breaker opened would dispatch.
  - The status bar and the Ask log show the refusal as `describe_error` renders it.
  - Reads are refused at once by the same breaker, and each view keeps its last paint,
    which gives a read-only view of the last known state. The attention feed logs the
    refusal once per outage, in one line.
  - Once the timeout has passed, the next poll read is the probe. When its connect
    succeeds, the writes come back and the status bar says the database is reachable
    again.
- *Fund CLI.* `DatabaseUnavailableError` is a `RuntimeError`, so the commands'
  existing `except RuntimeError` handlers print `error: database unavailable: ...` and
  exit 1.
- *Fund worker daemon (`fund-worker`), a known limit.*
  - Its scheduled ticks (`fund/src/fund/scheduler.py`: the rebalance sweep, the drift
    monitor, the NAV mark and the orphan reaper) catch the refusal with
    `logger.exception`, so each tick, and each portfolio inside a tick, logs a
    traceback while the breaker is open.
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
- New database access must go through `DatabaseManager.get_session`. Code that builds
  its own engine bypasses the breaker.
- Deploying this needs the running ingestion and fund images rebuilt and restarted.
  The change needs no migration. The two environment variables are optional and keep
  the defaults when unset. This record does not deploy it: when the live daemons pick
  it up is the owner's decision.
