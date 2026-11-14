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

`portopt_db.resilience.DbCircuitBreaker` guards each `DatabaseManager`: one breaker
per manager instance, no module global. `get_session` asks it before it builds
anything, and every exit from the session records exactly one outcome.

- **Closed.** Sessions pass. Consecutive connectivity failures are counted. A session
  that reaches the database resets the count.
- **Open.** After `failure_threshold` consecutive connectivity failures, sessions are
  refused at once with `DatabaseUnavailableError` (a `RuntimeError`). No engine or
  session is created. The error message is one line without host, credentials or SQL.
- **Half-open.** Once `reset_timeout_s` has passed, the next caller is admitted as the
  single probe and every other caller is refused until it ends. Success closes the
  breaker. A connectivity failure reopens it, and the full timeout restarts from that
  failure.

Every state change happens under one `threading.Lock`. The clock is injected, so the
tests never sleep.

**What counts.** `DisconnectionError` (the pool's pre-ping), any `DBAPIError` that
SQLAlchemy marked `connection_invalidated`, a SQLSTATE in class 08 or equal to 57P01,
57P02 or 57P03 (the server shutting down, crashing or still starting), and an
`OperationalError` without a SQLSTATE raised outside a statement, which is how a
refused or timed-out connect at pool checkout surfaces. psycopg2's `pgcode` and
psycopg 3's `sqlstate` are both read.

**What does not count.** Any other SQLSTATE: statement and lock timeouts, integrity,
programming and data errors. The server answered, so the database is reachable and
the outcome is recorded as a success: it resets the count and settles a probe. A
statement error without a SQLSTATE (SQLite) and a pool-exhaustion `TimeoutError` do
not count either.

**Inconclusive outcomes.** An exception from the caller's own code inside the `with`
block, `GeneratorExit`, `KeyboardInterrupt` or a closed manager says nothing about the
database. The count is left alone, and a held probe slot is released so the next
caller probes. The breaker can therefore never stay half-open with its slot taken.

**Late outcomes.** Outcomes are not tied to the session that produced them. While the
breaker is open they are ignored, so a session admitted before it opened can neither
close it nor extend it. While it is half-open the first outcome to land settles the
probe, even when it comes from a session admitted before the breaker opened: it is
equally fresh evidence about the same database.

**Defaults, on by default.** `DbConfig.breaker_failure_threshold = 5` and
`DbConfig.breaker_reset_timeout_s = 30.0`, both validated as finite and above zero.
The breaker is on by default because it only counts connectivity failures, so a
healthy database never sees it. A false open costs at most 30 s of refused sessions,
and then one probe decides. `pool_pre_ping` already absorbs a single dropped connection
by reconnecting at checkout. Five consecutive connect failures across the daemon's
threads therefore mean the server is gone, not a blip. 30 s matches the heartbeat
cadence and the health-check cache. It is short next to the 300 s orphan lease, so a
recovered database is noticed long before the reaper acts, and long enough that a
restarting server is not hammered.

**Fallbacks.**

- *Ingestion daemon.* A step that meets the open breaker stops. It writes one ERROR
  line, without a traceback, and increments `jobs_db_unavailable_total{domain}`, and the
  next scheduled run retries it. A row the breaker stopped mid-step is marked `failed`
  with an error starting `db_unavailable:` when that write gets through. While the
  breaker is open the write is refused before it reaches the database. The row then
  stays `running` and the heartbeat-lease orphan reaper (ADR-0004) fails it once the
  database answers again, so the failure is not lost. The reaper tick logs one line
  while the breaker is open. The gauge `portopt_db_breaker_state` reports 0 (closed),
  1 (open) and 2 (half-open), read at each scrape.
- *Fund cockpit (`fund-tui`).* While the breaker of `fund.database`'s manager is not
  closed, the bindings that write are disabled: approve, reject, killswitch, re-arm,
  save mandate, guardrail override, new portfolio, profile, run, barge-in, edit and
  steer. They show greyed out and do not run. The status bar shows the refusal as
  `describe_error` renders it. Reads are refused at once by the same breaker, and each
  view keeps its last paint, which gives a read-only view of the last known state. Once
  the timeout has passed, the next poll read is the probe. When it succeeds, the writes
  come back and the status bar says the database is reachable again. A confirm already
  open when the breaker opened ends in a refused write, which the action worker reports
  like any other failure.
- *Fund CLI.* `DatabaseUnavailableError` is a `RuntimeError`, so the commands'
  existing `except RuntimeError` handlers print `error: database unavailable: ...` and
  exit 1.

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
- The breaker sees only outcomes that cross `get_session`. A body that catches its own
  errors, such as the per-ticker loop of the yfinance sweep inside one long session, is
  neither counted nor cut short. That sweep keeps fetching from Yahoo after the
  database dies mid-run, until the step ends.
- Each process has its own breaker: the ingestion worker, the fund worker, the cockpit
  and each CLI run. They do not share state.
- A stopped row stays `running` until the reaper runs after the database is back, and
  its error then reads as an orphan rather than `db_unavailable`. A manual re-run of
  that step within the 300 s lease is refused as already running.
- The thresholds are `DbConfig` fields. Neither daemon reads them from the environment
  yet, so changing them needs a code change.
- New database access must go through `DatabaseManager.get_session`. Code that builds
  its own engine bypasses the breaker.
- Deploying this needs the running ingestion and fund images rebuilt and restarted.
  The change needs no migration and no new environment variable. This record does not
  deploy it: when the live daemons pick it up is the owner's decision.
