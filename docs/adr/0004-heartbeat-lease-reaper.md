# ADR-0004: Reap orphaned jobs by heartbeat lease

- Status: Accepted
- Date: 2026-10-24

## Context

The ingestion daemon and the fund worker claim a job slot in the database. A worker that
dies leaves its row `running` forever, so something must reap it. `docs/ARCHITECTURE.md`
§5.3 records why a fixed timeout cannot do it: synchronous steps such as the yfinance fetch
and the reference seed run longer than a flat 300 s, so a flat timeout reaps work that is
slow but alive.

Related commits: `4b942926` (2026-10-24) stamps job heartbeats on the same clock the
reaper compares, and `125c7f54` (2026-11-03, #1123) stops the heartbeat reporting a
finished job as reaped.

## Decision

The reaper is a heartbeat lease. `reconcile_orphans` fails any active row (`pending` or
`running`) whose `last_heartbeat_at` is NULL or older than the lease TTL
(`SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS`, 300 s). `worker_host` and `worker_pid` are
observability only; there is no host-scoped reap predicate. A synchronous step must wrap
its work in `scheduler._heartbeat()`, because only the heartbeat thread stamps the row.
Exactly one daemon runs per database. The fund worker's job slot mirrors this design.

## Alternatives considered

- **A flat timeout on the claim.** Rejected in `docs/ARCHITECTURE.md` §5.3: it reaps slow
  but live work.
- **A host-scoped reap predicate.** An earlier design used it; it was removed, and
  `CLAUDE.md` calls any description of it stale.
- **Reclaim and re-queue instead of failing the row.** `docs/ARCHITECTURE.md` §5.3 marks
  it adopted but gated on idempotent writes (§5.4). The implemented reaper fails the row,
  so re-queueing is not in effect.

## Consequences

- A synchronous step that outlives the TTL without stamping a heartbeat is falsely
  reaped and flips to `failed` mid-run, so new work is added as a step wrapped in
  `_heartbeat()`, never as a CLI-only branch.
- Two daemons on one database would reap each other's rows, so the deployment keeps one.
- A reaped job is recorded as `failed`; the reaper does not re-queue it (see the
  alternatives above).
