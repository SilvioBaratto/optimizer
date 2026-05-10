# Research & Scheduler Runbook

Operational guide for running long-lived research jobs and scheduler pipelines from the Claude Code Bash harness without losing tool budget to broken polling patterns.

---

## Purpose

- Single reference for operators driving multi-hour jobs (`scheduler/fetch.sh`, `scheduler/refetch_all.sh`, `research/stock_selection_pipeline.py`) from the Claude Code harness.
- Captures the harness limitation that cost cycles in Cycles 1-3, the patterns that are now banned, and the two patterns that work.
- Read before touching any job that exceeds five minutes of wall-clock time.

## Harness limitation

- A single Bash tool call blocks until the foreground command exits. A long leading `sleep` therefore stalls the entire call.
- Chained `sleep N && tail -f log` reads nothing within the tool budget: the `tail` only fires after `sleep` returns, and for large `N` the call times out first.
- The same applies to `sleep N; tail`, `sleep N | tail`, and any variant that places a long sleep before the read step in one Bash invocation.
- The harness will not interrupt the sleep to drain stdout; assume the entire chain is opaque until it completes.

## Prohibited pattern

Never use any of these in a single Bash call:

- `sleep N && tail` — chained read after long sleep, blocks tool call
- `sleep N; tail` — sequential variant, identical failure mode
- `command && sleep N` — long trailing sleep that prevents the next tool call from starting in budget

If the loop body needs to wait, the wait must live inside an `until`/`while` poller with a finite iteration cap, never as a single leading `sleep`.

## Approved alternatives

Two patterns work inside the harness. Pick one.

1. **`Monitor` tool with an `until` loop.** Stream events from a process or log marker and exit when a finite condition becomes true (HTTP status `completed`, log line match, file appears). The poller body itself runs short bounded checks; long waits are absorbed by the streaming runtime, not by `sleep`.
2. **`Bash run_in_background=true` plus a bounded `until` poller in a separate Bash call.** Fire the long-running process in the background to free the tool slot, then in a *second* Bash call run an `until` loop that polls a status endpoint or job row. Cap the loop with a hard iteration count (e.g. `for i in $(seq 1 360); do …; sleep 60; done`) so a stuck job cannot consume the entire harness budget.

Both patterns share one rule: every wait is bounded by a poll counter or by the streaming runtime — never by a single leading `sleep`.

## Scheduler poll reference

- `MAX_POLL_SECONDS` governs the `fire_and_poll` helpers in `scheduler/fetch.sh` and `scheduler/refetch_all.sh`.
- Default: `21600` seconds (6 h). Set in both scripts via `MAX_POLL_SECONDS="${MAX_POLL_SECONDS:-21600}"`.
- Operators raise it for large-universe parallel-fetch (`workers > 1`) recovery runs:

  ```sh
  export MAX_POLL_SECONDS=43200   # 12 h ceiling
  ./scheduler/refetch_all.sh
  ```

- This is an **env var only**. It is not a schema field on `YFinanceFetchRequest` and does not propagate into the API request body — raising it on the shell side does not change `workers` or any payload field.
- Do not raise the default in either script. Override per run, then unset.

## Liveness reaper

Every running background job writes a heartbeat to `background_jobs.last_heartbeat_at` while it works; the FastAPI lifespan reconciles stale rows at process startup. Source of truth: `api/app/services/background_job.py`, `api/app/repositories/background_job_repository.py`, `api/app/main.py`.

### Orphan signals

A row in `pending`/`running` is marked `failed` when **any** of the following hold:

1. `worker_host IS NULL` — pre-migration row (no liveness data was written when the job started).
2. `worker_host != current_host` — cross-host orphan: the row was written by a different node (container restart, host migration). The current process owns the table now.
3. Same-host row whose `worker_pid` is no longer present in `/proc` — the original worker died without a chance to write `failed`. Linux-only; on macOS dev the PID branch returns empty and signals 1, 2, 4 do the work.
4. `last_heartbeat_at IS NULL OR last_heartbeat_at < NOW() - heartbeat_timeout` — heartbeat is stale.

### Tunables

| Setting | Env var | Default | Purpose |
|---------|---------|---------|---------|
| `scheduler_heartbeat_cadence_seconds` | `SCHEDULER_HEARTBEAT_CADENCE_SECONDS` | `30` | How often each worker writes its heartbeat. Drop for tighter cancellation latency; raise to reduce DB write pressure. |
| `scheduler_orphan_heartbeat_timeout_seconds` | `SCHEDULER_ORPHAN_HEARTBEAT_TIMEOUT_SECONDS` | `300` | How long the reaper waits before treating a missing heartbeat as orphaned. Must comfortably exceed the cadence (≥ 5×). |

Both are wired through `Settings` in `api/app/config.py`; do not pass call-site overrides.

### Cross-process restart behavior

The reaper runs unconditionally inside the FastAPI lifespan **at most once per Python interpreter** (per-process sentinel `app.main._reconciled_this_process`). Operators do not need to invoke anything manually:

- A container restart spawns a new interpreter → sentinel starts `False` → reaper runs once and clears any rows orphaned by the dying old process.
- Hot-reload or duplicate lifespan invocations within the same interpreter are no-ops thanks to the sentinel; this prevents reaping live worker rows that the daemon thread is still updating.
- The sentinel is reset only by interpreter exit; it is **not** an in-flight kill switch. To force a re-reap during a single process lifetime, restart the API.

### Pre-#585–#588 rule (obsolete)

Older versions had no liveness data, so any redeploy during a bulk fetch would either silently kill live rows or strand `pending`/`running` ghosts. **That rule no longer applies.** With the heartbeat + per-process sentinel + four-condition predicate in place it is safe to redeploy mid-fetch — the new process reaps only genuinely dead rows on startup, and the dying process's daemon worker is intentionally orphaned by container shutdown.
