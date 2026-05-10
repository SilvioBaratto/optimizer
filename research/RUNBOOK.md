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
