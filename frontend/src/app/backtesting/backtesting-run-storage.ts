/**
 * localStorage persistence for the last completed backtest run (issue #996, 13a).
 *
 * Pure functions over an injectable `Storage` (defaulting to `localStorage`),
 * mirroring the save/load helpers in `optimization-studio/optimization.service.ts`.
 * Every access is wrapped in try/catch so a private-mode / quota failure degrades
 * to a no-op instead of crashing the page.
 */
export const BACKTEST_RUN_STORAGE_KEY = 'optimizer.lastBacktestRun';

export interface StoredBacktestRun {
  /** `BacktestRun.id` — re-fetched via GET /backtest/runs/{runId} on re-mount. */
  runId: string;
  /** Portfolio selected when the run completed; null when none was selected. */
  portfolioId: string | null;
}

function resolveStorage(storage?: Storage | null): Storage | null {
  if (storage !== undefined) return storage;
  return typeof localStorage !== 'undefined' ? localStorage : null;
}

/** Persist the last run. Storage failures are swallowed. */
export function saveBacktestRun(run: StoredBacktestRun, storage?: Storage | null): void {
  const store = resolveStorage(storage);
  if (!store) return;
  try {
    store.setItem(BACKTEST_RUN_STORAGE_KEY, JSON.stringify(run));
  } catch {
    /* storage unavailable (quota / private mode) — ignore */
  }
}

/** Read the last run, or null when absent, unreadable, or malformed. */
export function loadBacktestRun(storage?: Storage | null): StoredBacktestRun | null {
  const store = resolveStorage(storage);
  if (!store) return null;
  const raw = readRaw(store);
  return raw ? parseStoredRun(raw) : null;
}

/** Remove the persisted run. Storage failures are swallowed. */
export function clearBacktestRun(storage?: Storage | null): void {
  const store = resolveStorage(storage);
  if (!store) return;
  try {
    store.removeItem(BACKTEST_RUN_STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

function readRaw(store: Storage): string | null {
  try {
    return store.getItem(BACKTEST_RUN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function parseStoredRun(raw: string): StoredBacktestRun | null {
  try {
    const parsed = JSON.parse(raw) as Partial<StoredBacktestRun>;
    if (typeof parsed?.runId !== 'string' || parsed.runId.length === 0) return null;
    return { runId: parsed.runId, portfolioId: parsed.portfolioId ?? null };
  } catch {
    return null;
  }
}
