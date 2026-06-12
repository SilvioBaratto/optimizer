import {
  BACKTEST_RUN_STORAGE_KEY,
  clearBacktestRun,
  loadBacktestRun,
  saveBacktestRun,
  type StoredBacktestRun,
} from './backtesting-run-storage';

/** Minimal in-memory Storage for deterministic, isolated round-trip tests. */
function memoryStorage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => Array.from(map.keys())[i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  };
}

/** Storage whose every access throws — models private mode / quota failure. */
function throwingStorage(): Storage {
  const boom = (): never => {
    throw new Error('storage unavailable');
  };
  return {
    get length() {
      return 0;
    },
    clear: boom,
    getItem: boom,
    key: () => null,
    removeItem: boom,
    setItem: boom,
  };
}

describe('backtesting-run-storage (issue #996, 13a)', () => {
  it('when a run is saved then loaded, the same runId and portfolioId are returned', () => {
    const store = memoryStorage();
    const run: StoredBacktestRun = { runId: 'run-1', portfolioId: 'pf-1' };
    saveBacktestRun(run, store);
    expect(loadBacktestRun(store)).toEqual(run);
  });

  it('when nothing has been saved, load returns null', () => {
    expect(loadBacktestRun(memoryStorage())).toBeNull();
  });

  it('when the stored value is corrupt JSON, load returns null', () => {
    const store = memoryStorage();
    store.setItem(BACKTEST_RUN_STORAGE_KEY, '{not valid json');
    expect(loadBacktestRun(store)).toBeNull();
  });

  it('when the stored value lacks a runId, load returns null', () => {
    const store = memoryStorage();
    store.setItem(BACKTEST_RUN_STORAGE_KEY, JSON.stringify({ portfolioId: 'pf-1' }));
    expect(loadBacktestRun(store)).toBeNull();
  });

  it('when a saved run is cleared, load returns null', () => {
    const store = memoryStorage();
    saveBacktestRun({ runId: 'run-1', portfolioId: null }, store);
    clearBacktestRun(store);
    expect(loadBacktestRun(store)).toBeNull();
  });

  it('when setItem throws, save does not throw', () => {
    expect(() =>
      saveBacktestRun({ runId: 'run-1', portfolioId: 'pf-1' }, throwingStorage()),
    ).not.toThrow();
  });

  it('when getItem throws, load returns null instead of throwing', () => {
    expect(loadBacktestRun(throwingStorage())).toBeNull();
  });

  it('when removeItem throws, clear does not throw', () => {
    expect(() => clearBacktestRun(throwingStorage())).not.toThrow();
  });
});
