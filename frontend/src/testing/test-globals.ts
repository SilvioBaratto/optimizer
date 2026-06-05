// JSDOM/Karma has no ResizeObserver, but BreakpointService and every ECharts
// wrapper instantiate one. Install an idempotent no-op stub so those specs can
// mount the units under test without a real observer.

interface ResizeObserverLike {
  observe(): void;
  unobserve(): void;
  disconnect(): void;
}

class NoopResizeObserver implements ResizeObserverLike {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

/** Register a no-op `ResizeObserver` on the global scope when absent. Idempotent. */
export function installResizeObserverStub(): void {
  const scope = globalThis as { ResizeObserver?: unknown };
  if (scope.ResizeObserver) return;
  scope.ResizeObserver = NoopResizeObserver;
}
