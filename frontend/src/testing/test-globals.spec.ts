import { installResizeObserverStub } from './index';

type GlobalScope = { ResizeObserver?: unknown };
type ResizeObserverCtor = new (cb: () => void) => {
  observe: () => void;
  unobserve: () => void;
  disconnect: () => void;
};

describe('installResizeObserverStub', () => {
  it('when ResizeObserver is absent, a usable no-op stub is installed', () => {
    const scope = globalThis as GlobalScope;
    const original = scope.ResizeObserver;
    delete scope.ResizeObserver;

    installResizeObserverStub();

    expect(scope.ResizeObserver).toBeDefined();
    const ro = new (scope.ResizeObserver as ResizeObserverCtor)(() => {});
    expect(() => {
      ro.observe();
      ro.unobserve();
      ro.disconnect();
    }).not.toThrow();

    scope.ResizeObserver = original;
  });

  it('when ResizeObserver already exists, the existing one is kept (idempotent)', () => {
    const scope = globalThis as GlobalScope;
    const sentinel = scope.ResizeObserver ?? class {};
    scope.ResizeObserver = sentinel;

    installResizeObserverStub();

    expect(scope.ResizeObserver).toBe(sentinel);
  });
});
