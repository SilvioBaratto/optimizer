/**
 * Source-blind contract tests for issue #1042.
 *
 * Derived solely from the acceptance criteria:
 *   - `path: 'optimize'` must be the real loadComponent route for
 *     OptimizationStudioComponent with title 'Optimize'.
 *   - `path: 'optimization-studio'` must redirect to 'optimize'.
 *
 * These tests MUST FAIL on the current routing (which has the paths swapped)
 * and PASS once the implementation flips them.
 */
import type { Route } from '@angular/router';

import { routes } from '../app.routes';

describe('app.routes – /optimize flip (#1042)', () => {
  /**
   * The routes tree has a single layout shell at index 0; all page routes are
   * its children.  Assumption: this nesting is stable across the refactor.
   */
  function childRoutes(): Route[] {
    return routes[0]?.children ?? [];
  }

  function findChild(path: string): Route | undefined {
    return childRoutes().find((r) => r.path === path);
  }

  // ---------------------------------------------------------------------------
  // path: 'optimize' — must be the REAL route (loadComponent, not redirectTo)
  // ---------------------------------------------------------------------------

  describe("when path is 'optimize'", () => {
    it('then the route exists in the routing table', () => {
      expect(findChild('optimize')).toBeDefined();
    });

    it('then it has a loadComponent factory (is a real page, not a redirect)', () => {
      const route = findChild('optimize');
      expect(route!.loadComponent).toBeDefined();
    });

    it('then it does NOT have a redirectTo property', () => {
      const route = findChild('optimize');
      expect(route!.redirectTo).toBeUndefined();
    });

    it('then its title is exactly "Optimize"', () => {
      const route = findChild('optimize');
      expect(route!.title).toBe('Optimize');
    });
  });

  // ---------------------------------------------------------------------------
  // path: 'optimization-studio' — must REDIRECT to 'optimize'
  // ---------------------------------------------------------------------------

  describe("when path is 'optimization-studio'", () => {
    it('then the route exists in the routing table', () => {
      expect(findChild('optimization-studio')).toBeDefined();
    });

    it("then it redirects to 'optimize'", () => {
      const route = findChild('optimization-studio');
      expect(route!.redirectTo).toBe('optimize');
    });

    it('then it does NOT have a loadComponent factory (is purely a redirect)', () => {
      const route = findChild('optimization-studio');
      expect(route!.loadComponent).toBeUndefined();
    });
  });

  // ---------------------------------------------------------------------------
  // Structural invariants — held for any valid routing table shape
  // ---------------------------------------------------------------------------

  it('then "optimize" has loadComponent and "optimization-studio" does not (no duplicated real route)', () => {
    expect(findChild('optimize')?.loadComponent).toBeDefined();
    expect(findChild('optimization-studio')?.loadComponent).toBeUndefined();
  });

  it("then 'optimization-studio' is only a redirect — it has redirectTo and no loadComponent", () => {
    const studioRoute = findChild('optimization-studio');
    expect(studioRoute?.redirectTo).toBeDefined();
    expect(studioRoute?.loadComponent).toBeUndefined();
  });
});
