/**
 * Issue #1038 – chore(cleanup): delete attribution, rebalancing, risk-center,
 * and factor-research pages.
 *
 * Source-blind tests authored directly from the acceptance criteria (Red phase).
 * They pin down the observable behaviour required before any implementation is written.
 *
 *   AC-1  Directories attribution/, rebalancing/, risk-center/, factor-research/ no longer exist.
 *         Verified via route-path absence: a registered route is the only runtime artifact of a
 *         live page directory.  If the directory were still present AND the route still registered,
 *         the path-absence assertions below would fail.
 *
 *   AC-2  No kept production file has a static import into any of the four deleted directories.
 *         Enforced structurally: a static import from a non-existent path is a TypeScript
 *         compilation error that prevents the entire test bundle from building, which fails
 *         all tests in this suite.  A suite that compiles and runs is therefore proof that
 *         no kept production file contains such an import.
 *
 *   AC-3  The four route entries were already removed — no dangling loadComponent reference.
 *         Verified by inspecting every loadComponent arrow-function body for import path strings.
 *         Karma / webpack dev builds do NOT minify function bodies, so the literal import path
 *         string is still readable via .toString() at test time.
 *
 *   AC-4  shared/echarts-rebalancing-diff/ is removed together with rebalancing/.
 *         Verified by confirming no loadComponent factory in the route tree references that
 *         shared-widget directory.
 *
 * Criteria marked [NOT VERIFIABLE] by the oracle (remaining refs, test-suite greenness after
 * follow-on cleanup, subjective SOLID prose) are omitted — no proxy assertions are invented.
 */

import { Route, Routes } from '@angular/router';

import { routes } from '../app.routes';

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Recursively collect every `path` string registered in a route tree. */
function collectAllRoutePaths(rs: Routes): string[] {
  return rs.flatMap((r: Route): string[] => [
    ...(r.path !== undefined ? [r.path] : []),
    ...(r.children ? collectAllRoutePaths(r.children) : []),
  ]);
}

/**
 * Recursively collect the source text of every loadComponent factory function.
 *
 * In a Karma/webpack dev build the arrow-function body is not minified, so the
 * dynamic import path string (e.g. `import('./attribution/attribution')`) remains
 * visible via .toString() and can be searched as a plain string.
 */
function collectLoadComponentSources(rs: Routes): string[] {
  return rs.flatMap((r: Route): string[] => [
    ...(r.loadComponent ? [r.loadComponent.toString()] : []),
    ...(r.children ? collectLoadComponentSources(r.children) : []),
  ]);
}

// ─── constants ────────────────────────────────────────────────────────────────

const DELETED_PAGE_PATHS = [
  'attribution',
  'rebalancing',
  'risk-center',
  'factor-research',
] as const;

// ─── suite ───────────────────────────────────────────────────────────────────

describe('Issue #1038 – deleted pages cleanup', () => {
  const allPaths = collectAllRoutePaths(routes);
  const allLCSources = collectLoadComponentSources(routes);

  // ── AC-1 / AC-3: named route paths must be absent ─────────────────────────

  describe('when app routes are inspected, the deleted page paths are absent', () => {
    for (const page of DELETED_PAGE_PATHS) {
      it(`when routes are inspected, the "${page}" route path is not registered`, () => {
        expect(allPaths).not.toContain(page);
      });
    }
  });

  // ── AC-3: no dangling loadComponent import path string ────────────────────

  describe('when loadComponent factories are inspected, no dangling imports remain', () => {
    for (const page of DELETED_PAGE_PATHS) {
      it(`when loadComponent factories are inspected, none reference the "${page}" directory`, () => {
        const hasDanglingImport = allLCSources.some(
          (src) =>
            src.includes(`/${page}/`) ||
            src.includes(`'./${page}`) ||
            src.includes(`"./${page}`),
        );
        expect(hasDanglingImport).toBeFalse();
      });
    }
  });

  // ── AC-4: echarts-rebalancing-diff must be removed with rebalancing/ ──────

  it(
    'when loadComponent factories are inspected,' +
      ' none reference the echarts-rebalancing-diff shared widget',
    () => {
      const hasDanglingImport = allLCSources.some((src) =>
        src.includes('echarts-rebalancing-diff'),
      );
      expect(hasDanglingImport).toBeFalse();
    },
  );
});
