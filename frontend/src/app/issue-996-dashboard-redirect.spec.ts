/**
 * Source-blind example tests for issue #996 — authored from acceptance criteria ONLY.
 *
 * These tests were written before (and without reading) the implementation, against
 * the acceptance criteria and `.code-generator/requirements.md`. They must not be
 * weakened to match whatever the implementer produces.
 *
 * ---------------------------------------------------------------------------
 * COVERAGE NOTE — why this file contains a single test.
 *
 * The oracle report classified the cycle's acceptance criteria. Most are marked
 * NOT VERIFIABLE (no concrete runtime check inferable) and are skipped per the
 * oracle, as instructed. Of the criteria the oracle marked verifiable, only one
 * can be pinned down *source-blind without guessing internals*:
 *
 *   [T2] "Navigating to /dashboard redirects to / rather than showing the 404 page."
 *
 * This criterion is fully specified by the spec itself — requirements.md scope 13
 * states the route contract verbatim:
 *     { path: 'dashboard', redirectTo: '', pathMatch: 'full' }
 * so the redirect can be asserted against the public, exported `routes` table
 * without bootstrapping the full component tree (which a source-blind author
 * cannot wire up correctly, since it would require reading component/service
 * internals, providers, and selectors).
 *
 * The other verifiable criteria were deliberately SKIPPED rather than proxied,
 * because a faithful black-box test would require knowledge a source-blind author
 * must not have:
 *   - "Factor attribution 404 -> inline 'Factor scores not available'" [T3]:
 *       needs the panel's input contract to drive the 404 state; the spec does
 *       not define that input, so any driver would be a guessed internal.
 *   - "No API endpoint/parameter names in user-facing copy" [T3]:
 *       a negative property over every template; cannot be enumerated without
 *       searching implementation source.
 *   - "Every page/panel shows a visible error state on API error" [T3],
 *     "Optimization-studio / Factor-research sub-charts render" [T3]:
 *       require component classes, selectors, service signatures and ECharts
 *       wiring that may not be inferred from the criteria text.
 *   - "Every wired page has specs covering (a)(b)(c)" [T3]:
 *       a meta-assertion about test-file existence, not a runtime behaviour.
 *
 * No criterion implies a round-trip / idempotence / never-raises / ordering
 * invariant over a callable pure function, so no property-based test is emitted
 * (forcing one would be a junk property, which the brief forbids).
 * ---------------------------------------------------------------------------
 */
import { Route, Routes } from '@angular/router';

import { routes } from './app.routes';

/**
 * Resolve the route entry whose `path` exactly equals the supplied segment,
 * searching the route tree recursively. Feature routes (including the
 * `dashboard` redirect) live under the layout shell's `children`, not at the
 * top level, so a flat `.find` would miss them.
 */
function routeForPath(table: Routes, path: string): Route | undefined {
  for (const entry of table) {
    if (entry.path === path) return entry;
    const nested = entry.children && routeForPath(entry.children, path);
    if (nested) return nested;
  }
  return undefined;
}

describe('Issue #996 — /dashboard route redirect', () => {
  it('when navigating to /dashboard, the root path is returned instead of the 404 page', () => {
    // Derived solely from the global acceptance criterion
    //   "Navigating to /dashboard redirects to / rather than showing the 404 page"
    // and requirements.md scope 13's stated route contract. A bare /dashboard
    // path with no redirect would fall through to the wildcard 404 route; the
    // observable contract is that the router resolves it to the root path ('').
    const dashboard = routeForPath(routes, 'dashboard');

    expect(dashboard)
      .withContext('a route entry for "dashboard" must exist so it never falls through to the 404 page')
      .toBeDefined();
    expect(dashboard?.redirectTo)
      .withContext('"dashboard" must redirect to the root path ("") rather than render a component')
      .toBe('');
  });
});
