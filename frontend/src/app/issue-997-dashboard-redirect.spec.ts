/**
 * Source-blind example tests for issue #997 — authored from acceptance criteria ONLY.
 *
 * Written before (and without reading) the implementation, against the acceptance
 * criteria and `.code-generator/requirements.md`. These tests must not be weakened
 * to match whatever the implementer produces.
 *
 * ---------------------------------------------------------------------------
 * COVERAGE NOTE — why this file contains a single test.
 *
 * The oracle report classifies the cycle's acceptance criteria. Most are marked
 * NOT VERIFIABLE and are skipped per the oracle. Of the criteria the oracle marks
 * verifiable, only one can be pinned down *source-blind without guessing internals*:
 *
 *   [T2] "Navigating to /dashboard redirects to / rather than showing the 404 page."
 *
 * requirements.md scope 13 states the route contract verbatim:
 *     { path: 'dashboard', redirectTo: '', pathMatch: 'full' }
 * so it is assertable against the public, exported `routes` table without
 * bootstrapping the component tree (which a source-blind author cannot wire up
 * correctly). The search below recurses into `children`, so it holds whether the
 * redirect sits at the top level or nested under a layout-shell route.
 *
 * Criteria central to #997 that were DELIBERATELY SKIPPED (not proxied):
 *
 *   - "Factor attribution 404 -> inline 'Factor scores not available'" [T3]:
 *       The oracle marks this verifiable at the integration tier, but a faithful
 *       black-box test needs the panel/page class, the input or trigger method,
 *       the init/portfolio-selection flow, the chained snapshot->brinson->factor
 *       HTTP sequence, and each response shape — all internals a source-blind
 *       author must not guess. Asserting them on guesses would be a brittle proxy
 *       (more likely to fail on setup than to pin the criterion), which the brief
 *       forbids. Deferred to the implementation phase's own (source-aware) specs.
 *   - "No API endpoint or parameter names in user-facing UI copy" [T3]:
 *       a negative property over every template; cannot be enumerated without
 *       searching implementation source.
 *   - "Attribution form hint shows the actual portfolio name, not '(none)'":
 *       marked NOT VERIFIABLE by the oracle — skipped.
 *   - Error-state, sub-chart-render, compute round-trip, and "(a)(b)(c) specs
 *     exist" criteria [T3]: require component/service internals and a test harness
 *     not inferable from the criteria text.
 *
 * No property-based test is emitted: no criterion implies a round-trip /
 * idempotence / never-raises / ordering invariant over a callable pure function.
 * ---------------------------------------------------------------------------
 */
import { Route, Routes } from '@angular/router';

import { routes } from './app.routes';

/**
 * Resolve the route entry whose `path` exactly equals the supplied segment,
 * searching the route tree recursively. Feature routes (including any
 * `dashboard` redirect) may live under a layout-shell route's `children`, so a
 * flat `.find` could miss them. Reads only the public `path`/`children` fields.
 */
function routeForPath(table: Routes, path: string): Route | undefined {
  for (const entry of table) {
    if (entry.path === path) return entry;
    const nested = entry.children && routeForPath(entry.children, path);
    if (nested) return nested;
  }
  return undefined;
}

describe('Issue #997 — /dashboard route redirect', () => {
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
