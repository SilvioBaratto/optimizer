/**
 * Source-blind example tests for issue #999 — authored from acceptance criteria ONLY.
 *
 * Written before (and without reading) the implementation, against the acceptance
 * criteria and `.code-generator/requirements.md`. These tests must not be weakened
 * to match whatever the implementer produces.
 *
 * ---------------------------------------------------------------------------
 * COVERAGE NOTE — why this file contains a single test.
 *
 * The volatile context supplies the cycle's GLOBAL acceptance criteria (the same
 * list fed to every issue in this cycle), classified by an oracle report. Most are
 * marked NOT VERIFIABLE and are skipped per the oracle. Of the criteria the oracle
 * marks verifiable, only one can be pinned down *source-blind without guessing
 * internals*:
 *
 *   [T2] "Navigating to /dashboard redirects to / rather than showing the 404 page."
 *
 * requirements.md scope 13 states the route contract verbatim:
 *     { path: 'dashboard', redirectTo: '', pathMatch: 'full' }
 * so it is assertable against the public, exported `routes` table without
 * bootstrapping the component tree. The search below recurses into `children`, so
 * it holds whether the redirect sits at the top level or nested under a
 * layout-shell route.
 *
 * Why nothing here targets #999's own subject (request-shape parity specs for the
 * dashboard, risk, and attribution services):
 *   - The matching criterion — "Contract-parity specs assert request and response
 *     shapes for every service method" — is marked NOT VERIFIABLE by the oracle,
 *     so it is skipped.
 *   - The concrete service methods, URLs, query params, and body shapes to assert
 *     are not named in the supplied criteria; authoring tests against guessed
 *     endpoints would invent scope (forbidden) and read as a brittle proxy. Those
 *     contract specs are themselves the implementation deliverable, owned by the
 *     source-aware implementation phase.
 *
 * Other verifiable-but-skipped criteria (would require internals a source-blind
 * author must not guess): factor-404 inline message, per-page error states,
 * optimization/factor sub-chart rendering, "(a)(b)(c) specs exist", and
 * "no endpoint names in UI copy" (a negative property over every template).
 *
 * No property-based test is emitted: no supplied criterion implies a round-trip /
 * idempotence / never-raises / ordering invariant over a callable pure function.
 *
 * This guard overlaps the same-named guards authored for #996–#998 — an artifact
 * of the identical global criteria list being supplied to each issue. The overlap
 * is harmless: all assert the same public route contract.
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

describe('Issue #999 — /dashboard route redirect', () => {
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
