/**
 * Source-blind contract test (issue #995 cycle).
 *
 * Global acceptance criterion under test [oracle tier: T2]:
 *   "Navigating to `/dashboard` redirects to `/` rather than showing the 404 page."
 *
 * Authored from the acceptance criteria only. No implementation source was read.
 * The assertion is black-box on the application's PUBLIC routing table — the
 * artifact the criterion governs — so it needs no feature-component internals,
 * no DI graph, and no network or wall-clock dependency. It is deterministic and
 * self-contained.
 *
 * Structure-agnostic by design: a source-blind author does not know whether the
 * `/dashboard` redirect is declared at the top level or nested under a layout
 * shell route's `children`, so the test flattens the whole route tree before
 * locating the rule. This keeps the assertion faithful to the criterion
 * (navigation behaviour) without guessing the file's structure.
 *
 * Red-phase expectation: fails if no `dashboard` path exists anywhere in the tree
 * (it would fall through to the wildcard 404 route) or if it is configured as
 * anything other than a full-match redirect to the application root.
 *
 * Assumption (recorded per instructions): "redirects to `/`" is read as resolving
 * to the application root, whose Angular route path is the empty string `''`;
 * `pathMatch: 'full'` is required because an empty `redirectTo` with prefix
 * matching is illegal and would not pin the rule to the exact `/dashboard` path.
 */
import { Route, Routes } from '@angular/router';

import { routes } from './app.routes';

/** Depth-first flatten so a nested redirect rule is found wherever it lives. */
function flattenRoutes(routeList: Routes): Route[] {
  return routeList.flatMap((route) => [
    route,
    ...(route.children ? flattenRoutes(route.children) : []),
  ]);
}

describe('app routing — /dashboard redirect (issue #995 cycle)', () => {
  it('when navigating to /dashboard, the application root route is returned (redirect to /)', () => {
    const dashboardRoute = flattenRoutes(routes).find(
      (route) => route.path === 'dashboard',
    );

    expect(dashboardRoute)
      .withContext('a route for the `dashboard` path must exist so it is not handled by the 404/wildcard route')
      .toBeDefined();
    expect(dashboardRoute?.redirectTo)
      .withContext('`/dashboard` must redirect to the application root `/` (the empty path)')
      .toBe('');
    expect(dashboardRoute?.pathMatch)
      .withContext('the redirect must be a full-path match so only the exact `/dashboard` path redirects')
      .toBe('full');
  });
});
