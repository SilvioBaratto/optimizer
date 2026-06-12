import type { Route, Routes } from '@angular/router';

import { routes } from './app.routes';

/**
 * Source-blind example tests for issue #969, authored from the acceptance
 * criteria only — no implementation source was read.
 *
 * Scope decision (recorded per the test-designer mandate):
 * Of the criteria the oracle marked runtime-verifiable, all but one are
 * component-rendering behaviours ("the results panel renders all sub-charts",
 * "the IC chart renders", "an inline message is shown", "no endpoint names in
 * UI copy"). Those cannot be pinned down without knowing component classes,
 * selectors, and service shapes — i.e. without reading implementation source —
 * so they are deliberately skipped here rather than asserted via brittle
 * proxies. The remaining criteria are oracle-marked NOT VERIFIABLE.
 *
 * The single criterion that is both verifiable AND expressible as a black-box
 * assertion on a public contract is:
 *
 *   "Navigating to /dashboard redirects to / rather than showing the 404 page."
 *
 * The Angular `routes` array is the public input to the Router, so the routing
 * behaviour it encodes is observable without loading any component.
 *
 * Ambiguity resolution: "redirects to /" is asserted as a redirect whose target
 * is the application root. Angular spells the root route as the empty path `''`
 * (idiomatically with `pathMatch: 'full'`), and `'/'` is the equivalent
 * absolute spelling; both are accepted as "root". This is the simplest
 * behaviour consistent with the criterion text.
 */

function flattenRoutes(routeTable: Routes): Route[] {
  const flat: Route[] = [];
  for (const route of routeTable) {
    flat.push(route);
    if (route.children?.length) {
      flat.push(...flattenRoutes(route.children));
    }
  }
  return flat;
}

function isRedirectToRoot(redirectTo: Route['redirectTo']): boolean {
  // Angular 21 types redirectTo as `string | RedirectFunction | undefined`;
  // only a literal-string redirect to the empty/root path counts as "root".
  return redirectTo === '' || redirectTo === '/';
}

describe('app routes — /dashboard redirect (issue #969)', () => {
  it('when the dashboard path is configured, it redirects to the application root', () => {
    const dashboardRoutes = flattenRoutes(routes).filter((r) => r.path === 'dashboard');

    // A dashboard path must exist...
    expect(dashboardRoutes.length).toBeGreaterThan(0);

    // ...and it must be a redirect to root, never a leaf that would otherwise
    // fall through to the wildcard 404.
    const redirectsToRoot = dashboardRoutes.some((r) => isRedirectToRoot(r.redirectTo));
    expect(redirectsToRoot).toBe(true);
  });

  it('when an unknown path is requested, a wildcard route serves the 404 page', () => {
    // The criterion contrasts /dashboard ("rather than showing the 404 page")
    // against the generic not-found behaviour: a wildcard route must exist so
    // that genuinely-unknown paths — unlike /dashboard — reach the 404 page.
    const hasWildcard = flattenRoutes(routes).some((r) => r.path === '**');
    expect(hasWildcard).toBe(true);
  });

  it('when the dashboard path redirects, it is not itself the wildcard 404 route', () => {
    // Guards against a degenerate "fix" where /dashboard is matched only by the
    // wildcard: the redirect must be an explicit, dedicated dashboard entry.
    const dashboardRoute = flattenRoutes(routes).find(
      (r) => r.path === 'dashboard' && isRedirectToRoot(r.redirectTo),
    );
    expect(dashboardRoute).toBeDefined();
    expect(dashboardRoute?.path).not.toBe('**');
  });
});
