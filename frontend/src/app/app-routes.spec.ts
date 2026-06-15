/**
 * Source-blind route-configuration spec (issue #1012 / requirements §13).
 *
 * Criteria tested:
 *  [T2] Navigating to /dashboard redirects to / rather than showing the 404 page.
 *       → The route config must contain a `{ path: 'dashboard', redirectTo: '',
 *         pathMatch: 'full' }` entry so that Angular's router treats /dashboard
 *         as an alias for the root route instead of a 404.
 *
 * Design note (source-blind):
 *  This spec tests the ROUTE CONFIGURATION, not the rendered page.
 *  Checking the route array directly is more deterministic than driving the
 *  router with navigate() because it avoids lazy-loaded module resolution,
 *  guard execution, and component rendering — all of which would require
 *  additional test infrastructure with no benefit to pinning this criterion.
 */
import { TestBed } from '@angular/core/testing';
import { Router, type Route, type Routes } from '@angular/router';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';

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

describe('App route configuration — /dashboard redirect (issue #1012)', () => {
  // ── 1. Static route-array assertions ─────────────────────────────────────────
  // These tests read the exported `routes` array and assert the redirect entry
  // is present and correctly shaped. They do NOT need Angular's DI.
  // Note: the /dashboard entry is nested inside the layout wrapper's children.

  it('when the routes array is inspected, a /dashboard entry exists', () => {
    const entry = flattenRoutes(routes).find((r) => r.path === 'dashboard');
    expect(entry)
      .withContext('routes must contain a { path: "dashboard" } entry')
      .toBeDefined();
  });

  it('when the /dashboard route entry is found, its redirectTo is an empty string (root)', () => {
    const entry = flattenRoutes(routes).find((r) => r.path === 'dashboard');
    expect(entry?.redirectTo)
      .withContext('redirectTo must be "" (root path) so the router lands on /, not a 404')
      .toBe('');
  });

  it('when the /dashboard route entry is found, its pathMatch is "full"', () => {
    const entry = flattenRoutes(routes).find((r) => r.path === 'dashboard');
    expect(entry?.pathMatch)
      .withContext('pathMatch must be "full" to avoid partial-match ambiguity')
      .toBe('full');
  });

  it('when the /dashboard route entry is found, it has no component property (pure redirect)', () => {
    const entry = flattenRoutes(routes).find((r) => r.path === 'dashboard');
    expect((entry as { component?: unknown } | undefined)?.component)
      .withContext('a redirect route must not declare a component — it would never render')
      .toBeUndefined();
  });

  it('when the /dashboard route entry is found, it has no loadComponent property (pure redirect)', () => {
    const entry = flattenRoutes(routes).find((r) => r.path === 'dashboard');
    expect((entry as { loadComponent?: unknown } | undefined)?.loadComponent)
      .withContext('a redirect route must not declare loadComponent — it would never render')
      .toBeUndefined();
  });

  // ── 2. Router-level integration: navigate(["/dashboard"]) → URL becomes "/" ──
  // These tests use the actual router to confirm that Angular applies the
  // redirect at runtime, not just that the config array contains the entry.

  describe('with router provided', () => {
    let router: Router;

    beforeEach(() => {
      TestBed.configureTestingModule({
        providers: [provideZonelessChangeDetection(), provideRouter(routes)],
      });
      router = TestBed.inject(Router);
    });

    it('when navigating to /dashboard, the router leaves /dashboard (the redirect fired)', async () => {
      await router.navigate(['/dashboard']);
      // The router's `redirectTo: ''` fires so we never stay at /dashboard.
      // Guards on the target route may cause further redirects (e.g. to
      // /portfolio-builder), so we only assert that /dashboard is not the
      // final URL — not the specific landing page.
      expect(router.url).not.toBe('/dashboard');
    });

    it('when navigating to /dashboard, the router does not emit a 404 / wildcard URL', async () => {
      await router.navigate(['/dashboard']);
      expect(router.url).not.toContain('404');
      expect(router.url).not.toContain('not-found');
    });
  });
});
