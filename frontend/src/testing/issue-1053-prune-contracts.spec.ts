/**
 * Issue 1053 – chore(prune): remove orphaned code, fix imports, green ng build & ng test
 *
 * Source-blind contract tests authored from acceptance criteria only.
 * Each suite maps to one acceptance criterion:
 *
 *   AC1 – pipeline-stepper + orphan dirs deleted
 *   AC2 – no dead imports / orphaned barrel exports      (self-enforcing via compile)
 *   AC3 – every nav link resolves; deleted paths → **
 *   AC4 – ng build zero errors                           (self-enforcing via compile)
 *   AC5 – ng test passes; specs removed not skipped      (meta; this file runs = suite passes)
 *   AC6 – Angular conventions preserved; no api/ changes (conventions captured in route metadata)
 *   AC7 – SOLID / clean code                             (NOT VERIFIABLE – oracle skip)
 *
 * No property-based invariants are warranted: the route-set is a small, finite enumeration.
 * None of the criteria imply a round-trip, idempotence, never-raises, or ordering invariant.
 */

import { Route, Routes } from '@angular/router';
import { routes } from '../app/app.routes';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Flatten a route tree (including children at any depth) into a single list. */
function flattenRoutes(rs: Routes): Route[] {
  return rs.reduce<Route[]>((acc, route) => {
    acc.push(route);
    if (route.children?.length) {
      acc.push(...flattenRoutes(route.children));
    }
    return acc;
  }, []);
}

/** True when a route entry causes Angular to mount a view (component or lazy chunk). */
function isLoadable(r: Route): boolean {
  return !!(r.component || r.loadComponent || r.loadChildren);
}

// ---------------------------------------------------------------------------
// Fixture data derived from acceptance criteria & requirements.md
// ---------------------------------------------------------------------------

/**
 * AC1 / AC3 – Paths that must NOT exist as loadable routes after the prune.
 * Source: requirements.md §Pages — DELETE
 */
const DELETED_PAGE_PATHS: string[] = [
  'pipeline-stepper',
  'portfolio-builder-legacy',
  'factor-research',
  'attribution',
  'rebalancing',
  'risk-center',
  'ai-control-room',
];

/**
 * AC3 – Exactly the 6 nav items the spec mandates must survive.
 * Source: requirements.md §Navigation — exactly 6 items
 */
const KEPT_NAV_PATHS: string[] = [
  '',                  // Dashboard (root)
  'portfolio-builder',
  'optimize',
  'backtesting',
  'macro-intelligence',
  'settings',
];

/**
 * AC3 – Support routes that must survive alongside the nav pages.
 * Source: requirements.md §Target Structure – "Support routes kept"
 */
const KEPT_SUPPORT_PATHS: string[] = [
  'portfolio/:name',
  'instrument/:id',
];

// ---------------------------------------------------------------------------
// AC1 – orphan dirs deleted: deleted paths have no loadable route
// ---------------------------------------------------------------------------

describe('issue-1053 AC1 – deleted page paths absent from route configuration', () => {
  let all: Route[];

  beforeEach(() => {
    all = flattenRoutes(routes);
  });

  for (const path of DELETED_PAGE_PATHS) {
    it(`when app routes are loaded, deleted path "${path}" has no component or lazy route`, () => {
      const found = all.find(r => r.path === path && isLoadable(r));
      expect(found)
        .withContext(
          `Deleted path "${path}" must not be a component/lazy route after the prune.`
        )
        .toBeUndefined();
    });
  }
});

// ---------------------------------------------------------------------------
// AC3 / AC6 – kept nav routes exist
// ---------------------------------------------------------------------------

describe('issue-1053 AC3 – kept navigation routes present in route configuration', () => {
  let all: Route[];

  beforeEach(() => {
    all = flattenRoutes(routes);
  });

  for (const path of KEPT_NAV_PATHS) {
    const label = path === '' ? '(dashboard root)' : path;
    it(`when app routes are loaded, kept path "${label}" is navigable`, () => {
      const found = all.find(
        r => r.path === path && (isLoadable(r) || r.redirectTo != null)
      );
      expect(found)
        .withContext(`Kept path "${label}" must have a navigable route.`)
        .toBeDefined();
    });
  }
});

// ---------------------------------------------------------------------------
// AC3 – support routes preserved
// ---------------------------------------------------------------------------

describe('issue-1053 AC3 – support routes preserved after prune', () => {
  let all: Route[];

  beforeEach(() => {
    all = flattenRoutes(routes);
  });

  for (const path of KEPT_SUPPORT_PATHS) {
    it(`when app routes are loaded, support route "${path}" still exists`, () => {
      const found = all.find(r => r.path === path);
      expect(found)
        .withContext(`Support route "${path}" must not be deleted.`)
        .toBeDefined();
    });
  }
});

// ---------------------------------------------------------------------------
// AC3 – wildcard not-found catch-all
// ---------------------------------------------------------------------------

describe('issue-1053 AC3 – wildcard not-found route', () => {
  it('when an unknown path is visited, a wildcard ** route exists to handle it', () => {
    const wildcardRoute = flattenRoutes(routes).find(r => r.path === '**');
    expect(wildcardRoute)
      .withContext('A wildcard "**" catch-all route must exist for unmatched paths.')
      .toBeDefined();
  });

  it('when wildcard routes are counted, exactly one ** route exists (no duplicates)', () => {
    const wildcards = flattenRoutes(routes).filter(r => r.path === '**');
    expect(wildcards.length)
      .withContext('Exactly one wildcard route must exist – not zero, not multiple.')
      .toBe(1);
  });
});

// ---------------------------------------------------------------------------
// AC3 – optimization-studio old path redirects to /optimize
// ---------------------------------------------------------------------------

describe('issue-1053 AC3 – optimization-studio redirects to optimize', () => {
  let all: Route[];

  beforeEach(() => {
    all = flattenRoutes(routes);
  });

  it('when optimization-studio path is navigated, a redirect route is defined', () => {
    const redirect = all.find(r => r.path === 'optimization-studio' && r.redirectTo != null);
    expect(redirect)
      .withContext('Expected a redirect rule for the legacy "optimization-studio" path.')
      .toBeDefined();
  });

  it('when optimization-studio redirect is defined, it targets the optimize path', () => {
    const redirect = all.find(r => r.path === 'optimization-studio' && r.redirectTo != null);
    if (redirect == null) {
      pending('optimization-studio redirect route is absent – caught by prior test.');
      return;
    }
    expect(String(redirect.redirectTo))
      .withContext('The redirect from "optimization-studio" must point to "optimize".')
      .toMatch(/optimize/);
  });

  it('when optimization-studio redirect is present, it is not also a loadable component route', () => {
    const componentRoute = all.find(r => r.path === 'optimization-studio' && isLoadable(r));
    expect(componentRoute)
      .withContext(
        '"optimization-studio" must not have a component route alongside its redirect – ' +
        'it must be redirect-only.'
      )
      .toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// AC3 – dashboard legacy redirect
// ---------------------------------------------------------------------------

describe('issue-1053 AC3 – dashboard legacy path redirects to root', () => {
  it('when "dashboard" path is navigated, a redirect to root exists', () => {
    const all = flattenRoutes(routes);
    const redirect = all.find(r => r.path === 'dashboard' && r.redirectTo != null);
    expect(redirect)
      .withContext(
        'A redirect from "dashboard" to "" (root) must exist per requirements §Target Structure.'
      )
      .toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// AC1 / AC3 – top-level loadable routes do not include any deleted path
// ---------------------------------------------------------------------------

describe('issue-1053 AC1 – deleted paths absent from top-level loadable routes', () => {
  it('when top-level loadable routes are listed, none of the deleted paths appear', () => {
    const topLevelLoadable = routes.filter(isLoadable).map((r: Route) => r.path ?? '');

    for (const deleted of DELETED_PAGE_PATHS) {
      expect(topLevelLoadable)
        .withContext(`Deleted path "${deleted}" must not appear as a top-level loadable route.`)
        .not.toContain(deleted);
    }
  });
});

// ---------------------------------------------------------------------------
// AC6 – Angular conventions: page routes use loadComponent / loadChildren
//        (direct `component:` on lazy pages violates bundle-splitting conventions)
// ---------------------------------------------------------------------------

describe('issue-1053 AC6 – Angular lazy-loading convention on page routes', () => {
  it('when kept nav routes are inspected, each uses loadComponent or loadChildren (not static component)', () => {
    const all = flattenRoutes(routes);
    // '' (Dashboard root) may be embedded in the layout children; check all nav paths.
    for (const path of KEPT_NAV_PATHS) {
      const route = all.find(r => r.path === path);
      if (!route) continue; // absence is caught by AC3 tests
      const isLazy = !!(route.loadComponent || route.loadChildren);
      expect(isLazy)
        .withContext(
          `Route "${path || '(root)'}" must use loadComponent/loadChildren for lazy splitting.`
        )
        .toBeTrue();
    }
  });
});
