/**
 * Source-blind tests for issue #1037 — criteria 1 & 2.
 * Authored from acceptance criteria only; no implementation source was read.
 *
 * Criterion 1: app.routes.ts contains no loadComponent importing from
 *   factor-research/, attribution/, rebalancing/, risk-center/, ai-control-room/,
 *   or pipeline-stepper/pipeline-stepper.
 *
 * Criterion 2: All kept routes preserved:
 *   '', portfolio-builder, optimize, optimization-studio, backtesting, macro-intelligence,
 *   settings, portfolio/:name, instrument/:id, dashboard→'', optimization-studio→optimize, **
 *
 * Import assumption: routes are exported as `routes` from `../app.routes`.
 * Adjust the import if the export name differs.
 */
import { Route, Routes } from '@angular/router';
import { routes } from '../app.routes';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function flattenRoutes(rs: Routes, acc: Route[] = []): Route[] {
  for (const r of rs) {
    acc.push(r);
    if (r.children) flattenRoutes(r.children, acc);
  }
  return acc;
}

const ALL_ROUTES: Route[] = flattenRoutes(routes);
const ALL_PATHS: Array<string | undefined> = ALL_ROUTES.map(r => r.path);

// ---------------------------------------------------------------------------
// Criterion 1 — no doomed route entries remain
// ---------------------------------------------------------------------------

const DOOMED_PATHS = [
  'factor-research',
  'attribution',
  'rebalancing',
  'risk-center',
  'ai-control-room',
];

describe('app.routes.ts — criterion 1: doomed route paths are absent', () => {
  DOOMED_PATHS.forEach(doomed => {
    it(`when routes are inspected, "${doomed}" is not registered as a route path`, () => {
      expect(ALL_PATHS)
        .withContext(`Route path "${doomed}" should have been deleted but is still present`)
        .not.toContain(doomed);
    });
  });

  it('when routes are inspected, no loadComponent imports from pipeline-stepper/pipeline-stepper', () => {
    // Arrow-function source is preserved by the TypeScript compiler in non-minified builds.
    // If this assertion is unreliable in your toolchain, remove it and rely on the
    // portfolio-builder-legacy path absence test instead.
    const offending = ALL_ROUTES.filter(r => {
      if (!r.loadComponent) { return false; }
      return r.loadComponent.toString().includes('pipeline-stepper/pipeline-stepper');
    });
    expect(offending.length)
      .withContext('Found route(s) with loadComponent importing from pipeline-stepper/pipeline-stepper')
      .toBe(0);
  });

  it('when routes are inspected, "portfolio-builder-legacy" is not registered as a route path', () => {
    expect(ALL_PATHS)
      .withContext('The legacy builder duplicate route should have been deleted')
      .not.toContain('portfolio-builder-legacy');
  });
});

// ---------------------------------------------------------------------------
// Criterion 2 — all kept routes are present
// ---------------------------------------------------------------------------

const KEPT_PATHS = [
  '',
  'portfolio-builder',
  'optimize',
  'optimization-studio',
  'backtesting',
  'macro-intelligence',
  'settings',
  'portfolio/:name',
  'instrument/:id',
  '**',
];

describe('app.routes.ts — criterion 2: kept routes are preserved', () => {
  KEPT_PATHS.forEach(path => {
    const label = path === '' ? '(root path "")' : `"${path}"`;
    it(`when routes are inspected, the route ${label} is present`, () => {
      expect(ALL_PATHS)
        .withContext(`Kept route ${label} is missing from app.routes.ts`)
        .toContain(path);
    });
  });

  it('when routes are inspected, a redirect from "dashboard" to "" is present', () => {
    const redirect = ALL_ROUTES.find(r => r.path === 'dashboard' && r.redirectTo !== undefined);
    expect(redirect)
      .withContext('Expected { path: "dashboard", redirectTo: "" } but no such entry was found')
      .toBeDefined();
    expect(redirect!.redirectTo).toBe('');
  });

  it('when routes are inspected, a redirect from "optimization-studio" to "optimize" is present', () => {
    const redirect = ALL_ROUTES.find(r => r.path === 'optimization-studio' && r.redirectTo !== undefined);
    expect(redirect)
      .withContext('Expected { path: "optimization-studio", redirectTo: "optimize" } but no such entry was found')
      .toBeDefined();
    expect(redirect!.redirectTo).toBe('optimize');
  });
});
