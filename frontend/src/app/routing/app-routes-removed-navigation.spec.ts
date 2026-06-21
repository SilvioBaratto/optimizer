/**
 * Source-blind tests for issue #1037 — criterion 4.
 * Authored from acceptance criteria only; no implementation source was read.
 *
 * Criterion 4: Navigating to a removed path (e.g. /risk-center) resolves to
 *   the not-found page, i.e. the wildcard "**" route.
 *
 * Uses provideLocationMocks() from @angular/common/testing so tests run
 * without a real browser Location object.
 *
 * Import assumption: routes are exported as `routes` from `../app.routes`.
 */
import { TestBed } from '@angular/core/testing';
import { Router, RouterStateSnapshot, provideRouter } from '@angular/router';
import { provideLocationMocks } from '@angular/common/testing';
import { routes } from '../app.routes';

// ---------------------------------------------------------------------------
// Helper — walk down to the deepest matched route config
// ---------------------------------------------------------------------------

function deepestMatchedPath(snapshot: RouterStateSnapshot): string | null {
  let route = snapshot.root;
  while (route.firstChild) { route = route.firstChild; }
  return route.routeConfig?.path ?? null;
}

// ---------------------------------------------------------------------------
// Criterion 4 — removed paths resolve to the not-found wildcard route
// ---------------------------------------------------------------------------

const REMOVED_PATHS = [
  '/factor-research',
  '/attribution',
  '/rebalancing',
  '/risk-center',
  '/ai-control-room',
];

describe('Routing — removed paths resolve to not-found (criterion 4)', () => {
  let router: Router;

  beforeEach(async () => {
    TestBed.configureTestingModule({
      providers: [
        provideRouter(routes),
        provideLocationMocks(),
      ],
    });
    router = TestBed.inject(Router);
    await router.initialNavigation();
  });

  REMOVED_PATHS.forEach(removedPath => {
    it(`when navigating to "${removedPath}", navigation succeeds by matching the wildcard route`, async () => {
      const navigated = await router.navigate([removedPath]);
      expect(navigated)
        .withContext(`Navigation to "${removedPath}" should succeed via the "**" wildcard route`)
        .toBeTrue();
    });

    it(`when navigating to "${removedPath}", the deepest matched route config path is "**"`, async () => {
      await router.navigate([removedPath]);
      const matched = deepestMatchedPath(router.routerState.snapshot);
      expect(matched)
        .withContext(
          `Expected "${removedPath}" to be handled by the "**" wildcard route, ` +
          `but deepest matched route path was "${matched}"`
        )
        .toBe('**');
    });
  });
});
