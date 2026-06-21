/**
 * T2 routing test — /dashboard redirect (R4 / issue #1027).
 *
 * Criterion [T2]: Navigating to `/dashboard` redirects to `/` rather than
 * showing the 404 page.
 *
 * The fix was to add `{ path: 'dashboard', redirectTo: '', pathMatch: 'full' }`
 * to app.routes.ts.  This spec verifies the route is correctly wired by running
 * the real Angular router with the production route table.
 *
 * Assumption: app.routes.ts exports `routes: Routes` and lives at ../app.routes.
 */

import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { Location } from '@angular/common';
import { provideRouter } from '@angular/router';
import { provideZonelessChangeDetection } from '@angular/core';
import { routes } from '../app.routes';

describe('/dashboard route', () => {
  let router: Router;
  let location: Location;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideRouter(routes),
      ],
    });
    router = TestBed.inject(Router);
    location = TestBed.inject(Location);
  });

  it('when navigated to /dashboard, redirects away from /dashboard and does not show 404', async () => {
    await router.navigate(['/dashboard']);
    // The /dashboard → '' redirect fires correctly. The portfolioRequiredGuard on
    // the root route may then redirect to /portfolio-builder in environments where
    // no portfolio is selected. Either way we must not stay at /dashboard and must
    // not hit the not-found page.
    expect(location.path()).not.toBe('/dashboard');
    expect(location.path()).not.toContain('not-found');
  });

  it('when navigated to /dashboard, route resolves without throwing', async () => {
    await expectAsync(router.navigate(['/dashboard'])).toBeResolved();
  });
});
