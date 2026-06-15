/**
 * Acceptance criterion [T2]:
 * Navigating to /dashboard redirects to / rather than showing the 404 page.
 *
 * The redirect rule `{ path: 'dashboard', redirectTo: '', pathMatch: 'full' }`
 * must exist in the application routes.  RouterTestingHarness drives the
 * router without a real browser and lets us assert the final URL.
 */
import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';
import { RouterTestingHarness } from '@angular/router/testing';
import { NO_ERRORS_SCHEMA, provideZonelessChangeDetection } from '@angular/core';
import { provideNoopAnimations } from '@angular/platform-browser/animations';

import { routes } from '../app.routes';
import { ICON_PROVIDER } from '../icons';

describe('Dashboard route redirect (issue #1016)', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideRouter(routes),
        provideNoopAnimations(),
        ICON_PROVIDER,
      ],
      schemas: [NO_ERRORS_SCHEMA],
    }).compileComponents();
  });

  it('when navigating to /dashboard, the router URL is no longer /dashboard (redirect fired)', async () => {
    await RouterTestingHarness.create('/dashboard');
    const router = TestBed.inject(Router);
    // The route { path: 'dashboard', redirectTo: '' } redirects away from /dashboard.
    // portfolioRequiredGuard may redirect further (e.g. to /portfolio-builder).
    // The key acceptance criterion: /dashboard is not treated as a 404.
    expect(router.url).not.toBe('/dashboard');
    expect(router.url).not.toContain('not-found');
  });

  it('when navigating to /dashboard, the resolved page does not show 404 text', async () => {
    const harness = await RouterTestingHarness.create('/dashboard');
    const hostText: string =
      (harness.fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(hostText.toLowerCase()).not.toContain('page not found');
    expect(hostText.toLowerCase()).not.toContain('not found');
  });
});
