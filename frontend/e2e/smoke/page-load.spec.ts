import type { Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import { stubLandingReads } from '../support/api-stubs';

// Page-load smoke for every user-facing route (acceptance item 9a). Each route
// must render its primary landmark + root component, keep the console clean,
// issue no failing initial-load `/api/` request (both inherited from the #853
// base fixture), and leave no loading skeleton stuck after the network settles.

interface RouteCase {
  readonly path: string;
  readonly selector: string;
}

// 14 user-facing routes from app.routes.ts (excludes the `optimize` redirect and
// the `**` wildcard). `*` routes sit behind portfolioRequiredGuard — the seeded
// localStorage portfolio id (via seedPortfolioGuard) lets them render directly.
const ROUTES: readonly RouteCase[] = [
  { path: '/', selector: 'app-dashboard' }, // guarded default
  { path: '/portfolio-builder', selector: 'app-portfolio-builder' },
  { path: '/portfolio-builder-legacy', selector: 'app-pipeline-stepper' },
  { path: '/optimization-studio', selector: 'app-optimization-studio' },
  { path: '/backtesting', selector: 'app-backtesting' }, // guarded
  { path: '/risk-center', selector: 'app-risk-center' }, // guarded
  { path: '/factor-research', selector: 'app-factor-research' },
  { path: '/rebalancing', selector: 'app-rebalancing' }, // guarded
  { path: '/attribution', selector: 'app-attribution' }, // guarded
  { path: '/macro-intelligence', selector: 'app-macro-intelligence' },
  { path: '/ai-control-room', selector: 'app-ai-control-room' },
  { path: '/settings', selector: 'app-settings' },
  { path: '/portfolio/e2e-portfolio', selector: 'app-portfolio-detail' },
  { path: '/instrument/SM1', selector: 'app-instrument-detail' },
];

async function assertPageLoads(page: Page, route: RouteCase): Promise<void> {
  await stubLandingReads(page);
  await page.goto(route.path);
  await expect(page.locator('main#main-content')).toBeVisible();
  await expect(page.locator(route.selector).first()).toBeVisible();
  await page.waitForLoadState('networkidle');
  await expect(page.locator('app-loading-skeleton')).toHaveCount(0);
}

test.describe('page-load smoke', () => {
  for (const route of ROUTES) {
    test(`when ${route.path} loads, its landmark and root component render cleanly`, async ({
      authedPage,
    }) => {
      await assertPageLoads(authedPage, route);
    });
  }

  test('when a sidebar link is clicked, nav round-trips between pages', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await page.goto('/portfolio-builder');
    await page.getByRole('link', { name: 'Optimization Studio' }).click();
    await page.waitForURL(/optimization-studio/);
    await expect(page.locator('app-optimization-studio').first()).toBeVisible();
    await page.getByRole('link', { name: 'Risk Center' }).click();
    await page.waitForURL(/risk-center/);
    await expect(page.locator('app-risk-center').first()).toBeVisible();
  });
});
