import { test, expect } from './support/fixtures';

// Harness self-test (NOT a product journey — Cycle-5 owns those). Proves the
// runner + webServer + globalSetup(seed) + stubs wire end-to-end against the
// seeded docker stack.
test.describe('e2e harness self-test', () => {
  // The minimal smoke seed creates `e2e-portfolio` with NO price/sector data, so
  // the dashboard's five /portfolio-analytics/* boot calls correctly return 422
  // ("No price data for any portfolio ticker" / "No sector data available").
  // These are deliberate seed-gap responses, not defects — allow-list them
  // (covers both the network collector and the Chromium "Failed to load
  // resource: … 422" console echo) so the unstubbed-boot self-test is
  // deterministic. Product pages mask these via stubLandingReads.
  test.use({
    expectedFailures: [{ url: '/api/v1/portfolio-analytics/', status: 422 }],
  });

  test('when the app root loads against the seeded stack, no unexpected error is logged', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
    await page.waitForLoadState('networkidle');
    // Console-error and failed-request assertions run automatically via the
    // overridden `page` fixture in support/fixtures.ts.
  });
});
