import { expect, test } from '@playwright/test';

import { registerExternalStubs } from './harness/stubs';

// Harness self-test (NOT a product journey — Cycle-5 owns those). Proves the
// runner + webServer + globalSetup(seed) + stubs wire end-to-end against the
// seeded docker stack.
test.describe('e2e harness self-test', () => {
  test('when the app root loads against the seeded stack, no console error is logged', async ({
    page,
    context,
  }) => {
    await registerExternalStubs(context);

    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
    expect(errors).toEqual([]);
  });
});
