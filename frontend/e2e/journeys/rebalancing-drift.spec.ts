import type { Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import {
  stubBacktest,
  stubDrift,
  stubDriftByBase,
  stubLandingReads,
  stubOptimizeSync,
  stubPipelineSession,
} from '../support/api-stubs';

// Acceptance item 9b — drive the portfolio-builder Drift canvas as a real user:
// switch to the Drift view, see target-vs-actual render, toggle invested↔total
// and confirm the trade list recomputes (new GET …?base=total, incrementing
// X-Drift-Request-Id), and confirm each diagnostic badge appears on its fixture.
//
// BuilderDriftService only fetches once the builder reaches resultStatus==='ok'
// AND a portfolio name is set, so every test first drives the builder to a
// completed result (reusing the #855 session/step/optimize stubs) before the
// Drift view is opened. Waits are condition-based — never a fixed sleep.

const DRIFT_VIEW = '[data-view-btn][data-view-id="drift"]';

// ── Builder-drive helpers ─────────────────────────────────────────────────────

function universe(page: Page) {
  return page
    .locator('app-step-section')
    .filter({ has: page.getByText('Universe', { exact: true }) });
}

async function setupBuilderStubs(page: Page): Promise<void> {
  await stubLandingReads(page);
  await stubPipelineSession(page, { tickers: ['AAPL', 'MSFT'] });
  await stubOptimizeSync(page);
  await stubBacktest(page);
}

/** Run the V2 wizard to a completed optimize result (resultStatus → 'ok'). */
async function driveBuilderToOk(page: Page): Promise<void> {
  await universe(page).getByRole('button').first().click(); // expand Universe
  const nSelected = page.locator('input[formcontrolname="n_selected"]');
  await nSelected.fill('20');
  await nSelected.blur();
  await page.getByRole('button', { name: 'Start pipeline' }).click();
  await universe(page).locator('app-step-load-panel').getByTestId('run-step').click();
  await expect(universe(page).getByText('assembly_hash: e2e-hash')).toBeVisible();
  await page.locator('[data-action="optimize"]').click();
}

/** Drive to ok, open the Drift view, and await the auto-fired drift fetch. */
async function openDrift(page: Page): Promise<void> {
  await driveBuilderToOk(page);
  const fetched = page.waitForResponse((r) => /\/portfolio\/[^/]+\/drift/.test(r.url()));
  await page.locator(DRIFT_VIEW).click();
  await fetched;
}

// ── Journeys ────────────────────────────────────────────────────────────────

test.describe('drift overlay + trade-list journey', () => {
  test('when the Drift view opens after a result, the overlay and target-vs-actual chart render', async ({
    authedPage: page,
  }) => {
    await setupBuilderStubs(page);
    await stubDriftByBase(page);
    await page.goto('/portfolio-builder');

    await openDrift(page);

    await expect(page.locator('[data-region="drift-overlay"]')).toBeVisible();
    await expect(page.locator('[data-region="drift-chart"] canvas')).toBeVisible({
      timeout: 10_000,
    });
    // Trade list renders rows for the populated invested fixture.
    await expect(page.locator('[data-region="trade-table"]')).toBeVisible();
    await expect(page.locator('[data-region="trade-row"]')).toHaveCount(1);
  });

  test('when toggling invested→total, a new base=total request fires and the trade list recomputes', async ({
    authedPage: page,
  }) => {
    await setupBuilderStubs(page);
    await stubDriftByBase(page);
    await page.goto('/portfolio-builder');

    await driveBuilderToOk(page);

    // Register both request listeners before any click so neither can fire
    // before its listener is attached.
    const investedReq = page.waitForRequest(
      (r) => r.url().includes('/drift') && r.url().includes('base=invested'),
    );
    const totalReq = page.waitForRequest(
      (r) => r.url().includes('/drift') && r.url().includes('base=total'),
    );

    await page.locator(DRIFT_VIEW).click();
    const investedId = Number((await investedReq).headers()['x-drift-request-id']);
    await expect(page.locator('[data-region="trade-row"]')).toHaveCount(1);

    await page.locator('[data-region="base-toggle"] [data-base-id="total"]').click();
    const totalId = Number((await totalReq).headers()['x-drift-request-id']);

    // New request, incremented request id, recomputed trade list, header updated.
    expect(totalId).toBeGreaterThan(investedId);
    await expect(page.locator('[data-region="trade-row"]')).toHaveCount(2);
    await expect(
      page.locator('[data-region="drift-header"] [data-cell="active-base"]'),
    ).toHaveText('total');
  });

  // ── Diagnostic badges (one per category) ────────────────────────────────────

  const GREY_CATEGORIES = [
    { key: 'unmapped', greyed: 'true' },
    { key: 'fx_missing', greyed: 'false' },
    { key: 'stale_price', greyed: 'false' },
    { key: 'target_not_on_broker', greyed: 'true' },
  ] as const;

  for (const cat of GREY_CATEGORIES) {
    test(`when the ${cat.key} fixture loads, its count badge renders and the row greyed state is ${cat.greyed}`, async ({
      authedPage: page,
    }) => {
      await setupBuilderStubs(page);
      await stubDrift(page, cat.key);
      await page.goto('/portfolio-builder');

      await openDrift(page);

      const item = page.locator(
        `[data-region="diagnostics-count-item"][data-cat-code="${cat.key}"]`,
      );
      await expect(item).toBeVisible();
      await expect(item.locator('[data-cell="count-value"]')).toHaveText('1');
      await expect(
        page.locator('[data-region="drift-flag-row"][data-row-ticker="TEST"]'),
      ).toHaveAttribute('data-row-greyed', cat.greyed);
    });
  }

  test('when the reconciliation_mismatch fixture loads, the reconciliation strip renders', async ({
    authedPage: page,
  }) => {
    await setupBuilderStubs(page);
    await stubDrift(page, 'reconciliation_mismatch');
    await page.goto('/portfolio-builder');

    await openDrift(page);

    await expect(
      page.locator('[data-region="diagnostics-reconciliation"]'),
    ).toBeVisible();
  });
});
