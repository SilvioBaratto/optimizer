import type { Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import { stubBacktest, stubLandingReads } from '../support/api-stubs';

// Acceptance item 9b — drive the Backtesting Lab as a real user: configure a run
// and Run it, watch the background job poll create→running→completed, then read
// the results panel (equity, drawdown, metrics); and submit an inverted date
// range and confirm the API error surfaces as a handled, visible message.
//
// POST /api/v1/backtest fetches prices via yfinance server-side (unstubbable at
// the browser), so optimize/jobs/run-fetch are stubbed at the route layer
// (`stubBacktest` from #853). Completion is awaited on the results panel
// becoming visible — never a fixed sleep.

const INVALID_RANGE_DETAIL = 'end_date must be after start_date';

// ── Helpers ───────────────────────────────────────────────────────────────────

function runButton(page: Page) {
  return page.getByRole('button', { name: 'Run Backtest' });
}

function resultsPanel(page: Page) {
  return page.locator('[data-testid="panel"]');
}

async function selectTab(page: Page, name: string): Promise<void> {
  // Match the exact label span so 'Metrics' doesn't also hit 'Rolling Metrics',
  // and a badge count (e.g. 'Drawdowns 1') doesn't break the lookup.
  await page
    .getByRole('tab')
    .filter({ has: page.getByText(name, { exact: true }) })
    .click();
}

// ── Journeys ────────────────────────────────────────────────────────────────

test.describe('backtesting journey', () => {
  test('when a backtest runs, the job completes and the results panel renders equity, drawdown and metrics', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubBacktest(page);
    await page.goto('/backtesting');

    await runButton(page).click();

    // The results panel only appears once the job completes and the run is
    // fetched — asserting it visible awaits the running→completed poll (2s gap;
    // the generous timeout absorbs CI jitter without a fixed sleep).
    await expect(resultsPanel(page)).toBeVisible({ timeout: 10_000 });
    await expect(resultsPanel(page).locator('app-echarts-line')).toBeVisible();

    // Drawdowns tab → ≥1 drawdown row (the -4% event on 2025-03-01).
    await selectTab(page, 'Drawdowns');
    await expect(page.getByText('Top 10 Drawdowns')).toBeVisible();
    // start and trough both fall on 2025-03-01 → two cells; one is enough.
    await expect(page.getByRole('cell', { name: '2025-03-01' }).first()).toBeVisible();

    // Metrics tab → the portfolio-vs-benchmark metrics table.
    await selectTab(page, 'Metrics');
    await expect(page.getByText('Sharpe Ratio')).toBeVisible();
  });

  test.describe('invalid date range', () => {
    test.use({ expectedFailures: [{ url: '/api/v1/backtest', status: 400 }] });

    test('when end is before start, the API 400 surfaces as a handled error message', async ({
      authedPage: page,
    }) => {
      await stubLandingReads(page);
      await stubBacktest(page, { error: { status: 400, detail: INVALID_RANGE_DETAIL } });
      await page.goto('/backtesting');

      // Invert the range — start after end. onRunBacktest only guards empty
      // tickers, so the inverted range reaches the API.
      await page.getByLabel('Backtest start date').fill('2026-02-25');
      await page.getByLabel('Backtest end date').fill('2020-01-01');

      await runButton(page).click();

      // Handled, visible error banner — no crash, no uncaught console error.
      await expect(page.getByText(/Backtest failed:/)).toBeVisible();
    });
  });
});
