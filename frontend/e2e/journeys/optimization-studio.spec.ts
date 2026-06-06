import type { Locator, Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import {
  stubLandingReads,
  stubOptimizeAsync,
  stubOptimizeSync,
} from '../support/api-stubs';

// Acceptance item 9b — drive Optimization Studio as a real user: open the
// Optimization node, Run, and read weights + 4 stat cards + the frontier chart;
// for the async path watch the background job poll create→running→completed and
// render results once the tracker clears; and confirm the placeholder Export
// Report modal opens with its Generate control honestly disabled.
//
// `POST /api/v1/optimize` fetches prices via yfinance server-side (unstubbable
// at the browser), so the journey stubs optimize/jobs/run-fetch at the route
// layer (helpers from #853). Job completion is awaited on visible state — the
// tracker disappearing — never a fixed sleep.

const ASYNC_JOB_ID = 'job-opt-async'; // studio strips `job-` → run id `opt-async`
const ASYNC_RUN_ID = 'opt-async';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** A pipeline node card in the builder, located by its label. */
function pipelineNode(page: Page, label: string): Locator {
  return page.locator('app-pipeline-builder button').filter({ hasText: label });
}

function resultsPanel(page: Page): Locator {
  return page.locator('app-results-panel');
}

async function openOptimizerAndRun(page: Page): Promise<void> {
  await pipelineNode(page, 'Optimization').click();
  await page.getByRole('button', { name: 'Run Optimization' }).click();
}

/** Assert the rendered results: weights, 4 stat cards, frontier chart, no error. */
async function expectResults(page: Page): Promise<void> {
  await pipelineNode(page, 'Results').click();
  const results = resultsPanel(page);
  await expect(results).toBeVisible();
  await expect(results.getByText('AAPL')).toBeVisible(); // ≥1 weight row
  await expect(results.locator('app-stat-card')).toHaveCount(4);
  await expect(results.getByText('12.0%')).toBeVisible(); // Annual Return stat — metrics rendered
  await expect(
    results.locator('[data-testid="efficient-frontier-chart"]'),
  ).toBeVisible();
  await expect(page.getByText(/Optimization failed:/)).toHaveCount(0);
}

// ── Journeys ────────────────────────────────────────────────────────────────

test.describe('optimization-studio journey', () => {
  test('when a sync optimization runs, results render with weights, stats and the frontier chart', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubOptimizeSync(page);
    await page.goto('/optimization-studio');

    await openOptimizerAndRun(page);

    await expectResults(page);
  });

  test('when an async job runs, the tracker polls running→completed then clears and results render', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubOptimizeAsync(page, { jobId: ASYNC_JOB_ID, runId: ASYNC_RUN_ID });
    await page.goto('/optimization-studio');

    await openOptimizerAndRun(page);

    // The background-job tracker appears while polling, then disappears once the
    // job reaches completed and the run is fetched.
    const tracker = page.locator('app-job-progress-tracker');
    await expect(tracker).toBeVisible();
    await expect(tracker).toBeHidden();

    await expectResults(page);
  });

  test('when Export Report is opened, the modal shows and its Generate control is disabled', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await page.goto('/optimization-studio');

    await page.getByRole('button', { name: 'Export Report' }).click();

    await expect(page.getByRole('dialog')).toBeVisible();
    await expect(page.getByText('Report generation coming soon.')).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Generate Report' }),
    ).toBeDisabled();
  });
});
