import type { Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import {
  CREATED_PORTFOLIO_NAME,
  SEEDED_PORTFOLIO_NAME,
  stubFactorScore,
  stubLandingReads,
  stubMacroTransitional,
  stubPortfoliosRoundTrip,
  stubRiskCorrelation,
} from '../support/api-stubs';

// Acceptance item 9b — the remaining primary-workflow journeys: load each page,
// exercise its primary control, and assert the view updates, plus a Settings
// round-trip. All boot reads are seeded/stubbed (#853); waits are condition-based
// (visible state), never a fixed sleep; no live external service.

// ── Shared helper ─────────────────────────────────────────────────────────────

/** Click a tab by label, assert it becomes active and its panel renders. */
async function switchTabAndAssert(
  page: Page,
  name: string,
  panelSelector: string,
): Promise<void> {
  const tab = page.getByRole('tab', { name });
  await tab.click();
  await expect(tab).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator(panelSelector)).toBeVisible();
}

// ── Journeys ────────────────────────────────────────────────────────────────

test.describe('secondary-page journeys', () => {
  test('risk-center: clicking Correlation activates the tab and renders the heatmap', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubRiskCorrelation(page); // registered last → overrides the empty landing matrix
    await page.goto('/risk-center');

    await switchTabAndAssert(page, 'Correlation', 'app-correlation-panel');
    await expect(page.locator('app-correlation-panel app-echarts-heatmap canvas')).toBeVisible({
      timeout: 10_000,
    });
  });

  test('factor-research: running the score updates the results table', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubFactorScore(page);
    await page.goto('/factor-research');

    await page.getByRole('tab', { name: 'Scoring' }).click();
    const table = page.locator('app-score-panel app-data-table');
    // Before-anchor: no score row yet, so a passing NVDA assertion below is
    // causally tied to the run-score action (not a pre-existing default).
    await expect(table.getByText('NVDA')).toBeHidden();

    const runBtn = page.getByTestId('run-score-btn');
    await expect(runBtn).toBeVisible();
    await runBtn.click();

    await expect(table.getByText('NVDA')).toBeVisible();
  });

  test('attribution: the page renders and a tab switch shows the selected sub-panel', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await page.goto('/attribution');

    await expect(page.locator('app-brinson-panel')).toBeVisible(); // default tab
    await switchTabAndAssert(page, 'Multi-Level', 'app-saa-taa-panel');
  });

  test('rebalancing: switching to Policy renders the policy panel', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await page.goto('/rebalancing');

    await expect(page.locator('app-status-panel')).toBeVisible(); // default tab
    await switchTabAndAssert(page, 'Policy', 'app-policy-panel');
  });

  test('macro-intelligence: regime resolves to Transitional and a chart tab switches', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubMacroTransitional(page);
    await page.goto('/macro-intelligence');

    // Empty FRED/TE → composite score 0 → Transitional regime (the dot's regime;
    // the visible label is the derived business-cycle phase).
    await expect(page.locator('[data-region="regime-dot"]')).toHaveAttribute(
      'data-regime',
      'Transitional',
      { timeout: 15_000 },
    );

    // The chart tabs share one container; switching re-renders it in place.
    // aria-selected proves the switch; the scoped chart container (not the
    // sparkline svgs) proves the chart is present.
    const chartTab = page.getByRole('tab', { name: 'Credit Spreads' });
    await chartTab.click();
    await expect(chartTab).toHaveAttribute('aria-selected', 'true');
    await expect(
      page.locator('[role="img"][aria-label="Macro time-series chart"]'),
    ).toBeVisible();
  });

  test('settings: creating a portfolio round-trips into the list', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPortfoliosRoundTrip(page);
    await page.goto('/settings');

    const panel = page.locator('app-portfolios-panel');
    await page.getByRole('tab', { name: 'Portfolios' }).click();
    await expect(panel.getByText(SEEDED_PORTFOLIO_NAME)).toBeVisible();

    await page.getByTestId('toggle-create-portfolio').click();
    await page.getByTestId('portfolio-name').fill(CREATED_PORTFOLIO_NAME);
    await page.getByTestId('submit-portfolio').click();

    // The post-create list re-fetch surfaces the new portfolio.
    await expect(panel.getByText(CREATED_PORTFOLIO_NAME)).toBeVisible();
  });
});
