import type { Download, Locator, Page } from '@playwright/test';

import { test, expect } from '../support/fixtures';
import {
  PB_ARTIFACT_NAMES,
  stubBacktest,
  stubLandingReads,
  stubOptimizeSync,
  stubPipelineSession,
} from '../support/api-stubs';

// Acceptance item 9b — drive the V2 portfolio-builder wizard and the legacy
// pipeline-stepper as a real user: validate n_selected against the API band,
// advance steps with downstream steps gated, surface a coverage-gate abort as
// a blocking error, reach the weights view, and download each pipeline artifact.
//
// Step order is enforced server-side; here the #853-style pipeline-step stubs
// resolve each poll to a terminal state on its first tick, so every wait is
// condition-based (toBeVisible / waitForResponse) with no fixed sleep.

const VALID_N_SELECTED = '20'; // in the API band [15, 30]
const BELOW_BAND_N_SELECTED = '14'; // below ge=15 → invalid
const BAND_ERROR = 'Must be between 15 and 30.';

// Six steps that precede the coverage gate, in pipeline order.
const PRE_GATE_STEPS = [
  'Load',
  'Screen',
  'Clean returns',
  'Build history',
  'Validate in-sample',
  'Validate out-of-sample',
] as const;

// ── Shared helpers ────────────────────────────────────────────────────────────

/** A pipeline step section (legacy stepper or V2 accordion) by its title. */
function section(page: Page, title: string): Locator {
  return page
    .locator('app-step-section')
    .filter({ has: page.getByText(title, { exact: true }) });
}

function nSelectedInput(page: Page): Locator {
  return page.locator('input[formcontrolname="n_selected"]');
}

function startButton(page: Page): Locator {
  return page.getByRole('button', { name: 'Start pipeline' });
}

async function enterNSelected(page: Page, value: string): Promise<void> {
  const input = nSelectedInput(page);
  await input.fill(value);
  await input.blur(); // mark the control touched so the band error renders
}

/** Expand the V2 Universe accordion to mount the run-config + load/screen panels. */
async function expandUniverseSection(page: Page): Promise<void> {
  await section(page, 'Universe').getByRole('button').first().click();
}

/** Fetch one artifact from the browser context (hits the stubbed endpoint). */
function fetchArtifact(
  page: Page,
  name: string,
): Promise<{ status: number; length: number }> {
  return page.evaluate(async (artifact) => {
    const res = await fetch(
      `/api/v1/pipeline-builder/sessions/e2e-session/artifacts/${artifact}`,
    );
    return { status: res.status, length: (await res.text()).length };
  }, name);
}

// ── Legacy pipeline-stepper journeys ──────────────────────────────────────────

test.describe('legacy pipeline-stepper journey', () => {
  test('when n_selected is below the API band, the band error shows and no session starts', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page);
    await page.goto('/portfolio-builder-legacy');

    await enterNSelected(page, BELOW_BAND_N_SELECTED);

    await expect(page.getByText(BAND_ERROR)).toBeVisible();
    await expect(startButton(page)).toBeDisabled();
  });

  test('when n_selected is in-band, the session is created and the load step runs to completed', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page);
    await page.goto('/portfolio-builder-legacy');

    await enterNSelected(page, VALID_N_SELECTED);
    await startButton(page).click();

    // load auto-dispatches on session start and resolves to Done.
    await expect(section(page, 'Load').getByText('Done')).toBeVisible();
  });

  test('when a step completes, the next unlocks while a later step stays locked and unopenable', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page);
    await page.goto('/portfolio-builder-legacy');

    await enterNSelected(page, VALID_N_SELECTED);
    await startButton(page).click();
    await expect(section(page, 'Load').getByText('Done')).toBeVisible();

    // A downstream step is Locked; clicking its header must not open its panel.
    const cleanReturns = section(page, 'Clean returns');
    await expect(cleanReturns.getByText('Locked')).toBeVisible();
    await cleanReturns.getByRole('button').first().click();
    await expect(cleanReturns.locator('app-step-clean-returns-panel')).toHaveCount(0);

    // Advancing the active step unlocks and runs the next one.
    await page.getByRole('button', { name: 'Continue →' }).click();
    await expect(section(page, 'Screen').getByText('Done')).toBeVisible();
  });

  test('when the coverage gate fails, a blocking abort-gate error surfaces', async ({
    authedPage: page,
  }) => {
    const gateReason = 'Only 1 factor passed IS ∩ OOS; 2 required.';
    await stubLandingReads(page);
    await stubPipelineSession(page, { abortAtStep: 'coverage_gate', gateReason });
    await page.goto('/portfolio-builder-legacy');

    await enterNSelected(page, VALID_N_SELECTED);
    await startButton(page).click();

    // Walk each pre-gate step to Done and Continue; the final Continue dispatches
    // the coverage gate, which the stub fails with a gateReason.
    for (const title of PRE_GATE_STEPS) {
      await expect(section(page, title).getByText('Done')).toBeVisible();
      await page.getByRole('button', { name: 'Continue →' }).click();
    }

    await expect(page.getByText('Pipeline halted at Coverage Gate.')).toBeVisible();
    await expect(page.getByText(gateReason)).toBeVisible();
  });
});

// ── V2 portfolio-builder journeys ─────────────────────────────────────────────

test.describe('V2 portfolio-builder journey', () => {
  test('when n_selected is below the API band on the V2 wizard, the band error shows', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page);
    await page.goto('/portfolio-builder');

    await expandUniverseSection(page);
    await enterNSelected(page, BELOW_BAND_N_SELECTED);

    await expect(page.getByText(BAND_ERROR)).toBeVisible();
    await expect(startButton(page)).toBeDisabled();
  });

  test('when the load step runs and Optimize fires, the weights view renders weight rows', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page, { tickers: ['AAPL', 'MSFT'] });
    await stubOptimizeSync(page);
    await stubBacktest(page);
    await page.goto('/portfolio-builder');

    await expandUniverseSection(page);
    await enterNSelected(page, VALID_N_SELECTED);
    await startButton(page).click();

    // The seeded first step is Ready — run it and confirm it completes.
    const universe = section(page, 'Universe');
    await universe.locator('app-step-load-panel').getByTestId('run-step').click();
    await expect(universe.getByText('assembly_hash: e2e-hash')).toBeVisible();

    // Optimize triggers the result run (optimize + backtest) → builder weights.
    await page.locator('[data-action="optimize"]').click();

    await page.locator('[data-view-btn][data-view-id="bars"]').click();
    await expect(page.locator('[data-row="weight-bar"]').first()).toBeVisible({
      timeout: 15_000,
    });
    await expect(page.locator('[data-row="weight-bar"]')).toHaveCount(2);

    // The donut view leaves its empty state once weights exist.
    await page.locator('[data-view-btn][data-view-id="donut"]').click();
    await expect(page.locator('[data-region="allocation-donut-empty"]')).toHaveCount(0);
  });

  test('when Export is clicked, each of the four artifacts downloads successfully', async ({
    authedPage: page,
  }) => {
    await stubLandingReads(page);
    await stubPipelineSession(page);
    await page.goto('/portfolio-builder');

    await expandUniverseSection(page);
    await enterNSelected(page, VALID_N_SELECTED);
    await startButton(page).click();

    const exportBtn = page.locator('[data-action="export"]');
    await expect(exportBtn).toBeEnabled();

    // One click streams all four artifacts as browser downloads — the wired
    // Export button reaches every artifact (download-event criterion).
    const downloads: Download[] = [];
    page.on('download', (d) => downloads.push(d));
    await exportBtn.click();

    await expect.poll(() => downloads.length).toBe(PB_ARTIFACT_NAMES.length);
    const names = downloads.map((d) => d.suggestedFilename()).sort();
    expect(names).toEqual([...PB_ARTIFACT_NAMES].sort());

    // Each artifact endpoint returns 200 with a non-empty body (file response).
    for (const name of PB_ARTIFACT_NAMES) {
      const result = await fetchArtifact(page, name);
      expect(result.status).toBe(200);
      expect(result.length).toBeGreaterThan(0);
    }
  });
});
