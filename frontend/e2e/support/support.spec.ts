/**
 * Support-layer wiring tests (TDD — Red → Green).
 *
 * These specs assert the support infrastructure itself without exercising
 * product journeys. They do require the docker stack because `page.evaluate`
 * still runs inside the Chromium context that the webServer provides; but they
 * do NOT depend on any seeded DB data or Angular routing beyond `/`.
 */

import type { Page } from '@playwright/test';

import { test, expect, seedPortfolioGuard, SEEDED_PORTFOLIO_ID, isExpectedFailure } from './fixtures';
import { stubOptimizeSync, stubOptimizeAsync, stubDrift, stubLandingReads, DRIFT_FIXTURES } from './api-stubs';

// Stub-wiring tests assert the route registrars themselves, not the product
// app. Booting the real SPA at `/` fires uncontrolled boot API calls that 4xx
// against the seeded-only backend and pollute the console-error collector. A
// stubbed blank same-origin document gives `fetch('/api/v1/..')` the right
// origin to hit the page.route stubs, with zero Angular boot.
async function gotoBlank(page: Page): Promise<void> {
  await page.route('**/__e2e_blank__', (route) =>
    route.fulfill({ contentType: 'text/html', body: '<!doctype html><title>blank</title>' }),
  );
  await page.goto('/__e2e_blank__');
}

// ── isExpectedFailure unit tests (no browser needed) ─────────────────────────

test.describe('isExpectedFailure predicate', () => {
  test('matches substring url and any status when status is omitted', () => {
    expect(isExpectedFailure('/api/v1/optimize', 400, [{ url: '/api/v1/optimize' }])).toBe(true);
  });

  test('matches regexp url', () => {
    expect(isExpectedFailure('/api/v1/optimize', 422, [{ url: /optimize/, status: 422 }])).toBe(true);
  });

  test('does not match when status differs', () => {
    expect(isExpectedFailure('/api/v1/optimize', 500, [{ url: '/api/v1/optimize', status: 400 }])).toBe(false);
  });

  test('returns false for an empty allow-list', () => {
    expect(isExpectedFailure('/api/v1/optimize', 400, [])).toBe(false);
  });
});

// ── stubOptimizeSync ──────────────────────────────────────────────────────────

test.describe('stubOptimizeSync', () => {
  test('POST /api/v1/optimize returns the fixture with status 200', async ({ page }) => {
    await stubOptimizeSync(page);
    await gotoBlank(page);

    const result = await page.evaluate(async () => {
      const res = await fetch('/api/v1/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2025-01-01', optimizer_type: 'mean_risk' }),
      });
      return { status: res.status, body: await res.json() };
    });

    expect(result.status).toBe(200);
    expect(result.body.status).toBe('completed');
    expect(Object.keys(result.body.weights as Record<string, number>).length).toBeGreaterThan(0);
  });

  test('applies overrides to the fixture', async ({ page }) => {
    await stubOptimizeSync(page, { optimizerType: "regime_blended" });
    await gotoBlank(page);

    const result = await page.evaluate(async () => {
      const res = await fetch('/api/v1/optimize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      return res.json();
    });

    expect((result as { optimizerType: string }).optimizerType).toBe('regime_blended');
  });
});

// ── stubOptimizeAsync running→completed transition ────────────────────────────

test.describe('stubOptimizeAsync', () => {
  test('first job poll returns running, second returns completed', async ({ page }) => {
    await stubOptimizeAsync(page);
    await gotoBlank(page);

    const first = await page.evaluate(async () => {
      const res = await fetch('/api/v1/jobs/e2e-job-opt');
      return (await res.json() as { status: string }).status;
    });

    const second = await page.evaluate(async () => {
      const res = await fetch('/api/v1/jobs/e2e-job-opt');
      return (await res.json() as { status: string }).status;
    });

    expect(first).toBe('running');
    expect(second).toBe('completed');
  });

  test('POST /api/v1/optimize returns 202 with job_id and run_id (no weights key)', async ({ page }) => {
    await stubOptimizeAsync(page);
    await gotoBlank(page);

    const body = await page.evaluate(async () => {
      const res = await fetch('/api/v1/optimize', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      return res.json();
    });

    expect((body as { job_id: string }).job_id).toBe('e2e-job-opt');
    expect((body as { run_id: string }).run_id).toBe('e2e-run-opt');
    expect('weights' in (body as object)).toBe(false);
  });
});

// ── seedPortfolioGuard ────────────────────────────────────────────────────────

test.describe('seedPortfolioGuard', () => {
  test('localStorage key is set before first navigation', async ({ page }) => {
    await seedPortfolioGuard(page);
    // Use the blank doc (not `/`) so the real SPA never boots and overwrites the
    // seeded id with the active DB portfolio — this asserts the helper, not app boot.
    await gotoBlank(page);

    const storedId = await page.evaluate(() =>
      localStorage.getItem('optimizer.currentPortfolioId'),
    );

    expect(storedId).toBe(SEEDED_PORTFOLIO_ID);
  });

  test('navigating to /risk-center stays on /risk-center (guard passes)', async ({ page }) => {
    // Stub the boot analytics reads so the real risk-center page renders without
    // polluting the base fixture's console/failed-request collectors.
    await stubLandingReads(page);
    await seedPortfolioGuard(page);
    await page.goto('/risk-center');
    await page.waitForURL(/risk-center/);

    expect(page.url()).toContain('risk-center');
  });
});

// ── stubDrift ─────────────────────────────────────────────────────────────────

test.describe('stubDrift', () => {
  test('reconciliation_mismatch fixture has reconciliation_ok:false', async ({ page }) => {
    await stubDrift(page, "reconciliation_mismatch");
    await gotoBlank(page);

    const diagnostics = await page.evaluate(async () => {
      const res = await fetch('/api/v1/portfolio/any-id/drift');
      const data = await res.json() as { diagnostics: { reconciliation_ok: boolean; reconciliation_delta_pct: number | null } };
      return data.diagnostics;
    });

    expect(diagnostics.reconciliation_ok).toBe(false);
    expect(diagnostics.reconciliation_delta_pct).not.toBeNull();
  });

  test('base fixture has reconciliation_ok:true and no drift rows', async ({ page }) => {
    await stubDrift(page, "base");
    await gotoBlank(page);

    const data = await page.evaluate(async () => {
      const res = await fetch('/api/v1/portfolio/any-id/drift');
      return res.json() as Promise<{ diagnostics: { reconciliation_ok: boolean }; drift: unknown[] }>;
    });

    expect(data.diagnostics.reconciliation_ok).toBe(true);
    expect(data.drift).toHaveLength(0);
  });

  test('DRIFT_FIXTURES keys match expected DriftFixtureKey values', () => {
    const keys = Object.keys(DRIFT_FIXTURES).sort();
    expect(keys).toEqual(
      ['base', 'fx_missing', 'reconciliation_mismatch', 'stale_price', 'target_not_on_broker', 'total', 'unmapped'],
    );
  });
});
