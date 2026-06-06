import { test as base, expect, type Page, type BrowserContext } from '@playwright/test';

import { registerExternalStubs } from '../harness/stubs';

// ── Constants ─────────────────────────────────────────────────────────────────

export const SEEDED_PORTFOLIO_ID = '22222222-2222-2222-2222-222222222222';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ExpectedFailure {
  url: string | RegExp;
  status?: number;
}

// ── Predicates ────────────────────────────────────────────────────────────────

function matchesUrl(url: string, pattern: string | RegExp): boolean {
  if (pattern instanceof RegExp) return pattern.test(url);
  return url.includes(pattern);
}

export function isExpectedFailure(
  url: string,
  status: number,
  list: ExpectedFailure[],
): boolean {
  return list.some(
    (f) => matchesUrl(url, f.url) && (f.status === undefined || f.status === status),
  );
}

// ── Guard helpers ─────────────────────────────────────────────────────────────

/** Sets localStorage before the first navigation so the portfolio guard passes. */
export async function seedPortfolioGuard(page: Page): Promise<void> {
  await page.addInitScript((id: string) => {
    localStorage.setItem('optimizer.currentPortfolioId', id);
  }, SEEDED_PORTFOLIO_ID);
}

// ── Fixture types ─────────────────────────────────────────────────────────────

interface SupportFixtures {
  /** Allow-list of deliberate 4xx/5xx API responses for a specific test. */
  expectedFailures: ExpectedFailure[];
  /** A page with the portfolio guard pre-seeded via localStorage. */
  authedPage: Page;
}

// ── Collector helpers ─────────────────────────────────────────────────────────

// A deliberately allow-listed failed request (expectedFailures) is echoed by
// Chromium as a "Failed to load resource: … status of N" console error. That
// echo carries the status but not the URL, so match on status alone — if any
// expected failure shares it, the console echo is expected too. This is
// intentionally coarse; the network collector (attachFailedRequestCollector)
// remains the URL+status safety net for `/api/` requests.
function isAllowlistedResourceError(
  text: string,
  expectedFailures: ExpectedFailure[],
): boolean {
  const match = text.match(/Failed to load resource:.*status of (\d+)/);
  if (!match) return false;
  const status = Number(match[1]);
  return expectedFailures.some((f) => f.status === undefined || f.status === status);
}

function attachConsoleErrorCollector(
  page: Page,
  bucket: string[],
  expectedFailures: ExpectedFailure[],
): void {
  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    if (isAllowlistedResourceError(msg.text(), expectedFailures)) return;
    bucket.push(msg.text());
  });
  page.on('pageerror', (err) => bucket.push(err.message));
}

function attachFailedRequestCollector(
  page: Page,
  bucket: string[],
  expectedFailures: ExpectedFailure[],
): void {
  page.on('response', (res) => {
    const status = res.status();
    const url = res.url();
    if (url.includes('/api/') && status >= 400 && !isExpectedFailure(url, status, expectedFailures)) {
      bucket.push(`${status} ${url}`);
    }
  });
}

// ── Extended test ─────────────────────────────────────────────────────────────

export const test = base.extend<SupportFixtures>({
  expectedFailures: [[], { option: true }],

  // Overrides the base `page` fixture so every test gets stubs + post-test assertions.
  page: async ({ page, context, expectedFailures }, use) => {
    await registerExternalStubs(context as BrowserContext);

    const consoleErrors: string[] = [];
    const failedRequests: string[] = [];

    attachConsoleErrorCollector(page, consoleErrors, expectedFailures);
    attachFailedRequestCollector(page, failedRequests, expectedFailures);

    await use(page);

    expect(consoleErrors, 'No console errors expected').toEqual([]);
    expect(failedRequests, 'No unexpected API failures expected').toEqual([]);
  },

  authedPage: async ({ page }, use) => {
    await seedPortfolioGuard(page);
    await use(page);
  },
});

export { expect };
