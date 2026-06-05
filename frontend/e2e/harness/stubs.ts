import type { BrowserContext } from '@playwright/test';

// IMPORTANT — seam limitation:
// page/context.route only intercepts requests the BROWSER issues. Trading 212,
// FRED and yfinance are called SERVER-SIDE (api → external host), so these
// browser-layer stubs do NOT intercept them. They are registered here for
// Cycle 1 so journeys that trigger any browser-originated third-party call are
// deterministic; true isolation of server-side calls needs an API-boundary
// stub (env-gated fake clients in the api service) — the better long-term seam,
// owned by Cycle-5+.

const TRADING212_FIXTURE = { items: [], nextPagePath: null };
const FRED_FIXTURE = { observations: [] };

async function stubJson(
  context: BrowserContext,
  pattern: string,
  body: unknown,
): Promise<void> {
  await context.route(pattern, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(body),
    });
  });
}

/** Register deterministic external-service stubs on a browser context. */
export async function registerExternalStubs(context: BrowserContext): Promise<void> {
  await stubJson(context, '**/*trading212.com/**', TRADING212_FIXTURE);
  await stubJson(context, '**/api.stlouisfed.org/**', FRED_FIXTURE);
  // PLACEHOLDER (yfinance): server-side only — cannot be stubbed at the browser
  // layer. Add an API-boundary fake facade instead. Left intentionally unwired.
  // await stubJson(context, '**/query*.finance.yahoo.com/**', YFINANCE_FIXTURE);
}
