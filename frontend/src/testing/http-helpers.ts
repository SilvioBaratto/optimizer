import { HttpTestingController } from '@angular/common/http/testing';

export { BACKGROUND_REQUEST, backgroundContext } from '../app/core/http-context';

/** Maps a request URL to a stub response body. */
export type StubFor = (url: string) => Record<string, unknown>;

// Pre-built typed stubs for the dashboard bootstrap endpoints.
const STUB_INDICES = { indices: [], total: 0 };
const STUB_SNAPSHOT = {
  vix: 0,
  vixChange: 0,
  sp500Return: 0,
  tenYearYield: 0,
  yieldChange: 0,
  usdIndex: 0,
  usdChange: 0,
  asOf: '2026-01-01T00:00:00.000Z',
};
const STUB_EQUITY = { points: [] };
const STUB_METRICS = { kpis: [], nav: 0, navChangePct: 0, currency: 'EUR' };
const STUB_ALLOCATION = { nodes: [], totalPositions: 0, totalSectors: 0 };
const STUB_ASSET_RETURNS = { returns: [] };
const STUB_ROLLING = { window: 63, sharpe: [], volatility: [], beta: [] };

// Fragment → stub table. Extend by adding a row (OCP), not an if-branch.
const STUB_TABLE: ReadonlyArray<readonly [string, Record<string, unknown>]> = [
  ['/market/indices', STUB_INDICES],
  ['/market/snapshot', STUB_SNAPSHOT],
  ['/equity-curve', STUB_EQUITY],
  ['/performance-metrics', STUB_METRICS],
  ['/allocation', STUB_ALLOCATION],
  ['/asset-class-returns', STUB_ASSET_RETURNS],
  ['/rolling-metrics', STUB_ROLLING],
];

/** Default URL→body mapper over the pre-built stub constants. */
export function defaultStubFor(url: string): Record<string, unknown> {
  const match = STUB_TABLE.find(([fragment]) => url.includes(fragment));
  return match ? match[1] : {};
}

/** Flush every pending request, resolving each body via `stubFor`. */
export function drainRequests(
  http: HttpTestingController,
  stubFor: StubFor = defaultStubFor,
): void {
  for (const req of http.match(() => true)) {
    req.flush(stubFor(req.request.url));
  }
}

/** Flush all pending requests whose URL contains `fragment`; return the count. */
export function flushMatching(
  http: HttpTestingController,
  fragment: string,
  body: Record<string, unknown>,
): number {
  const reqs = http.match((r) => r.url.includes(fragment));
  for (const req of reqs) {
    req.flush(body);
  }
  return reqs.length;
}
