/**
 * Source-blind CRUD round-trip tests for issue #1021.
 *
 * Criterion pinned from acceptance-criteria text (oracle classification):
 *   [T2] Limits CRUD round-trips on /risk/{portfolio_name}/limits —
 *        create, update, and delete each refresh comp.limits() with the
 *        correct state.
 *
 * What makes these tests new relative to earlier committed specs:
 *   - risk-center.spec.ts tests each CRUD method in isolation.  These tests
 *     pin the sequential chain: list → create → verify → update → verify →
 *     delete → verify, checking comp.limits() at every step.
 *   - HTTP expectations use exact URLs, not `r.url.includes(fragment)`.
 *
 * Source-blind assumptions (inferred from .spec.ts artefacts):
 *   comp.onCreateLimit({ metric, limit_type, threshold })   → POST
 *   comp.onUpdateLimit({ id, current, breached })           → PUT
 *   comp.onDeleteLimit(id)                                  → DELETE
 *   comp.limits()          → WritableSignal<RiskLimit[]>    (display items)
 *   comp.limitsResponse()  → raw API response with breachCount
 *
 * Wire format (camelCase as returned by the API):
 *   { id, portfolioId, metric, limitType, threshold, isBreached,
 *     currentValue, lastCheckedAt, createdAt, updatedAt }
 *
 * Display-format status rules (from risk.mapper.spec.ts):
 *   isBreached === true                  → 'breached'
 *   currentValue / threshold >= 0.8     → 'warning'
 *   otherwise                           → 'ok'
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import type { HttpTestingController } from '@angular/common/http/testing';

import {
  configureTestBed,
  drainRequests,
  injectHttp,
  installResizeObserverStub,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { RiskCenterComponent } from './risk-center';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

// ─── Constants ────────────────────────────────────────────────────────────────

const API       = environment.apiUrl;
const PORTFOLIO = 'trading212';
const LIMITS_URL = `${API}risk/${PORTFOLIO}/limits`;
const ISO        = '2026-06-15T00:00:00Z';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeCtxStub(name: string | null = PORTFOLIO, id: string | null = 'stub-id') {
  const idSig   = signal<string | null>(id);
  const nameSig = signal<string | null>(name);
  return {
    currentPortfolioId:   idSig,
    currentPortfolioName: nameSig,
    hasPortfolio:         { _value: name !== null },
    _idSig:               idSig,
    _nameSig:             nameSig,
  };
}

/** Wire-format limit item as returned by GET /risk/{name}/limits. */
function wireLimit(overrides: Partial<{
  id: string;
  portfolioId: string;
  metric: string;
  limitType: string;
  threshold: number;
  isBreached: boolean;
  currentValue: number | null;
}> = {}) {
  return {
    id:            'rl-1',
    portfolioId:   'pf-1',
    metric:        'max_drawdown',
    limitType:     'upper',
    threshold:     0.2,
    isBreached:    false,
    currentValue:  null,
    lastCheckedAt: null,
    createdAt:     ISO,
    updatedAt:     ISO,
    ...overrides,
  };
}

/** List response wrapper. */
function limitsList(items: ReturnType<typeof wireLimit>[], breachCount = 0) {
  return { items, breachCount };
}

// Stubs for the five analytics endpoints (no data needed for these tests).
function analyticsStubFor(url: string): Record<string, unknown> {
  if (url.includes('/risk/var'))           return { var: { '95': 0.03 }, cvar: { '95': 0.05 }, method: 'historical', lookback: 252, nObservations: 252 };
  if (url.includes('/risk/correlation'))  return { assets: [], matrix: [], clusterLabels: [] };
  if (url.includes('/risk/factor-exposure')) return { exposures: {}, assetExposures: {} };
  if (url.includes('/risk/concentration')) return { assets: [], summary: { hhi: 0, effectiveN: 0, topNRatio: 0 } };
  if (url.includes('/risk/liquidity'))    return { assets: [], summary: { weightedAvgDaysToLiquidate: 0 } };
  return {};
}

// ─── Suite ───────────────────────────────────────────────────────────────────

describe('RiskCenterComponent — limits CRUD round-trip (issue #1021)', () => {
  let fixture: ComponentFixture<RiskCenterComponent>;
  let comp: RiskCenterComponent;
  let http: HttpTestingController;

  /** Flush the five analytics endpoints and the limits endpoint with the given items. */
  function settleWithLimits(limitItems: ReturnType<typeof wireLimit>[] = []): void {
    fixture.detectChanges();
    for (const req of http.match((r) => !r.url.includes('/limits'))) {
      req.flush(analyticsStubFor(req.request.url));
    }
    for (const req of http.match((r) => r.url.includes('/limits') && r.method === 'GET')) {
      req.flush(limitsList(limitItems));
    }
    fixture.detectChanges();
  }

  beforeEach(async () => {
    installResizeObserverStub();
    await configureTestBed({
      imports:   [RiskCenterComponent],
      withHttp:  true,
      providers: [
        ICON_PROVIDER,
        { provide: PortfolioContextService, useValue: makeCtxStub() },
      ],
    });
    fixture = TestBed.createComponent(RiskCenterComponent);
    comp    = fixture.componentInstance;
    http    = injectHttp();
  });

  afterEach(() => http.verify());

  // ── Sequential round-trip: single flow covering all four ops ─────────────

  it('sequential round-trip: list → create → update → delete tracks comp.limits() at each step', () => {
    // ── Step 0: Init — empty limits list ────────────────────────────────────
    settleWithLimits([]);
    expect(comp.limits().length)
      .withContext('Before any CRUD ops, limits list should be empty')
      .toBe(0);

    // ── Step 1: Create ───────────────────────────────────────────────────────
    comp.onCreateLimit({ metric: 'max_drawdown', limit_type: 'upper', threshold: 0.2 });

    const createReq = http.expectOne(
      (r) => r.url === LIMITS_URL && r.method === 'POST',
    );
    expect(createReq.request.body.threshold)
      .withContext('POST body must carry the threshold value')
      .toBe(0.2);
    createReq.flush({});

    // Reload after create — list now contains the new item
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET')
      .flush(limitsList([wireLimit({ id: 'rl-1', currentValue: null })]));
    fixture.detectChanges();

    expect(comp.limits().length)
      .withContext('After create + reload, limits list should have 1 item')
      .toBe(1);
    expect(comp.limits().some((l) => l.id === 'rl-1'))
      .withContext('Newly created limit must appear in comp.limits()')
      .toBe(true);
    expect(comp.limits().find((l) => l.id === 'rl-1')?.status)
      .withContext('New limit with no current value should have status "ok"')
      .toBe('ok');

    // ── Step 2: Update — push current value above 80 % of threshold ─────────
    comp.onUpdateLimit({ id: 'rl-1', current: 0.18, breached: false });

    const updateReq = http.expectOne(
      (r) => r.url === `${LIMITS_URL}/rl-1` && r.method === 'PUT',
    );
    // Wire-format body uses snake_case keys
    expect(updateReq.request.body.current_value ?? updateReq.request.body.currentValue)
      .withContext('PUT body must carry the new current value')
      .toBeDefined();
    updateReq.flush({});

    // Reload after update — currentValue=0.18, threshold=0.2 → 0.9 >= 0.8 → 'warning'
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET')
      .flush(limitsList([wireLimit({ id: 'rl-1', currentValue: 0.18, isBreached: false })]));
    fixture.detectChanges();

    const afterUpdate = comp.limits().find((l) => l.id === 'rl-1');
    expect(afterUpdate)
      .withContext('Limit must still exist after update')
      .toBeDefined();
    expect(afterUpdate?.status)
      .withContext('currentValue/threshold = 0.9 >= 0.8 should yield "warning" status')
      .toBe('warning');

    // ── Step 3: Breach — push current value above threshold ─────────────────
    comp.onUpdateLimit({ id: 'rl-1', current: 0.22, breached: true });

    const breachReq = http.expectOne(
      (r) => r.url === `${LIMITS_URL}/rl-1` && r.method === 'PUT',
    );
    breachReq.flush({});

    // Reload with isBreached flag → 'breached'
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET')
      .flush(limitsList([wireLimit({ id: 'rl-1', currentValue: 0.22, isBreached: true })], 1));
    fixture.detectChanges();

    const afterBreach = comp.limits().find((l) => l.id === 'rl-1');
    expect(afterBreach?.status)
      .withContext('isBreached=true must map to status "breached"')
      .toBe('breached');
    expect(comp.limitsResponse().breachCount)
      .withContext('breachCount must reflect the reloaded response')
      .toBe(1);

    // ── Step 4: Delete ───────────────────────────────────────────────────────
    comp.onDeleteLimit('rl-1');

    const deleteReq = http.expectOne(
      (r) => r.url === `${LIMITS_URL}/rl-1` && r.method === 'DELETE',
    );
    deleteReq.flush(null, { status: 204, statusText: 'No Content' });

    // Reload after delete — list is empty again
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET')
      .flush(limitsList([], 0));
    fixture.detectChanges();

    expect(comp.limits().length)
      .withContext('After delete + reload, limits list must be empty')
      .toBe(0);
    expect(comp.limits().some((l) => l.id === 'rl-1'))
      .withContext('Deleted limit must not appear in comp.limits()')
      .toBe(false);
    expect(comp.limitsResponse().breachCount)
      .withContext('breachCount must be zero after list reloads without items')
      .toBe(0);
  });

  // ── Isolated: each operation individually uses the correct URL and verb ───

  it('onCreateLimit fires POST to exact URL /risk/{name}/limits', () => {
    settleWithLimits([]);
    comp.onCreateLimit({ metric: 'var_95', limit_type: 'upper', threshold: 0.05 });

    const req = http.expectOne((r) => r.url === LIMITS_URL && r.method === 'POST');
    expect(req.request.method).toBe('POST');
    req.flush({});
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET').flush(limitsList([]));
    fixture.detectChanges();
  });

  it('onUpdateLimit fires PUT to exact URL /risk/{name}/limits/{id}', () => {
    settleWithLimits([wireLimit({ id: 'tgt' })]);
    comp.onUpdateLimit({ id: 'tgt', current: 0.1, breached: false });

    const req = http.expectOne((r) => r.url === `${LIMITS_URL}/tgt` && r.method === 'PUT');
    expect(req.request.method).toBe('PUT');
    req.flush({});
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET').flush(limitsList([]));
    fixture.detectChanges();
  });

  it('onDeleteLimit fires DELETE to exact URL /risk/{name}/limits/{id}', () => {
    settleWithLimits([wireLimit({ id: 'del-me' })]);
    comp.onDeleteLimit('del-me');

    const req = http.expectOne((r) => r.url === `${LIMITS_URL}/del-me` && r.method === 'DELETE');
    expect(req.request.method).toBe('DELETE');
    req.flush(null, { status: 204, statusText: 'No Content' });
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET').flush(limitsList([]));
    fixture.detectChanges();
  });

  // ── Reload after each mutation ────────────────────────────────────────────

  it('after a successful create, a GET /limits reload fires automatically', () => {
    settleWithLimits([]);
    comp.onCreateLimit({ metric: 'max_drawdown', limit_type: 'upper', threshold: 0.15 });
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'POST').flush({});

    // Expect the automatic reload
    const reloadReq = http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET');
    reloadReq.flush(limitsList([wireLimit()]));
    fixture.detectChanges();

    expect(comp.limits().length).toBe(1);
  });

  it('after a successful delete, a GET /limits reload fires automatically', () => {
    settleWithLimits([wireLimit({ id: 'to-delete' })]);
    comp.onDeleteLimit('to-delete');
    http.expectOne((r) => r.url === `${LIMITS_URL}/to-delete` && r.method === 'DELETE')
      .flush(null, { status: 204, statusText: 'No Content' });

    // Expect the automatic reload
    const reloadReq = http.expectOne((r) => r.url === LIMITS_URL && r.method === 'GET');
    reloadReq.flush(limitsList([]));
    fixture.detectChanges();

    expect(comp.limits().length).toBe(0);
  });

  // ── Error handling: failed mutation does not corrupt the local state ───────

  it('when create fails with 422, panelErrors["limits"] is set and list is unchanged', () => {
    settleWithLimits([wireLimit()]);
    const countBefore = comp.limits().length;

    comp.onCreateLimit({ metric: 'liquidity_ratio', limit_type: 'lower', threshold: 0.03 });
    http.expectOne((r) => r.url === LIMITS_URL && r.method === 'POST')
      .flush({ detail: 'validation error' }, { status: 422, statusText: 'Unprocessable Entity' });
    fixture.detectChanges();

    expect(comp.panelErrors()['limits'])
      .withContext('A failed create must set panelErrors["limits"]')
      .toBeTruthy();
    expect(comp.limits().length)
      .withContext('List must not change after a failed create')
      .toBe(countBefore);
  });

  it('when delete fails with 500, panelErrors["limits"] is set and list is unchanged', () => {
    settleWithLimits([wireLimit({ id: 'keep-me' })]);
    const countBefore = comp.limits().length;

    comp.onDeleteLimit('keep-me');
    http.expectOne((r) => r.url === `${LIMITS_URL}/keep-me` && r.method === 'DELETE')
      .flush({ detail: 'server error' }, { status: 500, statusText: 'Server Error' });
    fixture.detectChanges();

    expect(comp.panelErrors()['limits'])
      .withContext('A failed delete must set panelErrors["limits"]')
      .toBeTruthy();
    expect(comp.limits().length)
      .withContext('List must not shrink after a failed delete')
      .toBe(countBefore);
  });
});
