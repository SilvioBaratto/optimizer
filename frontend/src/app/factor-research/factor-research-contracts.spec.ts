/**
 * Source-blind contract spec for FactorResearchComponent (issue #955).
 *
 * Authored from acceptance criteria. Covers T3-verifiable items only.
 *
 * Criteria:
 *
 * T3-a — Init state:
 *   • computeJobId() null, computeError() null on mount
 *   • no POST to /factors/compute on init
 *
 * POST body contract:
 *   • snake_case keys: tickers[], start_date, end_date
 *   • response job_id (snake_case) → computeJobId() set
 *   • guard: second call while running → no second request
 *
 * onComputeCompleted fix (the bug):
 *   • must GET /factors/compute/{jobId} to retrieve result (currently no-op → RED)
 *   • clears computeJobId after polling
 *
 * T3-b — Visible error state:
 *   • 5xx → computeError() truthy, [role="alert"] visible with non-blank text
 *   • onComputeFailed clears computeJobId + sets computeError
 *
 * T3-c — Portfolio context → tickers in POST body:
 *   • portfolio selected → snapshot seeded → POST body carries portfolio tickers
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';

import { configureTestBed, injectHttp, installResizeObserverStub } from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { PortfolioContextService } from '../core/services/portfolio-context.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;
const COMPUTE_URL = `${API}factors/compute`;
const PORTFOLIO_LIST_URL = `${API}portfolio/`;

function snapshotUrl(name: string): string {
  return `${API}portfolio/${encodeURIComponent(name)}/snapshots/latest`;
}

function buildPortfolioList(portfolios: { id: string; name: string }[]) {
  return {
    items: portfolios.map((p) => ({
      id: p.id,
      name: p.name,
      description: null,
      currency: 'USD',
      benchmark_ticker: 'SPY',
    })),
    total: portfolios.length,
  };
}

function buildSnapshotDto(weights: Record<string, number>) {
  return {
    id: 'snap-fr-c',
    portfolio_id: 'pf-fr-c',
    snapshot_date: '2026-01-01',
    snapshot_type: 'manual',
    weights,
    sector_mapping: null,
    summary: null,
    optimizer_config: null,
    turnover: null,
    holding_count: Object.keys(weights).length,
    created_at: '2026-01-01T00:00:00Z',
  };
}

// Async response uses snake_case job_id — /factors/compute is NOT a CamelCaseModel endpoint.
const ASYNC_RESPONSE = { job_id: 'fc-1', status: 'pending', message: '' };

const COMPUTE_PROGRESS = {
  job_id: 'fc-1',
  status: 'completed' as const,
  current: 10,
  total: 10,
  errors: [] as string[],
  result: {
    ic_report: [
      {
        factor: 'momentum_12_1',
        ic: 0.05,
        icir: 0.8,
        t_stat: 2.1,
        p_value: 0.04,
        vif: 1.2,
        significant: true,
      },
    ],
  },
  error: null,
};

describe('FactorResearchComponent — compute contracts (issue #955)', () => {
  let fixture: ComponentFixture<FactorResearchComponent>;
  let comp: FactorResearchComponent;
  let http: HttpTestingController;
  let ctx: PortfolioContextService;

  beforeEach(async () => {
    localStorage.clear();
    installResizeObserverStub();
    await configureTestBed({
      imports: [FactorResearchComponent],
      withHttp: true,
      providers: [ICON_PROVIDER],
    });
    fixture = TestBed.createComponent(FactorResearchComponent);
    comp = fixture.componentInstance;
    http = injectHttp();
    ctx = TestBed.inject(PortfolioContextService);
    fixture.detectChanges();
  });

  afterEach(() => {
    // Drain the PortfolioContextService auto-select bootstrap (GET /portfolio/)
    // that fires when no portfolio id is stored, so verify() only asserts on the
    // requests each test explicitly exercises.
    http.match((r) => r.method === 'GET' && r.url.endsWith('/portfolio/'));
    http.verify();
    localStorage.clear();
  });

  // ── T3-a: Init state ──────────────────────────────────────────────────────────

  it('on init, computeJobId is null', () => {
    expect(comp.computeJobId()).toBeNull();
  });

  it('on init, computeError is null', () => {
    expect(comp.computeError()).toBeNull();
  });

  it('on init, no POST to /factors/compute is made', () => {
    http.expectNone((r) => r.url.includes('factors/compute') && r.method === 'POST');
    expect(comp.computeJobId()).toBeNull();
  });

  // ── POST body shape ───────────────────────────────────────────────────────────

  it('POSTs to /factors/compute with snake_case keys: tickers, start_date, end_date', () => {
    comp.onRunCompute();

    const req = http.expectOne(COMPUTE_URL);
    expect(req.request.method).toBe('POST');
    req.flush(ASYNC_RESPONSE);

    const body = req.request.body as Record<string, unknown>;
    expect(Array.isArray(body['tickers'])).toBe(true);
    expect(typeof body['start_date']).toBe('string');
    expect(typeof body['end_date']).toBe('string');
  });

  it('POST body tickers array is non-empty', () => {
    comp.onRunCompute();

    const req = http.expectOne(COMPUTE_URL);
    req.flush(ASYNC_RESPONSE);

    const body = req.request.body as Record<string, unknown>;
    expect((body['tickers'] as string[]).length).toBeGreaterThan(0);
  });

  it('POST body has no camelCase key leakage', () => {
    comp.onRunCompute();

    const req = http.expectOne(COMPUTE_URL);
    req.flush(ASYNC_RESPONSE);

    const body = req.request.body as Record<string, unknown>;
    expect(body['startDate']).toBeUndefined();
    expect(body['endDate']).toBeUndefined();
  });

  it('POST body start_date and end_date are ISO date strings', () => {
    comp.onRunCompute();

    const req = http.expectOne(COMPUTE_URL);
    req.flush(ASYNC_RESPONSE);

    const body = req.request.body as Record<string, unknown>;
    const iso = /^\d{4}-\d{2}-\d{2}/;
    expect(iso.test(body['start_date'] as string))
      .withContext('start_date should be ISO date')
      .toBe(true);
    expect(iso.test(body['end_date'] as string))
      .withContext('end_date should be ISO date')
      .toBe(true);
  });

  // ── Async response → computeJobId set ────────────────────────────────────────

  it('response job_id (snake_case) sets computeJobId', () => {
    comp.onRunCompute();

    http.expectOne(COMPUTE_URL).flush({ job_id: 'fc-job-42', status: 'pending', message: '' });

    expect(comp.computeJobId()).toBe('fc-job-42');
  });

  it('second onRunCompute while job running makes no additional request', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    expect(comp.computeJobId()).toBeTruthy();

    comp.onRunCompute();
    http.expectNone((r) => r.url.includes('factors/compute') && r.method === 'POST');
    expect(comp.computeJobId()).toBe('fc-1');
  });

  // ── onComputeCompleted → fetch result (bug: currently no-op → RED) ────────────

  it('onComputeCompleted GETs /factors/compute/{jobId} to retrieve result', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    expect(comp.computeJobId()).toBe('fc-1');

    comp.onComputeCompleted();

    const poll = http.expectOne(`${COMPUTE_URL}/fc-1`);
    expect(poll.request.method).toBe('GET');
    poll.flush(COMPUTE_PROGRESS);

    expect(comp.computeJobId()).toBeNull();
  });

  // ── T3-b: Visible error state ─────────────────────────────────────────────────

  it('when POST returns 5xx, computeError is set', () => {
    comp.onRunCompute();

    http.expectOne(COMPUTE_URL).flush(
      { detail: 'Server error' },
      { status: 500, statusText: 'Internal Server Error' },
    );

    expect(comp.computeError()).toBeTruthy();
  });

  it('when POST returns 5xx, [role="alert"] is visible with non-blank text', () => {
    comp.onRunCompute();

    http.expectOne(COMPUTE_URL).flush(
      { detail: 'compute crashed' },
      { status: 500, statusText: 'Internal Server Error' },
    );
    fixture.detectChanges();

    const alertEl = fixture.nativeElement.querySelector('[role="alert"]') as HTMLElement | null;
    expect(alertEl)
      .withContext('expected [role="alert"] element after compute error')
      .toBeTruthy();
    expect(alertEl?.textContent?.trim().length)
      .withContext('expected non-blank error text in alert')
      .toBeGreaterThan(0);
  });

  it('when POST returns 4xx, computeError is set and computeJobId is null', () => {
    comp.onRunCompute();

    http.expectOne(COMPUTE_URL).flush(
      { detail: 'bad request' },
      { status: 400, statusText: 'Bad Request' },
    );

    expect(comp.computeError()).toBeTruthy();
    expect(comp.computeJobId()).toBeNull();
  });

  // ── onComputeFailed ───────────────────────────────────────────────────────────

  it('onComputeFailed clears computeJobId and sets computeError to message', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);
    expect(comp.computeJobId()).toBe('fc-1');

    comp.onComputeFailed('GPU OOM');

    expect(comp.computeJobId()).toBeNull();
    expect(comp.computeError()).toBe('GPU OOM');
  });

  it('onComputeFailed with empty message stores a non-empty fallback error string', () => {
    comp.onRunCompute();
    http.expectOne(COMPUTE_URL).flush(ASYNC_RESPONSE);

    comp.onComputeFailed('');

    expect((comp.computeError() ?? '').length).toBeGreaterThan(0);
  });

  // ── T3-c: Portfolio context → tickers in POST body ────────────────────────────

  it('when portfolio is selected, snapshot tickers appear in the POST body', () => {
    ctx.currentPortfolioId.set('pf-research');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-research', name: 'research-fund' }]));
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === snapshotUrl('research-fund'))
      .flush(buildSnapshotDto({ TSLA: 0.5, NVDA: 0.5 }));
    fixture.detectChanges();

    expect(comp.tickers()).toEqual(jasmine.arrayContaining(['TSLA', 'NVDA']));

    comp.onRunCompute();

    const req = http.expectOne(COMPUTE_URL);
    req.flush(ASYNC_RESPONSE);

    const tickers = (req.request.body as Record<string, unknown>)['tickers'] as string[];
    expect(tickers).toContain('TSLA');
    expect(tickers).toContain('NVDA');
  });
});
