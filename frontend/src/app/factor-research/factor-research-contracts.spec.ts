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
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import {
  assertBodyKeys,
  assertMethod,
  assertQueryParams,
  assertUrl,
  configureTestBed,
  injectHttp,
  installResizeObserverStub,
} from '../../testing';
import { ICON_PROVIDER } from '../icons';
import { FactorResearchComponent } from './factor-research';
import { FactorsService } from './factors.service';
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

  it('POST body has exactly the FactorComputeRequest keys (no extra keys)', () => {
    comp.onRunCompute();
    const req = http.expectOne(COMPUTE_URL);
    expect(Object.keys(req.request.body as object).sort()).toEqual([
      'end_date',
      'start_date',
      'tickers',
    ]);
    req.flush(ASYNC_RESPONSE);
  });

  // ── AC4: active-portfolio reference is shown on the page ──────────────────────

  it('when no portfolio is selected, a non-blank active-portfolio reference is shown', () => {
    const ref = fixture.nativeElement.querySelector(
      '[data-testid="active-portfolio"]',
    ) as HTMLElement | null;
    expect(ref)
      .withContext('expected a [data-testid="active-portfolio"] reference element')
      .toBeTruthy();
    expect(ref?.textContent?.trim().length)
      .withContext('active-portfolio reference must not be blank')
      .toBeGreaterThan(0);
  });

  it('when a portfolio is selected, its name appears in the active-portfolio reference', () => {
    ctx.currentPortfolioId.set('pf-fr');
    fixture.detectChanges();

    http
      .expectOne((r) => r.method === 'GET' && r.url === PORTFOLIO_LIST_URL)
      .flush(buildPortfolioList([{ id: 'pf-fr', name: 'research-fund' }]));
    fixture.detectChanges();

    // Selecting a portfolio triggers ticker seeding (latest-snapshot fetch); drain it.
    http
      .match((r) => r.url.includes('snapshots/latest'))
      .forEach((r) => r.flush(buildSnapshotDto({ AAPL: 1 })));
    fixture.detectChanges();

    const ref = fixture.nativeElement.querySelector(
      '[data-testid="active-portfolio"]',
    ) as HTMLElement | null;
    expect(ref?.textContent).toContain('research-fund');
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

// ── Service-level request contract parity (issue #1000) ──────────────────────
// Covers every FactorsService method's method + URL (path params encoded) + query
// + body. Uses a plain (non-intercepted) HTTP testbed so `macroCalibration`, which
// tags its request with the background HttpContext, still reaches the test backend.
describe('FactorsService — request contract parity (issue #1000)', () => {
  let svc: FactorsService;
  let http: HttpTestingController;
  const JOB_ID = 'job 1';
  const JOB_ENC = encodeURIComponent(JOB_ID);
  const FWD = { tickers: ['AAPL'], start_date: '2024-01-01', end_date: '2024-12-31' };

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc = TestBed.inject(FactorsService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  function postForwards(
    call: () => void,
    url: string,
  ): void {
    call();
    const req = capture(url);
    assertMethod(req, 'POST');
    assertUrl(req, url);
    expect(req.request.body).toEqual(FWD);
  }

  it('when compute is called, a POST to /factors/compute carries the body', () => {
    postForwards(
      () => svc.compute(FWD as unknown as Parameters<typeof svc.compute>[0]).subscribe({ error: () => {} }),
      `${API}factors/compute`,
    );
  });

  it('when pollCompute is called, a GET to the interpolated /factors/compute/{id} URL is issued', () => {
    svc.pollCompute(JOB_ID).subscribe({ error: () => {} });
    const req = capture(`${API}factors/compute/${JOB_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}factors/compute/${JOB_ENC}`);
  });

  it('when validate is called, a POST to /factors/validate carries the body', () => {
    postForwards(
      () => svc.validate(FWD as unknown as Parameters<typeof svc.validate>[0]).subscribe({ error: () => {} }),
      `${API}factors/validate`,
    );
  });

  it('when score is called, a POST to /factors/score carries the body', () => {
    postForwards(
      () => svc.score(FWD as unknown as Parameters<typeof svc.score>[0]).subscribe({ error: () => {} }),
      `${API}factors/score`,
    );
  });

  it('when select is called, a POST to /factors/select carries the body', () => {
    postForwards(
      () => svc.select(FWD as unknown as Parameters<typeof svc.select>[0]).subscribe({ error: () => {} }),
      `${API}factors/select`,
    );
  });

  it('when exposureConstraints is called, a POST to /factors/exposure-constraints carries the body', () => {
    postForwards(
      () => svc.exposureConstraints(FWD as unknown as Parameters<typeof svc.exposureConstraints>[0]).subscribe({ error: () => {} }),
      `${API}factors/exposure-constraints`,
    );
  });

  it('when quintileSpread is called, a POST to /factors/quintile-spread carries the body', () => {
    postForwards(
      () => svc.quintileSpread(FWD as unknown as Parameters<typeof svc.quintileSpread>[0]).subscribe({ error: () => {} }),
      `${API}factors/quintile-spread`,
    );
  });

  it('when regimeTilt is called, a POST to /factors/regime-tilt carries the body', () => {
    postForwards(
      () => svc.regimeTilt(FWD as unknown as Parameters<typeof svc.regimeTilt>[0]).subscribe({ error: () => {} }),
      `${API}factors/regime-tilt`,
    );
  });

  it('when macroCalibration is called, a GET to /views/macro-calibration carries query params (not body)', () => {
    svc.macroCalibration({ country: 'US', macroText: 'recession', refresh: true }).subscribe({ error: () => {} });
    const req = capture(`${API}views/macro-calibration`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}views/macro-calibration`);
    assertQueryParams(req, { country: 'US', macro_text: 'recession', refresh: true });
  });

  it('when teObservations is called, a GET to /macro-data/te-observations carries query params', () => {
    svc.teObservations({ country: 'US', startDate: '2024-01-01', endDate: '2024-12-31', limit: 50 })
      .subscribe({ error: () => {} });
    const req = capture(`${API}macro-data/te-observations`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}macro-data/te-observations`);
    assertQueryParams(req, { country: 'US', start_date: '2024-01-01', end_date: '2024-12-31', limit: 50 });
  });
});
