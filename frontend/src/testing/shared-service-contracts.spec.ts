/**
 * Request-shape contract parity for the shared core services (issue #1001, Scope 12).
 *
 * Covers every method of UniverseService, MarketService, and JobsService —
 * asserting HTTP method, exact interpolated URL (path param `{id}` encoded, no raw
 * token), query-parameter placement (in `req.request.params`, not the body), and
 * request body — via the shared request-assertion helper (#998).
 */
import { TestBed } from '@angular/core/testing';
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
} from './contract-request.helper';
import { UniverseService } from '../app/core/services/universe.service';
import { MarketService } from '../app/core/services/market.service';
import { JobsService, type ListJobsQuery } from '../app/core/services/jobs.service';
import type { UniverseScreenRequest } from '../app/core/models/universe.model';
import { environment } from '../environments/environment';

const API = environment.apiUrl;

function setupHttp(): { http: HttpTestingController } {
  TestBed.configureTestingModule({
    providers: [
      provideZonelessChangeDetection(),
      provideHttpClient(),
      provideHttpClientTesting(),
    ],
  });
  return { http: TestBed.inject(HttpTestingController) };
}

function capture(http: HttpTestingController, url: string, flushBody: object = {}): TestRequest {
  const req = http.expectOne((r) => r.url === url);
  req.flush(flushBody);
  return req;
}

// ── UniverseService ──────────────────────────────────────────────────────────

describe('UniverseService — request contract parity (issue #1001)', () => {
  let svc: UniverseService;
  let http: HttpTestingController;
  const BUILD_ID = 'b 1';
  const BUILD_ENC = encodeURIComponent(BUILD_ID);

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(UniverseService);
  });

  afterEach(() => http.verify());

  it('when getStats is called, a GET to /universe/stats is issued', () => {
    svc.getStats().subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/stats`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}universe/stats`);
  });

  it('when getExchanges is called, a GET to /universe/exchanges is issued', () => {
    svc.getExchanges().subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/exchanges`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}universe/exchanges`);
  });

  it('when getInstruments is called, a GET carries page/page_size/search/exchange in params', () => {
    svc.getInstruments({ page: 1, page_size: 15, search: 'AAPL', exchange: 'NASDAQ' })
      .subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/instruments`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}universe/instruments`);
    assertQueryParams(req, { page: 1, page_size: 15, search: 'AAPL', exchange: 'NASDAQ' });
  });

  it('when buildUniverse is called, a POST to /universe/build with an empty body is issued', () => {
    svc.buildUniverse().subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/build`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}universe/build`);
    assertBodyKeys(req, []);
  });

  it('when getBuildProgress is called, a GET to the interpolated /universe/build/{id} URL is issued', () => {
    svc.getBuildProgress(BUILD_ID).subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/build/${BUILD_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}universe/build/${BUILD_ENC}`);
  });

  it('when screen is called, a POST to /universe/screen carries the screen request body', () => {
    const body: UniverseScreenRequest = { preset: 'broad_universe', min_trading_history: 252 };
    svc.screen(body).subscribe({ error: () => {} });
    const req = capture(http, `${API}universe/screen`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}universe/screen`);
    assertBodyKeys(req, ['preset', 'min_trading_history']);
  });
});

// ── MarketService ────────────────────────────────────────────────────────────

describe('MarketService — request contract parity (issue #1001)', () => {
  let svc: MarketService;
  let http: HttpTestingController;

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(MarketService);
  });

  afterEach(() => http.verify());

  it('when getIndices is called, a GET to /market/indices is issued', () => {
    svc.getIndices().subscribe({ error: () => {} });
    const req = capture(http, `${API}market/indices`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}market/indices`);
  });
});

// ── JobsService ──────────────────────────────────────────────────────────────

describe('JobsService — request contract parity (issue #1001)', () => {
  let svc: JobsService;
  let http: HttpTestingController;
  const JOB_ID = 'j 1';
  const JOB_ENC = encodeURIComponent(JOB_ID);

  beforeEach(() => {
    ({ http } = setupHttp());
    svc = TestBed.inject(JobsService);
  });

  afterEach(() => http.verify());

  it('when getDomainStatuses is called, a GET to /jobs carries limit=100 in params', () => {
    svc.getDomainStatuses().subscribe({ error: () => {} });
    const req = capture(http, `${API}jobs`, { jobs: [] });
    assertMethod(req, 'GET');
    assertUrl(req, `${API}jobs`);
    assertQueryParams(req, { limit: 100 });
  });

  it('when getJob is called, a GET to the interpolated /jobs/{id} URL is issued', () => {
    svc.getJob(JOB_ID).subscribe({ error: () => {} });
    const req = capture(http, `${API}jobs/${JOB_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${API}jobs/${JOB_ENC}`);
  });

  it('when listJobs is called, a GET to /jobs carries limit/offset/domain/status in params', () => {
    const query: ListJobsQuery = {
      limit: 20,
      offset: 40,
      domain: ['market_data'],
      status: ['completed'],
    };
    svc.listJobs(query).subscribe({ error: () => {} });
    const req = capture(http, `${API}jobs`, { jobs: [] });
    assertMethod(req, 'GET');
    assertUrl(req, `${API}jobs`);
    assertQueryParams(req, { limit: 20, offset: 40, domain: 'market_data', status: 'completed' });
  });
});
