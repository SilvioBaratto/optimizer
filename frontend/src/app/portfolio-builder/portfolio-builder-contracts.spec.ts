/**
 * Request-shape contract parity for PortfolioApiService (issue #1000, Scope 12).
 *
 * Asserts HTTP method, exact interpolated URL (path params `{name}`/`{jobId}`
 * encoded, no raw token), and request-body shape for all 10 methods, via the
 * shared request-assertion helper (#998).
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
  type TestRequest,
} from '@angular/common/http/testing';

import { assertBodyKeys, assertMethod, assertUrl } from '../../testing';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import type { CreatePortfolioDto, CreateSnapshotDto } from '../core/models/portfolio-api.model';
import { environment } from '../../environments/environment';

const BASE = `${environment.apiUrl}portfolio`;
const NAME = 'My Port';
const ENC = encodeURIComponent(NAME);
const FOR_NAME = `${BASE}/${ENC}`;
const JOB_ID = 'job 1';
const JOB_ENC = encodeURIComponent(JOB_ID);

describe('PortfolioApiService — request contract parity (issue #1000)', () => {
  let svc: PortfolioApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc = TestBed.inject(PortfolioApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when list is called, a GET to the collection URL is issued', () => {
    svc.list().subscribe({ error: () => {} });
    const req = capture(`${BASE}/`);
    assertMethod(req, 'GET');
    assertUrl(req, `${BASE}/`);
  });

  it('when get is called, a GET to the interpolated portfolio URL is issued', () => {
    svc.get(NAME).subscribe({ error: () => {} });
    const req = capture(FOR_NAME);
    assertMethod(req, 'GET');
    assertUrl(req, FOR_NAME);
  });

  it('when create is called, a POST to the collection URL carries the create payload', () => {
    const body: CreatePortfolioDto = { name: 'New', currency: 'USD', benchmark_ticker: 'SPY' };
    svc.create(body).subscribe({ error: () => {} });
    const req = capture(`${BASE}/`);
    assertMethod(req, 'POST');
    assertUrl(req, `${BASE}/`);
    assertBodyKeys(req, ['name', 'currency', 'benchmark_ticker']);
  });

  it('when getSnapshots is called, a GET to the interpolated snapshots URL is issued', () => {
    svc.getSnapshots(NAME).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/snapshots`);
    assertMethod(req, 'GET');
    assertUrl(req, `${FOR_NAME}/snapshots`);
  });

  it('when createSnapshot is called, a POST to the snapshots URL carries the snapshot payload', () => {
    const body: CreateSnapshotDto = { snapshot_date: '2024-01-01', weights: { AAPL: 1 } };
    svc.createSnapshot(NAME, body).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/snapshots`);
    assertMethod(req, 'POST');
    assertUrl(req, `${FOR_NAME}/snapshots`);
    assertBodyKeys(req, ['snapshot_date', 'weights']);
  });

  it('when getLatestSnapshot is called, a GET to the latest-snapshot URL is issued', () => {
    svc.getLatestSnapshot(NAME).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/snapshots/latest`);
    assertMethod(req, 'GET');
    assertUrl(req, `${FOR_NAME}/snapshots/latest`);
  });

  it('when syncPortfolio is called, a POST to the sync URL is issued', () => {
    svc.syncPortfolio(NAME).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/sync`);
    assertMethod(req, 'POST');
    assertUrl(req, `${FOR_NAME}/sync`);
  });

  it('when getSyncProgress is called, a GET to the interpolated sync-job URL is issued', () => {
    svc.getSyncProgress(NAME, JOB_ID).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/sync/${JOB_ENC}`);
    assertMethod(req, 'GET');
    assertUrl(req, `${FOR_NAME}/sync/${JOB_ENC}`);
  });

  it('when getPositions is called, a GET to the positions URL is issued', () => {
    svc.getPositions(NAME).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/positions`);
    assertMethod(req, 'GET');
    assertUrl(req, `${FOR_NAME}/positions`);
  });

  it('when getAccount is called, a GET to the account URL is issued', () => {
    svc.getAccount(NAME).subscribe({ error: () => {} });
    const req = capture(`${FOR_NAME}/account`);
    assertMethod(req, 'GET');
    assertUrl(req, `${FOR_NAME}/account`);
  });
});
