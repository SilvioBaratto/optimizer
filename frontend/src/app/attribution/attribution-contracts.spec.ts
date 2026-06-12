/**
 * Request-shape contract parity for AttributionService (issue #999, Scope 12).
 *
 * Both methods are synchronous POSTs whose weights are carried in the body (the
 * backend exposes no GET-by-name attribution endpoints). Asserts HTTP method,
 * exact URL, and the request body field-by-field via the #998 helper.
 */
import { TestBed } from '@angular/core/testing';
import type { HttpTestingController, TestRequest } from '@angular/common/http/testing';

import {
  assertBodyFieldTypes,
  assertBodyKeys,
  assertMethod,
  assertUrl,
  configureTestBed,
  injectHttp,
} from '../../testing';
import { AttributionService } from './attribution.service';
import type { BrinsonApiRequest, FactorAttributionApiRequest } from './attribution.model';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

describe('AttributionService — request contract parity (issue #999)', () => {
  let svc: AttributionService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(AttributionService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  function capture(url: string): TestRequest {
    const req = http.expectOne((r) => r.url === url);
    req.flush({});
    return req;
  }

  it('when brinson is called, a POST carries portfolio/benchmark weights and the date range field-by-field', () => {
    const body: BrinsonApiRequest = {
      portfolio_weights: { AAPL: 0.6, MSFT: 0.4 },
      benchmark_weights: { SPY: 1 },
      start_date: '2024-01-01',
      end_date: '2024-12-31',
    };
    svc.brinson(body).subscribe({ error: () => {} });
    const req = capture(`${API}attribution/brinson`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}attribution/brinson`);
    assertBodyKeys(req, ['portfolio_weights', 'benchmark_weights', 'start_date', 'end_date']);
    assertBodyFieldTypes(req, {
      portfolio_weights: 'object',
      benchmark_weights: 'object',
      start_date: 'string',
      end_date: 'string',
    });
  });

  it('when factor is called, a POST carries portfolio weights and the date range field-by-field', () => {
    const body: FactorAttributionApiRequest = {
      portfolio_weights: { AAPL: 0.6, MSFT: 0.4 },
      start_date: '2024-01-01',
      end_date: '2024-12-31',
    };
    svc.factor(body).subscribe({ error: () => {} });
    const req = capture(`${API}attribution/factor`);
    assertMethod(req, 'POST');
    assertUrl(req, `${API}attribution/factor`);
    assertBodyKeys(req, ['portfolio_weights', 'start_date', 'end_date']);
    assertBodyFieldTypes(req, {
      portfolio_weights: 'object',
      start_date: 'string',
      end_date: 'string',
    });
  });
});
