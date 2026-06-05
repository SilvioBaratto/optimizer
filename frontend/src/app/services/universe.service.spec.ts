import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { Subject, Subscription } from 'rxjs';

import {
  UniverseService,
  SEARCH_DEBOUNCE_MS,
} from './universe.service';
import { environment } from '../../environments/environment';
import {
  BuildProgress,
  Exchange,
  Instrument,
  InstrumentList,
  UniverseScreenRequest,
  UniverseScreenResponse,
  UniverseStats,
} from '../models/universe.model';

const API = environment.apiUrl;

function instrument(overrides: Partial<Instrument> = {}): Instrument {
  return {
    id: 'i-1',
    ticker: 'AAPL',
    short_name: 'AAPL',
    name: 'Apple',
    isin: 'US0378331005',
    instrument_type: 'STOCK',
    currency_code: 'USD',
    yfinance_ticker: 'AAPL',
    exchange_name: 'NASDAQ',
    ...overrides,
  };
}

describe('UniverseService', () => {
  let svc: UniverseService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: SEARCH_DEBOUNCE_MS, useValue: 0 },
        UniverseService,
      ],
    });
    svc = TestBed.inject(UniverseService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  describe('getStats()', () => {
    it('calls GET /universe/stats and returns the payload', () => {
      const payload: UniverseStats = {
        total_exchanges: 10,
        total_instruments: 500,
        last_updated: '2026-04-01',
      };
      let result: UniverseStats | undefined;
      svc.getStats().subscribe((s) => (result = s));

      const req = http.expectOne(`${API}universe/stats`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('falls back to a zeroed stats object on error', () => {
      let result: UniverseStats | undefined;
      svc.getStats().subscribe((s) => (result = s));

      http
        .expectOne(`${API}universe/stats`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toBeDefined();
      expect(result!.total_exchanges).toBe(0);
      expect(result!.total_instruments).toBe(0);
    });
  });

  describe('getExchanges()', () => {
    it('calls GET /universe/exchanges and returns the list', () => {
      const payload: Exchange[] = [
        { id: 'e1', name: 'NASDAQ', mic: 'XNAS', country: 'US', instrument_count: 300 },
      ];
      let result: Exchange[] | undefined;
      svc.getExchanges().subscribe((x) => (result = x));

      http.expectOne(`${API}universe/exchanges`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('falls back to an empty array on error', () => {
      let result: Exchange[] | undefined;
      svc.getExchanges().subscribe((x) => (result = x));

      http
        .expectOne(`${API}universe/exchanges`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getInstruments()', () => {
    it('passes search, exchange, page and page_size as query params', () => {
      svc
        .getInstruments({ page: 2, page_size: 25, search: 'AAPL', exchange: 'NASDAQ' })
        .subscribe();

      const req = http.expectOne((r) => r.url === `${API}universe/instruments`);
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('page')).toBe('2');
      expect(req.request.params.get('page_size')).toBe('25');
      expect(req.request.params.get('search')).toBe('AAPL');
      expect(req.request.params.get('exchange')).toBe('NASDAQ');
      req.flush({ items: [], total: 0, page: 2, page_size: 25 });
    });

    it('when called with no query fields, sends no query params', () => {
      svc.getInstruments({}).subscribe();
      const req = http.expectOne((r) => r.url === `${API}universe/instruments`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush({ items: [], total: 0, page: 1, page_size: 15 });
    });

    it('falls back to an empty page on error', () => {
      let result: InstrumentList | undefined;
      svc.getInstruments({}).subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${API}universe/instruments`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result?.items).toEqual([]);
      expect(result?.total).toBe(0);
    });
  });

  describe('buildUniverse()', () => {
    it('calls POST /universe/build and returns the job_id', () => {
      let result: { job_id: string } | undefined;
      svc.buildUniverse().subscribe((r) => (result = r));

      const req = http.expectOne(`${API}universe/build`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({});
      req.flush({ job_id: 'abc', build_id: 'abc', status: 'pending', message: '' });

      expect(result?.job_id).toBe('abc');
    });

    it('rethrows errors so callers can handle them (does NOT fall back)', () => {
      let error: unknown;
      svc.buildUniverse().subscribe({
        error: (e) => (error = e),
      });

      http
        .expectOne(`${API}universe/build`)
        .flush({ detail: 'already running' }, { status: 409, statusText: 'Conflict' });

      expect(error).toBeDefined();
    });
  });

  describe('getBuildProgress()', () => {
    it('calls GET /universe/build/{id} and returns the progress payload', () => {
      const payload: BuildProgress = {
        job_id: 'abc',
        status: 'running',
        current: 3,
        total: 10,
        detail: 'Fetching NASDAQ',
      };
      let result: BuildProgress | undefined;
      svc.getBuildProgress('abc').subscribe((p) => (result = p));

      const req = http.expectOne(`${API}universe/build/abc`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('when job id is missing, propagates 404', () => {
      let error: unknown;
      svc.getBuildProgress('missing').subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}universe/build/missing`)
        .flush({ detail: 'job not found' }, { status: 404, statusText: 'Not Found' });

      expect(error).toBeDefined();
    });

    it('URI-encodes the build id in the path', () => {
      let result: BuildProgress | undefined;
      svc.getBuildProgress('a b').subscribe((p) => (result = p));
      const req = http.expectOne(`${API}universe/build/a%20b`);
      expect(req.request.url).toContain('a%20b');
      req.flush({
        job_id: 'a b',
        status: 'pending',
        current: 0,
        total: 0,
        detail: '',
      });
      expect(result?.job_id).toBe('a b');
    });
  });

  describe('screen()', () => {
    it('POSTs the InvestabilityScreenConfig payload and returns the screen response', () => {
      const request: UniverseScreenRequest = {
        preset: 'developed_markets',
        current_members: ['AAPL', 'MSFT'],
      };
      const payload: UniverseScreenResponse = {
        passingTickers: ['AAPL', 'MSFT', 'NVDA'],
        totalScreened: 500,
        diagnostics: { AAPL: { market_cap: true, addv_3m: true } },
      };

      let result: UniverseScreenResponse | undefined;
      svc.screen(request).subscribe((r) => (result = r));

      const req = http.expectOne(`${API}universe/screen`);
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual(request);
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('propagates 422 errors from /universe/screen', () => {
      let error: unknown;
      svc.screen({}).subscribe({ error: (e) => (error = e) });

      http
        .expectOne(`${API}universe/screen`)
        .flush(
          { detail: 'invalid config' },
          { status: 422, statusText: 'Unprocessable' },
        );

      expect(error).toBeDefined();
    });
  });

  describe('searchTickers()', () => {
    it('cancels superseded queries — only the latest debounced value triggers a request', (done) => {
      const subj = new Subject<string>();
      const results: InstrumentList[] = [];
      const sub: Subscription = svc.searchTickers(subj.asObservable()).subscribe((r) => {
        results.push(r);
      });

      subj.next('A');
      subj.next('AP');
      subj.next('APPL');

      // With SEARCH_DEBOUNCE_MS=0, the debounce fires on the next microtask.
      setTimeout(() => {
        const reqs = http.match((r) => r.url === `${API}universe/instruments`);
        expect(reqs.length).toBe(1);
        expect(reqs[0].request.params.get('search')).toBe('APPL');
        reqs[0].flush({
          items: [instrument()],
          total: 1,
          page: 1,
          page_size: 15,
        });

        expect(results.length).toBe(1);
        expect(results[0].items[0].ticker).toBe('AAPL');
        sub.unsubscribe();
        done();
      }, 10);
    });
  });
});
