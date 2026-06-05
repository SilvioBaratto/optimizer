import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { YfinanceService } from './yfinance.service';
import { environment } from '../../environments/environment';
import {
  AnalystPriceTarget,
  AnalystRecommendation,
  Dividend,
  FinancialStatement,
  InsiderTransaction,
  InstitutionalHolder,
  MutualFundHolder,
  PriceHistory,
  StockSplit,
  TickerNews,
  TickerProfile,
} from '../models/yfinance.model';

const BASE = `${environment.apiUrl}yfinance-data/instruments`;
const ID = 'instrument-123';

describe('YfinanceService', () => {
  let svc: YfinanceService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        YfinanceService,
      ],
    });
    svc = TestBed.inject(YfinanceService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    http.verify();
  });

  // ─── Single-item endpoints ─────────────────────────────────────────────

  describe('getProfile()', () => {
    it('calls GET /{id}/profile and returns the payload', () => {
      const payload = { id: 'p1', instrument_id: ID, sector: 'Technology' } as TickerProfile;
      let result: TickerProfile | null | undefined;
      svc.getProfile(ID).subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/${ID}/profile`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('falls back to null on HTTP error', () => {
      let result: TickerProfile | null | undefined = undefined;
      svc.getProfile(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/profile`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toBeNull();
    });
  });

  describe('getPriceTargets()', () => {
    it('calls GET /{id}/price-targets and returns the payload', () => {
      const payload: AnalystPriceTarget = {
        id: 'pt1',
        instrument_id: ID,
        current: 200,
        low: 180,
        high: 230,
        mean: 210,
        median: 208,
        created_at: '',
        updated_at: '',
      };
      let result: AnalystPriceTarget | null | undefined;
      svc.getPriceTargets(ID).subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/${ID}/price-targets`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('returns null on HTTP error (single-item fallback)', () => {
      let result: AnalystPriceTarget | null | undefined = undefined;
      svc.getPriceTargets(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/price-targets`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toBeNull();
    });
  });

  // ─── List endpoints ────────────────────────────────────────────────────

  describe('getPrices()', () => {
    it('passes start_date, end_date, limit as query params', () => {
      svc
        .getPrices(ID, { startDate: '2026-01-01', endDate: '2026-02-01', limit: 100 })
        .subscribe();

      const req = http.expectOne(
        (r) => r.url === `${BASE}/${ID}/prices`,
      );
      expect(req.request.params.get('start_date')).toBe('2026-01-01');
      expect(req.request.params.get('end_date')).toBe('2026-02-01');
      expect(req.request.params.get('limit')).toBe('100');
      req.flush([]);
    });

    it('calls /prices without params when none are provided', () => {
      svc.getPrices(ID).subscribe();

      const req = http.expectOne((r) => r.url === `${BASE}/${ID}/prices`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush([]);
    });

    it('falls back to [] on HTTP error', () => {
      let result: PriceHistory[] | undefined;
      svc.getPrices(ID).subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${BASE}/${ID}/prices`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getFinancials()', () => {
    it('passes statement_type and period_type as query params', () => {
      svc
        .getFinancials(ID, { statementType: 'income', periodType: 'quarterly' })
        .subscribe();

      const req = http.expectOne(
        (r) => r.url === `${BASE}/${ID}/financials`,
      );
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('statement_type')).toBe('income');
      expect(req.request.params.get('period_type')).toBe('quarterly');
      req.flush([]);
    });

    it('when called with no query, sends no query params', () => {
      svc.getFinancials(ID).subscribe();
      const req = http.expectOne((r) => r.url === `${BASE}/${ID}/financials`);
      expect(req.request.params.keys().length).toBe(0);
      req.flush([]);
    });

    it('when getFinancials encounters an HTTP error, falls back to []', () => {
      let result: FinancialStatement[] | undefined;
      svc.getFinancials(ID).subscribe((r) => (result = r));

      http
        .expectOne((r) => r.url === `${BASE}/${ID}/financials`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getDividends()', () => {
    it('calls /dividends and returns the list', () => {
      const payload: Dividend[] = [
        { id: 'd1', instrument_id: ID, date: '2026-01-01', amount: 0.24, created_at: '', updated_at: '' },
      ];
      let result: Dividend[] | undefined;
      svc.getDividends(ID).subscribe((r) => (result = r));

      const req = http.expectOne(`${BASE}/${ID}/dividends`);
      expect(req.request.method).toBe('GET');
      req.flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getDividends encounters an HTTP error, falls back to []', () => {
      let result: Dividend[] | undefined;
      svc.getDividends(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/dividends`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('other list endpoints just hit the right URL', () => {
    const cases: [string, keyof YfinanceService, string][] = [
      ['splits', 'getSplits', 'splits'],
      ['recommendations', 'getRecommendations', 'recommendations'],
      ['institutional-holders', 'getInstitutionalHolders', 'institutional-holders'],
      ['mutualfund-holders', 'getMutualfundHolders', 'mutualfund-holders'],
      ['insider-transactions', 'getInsiderTransactions', 'insider-transactions'],
      ['news', 'getNews', 'news'],
    ];

    for (const [name, method, path] of cases) {
      it(`${name} → GET /${path}`, () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (svc[method] as (id: string) => { subscribe: (fn: () => void) => void })(ID)
          .subscribe(() => {});
        const req = http.expectOne(`${BASE}/${ID}/${path}`);
        expect(req.request.method).toBe('GET');
        req.flush([]);
      });
    }
  });

  describe('getSplits()', () => {
    it('when getSplits returns data, maps the payload', () => {
      const payload: StockSplit[] = [
        { id: 's1', instrument_id: ID, date: '2020-08-31', ratio: 4, created_at: '', updated_at: '' },
      ];
      let result: StockSplit[] | undefined;
      svc.getSplits(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/splits`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getSplits encounters an HTTP error, falls back to []', () => {
      let result: StockSplit[] | undefined;
      svc.getSplits(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/splits`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getRecommendations()', () => {
    it('when getRecommendations returns data, maps the payload', () => {
      const payload: AnalystRecommendation[] = [
        {
          id: 'r1',
          instrument_id: ID,
          period: '2026-04',
          strong_buy: 10,
          buy: 5,
          hold: 3,
          sell: 1,
          strong_sell: 0,
          created_at: '',
          updated_at: '',
        },
      ];
      let result: AnalystRecommendation[] | undefined;
      svc.getRecommendations(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/recommendations`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getRecommendations encounters an HTTP error, falls back to []', () => {
      let result: AnalystRecommendation[] | undefined;
      svc.getRecommendations(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/recommendations`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getInstitutionalHolders()', () => {
    it('when getInstitutionalHolders returns data, maps the payload', () => {
      const payload: InstitutionalHolder[] = [
        {
          id: 'ih1',
          instrument_id: ID,
          holder_name: 'Vanguard',
          date_reported: '2026-03-31',
          shares: 1000000,
          value: 180000000,
          pct_held: 0.06,
          created_at: '',
          updated_at: '',
        },
      ];
      let result: InstitutionalHolder[] | undefined;
      svc.getInstitutionalHolders(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/institutional-holders`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getInstitutionalHolders encounters an HTTP error, falls back to []', () => {
      let result: InstitutionalHolder[] | undefined;
      svc.getInstitutionalHolders(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/institutional-holders`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getMutualfundHolders()', () => {
    it('when getMutualfundHolders returns data, maps the payload', () => {
      const payload: MutualFundHolder[] = [
        {
          id: 'mf1',
          instrument_id: ID,
          holder_name: 'Fidelity 500',
          date_reported: '2026-03-31',
          shares: 500000,
          value: 90000000,
          pct_held: 0.03,
          created_at: '',
          updated_at: '',
        },
      ];
      let result: MutualFundHolder[] | undefined;
      svc.getMutualfundHolders(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/mutualfund-holders`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getMutualfundHolders encounters an HTTP error, falls back to []', () => {
      let result: MutualFundHolder[] | undefined;
      svc.getMutualfundHolders(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/mutualfund-holders`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getInsiderTransactions()', () => {
    it('when getInsiderTransactions returns data, maps the payload', () => {
      const payload: InsiderTransaction[] = [
        {
          id: 'it1',
          instrument_id: ID,
          insider_name: 'Tim Cook',
          position: 'CEO',
          transaction_type: 'Sale',
          shares: 50000,
          value: 9000000,
          start_date: '2026-02-15',
          ownership: 'D',
          created_at: '',
          updated_at: '',
        },
      ];
      let result: InsiderTransaction[] | undefined;
      svc.getInsiderTransactions(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/insider-transactions`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getInsiderTransactions encounters an HTTP error, falls back to []', () => {
      let result: InsiderTransaction[] | undefined;
      svc.getInsiderTransactions(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/insider-transactions`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });

  describe('getNews()', () => {
    it('when getNews returns data, maps the payload', () => {
      const payload: TickerNews[] = [
        {
          id: 'n1',
          instrument_id: ID,
          news_uuid: 'uuid-1',
          title: 'Apple tops estimates',
          publisher: 'Reuters',
          link: 'https://reuters.com/a',
          publish_time: '2026-05-01T10:00:00Z',
          news_type: 'STORY',
          ticker_name: 'AAPL',
          full_content: null,
          created_at: '',
          updated_at: '',
        },
      ];
      let result: TickerNews[] | undefined;
      svc.getNews(ID).subscribe((r) => (result = r));

      http.expectOne(`${BASE}/${ID}/news`).flush(payload);

      expect(result).toEqual(payload);
    });

    it('when getNews encounters an HTTP error, falls back to []', () => {
      let result: TickerNews[] | undefined;
      svc.getNews(ID).subscribe((r) => (result = r));

      http
        .expectOne(`${BASE}/${ID}/news`)
        .error(new ProgressEvent('network'), { status: 500 });

      expect(result).toEqual([]);
    });
  });
});
