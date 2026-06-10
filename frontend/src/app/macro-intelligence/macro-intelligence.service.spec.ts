import { TestBed } from '@angular/core/testing';
import type { HttpTestingController } from '@angular/common/http/testing';
import { throwError } from 'rxjs';

import { configureTestBed, injectHttp } from '../../testing';
import { MacroIntelligenceService } from './macro-intelligence.service';
import { environment } from '../../environments/environment';

const API = environment.apiUrl;

// Minimal wire-shape builders (only the fields the mappers read).
function calib(phase: string): Record<string, unknown> {
  return {
    phase,
    delta: 2.5,
    tau: 0.05,
    confidence: 0.8,
    rationale: 'r',
    macroSummary: 's',
    blConfig: { views: [], tau: 0.05, prior_config: { mu_estimator: 'x', risk_aversion: 1, cov_estimator: 'y' } },
  };
}

describe('MacroIntelligenceService', () => {
  let svc: MacroIntelligenceService;
  let http: HttpTestingController;

  beforeEach(async () => {
    await configureTestBed({ withHttp: true });
    svc = TestBed.inject(MacroIntelligenceService);
    http = injectHttp();
  });

  afterEach(() => http.verify());

  describe('getMacroCalibration()', () => {
    it('when called, it GETs views/macro-calibration with country=USA', () => {
      svc.getMacroCalibration().subscribe();
      const req = http.expectOne((r) => r.url === `${API}views/macro-calibration`);
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('country')).toBe('USA');
      expect(req.request.params.get('force_refresh')).toBeNull();
      req.flush(calib('MID_EXPANSION'));
    });

    it('when forceRefresh is true, force_refresh is added to the params', () => {
      svc.getMacroCalibration(true).subscribe();
      const req = http.expectOne((r) => r.url === `${API}views/macro-calibration`);
      expect(req.request.params.get('force_refresh')).toBe('true');
      req.flush(calib('MID_EXPANSION'));
    });

    it('when phase is RECESSION, it is remapped to CONTRACTION', () => {
      let phase: string | undefined;
      svc.getMacroCalibration().subscribe((r) => (phase = r?.phase));
      http.expectOne((r) => r.url === `${API}views/macro-calibration`).flush(calib('RECESSION'));
      expect(phase).toBe('CONTRACTION');
    });

    it('when phase is a valid cycle phase, it passes through unchanged', () => {
      let phase: string | undefined;
      svc.getMacroCalibration().subscribe((r) => (phase = r?.phase));
      http.expectOne((r) => r.url === `${API}views/macro-calibration`).flush(calib('LATE_EXPANSION'));
      expect(phase).toBe('LATE_EXPANSION');
    });

    it('when phase is unrecognised, it falls back to CONTRACTION', () => {
      let phase: string | undefined;
      svc.getMacroCalibration().subscribe((r) => (phase = r?.phase));
      http.expectOne((r) => r.url === `${API}views/macro-calibration`).flush(calib('NONSENSE'));
      expect(phase).toBe('CONTRACTION');
    });

    it('when the request errors, the calibration falls back to null', () => {
      let result: unknown = 'unset';
      svc.getMacroCalibration().subscribe((r) => (result = r));
      http
        .expectOne((r) => r.url === `${API}views/macro-calibration`)
        .flush({ detail: 'down' }, { status: 500, statusText: 'Server Error' });
      expect(result).toBeNull();
    });
  });

  describe('FRED series getters', () => {
    const cases: ReadonlyArray<[keyof MacroIntelligenceService, string]> = [
      ['getFredYieldSpread', 'T10Y2Y'],
      ['getFredHyOas', 'BAMLH0A0HYM2'],
      ['getFredVix', 'VIXCLS'],
      ['getFredIgOas', 'BAMLC0A0CM'],
    ];

    for (const [method, seriesId] of cases) {
      it(`when ${String(method)} is called, it GETs fred/series with series_id=${seriesId} and drops null values`, () => {
        let rows: Array<{ date: string; value: number }> = [];
        (svc[method] as () => { subscribe: (fn: (v: typeof rows) => void) => void })().subscribe((v) => (rows = v));
        const req = http.expectOne((r) => r.url === `${API}macro-data/fred/series`);
        expect(req.request.method).toBe('GET');
        expect(req.request.params.get('series_id')).toBe(seriesId);
        expect(req.request.params.get('limit')).toBe('500');
        expect(req.request.params.get('start_date')).toBeTruthy();
        req.flush([
          { id: '1', series_id: seriesId, date: '2026-01-01', value: 1.2, created_at: '', updated_at: '' },
          { id: '2', series_id: seriesId, date: '2026-02-01', value: null, created_at: '', updated_at: '' },
        ]);
        expect(rows).toEqual([{ date: '2026-01-01', value: 1.2 }]);
      });
    }

    it('when the series request errors, an empty list is returned', () => {
      let rows: unknown = 'unset';
      svc.getFredVix().subscribe((v) => (rows = v));
      http
        .expectOne((r) => r.url === `${API}macro-data/fred/series`)
        .flush('boom', { status: 500, statusText: 'Server Error' });
      expect(rows).toEqual([]);
    });
  });

  describe('getFredPmi()', () => {
    it('when called, it GETs te-observations for manufacturing_pmi and drops null rows', () => {
      let rows: Array<{ date: string; value: number }> = [];
      svc.getFredPmi().subscribe((v) => (rows = v));
      const req = http.expectOne((r) => r.url === `${API}macro-data/te-observations`);
      expect(req.request.params.get('country')).toBe('USA');
      expect(req.request.params.get('indicator_keys')).toBe('manufacturing_pmi');
      expect(req.request.params.get('limit')).toBe('500');
      req.flush([
        { id: '1', country: 'USA', indicator_key: 'manufacturing_pmi', value: 51, date: '2026-01-01', created_at: '', updated_at: '' },
        { id: '2', country: 'USA', indicator_key: 'manufacturing_pmi', value: null, date: '2026-02-01', created_at: '', updated_at: '' },
      ]);
      expect(rows).toEqual([{ date: '2026-01-01', value: 51 }]);
    });

    it('when the request errors, an empty list is returned', () => {
      let rows: unknown = 'unset';
      svc.getFredPmi().subscribe((v) => (rows = v));
      http.expectOne((r) => r.url === `${API}macro-data/te-observations`).flush('e', { status: 500, statusText: 'x' });
      expect(rows).toEqual([]);
    });
  });

  describe('getNews()', () => {
    it('when no theme is given, it GETs news with limit=50 and no theme param', () => {
      svc.getNews().subscribe();
      const req = http.expectOne((r) => r.url === `${API}macro-data/news`);
      expect(req.request.method).toBe('GET');
      expect(req.request.params.get('limit')).toBe('50');
      expect(req.request.params.get('theme')).toBeNull();
      req.flush([]);
    });

    it('when a theme is given, it is added to the params', () => {
      svc.getNews('inflation').subscribe();
      const req = http.expectOne((r) => r.url === `${API}macro-data/news`);
      expect(req.request.params.get('theme')).toBe('inflation');
      req.flush([]);
    });

    it('when rows are returned, they are mapped with safe fallbacks', () => {
      let items: Array<Record<string, string>> = [];
      svc.getNews().subscribe((v) => (items = v as never));
      http.expectOne((r) => r.url === `${API}macro-data/news`).flush([
        { id: 'x', news_id: 'n1', title: 'T', snippet: null, link: null, publisher: null, publish_time: null, themes: null, source_ticker: null, source_query: null, full_content: null, created_at: '', updated_at: '' },
      ]);
      expect(items[0]).toEqual({ id: 'n1', headline: 'T', snippet: '', url: '#', publisher: '', published_at: '', theme: '' });
    });

    it('when the request errors, an empty list is returned', () => {
      let items: unknown = 'unset';
      svc.getNews().subscribe((v) => (items = v));
      http.expectOne((r) => r.url === `${API}macro-data/news`).flush('e', { status: 500, statusText: 'x' });
      expect(items).toEqual([]);
    });
  });

  describe('getNewsThemes()', () => {
    it('when called, it GETs news/themes and maps value→id with count 0', () => {
      let themes: Array<Record<string, unknown>> = [];
      svc.getNewsThemes().subscribe((v) => (themes = v as never));
      http.expectOne((r) => r.url === `${API}macro-data/news/themes`).flush([{ value: 'infl', label: 'Inflation' }]);
      expect(themes).toEqual([{ id: 'infl', label: 'Inflation', count: 0 }]);
    });

    it('when the request errors, an empty list is returned', () => {
      let themes: unknown = 'unset';
      svc.getNewsThemes().subscribe((v) => (themes = v));
      http.expectOne((r) => r.url === `${API}macro-data/news/themes`).flush('e', { status: 500, statusText: 'x' });
      expect(themes).toEqual([]);
    });
  });

  describe('news summaries', () => {
    it('when getCountrySummaries is called, it GETs news/summaries and passes the list through', () => {
      let result: unknown[] = [];
      svc.getCountrySummaries().subscribe((v) => (result = v));
      http.expectOne((r) => r.url === `${API}macro-data/news/summaries`).flush([{ id: 's1' }]);
      expect(result.length).toBe(1);
    });

    it('when getCountrySummaries errors, an empty list is returned', () => {
      let result: unknown = 'unset';
      svc.getCountrySummaries().subscribe((v) => (result = v));
      http.expectOne((r) => r.url === `${API}macro-data/news/summaries`).flush('e', { status: 500, statusText: 'x' });
      expect(result).toEqual([]);
    });

    it('when getCountrySummary is called, the country code maps to its DB name', () => {
      svc.getCountrySummary('US').subscribe();
      const req = http.expectOne((r) => r.url === `${API}macro-data/news/summaries/USA`);
      expect(req.request.method).toBe('GET');
      req.flush({ id: 's1' });
    });

    it('when getCountrySummary errors, null is returned', () => {
      let result: unknown = 'unset';
      svc.getCountrySummary('US').subscribe((v) => (result = v));
      http.expectOne((r) => r.url === `${API}macro-data/news/summaries/USA`).flush('e', { status: 404, statusText: 'x' });
      expect(result).toBeNull();
    });
  });

  describe('bond yields', () => {
    it('when getBondYieldsToday is called, rows are grouped per country with the mapped code', () => {
      let snaps: Array<{ country: string; curve: unknown[] }> = [];
      svc.getBondYieldsToday().subscribe((v) => (snaps = v as never));
      http.expectOne((r) => r.url === `${API}macro-data/bond-yields`).flush([
        { id: '1', country: 'USA', maturity: '10Y', yield_value: 4, day_change: null, month_change: null, year_change: null, reference_date: null, created_at: '', updated_at: '' },
        { id: '2', country: 'USA', maturity: '2Y', yield_value: 3, day_change: null, month_change: null, year_change: null, reference_date: null, created_at: '', updated_at: '' },
      ]);
      expect(snaps.length).toBe(1);
      expect(snaps[0].country).toBe('US');
      expect(snaps[0].curve).toEqual([
        { maturity: '2Y', yield_pct: 3 },
        { maturity: '10Y', yield_pct: 4 },
      ]);
    });

    it('when getBondYieldsToday errors, an empty list is returned', () => {
      let snaps: unknown = 'unset';
      svc.getBondYieldsToday().subscribe((v) => (snaps = v));
      http.expectOne((r) => r.url === `${API}macro-data/bond-yields`).flush('e', { status: 500, statusText: 'x' });
      expect(snaps).toEqual([]);
    });

    it('when getBondYields1YAgo is called, it GETs observations with an ISO date window', () => {
      svc.getBondYields1YAgo().subscribe();
      const req = http.expectOne((r) => r.url === `${API}macro-data/bond-yield-observations`);
      expect(req.request.params.get('start_date')).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(req.request.params.get('end_date')).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      req.flush([]);
    });

    it('when getBondYields1YAgo returns rows, it dedupes and groups per country', () => {
      let snaps: Array<{ country: string; curve: unknown[] }> = [];
      svc.getBondYields1YAgo().subscribe((v) => (snaps = v as never));
      http.expectOne((r) => r.url === `${API}macro-data/bond-yield-observations`).flush([
        { id: '1', country: 'USA', maturity: '10Y', date: '2025-01-01', yield_value: 3.9, created_at: '', updated_at: '' },
        { id: '2', country: 'USA', maturity: '10Y', date: '2025-01-03', yield_value: 4, created_at: '', updated_at: '' },
        { id: '3', country: 'USA', maturity: '2Y', date: '2025-01-03', yield_value: 3, created_at: '', updated_at: '' },
      ]);
      expect(snaps.length).toBe(1);
      expect(snaps[0].country).toBe('US');
      expect(snaps[0].curve).toEqual([
        { maturity: '2Y', yield_pct: 3 },
        { maturity: '10Y', yield_pct: 4 },
      ]);
    });

    it('when getBondYields1YAgo errors, an empty list is returned', () => {
      let snaps: unknown = 'unset';
      svc.getBondYields1YAgo().subscribe((v) => (snaps = v));
      http.expectOne((r) => r.url === `${API}macro-data/bond-yield-observations`).flush('e', { status: 500, statusText: 'x' });
      expect(snaps).toEqual([]);
    });
  });

  describe('getCountryData()', () => {
    it('when called, it fetches every DB country and filters failed ones', () => {
      let data: Array<Record<string, number>> = [];
      svc.getCountryData().subscribe((v) => (data = v as never));

      const usa = http.expectOne((r) => r.url === `${API}macro-data/countries/USA`);
      usa.flush({
        country: 'USA',
        economic_indicators: [],
        te_indicators: [{ id: 't', country: 'USA', indicator_key: 'gdp_growth_rate', value: 2.5, created_at: '', updated_at: '' }],
        bond_yields: [
          { id: 'b1', country: 'USA', maturity: '10Y', yield_value: 4, day_change: null, month_change: null, year_change: null, reference_date: null, created_at: '', updated_at: '' },
          { id: 'b2', country: 'USA', maturity: '2Y', yield_value: 3, day_change: null, month_change: null, year_change: null, reference_date: null, created_at: '', updated_at: '' },
        ],
      });
      // Remaining three countries error → filtered out.
      for (const c of ['Germany', 'France', 'UK']) {
        http.expectOne((r) => r.url === `${API}macro-data/countries/${c}`).flush('e', { status: 500, statusText: 'x' });
      }

      expect(data.length).toBe(1);
      expect(data[0]['gdp_growth']).toBe(2.5);
      expect(data[0]['yield_spread_bps']).toBe(100);
    });

    it('when all country requests return null, filter removes them and the result is empty', () => {
      // All four per-country HTTP calls fail → each catchError(of(null)) emits null →
      // forkJoin resolves with [null, null, null, null] → filter removes all → [].
      // (The outer catchError at line 130 is a purely defensive guard; this test
      // confirms the filter path that precedes it.)
      let data: unknown = 'unset';
      svc.getCountryData().subscribe((v) => (data = v));

      for (const c of ['USA', 'Germany', 'France', 'UK']) {
        http.expectOne((r) => r.url === `${API}macro-data/countries/${c}`)
          .flush('e', { status: 500, statusText: 'x' });
      }

      expect(data).toEqual([]);
    });
  });

  describe('getCompositeHistory()', () => {
    it('when both FRED series resolve, a per-month composite score is computed', () => {
      let points: Array<{ month: string; score: number }> = [];
      svc.getCompositeHistory().subscribe((v) => (points = v));

      const reqs = http.match((r) => r.url === `${API}macro-data/fred/series`);
      expect(reqs.length).toBe(2);
      // Order: getFredYieldSpread (T10Y2Y) then getFredHyOas (BAMLH0A0HYM2).
      reqs[0].flush([{ id: '1', series_id: 'T10Y2Y', date: '2026-01-15', value: 1.5, created_at: '', updated_at: '' }]);
      reqs[1].flush([{ id: '2', series_id: 'BAMLH0A0HYM2', date: '2026-01-20', value: 300, created_at: '', updated_at: '' }]);

      expect(points).toEqual([{ month: '2026-01', score: 2 }]);
    });

    it('when both FRED series error, the composite history is empty', () => {
      let points: unknown = 'unset';
      svc.getCompositeHistory().subscribe((v) => (points = v));
      const reqs = http.match((r) => r.url === `${API}macro-data/fred/series`);
      reqs[0].flush('e', { status: 500, statusText: 'x' });
      reqs[1].flush('e', { status: 500, statusText: 'x' });
      expect(points).toEqual([]);
    });

    it('when spread is below YIELD_SPREAD_BEAR, the score decrements by 1', () => {
      // Exercises the else-if (spreadVal < YIELD_SPREAD_BEAR) branch (source line 273).
      let points: Array<{ month: string; score: number }> = [];
      svc.getCompositeHistory().subscribe((v) => (points = v));

      const reqs = http.match((r) => r.url === `${API}macro-data/fred/series`);
      // spreadVal = -0.5 < YIELD_SPREAD_BEAR (0) → score -= 1.
      // hyVal = 600 > HY_OAS_BEAR (500) → score -= 1.  Total: -2.
      reqs[0].flush([{ id: '1', series_id: 'T10Y2Y', date: '2026-02-15', value: -0.5, created_at: '', updated_at: '' }]);
      reqs[1].flush([{ id: '2', series_id: 'BAMLH0A0HYM2', date: '2026-02-20', value: 600, created_at: '', updated_at: '' }]);

      expect(points).toEqual([{ month: '2026-02', score: -2 }]);
    });

    it('when spread is between thresholds, neither spread branch fires and score remains 0', () => {
      // Exercises the else-path of the else-if on source line 273 (neutral spread).
      // hyVal between BULL and BEAR → neither HY branch fires either.
      let points: Array<{ month: string; score: number }> = [];
      svc.getCompositeHistory().subscribe((v) => (points = v));

      const reqs = http.match((r) => r.url === `${API}macro-data/fred/series`);
      // spreadVal = 0.5: 0 <= 0.5 <= 1.0 → neither > BULL nor < BEAR.
      // hyVal = 400: 350 < 400 < 500 → neither < BULL nor > BEAR. score = 0.
      reqs[0].flush([{ id: '1', series_id: 'T10Y2Y', date: '2026-03-15', value: 0.5, created_at: '', updated_at: '' }]);
      reqs[1].flush([{ id: '2', series_id: 'BAMLH0A0HYM2', date: '2026-03-20', value: 400, created_at: '', updated_at: '' }]);

      expect(points).toEqual([{ month: '2026-03', score: 0 }]);
    });

    it('when getFredYieldSpread itself errors, the outer catchError returns an empty list', () => {
      // Spy on the public getFredYieldSpread to return a plain throwError — bypassing
      // its own inner catchError — so the forkJoin inside getCompositeHistory propagates
      // the error to the outer catchError at source line 281.
      spyOn(svc, 'getFredYieldSpread').and.returnValue(throwError(() => new Error('inject')));

      let points: unknown = 'unset';
      svc.getCompositeHistory().subscribe((v) => (points = v));
      // getFredHyOas still fires one HTTP request — drain it so verify() passes.
      http.match((r) => r.url === `${API}macro-data/fred/series`).forEach(r => r.flush([]));

      expect(points).toEqual([]);
    });
  });

  describe('triggerRefresh()', () => {
    // interval(3000) is setInterval-backed; jasmine.clock drives it (zoneless).
    it('when called, it fires the 3 fetch jobs, polls them, then summarizes', () => {
      jasmine.clock().install();
      try {
        let done = false;
        svc.triggerRefresh().subscribe(() => (done = true));

        const fetchPaths = ['macro-data/fetch', 'macro-data/fred/fetch', 'macro-data/news/fetch'];
        for (const path of fetchPaths) {
          http
            .expectOne((r) => r.url === `${API}${path}` && r.method === 'POST')
            .flush({ job_id: 'j1', status: 'pending', message: '' });
        }

        jasmine.clock().tick(3000);
        for (const path of fetchPaths) {
          http
            .expectOne((r) => r.url === `${API}${path}/j1`)
            .flush({ job_id: 'j1', status: 'completed', current: 1, total: 1, errors: [], result: null, error: null });
        }

        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize` && r.method === 'POST')
          .flush({ job_id: 's1', status: 'pending', message: '' });
        jasmine.clock().tick(3000);
        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize/s1`)
          .flush({ job_id: 's1', status: 'completed', current: 1, total: 1, errors: [], result: null, error: null });

        expect(done).toBe(true);
      } finally {
        jasmine.clock().uninstall();
      }
    });

    const FETCH_PATHS = ['macro-data/fetch', 'macro-data/fred/fetch', 'macro-data/news/fetch'];

    function completed(): Record<string, unknown> {
      return { job_id: 'x', status: 'completed', current: 1, total: 1, errors: [], result: null, error: null };
    }

    it('when a fetch POST errors, its job is skipped and surviving jobs still poll', () => {
      jasmine.clock().install();
      try {
        let done = false;
        svc.triggerRefresh().subscribe(() => (done = true));

        // First fetch POST fails → catchError(of(null)) → filtered out of polls.
        http
          .expectOne((r) => r.url === `${API}macro-data/fetch` && r.method === 'POST')
          .flush('e', { status: 500, statusText: 'x' });
        http
          .expectOne((r) => r.url === `${API}macro-data/fred/fetch` && r.method === 'POST')
          .flush({ job_id: 'j2', status: 'pending', message: '' });
        http
          .expectOne((r) => r.url === `${API}macro-data/news/fetch` && r.method === 'POST')
          .flush({ job_id: 'j3', status: 'pending', message: '' });

        jasmine.clock().tick(3000);
        http.expectOne((r) => r.url === `${API}macro-data/fred/fetch/j2`).flush(completed());
        http.expectOne((r) => r.url === `${API}macro-data/news/fetch/j3`).flush(completed());

        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize` && r.method === 'POST')
          .flush({ job_id: 's1', status: 'pending', message: '' });
        jasmine.clock().tick(3000);
        http.expectOne((r) => r.url === `${API}macro-data/news/summarize/s1`).flush(completed());

        expect(done).toBe(true);
      } finally {
        jasmine.clock().uninstall();
      }
    });

    it('when all three fetch POSTs error, polling is skipped and summarize still runs', () => {
      jasmine.clock().install();
      try {
        let done = false;
        svc.triggerRefresh().subscribe(() => (done = true));

        for (const path of FETCH_PATHS) {
          http
            .expectOne((r) => r.url === `${API}${path}` && r.method === 'POST')
            .flush('e', { status: 500, statusText: 'x' });
        }

        // polls$ is empty → the switchMap returns of(undefined) and summarize runs.
        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize` && r.method === 'POST')
          .flush({ job_id: 's1', status: 'pending', message: '' });
        jasmine.clock().tick(3000);
        http.expectOne((r) => r.url === `${API}macro-data/news/summarize/s1`).flush(completed());

        expect(done).toBe(true);
      } finally {
        jasmine.clock().uninstall();
      }
    });

    it('when a fetch poll reports status failed, the poll terminates and refresh completes', () => {
      jasmine.clock().install();
      try {
        let done = false;
        svc.triggerRefresh().subscribe(() => (done = true));

        for (const path of FETCH_PATHS) {
          http
            .expectOne((r) => r.url === `${API}${path}` && r.method === 'POST')
            .flush({ job_id: 'j1', status: 'pending', message: '' });
        }

        jasmine.clock().tick(3000);
        // status:'failed' is terminal too → takeWhile(..., true) emits and completes.
        for (const path of FETCH_PATHS) {
          http
            .expectOne((r) => r.url === `${API}${path}/j1`)
            .flush({ job_id: 'j1', status: 'failed', current: 0, total: 1, errors: ['x'], result: null, error: 'fail' });
        }

        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize` && r.method === 'POST')
          .flush({ job_id: 's1', status: 'pending', message: '' });
        jasmine.clock().tick(3000);
        http.expectOne((r) => r.url === `${API}macro-data/news/summarize/s1`).flush(completed());

        expect(done).toBe(true);
      } finally {
        jasmine.clock().uninstall();
      }
    });

    it('when the summarize POST errors, the refresh completes with the error swallowed', () => {
      jasmine.clock().install();
      try {
        let done = false;
        svc.triggerRefresh().subscribe(() => (done = true));

        for (const path of FETCH_PATHS) {
          http
            .expectOne((r) => r.url === `${API}${path}` && r.method === 'POST')
            .flush({ job_id: 'j1', status: 'pending', message: '' });
        }
        jasmine.clock().tick(3000);
        for (const path of FETCH_PATHS) {
          http.expectOne((r) => r.url === `${API}${path}/j1`).flush(completed());
        }

        // summarize POST errors → catchError(of(undefined)) → refresh still completes.
        http
          .expectOne((r) => r.url === `${API}macro-data/news/summarize` && r.method === 'POST')
          .flush('e', { status: 500, statusText: 'x' });

        expect(done).toBe(true);
      } finally {
        jasmine.clock().uninstall();
      }
    });
  });
});
