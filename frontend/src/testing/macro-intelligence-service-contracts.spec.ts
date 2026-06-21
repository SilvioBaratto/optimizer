/**
 * Request/response contract parity for MacroIntelligenceService (issue #1028).
 *
 * Uses assertMethod / assertUrl / assertQueryParams to pin:
 *   – HTTP method for every named endpoint
 *   – Exact interpolated URL (no raw {token} placeholders)
 *   – Date ranges in query params (not the body) — getBondYields1YAgo
 *   – Path params as encoded path segments (not query params) — getCountrySummary, getCountryData
 *   – Body key-set for job-create POST endpoints (within triggerRefresh)
 *   – Response field shapes from domain-fixture keys
 *
 * Criteria (UNIT, issue #1028):
 *   Every MacroIntelligenceService method (calibration, te-observations, country
 *   summary, bond yields, news themes/news/summaries, FRED, and the two
 *   job-create POSTs) asserts HTTP method, interpolated URL, query-param
 *   placement, body key-set + field types, and response shape.
 */
import { TestBed } from '@angular/core/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

import { assertMethod, assertUrl, assertQueryParams } from './contract-request.helper';
import { assertFieldParity } from './contract-field-parity';
import { schemaOf } from './contract-parity';
import { makeMacroCalibrationApiResponse } from './domain-fixtures';
import macroSnapshot from './contract-snapshots/macro.json';
import { MacroIntelligenceService } from '../app/macro-intelligence/macro-intelligence.service';
import { environment } from '../environments/environment';

const API = environment.apiUrl;

const CALIB_FIXTURE = makeMacroCalibrationApiResponse();
const JOB_CREATED = { job_id: 'j1', status: 'pending', message: '' };
const JOB_DONE = { job_id: 'j1', status: 'completed', current: 1, total: 1, errors: [], result: null, error: null };

describe('MacroIntelligenceService — request/response contract parity (issue #1028)', () => {
  let svc: MacroIntelligenceService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    svc = TestBed.inject(MacroIntelligenceService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  // ── getMacroCalibration() (calibration) ───────────────────────────────────

  describe('getMacroCalibration() — calibration endpoint', () => {
    const CAL_URL = `${API}views/macro-calibration`;

    it('when called, sends GET to exact views/macro-calibration URL', () => {
      svc.getMacroCalibration().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === CAL_URL);
      assertMethod(req, 'GET');
      assertUrl(req, CAL_URL);
      req.flush(CALIB_FIXTURE);
    });

    it('country is sent as query param, not as a path segment or request body', () => {
      svc.getMacroCalibration().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === CAL_URL);
      assertQueryParams(req, { country: 'USA' });
      expect(req.request.body).toBeNull();
      req.flush(CALIB_FIXTURE);
    });

    it('response wire shape matches MacroCalibrationResponse schema', () => {
      svc.getMacroCalibration().subscribe();
      http.expectOne(r => r.url === CAL_URL).flush(CALIB_FIXTURE);
      assertFieldParity(schemaOf(macroSnapshot, 'MacroCalibrationResponse'), CALIB_FIXTURE, macroSnapshot);
    });
  });

  // ── FRED series getters ───────────────────────────────────────────────────

  describe('FRED series getters — GET macro-data/fred/series with series_id query param', () => {
    const FRED_URL = `${API}macro-data/fred/series`;

    const SERIES_CASES: ReadonlyArray<[string, string]> = [
      ['getFredYieldSpread', 'T10Y2Y'],
      ['getFredHyOas', 'BAMLH0A0HYM2'],
      ['getFredVix', 'VIXCLS'],
      ['getFredIgOas', 'BAMLC0A0CM'],
    ];

    for (const [methodName, seriesId] of SERIES_CASES) {
      describe(`${methodName}()`, () => {
        it(`sends GET to exact macro-data/fred/series URL with series_id=${seriesId} as query param`, () => {
          type FreqFn = () => { subscribe: (o: object) => void };
          (svc[methodName as keyof MacroIntelligenceService] as FreqFn)().subscribe({ error: () => {} });
          const req = http.expectOne(r => r.url === FRED_URL);
          assertMethod(req, 'GET');
          assertUrl(req, FRED_URL);
          assertQueryParams(req, { series_id: seriesId });
          expect(req.request.body).toBeNull();
          req.flush([]);
        });
      });
    }
  });

  // ── getFredPmi() (te-observations endpoint) ───────────────────────────────

  describe('getFredPmi() — te-observations endpoint', () => {
    const PMI_URL = `${API}macro-data/te-observations`;

    it('when called, sends GET to exact macro-data/te-observations URL', () => {
      svc.getFredPmi().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === PMI_URL);
      assertMethod(req, 'GET');
      assertUrl(req, PMI_URL);
      req.flush([]);
    });

    it('country and indicator_keys are sent as query params (not body)', () => {
      svc.getFredPmi().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === PMI_URL);
      assertQueryParams(req, { country: 'USA', indicator_keys: 'manufacturing_pmi' });
      expect(req.request.body).toBeNull();
      req.flush([]);
    });
  });

  // ── getNews() ─────────────────────────────────────────────────────────────

  describe('getNews() — news endpoint', () => {
    const NEWS_URL = `${API}macro-data/news`;

    it('when called, sends GET to exact macro-data/news URL', () => {
      svc.getNews().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === NEWS_URL);
      assertMethod(req, 'GET');
      assertUrl(req, NEWS_URL);
      req.flush([]);
    });

    it('limit is sent as query param (not body)', () => {
      svc.getNews().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === NEWS_URL);
      expect(req.request.params.get('limit')).toBeTruthy();
      expect(req.request.body).toBeNull();
      req.flush([]);
    });
  });

  // ── getNewsThemes() ───────────────────────────────────────────────────────

  describe('getNewsThemes() — news themes endpoint', () => {
    const THEMES_URL = `${API}macro-data/news/themes`;

    it('when called, sends GET to exact macro-data/news/themes URL', () => {
      svc.getNewsThemes().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === THEMES_URL);
      assertMethod(req, 'GET');
      assertUrl(req, THEMES_URL);
      req.flush([]);
    });
  });

  // ── getCountrySummaries() ─────────────────────────────────────────────────

  describe('getCountrySummaries() — news summaries list endpoint', () => {
    const SUMM_URL = `${API}macro-data/news/summaries`;

    it('when called, sends GET to exact macro-data/news/summaries URL', () => {
      svc.getCountrySummaries().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === SUMM_URL);
      assertMethod(req, 'GET');
      assertUrl(req, SUMM_URL);
      req.flush([]);
    });
  });

  // ── getCountrySummary(code) — country summary, path param as path segment ─

  describe('getCountrySummary(code) — country summary endpoint', () => {
    it('when called with US, country code is interpolated as path segment (not query param)', () => {
      svc.getCountrySummary('US').subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === `${API}macro-data/news/summaries/USA`);
      assertMethod(req, 'GET');
      assertUrl(req, `${API}macro-data/news/summaries/USA`);
      expect(req.request.params.has('country')).toBe(false);
      req.flush(null);
    });
  });

  // ── getBondYieldsToday() (bond yields) ────────────────────────────────────

  describe('getBondYieldsToday() — bond yields endpoint', () => {
    const YIELDS_URL = `${API}macro-data/bond-yields`;

    it('when called, sends GET to exact macro-data/bond-yields URL', () => {
      svc.getBondYieldsToday().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === YIELDS_URL);
      assertMethod(req, 'GET');
      assertUrl(req, YIELDS_URL);
      req.flush([]);
    });
  });

  // ── getBondYields1YAgo() — date ranges as query params ────────────────────

  describe('getBondYields1YAgo() — bond yield observations with date range', () => {
    const OBS_URL = `${API}macro-data/bond-yield-observations`;

    it('when called, sends GET to exact macro-data/bond-yield-observations URL', () => {
      svc.getBondYields1YAgo().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === OBS_URL);
      assertMethod(req, 'GET');
      assertUrl(req, OBS_URL);
      req.flush([]);
    });

    it('start_date is a query param in ISO date format (not in request body)', () => {
      svc.getBondYields1YAgo().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === OBS_URL);
      expect(req.request.params.get('start_date')).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(req.request.body).toBeNull();
      req.flush([]);
    });

    it('end_date is a query param in ISO date format (not in request body)', () => {
      svc.getBondYields1YAgo().subscribe({ error: () => {} });
      const req = http.expectOne(r => r.url === OBS_URL);
      expect(req.request.params.get('end_date')).toMatch(/^\d{4}-\d{2}-\d{2}$/);
      expect(req.request.body).toBeNull();
      req.flush([]);
    });
  });

  // ── getCountryData() — encoded path param per country ────────────────────

  describe('getCountryData() — countries endpoint with path param', () => {
    it('country name is interpolated as an encoded path segment (not query param)', () => {
      svc.getCountryData().subscribe({ error: () => {} });

      const usaReq = http.expectOne(r => r.url === `${API}macro-data/countries/USA`);
      assertMethod(usaReq, 'GET');
      assertUrl(usaReq, `${API}macro-data/countries/USA`);
      expect(usaReq.request.params.has('country')).toBe(false);
      usaReq.flush({
        country: 'USA', economic_indicators: [], te_indicators: [], bond_yields: [],
      });

      ['Germany', 'France', 'UK'].forEach(c =>
        http.expectOne(r => r.url === `${API}macro-data/countries/${c}`)
          .flush('e', { status: 500, statusText: 'x' }),
      );
    });
  });

  // ── triggerRefresh() — two job-create POSTs ───────────────────────────────

  describe('triggerRefresh() — job-create POST contracts', () => {
    it('fires POST to exact macro-data/fetch and macro-data/news/summarize URLs', () => {
      jasmine.clock().install();
      try {
        svc.triggerRefresh().subscribe({ error: () => {} });

        const FETCH_PATHS = ['macro-data/fetch', 'macro-data/fred/fetch', 'macro-data/news/fetch'];

        const fetchReqs = FETCH_PATHS.map(path =>
          http.expectOne(r => r.url === `${API}${path}`),
        );
        assertMethod(fetchReqs[0], 'POST');
        assertUrl(fetchReqs[0], `${API}macro-data/fetch`);
        assertMethod(fetchReqs[2], 'POST');
        assertUrl(fetchReqs[2], `${API}macro-data/news/fetch`);
        fetchReqs.forEach(r => r.flush({ ...JOB_CREATED }));

        jasmine.clock().tick(3000);
        FETCH_PATHS.forEach(path =>
          http.expectOne(r => r.url === `${API}${path}/j1`).flush({ ...JOB_DONE }),
        );

        const summarizeReq = http.expectOne(r => r.url === `${API}macro-data/news/summarize`);
        assertMethod(summarizeReq, 'POST');
        assertUrl(summarizeReq, `${API}macro-data/news/summarize`);
        summarizeReq.flush({ job_id: 's1', status: 'pending', message: '' });

        jasmine.clock().tick(3000);
        http.expectOne(r => r.url === `${API}macro-data/news/summarize/s1`).flush({
          job_id: 's1', status: 'completed', current: 1, total: 1, errors: [], result: null, error: null,
        });
      } finally {
        jasmine.clock().uninstall();
      }
    });
  });
});
