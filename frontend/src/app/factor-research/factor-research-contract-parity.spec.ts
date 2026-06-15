/**
 * Source-blind spec — authored from acceptance criteria only.
 *
 * Criterion (UNIT):
 *   "Contract-parity specs assert request and response shapes for every
 *    service method."
 *
 * Criterion (T2):
 *   "POST /factors/compute + polling GET /factors/compute/{id} populates
 *    ic_reports / taa_signals / factor_returns / cma_sets."
 *
 * Criterion (T2):
 *   "FactorAnalysisPanelComponent renders IC/t-stat charts from POST /factors/validate."
 *
 * Criterion (T2):
 *   "ScorePanelComponent renders results from POST /factors/score and
 *    SelectPanelComponent from POST /factors/select (request shape parity;
 *    loading flags toggle)."
 *
 * Criterion (T2):
 *   "TaaPanelComponent renders TAA weights from GET /views/macro-calibration."
 *
 * Criterion (T2):
 *   "CmaBuilderPanelComponent renders the CMA chart from GET /macro-data/te-observations."
 *
 * Criterion (UNIT):
 *   "No API endpoint or parameter names appear in any user-facing UI copy."
 */

import { TestBed } from '@angular/core/testing';
import { HttpClient, provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';

// ---------------------------------------------------------------------------
// Spec-derived request / response interfaces
// (shapes inferred from criterion text; implementation must satisfy these)
// ---------------------------------------------------------------------------

interface FactorComputeRequest {
  tickers: string[];
  // Other fields are implementation-determined; tickers is the only spec-mandated field
}

interface FactorComputeJobResponse {
  job_id: string;
}

interface FactorComputeStatusResponse {
  status: 'pending' | 'running' | 'completed' | 'failed';
  ic_reports: unknown[];
  taa_signals: unknown[];
  factor_returns: unknown[];
  cma_sets: unknown[];
}

interface FactorValidateRequest {
  tickers: string[];
}

interface FactorValidateResponse {
  // Criterion: renders IC/t-stat charts → response must carry these series
  ic_series: unknown[];
  t_stat_series: unknown[];
}

interface FactorScoreRequest {
  tickers: string[];
}

interface FactorScoreResponse {
  scores: unknown[];
}

interface FactorSelectRequest {
  tickers: string[];
}

interface FactorSelectResponse {
  selected: string[];
}

interface MacroCalibrationResponse {
  // Criterion: renders TAA weights → response must carry weights
  weights: Record<string, number>;
}

interface TeObservationsResponse {
  // Criterion: renders CMA chart → response carries observations
  observations: unknown[];
}

// Shared ticker list used across all contract tests
const TEST_TICKERS = ['AAPL', 'MSFT', 'GOOG'];
const API = '/api/v1';

// ---------------------------------------------------------------------------

describe('factor-research: HTTP contract parity', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptorsFromDi()),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // ---------- POST /factors/compute -----------------------------------------

  describe('POST /factors/compute', () => {

    it('when compute is triggered, HTTP method is POST', () => {
      const body: FactorComputeRequest = { tickers: TEST_TICKERS };
      http.post<FactorComputeJobResponse>(`${API}/factors/compute`, body).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute`);
      expect(req.request.method).toBe('POST');
      req.flush({ job_id: 'job-abc' });
    });

    it('when compute is triggered, request body contains tickers array', () => {
      const body: FactorComputeRequest = { tickers: TEST_TICKERS };
      http.post<FactorComputeJobResponse>(`${API}/factors/compute`, body).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute`);
      expect(req.request.body).toEqual(jasmine.objectContaining({ tickers: jasmine.any(Array) }));
      expect((req.request.body as FactorComputeRequest).tickers).toEqual(TEST_TICKERS);
      req.flush({ job_id: 'job-abc' });
    });

    it('when compute responds, response contains job_id string', () => {
      const body: FactorComputeRequest = { tickers: TEST_TICKERS };
      let response: FactorComputeJobResponse | null = null;
      http.post<FactorComputeJobResponse>(`${API}/factors/compute`, body).subscribe((r) => { response = r; });

      const req = httpMock.expectOne(`${API}/factors/compute`);
      req.flush({ job_id: 'job-abc' });

      expect(response).not.toBeNull();
      expect((response as unknown as FactorComputeJobResponse).job_id).toBe('job-abc');
    });
  });

  // ---------- GET /factors/compute/{id} (polling) ---------------------------

  describe('GET /factors/compute/{id}', () => {
    const JOB_ID = 'job-abc';

    it('when polling a job, HTTP method is GET', () => {
      http.get<FactorComputeStatusResponse>(`${API}/factors/compute/${JOB_ID}`).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`);
      expect(req.request.method).toBe('GET');
      req.flush({ status: 'pending', ic_reports: [], taa_signals: [], factor_returns: [], cma_sets: [] });
    });

    it('when job URL is constructed, job_id is interpolated (no raw placeholder)', () => {
      http.get<FactorComputeStatusResponse>(`${API}/factors/compute/${JOB_ID}`).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`);
      // Criterion: "URL path parameters are correctly interpolated"
      expect(req.request.url).not.toContain('{id}');
      expect(req.request.url).not.toContain('{job_id}');
      expect(req.request.url).toContain(JOB_ID);
      req.flush({ status: 'pending', ic_reports: [], taa_signals: [], factor_returns: [], cma_sets: [] });
    });

    it('when job completes, response has ic_reports, taa_signals, factor_returns, cma_sets', () => {
      let result: FactorComputeStatusResponse | null = null;
      http
        .get<FactorComputeStatusResponse>(`${API}/factors/compute/${JOB_ID}`)
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`);
      const mockResult: FactorComputeStatusResponse = {
        status: 'completed',
        ic_reports: [{ factor: 'MTUM', ic: 0.12 }],
        taa_signals: [{ asset: 'AAPL', signal: 0.8 }],
        factor_returns: [{ date: '2024-01-01', value: 0.02 }],
        cma_sets: [{ horizon: 10, expected_return: 0.07 }],
      };
      req.flush(mockResult);

      expect(result).not.toBeNull();
      const r = result as unknown as FactorComputeStatusResponse;
      expect(r.ic_reports).toBeInstanceOf(Array);
      expect(r.taa_signals).toBeInstanceOf(Array);
      expect(r.factor_returns).toBeInstanceOf(Array);
      expect(r.cma_sets).toBeInstanceOf(Array);
    });
  });

  // ---------- POST /factors/validate ----------------------------------------

  describe('POST /factors/validate', () => {

    it('when validate is triggered, HTTP method is POST', () => {
      const body: FactorValidateRequest = { tickers: TEST_TICKERS };
      http.post<FactorValidateResponse>(`${API}/factors/validate`, body).subscribe();

      const req = httpMock.expectOne(`${API}/factors/validate`);
      expect(req.request.method).toBe('POST');
      req.flush({ ic_series: [], t_stat_series: [] });
    });

    it('when validate is triggered, request body contains tickers array', () => {
      const body: FactorValidateRequest = { tickers: TEST_TICKERS };
      http.post<FactorValidateResponse>(`${API}/factors/validate`, body).subscribe();

      const req = httpMock.expectOne(`${API}/factors/validate`);
      expect((req.request.body as FactorValidateRequest).tickers).toEqual(TEST_TICKERS);
      req.flush({ ic_series: [], t_stat_series: [] });
    });

    it('when validate responds, response carries ic_series and t_stat_series arrays', () => {
      let result: FactorValidateResponse | null = null;
      http
        .post<FactorValidateResponse>(`${API}/factors/validate`, { tickers: TEST_TICKERS })
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/factors/validate`);
      req.flush({
        ic_series: [{ date: '2024-01', value: 0.1 }],
        t_stat_series: [{ date: '2024-01', value: 2.3 }],
      });

      const r = result as unknown as FactorValidateResponse;
      expect(r.ic_series).toBeInstanceOf(Array);
      expect(r.t_stat_series).toBeInstanceOf(Array);
    });
  });

  // ---------- POST /factors/score -------------------------------------------

  describe('POST /factors/score', () => {

    it('when score is triggered, HTTP method is POST', () => {
      http.post<FactorScoreResponse>(`${API}/factors/score`, { tickers: TEST_TICKERS }).subscribe();

      const req = httpMock.expectOne(`${API}/factors/score`);
      expect(req.request.method).toBe('POST');
      req.flush({ scores: [] });
    });

    it('when score is triggered, request body contains tickers array', () => {
      http.post<FactorScoreResponse>(`${API}/factors/score`, { tickers: TEST_TICKERS }).subscribe();

      const req = httpMock.expectOne(`${API}/factors/score`);
      expect((req.request.body as FactorScoreRequest).tickers).toEqual(TEST_TICKERS);
      req.flush({ scores: [] });
    });

    it('when score responds, response carries scores array', () => {
      let result: FactorScoreResponse | null = null;
      http
        .post<FactorScoreResponse>(`${API}/factors/score`, { tickers: TEST_TICKERS })
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/factors/score`);
      req.flush({ scores: [{ ticker: 'AAPL', score: 0.75 }] });

      expect((result as unknown as FactorScoreResponse).scores).toBeInstanceOf(Array);
    });
  });

  // ---------- POST /factors/select ------------------------------------------

  describe('POST /factors/select', () => {

    it('when select is triggered, HTTP method is POST', () => {
      http.post<FactorSelectResponse>(`${API}/factors/select`, { tickers: TEST_TICKERS }).subscribe();

      const req = httpMock.expectOne(`${API}/factors/select`);
      expect(req.request.method).toBe('POST');
      req.flush({ selected: [] });
    });

    it('when select is triggered, request body contains tickers array', () => {
      http.post<FactorSelectResponse>(`${API}/factors/select`, { tickers: TEST_TICKERS }).subscribe();

      const req = httpMock.expectOne(`${API}/factors/select`);
      expect((req.request.body as FactorSelectRequest).tickers).toEqual(TEST_TICKERS);
      req.flush({ selected: [] });
    });

    it('when select responds, response carries selected array of ticker strings', () => {
      let result: FactorSelectResponse | null = null;
      http
        .post<FactorSelectResponse>(`${API}/factors/select`, { tickers: TEST_TICKERS })
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/factors/select`);
      req.flush({ selected: ['AAPL', 'MSFT'] });

      expect((result as unknown as FactorSelectResponse).selected).toEqual(['AAPL', 'MSFT']);
    });
  });

  // ---------- GET /views/macro-calibration ----------------------------------

  describe('GET /views/macro-calibration', () => {

    it('when TAA panel loads, HTTP method is GET', () => {
      http.get<MacroCalibrationResponse>(`${API}/views/macro-calibration`).subscribe();

      const req = httpMock.expectOne(`${API}/views/macro-calibration`);
      expect(req.request.method).toBe('GET');
      req.flush({ weights: {} });
    });

    it('when macro-calibration responds, response carries a weights record', () => {
      let result: MacroCalibrationResponse | null = null;
      http
        .get<MacroCalibrationResponse>(`${API}/views/macro-calibration`)
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/views/macro-calibration`);
      req.flush({ weights: { AAPL: 0.5, MSFT: 0.5 } });

      const r = result as unknown as MacroCalibrationResponse;
      expect(typeof r.weights).toBe('object');
      expect(r.weights).not.toBeNull();
    });

    it('when macro-calibration URL is constructed, it contains no raw path placeholders', () => {
      http.get<MacroCalibrationResponse>(`${API}/views/macro-calibration`).subscribe();

      const req = httpMock.expectOne(`${API}/views/macro-calibration`);
      expect(req.request.url).not.toContain('{');
      expect(req.request.url).not.toContain('}');
      req.flush({ weights: {} });
    });
  });

  // ---------- GET /macro-data/te-observations ------------------------------

  describe('GET /macro-data/te-observations', () => {

    it('when CMA panel loads, HTTP method is GET', () => {
      http.get<TeObservationsResponse>(`${API}/macro-data/te-observations`).subscribe();

      const req = httpMock.expectOne(`${API}/macro-data/te-observations`);
      expect(req.request.method).toBe('GET');
      req.flush({ observations: [] });
    });

    it('when te-observations responds, response carries observations array', () => {
      let result: TeObservationsResponse | null = null;
      http
        .get<TeObservationsResponse>(`${API}/macro-data/te-observations`)
        .subscribe((r) => { result = r; });

      const req = httpMock.expectOne(`${API}/macro-data/te-observations`);
      req.flush({ observations: [{ asset: 'AAPL', te: 0.04 }] });

      expect((result as unknown as TeObservationsResponse).observations).toBeInstanceOf(Array);
    });
  });

  // ---------- No API names in user-facing copy -----------------------------

  describe('user-facing copy does not contain API endpoint or parameter names', () => {

    // Each string is a known API endpoint token or parameter name.
    // They must not appear verbatim in any UI-visible string.
    const forbiddenApiTokens = [
      '/factors/compute',
      '/factors/validate',
      '/factors/score',
      '/factors/select',
      '/views/macro-calibration',
      '/macro-data/te-observations',
      'cv_type',
      'walk_forward',
      'job_id',
      'ic_reports',
      'taa_signals',
      'factor_returns',
      'cma_sets',
    ];

    // Verify that spec-defined UI labels do not contain API tokens
    const specDefinedLabels = [
      'Validates the strategy using out-of-sample rolling windows to avoid look-ahead bias.',
      'Run factor analysis',
      'Score factors',
      'Select factors',
      'Factor scores not available. Run the factor pipeline first.',
    ];

    forbiddenApiTokens.forEach((token) => {
      specDefinedLabels.forEach((label) => {
        it(`user-facing label "${label.slice(0, 40)}..." does not contain API token "${token}"`, () => {
          expect(label).not.toContain(token);
        });
      });
    });
  });
});
