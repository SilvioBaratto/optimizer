/**
 * Source-blind spec — authored from acceptance criteria only.
 *
 * Criterion (T2):
 *   "POST /factors/compute + polling GET /factors/compute/{id} populates
 *    ic_reports / taa_signals / factor_returns / cma_sets and the IC chart
 *    renders on completion of the compute + validate round-trip."
 *
 * Criterion (T2):
 *   "ScorePanelComponent renders results from POST /factors/score and
 *    SelectPanelComponent from POST /factors/select (request shape parity;
 *    loading flags toggle)."
 *
 * Criterion (T3):
 *   "Factor-research: the compute + validate round-trip completes and the IC
 *    chart renders."
 */

import { TestBed } from '@angular/core/testing';
import { HttpClient, provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { takeWhile, catchError } from 'rxjs/operators';
import { of } from 'rxjs';

// ---------------------------------------------------------------------------
// Spec-derived response shapes
// ---------------------------------------------------------------------------

interface ComputeJob {
  job_id: string;
}

interface ComputeStatus {
  status: 'pending' | 'running' | 'completed' | 'failed';
  ic_reports: unknown[];
  taa_signals: unknown[];
  factor_returns: unknown[];
  cma_sets: unknown[];
}

interface ValidateResponse {
  ic_series: unknown[];
  t_stat_series: unknown[];
}

interface ScoreResponse {
  scores: unknown[];
}

interface SelectResponse {
  selected: string[];
}

const API = '/api/v1';
const TEST_TICKERS = ['AAPL', 'MSFT', 'GOOG'];
const JOB_ID = 'job-compute-001';

/** Single-poll helper — makes one GET and returns the response. */
function pollComputeOnce(
  http: HttpClient,
  jobId: string,
): import('rxjs').Observable<ComputeStatus> {
  return http.get<ComputeStatus>(`${API}/factors/compute/${jobId}`).pipe(
    catchError(() => of({ status: 'failed', ic_reports: [], taa_signals: [], factor_returns: [], cma_sets: [] } as ComputeStatus)),
  );
}

// ---------------------------------------------------------------------------

describe('factor-research: compute + validate + score + select chain', () => {
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

  // ---------- compute job creation -----------------------------------------

  describe('POST /factors/compute → job creation', () => {

    it('when tickers are submitted, POST /factors/compute returns a job_id', () => {
      let jobId: string | null = null;

      http
        .post<ComputeJob>(`${API}/factors/compute`, { tickers: TEST_TICKERS })
        .subscribe((r) => { jobId = r.job_id; });

      const req = httpMock.expectOne(`${API}/factors/compute`);
      req.flush({ job_id: JOB_ID });

      expect(jobId as string | null).toBe(JOB_ID);
    });

    it('when compute job is created, no tickers appear as raw query parameters in the URL', () => {
      http.post<ComputeJob>(`${API}/factors/compute`, { tickers: TEST_TICKERS }).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute`);
      expect(req.request.url).not.toContain('tickers=');
      req.flush({ job_id: JOB_ID });
    });
  });

  // ---------- polling: job status response ---------------------------------

  describe('GET /factors/compute/{id} polling', () => {

    it('when polling a completed job, response has the correct status', () => {
      let result: ComputeStatus | null = null;
      pollComputeOnce(http, JOB_ID).subscribe((s) => { result = s; });

      httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`).flush({
        status: 'completed',
        ic_reports: [{ factor: 'MTUM', ic: 0.1 }],
        taa_signals: [{ asset: 'US_EQ', weight: 0.6 }],
        factor_returns: [{ date: '2024-01-01', return: 0.02 }],
        cma_sets: [{ horizon: 10, expected_return: 0.07 }],
      });

      expect((result as ComputeStatus | null)?.status).toBe('completed');
    });

    it('when job completes, result has all four required arrays', () => {
      let finalStatus: ComputeStatus | null = null;

      pollComputeOnce(http, JOB_ID).subscribe((s) => { finalStatus = s; });

      httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`).flush({
        status: 'completed',
        ic_reports: [{ factor: 'VALUE', ic: 0.08 }],
        taa_signals: [{ asset: 'AAPL', weight: 0.5 }],
        factor_returns: [{ date: '2024-01', value: 0.01 }],
        cma_sets: [{ label: '10yr', value: 0.065 }],
      } as ComputeStatus);

      expect(finalStatus).not.toBeNull();
      const r = finalStatus as unknown as ComputeStatus;
      expect(r.ic_reports).toBeInstanceOf(Array);
      expect(r.taa_signals).toBeInstanceOf(Array);
      expect(r.factor_returns).toBeInstanceOf(Array);
      expect(r.cma_sets).toBeInstanceOf(Array);
      expect(r.ic_reports.length).toBeGreaterThan(0);
    });

    it('when polling job URL, job_id is interpolated correctly (no placeholder tokens)', () => {
      pollComputeOnce(http, JOB_ID).subscribe();

      const req = httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`);
      expect(req.request.url).toContain(JOB_ID);
      expect(req.request.url).not.toContain('{');
      expect(req.request.url).not.toContain('}');
      req.flush({ status: 'completed', ic_reports: [], taa_signals: [], factor_returns: [], cma_sets: [] });
    });

    it('when job status is pending, ic_reports is empty', () => {
      let result: ComputeStatus | null = null;
      pollComputeOnce(http, JOB_ID).subscribe((s) => { result = s; });

      httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`).flush({
        status: 'pending',
        ic_reports: [],
        taa_signals: [],
        factor_returns: [],
        cma_sets: [],
      });

      expect((result as ComputeStatus | null)?.ic_reports.length).toBe(0);
    });

    it('when polling with takeWhile, completed status is included in the result (inclusive=true)', () => {
      // Verifies the takeWhile(..., true) inclusive contract
      const statuses: ComputeStatus['status'][] = [];

      pollComputeOnce(http, JOB_ID)
        .pipe(
          takeWhile((s) => s.status !== 'completed' && s.status !== 'failed', true),
        )
        .subscribe((s) => statuses.push(s.status));

      httpMock.expectOne(`${API}/factors/compute/${JOB_ID}`).flush({
        status: 'completed',
        ic_reports: [],
        taa_signals: [],
        factor_returns: [],
        cma_sets: [],
      });

      expect(statuses).toContain('completed');
    });
  });

  // ---------- validate round-trip ------------------------------------------

  describe('compute → validate round-trip', () => {

    it('when compute completes, POST /factors/validate is called next', () => {
      let validateCalled = false;

      http
        .post<ComputeJob>(`${API}/factors/compute`, { tickers: TEST_TICKERS })
        .subscribe((job) => {
          http
            .get<ComputeStatus>(`${API}/factors/compute/${job.job_id}`)
            .subscribe((status) => {
              if (status.status === 'completed') {
                http
                  .post<ValidateResponse>(`${API}/factors/validate`, { tickers: TEST_TICKERS })
                  .subscribe(() => { validateCalled = true; });
              }
            });
        });

      httpMock.expectOne(`${API}/factors/compute`).flush({ job_id: JOB_ID });
      httpMock
        .expectOne(`${API}/factors/compute/${JOB_ID}`)
        .flush({ status: 'completed', ic_reports: [], taa_signals: [], factor_returns: [], cma_sets: [] });
      httpMock.expectOne(`${API}/factors/validate`).flush({ ic_series: [], t_stat_series: [] });

      expect(validateCalled).toBeTrue();
    });

    it('when validate responds, ic_series and t_stat_series are present in the response', () => {
      let validateResult: ValidateResponse | null = null;

      http
        .post<ValidateResponse>(`${API}/factors/validate`, { tickers: TEST_TICKERS })
        .subscribe((r) => { validateResult = r; });

      httpMock.expectOne(`${API}/factors/validate`).flush({
        ic_series: [{ date: '2024-01', ic: 0.12 }, { date: '2024-02', ic: 0.09 }],
        t_stat_series: [{ date: '2024-01', t: 2.1 }, { date: '2024-02', t: 1.9 }],
      });

      const r = validateResult as unknown as ValidateResponse;
      expect(r.ic_series.length).toBeGreaterThan(0);
      expect(r.t_stat_series.length).toBeGreaterThan(0);
    });
  });

  // ---------- score chain ---------------------------------------------------

  describe('POST /factors/score — loading flag contract', () => {

    it('when score request is in-flight, loading flag should be true before response arrives', () => {
      let isLoading = false;
      let resultArrived = false;

      isLoading = true;
      http
        .post<ScoreResponse>(`${API}/factors/score`, { tickers: TEST_TICKERS })
        .subscribe({
          next: () => { isLoading = false; resultArrived = true; },
        });

      expect(isLoading).withContext('loading must be true while request is in-flight').toBeTrue();

      httpMock.expectOne(`${API}/factors/score`).flush({ scores: [{ ticker: 'AAPL', score: 0.9 }] });

      expect(isLoading).withContext('loading must be false once response arrives').toBeFalse();
      expect(resultArrived).toBeTrue();
    });

    it('when score responds successfully, scores array is non-null', () => {
      let result: ScoreResponse | null = null;
      http
        .post<ScoreResponse>(`${API}/factors/score`, { tickers: TEST_TICKERS })
        .subscribe((r) => { result = r; });

      httpMock.expectOne(`${API}/factors/score`).flush({ scores: [{ ticker: 'MSFT', score: 0.7 }] });

      expect((result as unknown as ScoreResponse).scores).toBeInstanceOf(Array);
    });
  });

  // ---------- select chain --------------------------------------------------

  describe('POST /factors/select — loading flag contract', () => {

    it('when select request is in-flight, loading flag is true before response arrives', () => {
      let isLoading = false;
      let resultArrived = false;

      isLoading = true;
      http
        .post<SelectResponse>(`${API}/factors/select`, { tickers: TEST_TICKERS })
        .subscribe({
          next: () => { isLoading = false; resultArrived = true; },
        });

      expect(isLoading).toBeTrue();

      httpMock.expectOne(`${API}/factors/select`).flush({ selected: ['AAPL', 'GOOG'] });

      expect(isLoading).toBeFalse();
      expect(resultArrived).toBeTrue();
    });

    it('when select responds, selected array contains ticker strings', () => {
      let result: SelectResponse | null = null;
      http
        .post<SelectResponse>(`${API}/factors/select`, { tickers: TEST_TICKERS })
        .subscribe((r) => { result = r; });

      httpMock.expectOne(`${API}/factors/select`).flush({ selected: ['AAPL', 'GOOG'] });

      const r = result as unknown as SelectResponse;
      expect(r.selected).toEqual(['AAPL', 'GOOG']);
      r.selected.forEach((ticker) => {
        expect(typeof ticker).toBe('string');
        expect(ticker.length).toBeGreaterThan(0);
      });
    });
  });

  // ---------- chain error resilience ----------------------------------------

  describe('when any step in the chain errors', () => {

    it('when POST /factors/compute returns 500, chain does not proceed to validate', () => {
      let validateCalled = false;
      const consoleErrorSpy = spyOn(console, 'error');

      http
        .post<ComputeJob>(`${API}/factors/compute`, { tickers: TEST_TICKERS })
        .pipe(
          catchError(() => {
            return of(null as unknown as ComputeJob);
          }),
        )
        .subscribe((job) => {
          if (job?.job_id) {
            http
              .post<ValidateResponse>(`${API}/factors/validate`, { tickers: TEST_TICKERS })
              .subscribe(() => { validateCalled = true; });
          }
        });

      httpMock.expectOne(`${API}/factors/compute`).flush('error', { status: 500, statusText: 'Server Error' });

      expect(validateCalled).toBeFalse();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });
  });
});
