import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { WalkForwardPanelComponent } from './walk-forward-panel';
import { environment } from '../../../environments/environment';

const API = environment.apiUrl;

describe('WalkForwardPanelComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function createPanel() {
    const fx = TestBed.createComponent(WalkForwardPanelComponent);
    fx.componentRef.setInput('tickers', ['AAPL', 'MSFT']);
    fx.componentRef.setInput('startDate', '2024-01-01');
    fx.componentRef.setInput('endDate', '2024-12-31');
    fx.detectChanges();
    return fx;
  }

  it('POSTs /validate/walk-forward with cv_type=walk_forward', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();

    const req = http.expectOne(`${API}validate/walk-forward`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body.cv_type).toBe('walk_forward');
    expect(req.request.body.tickers).toEqual(['AAPL', 'MSFT']);
    req.flush({ job_id: 'v1', status: 'pending', message: '' });

    expect(fx.componentInstance.jobId()).toBe('v1');
    expect(fx.componentInstance.isRunning()).toBe(true);
  });

  it('when walk-forward runs, the POST body carries all 7 ValidateRequest fields', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();

    const req = http.expectOne(`${API}validate/walk-forward`);
    expect(Object.keys(req.request.body as object).sort()).toEqual([
      'cv_config',
      'cv_type',
      'end_date',
      'optimizer_config',
      'optimizer_type',
      'start_date',
      'tickers',
    ]);
    req.flush({ job_id: 'v1', status: 'pending', message: '' });
  });

  it('when walk-forward runs, cv_config and optimizer_config are objects', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();

    const req = http.expectOne(`${API}validate/walk-forward`);
    const body = req.request.body as Record<string, unknown>;
    expect(typeof body['cv_config']).toBe('object');
    expect(typeof body['optimizer_config']).toBe('object');
    req.flush({ job_id: 'v1', status: 'pending', message: '' });
  });

  it('surfaces the error when walk-forward POST fails', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();

    http
      .expectOne(`${API}validate/walk-forward`)
      .flush({ detail: 'busy' }, { status: 409, statusText: 'Conflict' });

    expect(fx.componentInstance.error()).toBeTruthy();
    expect(fx.componentInstance.jobId()).toBeNull();
  });

  it('blocks the run when no tickers are provided', () => {
    const fx = TestBed.createComponent(WalkForwardPanelComponent);
    fx.componentRef.setInput('tickers', []);
    fx.componentRef.setInput('startDate', '2024-01-01');
    fx.componentRef.setInput('endDate', '2024-12-31');
    fx.detectChanges();

    fx.componentInstance.onRun();
    http.expectNone((r) => r.url.includes('/validate/walk-forward'));
    expect(fx.componentInstance.error()).toBe('No tickers provided');
  });

  it('fetches the completed CV result and renders per-fold + aggregate rows', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({
      job_id: 'v1',
      status: 'pending',
      message: '',
    });

    fx.componentInstance.onJobCompleted();
    const poll = http.expectOne(`${API}validate/walk-forward/v1`);
    poll.flush({
      job_id: 'v1',
      status: 'completed',
      current: 3,
      total: 3,
      current_fold: 3,
      total_folds: 3,
      errors: [],
      result: {
        folds: [
          {
            weights: { AAPL: 0.5, MSFT: 0.5 },
            measures: {
              sharpe: 1.2,
              annualized_return: 0.1,
              volatility: 0.15,
              max_drawdown: -0.08,
            },
          },
          {
            weights: { AAPL: 0.7, MSFT: 0.3 },
            measures: {
              sharpe: 0.9,
              annualized_return: 0.08,
              volatility: 0.16,
              max_drawdown: -0.11,
            },
          },
        ],
        aggregate: {
          sharpe: 1.05,
          annualized_return: 0.09,
          volatility: 0.155,
          max_drawdown: -0.095,
        },
      },
      error: null,
    });

    expect(fx.componentInstance.folds().length).toBe(2);
    expect(fx.componentInstance.foldRows()[0].sharpe).toBe('1.200');
    expect(fx.componentInstance.aggregateRows().length).toBe(4);
    expect(fx.componentInstance.jobId()).toBeNull();
  });

  it('handles a failed job by clearing jobId and storing the error', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({
      job_id: 'v1', status: 'pending', message: '',
    });

    fx.componentInstance.onJobFailed('solver crashed');
    expect(fx.componentInstance.jobId()).toBeNull();
    expect(fx.componentInstance.error()).toBe('solver crashed');
  });

  it('when walk-forward POST fails, error element has role="alert" with non-blank text', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();

    http
      .expectOne(`${API}validate/walk-forward`)
      .flush({ detail: 'busy' }, { status: 409, statusText: 'Conflict' });
    fx.detectChanges();

    const alertEl = fx.nativeElement.querySelector('[role="alert"]') as HTMLElement | null;
    expect(alertEl).withContext('expected [role="alert"] after walk-forward error').toBeTruthy();
    expect(alertEl?.textContent?.trim().length).toBeGreaterThan(0);
  });

  // ── Branch coverage: guards, key fallbacks, result-shape variants ─────────────

  it('ignores a second onRun while a job is already running', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onRun();
    http.expectNone(`${API}validate/walk-forward`);
  });

  it('does not poll when onJobCompleted is called with no active job', () => {
    const fx = createPanel();
    fx.componentInstance.onJobCompleted();
    http.expectNone((r) => r.url.includes('/validate/walk-forward/'));
  });

  it('surfaces the error when pollWalkForward fails', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onJobCompleted();
    http
      .expectOne(`${API}validate/walk-forward/v1`)
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });

    expect(fx.componentInstance.error()).toBeTruthy();
  });

  it('renders "—" for fold cells when the fold has no recognised measures', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onJobCompleted();
    http.expectOne(`${API}validate/walk-forward/v1`).flush({
      job_id: 'v1', status: 'completed', current: 1, total: 1, current_fold: 1, total_folds: 1,
      errors: [], result: { folds: [{ weights: {}, measures: {} }] }, error: null,
    });

    const row = fx.componentInstance.foldRows()[0];
    expect(row['sharpe']).toBe('—');
    expect(row['annualizedReturn']).toBe('—');
    expect(row['maxDrawdown']).toBe('—');
  });

  it('reads Title-cased skfolio measure keys and negates the drawdown for a fold row', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onJobCompleted();
    http.expectOne(`${API}validate/walk-forward/v1`).flush({
      job_id: 'v1', status: 'completed', current: 1, total: 1, current_fold: 1, total_folds: 1,
      errors: [],
      result: { folds: [{ weights: {}, measures: { 'Annualized Sharpe Ratio': 1.5, 'MAX Drawdown': 0.2 } }] },
      error: null,
    });

    const row = fx.componentInstance.foldRows()[0];
    expect(row['sharpe']).toBe('1.500');
    expect(row['maxDrawdown']).toBe('-20.00%');
  });

  it('applies results delivered under the fold_results key with an aggregate_score', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onJobCompleted();
    http.expectOne(`${API}validate/walk-forward/v1`).flush({
      job_id: 'v1', status: 'completed', current: 1, total: 1, current_fold: 1, total_folds: 1,
      errors: [],
      result: { fold_results: [{ weights: {}, measures: { sharpe: 1.0 } }], aggregate_score: 0.9 },
      error: null,
    });

    expect(fx.componentInstance.folds().length).toBe(1);
    expect(fx.componentInstance.aggregate()['aggregate_score']).toBe(0.9);
  });

  it('handles a poll result with a null payload by leaving folds empty', () => {
    const fx = createPanel();
    fx.componentInstance.onRun();
    http.expectOne(`${API}validate/walk-forward`).flush({ job_id: 'v1', status: 'pending', message: '' });

    fx.componentInstance.onJobCompleted();
    http.expectOne(`${API}validate/walk-forward/v1`).flush({
      job_id: 'v1', status: 'completed', current: 0, total: 0, current_fold: 0, total_folds: 0,
      errors: [], result: null, error: null,
    });

    expect(fx.componentInstance.folds()).toEqual([]);
    expect(fx.componentInstance.aggregateRows()).toEqual([]);
  });
});
