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
});
