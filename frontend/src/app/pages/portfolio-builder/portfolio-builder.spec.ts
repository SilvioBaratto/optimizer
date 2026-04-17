import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { PortfolioBuilderComponent } from './portfolio-builder';
import { environment } from '../../../environments/environment';
import { ICON_PROVIDER } from '../../icons';

const BASE = `${environment.apiUrl}portfolio`;

describe('PortfolioBuilderComponent', () => {
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        ICON_PROVIDER,
      ],
    });
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  function createComponent() {
    const fx = TestBed.createComponent(PortfolioBuilderComponent);
    fx.detectChanges();
    return fx;
  }

  function flushPortfolioList() {
    http.expectOne(`${BASE}/`).flush({ items: [], total: 0 });
  }

  it('fetches /portfolio/ on init and renders the page when list is empty', () => {
    const fx = createComponent();
    flushPortfolioList();
    expect(fx.componentInstance.isLoading()).toBe(false);
    expect(fx.componentInstance.portfolios()).toEqual([]);
  });

  it('opens the create-portfolio modal and POSTs to /portfolio', () => {
    const fx = createComponent();
    flushPortfolioList();

    fx.componentInstance.openCreateModal();
    fx.componentInstance.formName.set('MyPortfolio');
    fx.componentInstance.formCurrency.set('USD');
    fx.componentInstance.formBenchmark.set('SPY');
    fx.componentInstance.submitCreate();

    const req = http.expectOne(`${BASE}/`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({
      name: 'MyPortfolio',
      description: null,
      currency: 'USD',
      benchmark_ticker: 'SPY',
    });

    req.flush({
      id: 'p1',
      name: 'MyPortfolio',
      description: null,
      currency: 'USD',
      benchmark_ticker: 'SPY',
      is_active: true,
      created_at: '',
      updated_at: '',
    });

    expect(fx.componentInstance.portfolios().length).toBe(1);
    expect(fx.componentInstance.activePortfolio()?.name).toBe('MyPortfolio');
    expect(fx.componentInstance.showCreateModal()).toBe(false);
  });

  it('disables create submission while name is shorter than 2 chars', () => {
    const fx = createComponent();
    flushPortfolioList();

    fx.componentInstance.formName.set('A');
    expect(fx.componentInstance.formIsValid()).toBe(false);

    fx.componentInstance.formName.set('AB');
    expect(fx.componentInstance.formIsValid()).toBe(true);
  });

  it('surfaces the error message when POST /portfolio fails (e.g. duplicate name)', () => {
    const fx = createComponent();
    flushPortfolioList();

    fx.componentInstance.formName.set('Dup');
    fx.componentInstance.submitCreate();

    http
      .expectOne(`${BASE}/`)
      .flush({ detail: 'duplicate name' }, { status: 409, statusText: 'Conflict' });

    expect(fx.componentInstance.formStatus()).toBe('error');
    expect(fx.componentInstance.formError()).toContain('duplicate');
  });

  it('POSTs to /portfolio/{name}/snapshots when Save as Snapshot fires', () => {
    const fx = createComponent();

    http.expectOne(`${BASE}/`).flush({
      items: [
        {
          id: 'p1',
          name: 'MyPortfolio',
          description: null,
          currency: 'USD',
          benchmark_ticker: 'SPY',
          is_active: true,
          created_at: '',
          updated_at: '',
        },
      ],
      total: 1,
    });

    fx.componentInstance.saveSnapshot();

    const req = http.expectOne(`${BASE}/MyPortfolio/snapshots`);
    expect(req.request.method).toBe('POST');
    expect(req.request.body.snapshot_type).toBe('manual');
    req.flush({
      id: 's1', portfolio_id: 'p1', snapshot_date: '2024-01-01',
      snapshot_type: 'manual', weights: {}, sector_mapping: null,
      summary: null, optimizer_config: null, turnover: null,
      holding_count: 0, created_at: '',
    });

    expect(fx.componentInstance.snapshotStatus()).toBe('success');
  });

  it('opens the flyout when a ticker is selected from the universe panel', () => {
    const fx = createComponent();
    flushPortfolioList();

    fx.componentInstance.onTickerSelected({
      id: 'abc',
      ticker: 'AAPL',
      name: 'Apple',
      exchange: 'NASDAQ',
      type: 'equity',
      currency: 'USD',
      isin: 'US0378331005',
    });

    expect(fx.componentInstance.flyoutInstrumentId()).toBe('abc');

    fx.componentInstance.closeFlyout();
    expect(fx.componentInstance.flyoutInstrumentId()).toBeNull();
  });
});
