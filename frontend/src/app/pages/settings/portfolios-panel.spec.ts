import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { PortfoliosPanelComponent } from './portfolios-panel';
import { environment } from '../../../environments/environment';
import type { PortfolioDto } from '../../core/models/portfolio-api.model';

const API = environment.apiUrl;

const EXISTING: PortfolioDto = {
  id: 'id-1',
  name: 'Core',
  description: null,
  currency: 'USD',
  benchmark_ticker: 'SPY',
  is_active: true,
  created_at: '2026-04-01T00:00:00Z',
  updated_at: '2026-04-01T00:00:00Z',
};

describe('PortfoliosPanelComponent', () => {
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

  it('loads portfolios on construction', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}market/indices`).flush({ indices: [{ ticker: 'SPY' }] });
    http.expectOne(`${API}portfolio/`).flush({ items: [EXISTING], total: 1 });

    expect(fx.componentInstance.portfolios().length).toBe(1);
    expect(fx.componentInstance.tableRows()[0]['name']).toBe('Core');
  });

  it('blocks submit when name is too short', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();
    http.expectOne(`${API}market/indices`).flush({ indices: [{ ticker: 'SPY' }] });
    http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });

    fx.componentInstance.toggleCreate();
    fx.componentInstance.updateField('name', 'A');
    expect(fx.componentInstance.isFormValid()).toBe(false);

    fx.componentInstance.submit();
    http.expectNone((r) => r.method === 'POST');
  });

  it('POSTs /portfolio/ with trimmed payload and refreshes on success', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();
    http.expectOne(`${API}market/indices`).flush({ indices: [{ ticker: 'SPY' }] });
    http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });

    fx.componentInstance.toggleCreate();
    fx.componentInstance.updateField('name', '  Core  ');
    fx.componentInstance.updateField('currency', 'USD');
    fx.componentInstance.updateField('benchmark_ticker', '  SPY ');
    fx.componentInstance.submit();

    const post = http.expectOne(
      (r) => r.method === 'POST' && r.url === `${API}portfolio/`,
    );
    expect(post.request.body.name).toBe('Core');
    expect(post.request.body.currency).toBe('USD');
    expect(post.request.body.benchmark_ticker).toBe('SPY');
    post.flush(EXISTING);

    http.expectOne(`${API}portfolio/`).flush({ items: [EXISTING], total: 1 });

    expect(fx.componentInstance.creating()).toBe(false);
    expect(fx.componentInstance.showCreate()).toBe(false);
    expect(fx.componentInstance.portfolios().length).toBe(1);
  });

  it('surfaces the error when create fails', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();
    http.expectOne(`${API}market/indices`).flush({ indices: [{ ticker: 'SPY' }] });
    http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });

    fx.componentInstance.toggleCreate();
    fx.componentInstance.updateField('name', 'Core');
    fx.componentInstance.submit();

    http
      .expectOne((r) => r.method === 'POST' && r.url === `${API}portfolio/`)
      .flush({ detail: 'duplicate' }, { status: 409, statusText: 'Conflict' });

    expect(fx.componentInstance.creating()).toBe(false);
    expect(fx.componentInstance.createError()).toBeTruthy();
    expect(fx.componentInstance.showCreate()).toBe(true);
  });
});
