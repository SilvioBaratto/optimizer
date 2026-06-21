import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import { PortfoliosPanelComponent } from './portfolios-panel';
import { ReferenceIndexService } from '../../core/services/reference-index.service';
import { environment } from '../../../environments/environment';
import type { PortfolioDto } from '../../core/models/portfolio-api.model';
import type { ReferenceIndexSeedProgress } from '../../core/services/reference-index.service';

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

  describe('error banner (issue #964)', () => {
    it('when portfolio list returns 500, loadError is set and app-page-error-banner is in the DOM', () => {
      const fx = TestBed.createComponent(PortfoliosPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}market/indices`).flush({ indices: [] });
      http.expectOne(`${API}portfolio/`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });

    it('clicking the retry button in the load-error banner re-fires the portfolio list request', () => {
      const fx = TestBed.createComponent(PortfoliosPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}market/indices`).flush({ indices: [] });
      http.expectOne(`${API}portfolio/`)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      fx.detectChanges();

      const banner: HTMLElement = fx.nativeElement.querySelector('app-page-error-banner');
      banner.querySelector<HTMLButtonElement>('button')!.click();
      fx.detectChanges();

      http.expectOne(`${API}portfolio/`).flush({ items: [EXISTING], total: 1 });
      fx.detectChanges();

      expect(fx.componentInstance.loadError()).toBeNull();
      expect(fx.componentInstance.portfolios().length).toBe(1);
    });

    it('when create returns 409, createError is set and app-page-error-banner is in the DOM', () => {
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
      fx.detectChanges();

      expect(fx.componentInstance.createError()).toBeTruthy();
      expect(fx.nativeElement.querySelector('app-page-error-banner')).not.toBeNull();
    });
  });

  // ── loadKnownBenchmarks catch branch (issue #1026) ─────────────────────────
  it('when GET /market/indices returns 500, the panel does not crash and loadError stays null', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();

    http.expectOne(`${API}market/indices`)
      .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
    http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });

    // The catchError in loadKnownBenchmarks must swallow this — no error banner
    expect(fx.componentInstance.loadError()).toBeNull();
    expect(fx.nativeElement.querySelector('app-page-error-banner')).toBeNull();
  });

  // ── isFormValid false branches (issue #1026) ──────────────────────────────
  it('when name is valid but currency is not 3 characters, isFormValid is false', () => {
    const fx = TestBed.createComponent(PortfoliosPanelComponent);
    fx.detectChanges();
    http.expectOne(`${API}market/indices`).flush({ indices: [] });
    http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });

    fx.componentInstance.toggleCreate();
    fx.componentInstance.updateField('name', 'MyPortfolio');
    fx.componentInstance.updateField('currency', 'US'); // 2 chars → invalid

    expect(fx.componentInstance.isFormValid()).toBe(false);
    http.expectNone((r) => r.method === 'POST');
  });

  // ── Seed error branches (issue #1026) ─────────────────────────────────────
  describe('benchmark seed flow', () => {
    function setupForUnknownBenchmark(): ReturnType<typeof TestBed.createComponent<PortfoliosPanelComponent>> {
      const fx = TestBed.createComponent(PortfoliosPanelComponent);
      fx.detectChanges();
      http.expectOne(`${API}market/indices`).flush({ indices: [] }); // empty → any ticker is unknown
      http.expectOne(`${API}portfolio/`).flush({ items: [], total: 0 });
      fx.componentInstance.toggleCreate();
      fx.componentInstance.updateField('name', 'MyPortfolio');
      fx.componentInstance.updateField('benchmark_ticker', 'XYZ'); // not in known benchmarks
      return fx;
    }

    it('when startSeed POST returns 500, createError is set and creating is false', () => {
      const fx = setupForUnknownBenchmark();
      fx.componentInstance.submit();

      http.expectOne(`${API}reference-indices/seed`)
        .flush({ detail: 'service unavailable' }, { status: 500, statusText: 'Server Error' });

      expect(fx.componentInstance.createError()).toBeTruthy();
      expect(fx.componentInstance.creating()).toBe(false);
    });

    it('when seed poll emits status="failed", createError is set and creating is false', () => {
      const fx = setupForUnknownBenchmark();

      const refIndex = TestBed.inject(ReferenceIndexService);
      const failedProgress: ReferenceIndexSeedProgress = {
        job_id: 'j1', status: 'failed',
        current: 0, total: 0, errors: [], result: null, error: 'Ticker not found',
      };
      spyOn(refIndex, 'pollUntilDone').and.returnValue(of(failedProgress));

      fx.componentInstance.submit();
      http.expectOne(`${API}reference-indices/seed`).flush({ job_id: 'j1', status: 'pending' });

      expect(fx.componentInstance.createError()).toBeTruthy();
      expect(fx.componentInstance.creating()).toBe(false);
    });
  });
});
