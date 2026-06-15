/**
 * Example tests for issue #1010 — factor attribution 404 criterion.
 *
 * Criteria covered:
 *  - (UNIT) AttributionService.factor() propagates HttpErrorResponse with .status intact.
 *  - (UNIT) A 404 from POST /attribution/factor can be detected by the subscriber (status 404).
 *  - (UNIT) Other 4xx/5xx statuses also propagate with correct status.
 *  - (UNIT) FactorAttributionPanelComponent shows the "Factor scores not available" inline
 *           message when notAvailable=true (the parent sets this on a 404 response).
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';

import { configureTestBed } from '../../testing';
import { FactorAttributionPanelComponent } from './factor-attribution-panel/factor-attribution-panel';

import { AttributionService } from './attribution.service';
import { environment } from '../../environments/environment';
import type { FactorAttributionApiRequest } from './attribution.model';

const API = environment.apiUrl;

function factorRequest(): FactorAttributionApiRequest {
  return {
    portfolio_weights: { AAPL: 0.5, MSFT: 0.5 },
    start_date: '2024-01-01',
    end_date: '2024-12-31',
  };
}

describe('AttributionService — factor() error propagation (issue #1009)', () => {
  let svc: AttributionService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(),
        provideHttpClientTesting(),
        AttributionService,
      ],
    });
    svc = TestBed.inject(AttributionService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('when factor call returns 404, subscriber receives HttpErrorResponse with status 404', () => {
    let error: HttpErrorResponse | undefined;
    svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

    http
      .expectOne(`${API}attribution/factor`)
      .flush(
        { detail: 'Factor scores not found' },
        { status: 404, statusText: 'Not Found' },
      );

    expect(error?.status).toBe(404);
    expect(error?.error?.detail).toBe('Factor scores not found');
  });

  it('when factor call returns 500, subscriber receives HttpErrorResponse with status 500', () => {
    let error: HttpErrorResponse | undefined;
    svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

    http
      .expectOne(`${API}attribution/factor`)
      .flush(
        { detail: 'Internal server error' },
        { status: 500, statusText: 'Internal Server Error' },
      );

    expect(error?.status).toBe(500);
  });

  it('when factor call returns 403, subscriber receives HttpErrorResponse with status 403', () => {
    let error: HttpErrorResponse | undefined;
    svc.factor(factorRequest()).subscribe({ error: (e) => (error = e) });

    http
      .expectOne(`${API}attribution/factor`)
      .flush(
        { detail: 'Forbidden' },
        { status: 403, statusText: 'Forbidden' },
      );

    expect(error?.status).toBe(403);
  });

  it('when factor call succeeds, error callback is not invoked', () => {
    let errorFired = false;
    svc.factor(factorRequest()).subscribe({
      error: () => { errorFired = true; },
    });

    http
      .expectOne(`${API}attribution/factor`)
      .flush({
        factors: [],
        portfolioReturn: 0.05,
        explainedReturn: 0.03,
        residual: 0.02,
      });

    expect(errorFired).toBeFalse();
  });
});

// ---------------------------------------------------------------------------
// Component-level: inline 404 message vs generic empty-state
// Criterion (issue #1010): A 404 response from POST /attribution/factor renders
// an inline empty-state message ("Factor scores not available. Run the factor
// pipeline first.") inside FactorAttributionPanelComponent rather than a generic
// error toast.  The parent AttributionComponent maps 404 → notAvailable=true and
// passes it as an input; the component owns the rendering contract tested here.
// ---------------------------------------------------------------------------

describe('FactorAttributionPanelComponent — 404 inline empty state (issue #1010)', () => {
  let fixture: ComponentFixture<FactorAttributionPanelComponent>;

  beforeEach(async () => {
    await configureTestBed({
      imports: [FactorAttributionPanelComponent],
      withHttp: false,
    });
    fixture = TestBed.createComponent(FactorAttributionPanelComponent);
  });

  it('when notAvailable is true, renders "Factor scores not available. Run the factor pipeline first." inline', () => {
    fixture.componentRef.setInput('notAvailable', true);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).toContain('Factor scores not available');
    expect(text).toContain('Run the factor pipeline first');
  });

  it('when notAvailable is false (non-404 error or no run), renders the generic "no run yet" state, not the 404 message', () => {
    fixture.componentRef.setInput('notAvailable', false);
    fixture.detectChanges();
    const text = (fixture.nativeElement as HTMLElement).textContent ?? '';
    expect(text).not.toContain('Factor scores not available');
    expect(text).toContain('No factor attribution result yet');
  });

  it('when notAvailable is true, no toast/snackbar/alert element is present — only the inline message paragraph', () => {
    fixture.componentRef.setInput('notAvailable', true);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    // The panel never mounts a toast or role="alert" widget for 404.
    expect(el.querySelector('[role="alert"]')).toBeNull();
    expect(el.querySelector('.toast')).toBeNull();
    // The inline "Factor scores not available" text is present instead.
    expect((el.textContent ?? '')).toContain('Factor scores not available');
  });
});
