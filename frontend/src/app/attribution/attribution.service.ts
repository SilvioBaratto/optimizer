import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import type {
  BrinsonApiRequest,
  BrinsonApiResponse,
  FactorAttributionApiRequest,
  FactorAttributionApiResponse,
} from './attribution.model';

/**
 * Identity error mapper: re-throws the raw error unchanged so subscribers
 * receive the original `HttpErrorResponse` (status + body on `error.error`).
 * The interceptor (`api-http.interceptor.ts`) owns message normalization and
 * toast dispatch — services do not duplicate that logic.
 */
function mapHttpError() {
  return (err: unknown) => throwError(() => err);
}

/**
 * HTTP client for /api/v1/attribution/{brinson,factor}.
 *
 * Both endpoints are synchronous POSTs that validate weights sum to 1.0 ± 0.01
 * and that start_date < end_date before running the attribution calculation.
 *
 * Portfolio-context methods (getBrinsonAttribution, getFactorAttribution) accept
 * a portfolio name as their first argument, matching the convention used by
 * PortfolioContextService across the dashboard.
 */
@Injectable({ providedIn: 'root' })
export class AttributionService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  // ── Portfolio-context wiring API (issue #960) ─────────────────────────────

  /**
   * Fetch Brinson-Fachler attribution for a named portfolio.
   * Uses portfolio name (not UUID) as the primary identifier.
   */
  getBrinsonAttribution(
    portfolioName: string,
    startDate: string,
    endDate: string,
    benchmarkWeights: Record<string, number> = { SPY: 1.0 },
  ): Observable<BrinsonApiResponse> {
    return this.http
      .get<BrinsonApiResponse>(
        `${this.api}attribution/brinson/${encodeURIComponent(portfolioName)}`,
        { params: { start_date: startDate, end_date: endDate } },
      )
      .pipe(catchError(mapHttpError()));
  }

  /**
   * Fetch factor attribution for a named portfolio.
   * Uses portfolio name (not UUID) as the primary identifier.
   */
  getFactorAttribution(
    portfolioName: string,
    startDate: string,
    endDate: string,
  ): Observable<FactorAttributionApiResponse> {
    return this.http
      .get<FactorAttributionApiResponse>(
        `${this.api}attribution/factor/${encodeURIComponent(portfolioName)}`,
        { params: { start_date: startDate, end_date: endDate } },
      )
      .pipe(catchError(mapHttpError()));
  }

  // ── Legacy form-based API (manual weight entry) ───────────────────────────

  brinson(body: BrinsonApiRequest): Observable<BrinsonApiResponse> {
    return this.http
      .post<BrinsonApiResponse>(`${this.api}attribution/brinson`, body)
      .pipe(catchError(mapHttpError()));
  }

  factor(body: FactorAttributionApiRequest): Observable<FactorAttributionApiResponse> {
    return this.http
      .post<FactorAttributionApiResponse>(`${this.api}attribution/factor`, body)
      .pipe(catchError(mapHttpError()));
  }
}
