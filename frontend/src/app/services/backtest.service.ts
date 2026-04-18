import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  BacktestApiRequest,
  BacktestAsyncResponse,
  BacktestProgressResponse,
  EquityCurvePoint,
  ValidateApiRequest,
  ValidateAsyncResponse,
  ValidateProgressResponse,
} from '../models/backtest.model';

/**
 * HTTP client for /api/v1/backtest and /api/v1/validate/walk-forward.
 *
 * Both endpoints are 202 + job_id flows. The validation route supports
 * multiple CV strategies via the `cv_type` field on the request body.
 */
@Injectable({ providedIn: 'root' })
export class BacktestService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  runBacktest(body: BacktestApiRequest): Observable<BacktestAsyncResponse> {
    return this.http.post<BacktestAsyncResponse>(`${this.api}backtest`, body);
  }

  pollBacktest(jobId: string): Observable<BacktestProgressResponse> {
    const encoded = encodeURIComponent(jobId);
    return this.http.get<BacktestProgressResponse>(`${this.api}backtest/${encoded}`);
  }

  runWalkForward(body: ValidateApiRequest): Observable<ValidateAsyncResponse> {
    return this.http.post<ValidateAsyncResponse>(
      `${this.api}validate/walk-forward`,
      body,
    );
  }

  pollWalkForward(jobId: string): Observable<ValidateProgressResponse> {
    const encoded = encodeURIComponent(jobId);
    return this.http.get<ValidateProgressResponse>(
      `${this.api}validate/walk-forward/${encoded}`,
    );
  }

  getEquityCurve(portfolioName: string): Observable<EquityCurvePoint[]> {
    const encoded = encodeURIComponent(portfolioName);
    return this.http.get<EquityCurvePoint[]>(
      `${this.api}portfolio-analytics/${encoded}/equity-curve`,
    );
  }
}
