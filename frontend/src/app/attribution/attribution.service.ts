import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import { mapHttpError } from '../core/http-error';
import type {
  BrinsonApiRequest,
  BrinsonApiResponse,
  FactorAttributionApiRequest,
  FactorAttributionApiResponse,
} from './attribution.model';

/**
 * HTTP client for /api/v1/attribution/{brinson,factor}.
 *
 * Both endpoints are synchronous POSTs that validate weights sum to 1.0 ± 0.01
 * and that start_date < end_date before running the attribution calculation.
 * The portfolio's snapshot weights are loaded by the caller (the page resolves
 * them via PortfolioApiService.getLatestSnapshot) and passed in the POST body —
 * the backend exposes no GET-by-name attribution endpoints.
 */
@Injectable({ providedIn: 'root' })
export class AttributionService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

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
