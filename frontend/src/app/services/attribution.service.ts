import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  BrinsonApiRequest,
  BrinsonApiResponse,
  FactorAttributionApiRequest,
  FactorAttributionApiResponse,
} from '../models/attribution.model';

/**
 * HTTP client for /api/v1/attribution/{brinson,factor}.
 *
 * Both endpoints are synchronous POSTs that validate weights sum to 1.0 ± 0.01
 * and that start_date < end_date before running the attribution calculation.
 */
@Injectable({ providedIn: 'root' })
export class AttributionService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  brinson(body: BrinsonApiRequest): Observable<BrinsonApiResponse> {
    return this.http.post<BrinsonApiResponse>(`${this.api}attribution/brinson`, body);
  }

  factor(body: FactorAttributionApiRequest): Observable<FactorAttributionApiResponse> {
    return this.http.post<FactorAttributionApiResponse>(
      `${this.api}attribution/factor`,
      body,
    );
  }
}
