import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../../environments/environment';
import type { ReferenceIndicesResponse } from '../models/dashboard-api.model';

/**
 * HTTP client for the dashboard's market reference endpoints.
 *
 *   GET /api/v1/market/indices → ReferenceIndicesResponse
 *     (ETF-filtered instruments for the benchmark selector)
 */
@Injectable({ providedIn: 'root' })
export class MarketService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  getIndices(): Observable<ReferenceIndicesResponse> {
    return this.http.get<ReferenceIndicesResponse>(`${this.api}market/indices`);
  }
}
