import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  FactorComputeAsyncResponse,
  FactorComputeProgress,
  FactorComputeRequest,
  FactorExposureConstraintsApiResponse,
  FactorExposureConstraintsRequest,
  FactorQuintileSpreadApiResponse,
  FactorQuintileSpreadRequest,
  FactorRegimeTiltApiResponse,
  FactorRegimeTiltRequest,
  FactorScoreApiResponse,
  FactorScoreRequest,
  FactorSelectApiResponse,
  FactorSelectRequest,
  FactorValidateRequest,
  FactorValidateResponse,
  TradingEconomicsObservation,
} from '../models/factor.model';
import type { MacroCalibrationResponse } from '../core/models/macro-intelligence.model';

interface MacroCalibrationQuery {
  country?: string;
  macroText?: string;
  refresh?: boolean;
}

interface TeObservationsQuery {
  country?: string;
  indicatorKeys?: string[];
  startDate?: string;
  endDate?: string;
  limit?: number;
}

/**
 * HTTP client for factor research endpoints:
 *
 *   Async (202 + job_id flow):
 *     POST /factors/compute       → FactorComputeAsyncResponse
 *     GET  /factors/compute/{id}  → FactorComputeProgress
 *
 *   Synchronous POSTs:
 *     /factors/{validate,score,select,exposure-constraints,quintile-spread,regime-tilt}
 *
 *   Macro reads:
 *     GET /views/macro-calibration
 *     GET /market/regime/history
 *     GET /macro-data/te-observations
 */
@Injectable({ providedIn: 'root' })
export class FactorsService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  // ── Async factor compute ─────────────────────────────────────────────────
  compute(body: FactorComputeRequest): Observable<FactorComputeAsyncResponse> {
    return this.http.post<FactorComputeAsyncResponse>(
      `${this.api}factors/compute`,
      body,
    );
  }

  pollCompute(jobId: string): Observable<FactorComputeProgress> {
    return this.http.get<FactorComputeProgress>(
      `${this.api}factors/compute/${encodeURIComponent(jobId)}`,
    );
  }

  // ── Synchronous factor endpoints ─────────────────────────────────────────
  validate(body: FactorValidateRequest): Observable<FactorValidateResponse> {
    return this.http.post<FactorValidateResponse>(`${this.api}factors/validate`, body);
  }

  score(body: FactorScoreRequest): Observable<FactorScoreApiResponse> {
    return this.http.post<FactorScoreApiResponse>(`${this.api}factors/score`, body);
  }

  select(body: FactorSelectRequest): Observable<FactorSelectApiResponse> {
    return this.http.post<FactorSelectApiResponse>(`${this.api}factors/select`, body);
  }

  exposureConstraints(
    body: FactorExposureConstraintsRequest,
  ): Observable<FactorExposureConstraintsApiResponse> {
    return this.http.post<FactorExposureConstraintsApiResponse>(
      `${this.api}factors/exposure-constraints`,
      body,
    );
  }

  quintileSpread(
    body: FactorQuintileSpreadRequest,
  ): Observable<FactorQuintileSpreadApiResponse> {
    return this.http.post<FactorQuintileSpreadApiResponse>(
      `${this.api}factors/quintile-spread`,
      body,
    );
  }

  regimeTilt(body: FactorRegimeTiltRequest): Observable<FactorRegimeTiltApiResponse> {
    return this.http.post<FactorRegimeTiltApiResponse>(
      `${this.api}factors/regime-tilt`,
      body,
    );
  }

  // ── Macro reads ──────────────────────────────────────────────────────────
  macroCalibration(query: MacroCalibrationQuery): Observable<MacroCalibrationResponse> {
    let params = new HttpParams();
    if (query.country) params = params.set('country', query.country);
    if (query.macroText) params = params.set('macro_text', query.macroText);
    if (query.refresh !== undefined) params = params.set('refresh', String(query.refresh));
    return this.http.get<MacroCalibrationResponse>(
      `${this.api}views/macro-calibration`,
      params.keys().length ? { params } : {},
    );
  }

  teObservations(
    query: TeObservationsQuery,
  ): Observable<TradingEconomicsObservation[]> {
    let params = new HttpParams();
    if (query.country) params = params.set('country', query.country);
    if (query.startDate) params = params.set('start_date', query.startDate);
    if (query.endDate) params = params.set('end_date', query.endDate);
    if (query.limit !== undefined) params = params.set('limit', String(query.limit));
    if (query.indicatorKeys) {
      for (const key of query.indicatorKeys) {
        params = params.append('indicator_keys', key);
      }
    }
    return this.http.get<TradingEconomicsObservation[]>(
      `${this.api}macro-data/te-observations`,
      params.keys().length ? { params } : {},
    );
  }
}
