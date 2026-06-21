import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import { PortfolioApiService } from '../core/services/portfolio-api.service';
import type {
  OptimizationRunResponse,
  OptimizeAsyncResponse,
  OptimizeRequest,
  TuneJobCreateResponse,
  TuneRequest,
} from '../core/models/optimization.model';

export type OptimizeResult = OptimizationRunResponse | OptimizeAsyncResponse;

/**
 * Optimizer types the backend `OptimizeRequest.optimizer_type` Literal accepts
 * (`api/app/schemas/optimization/optimization.py`: `Literal["mean_risk"]`).
 * Single source of truth so an unsupported value cannot reach POST /optimize.
 */
export const ACCEPTED_OPTIMIZER_TYPES = ['mean_risk'] as const;

/** True when `value` is an optimizer_type the backend `/optimize` endpoint accepts. */
export function isAcceptedOptimizerType(value: string): boolean {
  return (ACCEPTED_OPTIMIZER_TYPES as readonly string[]).includes(value);
}

/** Minimal run config accepted by buildOptimizeBody. */
export interface RunConfig {
  optimizerType: string;
  config: Record<string, unknown>;
}

@Injectable({ providedIn: 'root' })
export class OptimizationService {
  private readonly http = inject(HttpClient);
  private readonly portfolioApi = inject(PortfolioApiService);
  private readonly api = environment.apiUrl;

  optimize(body: OptimizeRequest): Observable<OptimizeResult> {
    return this.http.post<OptimizeResult>(`${this.api}optimize`, body);
  }

  getOptimizationRun(runId: string): Observable<OptimizationRunResponse> {
    const encoded = encodeURIComponent(runId);
    return this.http.get<OptimizationRunResponse>(`${this.api}optimize/${encoded}`);
  }

  tune(body: TuneRequest): Observable<TuneJobCreateResponse> {
    return this.http.post<TuneJobCreateResponse>(`${this.api}tune`, body);
  }

  buildOptimizeBody(
    request: RunConfig,
    tickers: string[],
    startDate: string,
    endDate: string,
  ): OptimizeRequest {
    return {
      tickers,
      start_date: startDate,
      end_date: endDate,
      optimizer_type: request.optimizerType,
      config: request.config,
    };
  }

  applyWeightsToPortfolio(
    portfolioRef: string,
    weights: Record<string, number>,
    today: string,
  ): Observable<string> {
    return this.portfolioApi.list().pipe(
      switchMap((list) => {
        const target = list.items.find(
          (p) => p.id === portfolioRef || p.name === portfolioRef,
        );
        if (!target) {
          return throwError(() => new Error(`Portfolio ${portfolioRef} not found.`));
        }
        return this.portfolioApi
          .createSnapshot(target.name, {
            snapshot_date: today,
            snapshot_type: 'optimization',
            weights,
          })
          .pipe(map(() => target.name));
      }),
    );
  }

  static isAsyncResponse(value: OptimizeResult): value is OptimizeAsyncResponse {
    return (
      typeof (value as OptimizeAsyncResponse).job_id === 'string' &&
      typeof (value as OptimizeAsyncResponse).run_id === 'string' &&
      !('weights' in value)
    );
  }
}
