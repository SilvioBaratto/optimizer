import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

import { environment } from '../../environments/environment';
import type {
  AdaptFactorWeightsRequest,
  AdaptFactorWeightsResponse,
  CalibrateDeltaRequest,
  CalibrateDeltaResponse,
  EntropyPoolingRequest,
  EntropyPoolingResponse,
  GenerateViewsRequest,
  GenerateViewsResponse,
  OpinionPoolRequest,
  OpinionPoolResponse,
  OptimizationRunResponse,
  OptimizeAsyncResponse,
  OptimizeRequest,
  SelectCovRegimeRequest,
  SelectCovRegimeResponse,
  TuneJobCreateResponse,
  TuneRequest,
} from '../models/optimization.model';

export type OptimizeResult = OptimizationRunResponse | OptimizeAsyncResponse;

@Injectable({ providedIn: 'root' })
export class OptimizationService {
  private readonly http = inject(HttpClient);
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

  calibrateDelta(body: CalibrateDeltaRequest): Observable<CalibrateDeltaResponse> {
    return this.http.post<CalibrateDeltaResponse>(
      `${this.api}llm-moments/calibrate-delta`,
      body,
    );
  }

  adaptFactorWeights(
    body: AdaptFactorWeightsRequest,
  ): Observable<AdaptFactorWeightsResponse> {
    return this.http.post<AdaptFactorWeightsResponse>(
      `${this.api}llm-moments/adapt-factor-weights`,
      body,
    );
  }

  selectCovRegime(
    body: SelectCovRegimeRequest,
  ): Observable<SelectCovRegimeResponse> {
    return this.http.post<SelectCovRegimeResponse>(
      `${this.api}llm-moments/select-cov-regime`,
      body,
    );
  }

  generateViews(body: GenerateViewsRequest): Observable<GenerateViewsResponse> {
    return this.http.post<GenerateViewsResponse>(`${this.api}views/generate`, body);
  }

  opinionPool(body: OpinionPoolRequest): Observable<OpinionPoolResponse> {
    return this.http.post<OpinionPoolResponse>(`${this.api}views/opinion-pool`, body);
  }

  entropyPooling(body: EntropyPoolingRequest): Observable<EntropyPoolingResponse> {
    return this.http.post<EntropyPoolingResponse>(`${this.api}views/entropy-pooling`, body);
  }

  static isAsyncResponse(value: OptimizeResult): value is OptimizeAsyncResponse {
    return (
      typeof (value as OptimizeAsyncResponse).job_id === 'string' &&
      typeof (value as OptimizeAsyncResponse).run_id === 'string' &&
      !('weights' in value)
    );
  }
}
