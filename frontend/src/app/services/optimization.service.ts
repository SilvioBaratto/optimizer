import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

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

interface ApiErrorBody {
  error?: {
    message?: string;
    details?: { validation_errors?: Record<string, string[]> };
  };
  detail?: string;
  validation_errors?: Record<string, string[]>;
}

function formatValidationErrors(
  errors: Record<string, string[]> | undefined,
): string | null {
  if (!errors) return null;
  const fields = Object.entries(errors)
    .map(([f, msgs]) => `${f || 'request'}: ${msgs.join(', ')}`)
    .join('; ');
  return fields || null;
}

interface ApiErrorLike {
  status?: number;
  message?: string;
  details?: unknown;
}

function readBodyMessage(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const ab = body as ApiErrorBody;
  const nested = formatValidationErrors(ab.error?.details?.validation_errors);
  if (nested) return nested;
  const top = formatValidationErrors(ab.validation_errors);
  if (top) return top;
  if (ab.error?.message) return ab.error.message;
  if (ab.detail) return ab.detail;
  return null;
}

function extractApiMessage(err: unknown, fallback: string): Error {
  if (err instanceof HttpErrorResponse) {
    const body = err.error as ApiErrorBody | string | undefined;
    const fromBody = readBodyMessage(body);
    if (fromBody) return new Error(fromBody);
    if (typeof body === 'string' && body) return new Error(body);
    return new Error(`${fallback} (HTTP ${err.status})`);
  }
  const apiErr = err as ApiErrorLike | null | undefined;
  if (apiErr && typeof apiErr === 'object') {
    const fromDetails = readBodyMessage(apiErr.details);
    if (fromDetails) return new Error(fromDetails);
    if (typeof apiErr.message === 'string' && apiErr.message) {
      return new Error(apiErr.message);
    }
  }
  return new Error(fallback);
}

function mapHttpError(fallback: string) {
  return (err: unknown) => throwError(() => extractApiMessage(err, fallback));
}

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
    return this.http
      .post<CalibrateDeltaResponse>(`${this.api}llm-moments/calibrate-delta`, body)
      .pipe(catchError(mapHttpError('calibrate-delta failed')));
  }

  adaptFactorWeights(
    body: AdaptFactorWeightsRequest,
  ): Observable<AdaptFactorWeightsResponse> {
    return this.http
      .post<AdaptFactorWeightsResponse>(
        `${this.api}llm-moments/adapt-factor-weights`,
        body,
      )
      .pipe(catchError(mapHttpError('adapt-factor-weights failed')));
  }

  selectCovRegime(
    body: SelectCovRegimeRequest,
  ): Observable<SelectCovRegimeResponse> {
    return this.http
      .post<SelectCovRegimeResponse>(`${this.api}llm-moments/select-cov-regime`, body)
      .pipe(catchError(mapHttpError('select-cov-regime failed')));
  }

  generateViews(body: GenerateViewsRequest): Observable<GenerateViewsResponse> {
    return this.http
      .post<GenerateViewsResponse>(`${this.api}views/generate`, body)
      .pipe(catchError(mapHttpError('generate-views failed')));
  }

  opinionPool(body: OpinionPoolRequest): Observable<OpinionPoolResponse> {
    return this.http
      .post<OpinionPoolResponse>(`${this.api}views/opinion-pool`, body)
      .pipe(catchError(mapHttpError('opinion-pool failed')));
  }

  entropyPooling(body: EntropyPoolingRequest): Observable<EntropyPoolingResponse> {
    return this.http
      .post<EntropyPoolingResponse>(`${this.api}views/entropy-pooling`, body)
      .pipe(catchError(mapHttpError('entropy-pooling failed')));
  }

  static isAsyncResponse(value: OptimizeResult): value is OptimizeAsyncResponse {
    return (
      typeof (value as OptimizeAsyncResponse).job_id === 'string' &&
      typeof (value as OptimizeAsyncResponse).run_id === 'string' &&
      !('weights' in value)
    );
  }
}
