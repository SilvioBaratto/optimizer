import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import type {
  BrinsonApiRequest,
  BrinsonApiResponse,
  FactorAttributionApiRequest,
  FactorAttributionApiResponse,
} from './attribution.model';

interface ApiErrorBody {
  error?: {
    message?: string;
    details?: { validation_errors?: Record<string, string[]> };
  };
  detail?: string;
  validation_errors?: Record<string, string[]>;
}

interface ApiErrorLike {
  status?: number;
  message?: string;
  details?: unknown;
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

function mapHttpError() {
  return (err: unknown) => throwError(() => err);
}

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
