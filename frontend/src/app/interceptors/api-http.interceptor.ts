import { inject, InjectionToken } from '@angular/core';
import {
  HttpErrorResponse,
  HttpInterceptorFn,
  HttpRequest,
} from '@angular/common/http';
import { Observable, throwError, timer } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

import { environment } from '../../environments/environment';
import { ApiError } from '../models/api-error.model';
import { NotificationService } from '../shared/notification/notification.service';

const MAX_RETRIES = 3;
const BASE_BACKOFF_MS = 1000;
const RETRIABLE_STATUSES = new Set([0, 408]);
const SILENT_STATUSES = new Set([401, 409]);

export type BackoffFn = (attempt: number) => Observable<number>;

export const RETRY_BACKOFF = new InjectionToken<BackoffFn>('RETRY_BACKOFF', {
  providedIn: 'root',
  factory: () => (attempt: number) =>
    timer(Math.pow(2, attempt - 1) * BASE_BACKOFF_MS),
});

function isInternalRequest(req: HttpRequest<unknown>): boolean {
  return req.url.startsWith(environment.apiUrl);
}

function withApiKey<T>(req: HttpRequest<T>): HttpRequest<T> {
  if (!isInternalRequest(req)) {
    return req;
  }
  return req.clone({ setHeaders: { 'X-API-Key': environment.apiKey } });
}

function isRetriable(error: HttpErrorResponse, method: string): boolean {
  if (method !== 'GET') {
    return false;
  }
  return error.status >= 500 || RETRIABLE_STATUSES.has(error.status);
}

function extractMessage(error: HttpErrorResponse): string {
  const body = error.error;
  if (
    body &&
    typeof body === 'object' &&
    typeof (body as { detail?: unknown }).detail === 'string'
  ) {
    return (body as { detail: string }).detail;
  }
  if (typeof body === 'string' && body.length > 0) {
    return body;
  }
  return error.message || error.statusText || 'Unknown error';
}

function normalize(error: HttpErrorResponse): ApiError {
  return {
    status: error.status,
    message: extractMessage(error),
    details: error.error,
  };
}

function notifyIfNeeded(error: ApiError, notifier: NotificationService): void {
  if (SILENT_STATUSES.has(error.status)) {
    return;
  }
  if (error.status >= 400) {
    notifier.error(error.message);
  }
}

export const apiHttpInterceptor: HttpInterceptorFn = (req, next) => {
  const notifier = inject(NotificationService);
  const backoff = inject(RETRY_BACKOFF);
  const authed = withApiKey(req);

  return next(authed).pipe(
    retry({
      count: MAX_RETRIES,
      delay: (err, attempt) =>
        err instanceof HttpErrorResponse && isRetriable(err, authed.method)
          ? backoff(attempt)
          : throwError(() => err),
    }),
    catchError((err) => {
      if (!(err instanceof HttpErrorResponse)) {
        return throwError(() => err);
      }
      const apiError = normalize(err);
      notifyIfNeeded(apiError, notifier);
      return throwError(() => apiError);
    }),
  );
};
