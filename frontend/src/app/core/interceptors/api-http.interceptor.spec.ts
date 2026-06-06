import { TestBed } from '@angular/core/testing';
import {
  HttpClient,
  HttpContext,
  HttpErrorResponse,
  provideHttpClient,
  withInterceptors,
} from '@angular/common/http';
import {
  HttpTestingController,
  provideHttpClientTesting,
} from '@angular/common/http/testing';
import { provideZonelessChangeDetection } from '@angular/core';
import { of } from 'rxjs';

import {
  apiHttpInterceptor,
  RETRY_BACKOFF,
  SUPPRESS_TOAST_STATUSES,
  BackoffFn,
} from './api-http.interceptor';
import { environment } from '../../../environments/environment';
import { NotificationService } from '../../shared/notification/notification.service';
import { ApiError } from '../models/api-error.model';

describe('apiHttpInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let notifications: jasmine.SpyObj<NotificationService>;
  let attempts: number[];

  const apiUrl = environment.apiUrl;
  const apiEndpoint = `${apiUrl}portfolio/`;
  const externalUrl = 'https://cdn.example.com/asset.json';

  beforeEach(() => {
    attempts = [];
    const spy = jasmine.createSpyObj<NotificationService>(
      'NotificationService',
      ['success', 'error', 'warning', 'info'],
    );
    const immediateBackoff: BackoffFn = (attempt) => {
      attempts.push(attempt);
      return of(0);
    };

    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        provideHttpClient(withInterceptors([apiHttpInterceptor])),
        provideHttpClientTesting(),
        { provide: NotificationService, useValue: spy },
        { provide: RETRY_BACKOFF, useValue: immediateBackoff },
      ],
    });

    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
    notifications = TestBed.inject(
      NotificationService,
    ) as jasmine.SpyObj<NotificationService>;
  });

  afterEach(() => {
    httpMock.verify();
  });

  describe('API-key injection', () => {
    it('injects X-API-Key header when URL starts with apiUrl', () => {
      http.get(apiEndpoint).subscribe();

      const req = httpMock.expectOne(apiEndpoint);
      expect(req.request.headers.get('X-API-Key')).toBe(environment.apiKey);
      req.flush({});
    });

    it('does NOT inject X-API-Key header on external URLs', () => {
      http.get(externalUrl).subscribe();

      const req = httpMock.expectOne(externalUrl);
      expect(req.request.headers.has('X-API-Key')).toBe(false);
      req.flush({});
    });
  });

  describe('retry with exponential backoff', () => {
    it('retries GET on 503 and then succeeds', () => {
      let result: unknown;
      http.get(apiEndpoint).subscribe((r) => (result = r));

      httpMock
        .expectOne(apiEndpoint)
        .flush('fail', { status: 503, statusText: 'Service Unavailable' });
      httpMock.expectOne(apiEndpoint).flush({ ok: true });

      expect(result).toEqual({ ok: true });
      expect(attempts).toEqual([1]);
    });

    it('retries GET on 408 Request Timeout', () => {
      let result: unknown;
      http.get(apiEndpoint).subscribe((r) => (result = r));

      httpMock
        .expectOne(apiEndpoint)
        .flush('', { status: 408, statusText: 'Request Timeout' });
      httpMock.expectOne(apiEndpoint).flush({ ok: true });

      expect(result).toEqual({ ok: true });
    });

    it('retries GET on network error (status 0)', () => {
      let result: unknown;
      http.get(apiEndpoint).subscribe((r) => (result = r));

      httpMock
        .expectOne(apiEndpoint)
        .error(new ProgressEvent('network error'), { status: 0 });
      httpMock.expectOne(apiEndpoint).flush({ ok: true });

      expect(result).toEqual({ ok: true });
    });

    it('retries up to 3 times using exponential attempt numbering', () => {
      let error: ApiError | undefined;
      http.get(apiEndpoint).subscribe({ error: (e: ApiError) => (error = e) });

      for (let i = 0; i < 4; i++) {
        httpMock
          .expectOne(apiEndpoint)
          .flush('', { status: 503, statusText: 'Service Unavailable' });
      }

      httpMock.expectNone(apiEndpoint);
      expect(attempts).toEqual([1, 2, 3]);
      expect(error?.status).toBe(503);
    });

    it('does NOT retry POST on 503', () => {
      let error: ApiError | undefined;
      http.post(apiEndpoint, {}).subscribe({
        error: (e: ApiError) => (error = e),
      });

      httpMock
        .expectOne(apiEndpoint)
        .flush('nope', { status: 503, statusText: 'Service Unavailable' });

      httpMock.expectNone(apiEndpoint);
      expect(attempts).toEqual([]);
      expect(error?.status).toBe(503);
    });

    it('does NOT retry GET on 400 Bad Request', () => {
      let error: ApiError | undefined;
      http.get(apiEndpoint).subscribe({ error: (e: ApiError) => (error = e) });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'bad' }, { status: 400, statusText: 'Bad Request' });

      httpMock.expectNone(apiEndpoint);
      expect(attempts).toEqual([]);
      expect(error?.status).toBe(400);
    });
  });

  describe('error normalization', () => {
    it('normalizes 404 into ApiError with status, message, details', () => {
      let error: ApiError | undefined;
      http.get(apiEndpoint).subscribe({ error: (e: ApiError) => (error = e) });

      httpMock.expectOne(apiEndpoint).flush(
        { detail: 'Portfolio not found', code: 'not_found' },
        { status: 404, statusText: 'Not Found' },
      );

      expect(error?.status).toBe(404);
      expect(error?.message).toBe('Portfolio not found');
      expect(error?.details).toEqual({
        detail: 'Portfolio not found',
        code: 'not_found',
      });
    });

    it('falls back to statusText when no detail is provided', () => {
      let error: ApiError | undefined;
      http.post(apiEndpoint, {}).subscribe({
        error: (e: ApiError) => (error = e),
      });

      httpMock
        .expectOne(apiEndpoint)
        .flush(null, { status: 500, statusText: 'Internal Server Error' });

      expect(error?.status).toBe(500);
      expect(error?.message).toContain('Internal Server Error');
    });

    it('wraps HttpErrorResponse rather than passing it through raw', () => {
      let error: unknown;
      http.get(apiEndpoint).subscribe({ error: (e) => (error = e) });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'x' }, { status: 400, statusText: 'Bad Request' });

      expect(error instanceof HttpErrorResponse).toBe(false);
      expect((error as ApiError).status).toBe(400);
    });
  });

  describe('toast dispatch', () => {
    it('dispatches NotificationService.error on 500', () => {
      http.post(apiEndpoint, {}).subscribe({ error: () => {} });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });

      expect(notifications.error).toHaveBeenCalledTimes(1);
      expect(notifications.error.calls.mostRecent().args[0]).toContain('boom');
    });

    it('dispatches NotificationService.error on 404', () => {
      http.get(apiEndpoint).subscribe({ error: () => {} });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'missing' }, { status: 404, statusText: 'Not Found' });

      expect(notifications.error).toHaveBeenCalledTimes(1);
    });

    it('suppresses toast on 401 Unauthorized', () => {
      http.get(apiEndpoint).subscribe({ error: () => {} });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'unauth' }, { status: 401, statusText: 'Unauthorized' });

      expect(notifications.error).not.toHaveBeenCalled();
    });

    it('suppresses toast on 409 Conflict', () => {
      http.post(apiEndpoint, {}).subscribe({ error: () => {} });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'conflict' }, { status: 409, statusText: 'Conflict' });

      expect(notifications.error).not.toHaveBeenCalled();
    });

    it('does NOT dispatch toast on successful response', () => {
      http.get(apiEndpoint).subscribe();

      httpMock.expectOne(apiEndpoint).flush({});

      expect(notifications.error).not.toHaveBeenCalled();
    });
  });

  describe('per-call toast suppression via SUPPRESS_TOAST_STATUSES (issue #438)', () => {
    function suppressing(statuses: readonly number[]): HttpContext {
      return new HttpContext().set(SUPPRESS_TOAST_STATUSES, statuses);
    }

    it('suppresses the toast when the response status is in the per-call list', () => {
      http.get(apiEndpoint, { context: suppressing([404]) }).subscribe({
        error: () => {},
      });

      httpMock
        .expectOne(apiEndpoint)
        .flush(null, { status: 404, statusText: 'Not Found' });

      expect(notifications.error).not.toHaveBeenCalled();
    });

    it('still toasts other statuses when 404 is suppressed (e.g. 500)', () => {
      // 5xx is retriable for GET, so the interceptor will retry MAX_RETRIES
      // times. Flush every attempt with the same 500 to drive the chain to
      // its final propagated error.
      http.get(apiEndpoint, { context: suppressing([404]) }).subscribe({
        error: () => {},
      });

      for (let i = 0; i < 4; i++) {
        httpMock
          .expectOne(apiEndpoint)
          .flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
      }

      expect(notifications.error).toHaveBeenCalledTimes(1);
      expect(notifications.error.calls.mostRecent().args[0]).toContain('boom');
    });

    it('still propagates the ApiError downstream so the page can render an empty state', () => {
      let error: ApiError | undefined;
      http.get(apiEndpoint, { context: suppressing([404]) }).subscribe({
        error: (e: ApiError) => (error = e),
      });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'gone' }, { status: 404, statusText: 'Not Found' });

      expect(error?.status).toBe(404);
      expect(error?.message).toBe('gone');
    });

    it('does not affect other requests that did not opt in', () => {
      http.get(apiEndpoint).subscribe({ error: () => {} });

      httpMock
        .expectOne(apiEndpoint)
        .flush({ detail: 'missing' }, { status: 404, statusText: 'Not Found' });

      expect(notifications.error).toHaveBeenCalledTimes(1);
    });
  });
});
