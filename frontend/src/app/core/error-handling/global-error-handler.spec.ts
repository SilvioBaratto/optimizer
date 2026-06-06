import { TestBed } from '@angular/core/testing';
import { ErrorHandler, provideZonelessChangeDetection } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';

import {
  GlobalErrorHandler,
  ERROR_CLOCK,
  DEDUP_WINDOW_MS,
  GENERIC_ERROR_MESSAGE,
} from './global-error-handler';
import { NotificationService } from '../../shared/notification/notification.service';

describe('GlobalErrorHandler', () => {
  let handler: GlobalErrorHandler;
  let notifications: jasmine.SpyObj<NotificationService>;
  let consoleSpy: jasmine.Spy;
  let now: number;

  beforeEach(() => {
    now = 1_000_000;
    notifications = jasmine.createSpyObj<NotificationService>(
      'NotificationService',
      ['success', 'error', 'warning', 'info'],
    );

    TestBed.configureTestingModule({
      providers: [
        provideZonelessChangeDetection(),
        { provide: NotificationService, useValue: notifications },
        { provide: ERROR_CLOCK, useValue: () => now },
        { provide: ErrorHandler, useClass: GlobalErrorHandler },
      ],
    });

    handler = TestBed.inject(ErrorHandler) as GlobalErrorHandler;
    consoleSpy = spyOn(console, 'error');
  });

  describe('logging', () => {
    it('logs the original error to console.error', () => {
      const err = new Error('boom');
      handler.handleError(err);

      expect(consoleSpy).toHaveBeenCalledWith(jasmine.anything(), err);
    });
  });

  describe('message extraction', () => {
    it('extracts message from Error instance', () => {
      handler.handleError(new Error('Something went wrong'));

      expect(notifications.error).toHaveBeenCalledWith('Something went wrong');
    });

    it('uses string value directly when error is a string', () => {
      handler.handleError('simple string error');

      expect(notifications.error).toHaveBeenCalledWith('simple string error');
    });

    it('falls back to generic message for unknown shapes', () => {
      handler.handleError({ weird: 'shape' });

      expect(notifications.error).toHaveBeenCalledWith(GENERIC_ERROR_MESSAGE);
    });

    it('falls back to generic message for null/undefined', () => {
      handler.handleError(null);
      now += DEDUP_WINDOW_MS + 1;
      handler.handleError(undefined);

      expect(notifications.error).toHaveBeenCalledWith(GENERIC_ERROR_MESSAGE);
      expect(notifications.error).toHaveBeenCalledTimes(2);
    });
  });

  describe('HttpErrorResponse handling', () => {
    it('does NOT dispatch toast for HttpErrorResponse (interceptor handles it)', () => {
      const httpErr = new HttpErrorResponse({
        status: 500,
        statusText: 'Server Error',
      });

      handler.handleError(httpErr);

      expect(notifications.error).not.toHaveBeenCalled();
    });

    it('still logs HttpErrorResponse to console.error', () => {
      const httpErr = new HttpErrorResponse({
        status: 500,
        statusText: 'Server Error',
      });

      handler.handleError(httpErr);

      expect(consoleSpy).toHaveBeenCalled();
    });

    it('also skips HttpErrorResponse wrapped by Angular rethrow (error.rejection)', () => {
      const httpErr = new HttpErrorResponse({
        status: 404,
        statusText: 'Not Found',
      });
      const wrapped = { rejection: httpErr };

      handler.handleError(wrapped);

      expect(notifications.error).not.toHaveBeenCalled();
    });
  });

  describe('deduplication', () => {
    it('suppresses the same message within the dedup window', () => {
      handler.handleError(new Error('duplicate'));
      now += DEDUP_WINDOW_MS - 1;
      handler.handleError(new Error('duplicate'));

      expect(notifications.error).toHaveBeenCalledTimes(1);
    });

    it('allows the same message again after the dedup window elapses', () => {
      handler.handleError(new Error('duplicate'));
      now += DEDUP_WINDOW_MS + 1;
      handler.handleError(new Error('duplicate'));

      expect(notifications.error).toHaveBeenCalledTimes(2);
    });

    it('does not suppress distinct messages', () => {
      handler.handleError(new Error('first'));
      handler.handleError(new Error('second'));

      expect(notifications.error).toHaveBeenCalledTimes(2);
      expect(notifications.error).toHaveBeenCalledWith('first');
      expect(notifications.error).toHaveBeenCalledWith('second');
    });
  });
});
