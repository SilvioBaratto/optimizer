import { ErrorHandler, Injectable, InjectionToken, inject } from '@angular/core';
import { HttpErrorResponse } from '@angular/common/http';

import { NotificationService } from '../shared/notification/notification.service';

export const DEDUP_WINDOW_MS = 5000;
export const GENERIC_ERROR_MESSAGE = 'An unexpected error occurred.';

export type Clock = () => number;

export const ERROR_CLOCK = new InjectionToken<Clock>('ERROR_CLOCK', {
  providedIn: 'root',
  factory: () => () => Date.now(),
});

function isHttpError(error: unknown): boolean {
  if (error instanceof HttpErrorResponse) {
    return true;
  }
  const rejection = (error as { rejection?: unknown } | null)?.rejection;
  return rejection instanceof HttpErrorResponse;
}

function extractMessage(error: unknown): string {
  if (typeof error === 'string' && error.length > 0) {
    return error;
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return GENERIC_ERROR_MESSAGE;
}

@Injectable()
export class GlobalErrorHandler implements ErrorHandler {
  private readonly notifications = inject(NotificationService);
  private readonly clock = inject(ERROR_CLOCK);
  private readonly recent = new Map<string, number>();

  handleError(error: unknown): void {
    console.error('[GlobalErrorHandler]', error);
    if (isHttpError(error)) {
      return;
    }
    const message = extractMessage(error);
    if (this.wasRecentlyShown(message)) {
      return;
    }
    this.recent.set(message, this.clock());
    this.notifications.error(message);
  }

  private wasRecentlyShown(message: string): boolean {
    const last = this.recent.get(message);
    return last !== undefined && this.clock() - last < DEDUP_WINDOW_MS;
  }
}
