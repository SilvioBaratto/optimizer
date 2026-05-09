import { Injectable, signal, computed } from '@angular/core';

function generateNotificationId(): string {
  const c = (globalThis as { crypto?: Crypto }).crypto;
  if (c?.randomUUID) {
    try {
      return c.randomUUID();
    } catch {
      /* secure-context guard threw; fall through */
    }
  }
  return `n-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export type NotificationLevel = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  level: NotificationLevel;
  message: string;
  autoDismissMs: number;
}

export interface NotificationOptions {
  autoDismissMs?: number;
}

@Injectable({ providedIn: 'root' })
export class NotificationService {
  private _toasts = signal<Notification[]>([]);
  readonly toasts = computed(() => this._toasts());

  success(message: string, opts?: NotificationOptions) {
    this.add('success', message, opts);
  }

  error(message: string, opts?: NotificationOptions) {
    this.add('error', message, opts);
  }

  warning(message: string, opts?: NotificationOptions) {
    this.add('warning', message, opts);
  }

  info(message: string, opts?: NotificationOptions) {
    this.add('info', message, opts);
  }

  dismiss(id: string) {
    this._toasts.update(list => list.filter(t => t.id !== id));
  }

  private add(level: NotificationLevel, message: string, opts?: NotificationOptions) {
    const autoDismissMs = opts?.autoDismissMs ?? 5000;
    const id = generateNotificationId();
    const notification: Notification = { id, level, message, autoDismissMs };

    this._toasts.update(list => [...list, notification]);

    if (autoDismissMs > 0) {
      setTimeout(() => this.dismiss(id), autoDismissMs);
    }
  }
}
