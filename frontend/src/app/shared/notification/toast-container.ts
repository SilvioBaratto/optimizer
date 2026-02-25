import { Component, inject, ChangeDetectionStrategy } from '@angular/core';
import { NotificationService, NotificationLevel } from './notification.service';

@Component({
  selector: 'app-toast-container',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div role="status" aria-live="polite" aria-atomic="false" class="fixed bottom-4 right-4 z-[var(--z-toast,400)] flex flex-col gap-2 w-80">
      @for (toast of notificationService.toasts(); track toast.id) {
        <div class="flex items-start gap-3 px-4 py-3 bg-surface-raised border border-border rounded-lg shadow-lg animate-slide-in"
             [class]="borderClass(toast.level)">
          <div class="flex-1 text-sm text-text">{{ toast.message }}</div>
          <button (click)="notificationService.dismiss(toast.id)"
                  class="shrink-0 p-0.5 text-text-tertiary hover:text-text transition-colors"
                  aria-label="Dismiss">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      }
    </div>
  `,
  styles: [`
    @keyframes slide-in {
      from { opacity: 0; transform: translateX(100%); }
      to { opacity: 1; transform: translateX(0); }
    }
    .animate-slide-in {
      animation: slide-in 0.2s ease-out;
    }
  `],
})
export class ToastContainerComponent {
  notificationService = inject(NotificationService);

  borderClass(level: NotificationLevel): string {
    const map: Record<NotificationLevel, string> = {
      success: 'border-l-4 border-l-gain',
      error: 'border-l-4 border-l-loss',
      warning: 'border-l-4 border-l-warning',
      info: 'border-l-4 border-l-accent',
    };
    return map[level];
  }
}
