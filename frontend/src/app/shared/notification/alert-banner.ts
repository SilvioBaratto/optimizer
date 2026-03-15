import { Component, input, output, signal, ChangeDetectionStrategy } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';

@Component({
  selector: 'app-alert-banner',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (!isDismissed()) {
      <div class="flex items-start gap-3 px-4 py-3 rounded-lg border text-sm"
           [class]="levelClasses()">
        <div class="flex-1 text-text">{{ message() }}</div>
        @if (dismissible()) {
          <button (click)="onDismiss()"
                  class="shrink-0 p-0.5 text-text-tertiary hover:text-text transition-colors"
                  aria-label="Dismiss">
            <i-lucide name="x" />
          </button>
        }
      </div>
    }
  `,
})
export class AlertBannerComponent {
  level = input<'success' | 'error' | 'warning' | 'info'>('info');
  message = input.required<string>();
  dismissible = input(true);
  dismissed = output<void>();

  isDismissed = signal(false);

  levelClasses(): string {
    const map: Record<string, string> = {
      success: 'bg-gain/10 border-gain/20',
      error: 'bg-loss/10 border-loss/20',
      warning: 'bg-warning/10 border-warning/20',
      info: 'bg-accent/10 border-accent/20',
    };
    return map[this.level()] ?? map['info'];
  }

  onDismiss() {
    this.isDismissed.set(true);
    this.dismissed.emit();
  }
}
