import { Component, input, output, ChangeDetectionStrategy } from '@angular/core';
import { LucideAngularModule } from 'lucide-angular';

@Component({
  selector: 'app-page-error-banner',
  imports: [LucideAngularModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div role="alert" class="flex flex-col items-center justify-center py-20 px-4">
      <p class="text-sm text-text-secondary mb-4">{{ message() }}</p>
      <button type="button" (click)="retry.emit()"
        class="px-4 py-2 text-sm font-medium bg-accent text-white rounded-lg hover:bg-accent/90 transition-colors"
        aria-label="Retry">
        Try Again
      </button>
    </div>
  `,
})
export class PageErrorBannerComponent {
  message = input.required<string>();
  retry = output<void>();
}
