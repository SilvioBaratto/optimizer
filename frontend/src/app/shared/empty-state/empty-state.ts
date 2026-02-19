import { Component, input, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-empty-state',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="flex flex-col items-center justify-center py-12 text-center">
      <div class="text-text-tertiary mb-3">
        <ng-content select="[icon]"></ng-content>
      </div>
      <h3 class="text-sm font-medium text-text">{{ title() }}</h3>
      @if (description()) {
        <p class="mt-1 text-xs text-text-secondary max-w-sm">{{ description() }}</p>
      }
      <div class="mt-4">
        <ng-content select="[action]"></ng-content>
      </div>
    </div>
  `,
})
export class EmptyStateComponent {
  title = input.required<string>();
  description = input<string>('');
}
