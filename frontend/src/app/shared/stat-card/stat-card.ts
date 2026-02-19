import { Component, input, ChangeDetectionStrategy } from '@angular/core';

@Component({
  selector: 'app-stat-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="bg-surface-raised border border-border rounded-lg p-4">
      <p class="text-xs font-medium text-text-secondary uppercase tracking-wide">{{ label() }}</p>
      <p class="mt-1 text-2xl font-semibold text-text">{{ value() }}</p>
      @if (subtitle()) {
        <p class="mt-0.5 text-xs text-text-tertiary">{{ subtitle() }}</p>
      }
    </div>
  `,
})
export class StatCardComponent {
  label = input.required<string>();
  value = input.required<string | number>();
  subtitle = input<string>('');
}
